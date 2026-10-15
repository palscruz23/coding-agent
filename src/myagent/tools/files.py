import json
import os
import stat
from contextlib import ExitStack
from fnmatch import fnmatchcase
from pathlib import Path


def list_files(workspace: Path) -> list[str]:
    """List immediate files and directories, excluding symlinks.

    Args:
        workspace (Path): Directory whose immediate entries are inspected.

    Returns:
        list[str]: Sorted entry names, with a trailing '/' for directories.
    """
    return sorted(
        entry.name + ("/" if entry.is_dir() else "")
        for entry in workspace.iterdir()
        if not entry.is_symlink() and (entry.is_file() or entry.is_dir())
    )


def safe_path(workspace: Path, requested: str) -> Path:
    """Resolve an existing relative path inside the workspace.

    Args:
        workspace (Path): Existing directory that bounds the requested path.
        requested (str): Relative path to an existing workspace entry.

    Returns:
        Path: Absolute resolved path contained within the resolved workspace.

    Raises:
        PermissionError: The path is absolute, contains '..', or resolves outside
            the workspace.
        OSError: The workspace or requested path cannot be resolved strictly.
    """
    base = workspace.resolve(strict=True)
    relative = Path(requested)

    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Use a relative path without '..'")

    candidate = (base / relative).resolve(strict=True)

    if not candidate.is_relative_to(base):
        raise PermissionError("Path escapes the workspace")

    return candidate


ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml"}
DENIED_PATTERNS = (
    ".env",
    ".env.*",
    ".git",
    ".ssh",
    ".aws",
    "id_*",
    "*.pem",
    "*.key",
    "*credentials*",
    "*secret*",
)

MAX_FILE_BYTES = 64 * 1024


def allowed_file(relative: Path) -> bool:
    """Check a relative file path against the name and extension policy.

    Args:
        relative (Path): Path to check without accessing the filesystem.

    Returns:
        bool: True when the path is relative, contains no '..' or denied name
            components, and has an allowed suffix; otherwise False.
    """
    if relative.is_absolute() or ".." in relative.parts:
        return False

    for part in relative.parts:
        if any(fnmatchcase(part.lower(), pattern) for pattern in DENIED_PATTERNS):
            return False

    return relative.suffix.lower() in ALLOWED_SUFFIXES


def read_bounded_bytes(base: Path, relative: Path) -> bytes:
    """Read a regular file with a byte limit and no symlink traversal.

    Args:
        base (Path): Directory from which file descriptor traversal begins.
        relative (Path): Validated relative file path beneath base. The caller
            must reject absolute paths and '..' components before calling.

    Returns:
        bytes: At most MAX_FILE_BYTES + 1 bytes. The extra byte lets the caller
            detect files exceeding the permitted size.

    Raises:
        ValueError: The opened entry is not a regular file.
        OSError: A directory or file cannot be opened or read, including when
            symlink traversal is rejected by the operating system.
    """
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    with ExitStack() as cleanup:
        directory_fd = os.open(base, directory_flags)
        cleanup.callback(os.close, directory_fd)

        for part in relative.parts[:-1]:
            directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            cleanup.callback(os.close, directory_fd)

        file_fd = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        cleanup.callback(os.close, file_fd)

        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError("Expected a regular file")

        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            return stream.read(MAX_FILE_BYTES + 1)


def read_file(workspace: Path, path: str) -> dict[str, str]:
    """Read a permitted workspace file as bounded UTF-8 text.

    Args:
        workspace (Path): Existing directory that bounds file access.
        path (str): Relative path to a file permitted by the file policy.

    Returns:
        dict[str, str]: A 'path' key containing the resolved workspace-relative
            POSIX path and a 'content' key containing the decoded text.

    Raises:
        PermissionError: The requested or resolved path is blocked by policy
            or escapes the workspace.
        ValueError: The file is not regular, exceeds MAX_FILE_BYTES, contains
            a null byte, or cannot be decoded as UTF-8.
        OSError: The workspace or file cannot be resolved, opened, or read.
    """
    base = workspace.resolve(strict=True)

    if not allowed_file(Path(path)):
        raise PermissionError("File blocked by policy")

    file = safe_path(base, path)
    relative = file.relative_to(base)

    if not allowed_file(relative):
        raise PermissionError("Resolved file blocked by policy")

    data = read_bounded_bytes(base, relative)

    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File is too large")

    if b"\x00" in data:
        raise ValueError("Binary content is not supported")

    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File must contain UTF-8 text") from exc

    return {"path": relative.as_posix(), "content": content}


def execute_tool(
    workspace: Path, name: str, arguments: str
) -> list[str] | dict[str, str]:
    """Validate JSON arguments and dispatch a supported workspace tool.

    Args:
        workspace (Path): Directory used for listing or reading files.
        name (str): Supported tool name: 'list_files' or 'read_file'.
        arguments (str): JSON object string; '{}' for list_files or an object
            with exactly one nonempty string 'path' field for read_file.

    Returns:
        list[str] | dict[str, str]: Sorted entry names for list_files, or a
            dictionary containing 'path' and 'content' for read_file.

    Raises:
        ValueError: The tool name or arguments are invalid, JSON decoding fails,
            or read_file rejects the file's type, size, or content.
        PermissionError: File access is blocked by the workspace policy.
        OSError: A filesystem operation fails.
    """
    if name not in {"list_files", "read_file"}:
        raise ValueError(f"Unknown tool: {name}")

    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be an object")

    if name == "list_files":
        if parsed:
            raise ValueError("list_files requires an empty object")
        return list_files(workspace)

    if set(parsed) != {"path"}:
        raise ValueError("read_file requires exactly one path argument")

    path = parsed["path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a nonempty string")

    return read_file(workspace, path)
