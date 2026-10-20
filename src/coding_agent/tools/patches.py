from difflib import unified_diff
from pathlib import Path

from coding_agent.tools.files import MAX_FILE_BYTES, read_file


def propose_patch(
    workspace: Path,
    path: str,
    new_content: str,
) -> dict[str, str]:
    """Generate a unified diff for an existing permitted text file.

    Reads the current file and compares it with proposed replacement text.
    Does not write files or apply the generated patch.

    Args:
        workspace (Path): Directory that bounds file access.
        path (str): Workspace-relative path to an existing permitted file.
        new_content (str): Complete proposed replacement text, limited to
            MAX_FILE_BYTES when encoded as UTF-8 and containing no null bytes.

    Returns:
        dict[str, str]: Resolved relative path and unified diff under the
            'patch' key. The patch is empty when the content is unchanged.
            Missing final newlines are marked explicitly.

    Raises:
        ValueError: Replacement text is invalid or exceeds the byte limit,
            or read_file rejects the current file's content.
        PermissionError: Reading the file violates the workspace policy.
        OSError: A filesystem operation fails.
    """
    if not isinstance(new_content, str):
        raise ValueError("new_content must be a string")

    if "\x00" in new_content:
        raise ValueError("Replacement text must not contain null bytes")

    try:
        encoded = new_content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Replacement text must support UTF-8") from exc

    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError("Replacement text is too large")

    document = read_file(workspace, path)
    relative = document["path"]

    lines = unified_diff(
        document["content"].splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{relative}",
        tofile=f"b/{relative}",
    )

    patch_parts = []
    for line in lines:
        patch_parts.append(line)
        if not line.endswith("\n"):
            patch_parts.append("\n\\ No newline at end of file\n")

    return {"path": relative, "patch": "".join(patch_parts)}
