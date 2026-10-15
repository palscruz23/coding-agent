"""Read-only listing and dispatcher validation boundaries."""

import errno
import os

import pytest
from pathlib import Path

from myagent.tools.files import MAX_FILE_BYTES, allowed_file, execute_tool, list_files, read_file, safe_path


def test_listing_is_sorted_and_nonrecursive(tmp_path):
    (tmp_path / "z.py").write_text("# fixture\n", encoding="utf-8")
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "nested.py").write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("# fixture\n", encoding="utf-8")

    assert list_files(tmp_path) == ["a.py", "src/", "z.py"]
    assert execute_tool(tmp_path, "list_files", "{}") == ["a.py", "src/", "z.py"]


def test_listing_skips_file_directory_and_broken_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "visible.py").write_text("# fixture\n", encoding="utf-8")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("fixture", encoding="utf-8")
    outside_folder = tmp_path / "outside"
    outside_folder.mkdir()
    (workspace / "file-link").symlink_to(outside_file)
    (workspace / "folder-link").symlink_to(outside_folder, target_is_directory=True)
    (workspace / "broken-link").symlink_to(tmp_path / "missing")

    assert list_files(workspace) == ["visible.py"]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", "{}"),
        ("list_files", "invalid JSON"),
        ("list_files", "[]"),
        ("list_files", "null"),
        ("list_files", '"text"'),
        ("list_files", '{"path": "../outside"}'),
    ],
    ids=["unknown-tool", "malformed-json", "array", "null", "string", "extra-argument"],
)
def test_dispatch_rejects_invalid_requests_before_listing(monkeypatch, tmp_path, name, arguments):
    def unexpected_listing(workspace):
        pytest.fail("Invalid request must not access the directory")

    monkeypatch.setattr("myagent.tools.files.list_files", unexpected_listing)
    with pytest.raises(ValueError):
        execute_tool(tmp_path, name, arguments)


def test_safe_path_resolves_existing_nested_file(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    file = folder / "main.py"
    file.write_text("# fixture\n", encoding="utf-8")

    assert safe_path(tmp_path, "src/main.py") == file.resolve()


@pytest.mark.parametrize("requested", ["../outside.txt", "src/../main.py"])
def test_safe_path_rejects_parent_components(tmp_path, requested):
    with pytest.raises(PermissionError):
        safe_path(tmp_path, requested)


def test_safe_path_rejects_absolute_path_even_inside_workspace(tmp_path):
    file = tmp_path / "main.py"
    file.write_text("# fixture\n", encoding="utf-8")

    with pytest.raises(PermissionError):
        safe_path(tmp_path, str(file.resolve()))


@pytest.mark.parametrize("link_kind", ["file", "directory", "prefix-sibling"])
def test_safe_path_rejects_symlink_escape(tmp_path, link_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if link_kind == "file":
        target = tmp_path / "outside.txt"
        target.write_text("fixture", encoding="utf-8")
        (workspace / "link").symlink_to(target)
        requested = "link"
    else:
        target = tmp_path / ("workspace-extra" if link_kind == "prefix-sibling" else "outside")
        target.mkdir()
        (target / "main.py").write_text("# fixture\n", encoding="utf-8")
        (workspace / "link").symlink_to(target, target_is_directory=True)
        requested = "link/main.py"

    with pytest.raises(PermissionError, match="escapes"):
        safe_path(workspace, requested)


def test_safe_path_allows_symlink_whose_target_stays_inside(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "link").symlink_to(target)

    assert safe_path(tmp_path, "link") == target.resolve()


def test_safe_path_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        safe_path(tmp_path, "missing.py")


def test_safe_path_rejects_broken_symlink(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "missing.py")

    with pytest.raises(FileNotFoundError):
        safe_path(tmp_path, "link")


def test_safe_path_rejects_missing_workspace(tmp_path):
    with pytest.raises(FileNotFoundError):
        safe_path(tmp_path / "missing", "main.py")


@pytest.mark.parametrize(
    "requested",
    ["src/main.py", "README.md", "notes.txt", "data.json", "pyproject.toml", "src/MAIN.PY"],
)
def test_file_policy_allows_supported_relative_paths(requested):
    assert allowed_file(Path(requested)) is True


@pytest.mark.parametrize(
    "requested",
    [
        ".env", ".env.local", ".ENV.production.txt", ".git/config.json",
        ".GIT/config.json", ".ssh/settings.json", ".aws/settings.toml",
        "id_rsa.txt", "private.pem", "private.key", "credentials.json",
        "src/SECRET.txt", "secrets/main.py", "credentials/settings.json",
        "image.png", "program.exe", "no_extension", "../main.py",
        "src/../main.py", "/tmp/main.py",
    ],
)
def test_file_policy_rejects_secrets_and_unsupported_paths(requested):
    assert allowed_file(Path(requested)) is False


def test_file_policy_honors_configured_deny_patterns(monkeypatch):
    monkeypatch.setattr("myagent.tools.files.DENIED_PATTERNS", ("*.json",))
    assert allowed_file(Path("data.json")) is False
    assert allowed_file(Path("main.py")) is True


def test_read_file_returns_utf8_content_and_relative_path(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    content = "# café\nprint('hello')\n"
    (folder / "main.py").write_text(content, encoding="utf-8")

    assert read_file(tmp_path, "src/main.py") == {
        "path": "src/main.py", "content": content
    }


@pytest.mark.parametrize("size", [0, MAX_FILE_BYTES, MAX_FILE_BYTES + 1])
def test_read_file_enforces_exact_byte_limit(tmp_path, size):
    file = tmp_path / "notes.txt"
    file.write_bytes(b"a" * size)

    if size > MAX_FILE_BYTES:
        with pytest.raises(ValueError, match="too large"):
            read_file(tmp_path, "notes.txt")
    else:
        assert read_file(tmp_path, "notes.txt")["content"] == "a" * size


@pytest.mark.parametrize("data", [b"hello\x00world", b"\xff\xfe"])
def test_read_file_rejects_nul_and_invalid_utf8(tmp_path, data):
    (tmp_path / "notes.txt").write_bytes(data)
    with pytest.raises(ValueError):
        read_file(tmp_path, "notes.txt")


@pytest.mark.parametrize("requested", [".env", "credentials.json", ".git/config.json", "../outside.txt"])
def test_read_file_rejects_denied_path_before_opening(monkeypatch, tmp_path, requested):
    def unexpected_open(*args, **kwargs):
        pytest.fail("Denied paths must not be opened")

    monkeypatch.setattr(Path, "open", unexpected_open)
    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(PermissionError):
        read_file(tmp_path, requested)


def test_read_file_rejects_directory(tmp_path):
    (tmp_path / "folder.txt").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        read_file(tmp_path, "folder.txt")


def test_read_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path, "missing.py")


def test_read_file_rejects_symlink_to_denied_internal_target(tmp_path):
    (tmp_path / "secret.txt").write_text("synthetic fixture", encoding="utf-8")
    (tmp_path / "notes.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(PermissionError, match="Resolved file"):
        read_file(tmp_path, "notes.txt")


def test_read_file_rejects_symlink_to_external_target(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("synthetic fixture", encoding="utf-8")
    (workspace / "notes.txt").symlink_to(target)
    with pytest.raises(PermissionError):
        read_file(workspace, "notes.txt")


def test_read_file_blocks_symlink_swap_after_validation(monkeypatch, tmp_path):
    """Simulate replacement after resolution, without threads or real secrets."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "notes.txt"
    file.write_text("original fixture", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside fixture", encoding="utf-8")

    def resolve_then_swap(root, requested):
        resolved = safe_path(root, requested)
        file.unlink()
        file.symlink_to(outside)
        return resolved

    monkeypatch.setattr("myagent.tools.files.safe_path", resolve_then_swap)
    with pytest.raises((PermissionError, OSError)):
        read_file(workspace, "notes.txt")


def test_read_file_blocks_parent_symlink_swap_before_open(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    folder = workspace / "src"
    folder.mkdir()
    (folder / "main.py").write_text("original fixture", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "main.py").write_text("outside fixture", encoding="utf-8")

    def resolve_then_swap(root, requested):
        resolved = safe_path(root, requested)
        folder.rename(workspace / "old-src")
        folder.symlink_to(outside, target_is_directory=True)
        return resolved

    monkeypatch.setattr("myagent.tools.files.safe_path", resolve_then_swap)
    with pytest.raises(OSError):
        read_file(workspace, "src/main.py")


def test_read_file_stays_anchored_after_parent_is_open(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    folder = workspace / "src"
    folder.mkdir()
    (folder / "main.py").write_text("original fixture", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "main.py").write_text("outside fixture", encoding="utf-8")
    original_open = os.open
    swapped = False

    def open_then_swap(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "src" and kwargs.get("dir_fd") is not None:
            folder.rename(workspace / "old-src")
            folder.symlink_to(outside, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(os, "open", open_then_swap)
    result = read_file(workspace, "src/main.py")

    assert swapped
    assert result == {"path": "src/main.py", "content": "original fixture"}


@pytest.mark.parametrize("regular_file", [True, False], ids=["success", "failure"])
def test_read_file_closes_opened_descriptors(monkeypatch, tmp_path, regular_file):
    folder = tmp_path / "src"
    folder.mkdir()
    target = folder / "notes.txt"
    if regular_file:
        target.write_text("fixture", encoding="utf-8")
    else:
        target.mkdir()
    original_open = os.open
    descriptors = []

    def tracked_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", tracked_open)
    if regular_file:
        assert read_file(tmp_path, "src/notes.txt")["content"] == "fixture"
    else:
        with pytest.raises(ValueError, match="regular file"):
            read_file(tmp_path, "src/notes.txt")

    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF
