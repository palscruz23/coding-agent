"""Single-file text search behavior, file policy, and tool integration."""

import json

import pytest

from coding_agent.tools.files import MAX_FILE_BYTES, execute_tool, search_text


@pytest.mark.parametrize(
    ("query", "limit", "lines", "truncated"),
    [
        ("needle", 20, [2, 4], False),
        ("missing", 20, [], False),
        ("NEEDLE", 20, [3], False),
        ("needle", 2, [2, 4], False),
        ("needle", 1, [2], True),
        (".*", 20, [5], False),
    ],
)
def test_search_matches_and_truncation(tmp_path, query, limit, lines, truncated):
    """Verify literal matching, line numbers, and exact versus exceeded limits.

    Args:
        tmp_path (Path): Disposable workspace supplied by pytest.
        query (str): Literal search term.
        limit (int): Maximum returned matching lines.
        lines (list[int]): Expected one-based matching line numbers.
        truncated (bool): Whether additional matching lines must exist.

    Returns:
        None: Assertions verify the complete search result.
    """
    content = "first\nneedle needle\nNEEDLE\nlast needle\nliteral .*\n"
    (tmp_path / "example.py").write_text(content, encoding="utf-8")

    assert search_text(tmp_path, "example.py", query, max_matches=limit) == {
        "path": "example.py",
        "matches": [
            {"line": number, "text": content.splitlines()[number - 1]}
            for number in lines
        ],
        "truncated": truncated,
    }


@pytest.mark.parametrize("query", ["", None, 1])
def test_search_rejects_invalid_query(tmp_path, query):
    """Reject invalid queries before attempting a file read.

    Args:
        tmp_path (Path): Disposable workspace with no requested file.
        query (object): Invalid query value.

    Returns:
        None: Asserts that validation raises ValueError.
    """
    with pytest.raises(ValueError, match="query"):
        search_text(tmp_path, "missing.py", query)


@pytest.mark.parametrize("limit", [0, -1, 101, True, False, 1.5, "2", None])
def test_search_rejects_invalid_limit(tmp_path, limit):
    """Reject invalid result limits before attempting a file read.

    Args:
        tmp_path (Path): Disposable workspace with no requested file.
        limit (object): Invalid result limit.

    Returns:
        None: Asserts that validation raises ValueError.
    """
    with pytest.raises(ValueError, match="max_matches"):
        search_text(tmp_path, "missing.py", "text", max_matches=limit)


@pytest.mark.parametrize("kind", ["secret", "traversal", "absolute", "symlink", "binary", "oversized"])
def test_search_preserves_file_read_boundaries(tmp_path, kind):
    """Ensure search inherits blocked paths and content limits from read_file.

    Args:
        tmp_path (Path): Disposable parent of the workspace and outside fixture.
        kind (str): File access or content boundary to exercise.

    Returns:
        None: Asserts the appropriate read failure instead of returning content.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("needle", encoding="utf-8")
    path = "example.txt"
    expected = OSError
    if kind == "secret":
        path = ".env"
        (workspace / path).write_text("needle", encoding="utf-8")
    elif kind == "traversal":
        path = "../outside.txt"
    elif kind == "absolute":
        path = str(outside)
    elif kind == "symlink":
        (workspace / path).symlink_to(outside)
    elif kind == "binary":
        (workspace / path).write_bytes(b"needle\x00")
        expected = ValueError
    else:
        (workspace / path).write_bytes(b"x" * (MAX_FILE_BYTES + 1))
        expected = ValueError

    with pytest.raises(expected):
        search_text(workspace, path, "needle")


def test_search_tool_dispatch(tmp_path):
    """Verify the advertised search tool can be invoked through its dispatcher.

    Args:
        tmp_path (Path): Disposable workspace containing a permitted text file.

    Returns:
        None: Asserts a structured search result for valid JSON arguments.
    """
    (tmp_path / "example.py").write_text("def main():\n", encoding="utf-8")
    assert execute_tool(
        tmp_path, "search_text", json.dumps({"path": "example.py", "query": "main"})
    ) == {
        "path": "example.py",
        "matches": [{"line": 1, "text": "def main():"}],
        "truncated": False,
    }
