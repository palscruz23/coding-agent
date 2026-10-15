"""Smoke tests for the installed Coding Agent console command."""

import subprocess
from pathlib import Path

import pytest


def test_version() -> None:
    result = subprocess.run(
        ["coding-agent", "--version"],
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "coding-agent 0.1.0"


def test_missing_workspace(tmp_path: Path) -> None:
    result = subprocess.run(
        ["coding-agent", "--workspace", str(tmp_path / "missing")],
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 2, result.stderr
    assert "Workspace must be an existing directory" in result.stderr


def test_explicit_workspace(tmp_path: Path) -> None:
    result = subprocess.run(
        ["coding-agent", "--workspace", str(tmp_path.resolve())],
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert str(tmp_path.resolve()) in [line.strip() for line in result.stdout.splitlines()]


def test_default_workspace(tmp_path: Path) -> None:
    result = subprocess.run(
        ["coding-agent"],
        cwd=tmp_path,
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert str(tmp_path.resolve()) in [line.strip() for line in result.stdout.splitlines()]


def test_help() -> None:
    result = subprocess.run(
        ["coding-agent", "--help"], input="", capture_output=True, text=True, timeout=5
    )

    assert result.returncode == 0, result.stderr
    assert "--version" in result.stdout
    assert "--workspace" in result.stdout


def test_unknown_argument() -> None:
    result = subprocess.run(
        ["coding-agent", "--unknown"], input="", capture_output=True, text=True, timeout=5
    )

    assert result.returncode == 2, result.stderr
    assert "unrecognized arguments: --unknown" in result.stderr


def test_file_workspace(tmp_path: Path) -> None:
    workspace_file = tmp_path / "example.txt"
    workspace_file.write_text("example", encoding="utf-8")
    result = subprocess.run(
        ["coding-agent", "--workspace", str(workspace_file)],
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 2, result.stderr
    assert "Workspace must be an existing directory" in result.stderr


def test_relative_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project with spaces"
    workspace.mkdir()
    result = subprocess.run(
        ["coding-agent", "--workspace", workspace.name],
        cwd=tmp_path,
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert str(workspace.resolve()) in [line.strip() for line in result.stdout.splitlines()]


@pytest.mark.parametrize("project_name", ["project-one", "project-two"])
def test_launch_from_other_project(tmp_path: Path, project_name: str) -> None:
    project = tmp_path / project_name
    project.mkdir()
    result = subprocess.run(
        ["coding-agent"],
        cwd=project,
        input="",
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert str(project.resolve()) in [line.strip() for line in result.stdout.splitlines()]
