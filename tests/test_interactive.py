"""Offline checks of async terminal input and request recovery."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from myagent import cli


@pytest.mark.parametrize(
    ("inputs", "messages"),
    [
        (["hello", "second request", "/exit"], ["hello", "second request"]),
        (["", "   ", " hello ", "/exit"], ["hello"]),
        (["/exit"], []),
        ([EOFError()], []),
        ([KeyboardInterrupt(), "hello", "/exit"], []),
    ],
    ids=["multiple-requests", "blank-input", "exit-command", "eof", "ctrl-c"],
)
def test_interactive_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    inputs: list[str | BaseException],
    messages: list[str],
) -> None:
    session = Mock(prompt_async=AsyncMock(side_effect=inputs))
    monkeypatch.setattr(cli, "PromptSession", lambda: session)

    respond = AsyncMock(
        side_effect=[(f"answer {i}", f"resp_{i}") for i in range(len(messages))]
    )
    monkeypatch.setattr(cli, "respond", respond)

    asyncio.run(cli.interactive_loop(tmp_path))

    assert respond.await_args_list == [
        call(
            message=message,
            previous_response_id=None if i == 0 else f"resp_{i - 1}",
            workspace=tmp_path,
        )
        for i, message in enumerate(messages)
    ]
    output = capsys.readouterr().out
    assert output.count("agent>") == len(messages)
    for i in range(len(messages)):
        assert f"agent> answer {i}" in output
    assert "agent> /exit" not in output


def test_slash_commands_stay_local(monkeypatch, capsys, tmp_path):
    session = Mock(prompt_async=AsyncMock(
        side_effect=["/help", "/status", "/foo", "/unknown", "/exit"]
    ))
    monkeypatch.setattr(cli, "PromptSession", lambda: session)
    respond = AsyncMock()
    monkeypatch.setattr(cli, "respond", respond)

    asyncio.run(cli.interactive_loop(tmp_path))

    respond.assert_not_awaited()
    output = capsys.readouterr().out
    assert "/help   Show commands" in output
    assert "/exit   Exit MyAgent" in output
    assert f"Workspace: {tmp_path}" in output
    assert "Unknown command: /status" not in output
    assert "Unknown command: /foo. Type /help for commands." in output
    assert "Unknown command: /unknown. Type /help for commands." in output


def test_runtime_error_returns_to_prompt_without_retry(monkeypatch, capsys, tmp_path):
    session = Mock(
        prompt_async=AsyncMock(side_effect=["first", "fails", "next", "/exit"])
    )
    monkeypatch.setattr(cli, "PromptSession", lambda: session)
    respond = AsyncMock(side_effect=[
        ("first answer", "resp_first"),
        RuntimeError("LLM returned no final text"),
        ("next answer", "resp_next"),
    ])
    monkeypatch.setattr(cli, "respond", respond)

    asyncio.run(cli.interactive_loop(tmp_path))

    assert respond.await_args_list == [
        call(message="first", previous_response_id=None, workspace=tmp_path),
        call(message="fails", previous_response_id="resp_first", workspace=tmp_path),
        call(message="next", previous_response_id="resp_first", workspace=tmp_path),
    ]
    output = capsys.readouterr().out
    assert output.count("Agent error: LLM returned no final text") == 1
    assert output.count("first answer") == 1
    assert "agent> next answer" in output


def test_first_request_failure_does_not_crash_loop(monkeypatch, capsys, tmp_path):
    session = Mock(prompt_async=AsyncMock(side_effect=["fails", "next", "/exit"]))
    monkeypatch.setattr(cli, "PromptSession", lambda: session)
    respond = AsyncMock(side_effect=[RuntimeError("failed"), ("recovered", "resp_next")])
    monkeypatch.setattr(cli, "respond", respond)

    asyncio.run(cli.interactive_loop(tmp_path))

    assert respond.await_args_list == [
        call(message="fails", previous_response_id=None, workspace=tmp_path),
        call(message="next", previous_response_id=None, workspace=tmp_path),
    ]
    output = capsys.readouterr().out
    assert output.count("Agent error: failed") == 1
    assert "agent> recovered" in output


def test_main_starts_interactive_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["myagent"])
    loop = AsyncMock()
    monkeypatch.setattr(cli, "interactive_loop", loop)

    assert cli.main() == 0
    loop.assert_awaited_once_with(tmp_path.resolve())
