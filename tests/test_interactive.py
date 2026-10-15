"""Offline checks of async terminal input and request recovery."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

from coding_agent import cli


@pytest.fixture
def session_client(monkeypatch):
    """Supply a fake session client and verify its context-manager lifecycle.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture replacing CLI client creation.

    Yields:
        Mock: Shared client expected in every agent request during the session.
    """
    client = Mock()
    manager = AsyncMock()
    manager.__aenter__.return_value = client
    manager.__aexit__.return_value = False
    factory = Mock(return_value=manager)
    monkeypatch.setattr(cli, "AsyncOpenAI", factory)

    yield client

    factory.assert_called_once_with()
    manager.__aenter__.assert_awaited_once_with()
    manager.__aexit__.assert_awaited_once()


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
    session_client,
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
            client=session_client,
        )
        for i, message in enumerate(messages)
    ]
    output = capsys.readouterr().out
    assert output.count("Agent>") == len(messages)
    for i in range(len(messages)):
        assert f"Agent> answer {i}" in output
    assert "Agent> /exit" not in output


def test_slash_commands_stay_local(monkeypatch, capsys, tmp_path, session_client):
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
    assert "/exit   Exit Coding Agent" in output
    assert f"Workspace: {tmp_path}" in output
    assert "Unknown command: /status" not in output
    assert "Unknown command: /foo. Type /help for commands." in output
    assert "Unknown command: /unknown. Type /help for commands." in output


def test_runtime_error_returns_to_prompt_without_retry(
    monkeypatch, capsys, tmp_path, session_client
):
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
        call(message="first", previous_response_id=None, workspace=tmp_path,
             client=session_client),
        call(message="fails", previous_response_id="resp_first", workspace=tmp_path,
             client=session_client),
        call(message="next", previous_response_id="resp_first", workspace=tmp_path,
             client=session_client),
    ]
    output = capsys.readouterr().out
    assert output.count("Agent error: LLM returned no final text") == 1
    assert output.count("first answer") == 1
    assert "Agent> next answer" in output


def test_first_request_failure_does_not_crash_loop(
    monkeypatch, capsys, tmp_path, session_client
):
    session = Mock(prompt_async=AsyncMock(side_effect=["fails", "next", "/exit"]))
    monkeypatch.setattr(cli, "PromptSession", lambda: session)
    respond = AsyncMock(side_effect=[RuntimeError("failed"), ("recovered", "resp_next")])
    monkeypatch.setattr(cli, "respond", respond)

    asyncio.run(cli.interactive_loop(tmp_path))

    assert respond.await_args_list == [
        call(message="fails", previous_response_id=None, workspace=tmp_path,
             client=session_client),
        call(message="next", previous_response_id=None, workspace=tmp_path,
             client=session_client),
    ]
    output = capsys.readouterr().out
    assert output.count("Agent error: failed") == 1
    assert "Agent> recovered" in output


def test_unexpected_error_exits_client_context(monkeypatch, tmp_path, session_client):
    """Verify that an unhandled request error unwinds the client context.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture replacing prompt and responder.
        tmp_path (Path): Disposable workspace supplied to the interactive loop.
        session_client (Mock): Fake client whose fixture checks context cleanup.

    Returns:
        None: Assertions verify error propagation and use of the shared client.
    """
    session = Mock(prompt_async=AsyncMock(side_effect=["hello"]))
    monkeypatch.setattr(cli, "PromptSession", lambda: session)
    responder = AsyncMock(side_effect=ValueError("Unexpected failure"))
    monkeypatch.setattr(cli, "respond", responder)

    with pytest.raises(ValueError, match="Unexpected failure"):
        asyncio.run(cli.interactive_loop(tmp_path))

    responder.assert_awaited_once_with(
        message="hello", previous_response_id=None, workspace=tmp_path,
        client=session_client,
    )


def test_main_starts_interactive_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["coding-agent"])
    loop = AsyncMock()
    monkeypatch.setattr(cli, "interactive_loop", loop)

    assert cli.main() == 0
    loop.assert_awaited_once_with(tmp_path.resolve())
