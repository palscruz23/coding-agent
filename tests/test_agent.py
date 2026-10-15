"""Responses protocol checks using a fake client and real file dispatcher."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from coding_agent import agent


def response(*, text="", calls=(), status="completed", response_id="resp_first"):
    return SimpleNamespace(
        status=status, output=list(calls), output_text=text, id=response_id
    )


def tool_call(*, name="list_files", arguments="{}", call_id="call_files"):
    return SimpleNamespace(
        type="function_call", name=name, arguments=arguments, call_id=call_id
    )


@pytest.fixture
def fake_api(monkeypatch):
    create = AsyncMock()
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(agent, "AsyncOpenAI", lambda: client)
    return create


@pytest.mark.parametrize("previous_id", [None, "resp_previous"])
def test_final_answer_preserves_conversation_id(fake_api, tmp_path, previous_id):
    fake_api.return_value = response(text="Hello", response_id="resp_answer")

    result = asyncio.run(agent.respond("hello", previous_id, tmp_path))

    assert result == ("Hello", "resp_answer")
    assert fake_api.await_count == 1
    request = fake_api.await_args.kwargs
    assert request["input"] == "hello"
    if previous_id is None:
        assert "previous_response_id" not in request
    else:
        assert request["previous_response_id"] == previous_id


@pytest.mark.parametrize("supply_client", [False, True], ids=["default", "supplied"])
def test_client_selection_persists_across_model_turns(
    monkeypatch, tmp_path, supply_client
):
    """Check client selection and reuse across a tool request and final answer.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture replacing the SDK constructor.
        tmp_path (Path): Empty disposable workspace supplied by pytest.
        supply_client (bool): Whether to pass a client explicitly to respond.

    Returns:
        None: Assertions verify client construction and both model requests.
    """
    create = AsyncMock(side_effect=[
        response(calls=[tool_call()], response_id="resp_tool"),
        response(text="Finished", response_id="resp_final"),
    ])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    factory = Mock(return_value=client)
    monkeypatch.setattr(agent, "AsyncOpenAI", factory)
    options = {"client": client} if supply_client else {}

    result = asyncio.run(agent.respond("List files", None, tmp_path, **options))

    assert result == ("Finished", "resp_final")
    assert create.await_count == 2
    if supply_client:
        factory.assert_not_called()
    else:
        factory.assert_called_once_with()
    follow_up = create.await_args_list[1].kwargs
    assert follow_up["previous_response_id"] == "resp_tool"
    assert follow_up["input"][0]["call_id"] == "call_files"


def test_tool_result_returns_to_model_with_matching_ids(fake_api, tmp_path):
    (tmp_path / "main.py").write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    fake_api.side_effect = [
        response(calls=[tool_call()], response_id="resp_tool"),
        response(text="Found main.py and src/", response_id="resp_answer"),
    ]

    result = asyncio.run(agent.respond("list the workspace", "resp_previous", tmp_path))

    assert result == ("Found main.py and src/", "resp_answer")
    assert fake_api.await_count == 2
    first, second = [entry.kwargs for entry in fake_api.await_args_list]
    assert first["previous_response_id"] == "resp_previous"
    assert second["previous_response_id"] == "resp_tool"
    assert len(second["input"]) == 1
    item = second["input"][0]
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_files"
    assert json.loads(item["output"]) == ["main.py", "src/"]
    assert second["tool_choice"] == "auto"


@pytest.mark.parametrize(
    ("bad_response", "error"),
    [
        (response(status="incomplete"), "status: incomplete"),
        (response(), "no text"),
        (response(calls=[tool_call(call_id=f"call_{i}") for i in range(5)]), "per-turn tool-call limit"),
    ],
    ids=["incomplete", "empty", "oversized-batch"],
)
def test_invalid_first_response_fails_before_dispatch(
    fake_api, monkeypatch, tmp_path, bad_response, error
):
    fake_api.return_value = bad_response

    def unexpected_dispatch(*args):
        pytest.fail("Invalid response must not execute tools")

    monkeypatch.setattr(agent, "execute_tool", unexpected_dispatch)
    with pytest.raises(RuntimeError, match=error):
        asyncio.run(agent.respond("hello", None, tmp_path))
    assert fake_api.await_count == 1


@pytest.mark.parametrize(
    ("bad_response", "error"),
    [
        (response(status="incomplete"), "status: incomplete"),
        (response(), "no text"),
    ],
    ids=["incomplete-follow-up", "empty-follow-up"],
)
def test_invalid_follow_up_is_not_returned_as_answer(fake_api, tmp_path, bad_response, error):
    fake_api.side_effect = [response(calls=[tool_call()]), bad_response]
    with pytest.raises(RuntimeError, match=error):
        asyncio.run(agent.respond("list files", None, tmp_path))
    assert fake_api.await_count == 2


def test_successive_tool_turns(fake_api, tmp_path):
    """Check that successive model turns receive their matching tool results.

    Args:
        fake_api (AsyncMock): Mock Responses API create method.
        tmp_path (Path): Empty disposable workspace supplied by pytest.

    Returns:
        None: Assertions check the answer, request count, and result identities.
    """
    fake_api.side_effect = [
        response(calls=[tool_call(call_id="call_first")], response_id="resp_first"),
        response(calls=[tool_call(call_id="call_second")], response_id="resp_second"),
        response(text="Finished", response_id="resp_final"),
    ]

    result = asyncio.run(agent.respond("List files twice", None, tmp_path))

    assert result == ("Finished", "resp_final")
    assert fake_api.await_count == 3
    requests = [entry.kwargs for entry in fake_api.await_args_list]
    for request, response_id, call_id in [
        (requests[1], "resp_first", "call_first"),
        (requests[2], "resp_second", "call_second"),
    ]:
        assert request["previous_response_id"] == response_id
        assert request["tool_choice"] == "auto"
        assert request["input"] == [{
            "type": "function_call_output",
            "call_id": call_id,
            "output": "[]",
        }]


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("invalid_first", [False, True])
def test_batch_results_preserve_identity_after_tool_failure(
    fake_api, tmp_path, capsys, batch_size, invalid_first
):
    """Verify batch boundaries, result identity, and continuation after errors.

    Args:
        fake_api (AsyncMock): Fake Responses API create method.
        tmp_path (Path): Empty disposable workspace for successful list calls.
        capsys (pytest.CaptureFixture[str]): Captures user-facing execution logs.
        batch_size (int): Number of tool calls in an accepted batch.
        invalid_first (bool): Whether the first call has malformed JSON.

    Returns:
        None: Assertions check every result and the literal action description.
    """
    calls = [
        tool_call(
            call_id=f"call_{index}",
            arguments="{" if invalid_first and index == 0 else "{}",
        )
        for index in range(batch_size)
    ]
    fake_api.side_effect = [
        response(text="Inspect [files].", calls=calls, response_id="resp_batch"),
        response(text="Finished", response_id="resp_final"),
    ]

    assert asyncio.run(agent.respond("Inspect files", None, tmp_path)) == (
        "Finished", "resp_final"
    )
    assert fake_api.await_count == 2
    follow_up = fake_api.await_args_list[1].kwargs
    assert follow_up["previous_response_id"] == "resp_batch"
    outputs = follow_up["input"]
    assert len(outputs) == batch_size
    for index, item in enumerate(outputs):
        assert item["type"] == "function_call_output"
        assert item["call_id"] == f"call_{index}"
        result = json.loads(item["output"])
        if invalid_first and index == 0:
            assert result["error"]["type"] == "JSONDecodeError"
        else:
            assert result == []
    output = capsys.readouterr().out
    assert "Step 1: Inspect [files]." in output
    assert output.count("Running list_files:") == batch_size


@pytest.mark.parametrize("final_requests_tool", [False, True], ids=["answer", "tool"])
@pytest.mark.parametrize("max_steps", [1, 3, 5])
def test_final_turn_disables_tools_and_enforces_limit(
    fake_api, monkeypatch, tmp_path, final_requests_tool, max_steps
):
    """Check final-turn answering and rejection of a tool beyond the budget.

    Args:
        fake_api (AsyncMock): Mock Responses API create method.
        monkeypatch (pytest.MonkeyPatch): Fixture replacing tool dispatch.
        tmp_path (Path): Disposable workspace passed to the agent.
        final_requests_tool (bool): Whether the final turn requests a tool.
        max_steps (int): Positive model-turn budget under test.

    Returns:
        None: Assertions verify the model budget and one fewer tool executions.
    """
    dispatch = Mock(return_value=[])
    monkeypatch.setattr(agent, "execute_tool", dispatch)
    fake_api.side_effect = [
        response(
            calls=[tool_call(call_id=f"call_{index}")],
            response_id=f"resp_{index}",
        )
        for index in range(max_steps - 1)
    ] + [
        response(calls=[tool_call(call_id="call_forbidden")])
        if final_requests_tool
        else response(text="Finished", response_id="resp_final")
    ]

    if final_requests_tool:
        with pytest.raises(RuntimeError, match="model-turn limit"):
            asyncio.run(agent.respond(
                "Keep inspecting", None, tmp_path, max_steps=max_steps
            ))
    else:
        assert asyncio.run(agent.respond(
            "Keep inspecting", None, tmp_path, max_steps=max_steps
        )) == (
            "Finished", "resp_final"
        )

    assert fake_api.await_count == max_steps
    assert dispatch.call_count == max_steps - 1
    requests = [entry.kwargs for entry in fake_api.await_args_list]
    assert [request["tool_choice"] for request in requests] == (
        ["auto"] * (max_steps - 1) + ["none"]
    )
    assert all(request["parallel_tool_calls"] is True for request in requests)
    for index, request in enumerate(requests[1:]):
        assert request["previous_response_id"] == f"resp_{index}"
        assert request["input"][0]["call_id"] == f"call_{index}"


@pytest.mark.parametrize("budget", [0, -1, True, False, 1.5, "3", None])
def test_invalid_budget_fails_before_client_creation(monkeypatch, tmp_path, budget):
    """Reject invalid budgets before creating a client or dispatching tools.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture replacing client and dispatcher.
        tmp_path (Path): Disposable workspace passed to the agent.
        budget (object): Invalid model-turn budget under test.

    Returns:
        None: Assertions verify ValueError and no client or tool activity.
    """
    client_factory = Mock()
    dispatch = Mock()
    monkeypatch.setattr(agent, "AsyncOpenAI", client_factory)
    monkeypatch.setattr(agent, "execute_tool", dispatch)

    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        asyncio.run(agent.respond("Hello", None, tmp_path, max_steps=budget))

    client_factory.assert_not_called()
    dispatch.assert_not_called()


def test_tool_validation_error_is_returned_before_recovery(fake_api, tmp_path):
    """Check that invalid arguments become a tool result and permit recovery.

    Args:
        fake_api (AsyncMock): Mock Responses API create method.
        tmp_path (Path): Empty disposable workspace used by list_files.

    Returns:
        None: Assertions verify error identity and a later successful result.
    """
    fake_api.side_effect = [
        response(
            calls=[tool_call(arguments="{", call_id="call_bad")],
            response_id="resp_bad",
        ),
        response(
            calls=[tool_call(call_id="call_retry")], response_id="resp_retry"
        ),
        response(text="Recovered", response_id="resp_final"),
    ]

    assert asyncio.run(agent.respond("List files", None, tmp_path)) == (
        "Recovered", "resp_final"
    )
    assert fake_api.await_count == 3
    requests = [entry.kwargs for entry in fake_api.await_args_list]
    failed = requests[1]["input"][0]
    assert requests[1]["previous_response_id"] == "resp_bad"
    assert failed["call_id"] == "call_bad"
    assert json.loads(failed["output"]) == {
        "error": {
            "type": "JSONDecodeError",
            "message": "The tool request failed validation or file access.",
        }
    }
    assert requests[2]["previous_response_id"] == "resp_retry"
    assert requests[2]["input"][0]["call_id"] == "call_retry"
    assert json.loads(requests[2]["input"][0]["output"]) == []
