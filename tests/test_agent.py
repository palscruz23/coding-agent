"""Responses protocol checks using a fake client and real file dispatcher."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from myagent import agent


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
        (response(calls=[tool_call(), tool_call(call_id="call_other")]), "Multiple tool calls"),
    ],
    ids=["incomplete", "empty", "multiple-tool-calls"],
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


@pytest.mark.parametrize("final_requests_tool", [False, True], ids=["answer", "tool"])
def test_final_turn_disables_tools_and_enforces_limit(
    fake_api, monkeypatch, tmp_path, final_requests_tool
):
    """Check final-turn answering and rejection of an unexpected fifth tool.

    Args:
        fake_api (AsyncMock): Mock Responses API create method.
        monkeypatch (pytest.MonkeyPatch): Fixture replacing tool dispatch.
        tmp_path (Path): Disposable workspace passed to the agent.
        final_requests_tool (bool): Whether turn five violates tool_choice none.

    Returns:
        None: Assertions verify five model requests and four tool executions.
    """
    dispatch = Mock(return_value=[])
    monkeypatch.setattr(agent, "execute_tool", dispatch)
    fake_api.side_effect = [
        response(
            calls=[tool_call(call_id=f"call_{index}")],
            response_id=f"resp_{index}",
        )
        for index in range(4)
    ] + [
        response(calls=[tool_call(call_id="call_forbidden")])
        if final_requests_tool
        else response(text="Finished", response_id="resp_final")
    ]

    if final_requests_tool:
        with pytest.raises(RuntimeError, match="model-turn limit"):
            asyncio.run(agent.respond("Keep inspecting", None, tmp_path))
    else:
        assert asyncio.run(agent.respond("Keep inspecting", None, tmp_path)) == (
            "Finished", "resp_final"
        )

    assert fake_api.await_count == 5
    assert dispatch.call_count == 4
    requests = [entry.kwargs for entry in fake_api.await_args_list]
    assert [request["tool_choice"] for request in requests] == [
        "auto", "auto", "auto", "auto", "none"
    ]
    assert all(request["parallel_tool_calls"] is False for request in requests)
    for index, request in enumerate(requests[1:]):
        assert request["previous_response_id"] == f"resp_{index}"
        assert request["input"][0]["call_id"] == f"call_{index}"


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
