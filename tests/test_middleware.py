"""Tests for research limit middleware."""

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from research_agent.middleware import ResearchLimitsMiddleware


@dataclass
class FakeRuntime:
    """Minimal runtime shape used by middleware tests."""

    state: dict[str, Any]


@dataclass
class FakeRequest:
    """Minimal tool-call request shape used by middleware tests."""

    tool_call: dict[str, Any]
    runtime: FakeRuntime


def _request(
    name: str,
    args: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> FakeRequest:
    return FakeRequest(
        tool_call={"id": "call-1", "name": name, "args": args or {}},
        runtime=FakeRuntime(state=state or {}),
    )


def _handler(request: FakeRequest) -> ToolMessage:
    return ToolMessage(
        content="tool ok",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
    )


def _command_handler(request: FakeRequest) -> Command:
    return Command(
        update={
            "research_limit_counts": {"web_search": 2},
            "messages": [
                ToolMessage(
                    content="subagent ok",
                    tool_call_id=request.tool_call["id"],
                    name=request.tool_call["name"],
                )
            ],
        }
    )


def _model_request(state: dict[str, Any] | None = None) -> ModelRequest:
    return ModelRequest(
        model=None,
        messages=[],
        runtime=FakeRuntime(state=state or {}),
        state=state or {},
    )


def _model_handler(request: ModelRequest) -> ModelResponse:
    assert request.system_message is not None
    assert "Runtime Research Budgets" in request.system_message.text
    assert "orchestrator model calls" not in request.system_message.text
    return ModelResponse(result=[AIMessage(content="model ok")])


def test_allows_web_search_under_budget() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=1,
        max_task_calls=1,
    )

    result = middleware.wrap_tool_call(_request("web_search"), _handler)

    assert isinstance(result, Command)
    assert result.update["research_limit_counts"] == {"web_search": 1}
    assert result.update["messages"][0].content == "tool ok"


def test_blocks_web_search_at_budget() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=1,
        max_task_calls=1,
    )
    state = {"research_limit_counts": {"web_search": 1}}

    result = middleware.wrap_tool_call(_request("web_search", state=state), _handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "research budget exceeded for web_search" in result.content


def test_blocks_research_agent_task_at_budget() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=5,
        max_task_calls=1,
    )
    state = {"research_limit_counts": {"research_agent_tasks": 1}}
    request = _request(
        "task",
        args={"subagent_type": "research-agent", "description": "AI infra demand"},
        state=state,
    )

    result = middleware.wrap_tool_call(request, _handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "research budget exceeded for research-agent delegation" in result.content


def test_blocks_unsupported_task_delegation() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=5,
        max_task_calls=5,
    )
    request = _request(
        "task",
        args={"subagent_type": "general-purpose", "description": "Research with GP"},
    )

    result = middleware.wrap_tool_call(request, _handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "research budget exceeded for unsupported subagent delegation" in result.content


def test_command_results_keep_existing_counter_updates() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=5,
        max_task_calls=3,
    )
    request = _request(
        "task",
        args={"subagent_type": "research-agent", "description": "AI infra demand"},
    )

    result = middleware.wrap_tool_call(request, _command_handler)

    assert isinstance(result, Command)
    assert result.update["research_limit_counts"] == {
        "web_search": 2,
        "research_agent_tasks": 1,
    }
    assert result.update["messages"][0].content == "subagent ok"


def test_allows_model_call_under_budget() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=0,
        max_task_calls=3,
        max_model_calls=2,
        model_call_counter="orchestrator_model_calls",
    )

    result = middleware.wrap_model_call(_model_request(), _model_handler)

    assert result.command.update["research_limit_counts"] == {
        "orchestrator_model_calls": 1
    }
    assert result.model_response.result[0].content == "model ok"


def test_model_call_without_model_budget_still_adds_budget_notice() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=5,
        max_task_calls=0,
    )

    result = middleware.wrap_model_call(_model_request(), _model_handler)

    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "model ok"


def test_blocks_model_call_at_budget() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=0,
        max_task_calls=3,
        max_model_calls=1,
        model_call_counter="orchestrator_model_calls",
    )
    state = {"research_limit_counts": {"orchestrator_model_calls": 1}}

    def handler(request: ModelRequest) -> ModelResponse:  # noqa: ARG001
        raise AssertionError("handler should not be called")

    result = middleware.wrap_model_call(_model_request(state=state), handler)

    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "Orchestrator model call budget exhausted. Ending this run."


def test_counts_researcher_model_calls() -> None:
    middleware = ResearchLimitsMiddleware(
        max_search_calls=5,
        max_task_calls=0,
        max_model_calls=2,
        model_call_counter="researcher_model_calls",
    )

    result = middleware.wrap_model_call(_model_request(), _model_handler)

    assert result.command.update["research_limit_counts"] == {
        "researcher_model_calls": 1
    }
