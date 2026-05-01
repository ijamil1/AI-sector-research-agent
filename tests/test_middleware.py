"""Tests for research limit middleware."""

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
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


def _request(name: str, args: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> FakeRequest:
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


def test_allows_web_search_under_budget() -> None:
    middleware = ResearchLimitsMiddleware(max_search_calls=1, max_task_calls=1)

    result = middleware.wrap_tool_call(_request("web_search"), _handler)

    assert isinstance(result, Command)
    assert result.update["research_limit_counts"] == {"web_search": 1}
    assert result.update["messages"][0].content == "tool ok"


def test_blocks_web_search_at_budget() -> None:
    middleware = ResearchLimitsMiddleware(max_search_calls=1, max_task_calls=1)
    state = {"research_limit_counts": {"web_search": 1}}

    result = middleware.wrap_tool_call(_request("web_search", state=state), _handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "research budget exceeded for web_search" in result.content


def test_blocks_research_agent_task_at_budget() -> None:
    middleware = ResearchLimitsMiddleware(max_search_calls=5, max_task_calls=1)
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


def test_unrelated_tool_is_not_counted() -> None:
    middleware = ResearchLimitsMiddleware(max_search_calls=0, max_task_calls=0)

    result = middleware.wrap_tool_call(_request("think"), _handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "tool ok"
