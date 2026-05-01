"""Custom middleware for enforcing research budgets."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, TypedDict

from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import Annotated


class ResearchLimitCounts(TypedDict, total=False):
    """Accumulated research tool usage counters."""

    web_search: int
    research_agent_tasks: int


def _counter_reducer(
    left: ResearchLimitCounts | None,
    right: ResearchLimitCounts | None,
) -> ResearchLimitCounts:
    """Merge counter deltas into accumulated state."""
    counts: ResearchLimitCounts = dict(left or {})
    for key, value in (right or {}).items():
        counts[key] = counts.get(key, 0) + value
    return counts


class ResearchLimitsState(AgentState):
    """State extension for research limit counters."""

    research_limit_counts: Annotated[NotRequired[ResearchLimitCounts], _counter_reducer]


class ResearchLimitsMiddleware(AgentMiddleware[ResearchLimitsState, Any, Any]):
    """Enforce hard budgets for research-related tool calls.

    This middleware tracks successful `web_search` calls and `task` calls to the
    `research-agent` subagent. It blocks calls once the configured limit has
    already been reached.
    """

    state_schema = ResearchLimitsState

    def __init__(self, *, max_search_calls: int, max_task_calls: int) -> None:
        """Create research limit middleware.

        Args:
            max_search_calls: Maximum successful `web_search` calls per graph run.
            max_task_calls: Maximum successful `task` calls to `research-agent`.

        Raises:
            ValueError: If either limit is negative.
        """
        if max_search_calls < 0:
            msg = f"max_search_calls must be non-negative, got {max_search_calls}"
            raise ValueError(msg)
        if max_task_calls < 0:
            msg = f"max_task_calls must be non-negative, got {max_task_calls}"
            raise ValueError(msg)
        self.max_search_calls = max_search_calls
        self.max_task_calls = max_task_calls

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Block research tool calls that would exceed configured budgets."""
        counter = self._counter_for_tool_call(request.tool_call)
        if counter is None:
            return handler(request)

        blocked = self._blocked_message(request, counter)
        if blocked is not None:
            return blocked

        result = handler(request)
        return self._with_counter_delta(result, counter)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of `wrap_tool_call`."""
        counter = self._counter_for_tool_call(request.tool_call)
        if counter is None:
            return await handler(request)

        blocked = self._blocked_message(request, counter)
        if blocked is not None:
            return blocked

        result = await handler(request)
        return self._with_counter_delta(result, counter)

    @staticmethod
    def _counter_for_tool_call(tool_call: dict[str, Any]) -> str | None:
        name = tool_call.get("name")
        if name == "web_search":
            return "web_search"
        if name != "task":
            return None

        args = tool_call.get("args") or {}
        if args.get("subagent_type") == "research-agent":
            return "research_agent_tasks"
        return None

    def _blocked_message(
        self,
        request: ToolCallRequest,
        counter: str,
    ) -> ToolMessage | None:
        counts = request.runtime.state.get("research_limit_counts") or {}
        current = counts.get(counter, 0)
        limit = self._limit_for_counter(counter)
        if current < limit:
            return None

        return ToolMessage(
            content=(
                f"Error: research budget exceeded for {self._label_for_counter(counter)}. "
                f"Limit is {limit} successful call(s) for this run."
            ),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call.get("name"),
            status="error",
        )

    def _limit_for_counter(self, counter: str) -> int:
        if counter == "web_search":
            return self.max_search_calls
        return self.max_task_calls

    @staticmethod
    def _label_for_counter(counter: str) -> str:
        if counter == "web_search":
            return "web_search"
        return "research-agent delegation"

    @staticmethod
    def _with_counter_delta(result: ToolMessage | Command, counter: str) -> ToolMessage | Command:
        delta: ResearchLimitCounts = {counter: 1}  # type: ignore[typeddict-unknown-key]
        if isinstance(result, Command):
            update = result.update or {}
            return Command(update={**update, "research_limit_counts": delta})

        return Command(
            update={
                "research_limit_counts": delta,
                "messages": [result],
            }
        )
