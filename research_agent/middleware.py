"""Custom middleware for enforcing research budgets."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from deepagents.middleware._utils import append_to_system_message
from langchain.agents import AgentState
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

ResearchModelCallCounter = Literal["orchestrator_model_calls", "researcher_model_calls"]


class ResearchLimitCounts(TypedDict, total=False):
    """Accumulated research tool usage counters."""

    web_search: int
    research_agent_tasks: int
    orchestrator_model_calls: int
    researcher_model_calls: int


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

    def __init__(
        self,
        *,
        max_search_calls: int,
        max_task_calls: int,
        max_model_calls: int | None = None,
        model_call_counter: ResearchModelCallCounter | None = None,
    ) -> None:
        """Create research limit middleware.

        Args:
            max_search_calls: Maximum successful `web_search` calls per graph run.
            max_task_calls: Maximum successful `task` calls to `research-agent`.
            max_model_calls: Maximum model calls for this middleware instance.
            model_call_counter: State counter to increment for model calls.

        Raises:
            ValueError: If either limit is negative.
        """
        if max_search_calls < 0:
            msg = f"max_search_calls must be non-negative, got {max_search_calls}"
            raise ValueError(msg)
        if max_task_calls < 0:
            msg = f"max_task_calls must be non-negative, got {max_task_calls}"
            raise ValueError(msg)
        if max_model_calls is not None and max_model_calls < 0:
            msg = f"max_model_calls must be non-negative, got {max_model_calls}"
            raise ValueError(msg)
        if (max_model_calls is None) != (model_call_counter is None):
            msg = "max_model_calls and model_call_counter must be provided together"
            raise ValueError(msg)
        self.max_search_calls = max_search_calls
        self.max_task_calls = max_task_calls
        self.max_model_calls = max_model_calls
        self.model_call_counter = model_call_counter

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Block model calls that would exceed the configured model budget."""
        request = self._with_budget_system_prompt(request)
        if self.model_call_counter is None or self.max_model_calls is None:
            return handler(request)

        blocked = self._blocked_model_response(request)
        if blocked is not None:
            return blocked

        result = handler(request)
        return self._with_counter_delta_for_model_response(result, self.model_call_counter)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Async version of `wrap_model_call`."""
        request = self._with_budget_system_prompt(request)
        if self.model_call_counter is None or self.max_model_calls is None:
            return await handler(request)

        blocked = self._blocked_model_response(request)
        if blocked is not None:
            return blocked

        result = await handler(request)
        return self._with_counter_delta_for_model_response(result, self.model_call_counter)

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
        return "unsupported_task"

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
        if counter in {"orchestrator_model_calls", "researcher_model_calls"}:
            if self.max_model_calls is None:
                msg = f"No model-call limit configured for {counter}"
                raise ValueError(msg)
            return self.max_model_calls
        if counter == "unsupported_task":
            return 0
        return self.max_task_calls

    @staticmethod
    def _label_for_counter(counter: str) -> str:
        if counter == "web_search":
            return "web_search"
        if counter == "orchestrator_model_calls":
            return "orchestrator model"
        if counter == "researcher_model_calls":
            return "researcher model"
        if counter == "unsupported_task":
            return "unsupported subagent delegation"
        return "research-agent delegation"

    def _blocked_model_response(self, request: ModelRequest[Any]) -> ModelResponse[Any] | None:
        if self.model_call_counter is None:
            return None

        counts = request.runtime.state.get("research_limit_counts") or {}
        current = counts.get(self.model_call_counter, 0)
        limit = self._limit_for_counter(self.model_call_counter)
        if current < limit:
            return None

        label = self._label_for_counter(self.model_call_counter)
        return ModelResponse(
            result=[
                AIMessage(
                    content=f"{label.capitalize()} call budget exhausted. Ending this run."
                )
            ]
        )

    def _with_budget_system_prompt(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        notice = self._budget_notice(request)
        return request.override(
            system_message=append_to_system_message(request.system_message, notice)
        )

    def _budget_notice(self, request: ModelRequest[Any]) -> str:
        counts = request.runtime.state.get("research_limit_counts") or {}
        sections = [
            "## Runtime Research Budgets",
            (
                "This section is a current runtime artifact injected by middleware, "
                "not part of the user's request. It reflects the latest budget state "
                "available immediately before this model call."
            ),
        ]

        search_current = counts.get("web_search", 0)
        sections.append(
            "- "
            f"web_search calls: {search_current}/{self.max_search_calls} used. "
            "Do not attempt web_search if this budget is exhausted or the tool is unavailable."
        )

        task_current = counts.get("research_agent_tasks", 0)
        sections.append(
            "- "
            f"research-agent delegations: {task_current}/{self.max_task_calls} used. "
            "Do not attempt delegation if this budget is exhausted or the task tool is unavailable."
        )

        sections.append(
            "Spend remaining calls deliberately. When a budget is exhausted, stop using "
            "that capability and synthesize from available evidence."
        )
        return "\n".join(sections)

    @staticmethod
    def _with_counter_delta(result: ToolMessage | Command, counter: str) -> ToolMessage | Command:
        delta: ResearchLimitCounts = {counter: 1}  # type: ignore[typeddict-unknown-key]
        if isinstance(result, Command):
            update = result.update or {}
            existing = update.get("research_limit_counts") or {}
            return Command(
                graph=result.graph,
                update={
                    **update,
                    "research_limit_counts": _counter_reducer(existing, delta),
                },
                resume=result.resume,
                goto=result.goto,
            )

        return Command(
            update={
                "research_limit_counts": delta,
                "messages": [result],
            }
        )

    @staticmethod
    def _with_counter_delta_for_model_response(
        result: ModelResponse[Any] | ExtendedModelResponse[Any],
        counter: str,
    ) -> ExtendedModelResponse[Any]:
        delta: ResearchLimitCounts = {counter: 1}  # type: ignore[typeddict-unknown-key]
        if isinstance(result, ExtendedModelResponse):
            command = result.command
            update = command.update if command is not None and command.update is not None else {}
            existing = update.get("research_limit_counts") or {}
            return ExtendedModelResponse(
                model_response=result.model_response,
                command=Command(
                    graph=command.graph if command is not None else None,
                    update={
                        **update,
                        "research_limit_counts": _counter_reducer(existing, delta),
                    },
                    resume=command.resume if command is not None else None,
                    goto=command.goto if command is not None else (),
                ),
            )

        return ExtendedModelResponse(
            model_response=result,
            command=Command(update={"research_limit_counts": delta}),
        )
