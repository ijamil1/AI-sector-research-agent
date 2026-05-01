"""Runtime configuration for the research agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_ORCHESTRATOR_MODEL = "deepseek:deepseek-v4-pro"
DEFAULT_RESEARCHER_MODEL = "deepseek:deepseek-v4-flash"


@dataclass(frozen=True)
class Settings:
    """Settings loaded from environment variables."""

    orchestrator_model: str = DEFAULT_ORCHESTRATOR_MODEL
    researcher_model: str = DEFAULT_RESEARCHER_MODEL
    temperature: float = 0.0
    max_search_calls: int = 8
    max_task_calls: int = 10
    max_orchestrator_model_calls: int = 30
    max_researcher_model_calls: int = 30

    @classmethod
    def from_env(cls) -> Settings:
        """Create settings from process environment variables.

        Returns:
            Parsed settings with defaults for omitted values.

        Raises:
            ValueError: If numeric environment variables are malformed.
        """
        return cls(
            orchestrator_model=os.getenv(
                "ORCHESTRATOR_MODEL",
                DEFAULT_ORCHESTRATOR_MODEL,
            ),
            researcher_model=os.getenv(
                "RESEARCHER_MODEL",
                DEFAULT_RESEARCHER_MODEL,
            ),
            temperature=_parse_float("RESEARCH_AGENT_TEMPERATURE", 0.0),
            max_search_calls=_parse_int("MAX_SEARCH_CALLS", 8),
            max_task_calls=_parse_int("MAX_TASK_CALLS", 10),
            max_orchestrator_model_calls=_parse_int("MAX_ORCHESTRATOR_MODEL_CALLS", 30),
            max_researcher_model_calls=_parse_int("MAX_RESEARCHER_MODEL_CALLS", 30),
        )


def _parse_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg) from exc
    if parsed < 0:
        msg = f"{name} must be non-negative, got {parsed}"
        raise ValueError(msg)
    return parsed


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        msg = f"{name} must be a float, got {value!r}"
        raise ValueError(msg) from exc
