"""Tests for environment configuration."""

import pytest

from research_agent.config import DEFAULT_MODEL, Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_AGENT_MODEL", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_TEMPERATURE", raising=False)
    monkeypatch.delenv("MAX_SEARCH_CALLS", raising=False)
    monkeypatch.delenv("MAX_TASK_CALLS", raising=False)
    monkeypatch.delenv("MAX_ORCHESTRATOR_MODEL_CALLS", raising=False)
    monkeypatch.delenv("MAX_RESEARCHER_MODEL_CALLS", raising=False)

    settings = Settings.from_env()

    assert settings.model == DEFAULT_MODEL
    assert settings.temperature == 0.0
    assert settings.max_search_calls == 8
    assert settings.max_task_calls == 10
    assert settings.max_orchestrator_model_calls == 30
    assert settings.max_researcher_model_calls == 30


def test_settings_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_AGENT_MODEL", "openai:gpt-5")
    monkeypatch.setenv("RESEARCH_AGENT_TEMPERATURE", "0.2")
    monkeypatch.setenv("MAX_SEARCH_CALLS", "2")
    monkeypatch.setenv("MAX_TASK_CALLS", "1")
    monkeypatch.setenv("MAX_ORCHESTRATOR_MODEL_CALLS", "4")
    monkeypatch.setenv("MAX_RESEARCHER_MODEL_CALLS", "5")

    settings = Settings.from_env()

    assert settings.model == "openai:gpt-5"
    assert settings.temperature == 0.2
    assert settings.max_search_calls == 2
    assert settings.max_task_calls == 1
    assert settings.max_orchestrator_model_calls == 4
    assert settings.max_researcher_model_calls == 5


def test_settings_rejects_bad_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_SEARCH_CALLS", "many")

    with pytest.raises(ValueError, match="MAX_SEARCH_CALLS must be an integer"):
        Settings.from_env()
