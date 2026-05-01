"""LangGraph entrypoint for the AI sector research agent."""

from __future__ import annotations

from datetime import datetime

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from research_agent.config import Settings
from research_agent.middleware import ResearchLimitsMiddleware
from research_agent.prompts import (
    ORCHESTRATOR_PROMPT,
    RESEARCHER_PROMPT,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import web_search

load_dotenv()

settings = Settings.from_env()
orchestrator_model = init_chat_model(
    settings.orchestrator_model,
    temperature=settings.temperature,
)
researcher_model = init_chat_model(
    settings.researcher_model,
    temperature=settings.temperature,
)

current_date = datetime.now().strftime("%Y-%m-%d")

RESEARCH_INSTRUCTIONS = (
    RESEARCHER_PROMPT.format(date=current_date, max_search_calls=settings.max_search_calls)
)

research_subagent = {
    "name": "research-agent",
    "description": (
        "Research one focused AI sector topic and return concise findings with source URLs."
    ),
    "model": researcher_model,
    "system_prompt": RESEARCH_INSTRUCTIONS,
    "tools": [web_search],
    "middleware": [
        ResearchLimitsMiddleware(
            max_search_calls=settings.max_search_calls,
            max_task_calls=0,
            max_model_calls=settings.max_researcher_model_calls,
            model_call_counter="researcher_model_calls",
        )
    ],
}

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
INSTRUCTIONS = (
    ORCHESTRATOR_PROMPT.format(date=current_date)
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_task_calls=settings.max_task_calls,
    )
)

agent = create_deep_agent(
    model=orchestrator_model,
    tools=[],
    system_prompt=INSTRUCTIONS,
    subagents=[research_subagent],
    middleware=[
        ResearchLimitsMiddleware(
            max_search_calls=0,
            max_task_calls=settings.max_task_calls,
            max_model_calls=settings.max_orchestrator_model_calls,
            model_call_counter="orchestrator_model_calls",
        )
    ],
)
