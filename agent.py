"""LangGraph entrypoint for the AI sector research agent."""

from __future__ import annotations

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from research_agent.config import Settings
from research_agent.middleware import ResearchLimitsMiddleware
from research_agent.prompts import ORCHESTRATOR_PROMPT, RESEARCHER_PROMPT
from research_agent.tools import think, web_search

load_dotenv()

settings = Settings.from_env()
model = init_chat_model(settings.model, temperature=settings.temperature)

research_subagent = {
    "name": "research-agent",
    "description": (
        "Research one focused AI sector topic and return concise findings with source URLs."
    ),
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [web_search, think],
}

agent = create_deep_agent(
    model=model,
    tools=[web_search, think],
    system_prompt=ORCHESTRATOR_PROMPT,
    subagents=[research_subagent],
    middleware=[
        ResearchLimitsMiddleware(
            max_search_calls=settings.max_search_calls,
            max_task_calls=settings.max_task_calls,
        )
    ],
)
