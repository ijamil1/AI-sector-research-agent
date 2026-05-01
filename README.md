# AI Sector Research Agent

Standalone Deep Agents research agent for source-backed analysis of the AI sector.

The project is intentionally small: `agent.py` exports a LangGraph-compatible
graph named `research`, while `research_agent/` contains the prompts, tools,
configuration, and custom middleware.

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` with a model provider key and a Tavily key:

```bash
ORCHESTRATOR_MODEL=deepseek:deepseek-v4-pro
RESEARCHER_MODEL=deepseek:deepseek-v4-flash
DEEPSEEK_API_KEY=...
TAVILY_API_KEY=...
```

The default model IDs use DeepSeek's V4 models: Pro for orchestration and
synthesis, Flash for focused research subagents. You can switch providers by
changing `ORCHESTRATOR_MODEL` and `RESEARCHER_MODEL` to other LangChain model
strings and setting the matching provider key.

## Run Locally

```bash
uv run langgraph dev
```

The graph is exposed as `research` through `langgraph.json`.

## Invoke With The LangGraph SDK

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")
thread = client.threads.create()

result = client.runs.wait(
    thread["thread_id"],
    "research",
    input={
        "messages": [
            {
                "role": "user",
                "content": "Research the AI infrastructure market and identify key themes.",
            }
        ]
    },
)

print(result["messages"][-1]["content"])
```

## Research Budgets

`ResearchLimitsMiddleware` enforces real runtime budgets for:

- `web_search`
- `task` calls to the `research-agent` subagent
- model calls made by the orchestrator
- model calls made by the researcher subagent

Configure them in `.env`:

```bash
MAX_SEARCH_CALLS=8
MAX_TASK_CALLS=10
MAX_ORCHESTRATOR_MODEL_CALLS=30
MAX_RESEARCHER_MODEL_CALLS=30
```

These limits are enforced by middleware, not only by prompt instructions.

## Git

This repository is intended to use:

```bash
git remote add origin https://github.com/ijamil1/AI-sector-research-agent.git
git branch -M main
```
