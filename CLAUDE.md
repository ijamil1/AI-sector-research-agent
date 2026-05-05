# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                        # Install all dependencies
uv run --extra dev pytest      # Run all tests
uv run --extra dev pytest tests/test_middleware.py  # Run a single test file
uv run langgraph dev           # Start dev server on port 2024
ruff check .                   # Lint
ruff format .                  # Format
```

## Architecture

This is a **two-tier research agent** built on [Deep Agents](https://github.com/langchain-ai/deepagents) (a LangChain wrapper around LangGraph). The main graph is exported as `agent` from [agent.py](agent.py) and referenced in [langgraph.json](langgraph.json) as the `research` graph.

### Agent tiers

**Orchestrator** (`agent` in [agent.py](agent.py))
- No direct tool access — only delegates via `task` calls to the research subagent
- Responsibilities: plan with TODOs, delegate focused research tasks, synthesize, write final report
- Budget: `MAX_TASK_CALLS` delegations, `MAX_ORCHESTRATOR_MODEL_CALLS` model calls

**Research Subagent** (`research-agent` in [agent.py](agent.py))
- Has `web_search` (Tavily-backed) as its only tool
- Receives a focused topic from the orchestrator, searches, and returns structured findings
- Budget: `MAX_SEARCH_CALLS` searches, `MAX_RESEARCHER_MODEL_CALLS` model calls
- Cannot delegate further (its `max_task_calls=0`)

### Budget enforcement

[research_agent/middleware.py](research_agent/middleware.py) defines `ResearchLimitsMiddleware`, an `AgentMiddleware` subclass applied **separately** to both agent tiers. It:
- Appends remaining budget counts to the system prompt before each model call
- Blocks tool/model calls when the budget is exhausted (returns error strings instead)
- Merges counter updates from subagent `Command` responses into `ResearchLimitsState`

The orchestrator's middleware instance tracks `web_search` calls at 0 (it can't call search directly) and `research_agent_tasks`. The subagent's instance tracks `web_search` and `researcher_model_calls`.

### Key files

| File | Purpose |
|------|---------|
| [agent.py](agent.py) | Wires models, subagent, and middleware; exports the LangGraph `agent` |
| [research_agent/config.py](research_agent/config.py) | `Settings` dataclass — parsed from `.env` via `Settings.from_env()` |
| [research_agent/prompts.py](research_agent/prompts.py) | `ORCHESTRATOR_PROMPT`, `RESEARCHER_PROMPT`, `SUBAGENT_DELEGATION_INSTRUCTIONS` |
| [research_agent/tools.py](research_agent/tools.py) | `web_search` LangChain tool — HTTP POST to Tavily, formats results |
| [research_agent/middleware.py](research_agent/middleware.py) | `ResearchLimitsMiddleware` + `ResearchLimitsState` |

### Model selection

Models are initialized with `langchain.chat_models.init_chat_model()` using the `ORCHESTRATOR_MODEL` / `RESEARCHER_MODEL` env vars. The format `provider:model-id` is passed directly (e.g., `deepseek:deepseek-v4-pro`, `anthropic:claude-sonnet-4-6`). This means switching LLM providers requires only an env var change and the corresponding API key.

## Environment variables

Copy `.env.example` to `.env`. Required:

| Variable | Default | Notes |
|----------|---------|-------|
| `TAVILY_API_KEY` | — | Required — no fallback |
| `DEEPSEEK_API_KEY` | — | Required if using DeepSeek models (default) |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic models |
| `ORCHESTRATOR_MODEL` | `deepseek:deepseek-v4-pro` | LangChain model string |
| `RESEARCHER_MODEL` | `deepseek:deepseek-v4-flash` | LangChain model string |
| `MAX_SEARCH_CALLS` | `8` | Web searches per research subagent run |
| `MAX_TASK_CALLS` | `10` | Orchestrator delegations to research-agent |
| `MAX_ORCHESTRATOR_MODEL_CALLS` | `30` | Orchestrator LLM calls |
| `MAX_RESEARCHER_MODEL_CALLS` | `30` | Researcher LLM calls |

## Deep Agents behavior notes

- `create_deep_agent()` automatically adds built-in middleware (todo list, filesystem, subagent support, summarization) before user middleware. `ResearchLimitsMiddleware` is user middleware, so it runs after built-ins and sees all tool calls.
- Deep Agents adds a `general-purpose` subagent unless you supply your own list. This project supplies `[research_agent_subagent]` explicitly to prevent the orchestrator from using the general-purpose agent.
- The `StateBackend` manages long-horizon state across many model calls; the middleware's `ResearchLimitsState` extends `AgentState` to persist counters across the full run.

See [AGENTS.md](AGENTS.md) for deeper architecture notes on the Deep Agents library internals.
