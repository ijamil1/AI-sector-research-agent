# AI Sector Research Agent

This repo contains a small LangGraph-compatible Deep Agents research agent for source-backed analysis of the GenAI supply chain and vertical stack.

The intended use case is researching the latest news, trends, developments, and breakthroughs across the GenAI ecosystem, including:

- frontier model providers such as OpenAI, Anthropic, and Google
- hyperscalers and AI cloud infrastructure such as Amazon, Microsoft, and CoreWeave
- chip designers such as NVIDIA, Amazon, AMD, and Intel
- chip manufacturers and foundries such as TSMC and Intel
- energy providers, power sources, grid constraints, and data-center power demand
- raw materials and upstream dependencies such as silicon, rare earth materials, and related supply-chain inputs

## Project Shape

- `agent.py` is the LangGraph entrypoint. It exports `agent`, which is exposed as graph `research` in `langgraph.json`.
- `research_agent/prompts.py` contains the orchestrator and researcher prompts.
- `research_agent/tools.py` contains the Tavily-backed `web_search` tool.
- `research_agent/middleware.py` contains `ResearchLimitsMiddleware`, which enforces runtime budgets for web search and research-subagent delegation.
- `research_agent/config.py` loads runtime settings from environment variables.
- `tests/` contains focused unit tests for config, tools, and middleware.

## Deep Agents Notes

The project uses `deepagents.create_deep_agent`.

Important library behavior:

- `create_deep_agent` adds a built-in middleware stack before user middleware: todo list, filesystem tools, subagent/task support, summarization, and dangling tool-call patching.
- User middleware is inserted after the built-in stack. This means `ResearchLimitsMiddleware` sees tool calls from both user tools and Deep Agents middleware-injected tools such as `task`.
- If no subagent named `general-purpose` is supplied, Deep Agents auto-adds one. This project also supplies a custom `research-agent` subagent.
- Declarative subagents are invoked through the `task` tool with `subagent_type`.
- Subagent calls return only the subagent's final message to the parent agent.
- The default backend is `StateBackend`, so built-in filesystem state is ephemeral per graph/thread. The `execute` tool is filtered out because `StateBackend` does not support command execution.
- Built-in summarization can compact long conversations and offload history into the Deep Agents virtual filesystem state.

## Current Agent Wiring

`agent.py` creates:

- the main orchestrator agent with no custom tools
- a focused `research-agent` subagent with tools `[web_search]`
- `ResearchLimitsMiddleware` configured from environment settings

Default budgets from `Settings` are:

- `MAX_SEARCH_CALLS=8`
- `MAX_TASK_CALLS=10`
- `MAX_ORCHESTRATOR_MODEL_CALLS=30`
- `MAX_RESEARCHER_MODEL_CALLS=30`

The middleware counts:

- `web_search` tool calls as `web_search`
- `task` calls where `subagent_type == "research-agent"` as `research_agent_tasks`
- outer-orchestrator model calls as `orchestrator_model_calls`
- research-subagent model calls as `researcher_model_calls`

The outer orchestrator does not expose `web_search` to its model and sets its
search limit to `0`. The `research-agent` subagent exposes `web_search` and has
its own `ResearchLimitsMiddleware` with task delegation disabled via
`max_task_calls=0`. Both the orchestrator and research subagent middleware append
current budget notices to the system prompt. The orchestrator also blocks model
calls once its configured model-call budget is exhausted. The research subagent
also has a model-call budget, and its research activity is bounded through the
`web_search` tool budget.

The middleware does not count filesystem tools. It blocks `task` calls to any
subagent type other than `research-agent`, including the auto-added
`general-purpose` subagent.

## Commands

Use `uv`.

```bash
uv sync
uv run --extra dev pytest
uv run langgraph dev
```

The basic test baseline is currently:

```bash
uv run --extra dev pytest
```

## Environment

Copy `.env.example` to `.env` and configure:

- `ORCHESTRATOR_MODEL`
- `RESEARCHER_MODEL`
- provider API key, such as `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- optional `MAX_SEARCH_CALLS`
- optional `MAX_TASK_CALLS`

Do not commit `.env` or secrets.
