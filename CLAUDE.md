# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test & Run

```bash
uv pip install .                    # install the package in editable mode
pytest                           # run all tests
pytest -m unit                   # unit tests only (no external services needed)
pytest tests/test_signal_processing.py -k test_rendered_pm_markdown_shape  # single test
tradingagents                    # launch interactive CLI
uv run main.py                   # run a single ticker directly (edit ticker/date in main.py)
docker compose run --rm tradingagents  # run via Docker
```

Tests use `pytest` with markers `unit`, `integration`, `smoke`. The `conftest.py` auto-patches all API key env vars with `"placeholder"` so tests never block waiting for credentials.

## Architecture

TradingAgents is a multi-agent financial trading framework built on **LangGraph**. The graph orchestrates a pipeline of LLM-powered agents that collaborate to produce a trading decision (Buy/Overweight/Hold/Underweight/Sell).

### Graph pipeline (defined in `tradingagents/graph/setup.py`)

```
START → Analysts (Market, Social, News, Fundamentals) [sequential]
      → Bull Researcher ↔ Bear Researcher ← debates up to max_debate_rounds
      → Research Manager → Trader
      → Aggressive ↔ Conservative ↔ Neutral ← risk debate up to max_risk_discuss_rounds
      → Portfolio Manager → END
```

- **`tradingagents/graph/trading_graph.py`** — `TradingAgentsGraph` is the main orchestrator. Initializes LLM clients, sets up the graph, manages checkpoint resume, resolves pending memory log entries, and invokes the LangGraph workflow.
- **`tradingagents/graph/setup.py`** — `GraphSetup` builds the `StateGraph` with conditional edges for tool-loop and debate routing.
- **`tradingagents/graph/propagation.py`** — `Propagator` creates the initial `AgentState` and provides `stream`/`invoke` args.
- **`tradingagents/graph/conditional_logic.py`** — `ConditionalLogic` controls when the analyst tool loop ends and when debates advance (count-based limits on back-and-forth rounds).
- **`tradingagents/graph/checkpointer.py`** — Per-ticker SQLite checkpoints via LangGraph's `SqliteSaver`. Lets a crashed run resume from the last successful node. Clears on completion.
- **`tradingagents/graph/reflection.py`** — `Reflector` calls an LLM post-trade to produce a one-paragraph reflection on what worked/didn't.
- **`tradingagents/graph/signal_processing.py`** — `SignalProcessor` extracts the 5-tier rating from the PM's structured output markdown using a deterministic regex heuristic (no LLM call).

### Agents (`tradingagents/agents/`)

Agent functions are factory functions (e.g., `create_market_analyst(llm)`) that return LangGraph nodes. Each is a function wrapping an LLM call with a system prompt.

| Module | Role |
|---|---|
| `analysts/market_analyst.py` | Technical analysis via stock data + indicators |
| `analysts/social_media_analyst.py` | Social media sentiment analysis |
| `analysts/news_analyst.py` | News + insider-transaction analysis |
| `analysts/fundamentals_analyst.py` | Financial statements analysis |
| `researchers/bull_researcher.py` | Argues the bullish case |
| `researchers/bear_researcher.py` | Argues the bearish case |
| `managers/research_manager.py` | Judges the bull/bear debate, produces `ResearchPlan` |
| `trader/trader.py` | Translates plan into `TraderProposal` |
| `risk_mgmt/*.py` | Three risk analysts (aggressive/conservative/neutral) debate the proposal |
| `managers/portfolio_manager.py` | Final decision — produces `PortfolioDecision` |

Three agents use **structured output** via Pydantic schemas (`agents/schemas.py`): Research Manager (`ResearchPlan`), Trader (`TraderProposal`), Portfolio Manager (`PortfolioDecision`). The structured-output pattern is centralized in `agents/utils/structured.py` with graceful fallback to free-text generation.

`AgentState` (in `agents/utils/agent_states.py`) carries all graph state including analyst reports, debate histories, and the final decision. It extends LangGraph's `MessagesState`.

### LLM Clients (`tradingagents/llm_clients/`)

Provider-agnostic LLM instantiation via the factory pattern:

- **`factory.py`** — `create_llm_client(provider, model, base_url, **kwargs)` lazily imports the right client.
- **`base_client.py`** — `BaseLLMClient` abstract class with `normalize_content()` helper for providers that return list-structured content.
- **`openai_client.py`** — Handles OpenAI + the six OpenAI-compatible providers (xAI, DeepSeek, Qwen, GLM, OpenRouter, Ollama). Each has its own default base URL and API key env var. DeepSeek gets a dedicated `DeepSeekChatOpenAI` subclass for thinking-mode round-trip (echoing `reasoning_content` back to the API).
- **`anthropic_client.py`**, **`google_client.py`**, **`azure_client.py`** — Provider-specific clients.
- **`model_catalog.py`** — Known model list per provider, used for validation and CLI model selection.

### Dataflows (`tradingagents/dataflows/`)

Vendor-agnostic data layer that routes tool calls to either **yfinance** or **Alpha Vantage** based on `config["data_vendors"]` and `config["tool_vendors"]`. Supports fallback chains (e.g., try Alpha Vantage first, fall back to yfinance on rate-limit). The routing logic lives in `interface.py` with `route_to_vendor()`.

### Memory / Decision Log (`tradingagents/agents/utils/memory.py`)

`TradingMemoryLog` is an append-only markdown file at `~/.tradingagents/memory/trading_memory.md`. Each run appends a pending entry with the decision; on the next same-ticker run, the system fetches realized returns, generates a reflection via LLM, and resolves pending entries. Past context (same-ticker + cross-ticker lessons) is injected into the Portfolio Manager prompt.

## Configuration (`tradingagents/default_config.py`)

`DEFAULT_CONFIG` holds all tunable settings: LLM provider/model, debate rounds, data vendors, checkpoint toggle, output language. The CLI builds atop this by prompting the user and overriding relevant keys.

## CLI (`cli/`)

Typer-based app with Rich UI (`cli/main.py`). Launches an interactive panel that lets users select tickers, date, LLM provider, analysts, etc. The `tradingagents` console script entry point is defined in `pyproject.toml` and maps to `cli.main:app`.

## Key env vars

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `ZHIPU_API_KEY`, `OPENROUTER_API_KEY` | LLM provider auth |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage data vendor |
| `TRADINGAGENTS_CACHE_DIR` | Override cache/checkpoint directory (default `~/.tradingagents/cache`) |
| `TRADINGAGENTS_RESULTS_DIR` | Override results/log directory (default `~/.tradingagents/logs`) |
| `TRADINGAGENTS_MEMORY_LOG_PATH` | Override decision log path |

## Security note

`tradingagents/dataflows/utils.py:safe_ticker_component()` validates ticker symbols before they're interpolated into filesystem paths. Tickers come from user input and from LLM tool calls (prompt injection risk in fetched content). Any code that writes files keyed by ticker must route through this function.
