# TradingAgents

Multi-agent LLM framework for financial trading analysis and decision-making using LangGraph.

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | ≥ 3.10 |
| Agent orchestration | LangGraph | ≥ 0.4.8 |
| LLM integrations | LangChain (OpenAI, Anthropic, Google GenAI) | ≥ 0.3.x |
| CLI | Typer + Questionary + Rich | ≥ 0.21.0 |
| Market data | yFinance / Alpha Vantage | ≥ 0.2.63 |
| Technical indicators | stockstats | ≥ 0.6.5 |
| State persistence | langgraph-checkpoint-sqlite | ≥ 2.0.0 |
| Caching (optional) | Redis | ≥ 6.2.0 |
| Backtesting | backtrader | ≥ 1.9.78.123 |
| Packaging | setuptools / uv | — |

## Commands

```bash
# Interactive CLI (after install)
tradingagents

# Install in editable mode
pip install -e .

# Run a single analysis programmatically
python main.py

# Run all tests
pytest tests/ -ra

# Run only unit tests
pytest tests/ -m unit -ra

# Run smoke tests
pytest tests/ -m smoke -ra

# Docker — default profile
docker-compose up tradingagents

# Docker — with local Ollama
docker-compose --profile ollama up
```

## Project Structure

```
tradingagents/              # Core library package
  agents/
    analysts/               # Market, news, fundamentals, sentiment, social analysts
    managers/               # Portfolio Manager, Research Manager
    researchers/            # Bull Researcher, Bear Researcher
    risk_mgmt/              # Aggressive / Neutral / Conservative debators
    trader/                 # Trader agent
    utils/                  # Shared agent state, memory log, LangChain tools
  dataflows/                # Data ingestion layer
    y_finance.py            # yFinance market data + news
    alpha_vantage*.py       # Alpha Vantage stock, indicators, fundamentals, news
    reddit.py / stocktwits.py  # Social sentiment sources
    interface.py            # Vendor-routing facade (respects config data_vendors)
  graph/                    # LangGraph orchestration
    trading_graph.py        # TradingAgentsGraph — main entry point
    setup.py                # Graph node/edge wiring
    propagation.py          # Forward propagation (analysis → decision)
    conditional_logic.py    # Routing conditions between nodes
    signal_processing.py    # Signal extraction from final state
    reflection.py           # Post-trade reflection and memory update
    analyst_execution.py    # Analyst timing and concurrency planning
    checkpointer.py         # SQLite checkpoint helpers
  llm_clients/              # Multi-provider LLM client factory
    factory.py              # create_llm_client() dispatcher
    model_catalog.py        # Model name / capability registry
    *_client.py             # Provider-specific clients (OpenAI, Anthropic, Google, …)
  default_config.py         # Central config + TRADINGAGENTS_* env-var override system
cli/                        # Typer-based interactive CLI
  main.py                   # app entrypoint, streaming output, TUI
  models.py                 # Pydantic / enum models (AnalystType, …)
  config.py                 # CLI-side provider/model selection helpers
  stats_handler.py          # LangChain callback for token/latency stats
tests/                      # pytest suite (unit, integration, smoke markers)
scripts/                    # Ad-hoc utility scripts
docs/                       # Documentation
main.py                     # Minimal programmatic example
```

## External Dependencies

| Service | Purpose | Required? |
|---|---|---|
| OpenAI API | LLM inference (default provider) | One provider required |
| Anthropic API | Claude models | Optional |
| Google Generative AI | Gemini models | Optional |
| xAI (Grok), DeepSeek, DashScope, Zhipu, MiniMax, OpenRouter | Alternative LLM providers | Optional |
| Ollama | Local LLM inference | Optional |
| Alpha Vantage API | Stock data, indicators, fundamentals, news | Optional (yFinance is default) |
| Reddit API | Social sentiment data | Optional |
| StockTwits | Social sentiment data | Optional |
| Redis | Response caching | Optional |

Configuration is driven by `.env` (see `.env.example`). Any `DEFAULT_CONFIG` key can be overridden at runtime via a matching `TRADINGAGENTS_*` environment variable without code changes.
