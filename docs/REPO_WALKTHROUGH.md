# TradingAgents: A Linear Walkthrough

*2026-05-21T15:50:56Z by Showboat dev*
<!-- showboat-id: ffa88f8a-e8a3-4a90-bd46-42b990920c07 -->

## What this document is

This is a top-to-bottom tour of the TradingAgents codebase — the multi-agent LLM trading framework whose public-facing shape you can see in `README.md`. We start at the entry points, descend through configuration, the LLM client layer, and the state graph, then walk every agent in the order they actually run inside a live propagation. Along the way we look at the supporting machinery: data-vendor routing, structured-output schemas, memory and reflection, checkpointing, and the Rich-based CLI that wraps it all.

The goal is to give you a single linear narrative that explains *how the system works*, with every piece of commentary backed by a snippet pulled from the source. By the end you should be able to point at any node in the graph and know which file owns it, which LLM call it makes, which state it mutates, and which tools it can reach for.

## 1. The thirty-thousand-foot view

TradingAgents is a LangGraph-orchestrated multi-agent system that simulates a trading desk. A run takes `(ticker, trade_date)` and produces a five-tier trading rating (**Buy / Overweight / Hold / Underweight / Sell**), backed by a stack of analyst reports, a structured debate, and a final portfolio-manager decision.

The repository is small — under ten thousand lines of Python — and breaks cleanly into four layers:

```bash
find /home/evan/opensource/TradingAgents -maxdepth 2 -type d -not -path '*/.*' -not -path '*egg-info*' -not -path '*/.venv*' | sort
```

```output
/home/evan/opensource/TradingAgents
/home/evan/opensource/TradingAgents/assets
/home/evan/opensource/TradingAgents/assets/cli
/home/evan/opensource/TradingAgents/cli
/home/evan/opensource/TradingAgents/cli/static
/home/evan/opensource/TradingAgents/docs
/home/evan/opensource/TradingAgents/scripts
/home/evan/opensource/TradingAgents/tests
/home/evan/opensource/TradingAgents/tradingagents
/home/evan/opensource/TradingAgents/tradingagents/agents
/home/evan/opensource/TradingAgents/tradingagents/dataflows
/home/evan/opensource/TradingAgents/tradingagents/graph
/home/evan/opensource/TradingAgents/tradingagents/llm_clients
```

- **`cli/`** — interactive Typer + Rich front-end (`tradingagents` console script).
- **`tradingagents/graph/`** — the LangGraph wiring: state graph, conditional logic, propagation, reflection, signal processing, checkpointing.
- **`tradingagents/agents/`** — the LangGraph nodes themselves: four analysts, two researchers (bull/bear), one trader, three risk debators, two managers.
- **`tradingagents/dataflows/`** — data-vendor implementations (yfinance, Alpha Vantage, StockTwits, Reddit) and the router that dispatches tool calls to the configured vendor.
- **`tradingagents/llm_clients/`** — a thin abstraction over LangChain provider classes (OpenAI, Anthropic, Google, Azure plus seven OpenAI-compatible providers) with model validation and per-model capability flags.
- **`tests/`** — pytest suite covering memory log, structured outputs, capability dispatch, ticker hardening, checkpoint resume, etc.

A reading order that makes sense: `README.md` → `main.py` (Python API) → `tradingagents/__init__.py` (bootstrap) → `tradingagents/default_config.py` → `tradingagents/graph/trading_graph.py` → `tradingagents/graph/setup.py` → each agent file → `tradingagents/dataflows/interface.py` → `cli/main.py` (UI). This document follows that order.

## 2. The two entry points

There are exactly two ways to start a run, and `pyproject.toml` is where you see them registered:

```bash
sed -n '1,40p' /home/evan/opensource/TradingAgents/pyproject.toml
```

```output
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "tradingagents"
version = "0.2.5"
description = "TradingAgents: Multi-Agents LLM Financial Trading Framework"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "langchain-core>=0.3.81",
    "backtrader>=1.9.78.123",
    "langchain-anthropic>=0.3.15",
    "langchain-experimental>=0.3.4",
    "langchain-google-genai>=4.0.0",
    "langchain-openai>=0.3.23",
    "langgraph>=0.4.8",
    "langgraph-checkpoint-sqlite>=2.0.0",
    "pandas>=2.3.0",
    "parsel>=1.10.0",
    "pytz>=2025.2",
    "questionary>=2.1.0",
    "redis>=6.2.0",
    "requests>=2.32.4",
    "rich>=14.0.0",
    "typer>=0.21.0",
    "setuptools>=80.9.0",
    "stockstats>=0.6.5",
    "tqdm>=4.67.1",
    "typing-extensions>=4.14.0",
    "yfinance>=0.2.63",
]

[project.scripts]
tradingagents = "cli.main:app"

[tool.setuptools.packages.find]
include = ["tradingagents*", "cli*"]

```

The `tradingagents` console script points at `cli.main:app` — a Typer application. The Python API entry sits at the repository root in `main.py` and is a literal copy-paste example for embedding the framework in user code:

```bash
cat /home/evan/opensource/TradingAgents/main.py
```

```output
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
```

Both paths funnel into the same class — `TradingAgentsGraph` — whose `.propagate(ticker, date)` is the one public API of the whole framework. That's the only call you need to remember. Everything else in this walkthrough explains what happens inside it.

## 3. Bootstrapping: `tradingagents/__init__.py`

When the package is first imported, three things happen before any of your code runs:

```bash
sed -n '1,40p' /home/evan/opensource/TradingAgents/tradingagents/__init__.py
```

```output
import warnings

# Load .env files at package import so DEFAULT_CONFIG's env-var overlay
# (and every llm_clients consumer) sees the user's keys regardless of
# which entry point started the process. find_dotenv(usecwd=True) walks
# from the CWD, so the installed `tradingagents` console script picks up
# the project's .env instead of stepping up from site-packages.
# load_dotenv defaults to override=False, so it never clobbers values
# the caller has already exported.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(find_dotenv(".env.enterprise", usecwd=True), override=False)
except ImportError:
    pass

# langchain-core 1.3.3 calls surface_langchain_deprecation_warnings() in
# its own __init__, which prepends default-action filters for its
# subclassed warning categories. To suppress a specific warning we must
# install our filter AFTER langchain-core has installed its own, so import
# it first. The package is a guaranteed transitive dep via langgraph.
try:
    import langchain_core  # noqa: F401
except ImportError:
    pass

# langgraph-checkpoint 4.0.3 calls Reviver() at module load without an
# explicit allowed_objects, which triggers a noisy pending-deprecation
# warning from langchain-core 1.3.3 on every interpreter start. The fix
# is already merged upstream (langchain-ai/langgraph#7743, 2026-05-08)
# and will arrive in the next langgraph-checkpoint release. Remove this
# block (and the langchain_core preload above) when we bump past it.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects`.*",
    category=PendingDeprecationWarning,
)
```

1. **`.env` is loaded from CWD** — both `.env` and `.env.enterprise` are walked from the working directory, not from where the package is installed. That's why running the installed `tradingagents` script in your project folder picks up your local API keys.
2. **`langchain_core` is preloaded** — a workaround for a noisy `langgraph-checkpoint 4.0.3` deprecation warning that fires on every interpreter start. The comment in the source documents the upstream fix and tells you when to delete this block.
3. **`override=False`** — env vars exported by the parent shell always win over what's in the `.env` file. This matters when running inside Docker or CI.

## 4. Configuration as data: `default_config.py`

This is the single source of truth for runtime configuration. There is no settings class, no Pydantic model: it's a dict, and every consumer reads it the same way. The interesting parts are at the top:

```bash
sed -n '46,84p' /home/evan/opensource/TradingAgents/tradingagents/default_config.py
```

```output
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
```

And the env-var overlay table that lets users override any of these from `.env` without touching code:

```bash
sed -n '6,32p' /home/evan/opensource/TradingAgents/tradingagents/default_config.py
```

```output
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value

```

The pattern is worth pausing on: `_coerce` reads the type of the *current default value*, so a user writing `TRADINGAGENTS_CHECKPOINT_ENABLED=true` in their `.env` gets a real Python `bool`, not the string `"true"`. Adding a new overridable key requires only one row in the table.

A second internal detail: `backend_url` defaults to `None`, not to a provider-specific URL. The comment explains why — previously OpenAI's `/v1` was leaking into Gemini calls and producing malformed URLs.

The runtime layer in `tradingagents/dataflows/config.py` exposes the config to the rest of the codebase, with a small but careful detail: dict-valued keys (like `data_vendors`) are merged one level deep, scalars are replaced:

```bash
sed -n '17,32p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/config.py
```

```output
def set_config(config: Dict):
    """Update the configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    global _config
    initialize_config()
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(_config.get(key), dict):
            _config[key].update(value)
        else:
            _config[key] = value

```

## 5. LLM provider abstraction

`tradingagents/llm_clients/` is a tiny, opinionated layer over LangChain. It exists because the framework needs to support a long tail of providers (OpenAI, Anthropic, Google, Azure, plus seven OpenAI-compatible vendors: xAI, DeepSeek, Qwen × 2 regions, GLM × 2 regions, MiniMax × 2 regions, OpenRouter, Ollama) and each has its own quirks — base URL, API-key env-var name, structured-output capability, reasoning-content roundtrip requirements.

The dispatch happens in `factory.py`:

```bash
sed -n '1,57p' /home/evan/opensource/TradingAgents/tradingagents/llm_clients/factory.py
```

```output
from typing import Optional

from .base_client import BaseLLMClient

# Providers that use the OpenAI-compatible chat completions API
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek",
    "qwen", "qwen-cn",
    "glm", "glm-cn",
    "minimax", "minimax-cn",
    "ollama", "openrouter",
)


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider modules are imported lazily so that simply importing this
    factory (e.g. during test collection) does not pull in heavy LLM SDKs
    or fail when their API keys are absent.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower in _OPENAI_COMPATIBLE:
        from .openai_client import OpenAIClient
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
```

Two design choices stand out. **Lazy imports**: provider modules are imported inside the branch that uses them, so simply importing the factory (e.g. during `pytest` collection) doesn't pull in `langchain-anthropic` or fail when API keys are absent. **Bucketing**: all OpenAI-compatible vendors collapse into one class — `OpenAIClient` — because the Chat Completions wire format is the same; provider-specific quirks live in subclasses or in a capability table.

### 5.1 Content normalisation

Multi-provider support has one ugly trap: some providers (OpenAI Responses API, Gemini 3, Claude with extended thinking) return `content` as a list of typed blocks, not a string. Downstream agents expect `response.content` to be a plain string. The fix sits in `base_client.py`:

```bash
sed -n '6,21p' /home/evan/opensource/TradingAgents/tradingagents/llm_clients/base_client.py
```

```output
def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
```

Each concrete client subclasses its LangChain class to override `invoke` with this normaliser. Reasoning blocks are discarded, text blocks are joined.

### 5.2 The per-model capability table

The hard problem isn't the wire format — it's that each provider has model-specific landmines. DeepSeek thinking models 400 if you don't echo `reasoning_content` back. MiniMax M2.x models 400 if you don't set `reasoning_split=True`. Both providers reject langchain's `tool_choice` dict in different ways. Instead of an `if model.startswith("deepseek-v4")` ladder, capabilities are encoded declaratively:

```bash
sed -n '23,90p' /home/evan/opensource/TradingAgents/tradingagents/llm_clients/capabilities.py
```

```output
    "function_calling",  # uses tools; respects supports_tool_choice
    "json_mode",         # uses response_format={"type":"json_object"}
    "json_schema",       # uses response_format={"type":"json_schema",...}
    "none",              # no structured output available; caller falls back to free-text
]


@dataclass(frozen=True)
class ModelCapabilities:
    """What an OpenAI-compatible model accepts at the API level."""

    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: StructuredMethod
    # DeepSeek thinking-mode models 400 if reasoning_content from prior
    # assistant turns is not echoed back on the next request.
    requires_reasoning_content_roundtrip: bool = False
    # MiniMax M2.x reasoning models need ``reasoning_split=True`` so the
    # <think> block lands in ``reasoning_details`` instead of polluting
    # ``content``. The flag is rejected by non-reasoning MiniMax models
    # (Coding Plan, MiniMax-Text-01, etc.), so we only set it where the
    # model actually consumes it. (#826)
    requires_reasoning_split: bool = False


# DeepSeek's thinking models accept the ``tools`` array but reject the
# ``tool_choice`` parameter (official Oh My Pi integration guide and the
# 400 response in issue #678). Their official tool-calling examples
# (api-docs.deepseek.com/guides/tool_calls) pass ``tools=[...]`` without
# ``tool_choice`` — we mirror that pattern by setting supports_tool_choice
# to False and letting the client suppress the kwarg.
_DEEPSEEK_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_content_roundtrip=True,
)

_DEEPSEEK_CHAT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
)

# MiniMax M2.x reasoning models accept the tools array, but their
# tool_choice parameter is restricted to the enum {"none", "auto"}
# (platform.minimax.io/docs/api-reference/text-post). Langchain's
# function_calling path sends tool_choice as a function-spec dict, which
# MiniMax 400s — same shape as the DeepSeek bug. supports_tool_choice=False
# makes the dispatch in NormalizedChatOpenAI suppress the kwarg; the schema
# still ships as a tool. json_mode response_format is only for
# MiniMax-Text-01, not M2.x.
_MINIMAX_THINKING = ModelCapabilities(
    supports_tool_choice=False,
    supports_json_mode=False,
    supports_json_schema=False,
    preferred_structured_method="function_calling",
    requires_reasoning_split=True,
)

_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
```

And the consumer side in `NormalizedChatOpenAI.with_structured_output` reads it:

```bash
sed -n '14,42p' /home/evan/opensource/TradingAgents/tradingagents/llm_clients/openai_client.py
```

```output
    """ChatOpenAI with normalized content output and capability-aware binding.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )
        method = method or caps.preferred_structured_method
```

The result of this design: adding a new model means editing one table, not the client code. DeepSeek thinking models also need a *reasoning-content roundtrip* — the previous-turn `reasoning_content` has to be echoed back as part of the assistant message or the API 400s. That lives in `DeepSeekChatOpenAI` as overrides on `_get_request_payload` and `_create_chat_result`. MiniMax M2.x gets a parallel subclass that injects `reasoning_split=True` into the payload.

### 5.3 API-key environment-variable mapping

The CLI's interactive key prompt and the runtime clients both consult one canonical mapping:

```bash
sed -n '15,38p' /home/evan/opensource/TradingAgents/tradingagents/llm_clients/api_key_env.py
```

```output


PROVIDER_API_KEY_ENV: dict[str, Optional[str]] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "azure":      "AZURE_OPENAI_API_KEY",
    "xai":        "XAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    # Dual-region providers each carry their own account; keys are not
    # interchangeable between the international and China endpoints.
    "qwen":       "DASHSCOPE_API_KEY",
    "qwen-cn":    "DASHSCOPE_CN_API_KEY",
    "glm":        "ZHIPU_API_KEY",
    "glm-cn":     "ZHIPU_CN_API_KEY",
    "minimax":    "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    # Local runtimes do not authenticate.
    "ollama":     None,
}


def get_api_key_env(provider: str) -> Optional[str]:
```

Note the dual-region split for Qwen, GLM, and MiniMax. Their China-mainland and international endpoints require separate accounts, so keys are not interchangeable. The CLI surfaces this as a follow-up question after the provider is picked.

## 6. The shared state schema

Every node in the graph reads from and writes to a single `AgentState` TypedDict. It's the closest thing this project has to a 'message bus':

```bash
sed -n '47,76p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/agent_states.py
```

```output
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    asset_type: Annotated[str, "Asset type under analysis such as stock or crypto"]
    trade_date: Annotated[str, "What date we are trading at"]

    sender: Annotated[str, "Agent that sent this message"]

    # research step
    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    news_report: Annotated[
        str, "Report from the News Researcher of current world affairs"
    ]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Researcher"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Plan generated by the Analyst"]

    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    final_trade_decision: Annotated[str, "Final decision made by the Risk Analysts"]
    past_context: Annotated[str, "Memory log context injected at run start (same-ticker decisions + cross-ticker lessons)"]
```

`AgentState` extends LangGraph's `MessagesState`, which gives it a `messages` field for tool-using analysts. Each report field is a string slot owned by exactly one analyst. The two embedded `TypedDict`s — `InvestDebateState` and `RiskDebateState` — hold full conversation histories for the two debates.

Crucially, every field is *additive*: a node returns a dict patch, LangGraph merges it into state. So when the Market Analyst finishes, it returns `{"messages": [...], "market_report": ...}` and nothing else gets touched.

## 7. The `TradingAgentsGraph` orchestrator

This is the class `main.py` instantiates. Its constructor (`tradingagents/graph/trading_graph.py`) is where the framework's lazy initialisation lives:

```bash
sed -n '56,141p' /home/evan/opensource/TradingAgents/tradingagents/graph/trading_graph.py
```

```output
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            analyst_concurrency_limit=self.config.get("analyst_concurrency_limit", 1),
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
```

A few observations:

- **Two LLMs, not one**: `deep_thinking_llm` (`deep_think_llm` config key) is reserved for the Research Manager and the Portfolio Manager — the agents that produce the final structured outputs. Every other agent uses `quick_thinking_llm`. The split lets you spend $30/$180-per-1M-tokens on the two decisions that matter and the cheap model on everything else.
- **`set_config(self.config)`** propagates the user-supplied config into `dataflows/config.py`'s module-global, so the tools layer sees the same values.
- **One `workflow`, one `graph`** — the `workflow` (StateGraph) is kept around as a separate attribute because it gets *recompiled* with a checkpointer when resume is enabled (see §14).

### 7.1 Tool nodes

Each analyst gets a `ToolNode` configured with only the tools its prompt instructs it to call:

```bash
sed -n '160,190p' /home/evan/opensource/TradingAgents/tradingagents/graph/trading_graph.py
```

```output
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
```

The keys here — `market`, `social`, `news`, `fundamentals` — match the analyst dispatch keys used everywhere else in the system. Note that `social` is the wire key for what the user-facing CLI calls 'Sentiment Analyst'; the rename happened in v0.2.5 but the wire value stayed for back-compat with saved configs.

## 8. Graph composition: `GraphSetup`

This is where the LangGraph wiring lives — analyst nodes get added in the order the user selected them, and a sequence of conditional edges threads them together. The high-level shape is:

```
START
  → first selected analyst → tools loop → clear messages
  → next selected analyst → tools loop → clear messages
  …
  → Bull Researcher ⇄ Bear Researcher (configurable rounds)
  → Research Manager
  → Trader
  → Aggressive Analyst → Conservative Analyst → Neutral Analyst (cycle, configurable rounds)
  → Portfolio Manager
  → END
```

The interesting code:

```bash
sed -n '79,124p' /home/evan/opensource/TradingAgents/tradingagents/graph/setup.py
```

```output
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
```

Each analyst contributes three nodes: an *agent node* (the LLM call), a *tool node* (the `ToolNode` from §7.1), and a *clear node* that wipes the message history before moving on. The pattern lets one analyst's tool-call chatter not contaminate the next analyst's context window.

The clear node is a tiny helper:

```bash
sed -n '46,57p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/agent_utils.py
```

```output
    )
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
```

```bash
sed -n '55,67p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/agent_utils.py
```

```output
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages
```

A `"Continue"` placeholder is left behind because Anthropic's API rejects a thread where the only message is from the assistant — there has to be a user turn after the deletes for the next agent to start cleanly.

The analyst execution plan that drives the loop is itself a dataclass:

```bash
sed -n '20,68p' /home/evan/opensource/TradingAgents/tradingagents/graph/analyst_execution.py
```

```output

ANALYST_NODE_SPECS: Dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        clear_node="Msg Clear Market",
        tool_node="tools_market",
        report_key="market_report",
    ),
    "social": AnalystNodeSpec(
        # Wire key stays "social" for saved-config back-compat; the
        # user-facing label is "Sentiment Analyst" to match the rename
        # that landed in v0.2.5 (sentiment_analyst now ingests news +
        # StockTwits + Reddit, not just social media).
        key="social",
        agent_node="Sentiment Analyst",
        clear_node="Msg Clear Sentiment",
        tool_node="tools_social",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        clear_node="Msg Clear News",
        tool_node="tools_news",
        report_key="news_report",
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        clear_node="Msg Clear Fundamentals",
        tool_node="tools_fundamentals",
        report_key="fundamentals_report",
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
    concurrency_limit: int = 1,
) -> AnalystExecutionPlan:
    if concurrency_limit < 1:
        raise ValueError("analyst concurrency limit must be >= 1")

    specs: List[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
```

Every node name (`"Market Analyst"`, `"tools_market"`, `"Msg Clear Market"`) and every report-state key (`"market_report"`) is bundled into one `AnalystNodeSpec`. The CLI and the graph wiring both consume the same plan, which is why renaming a node only requires editing this one table.

## 9. Conditional routing

The conditional logic that decides where to go next has only two interesting cases:

```bash
sed -n '13,60p' /home/evan/opensource/TradingAgents/tradingagents/graph/conditional_logic.py
```

```output

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if sentiment-analyst tool round should continue.

        Method name keeps the legacy ``social`` suffix to match the
        ``AnalystType.SOCIAL = "social"`` wire value (saved-config
        back-compat); the returned ``clear_node`` label uses the v0.2.5
        rename so it matches the node registered by the execution plan.
        """
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Sentiment"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
```

The four `should_continue_*` methods all do the same thing: if the last AI message contains `tool_calls`, route to the tools node; otherwise route to the clear node and move on. This is the classic tool-using-agent loop, just one per analyst.

The debate routers count turns:

```bash
sed -n '52,75p' /home/evan/opensource/TradingAgents/tradingagents/graph/conditional_logic.py
```

```output
    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
```

`max_debate_rounds` and `max_risk_discuss_rounds` are user-configurable (the CLI's 'Research Depth' picker sets both — Shallow=1, Medium=3, Deep=5). The bull/bear loop runs `2 × max_debate_rounds` turns total; the three-way risk loop runs `3 × max_risk_discuss_rounds` turns. The speaker order is enforced by reading `latest_speaker` / `current_response` and rotating.

## 10. Initial state: `Propagator`

When `.propagate()` is called, the propagator builds the seed state:

```bash
sed -n '16,68p' /home/evan/opensource/TradingAgents/tradingagents/graph/propagation.py
```

```output
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
```

Three things to notice:

1. The single seed message is `("human", company_name)` — just the ticker. Each analyst's first turn sees only this; it relies on its system prompt to interpret it.
2. `past_context` is a string injected only into the *Portfolio Manager*'s prompt (see §17). It carries forward the most recent same-ticker decisions and a few cross-ticker lessons from the memory log.
3. `asset_type` flows through to every agent so the prompts can adapt — `"crypto"` switches the analyst stack and skips fundamentals (no balance sheet for Bitcoin).

The runtime config also caps recursion at `max_recur_limit` (default 100), bounding the worst case if a tool-loop misbehaves.

## 11. The analysts

### 11.1 Market Analyst — technicals

This is the canonical 'analyst that uses tools' shape. The system prompt is a wall of indicator descriptions (50 SMA, MACD, RSI, etc.), and the node binds the two relevant tools to the LLM:

```bash
sed -n '10,28p' /home/evan/opensource/TradingAgents/tradingagents/agents/analysts/market_analyst.py
```

```output

def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(
            state["company_of_interest"], asset_type
        )

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
```

```bash
sed -n '60,99p' /home/evan/opensource/TradingAgents/tradingagents/agents/analysts/market_analyst.py
```

```output
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
```

Each analyst follows the same shape: a partial-filled `ChatPromptTemplate`, `llm.bind_tools(tools)`, a single invoke. If the LLM emits tool calls, the report stays empty and the graph loops back via the conditional edge (`should_continue_market`) into the tools node, then back into the analyst. If the LLM finally produces a free-text response, that response is stored as `market_report`.

`build_instrument_context` is a small but important helper that pins the ticker to the LLM:

```bash
sed -n '36,53p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/agent_utils.py
```

```output
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )

```

Without this, LLMs love to drop the exchange suffix (`0700.HK` → `0700`, `7203.T` → `7203`) and then make tool calls against the wrong instrument. The system also runs all tickers through `safe_ticker_component()` before they touch the filesystem — preventing path-traversal injection via either user input or LLM-generated tool args.

### 11.2 Sentiment Analyst — pre-fetched data, no tool calls

This one is the most interesting analyst because it inverts the pattern. Instead of giving the LLM tools and letting it explore, the node fetches all three sentiment sources up front and embeds them in the prompt. The motivation is in the file header:

```bash
sed -n '1,21p' /home/evan/opensource/TradingAgents/tradingagents/agents/analysts/sentiment_analyst.py
```

```output
"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. The LLM produces the sentiment report in a single invocation.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

```

```bash
sed -n '45,60p' /home/evan/opensource/TradingAgents/tradingagents/agents/analysts/sentiment_analyst.py
```

```output

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
        reddit_block = fetch_reddit_posts(ticker)

        system_message = _build_system_message(
            ticker=ticker,
```

Three data fetchers run synchronously up front. Each is designed to degrade gracefully — if the network is down or the symbol has zero StockTwits messages, the fetcher returns a placeholder string instead of raising, so the LLM always sees a structured block. The chain that follows is just `prompt | llm` with **no `bind_tools`** — one LLM call, one report.

### 11.3 News & Fundamentals Analysts

These follow the Market Analyst shape. The News Analyst gets `get_news`, `get_global_news`, and `get_insider_transactions`; the Fundamentals Analyst gets `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, and `get_income_statement`. The prompts steer them to call those tools and produce a multi-paragraph report ending in a markdown summary table.

The Fundamentals Analyst is excluded automatically from the analyst list when the asset type is crypto — that's enforced both at the CLI layer (`filter_analysts_for_asset_type` in `cli/utils.py`) and at the prompt level (`build_instrument_context` with `asset_type="crypto"` adds 'do not assume company fundamentals are available').

## 12. The tools layer: vendor routing

Every analyst tool — `get_stock_data`, `get_indicators`, `get_news`, etc. — is a thin `@tool` shim that delegates to one function: `route_to_vendor`. The tool definitions live in `tradingagents/agents/utils/`:

```bash
sed -n '6,25p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/core_stock_tools.py
```

```output
@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)
```

The router in `tradingagents/dataflows/interface.py` does the heavy lifting. There are two layers of indirection — *category* (the four logical groupings: `core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`) and *vendor* (`yfinance` or `alpha_vantage`). The user can configure both:

```bash
sed -n '116,156p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/interface.py
```

```output
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

```

```bash
sed -n '155,161p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/interface.py
```

```output
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError:
            continue  # Only rate limits trigger fallback

```

Two things to flag:

- The fallback chain is **rate-limit only**. `AlphaVantageRateLimitError` is the one exception that drops to the next vendor; every other error propagates. This is deliberate — if the user has selected Alpha Vantage but it's returning 'invalid symbol', you want the LLM to see that and try a different ticker, not silently swap to yfinance.
- The tool-level override (`tool_vendors`) takes precedence over the category-level (`data_vendors`). That lets you say 'use yfinance for everything except `get_news`, where I want Alpha Vantage's NEWS_SENTIMENT'.

## 13. Data vendors

### 13.1 yfinance — the default

`tradingagents/dataflows/y_finance.py` is the largest dataflow module. It does six things: OHLCV download, indicator time-series via the `stockstats` library, fundamentals overview, balance sheet / cash flow / income statement, and insider transactions. The OHLCV path caches per-symbol CSVs to disk so subsequent indicator calls don't re-hit Yahoo:

```bash
sed -n '45,92p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/stockstats_utils.py
```

```output
    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 15 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    # Reject ticker values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    safe_symbol = safe_ticker_component(symbol)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache uses a fixed window (15y to today) so one file per symbol
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

```

Two safety properties:

- `safe_ticker_component(symbol)` runs before the symbol is interpolated into a filesystem path. A ticker like `../../../etc/passwd` is rejected. This matters because tool args ultimately come from an LLM, which can be steered by prompt injection embedded in fetched news.
- The cache is filtered to `curr_date` *after* loading — so a backtest run dated 2024-01-01 never sees prices from 2024-02-01, even though they're sitting in the same cache file. This is the *look-ahead prevention* the whole framework cares about.

The same filter shows up on the financial-statement side as `filter_financials_by_date` (drops columns whose fiscal period ends after the cutoff) and on the Alpha Vantage side as `_filter_reports_by_date` (drops `annualReports` / `quarterlyReports` entries past the cutoff).

### 13.2 Indicator computation

Indicators are computed in bulk via the `stockstats` library, then sliced by date. `get_stock_stats_indicators_window` accepts a single indicator name (e.g. `rsi`) and a lookback window, returns a date → value string for each day in the window:

```bash
sed -n '169,200p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/y_finance.py
```

```output
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, curr_date)
```

The first attempt is the bulk path (`_get_stock_stats_bulk`): load the OHLCV once, call `df[indicator]` to trigger stockstats's lazy computation for *every* row at once, then look up the dates you want. If that fails, the fallback walks the window day-by-day, calling `get_stockstats_indicator` per date — slower but more resilient when stockstats hits an edge case.

### 13.3 StockTwits & Reddit — keyless public endpoints

The two social-data fetchers consumed by the Sentiment Analyst hit public, unauthenticated endpoints:

```bash
sed -n '26,52p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/stocktwits.py
```

```output
_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"


def fetch_stocktwits_messages(ticker: str, limit: int = 30, timeout: float = 10.0) -> str:
    """Fetch recent StockTwits messages for ``ticker`` and return them as a
    formatted plaintext block ready for prompt injection.

    Returns a placeholder string when the endpoint is unreachable, the
    symbol has no messages, or the response shape is unexpected — the
    caller never has to special-case None or exceptions.
    """
    url = _API.format(ticker=ticker.upper())
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        return f"<stocktwits unavailable: {type(exc).__name__}>"

    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not messages:
        return f"<no StockTwits messages found for ${ticker.upper()}>"

    lines = []
    bullish = bearish = unlabeled = 0
```

StockTwits returns each message with a user-labeled `Bullish`/`Bearish` tag. The fetcher counts them and prepends a one-line summary (`Bullish: 18 (60%) · Bearish: 6 (20%) · …`) before the message bodies — that ratio is exactly what the analyst's prompt is trained to read.

The Reddit fetcher hits `reddit.com/r/{sub}/search.json` (the public, no-auth endpoint) across `wallstreetbets`, `stocks`, and `investing`, throttled at ~10 req/min by an inter-request sleep:

```bash
sed -n '63,82p' /home/evan/opensource/TradingAgents/tradingagents/dataflows/reddit.py
```

```output
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` keeps us under Reddit's public rate limit
    (~10 req/min per IP) even if the caller queries many subreddits.
    """
    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(ticker, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

```

### 13.4 Alpha Vantage — the alternative

Alpha Vantage is the alternate vendor for every category. It's behind the rate-limit-fallback mechanic from §12. The unique thing about its module is the rate-limit signal: `_make_api_request` parses the response JSON and raises `AlphaVantageRateLimitError` when it sees Alpha Vantage's textual quota message — that's the exception the router catches to fall back to yfinance.

## 14. The bull/bear debate

After all selected analysts are done, the graph routes to the Bull Researcher. The bull/bear loop is the simplest possible debate structure — each researcher reads the analyst reports, the conversation history, and the opponent's last response, then produces a new argument. State is updated by appending to the histories:

```bash
sed -n '46,62p' /home/evan/opensource/TradingAgents/tradingagents/agents/researchers/bull_researcher.py
```

```output
        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
```

Each turn:
- prepends "Bull Analyst:" (or "Bear Analyst:") to the response so `should_continue_debate` can read the speaker from `current_response` (§9)
- appends to the joint `history` (which the opposing researcher reads as 'conversation history of the debate')
- appends to its own `bull_history` / `bear_history` (which the Research Manager reads at the end)
- bumps `count`

`count` is shared across both researchers, so the loop terminates after `2 × max_debate_rounds` total turns regardless of who started.

## 15. Structured outputs: Research Manager, Trader, Portfolio Manager

The three decision-making agents — Research Manager, Trader, Portfolio Manager — don't produce free-form prose. They use LangChain's `with_structured_output` to generate typed Pydantic instances. The schemas live in `tradingagents/agents/schemas.py`:

```bash
sed -n '23,42p' /home/evan/opensource/TradingAgents/tradingagents/agents/schemas.py
```

```output

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
```

```bash
sed -n '154,196p' /home/evan/opensource/TradingAgents/tradingagents/agents/schemas.py
```

```output
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
```

Field `description` strings double as the model's output instructions — that's why the prompt body for each of these agents doesn't have to re-explain what the fields mean.

The wrapper that ties schemas to LLMs lives in `tradingagents/agents/utils/structured.py` and codifies the canonical fall-back pattern:

```bash
sed -n '34,77p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/structured.py
```

```output
    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content
```

Two safety nets:

1. **Bind-time fallback**: if the provider's class raises `NotImplementedError` or `AttributeError` on `.with_structured_output` (mostly older Ollama models, or DeepSeek/MiniMax models the capability table marks `"none"`), the agent gets `structured_llm = None` and runs free-text generation for every call.
2. **Invocation-time fallback**: even if binding succeeded, a single failed `.invoke()` (malformed JSON from a weak model, transient provider error) triggers one retry with the plain LLM. The render-to-markdown path produces the same string shape downstream consumers expect.

### 15.1 Research Manager

Bull/bear is done. The Research Manager reads the joint debate history and emits a `ResearchPlan` with `recommendation`, `rationale`, and `strategic_actions`:

```bash
sed -n '17,52p' /home/evan/opensource/TradingAgents/tradingagents/agents/managers/research_manager.py
```

```output
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

**Debate History:**
{history}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

```

Note: the Research Manager runs on the **deep-thinking LLM** (passed in from `TradingAgentsGraph` constructor as `deep_thinking_llm`), not the quick model. Its output is rendered to a markdown block and saved as `investment_plan` in state.

### 15.2 Trader

The Trader is unique in two ways. First, it returns a `TraderProposal` whose rendering ends in a hardcoded `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**` line — a backward-compatibility marker that the analyst stop-signal text from earlier prompts can grep for:

```bash
sed -n '141,165p' /home/evan/opensource/TradingAgents/tradingagents/agents/schemas.py
```

```output
def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


```

Second, the Trader uses only three actions — `Buy`, `Hold`, `Sell` — not the five-tier rating. Position sizing nuance (`Overweight` / `Underweight`) is the Portfolio Manager's job, not the Trader's.

## 16. The risk debate

Three risk-management debators — Aggressive, Conservative, Neutral — argue over the Trader's proposal in a strict rotation that's enforced by `should_continue_risk_analysis` (§9). Each reads the analyst reports, the conversation history so far, and the other two debators' last responses, then makes its case.

The three personas are deliberate counterweights: the Aggressive analyst champions upside, the Conservative analyst protects against downside, the Neutral analyst balances both:

```bash
sed -n '17,21p' /home/evan/opensource/TradingAgents/tradingagents/agents/risk_mgmt/aggressive_debator.py /home/evan/opensource/TradingAgents/tradingagents/agents/risk_mgmt/conservative_debator.py /home/evan/opensource/TradingAgents/tradingagents/agents/risk_mgmt/neutral_debator.py
```

```output

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Aggressive Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:

```

After `3 × max_risk_discuss_rounds` total turns, control routes to the Portfolio Manager.

## 17. The Portfolio Manager

This is the final decision-maker. It reads everything: the Trader's proposal, the Research Manager's investment plan, the risk debate history, and crucially — `past_context`, the memory-log lessons injected at the start of the run:

```bash
sed -n '20,68p' /home/evan/opensource/TradingAgents/tradingagents/agents/managers/portfolio_manager.py
```

```output
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
```

Output goes into `final_trade_decision` (also mirrored into `risk_debate_state.judge_decision` so the CLI can display it under the right section). That's the string that gets returned by `.propagate()`.

## 18. Rating parsing & `SignalProcessor`

The Portfolio Manager always renders `**Rating**: X` into its markdown — `parse_rating` extracts it deterministically. No second LLM call:

```bash
sed -n '15,40p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/rating.py
```

```output
from typing import Tuple


# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Two-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Fall back to the first 5-tier rating word found anywhere in the text.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
```

Two passes: explicit `Rating: X` label (tolerant of `**Rating**: **Buy**`), then a fall-back scan for any 5-tier word. `SignalProcessor` in `signal_processing.py` is a back-compat wrapper around this function — older code expected a class with a `process_signal` method.

## 19. Memory log: Phase A and Phase B

This is the system's long-term memory. It's an append-only markdown file at `~/.tradingagents/memory/trading_memory.md` (overridable via `TRADINGAGENTS_MEMORY_LOG_PATH`).

### Phase A — write a pending entry at the end of each run

`store_decision` runs from `_run_graph` after every successful propagation. It tags the entry with rating + `pending` outcome:

```bash
sed -n '30,52p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/memory.py
```

```output

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
    ) -> None:
        """Append pending entry at end of propagate(). No LLM call."""
        if not self._log_path:
            return
        # Idempotency guard: fast raw-text scan instead of full parse
        if self._log_path.exists():
            raw = self._log_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                if line.startswith(f"[{trade_date} | {ticker} |") and line.endswith("| pending]"):
                    return
        rating = parse_rating(final_trade_decision)
        tag = f"[{trade_date} | {ticker} | {rating} | pending]"
        entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # --- Read path (Phase A) ---
```

An entry looks like `[2026-01-15 | NVDA | Buy | pending]` followed by the decision text. The HTML-comment separator (`<!-- ENTRY_END -->`) is the delimiter — chosen because LLM prose can't legally produce it.

### Phase B — resolve pending entries on the next same-ticker run

At the start of every `.propagate()` call, `_resolve_pending_entries` walks all pending entries for the current ticker, fetches the realised return, generates a one-paragraph reflection, and atomically rewrites the log:

```bash
sed -n '237,275p' /home/evan/opensource/TradingAgents/tradingagents/graph/trading_graph.py
```

```output
            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker, entry["date"], benchmark=benchmark,
            )
            if raw is None:
```

After resolution, the entry tag becomes `[2026-01-15 | NVDA | Buy | +5.4% | +1.2% | 5d]` — raw return, alpha vs benchmark, and the realized holding period. A short reflection is appended after the original decision text.

The benchmark for each ticker is auto-selected via the suffix map in `default_config.py`:

```bash
sed -n '194,214p' /home/evan/opensource/TradingAgents/tradingagents/graph/trading_graph.py
```

```output
    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

```

So a `.T` ticker gets benchmarked against the Nikkei 225 (`^N225`), `.HK` against the Hang Seng, US tickers against SPY. The user can override with `TRADINGAGENTS_BENCHMARK_TICKER`.

### 19.1 Memory injection into the next run

When `get_past_context(ticker)` is called from `_run_graph`, it returns a formatted block with the most recent 5 same-ticker decisions plus 3 cross-ticker lessons:

```bash
sed -n '74,103p' /home/evan/opensource/TradingAgents/tradingagents/agents/utils/memory.py
```

```output
        if not entries:
            return ""

        same, cross = [], []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    # --- Update path (Phase B) ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
```

The string ends up in the Portfolio Manager's prompt under 'Lessons from prior decisions and outcomes', so the PM literally sees a list of past Buy/Hold/Sell calls with their realised alpha and a 2-4 sentence post-mortem. That feedback loop is the whole point of the memory log.

### 19.2 Reflection prompt

The reflection itself is one quick-LLM call per pending entry:

```bash
sed -n '14,52p' /home/evan/opensource/TradingAgents/tradingagents/graph/reflection.py
```

```output
    def _get_log_reflection_prompt(self) -> str:
        """Concise prompt for reflect_on_final_decision (Phase B log entries).

        Produces 2-4 sentences of plain prose — compact enough to be re-injected
        into future agent prompts without bloating the context window.
        """
        return (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            "1. Was the directional call correct? (cite the alpha figure)\n"
            "2. Which part of the investment thesis held or failed?\n"
            "3. One concrete lesson to apply to the next similar analysis.\n\n"
            "Be specific and terse. Your output will be stored verbatim in a decision log "
            "and re-read by future analysts, so every word must earn its place."
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark_name: str = "SPY",
    ) -> str:
        """Single reflection call on the final trade decision with outcome context.

        Used by Phase B deferred reflection. The final_trade_decision already
        synthesises all analyst insights, so no separate market context is needed.
        ``benchmark_name`` is the label used for the alpha line (e.g. ``"SPY"``
        for US tickers, ``"^N225"`` for ``.T`` listings); defaults to SPY for
        callers that haven't been updated to thread the benchmark through.
        """
        messages = [
            ("system", self.log_reflection_prompt),
            (
                "human",
                (
                    f"Raw return: {raw_return:+.1%}\n"
                    f"Alpha vs {benchmark_name}: {alpha_return:+.1%}\n\n"
```

The prompt is *deliberately* terse — 2-4 sentences max, because each reflection ends up living in future prompts' context windows.

## 20. Checkpoint resume

Opt-in (`--checkpoint` on the CLI, `config["checkpoint_enabled"] = True` in code). When enabled, LangGraph saves state after each node into a per-ticker SQLite database, and a crashed run can pick up where it left off. The plumbing:

```bash
sed -n '21,42p' /home/evan/opensource/TradingAgents/tradingagents/graph/checkpointer.py
```

```output
    # Reject ticker values that would escape the checkpoints directory.
    safe = safe_ticker_component(ticker).upper()
    p = Path(data_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.db"


def thread_id(ticker: str, date: str) -> str:
    """Deterministic thread ID for a ticker+date pair."""
    return hashlib.sha256(f"{ticker.upper()}:{date}".encode()).hexdigest()[:16]


@contextmanager
def get_checkpointer(data_dir: str | Path, ticker: str) -> Generator[SqliteSaver, None, None]:
    """Context manager yielding a SqliteSaver backed by a per-ticker DB."""
    db = _db_path(data_dir, ticker)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
```

Three design choices:

- **One database per ticker** so concurrent runs on different tickers don't contend on a single SQLite file.
- **Thread ID is a hash of `(ticker, date)`** — same ticker + same date resumes; same ticker + different date starts fresh.
- **`safe_ticker_component`** again guards path traversal.

The recompile happens inside `.propagate()`:

```bash
sed -n '299,332p' /home/evan/opensource/TradingAgents/tradingagents/graph/trading_graph.py
```

```output
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date, asset_type=asset_type)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
```

On success, `clear_checkpoint` wipes the rows for that thread-id so stale state doesn't accidentally resume on a future re-run.

## 21. State dump and saved reports

When the graph finishes, `_log_state` writes the full state to disk:

```bash
sed -n '386,408p' /home/evan/opensource/TradingAgents/tradingagents/graph/trading_graph.py
```

```output

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
```

Each run leaves behind a JSON snapshot at `<results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<DATE>.json`. The CLI also saves a human-readable markdown report (`save_report_to_disk` in `cli/main.py`) that bundles each report into a numbered section folder (`1_analysts/`, `2_research/`, …, `5_portfolio/`).

## 22. The CLI display layer

`cli/main.py` is large (1287 lines) but its three responsibilities are crisp:

1. **Pre-run user-selection wizard** (`get_user_selections`) — eight steps, in order: ticker → date → language → analyst checkbox → research depth → LLM provider → models → provider-specific thinking config.
2. **Live Rich layout** (`update_display`) — a four-panel layout with a per-agent status table, a scrolling message log, the current report, and a footer with token/call stats.
3. **`run_analysis`** — wires the user selections into a `TradingAgentsGraph`, then streams the graph and updates the Rich layout on every chunk.

The state machine for analyst status transitions is interesting because the graph itself doesn't emit explicit start/end events — the CLI infers them from accumulated report state:

```bash
sed -n '855,899p' /home/evan/opensource/TradingAgents/cli/main.py
```

```output
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
```

**Inference, not events**: the CLI looks at `message_buffer.report_sections` — the running accumulation of report content seen across chunks — and decides who's done, who's running, and who's still waiting. The first analyst with no report (and a non-empty pending state) is treated as 'in_progress'. When all analyst reports are populated, the Bull Researcher flips to 'in_progress'.

The stats footer is fed by a tiny LangChain callback handler:

```bash
sed -n '8,76p' /home/evan/opensource/TradingAgents/cli/stats_handler.py
```

```output

class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }
```

The handler is bound to **both** the LLM constructors (so it sees `on_chat_model_start` / `on_llm_end`) and the graph config (so it sees `on_tool_start`). The lock is necessary because LangGraph can run tool nodes in parallel.

## 23. End-to-end: tracing a single propagation

Putting it all together — here's the sequence of calls when you run `python main.py` with the default config for `NVDA / 2024-05-10`:

1. **Import-time**: `tradingagents/__init__.py` loads `.env`, suppresses langgraph-checkpoint warnings.
2. **`DEFAULT_CONFIG` is built**: `_apply_env_overrides` overlays any `TRADINGAGENTS_*` env vars onto the defaults.
3. **`TradingAgentsGraph(debug=True, config=...)`** is constructed:
   - `set_config(config)` propagates into `dataflows/config.py`.
   - `create_llm_client('openai', 'gpt-5.4', …)` for the deep model; same for the quick model.
   - `TradingMemoryLog` is loaded from `~/.tradingagents/memory/trading_memory.md`.
   - Four `ToolNode`s are created (one per analyst type), each holding the `@tool`-decorated functions.
   - `GraphSetup.setup_graph(["market", "social", "news", "fundamentals"])` builds the StateGraph: adds each analyst's three nodes, threads the conditional edges from §8, terminates at the Portfolio Manager.
   - `workflow.compile()` produces the LangGraph executor.
4. **`.propagate("NVDA", "2024-05-10")`** is called:
   - `_resolve_pending_entries("NVDA")` walks the memory log, resolves any `[… | NVDA | … | pending]` entries via yfinance prices and one reflection LLM call each. The pending tags become full `raw/alpha/holding` tags.
   - `_run_graph` builds initial state via `Propagator.create_initial_state`, prepends `past_context` from the memory log.
   - The StateGraph streams chunks. Each chunk is a per-node delta:
     - **Market Analyst** loops with `get_stock_data` → `get_indicators` (yfinance) → final report.
     - **Sentiment Analyst** pre-fetches `get_news` + StockTwits + Reddit, then one LLM call.
     - **News Analyst** loops over `get_news` / `get_global_news` / `get_insider_transactions`.
     - **Fundamentals Analyst** loops over `get_fundamentals` / `get_balance_sheet` / `get_cashflow` / `get_income_statement`.
     - **Bull / Bear Researcher** alternate for `2 × max_debate_rounds` turns.
     - **Research Manager** emits a `ResearchPlan` (deep model).
     - **Trader** emits a `TraderProposal` (quick model) ending in `FINAL TRANSACTION PROPOSAL: **BUY**`.
     - **Aggressive / Conservative / Neutral** debators rotate for `3 × max_risk_discuss_rounds` turns.
     - **Portfolio Manager** emits a `PortfolioDecision` (deep model) — `Rating: Buy`, executive summary, thesis.
   - `_log_state` writes the JSON snapshot.
   - `memory_log.store_decision` appends a `pending` entry for the next run.
5. **`signal_processor.process_signal(final_state["final_trade_decision"])`** parses the rating word.
6. **`return final_state, decision`** — `decision` is one of `Buy`, `Overweight`, `Hold`, `Underweight`, `Sell`.

Counting LLM calls: with default `max_debate_rounds = max_risk_discuss_rounds = 1` and all four analysts selected, a typical run makes roughly **15-20 LLM calls** — four analyst rounds (each spanning multiple LLM ↔ tool turns), two researcher turns, one Research Manager, one Trader, three risk debators, one Portfolio Manager, plus any deferred reflection calls from §19.

## 24. Files map: what to read next

If you need to:

- **Add a new LLM provider** → `tradingagents/llm_clients/{factory,api_key_env,validators,model_catalog}.py` plus a new subclass of `BaseLLMClient` modelled on `OpenAIClient` / `AnthropicClient`.
- **Add a new data vendor** → register it in `tradingagents/dataflows/interface.py`'s `VENDOR_METHODS` table and add the implementation module.
- **Change the agent prompts** → `tradingagents/agents/{analysts,researchers,managers,trader,risk_mgmt}/*.py`.
- **Rewire the graph** → `tradingagents/graph/setup.py` plus the matching `ConditionalLogic` method.
- **Tune debate length** → `max_debate_rounds` and `max_risk_discuss_rounds` (CLI 'Research Depth' or config).
- **Persist new state across runs** → `tradingagents/agents/utils/memory.py` (add a Phase B-style outcome update), or add a new context-injection point in `_run_graph`.
- **Track new run metrics** → `cli/stats_handler.py` (or write your own `BaseCallbackHandler` and pass it to `TradingAgentsGraph(callbacks=[...])`).

The README at the root of the repo, `CHANGELOG.md`, and the file headers of each module are the canonical places to go deeper. This walkthrough has tried to give you the shape; the source — as always — is the truth.
