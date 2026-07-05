## agent/config.py — HarnessConfig + the four sub-configs (3.6);
## ToolConfig default updated to the real suite (5.6); context budgets (6.4);
## guardrail settings (7.2–7.3); per-role scoped configs + supervisor budget (8.3–8.6).
import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "gpt-4o"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    fallback_model: str | None = "gpt-4o-mini"   # used if primary fails
    window_strategy: str = "summarize"            # "truncate"|"summarize"|"retain"


@dataclass
class ToolConfig:
    enabled_tools: list[str] = field(default_factory=lambda: [
        "calculator", "web_search", "read_file", "write_file", "remember",
    ])
    denied_tools: list[str] = field(default_factory=list)


@dataclass
class MemoryConfig:
    backend: str = "memory"          # "memory" | "postgres" | "redis"
    connection_string: str | None = None
    namespace: str = "default"       # isolate users/tenants


@dataclass
class RetryConfig:
    max_retries: int = 2
    base_backoff_seconds: float = 1.0
    # retryable errors are resolved at runtime from the provider tuple in 3.5
    on_exhausted: str = "fail"       # "fail" | "fallback_model" | "degrade"


@dataclass
class HarnessConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    # --- loop budgets (Section 2 uses these heavily) ---
    max_iterations: int = 12
    max_tokens_budget: int = 100_000
    max_seconds: float = 90.0
    recursion_limit: int = 50          # mechanical catastrophe net (3.11)
    # --- harness internals ---
    invoke_workers: int = 8            # thread pool for time-bounded invokes
    stall_threshold: int = 3           # iterations of no progress -> divergence (3.12)
    # re-entry policy (3.15): "reject" | "queue" | "interrupt" | "incorporate"
    re_entry_policy: str = "queue"
    # --- action boundaries (5.7) — pre-wired in Ch5, consumed by Ch7's approval gate ---
    require_approval: bool = False
    # --- context management budgets (6.4) ---
    context_token_budget: int = 24_000   # compress when the window exceeds this
    keep_recent: int = 8                 # messages kept verbatim after compression
    # --- guardrails (7.2–7.3) ---
    max_input_chars: int = 4_000         # front-door structural bound (7.2)
    scope: str = "general"               # topic scope for the input guard (7.2)
    allowed_pii: list[str] = field(default_factory=list)   # output guard (7.4)
    max_actions_per_run: int = 24        # per-run action budget (7.3)
    max_supervisor_steps: int = 8        # NEW: ceiling on supervisor delegations
    # --- observability (9.4) ---
    tracing_enabled: bool = False        # Langfuse tracing, wired in the harness
    env: str = "dev"                     # trace tag: dev | staging | prod


## The shared module-level config (imported as `from agent.config import CONFIG`
## by Ch6's memory backend and the Ch6/Ch7 graph wiring).
CONFIG = HarnessConfig(
    memory=MemoryConfig(
        backend="postgres" if os.environ.get("DATABASE_URL") else "memory",
        connection_string=os.environ.get("DATABASE_URL"),
    ),
)

## The shared model client, used throughout the book (Ch8's supervisor chain,
## the graph factory, and the Ch10 judge all import this).
from langchain_openai import ChatOpenAI    # swap for your provider

llm = ChatOpenAI(model=CONFIG.model.name, temperature=CONFIG.model.temperature)


## agent/config.py — per-role scoped configs. Each compiles (via worker_for) into
## the SAME guarded agent (Ch7), differing only in tool scope and system prompt.
## Least privilege = specialization.
ROLE_CONFIGS = {
    "research": HarnessConfig(
        tools=ToolConfig(enabled_tools=["web_search", "read_file"]),  # no write, no compute
        # system prompt: "You research and report findings. You do not write files."
    ),
    "compute": HarnessConfig(
        tools=ToolConfig(enabled_tools=["calculator", "run_python"]),  # no web, no write
    ),
    "writer": HarnessConfig(
        tools=ToolConfig(enabled_tools=["read_file", "write_file"]),   # can persist results
        require_approval=True,        # writing is high-stakes -> Ch7 approval gate
    ),
}

def build_subagent_config(role: str) -> HarnessConfig:
    """Return the scoped config for a role; raises on an unknown role (fail closed)."""
    if role not in ROLE_CONFIGS:
        raise ValueError(f"unknown subagent role: {role}")
    return ROLE_CONFIGS[role]
