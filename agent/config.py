## agent/config.py — HarnessConfig + the four sub-configs (3.6);
## ToolConfig default updated to the real suite (5.6); context budgets (6.4).
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
    # --- action boundaries (5.7) — pre-wired here, consumed by Ch7's approval gate ---
    require_approval: bool = False
    # --- context management budgets (6.4) ---
    context_token_budget: int = 24_000   # compress when the window exceeds this
    keep_recent: int = 8                 # messages kept verbatim after compression


## The shared module-level config (imported as `from agent.config import CONFIG`
## by Ch6's memory backend and the Ch6/Ch7 graph wiring).
CONFIG = HarnessConfig(
    memory=MemoryConfig(
        backend="postgres" if os.environ.get("DATABASE_URL") else "memory",
        connection_string=os.environ.get("DATABASE_URL"),
    ),
)
