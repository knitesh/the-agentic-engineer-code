## agent/config.py — HarnessConfig + the four sub-configs (3.6)
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
    enabled_tools: list[str] = field(default_factory=lambda: ["calculator", "web_search"])
    # tools the agent may NOT use in this context, even if registered globally
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
