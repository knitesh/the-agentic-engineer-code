## agent/memory_backend.py — one Postgres, two roles.
from typing import cast

from psycopg import Connection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import Conn, PostgresSaver
from agent.memory import SemanticMemory

## Repo glue: the pinned embedder (6.2). One model, one dimension — changing it
## means re-embedding everything (EMBED_DIM in agent/memory.py must match).
from langchain_openai import OpenAIEmbeddings

def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-small")   # 1536-dim


def build_memory(config) -> tuple[PostgresSaver, SemanticMemory]:
    """One connection string (MemoryConfig) serves BOTH:
      - the checkpointer  (session persistence / short-term durability)
      - the vector store  (long-term semantic memory)
    """
    conn = Connection.connect(
        config.memory.connection_string,
        row_factory=dict_row,   # pyright: ignore[reportArgumentType]
    )

    # Session / short-term durability — Chapter 3's checkpointer.
    checkpointer = PostgresSaver(cast(Conn, conn))
    checkpointer.setup()

    # Long-term semantic memory — this chapter's vector store, same DB.
    semantic = SemanticMemory(
        conn=conn,
        embedder=get_embedder(),
        namespace=config.memory.namespace,     # scope from MemoryConfig
    )
    return checkpointer, semantic


## Repo glue: a dev fallback so the project runs without Postgres.
## MemoryConfig.backend defaults to "memory" (3.6); "postgres" selects the
## durable pair above. The in-memory stand-ins honor the same interface.
from langgraph.checkpoint.memory import MemorySaver

class _InMemorySemanticMemory:
    """Interface-compatible stand-in for SemanticMemory (dev only, not persistent)."""
    def __init__(self):
        self._facts: list[tuple[str, dict]] = []

    def remember(self, text: str, metadata: dict | None = None) -> None:
        self._facts.append((text, metadata or {}))

    def recall(self, query: str, k: int = 5, min_score: float = 0.75) -> list:
        return []                       # no similarity search without a vector store

    def archive_messages(self, messages) -> None:
        for m in messages:
            self._facts.append((getattr(m, "content", str(m)), {"kind": "archive"}))


def build_memory_inmemory():
    return MemorySaver(), _InMemorySemanticMemory()


## agent/memory_backend.py — the single source of truth for the stores.
from agent.config import CONFIG
## build_memory is the function from 6.6.
if CONFIG.memory.backend == "postgres":
    checkpointer, memory = build_memory(CONFIG)     # one connection, two roles
    # Repo glue for 6.4's archive step: archived turns become recallable facts.
    def _archive_messages(messages, _m=memory):
        for msg in messages:
            _m.remember(getattr(msg, "content", str(msg)), metadata={"kind": "archive"})
    setattr(memory, "archive_messages", _archive_messages)
else:
    checkpointer, memory = build_memory_inmemory()  # dev fallback (see above)
