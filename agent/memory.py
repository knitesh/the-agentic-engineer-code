## agent/memory.py
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb       # adapt dict -> JSONB (see remember())
from pgvector.psycopg import register_vector
from langchain_core.embeddings import Embeddings

## One embedding model, PINNED. Changing it means re-embedding everything.
EMBED_DIM = 1536                         # must match the model's output dim
                                         # AND the vector(1536) column in the DDL


@dataclass
class Memory:
    text: str
    score: float                          # similarity, for thresholding
    metadata: dict


class SemanticMemory:
    """Long-term semantic memory backed by pgvector — the SAME Postgres
    instance as the Chapter 3 checkpointer."""

    def __init__(self, conn: psycopg.Connection[Any], embedder: Embeddings,
                 namespace: str):
        self.conn = conn
        self.embedder = embedder
        self.namespace = namespace        # tenant/user isolation (MemoryConfig)
        register_vector(self.conn)

    def remember(self, text: str, metadata: dict | None = None) -> None:
        """Promote a fact into long-term storage. Called deliberately,
        not on every message (see 6.5)."""
        vec = self.embedder.embed_query(text)
        self.conn.execute(
            "INSERT INTO memories (namespace, text, embedding, metadata, created_at) "
            "VALUES (%s, %s, %s, %s, now())",
            # Jsonb(...) wrapper required — psycopg3 won't adapt a bare dict to JSONB.
            (self.namespace, text, vec, Jsonb(metadata or {})),
        )
        self.conn.commit()

    def recall(self, query: str, k: int = 5,
               min_score: float = 0.75) -> list[Memory]:
        """Retrieve up to k relevant memories for THIS namespace, above a
        similarity floor. Filter by namespace FIRST — never cross tenants."""
        qvec = self.embedder.embed_query(query)
        rows = self.conn.execute(
            # cosine distance operator <=>; similarity = 1 - distance.
            "SELECT text, metadata, 1 - (embedding <=> %s) AS score "
            "FROM memories "
            "WHERE namespace = %s "                 # metadata filter — critical
            "ORDER BY embedding <=> %s "            # nearest neighbors
            "LIMIT %s",
            (qvec, self.namespace, qvec, k),
        ).fetchall()
        # Apply the relevance floor — discard the index's nearest-but-irrelevant.
        return [
            Memory(text=r[0], metadata=r[1], score=r[2])
            for r in rows if r[2] >= min_score
        ]
