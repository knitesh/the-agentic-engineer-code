-- migrations/001_memory.sql  — run once, on the SAME Postgres as the checkpointer.
CREATE EXTENSION IF NOT EXISTS vector;          -- the pgvector extension

CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    namespace   TEXT        NOT NULL,            -- tenant/user scope (6.5)
    text        TEXT        NOT NULL,            -- the stored fact / chunk
    embedding   vector(1536) NOT NULL,           -- dim MUST match EMBED_DIM
    metadata    JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Filter by namespace cheaply (we WHERE on it before every similarity search).
CREATE INDEX IF NOT EXISTS memories_namespace_idx ON memories (namespace);

-- Approximate-nearest-neighbor index for fast similarity search. HNSW gives
-- better recall/latency than ivfflat at the cost of build time and memory;
-- vector_cosine_ops matches the <=> cosine operator we query with.
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
