## service.py — the agent as an HTTP service. The harness IS the core;
## this is a thin transport layer over run_safe (Ch3) + graceful fallback (11.5).
from fastapi import FastAPI
from pydantic import BaseModel

from agent.fallback import graceful_response

## ---------------------------------------------------------------------------
## Harness wiring. Dev default: the shared guarded app (in-memory checkpointer
## unless DATABASE_URL selects Postgres — see agent/memory_backend.py).
##
## The full production wiring from 11.7 — config-driven, durable checkpointer,
## tracing on, tuned budgets — looks like this:
##
##   from langgraph.checkpoint.postgres import PostgresSaver
##
##   prod_config = HarnessConfig(
##       model=ModelConfig(name="gpt-4o", temperature=0.0),
##       env="prod",
##       tracing_enabled=True,                 # Ch9: every run traced
##       recursion_limit=25,                   # Ch3: catastrophe net
##       max_seconds=90,                       # Ch3: real wall-clock timeout
##       invoke_workers=16,                    # 11.3: bounded concurrency
##       retry=RetryConfig(max_retries=4, base_backoff_seconds=0.5),  # 11.2 + jitter
##       # memory_backend selects the durable checkpointer (Ch3 §3.6, §3.18)
##   )
##
##   with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
##       checkpointer.setup()
##       app = guarded_graph.compile(checkpointer=checkpointer)   # Ch7 guarded graph
##       harness = AgentHarness(app, prod_config, sinks)          # Ch3 harness
##       # service.py (11.6) imports `harness`; graceful_response (11.5) wraps results.
## ---------------------------------------------------------------------------
from agent.config import CONFIG
from agent.graph import app
from agent.harness import AgentHarness
from agent.sinks import Sinks

harness = AgentHarness(app, CONFIG, Sinks())

app_http = FastAPI()

class RunRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str | None = None

@app_http.post("/run")
def run(req: RunRequest):
    result = harness.run_safe(req.message, thread_id=req.thread_id,
                              user_id=req.user_id)
    return graceful_response(result)          # honest exits → user-facing responses

@app_http.get("/healthz")
def health():
    return {"status": "ok"}                   # liveness for the load balancer
