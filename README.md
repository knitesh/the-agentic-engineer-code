# The Agentic Engineer — Companion Code

The official companion repository for **[The Agentic Engineer: Designing, Evaluating, and Operating Multi-Agent AI Systems in Production](https://www.amazon.com/dp/B0H7GXBVNC)** by Nitesh Mishra.

> 📖 **Get the book on Amazon:** <https://www.amazon.com/dp/B0H7GXBVNC>
>
> The book builds one production-grade agent progressively — a harness-wrapped, guarded, memory-backed, traced, evaluated, multi-agent system — one capability per chapter. This repository is that project, with a tagged milestone for every hands-on step, so you can check out the code at any point and run exactly the state the book describes.

## What's in the repo

The repo is organized so the agent package evolves coherently rather than restarting each chapter:

```
agent/
  state.py           # AgentState (Ch3), evolves with declared additions
  config.py          # HarnessConfig + sub-configs (Ch3, 9, 11, 12)
  loop.py            # evaluate_exit, finalize (Ch3)
  harness.py         # AgentHarness.run_safe + run modes (Ch3, 9, 11)
  graph.py           # the compiled guarded ReAct agent (Ch4, 7)
  tools.py           # the scoped tool suite (Ch5)
  memory.py          # long-term semantic memory over pgvector (Ch6)
  memory_backend.py  # stores + checkpointer selection (Ch6, 11)
  guardrails.py      # input/output/tool-call layer (Ch7)
  supervisor.py      # multi-agent supervisor + workers (Ch8)
  reliability.py     # retries, rate limiter (Ch11)
  fallback.py        # graceful degradation (Ch11)
  approval_policy.py # human-gate policy (Ch12)
evals/               # datasets + experiment runner (Ch10)
migrations/          # Postgres/pgvector DDL (Ch6)
service.py           # the FastAPI serving layer (Ch11)
examples/            # Chapter 13's coding-agent case-study extension
```

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/knitesh/the-agentic-engineer-code.git
cd the-agentic-engineer-code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your OPENAI_API_KEY
```

Run the agent:

```bash
export $(grep -v '^#' .env | xargs)   # or use your preferred env loader
python -m agent.run
```

Everything runs **without** Postgres or Langfuse by default (in-memory checkpointer, tracing off). To enable the durable pieces:

- **Postgres + pgvector (Ch6, Ch11):** set `DATABASE_URL` in `.env`, then apply `migrations/001_memory.sql`.
- **Langfuse tracing (Ch9) and evals (Ch10):** set the `LANGFUSE_*` keys in `.env` and turn on `tracing_enabled` in `HarnessConfig`.
- **The file tools (Ch5):** set `AGENT_WORKSPACE` to an **absolute** path the agent may read/write.

## Milestones — follow along chapter by chapter

Each chapter's hands-on step (the 🔨 markers in the book) corresponds to a tagged commit. To check out the project as it stood at the end of a chapter:

```bash
git checkout ch07-guardrails     # the agent at the end of Chapter 7
```

| Tag | Chapter | The agent gains |
|---|---|---|
| `ch02-scaffold` | 2. The Building Blocks | state, a safe tool, a prompt, the first reason/act loop |
| `ch03-harness-loop` | 3. Harness & Loop Engineering | AgentHarness.run_safe, budgets, deadlock detection, honest exits |
| `ch04-react` | 4. Agent Patterns | the full ReAct graph wired to Ch3's exit logic |
| `ch05-tools` | 5. Tool Design & Integration | the real four-tool suite with contracts and failure-as-observation |
| `ch06-memory` | 6. Memory & State Management | pgvector long-term memory, recall, context compression |
| `ch07-guardrails` | 7. Guardrails | input/output guards + per-call tool authorization |
| `ch08-supervisor` | 8. Orchestration & Multi-Agent | supervisor graph delegating to scoped, isolated workers |
| `ch09-tracing` | 9. Observability & Tracing | Langfuse tracing wired in the harness — nodes untouched |
| `ch10-evals` | 10. Evaluation & Testing | dataset-driven experiments, layered evaluators, CI gates |
| `ch11-production` | 11. Production & Reliability | rate limiting, graceful fallback, the FastAPI service |
| `ch12-approval-gate` | 12. Safety, Trust & Human Oversight | the interrupt-based human approval gate |

`main` is `ch12-approval-gate` plus Chapter 13's coding-agent case-study extension (`examples/ch13_coding_agent.py`).

Start from the chapter you're reading; each milestone is a runnable state, not a fragment.

To return to the latest state:

```bash
git checkout main
```

## How the code maps to the book

Every listing that names a file in the book (e.g. `## agent/loop.py`) appears here verbatim, in that file. Where the book explicitly defers plumbing to the repo ("see repo") — the sandbox primitive, the supervisor prompt, dev fallbacks so everything runs without external services — those pieces are marked with `## Repo glue` / `## Repo helper` comments so you can tell the book's teaching code from the connective tissue.

## Trying the bigger pieces

```bash
# The multi-agent supervisor (Ch8)
python -c "
from agent.supervisor import supervisor_app
out = supervisor_app.invoke({'goal': 'What is 17% of 4,200? Verify with a calculation.',
                             'results': [], 'iterations': 0})
print(out['final_answer'])"

# The HTTP service (Ch11)
uvicorn service:app_http --reload
curl -X POST localhost:8000/run -H 'content-type: application/json' \
     -d '{"message": "what is 17% of 4,200, then add 90?"}'

# Evals (Ch10) — needs Langfuse keys; seed once, then run
python -m evals.seed_dataset
python -m evals.run_experiment
```

## The book

If this repository landed here without the book: it is the reference implementation for *The Agentic Engineer*, which walks through **why** every one of these pieces exists — the failure modes, the tradeoffs, and the engineering decisions behind each line.

**⭐ Available now on Amazon (Kindle, paperback & hardcover): <https://www.amazon.com/dp/B0H7GXBVNC>**

If the book or this code helped you build something, a review on Amazon genuinely helps other engineers find it.

---

© 2026 Nitesh Mishra. Code licensed for use with the book; see the book for the full context each module assumes.
