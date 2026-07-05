## agent/supervisor_runtime.py — the delegation contract and the 8.3 helper
## (scoped + isolated subagent execution), imported by agent/supervisor.py.
import time, uuid
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from agent.graph import worker_for               # scoped guarded agent per role (Ch7)
from agent.config import build_subagent_config


class SubagentTask(BaseModel):
    """Supervisor -> subagent. A clean contract, like a tool's input (Ch5)."""
    goal: str = Field(description="The specific, self-contained task for the subagent.")
    context: str = Field(default="", description="Only the context this task needs.")

class SubagentResult(BaseModel):
    """Subagent -> supervisor. The result, NOT the transcript."""
    answer: str = Field(description="The subagent's result for its assigned task.")
    status: str = Field(description="'complete' | 'partial' | 'failed'.")
    evidence: str = Field(default="", description="Brief support; not the full trace.")


def _compose(task: "SubagentTask") -> str:
    """Repo helper: render the delegation contract as the subagent's request."""
    if task.context:
        return f"{task.goal}\n\n[Context]\n{task.context}"
    return task.goal


def fresh_worker_state(task: "SubagentTask") -> dict:
    """A complete AgentState seed — EVERY field the Ch3/Ch7 loop reads, so
    evaluate_exit's time/budget checks never KeyError. (Ch7 discipline.)"""
    return {
        "goal": task.goal,
        "messages": [HumanMessage(content=_compose(task))],
        "iterations": 0, "tokens_used": 0, "started_at": time.time(),
        "action_history": [], "seen_signatures": set(), "stall_count": 0,
        "exit_reason": None, "final_answer": None, "status": None,
        "guard_block": None, "approval_granted": False,   # Ch7-added fields
    }

def run_subagent(role: str, task: "SubagentTask") -> "SubagentResult":
    """Delegate ONE task to a SCOPED subagent with an ISOLATED context.
    Fresh state (no supervisor history) + fresh thread_id (no checkpoint
    collision) + role-scoped worker (tool scoping actually enforced)."""
    worker = worker_for(role)                     # scoped, compiled once per role
    fresh = fresh_worker_state(task)              # ISOLATION: brand-new context
    # The worker is checkpointer-compiled (Ch6/Ch7), so it REQUIRES a thread_id.
    # A fresh id per delegation keeps subagent runs from colliding on one thread.
    cfg = {"configurable": {"thread_id": f"sub-{uuid.uuid4()}"}}
    out = worker.invoke(fresh, config=cfg)
    return SubagentResult(                        # distill — NOT the transcript
        answer=out.get("final_answer") or out["messages"][-1].content,
        status=out.get("status") or "complete",
    )
