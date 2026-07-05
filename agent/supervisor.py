## agent/supervisor.py
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from agent.config import CONFIG
from agent.config import llm      # the shared model client (see agent/config.py)
from agent.supervisor_runtime import run_subagent          # the 8.3 helper (scoped+isolated)
## SubagentTask / SubagentResult: as defined in 8.2
from agent.supervisor_runtime import SubagentTask, SubagentResult
## SUPERVISOR_PROMPT: a ChatPromptTemplate (see repo), analogous to Ch4's PLANNER_PROMPT
## llm: the shared model client from agent/config.py, used throughout the book
from langchain_core.prompts import ChatPromptTemplate

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a supervisor coordinating specialist subagents to complete a goal.\n"
     "Available roles:\n"
     "- research: searches the web and reads files; cannot write or compute.\n"
     "- compute:  does arithmetic and computation; no web access, no writes.\n"
     "- writer:   reads and writes files to persist results; writes are gated.\n\n"
     "Look at the goal and the results so far. If the goal is fully met, set\n"
     "is_done=true and produce the final answer. Otherwise pick ONE role and\n"
     "give it ONE specific, self-contained task. Delegate the smallest useful\n"
     "unit of work; do not re-request work that already succeeded."),
    ("human", "Goal:\n{goal}\n\nResults so far:\n{results}"),
])

class SupervisorState(TypedDict):
    goal: str
    results: Annotated[list, operator.add]                # accumulates (Ch3 reducer)
    next_agent: str | None
    next_task: str | None                                 # declared, not bolted on (Ch7)
    final_answer: str | None
    status: str | None
    iterations: int                                       # supervisor budget (Ch3)

class Delegation(BaseModel):
    """Supervisor's structured decision each step."""
    is_done: bool = Field(description="True if the goal is fully met.")
    final_answer: str | None = Field(default=None)
    role: str | None = Field(default=None, description="Subagent to run if not done.")
    task: str | None = Field(default=None, description="The subtask for that subagent.")

## Structured-output chain: the supervisor reasons over goal + results so far.
supervisor_chain = SUPERVISOR_PROMPT | llm.with_structured_output(Delegation)


def _summarize(results: list) -> str:
    """Repo helper: distill accumulated subagent results for the honest exit."""
    if not results:
        return "none"
    return " | ".join(f"[{r.get('status', '?')}] {str(r.get('answer', ''))[:200]}"
                      for r in results)


def supervise_node(state: SupervisorState) -> dict:
    """Decide: delegate to a subagent, or declare done. Bounded by iterations."""
    decision = supervisor_chain.invoke({"goal": state["goal"],
                                        "results": state["results"]})
    if decision.is_done:
        return {"status": "complete", "final_answer": decision.final_answer,
                "next_agent": None}
    return {"next_agent": decision.role, "next_task": decision.task,
            "iterations": state["iterations"] + 1}

def delegate_node(state: SupervisorState) -> dict:
    """Run the chosen subagent (scoped + isolated, via 8.3's run_subagent),
    then integrate its result WITH a status check — don't blindly chain (8.5)."""
    task = SubagentTask(goal=state["next_task"], context="")
    result = run_subagent(state["next_agent"], task)      # scoped worker, fresh thread (8.3)
    # Status-aware integration (8.5): surface failures instead of silently chaining.
    if result.status == "failed":
        note = (f"[NOTE] The {state['next_agent']} subagent FAILED its task "
                f"('{state['next_task']}'). Re-plan: try another approach, a "
                f"different subagent, or report the task as not fully completable.")
        result = SubagentResult(answer=note, status="failed")
    return {"results": [result.model_dump()]}             # accumulates via reducer

def route(state: SupervisorState) -> str:
    """Exit valve — bounded, like every loop in this book."""
    if state.get("status") == "complete":
        return "finalize"
    if state["iterations"] >= CONFIG.max_supervisor_steps:   # hard budget (Ch3 discipline)
        return "finalize"
    return "delegate"

def supervisor_finalize(state: SupervisorState) -> dict:
    """Honest exit (Ch3 finalize_node discipline): if we stopped without a
    complete status, say so — never fabricate a finish."""
    if state.get("status") == "complete" and state.get("final_answer"):
        return {}                                         # already finalized cleanly
    return {"status": "incomplete",
            "final_answer": ("Could not fully complete the task within the "
                             "delegation budget. Partial results: "
                             + _summarize(state["results"]))}

graph = StateGraph(SupervisorState)
graph.add_node("supervise", supervise_node)
graph.add_node("delegate", delegate_node)
graph.add_node("finalize", supervisor_finalize)          # honest exit, like Ch3
graph.set_entry_point("supervise")
graph.add_conditional_edges("supervise", route,
                            {"delegate": "delegate", "finalize": "finalize"})
graph.add_edge("delegate", "supervise")                  # integrate -> decide again
graph.add_edge("finalize", END)
supervisor_app = graph.compile()
