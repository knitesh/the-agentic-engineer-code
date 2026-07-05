## agent/state.py  — the shared schema Chapters 4–12 import unchanged
import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages   # ID-based merge (NOT operator.add)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]      # merge semantics, deliberate
    goal: str
    required_keys: list[str]                      # acceptance criteria (3.10)
    known_facts: dict                             # populated as the agent learns
    iterations: int
    tokens_used: int                             # token budget tracking (3.13)
    started_at: float                            # time budget tracking (3.13)
    action_history: Annotated[list, operator.add]  # additive: deadlock detect (3.14)
    seen_signatures: set                         # repetition (union'd in act_node)
    stall_count: int                             # convergence tracking (3.12)
    fact_count_last_iter: int                    # convergence bookkeeping
    criteria_met_last_iter: int                  # convergence bookkeeping
    cancel_requested: bool                        # re-entry cancellation (3.15)
    pending_feedback: str | None                  # criteria-not-met feedback (3.10)
    exit_reason: str | None                       # WHY we stopped — recorded for traces
    final_answer: str | None
    status: str | None                            # "complete" | "incomplete"
