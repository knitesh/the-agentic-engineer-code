## agent/loop.py — evaluate_exit, detect_deadlock, finalize_node + helpers (Section 2)
import time
from dataclasses import dataclass


@dataclass
class ExitDecision:
    stop: bool
    reason: str
    feedback: str | None = None     # set when we push back into the loop

def model_says_done(state) -> bool:
    """The model signals 'done' by producing a final answer with no tool calls.
    This is the SOFT signal — necessary, never sufficient on its own."""
    last = state["messages"][-1]
    return not getattr(last, "tool_calls", None)


## The goal is decomposed (at setup) into the keys that must be satisfied.
## For "compare competitor A and B pricing", that might be these four.
REQUIRED_KEYS = ("a_pricing", "b_pricing", "a_positioning", "b_positioning")

def unmet_criteria(state) -> list[str]:
    """Which required pieces of the goal are still missing from state?"""
    found = state.get("known_facts", {})
    return [k for k in state.get("required_keys", REQUIRED_KEYS) if not found.get(k)]

def meets_acceptance_criteria(state) -> bool:
    """Cheap, deterministic acceptance: every required key is populated.
    No extra model call. This is the version I reach for first."""
    return len(unmet_criteria(state)) == 0

def count_criteria_met(state) -> int:
    required = state.get("required_keys", REQUIRED_KEYS)
    return len(required) - len(unmet_criteria(state))


def measure_progress(state) -> bool:
    """Return True if this iteration made progress. A stall is a divergence signal.
    'Progress' = more facts OR more acceptance criteria satisfied than last iteration."""
    facts_now = len(state.get("known_facts", {}))
    facts_before = state.get("fact_count_last_iter", 0)
    criteria_now = count_criteria_met(state)       # reuses 3.10's helper
    criteria_before = state.get("criteria_met_last_iter", 0)
    return (facts_now > facts_before) or (criteria_now > criteria_before)

def update_convergence_tracking(state) -> dict:
    """Run once per iteration (in the observe step). Returns a state delta."""
    if measure_progress(state):
        return {
            "stall_count": 0,
            "fact_count_last_iter": len(state.get("known_facts", {})),
            "criteria_met_last_iter": count_criteria_met(state),
        }
    return {"stall_count": state.get("stall_count", 0) + 1}


def detect_deadlock(state, config) -> str | None:
    """Return a deadlock type if detected, else None. A few list comparisons."""
    actions = state.get("action_history", [])     # list of (tool, frozenset(args))
    if len(actions) < 2:
        return None

    # 1. Identical repetition: last action equals the one before
    if actions[-1] == actions[-2]:
        return "identical_repetition"

    # 2. Cyclic oscillation: a short cycle repeating (A,B,A,B...)
    if len(actions) >= 4 and actions[-4:-2] == actions[-2:]:
        return "oscillation"

    # 3. No-progress churn: many distinct actions, no convergence (3.12)
    if state.get("stall_count", 0) >= config.stall_threshold:
        return "no_progress"

    return None


def evaluate_exit(state, config) -> ExitDecision:
    """All exit logic in one place, in code we control.
    The model's signal is ONE input among several, and the lowest-authority one."""

    # 1. HARD BUDGET GUARDS — checked first, cannot be argued with.
    elapsed = time.time() - state["started_at"]
    if state["iterations"] >= config.max_iterations:
        return ExitDecision(True, "iteration_budget")
    if state["tokens_used"] >= config.max_tokens_budget:
        return ExitDecision(True, "token_budget")
    if elapsed >= config.max_seconds:
        return ExitDecision(True, "time_budget")

    # 2. COOPERATIVE CANCELLATION (re-entry, 3.15)
    if state.get("cancel_requested"):
        return ExitDecision(True, "cancelled")

    # 3. STUCK DETECTION (3.14)
    deadlock = detect_deadlock(state, config)
    if deadlock:
        return ExitDecision(True, f"deadlock:{deadlock}")

    # 4. MODEL SIGNAL — but VERIFIED, not trusted blindly.
    if model_says_done(state):
        if meets_acceptance_criteria(state):
            return ExitDecision(True, "goal_achieved")
        else:
            # Model thinks it's done but criteria say otherwise:
            # push back into the loop with the gap made explicit.
            return ExitDecision(
                False, "criteria_not_met",
                feedback="Goal not yet satisfied. Missing: "
                         + ", ".join(unmet_criteria(state)))

    # 5. DEFAULT: keep going
    return ExitDecision(False, "continue")


def extract_answer(state) -> str:
    """The model's last textual content — used when the goal was genuinely achieved."""
    return state["messages"][-1].content

def summarize_partial_progress(state) -> str:
    """An HONEST partial result: what we found, what we didn't reach.
    Cheap, deterministic version — list what's known and what's still missing."""
    known = state.get("known_facts", {})
    found = "; ".join(f"{k}: {v}" for k, v in known.items()) or "nothing conclusive"
    missing = ", ".join(unmet_criteria(state)) or "none"
    return (f"I ran out of budget before fully completing the goal. "
            f"Here is what I found: {found}. Still unresolved: {missing}.")

def finalize_node(state):
    reason = state.get("exit_reason", "")
    if reason == "goal_achieved":
        return {"final_answer": extract_answer(state), "status": "complete"}
    # budget, deadlock, cancellation: report honestly, NEVER fabricate completion
    return {
        "final_answer": summarize_partial_progress(state),
        "status": "incomplete",
        "exit_reason": reason,
    }
