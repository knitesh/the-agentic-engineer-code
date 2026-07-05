## agent/approval_policy.py — the operator-level (most-trusted) gate policy.
## Lives in code, not the prompt, so no user input or injection can override it.
HIGH_STAKES_TOOLS = {"send_email", "transfer_funds", "delete_record",
                     "deploy", "post_message"}

def needs_approval(tool: str, args: dict, state: dict, config) -> bool:
    if not config.require_approval:          # Ch7 flag: gating is configurable
        return False
    if tool in HIGH_STAKES_TOOLS:            # categorically high blast radius
        return True
    # Graduated: small money flows freely, large money stops for a human.
    if tool == "make_payment" and args.get("amount", 0) > config.approval_threshold:
        return True
    # Low confidence on a write action → gate even if normally autonomous.
    if args.get("_write") and state.get("low_confidence"):
        return True
    return False


## In the routing after `reason` (extends Ch7's guarded_route).
from langgraph.types import Command
from agent.config import CONFIG

def route_after_reason(state):
    pending = state.get("pending_action")
    if pending and needs_approval(pending["tool"], pending["args"], state, CONFIG):
        return "approval"          # pause for a human (interrupt) before acting
    return "act"                   # safe to execute autonomously


def present_to_human(request: dict) -> dict:
    """Repo glue: the simplest possible approval surface — the terminal.
    In production this is your UI / queue / Slack integration (12.4)."""
    print("\n=== APPROVAL REQUIRED ===")
    for key, value in request.items():
        print(f"  {key}: {value}")
    answer = input("Approve? [y/N] ").strip().lower()
    if answer == "y":
        return {"approved": True}
    return {"approved": False, "reason": "rejected by human at the console"}


## The caller handles the pause/resume around the harness (Ch11 checkpointer).
def run_with_approval(harness, message, thread_id):
    result = harness.run_safe(message, thread_id=thread_id)
    while result.get("status") == "interrupted":         # gate fired
        request = result["interrupt_payload"]            # what needs approval
        decision = present_to_human(request)             # your UI / queue / Slack
        # Resume the SAME run from the checkpoint with the human's decision.
        result = harness.resume(thread_id, Command(resume=decision))
    return result
