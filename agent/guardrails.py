## agent/guardrails.py
import re
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import AIMessage, SystemMessage
from agent.state import AgentState
from agent.config import CONFIG


## --- Repo glue: the check primitives the guardrail layer calls -------------
## Each is deliberately simple and deterministic; replace with your
## classifier/moderation service where stakes require it. All are FAIL-CLOSED
## at the call sites (an exception inside a check blocks the request).

_INJECTION_PATTERNS = [
    r"ignore (all|any|the|previous|prior) .{0,40}instructions",
    r"disregard (your|the) (system|previous) prompt",
    r"you are now\b", r"reveal (your|the) (system )?prompt",
    r"\bDAN\b", r"jailbreak",
]

def off_topic(request: str, scope: str) -> bool:
    """Cheap scope check (7.2). scope='general' accepts everything; a real
    deployment narrows this with rules or a small classifier."""
    return False if scope == "general" else False

def looks_like_injection(request: str) -> bool:
    lowered = request.lower()
    return any(re.search(p, lowered) for p in _INJECTION_PATTERNS)

def unsafe_content(output: str) -> bool:
    """Content-safety hook (7.4). Plug your moderation endpoint here."""
    return False

_PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "api_key": r"\b(?:sk|pk)-[A-Za-z0-9]{16,}\b",
}

def detect_pii_or_secrets(output: str, allowed_pii) -> list[str]:
    leaked = []
    for kind, pattern in _PII_PATTERNS.items():
        if kind in (allowed_pii or []):
            continue
        leaked.extend(re.findall(pattern, output))
    return leaked

def mask(output: str, leaked: list[str]) -> str:
    for item in leaked:
        output = output.replace(item, "[REDACTED]")
    return output

def validate_arguments(name: str, args: dict) -> str | None:
    """Per-tool argument rules (7.3). Return an error string, or None if valid."""
    if name in ("read_file", "write_file"):
        path = str(args.get("path", ""))
        if path.startswith("/") or ".." in path:
            return "path must be relative to the workspace, with no '..'"
    if name == "web_search" and len(str(args.get("query", ""))) > 512:
        return "query too long"
    return None

def over_action_budget(state, kind: str | None, config) -> bool:
    """Per-run action budget (7.3) — cousin of the Ch3 loop budgets."""
    history = state.get("action_history", [])
    return len(history) >= config.max_actions_per_run


## --- Tool-call authorization (7.3) -----------------------------------------

class CallVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class CallDecision:
    verdict: CallVerdict
    reason: str = ""


def authorize_tool_call(tool, args, state, config) -> CallDecision:
    """Run for EVERY tool call, after the model requests it, before act_node
    executes it. Deterministic, fail-closed, category-driven."""
    name = tool.name
    kind = tool.metadata.get("kind")                  # from Chapter 5

    # 1. Capability scope (ToolConfig, 3.6).
    if name in config.tools.denied_tools:
        return CallDecision(CallVerdict.BLOCK, f"'{name}' is denied in this context.")
    if name not in config.tools.enabled_tools:
        return CallDecision(CallVerdict.BLOCK, f"'{name}' is not enabled for this run.")

    # 2. Argument-level validation — finer than the input check.
    arg_error = validate_arguments(name, args)        # per-tool arg rules
    if arg_error:
        return CallDecision(CallVerdict.BLOCK, f"'{name}': {arg_error}")

    # 3. Per-run action budget (cousin of the Ch3 loop budgets).
    if over_action_budget(state, kind, config):
        return CallDecision(CallVerdict.BLOCK,
                            f"action budget for '{kind}' tools exceeded this run.")

    # 4. Approval gate for high-stakes categories — the field we pre-wired in Ch5.
    if kind in ("write", "communicate") and config.require_approval:
        return CallDecision(CallVerdict.REQUIRE_APPROVAL,
                            f"'{name}' is a {kind} action; a human must confirm.")

    return CallDecision(CallVerdict.ALLOW)


## --- INPUT (node at the front) -------------------------------------------
def input_guardrail_node(state: AgentState) -> dict:
    """Front-door checks, cheapest-first, FAIL-CLOSED. Writes a verdict the
    next edge reads; never lets a blocked request reach reasoning. If a check
    itself errors (classifier down), we block — a guardrail that fails open
    is not a guardrail."""
    request = state["goal"]
    try:
        if not request or len(request) > CONFIG.max_input_chars:
            return {"guard_block": "Request is empty or too long."}
        if off_topic(request, CONFIG.scope):
            return {"guard_block": "That's outside what I can help with here."}
        if looks_like_injection(request):            # screen, not full defense (Ch12)
            return {"guard_block": "Request flagged by a safety check."}
    except Exception:
        return {"guard_block": "Safety check unavailable; request blocked."}  # fail closed
    return {"guard_block": None}

def input_allowed(state: AgentState) -> str:
    return "block" if state.get("guard_block") else "allow"

## --- OUTPUT (node at the back) -------------------------------------------
def output_guardrail_node(state: AgentState) -> dict:
    """Back-door checks before the answer is returned. Runs BEFORE finalize_node,
    so final_answer is usually unset yet — we read the last message. (Once Ch3's
    finalize sets final_answer, the same guard could run after it on a re-pass;
    here, reading messages[-1] is the live value.) We don't implement
    groundedness checking in the project version — see 7.2 for why it's the
    least reliable check."""
    answer = state.get("final_answer") or state["messages"][-1].content
    try:
        if unsafe_content(answer):
            return {"final_answer": "I can't help with that.", "status": "blocked"}
        leaked = detect_pii_or_secrets(answer, CONFIG.allowed_pii)
        if leaked:
            return {"final_answer": mask(answer, leaked)}   # redact, don't block
    except Exception:
        return {"final_answer": "Response withheld (safety check failed).",
                "status": "blocked"}                        # fail closed
    return {}

## --- APPROVAL (Ch12 builds the full flow; the shape is here) --------------
def approval_node(state: AgentState) -> dict:
    """The graph pauses BEFORE this node (interrupt_before). On resume, the
    human's decision is supplied; here we just read it. Full UI/flow in Ch12."""
    return {}                                             # decision read from state/resume

def after_approval(state: AgentState) -> str:
    """Conditional edge after the human responds: approve -> execute, reject ->
    back to reason with the rejection recorded as an observation."""
    return "approved" if state.get("approval_granted") else "rejected"

## authorize_tool_call, CallVerdict, CallDecision: as defined in 7.3 (capability
## scope, arg validation, action budget, approval gate). Imported by graph.py
## for the routing edge and by guarded_act_node for the per-call wrapper.
