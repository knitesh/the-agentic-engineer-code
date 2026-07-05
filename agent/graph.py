## agent/graph.py — guardrails added to the Ch6 graph; loop shape unchanged.
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage, RemoveMessage
from agent.state import AgentState
from agent.loop import evaluate_exit, finalize_node       # Ch3, unchanged
from agent.guardrails import (                            # this chapter
    input_guardrail_node, input_allowed, output_guardrail_node,
    authorize_tool_call, CallVerdict, approval_node, after_approval,
)

## route_with_compression, recall_node, reason_node, compress_node live in THIS
## file (from Ch6's 6.7); CONFIG and TOOLS_BY_NAME are cross-module imports.
from agent.config import CONFIG
from agent.tools import TOOLS_BY_NAME
from agent.tools import TOOLS

## Repo wiring: the shared model client, bound to the tool suite.
from langchain_openai import ChatOpenAI    # swap for your provider

llm = ChatOpenAI(model=CONFIG.model.name, temperature=CONFIG.model.temperature)
llm_with_tools = llm.bind_tools(TOOLS)

REASON_SYSTEM_PROMPT = """You are a general-purpose assistant.
Think briefly about what you need before each action, then either call
a tool or give your final answer. Call a tool only when you need
information or computation you don't already have. When you have enough
to answer, respond directly with no tool call."""

## --- Ch4 nodes, unchanged (4.5) -------------------------------------------

def reason_node(state: AgentState) -> dict:
    """PERCEIVE + REASON. Tracks the token budget Ch3's exit logic reads."""
    response = llm_with_tools.invoke(
        [SystemMessage(content=REASON_SYSTEM_PROMPT)] + state["messages"]
    )
    usage = response.response_metadata.get("token_usage", {})
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
        "tokens_used": state["tokens_used"] + usage.get("total_tokens", 0),
    }

def route(state: AgentState) -> str:
    """Exit condition — the SAME evaluate_exit from Ch3, not a simplified one."""
    decision = evaluate_exit(state, CONFIG)    # all budgets + deadlock + verify
    if decision.stop:
        return "finalize"
    if getattr(decision, "feedback", None):
        return "reason_with_feedback"          # criteria-not-met push-back (Ch3)
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "finalize"
    return "act"

## --- Ch6: context management + recall (6.4, 6.7) ----------------------------
## Repo glue: cheap token estimate + a guided, lossy summary.

def count_tokens(messages) -> int:
    return sum(len(str(getattr(m, "content", m))) for m in messages) // 4

def needs_compression(state, config) -> bool:
    return count_tokens(state["messages"]) > config.context_token_budget

def summarize_preserving(messages, must_keep=("decisions", "commitments",
                                              "key facts", "open questions")) -> str:
    """Lossy, but guided (6.4): summarize old context, preserving what matters."""
    text = "\n".join(str(getattr(m, "content", m)) for m in messages)
    try:
        response = llm.invoke(
            f"Summarize this conversation excerpt in under 200 words. "
            f"You MUST preserve: {', '.join(must_keep)}.\n\n{text[:20_000]}"
        )
        return response.content
    except Exception:
        return text[:2_000] + " …[truncated summary — summarizer unavailable]"

def recall_node(state: AgentState) -> dict:
    """Retrieve long-term memories relevant to the goal and inject them
    as context BEFORE reasoning. Namespace-scoped, thresholded (6.2)."""
    memories = memory.recall(state["goal"], k=5, min_score=0.75)
    if not memories:
        return {}                                   # nothing relevant; no-op
    recalled = "\n".join(f"- {m.text}" for m in memories)
    return {"messages": [SystemMessage(
        content=f"[Relevant memories about this user]\n{recalled}"
    )]}

def compress_node(state: AgentState) -> dict:
    """Run when context exceeds budget: archive + summarize the old middle."""
    msgs = state["messages"]
    if count_tokens(msgs) <= CONFIG.context_token_budget:
        return {}                                   # no-op under budget

    keep_recent = msgs[-CONFIG.keep_recent:]
    old = msgs[2:-CONFIG.keep_recent]               # preserve system+goal at [0:2]

    memory.archive_messages(old)                    # to pgvector (6.2), retrievable
    digest = summarize_preserving(old)

    removals = [RemoveMessage(id=m.id) for m in old]
    return {"messages": removals + [SystemMessage(content=f"[Earlier context] {digest}")]}

def route_with_compression(state: AgentState) -> str:
    """Same exit logic as Ch4's route, plus a context-budget check (6.7).
    Order matters: a STOP decision always wins."""
    decision = route(state)                    # Ch4 route: act/finalize/feedback
    if decision == "finalize":                 # evaluate_exit said stop — honor it
        return "finalize"
    if needs_compression(state, CONFIG):       # continuing, but context too big?
        return "compress"
    return decision                            # act, or reason_with_feedback

## --- Ch7: the guarded act node (7.8) ----------------------------------------

def guarded_act_node(state: AgentState) -> dict:
    """act_node (4.5/5.6) + per-call guardrail. Reaches here only when the turn
    has no pending approval (guarded_route diverts approval turns first). Blocked
    calls become observations; the Ch4 deadlock bookkeeping is preserved."""
    last = state["messages"][-1]
    results = []
    seen = set(state.get("seen_signatures", set()))       # Ch4 union-write
    history = list(state.get("action_history", []))       # Ch4 accumulation
    for call in last.tool_calls:
        signature = (call["name"], frozenset(call["args"].items()))
        history.append(signature)                         # record EVERY attempt...
        tool = TOOLS_BY_NAME[call["name"]]
        verdict = authorize_tool_call(tool, call["args"], state, CONFIG)
        if verdict.verdict == CallVerdict.BLOCK:
            # Failure-as-observation (Ch5): the agent SEES the denial and adapts.
            # We still added the signature above, so repeated blocked retries
            # trip the Ch4 deadlock guard instead of looping forever.
            seen.add(signature)
            results.append(ToolMessage(
                content=f"BLOCKED: {verdict.reason}", tool_call_id=call["id"]))
            continue
        if signature in seen:                             # Ch4 (3.14) repetition guard
            results.append(ToolMessage(
                content=(f"NOTE: you already ran {call['name']} with these exact "
                         f"arguments. Try a materially different approach."),
                tool_call_id=call["id"]))
            continue
        try:
            output = tool.invoke(call["args"])
        except Exception as e:
            output = f"ERROR running {call['name']}: {e}"  # Ch5 failure-as-observation
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        seen.add(signature)
    return {"messages": results, "seen_signatures": seen, "action_history": history}

## --- Ch7: the guarded routing edge (7.8) -------------------------------------

def guarded_route(state: AgentState) -> str:
    """Edge condition: exit logic, then turn-level tool-call authorization.

    Approval is decided at the TURN level: if ANY call in this turn needs
    human sign-off, the whole turn diverts to `approval` and pauses. So we
    must scan EVERY call and pick the most restrictive verdict — never return
    on the first one, or a later require_approval call slips through to act."""
    decision = route_with_compression(state)              # Ch6 route (stop wins)
    if decision != "act":
        return decision                                   # finalize / compress / feedback

    last = state["messages"][-1]
    verdicts = [
        authorize_tool_call(TOOLS_BY_NAME[c["name"]], c["args"], state, CONFIG)
        for c in last.tool_calls
    ]
    # require_approval > block > allow. Only require_approval reroutes the turn;
    # blocks are turned into observations inside guarded_act_node, so they still
    # go to "act". Scanning ALL calls is the fix for the first-verdict bug.
    if any(v.verdict == CallVerdict.REQUIRE_APPROVAL for v in verdicts):
        return "approval"                                 # pause for a human (Ch12)
    return "act"

## Repo glue: LangGraph routers receive a state snapshot — writes made inside
## `route` don't persist to channels. Shadow-wrap Ch3's finalize_node so the
## exit decision is recorded in-node before it reads exit_reason (3.16).
_finalize_node_impl = finalize_node

def finalize_node(state):
    if state.get("guard_block"):
        return {"final_answer": state["guard_block"], "status": "blocked",
                "exit_reason": "input_blocked"}
    if not state.get("exit_reason"):
        decision = evaluate_exit(state, CONFIG)
        if decision.stop:
            state = {**state, "exit_reason": decision.reason}
    update = _finalize_node_impl(state)
    exit_reason = state.get("exit_reason")
    if "exit_reason" not in update and isinstance(exit_reason, str):
        update["exit_reason"] = exit_reason
    return update

graph = StateGraph(AgentState)
graph.add_node("input_guard", input_guardrail_node)       # NODE: front door
graph.add_node("recall", recall_node)                     # Ch6
graph.add_node("reason", reason_node)                     # Ch4
graph.add_node("act", guarded_act_node)                   # WRAPPER: per-call check
graph.add_node("approval", approval_node)                 # Ch12 human gate
graph.add_node("compress", compress_node)                 # Ch6
graph.add_node("output_guard", output_guardrail_node)     # NODE: back door
graph.add_node("finalize", finalize_node)                 # Ch3

graph.set_entry_point("input_guard")                      # guard BEFORE anything
graph.add_conditional_edges("input_guard", input_allowed,
                            {"allow": "recall", "block": "finalize"})
graph.add_edge("recall", "reason")
graph.add_conditional_edges("reason", guarded_route,      # EDGE: tool-call auth
                            {"act": "act", "approval": "approval",
                             "compress": "compress",
                             "reason_with_feedback": "reason",
                             "finalize": "output_guard"})  # exit via output guard
graph.add_edge("act", "reason")
## Approval is conditional, NOT a static edge: a human can approve OR reject.
## Approve -> execute the turn (act). Reject -> back to reason with a denial
## observation, so the agent adapts instead of executing the rejected call.
graph.add_conditional_edges("approval", after_approval,
                            {"approved": "act", "rejected": "reason"})
graph.add_edge("compress", "reason")
graph.add_edge("output_guard", "finalize")               # guard BEFORE returning
graph.add_edge("finalize", END)

from agent.memory_backend import checkpointer, memory     # Ch6
app = graph.compile(checkpointer=checkpointer,
                    interrupt_before=["approval"])         # pause at the human gate


def build_graph(config):
    """Kept for the harness/run entry point: the compiled guarded agent above."""
    return app
