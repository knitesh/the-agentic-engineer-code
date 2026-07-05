## agent/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, SystemMessage, RemoveMessage
from agent.state import AgentState
from agent.config import CONFIG
from agent.tools import TOOLS, TOOLS_BY_NAME
from agent.loop import evaluate_exit, finalize_node    # Ch3, unchanged
## reason_node, act_node, route: from Ch4 (4.5), unchanged
## compress_node: from 6.4

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

def act_node(state: AgentState) -> dict:
    """ACT + OBSERVE — using the Ch3 seen_signatures deadlock guard (3.14)."""
    last = state["messages"][-1]
    results = []
    seen = set(state.get("seen_signatures", set()))
    history = list(state.get("action_history", []))
    for call in last.tool_calls:                         # dict access (LangChain)
        signature = (call["name"], frozenset(call["args"].items()))
        history.append(signature)
        if signature in seen:
            results.append(ToolMessage(
                content=(f"NOTE: you already ran {call['name']} with these exact "
                         f"arguments and it did not advance the goal. Do NOT "
                         f"repeat it — try a materially different approach."),
                tool_call_id=call["id"],
            ))
            continue
        try:
            output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        except Exception as e:
            output = f"ERROR running {call['name']}: {e}"
        results.append(ToolMessage(content=str(output),
                                   tool_call_id=call["id"]))
        seen.add(signature)
    return {"messages": results,
            "seen_signatures": seen,           # union write — the Ch3 footgun fix
            "action_history": history}

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

## --- Ch6: context management helpers (6.4) ---------------------------------
## Repo glue: cheap token estimate + a guided, lossy summary. Swap count_tokens
## for your provider's tokenizer if you need exact counts.

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

## --- Ch6: recall before reasoning (6.7) ------------------------------------
## agent/graph.py — a recall step before the first reason step.
from langchain_core.messages import SystemMessage

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

## --- Ch6: compression node (6.4) -------------------------------------------
from langchain_core.messages import RemoveMessage, SystemMessage

def compress_node(state: AgentState) -> dict:
    """Run when context exceeds budget: archive + summarize the old middle."""
    msgs = state["messages"]
    if count_tokens(msgs) <= CONFIG.context_token_budget:
        return {}                                   # no-op under budget

    keep_recent = msgs[-CONFIG.keep_recent:]
    old = msgs[2:-CONFIG.keep_recent]               # preserve system+goal at [0:2]

    memory.archive_messages(old)                    # to pgvector (6.2), retrievable
    digest = summarize_preserving(old)

    # Remove the old messages (by id) and inject the digest in their place.
    removals = [RemoveMessage(id=m.id) for m in old]
    return {"messages": removals + [SystemMessage(content=f"[Earlier context] {digest}")]}

## --- Ch6: routing with compression (6.7) ------------------------------------

def route_with_compression(state: AgentState) -> str:
    """Same exit logic as Ch4's route, plus a context-budget check that
    diverts to compression instead of growing the window unbounded.

    Order matters: a STOP decision always wins. We let the Ch4 route run
    first so that an over-budget/deadlocked run finalizes; only if the run
    is continuing do we consider compressing. This guarantees an over-budget
    AND over-context run exits cleanly rather than compressing forever."""
    decision = route(state)                    # Ch4 route: act/finalize/feedback
    if decision == "finalize":                 # evaluate_exit said stop — honor it
        return "finalize"
    if needs_compression(state, CONFIG):       # continuing, but context too big?
        return "compress"
    return decision                            # act, or reason_with_feedback

## Repo glue: LangGraph routers receive a state snapshot — writes made inside
## `route` don't persist to channels. Shadow-wrap Ch3's finalize_node so the
## exit decision is recorded in-node before it reads exit_reason (3.16).
_finalize_node_impl = finalize_node

def finalize_node(state):
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
graph.add_node("recall", recall_node)          # NEW: retrieve before reasoning
graph.add_node("reason", reason_node)          # Ch4, unchanged
graph.add_node("act", act_node)                # Ch5 tools, Ch4 node, unchanged
graph.add_node("compress", compress_node)      # NEW: bound the context (6.4)
graph.add_node("finalize", finalize_node)      # Ch3, unchanged

graph.set_entry_point("recall")                # recall -> reason at the start
graph.add_edge("recall", "reason")
graph.add_conditional_edges("reason", route_with_compression,
                            {"act": "act",
                             "compress": "compress",
                             "finalize": "finalize",
                             "reason_with_feedback": "reason"})
graph.add_edge("act", "reason")
graph.add_edge("compress", "reason")           # after compressing, keep reasoning
graph.add_edge("finalize", END)

## The shared stores, imported (not rebuilt) — see agent/memory_backend.py.
## Same Postgres backs both the checkpointer and long-term memory (6.6).
from agent.memory_backend import checkpointer, memory
app = graph.compile(checkpointer=checkpointer)


def build_graph(config):
    """Kept for the harness/run entry point: the compiled agent above."""
    return app
