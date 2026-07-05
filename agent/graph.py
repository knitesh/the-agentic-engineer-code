## agent/graph.py — refactor Ch7's wiring into a config-parameterized factory.
## Node bodies are UNCHANGED from 7.8; they now close over `config` instead of
## a module-level CONFIG, so per-role tool scoping actually takes effect.
from functools import lru_cache
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, SystemMessage, RemoveMessage
from agent.state import AgentState
from agent.loop import evaluate_exit, finalize_node       # Ch3, unchanged
from agent.guardrails import (                            # Ch7
    input_guardrail_node, input_allowed, output_guardrail_node,
    authorize_tool_call, CallVerdict, approval_node, after_approval,
)
from agent.config import CONFIG, llm, build_subagent_config
from agent.tools import TOOLS, TOOLS_BY_NAME
from agent.memory_backend import checkpointer, memory      # Ch6 shared stores

if TYPE_CHECKING:
    from agent.config import HarnessConfig

REASON_SYSTEM_PROMPT = """You are a general-purpose assistant.
Think briefly about what you need before each action, then either call
a tool or give your final answer. Call a tool only when you need
information or computation you don't already have. When you have enough
to answer, respond directly with no tool call."""

## --- Context-management helpers (6.4) — config-independent ------------------

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
        content = response.content
        return content if isinstance(content, str) else str(content)
    except Exception:
        return text[:2_000] + " …[truncated summary — summarizer unavailable]"


def build_guarded_agent(config: "HarnessConfig"):
    """Compile a guarded ReAct agent (Ch7) bound to THIS config."""
    # Tool scoping actually takes effect: the model only sees enabled tools.
    scoped_tools = [t for t in TOOLS if t.name in config.tools.enabled_tools]
    llm_with_tools = llm.bind_tools(scoped_tools)

    def reason_node(state: AgentState) -> dict:              # 7.8 body, reads `config`
        response = llm_with_tools.invoke(
            [SystemMessage(content=REASON_SYSTEM_PROMPT)] + state["messages"]
        )
        usage = response.response_metadata.get("token_usage", {})
        update = {
            "messages": [response],
            "iterations": state["iterations"] + 1,
            "tokens_used": state["tokens_used"] + usage.get("total_tokens", 0),
        }
        # Ch12: declare the proposed action so the approval gate (12.3) can show
        # the human exactly what it is about to approve.
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            update["pending_action"] = {"tool": call["name"], "args": call["args"]}
        return update

    def route(state: AgentState) -> str:
        decision = evaluate_exit(state, config)    # all budgets + deadlock + verify
        if decision.stop:
            return "finalize"
        if getattr(decision, "feedback", None):
            return "reason_with_feedback"          # criteria-not-met push-back (Ch3)
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "finalize"
        return "act"

    def recall_node(state: AgentState) -> dict:              # Ch6, unchanged
        memories = memory.recall(state["goal"], k=5, min_score=0.75)
        if not memories:
            return {}
        recalled = "\n".join(f"- {m.text}" for m in memories)
        return {"messages": [SystemMessage(
            content=f"[Relevant memories about this user]\n{recalled}"
        )]}

    def compress_node(state: AgentState) -> dict:            # Ch6, unchanged
        msgs = state["messages"]
        if count_tokens(msgs) <= config.context_token_budget:
            return {}
        old = msgs[2:-config.keep_recent]           # preserve system+goal at [0:2]
        memory.archive_messages(old)
        digest = summarize_preserving(old)
        removals = [RemoveMessage(id=m.id) for m in old]
        return {"messages": removals + [SystemMessage(content=f"[Earlier context] {digest}")]}

    def route_with_compression(state: AgentState) -> str:    # Ch6, unchanged
        decision = route(state)
        if decision == "finalize":
            return "finalize"
        if needs_compression(state, config):
            return "compress"
        return decision

    def guarded_act_node(state: AgentState) -> dict:         # 7.8 body, but:
        last = state["messages"][-1]                         #   authorize_tool_call(tool, a, state, config)
        results = []
        seen = set(state.get("seen_signatures", set()))
        history = list(state.get("action_history", []))
        for call in getattr(last, "tool_calls", None) or []:
            signature = (call["name"], frozenset(call["args"].items()))
            history.append(signature)
            tool = TOOLS_BY_NAME[call["name"]]
            verdict = authorize_tool_call(tool, call["args"], state, config)
            if verdict.verdict == CallVerdict.BLOCK:
                seen.add(signature)
                results.append(ToolMessage(
                    content=f"BLOCKED: {verdict.reason}", tool_call_id=call["id"]))
                continue
            if signature in seen:
                results.append(ToolMessage(
                    content=(f"NOTE: you already ran {call['name']} with these exact "
                             f"arguments. Try a materially different approach."),
                    tool_call_id=call["id"]))
                continue
            try:
                output = tool.invoke(call["args"])
            except Exception as e:
                output = f"ERROR running {call['name']}: {e}"
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
            seen.add(signature)
        return {"messages": results, "seen_signatures": seen, "action_history": history}

    def guarded_route(state: AgentState) -> str:             # 7.8 body, reads `config`
        decision = route_with_compression(state)
        if decision != "act":
            return decision
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        verdicts = [
            authorize_tool_call(TOOLS_BY_NAME[c["name"]], c["args"], state, config)
            for c in tool_calls
        ]
        if any(v.verdict == CallVerdict.REQUIRE_APPROVAL for v in verdicts):
            return "approval"
        return "act"

    def record_exit_and_finalize(state):
        """Repo glue: record WHY we stopped before finalize_node reads it."""
        if state.get("guard_block"):
            return {"final_answer": state["guard_block"], "status": "blocked",
                    "exit_reason": "input_blocked"}
        if not state.get("exit_reason"):
            decision = evaluate_exit(state, config)
            if decision.stop:
                state = {**state, "exit_reason": decision.reason}
        update = finalize_node(state)
        exit_reason = state.get("exit_reason")
        if "exit_reason" not in update and isinstance(exit_reason, str):
            update["exit_reason"] = exit_reason
        return update

    # ... input/output guard nodes, approval node, edges — all exactly as 7.8 ...
    graph = StateGraph(AgentState)
    graph.add_node("input_guard", input_guardrail_node)
    graph.add_node("recall", recall_node)
    graph.add_node("reason", reason_node)
    graph.add_node("act", guarded_act_node)
    graph.add_node("approval", approval_node)
    graph.add_node("compress", compress_node)
    graph.add_node("output_guard", output_guardrail_node)
    graph.add_node("finalize", record_exit_and_finalize)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges("input_guard", input_allowed,
                                {"allow": "recall", "block": "finalize"})
    graph.add_edge("recall", "reason")
    graph.add_conditional_edges("reason", guarded_route,
                                {"act": "act", "approval": "approval",
                                 "compress": "compress",
                                 "reason_with_feedback": "reason",
                                 "finalize": "output_guard"})
    graph.add_edge("act", "reason")
    graph.add_conditional_edges("approval", after_approval,
                                {"approved": "act", "rejected": "reason"})
    graph.add_edge("compress", "reason")
    graph.add_edge("output_guard", "finalize")
    graph.add_edge("finalize", END)

    # Ch12: the approval node now pauses ITSELF via interrupt() (12.3), so the
    # static interrupt_before from Ch7 is no longer needed — the node runs,
    # halts inside interrupt(), and resumes on Command(resume=...).
    return graph.compile(checkpointer=checkpointer)      # Ch6 shared checkpointer


@lru_cache(maxsize=None)
def worker_for(role: str):
    """One compiled, scoped worker per role (cached so we compile once each)."""
    return build_guarded_agent(build_subagent_config(role))


## The default guarded agent — same compiled app prior chapters exposed.
app = build_guarded_agent(CONFIG)


def build_graph(config):
    """Kept for the harness/run entry point: compile for the given config."""
    return build_guarded_agent(config)
