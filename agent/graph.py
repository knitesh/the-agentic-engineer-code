## agent/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, SystemMessage
from agent.state import AgentState
from agent.config import HarnessConfig
from agent.tools import TOOLS, TOOLS_BY_NAME
from agent.prompt import SYSTEM_PROMPT
from agent.loop import (
    evaluate_exit,
    detect_deadlock,
    finalize_node as loop_finalize_node,
    update_convergence_tracking,
)  # from Ch3


def format_tool_result(call, result) -> str:
    """Repo helper: normalize a tool result into observation text (3.14)."""
    return str(result)

## Repo wiring (implied by the chapter text): the shared model client, bound to
## the tool suite, and the module-level CONFIG this graph closes over.
from langchain_openai import ChatOpenAI    # swap for your provider

CONFIG = HarnessConfig()
llm = ChatOpenAI(model=CONFIG.model.name, temperature=CONFIG.model.temperature)
llm_with_tools = llm.bind_tools(TOOLS)

REASON_SYSTEM_PROMPT = """You are a general-purpose assistant.
Think briefly about what you need before each action, then either call
a tool or give your final answer. Call a tool only when you need
information or computation you don't already have. When you have enough
to answer, respond directly with no tool call."""

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
    for call in getattr(last, "tool_calls", None) or []:
        signature = (call["name"], frozenset(call["args"].items()))
        history.append(signature)
        if signature in seen:
            # Same guard as Ch3 3.14: feed the fact of repetition back into
            # perception so the next reason step can change approach.
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
        return "finalize"                      # model declined a tool, and
                                               # evaluate_exit verified it's done
    return "act"

## Repo glue: LangGraph routers receive a state snapshot — writes made inside
## `route` don't persist to channels. Shadow-wrap Ch3's finalize_node so the
## exit decision is recorded in-node before it reads exit_reason (3.16).
def finalize_node(state):
    if not state.get("exit_reason"):
        decision = evaluate_exit(state, CONFIG)
        if decision.stop:
            state = {**state, "exit_reason": decision.reason}
    update = loop_finalize_node(state)
    exit_reason = state.get("exit_reason")
    if "exit_reason" not in update and isinstance(exit_reason, str):
        update["exit_reason"] = exit_reason
    return update

graph = StateGraph(AgentState)
graph.add_node("reason", reason_node)
graph.add_node("act", act_node)
graph.add_node("finalize", finalize_node)      # from Ch3 — honest partial output
graph.set_entry_point("reason")
graph.add_conditional_edges("reason", route,
                            {"act": "act",
                             "finalize": "finalize",
                             "reason_with_feedback": "reason"})
graph.add_edge("act", "reason")
graph.add_edge("finalize", END)
app = graph.compile()


def build_graph(config):
    """Factory: build and compile the agent graph from a config.
    The harness calls build_graph(config) and wraps the result."""

    llm = ChatOpenAI(model=config.model.name, temperature=config.model.temperature)
    llm_with_tools = llm.bind_tools(TOOLS)

    def reason_node(state):
        # criteria-not-met push-back (3.10): surface the gap before re-reasoning
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        if state.get("pending_feedback"):
            messages.append(SystemMessage(content=state["pending_feedback"]))
        response = llm_with_tools.invoke(messages)
        usage = response.response_metadata.get("token_usage", {})
        return {
            "messages": [response],                                  # add_messages reducer
            "iterations": state["iterations"] + 1,
            "tokens_used": state["tokens_used"] + usage.get("total_tokens", 0),
            "pending_feedback": None,
        }

    def act_node(state):
        observations = []
        seen = state.get("seen_signatures", set())     # read existing accumulator
        new_signatures = set()
        new_history = []
        last = state["messages"][-1]
        for call in getattr(last, "tool_calls", None) or []:
            sig = (call["name"], frozenset(call["args"].items()))
            new_history.append(sig)
            if sig in seen:
                observations.append(ToolMessage(
                    tool_call_id=call["id"],
                    content=(f"NOTE: you already ran {call['name']} with these exact "
                             f"arguments and it did not advance the goal. Do NOT repeat it. "
                             f"Try a materially different approach or report what's blocking you."),
                ))
                continue
            result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            new_signatures.add(sig)
            observations.append(ToolMessage(tool_call_id=call["id"],
                                            content=format_tool_result(call, result)))
        return {
            "messages": observations,                       # add_messages merges by id
            "seen_signatures": seen | new_signatures,        # read-then-return the UNION
            "action_history": new_history,                   # additive reducer appends
            # convergence tracking runs once per iteration, in the observe step (3.12)
            **update_convergence_tracking(state),
        }

    def route(state) -> str:
        if state.get("cancel_requested"):
            state["exit_reason"] = "cancelled"
            return "finalize"        # clean exit on cancellation (3.15)
        decision = evaluate_exit(state, config)      # all the logic in one call (3.10)
        if decision.stop:
            # record WHY in state so finalize and the trace can read it
            state["exit_reason"] = decision.reason
            return "finalize"                        # a node that writes the final answer
        if decision.feedback:
            # model thought it was done but criteria failed: inject the gap, re-reason
            state["pending_feedback"] = decision.feedback
            return "reason_with_feedback"
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "act"
        return "reason"

    def record_exit_and_finalize(state):
        """Repo glue: LangGraph routers receive a state snapshot — writes made
        inside `route` don't persist to channels. Recompute the exit decision
        here, in a node, so finalize_node (3.16) sees WHY we stopped."""
        if not state.get("exit_reason"):
            decision = evaluate_exit(state, config)
            if decision.stop:
                state = {**state, "exit_reason": decision.reason}
        update = loop_finalize_node(state)
        exit_reason = state.get("exit_reason")
        if "exit_reason" not in update and isinstance(exit_reason, str):
            update["exit_reason"] = exit_reason
        return update

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node("finalize", record_exit_and_finalize)   # honest exits (3.16)
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", route,
                                {"act": "act",
                                 "finalize": "finalize",
                                 "reason_with_feedback": "reason",
                                 "reason": "reason"})
    graph.add_edge("act", "reason")                  # observe -> back to reason
    graph.add_edge("finalize", END)
    return graph.compile()
