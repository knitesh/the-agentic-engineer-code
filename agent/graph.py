## agent/graph.py — build_graph(config): reason/act nodes + conditional edge,
## now wired to the Chapter 3 loop (evaluate_exit, finalize_node).
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI    # swap for your provider

from agent.state import AgentState
from agent.tools import TOOLS
from agent.prompt import SYSTEM_PROMPT
from agent.loop import evaluate_exit, finalize_node, update_convergence_tracking


def format_tool_result(call, result) -> str:
    """Repo helper: normalize a tool result into observation text (3.14)."""
    return str(result)


def build_graph(config):
    """Factory: build and compile the agent graph from a config.
    The harness calls build_graph(config) and wraps the result."""

    llm = ChatOpenAI(model=config.model.name, temperature=config.model.temperature)
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

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
        for call in state["messages"][-1].tool_calls:
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
            result = TOOLS[call["name"]](**call["args"])
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
        update = finalize_node(state)
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
