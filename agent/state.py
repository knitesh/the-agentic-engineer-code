## agent/state.py
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages   # ID-based message merge

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # short-term memory; reducer merges by ID
    goal: str                                 # the original objective, kept stable
    iterations: int                           # how many reasoning steps we've taken
    final_answer: str | None                  # set when the agent produces an answer
