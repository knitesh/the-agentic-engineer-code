# agent/run.py
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import build_graph

load_dotenv()


@dataclass
class MiniConfig:                # Chapter 3 replaces this with HarnessConfig
    model_name: str = "gpt-4o"
    temperature: float = 0.0


def main():
    app = build_graph(MiniConfig())
    goal = "What is 17% of 4,200, then add 90?"
    initial_state = {
        "messages": [HumanMessage(content=goal)],
        "goal": goal,
        "iterations": 0,
        "final_answer": None,
    }
    final_state = app.invoke(initial_state)
    print(final_state["final_answer"])


if __name__ == "__main__":
    main()
