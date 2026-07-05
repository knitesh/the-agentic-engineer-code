# agent/run.py
from dotenv import load_dotenv

from agent.config import HarnessConfig, ModelConfig
from agent.graph import build_graph
from agent.harness import AgentHarness
from agent.sinks import Sinks

load_dotenv()


def main():
    config = HarnessConfig(model=ModelConfig(name="gpt-4o", temperature=0.0))
    harness = AgentHarness(build_graph(config), config, sinks=Sinks())

    # single-shot
    print(harness.run_single("what is 17% of 4,200, then add 90?"))

    # interactive (same agent, threaded session)
    print(harness.run_interactive("sess-1", "and what's that times 3?"))


if __name__ == "__main__":
    main()
