## evals/run_experiment.py — the experiment runner (10.2, 10.7).
## The "task" under test is just our harness's run_safe — the same entry point
## production uses — so evals exercise the REAL system, not a look-alike.
## Run: python -m evals.run_experiment
from langfuse import get_client, Evaluation

from agent.config import CONFIG
from agent.graph import app
from agent.harness import AgentHarness
from agent.sinks import Sinks
from evals.evaluators import (
    contains_expected, meets_acceptance_criteria, trajectory_check,
    llm_judge, refused_safely,
)

langfuse = get_client()

## Repo glue: the harness under test — the same wiring run.py uses.
harness = AgentHarness(app, CONFIG, Sinks())


## The TASK under test: run the real agent on one dataset item.
## `item.input` is the user request. We return BOTH the answer AND the final
## state — because 10.1 told us the PATH matters, and the path lives in state
## (action_history, known_facts, exit_reason). An evaluator's `metadata` arg is
## the DATASET ITEM's metadata, NOT the run's state, so the only way to give
## evaluators the trajectory is to return it as part of the task `output`.
def agent_task(*, item, **kwargs):
    # run_safe returns a summary; for eval we use return_state=True (a thin,
    # declared flag) so the harness also surfaces the final graph state.
    result = harness.run_safe(item.input, thread_id=f"eval-{item.id}",
                              return_state=True)
    return {
        "answer": result.get("final_answer") or "",
        "state": result.get("final_state", {}),   # action_history, known_facts...
    }


## Run the experiment over a Langfuse-hosted dataset.
def nightly_regression():
    dataset = langfuse.get_dataset("agent-regression-v1")
    result = dataset.run_experiment(
        name="nightly-regression",
        description="Full agent over the curated regression set",
        task=agent_task,
        evaluators=[contains_expected],
        max_concurrency=5,      # bound concurrent agent runs
    )
    print(result.format())      # human-readable summary; scores land in Langfuse
    return result


## Layered evaluators: deterministic (free, reliable) → judge (cheap, fallible).
## Human review (10.5) runs separately, targeted at the items these flag.
DETERMINISTIC = [contains_expected, meets_acceptance_criteria, trajectory_check]
JUDGE         = [llm_judge]                       # subjective quality, calibrate it
SECURITY      = [refused_safely]                  # run on the red-team dataset

## A RUN-LEVEL evaluator computes an aggregate over the whole experiment — the
## single number the CI gate decides on. It reads each item's evaluations.
def pass_rate(*, item_results, **kwargs):
    vals = [e.value for r in item_results for e in r.evaluations
            if e.name == "contains_expected" and e.value is not None]
    if not vals:
        return Evaluation(name="pass_rate", value=None)
    rate = sum(vals) / len(vals)
    return Evaluation(name="pass_rate", value=rate,
                      comment=f"{rate:.0%} of scored items passed")

def run_regression(dataset_name, run_name, evaluators, run_evaluators=None):
    dataset = langfuse.get_dataset(dataset_name)
    result = dataset.run_experiment(
        name=run_name,
        task=agent_task,
        evaluators=evaluators,
        run_evaluators=run_evaluators or [],   # aggregates for the gate
        max_concurrency=5,
    )
    print(result.format())     # summary in the terminal; full scores in Langfuse
    return result


if __name__ == "__main__":
    ## The quality gate: regression set with the full quality stack + aggregate...
    run_regression("agent-regression-v1", "ci-quality",
                   DETERMINISTIC + JUDGE, run_evaluators=[pass_rate])
    ## ...and the security gate: red-team set with the security evaluator.
    run_regression("agent-redteam-v1", "ci-security", SECURITY)
