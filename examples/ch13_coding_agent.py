## examples/ch13_coding_agent.py — Chapter 13's coding-domain extension of the
## shared project: CodingState, CodingConfig, and the reflect-on-tests node,
## exactly as presented in the book. The chapter is a case study — these pieces
## show how the Ch3-Ch12 machinery is repointed at a test-gated coding loop.
## Helpers the chapter references (diffing, observation shaping) are provided
## below as minimal repo glue.
from dataclasses import dataclass
from langchain_core.messages import SystemMessage

from agent.state import AgentState
from agent.config import HarnessConfig
from agent.sandbox import run_in_sandbox      # no net, no creds (agent/sandbox.py)

## Coding-domain extension of AgentState. seen_signatures originates in Ch3 and
## is reused by Ch4's act_node; here a "signature" is a (failing, passing) test
## split, and the field is overridden from Ch3/Ch4's set to a list, because
## thrash detection counts how many times a split recurs (a set would collapse
## the duplicates), and lists also serialize cleanly to the durable checkpointer.
## The graph seeds it to [].
class CodingState(AgentState):
    workdir: str                  # the sandboxed working copy (never prod)
    best_diff: str | None         # best partial seen so far, kept for honest exit
    final_diff: str | None        # the patch we return
    seen_signatures: list         # overrides Ch3/Ch4's set: ordered, keeps duplicates
    # messages, iterations, exit_reason: inherited from Ch3/Ch4


## Coding-domain extension of HarnessConfig. Both fields are new — the shared
## config has no notion of "test run" or "oscillation" — so they're added here
## rather than overloaded onto Ch3's stall_threshold, which counts a different
## thing (consecutive no-progress iterations, not recurring test-outcome splits).
@dataclass
class CodingConfig(HarnessConfig):
    thrash_threshold: int = 3       # signature recurrences before declaring thrash
    test_timeout: float = 60.0      # seconds allowed for one sandboxed test run



## --- Repo glue: the helpers reflect_on_tests references ---------------------

def current_diff(state: "CodingState") -> str:
    """The working diff of the sandboxed copy vs. its base (git-based)."""
    result = run_in_sandbox(["git", "diff"], cwd=state["workdir"])
    return result.stdout

def fewer_failures(state: "CodingState", result) -> str | None:
    """Keep the best partial diff seen so far (fewest failing tests)."""
    best = state.get("best_diff")
    prior_failures = state.get("_best_failure_count")
    failures_now = len(result.failed_tests)
    if best is None or prior_failures is None or failures_now < prior_failures:
        state["_best_failure_count"] = failures_now
        return current_diff(state)
    return best

def test_failures_as_observation(result) -> SystemMessage:
    """Feed the REAL failures back into the loop — the tests grade, not the model."""
    failed = "\n".join(result.failed_tests) or result.stderr[:2_000]
    return SystemMessage(content=f"[Test results] Failing:\n{failed}")


def reflect_on_tests(state: CodingState, config: CodingConfig) -> dict:
    """Run the suite in the sandbox, turn the result into an observation, and
    track convergence. The agent never grades itself here — the tests do."""
    result = run_in_sandbox(["pytest", "-q"], cwd=state["workdir"],
                            timeout=config.test_timeout)   # no net, no creds

    signature = (result.failed_tests, result.passed_tests)   # the loop's "where am I"
    # seen_signatures is a list in CodingState (it overrides Ch3/Ch4's set), so
    # append + count give a true recurrence count instead of a deduped membership test.
    state["seen_signatures"].append(signature)               # Ch3/Ch4 field, retyped

    # Thrash detection: same failing/passing split seen before => oscillating.
    # This is Ch3's deadlock check, repointed at test outcomes.
    if state["seen_signatures"].count(signature) >= config.thrash_threshold:
        return {"exit_reason": "thrash:no_convergence",      # leave with a partial
                "final_diff": state["best_diff"]}

    if not result.failed_tests:                              # all green
        return {"exit_reason": "tests_pass", "final_diff": current_diff(state)}

    # Still failing but making progress — feed the real failures back in.
    return {"messages": [test_failures_as_observation(result)],
            "best_diff": fewer_failures(state, result)}      # keep the best so far
