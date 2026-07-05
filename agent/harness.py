## agent/harness.py — AgentHarness: the new center of gravity (3.5)
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError   # LangGraph's own loop-limit error

from agent.errors import InvalidInput, HarnessTimeout

if TYPE_CHECKING:
    from agent.config import HarnessConfig
## --- Provider transient errors ------------------------------------------------
## Different providers raise different transient errors. Rather than scatter
## provider-specific imports through the harness, we normalize them into one
## tuple the retry logic checks. Adjust the imports to YOUR provider.
try:
    from openai import APITimeoutError, RateLimitError, APIConnectionError
    PROVIDER_TRANSIENT_ERRORS = (APITimeoutError, RateLimitError, APIConnectionError)
except ImportError:                       # provider SDK not installed in this env
    PROVIDER_TRANSIENT_ERRORS = ()

MAX_INPUT_CHARS = 4_000   # bound on raw request size, enforced by the injector (3.2)


## Framework-agnostic: an input injector
def inject_input(raw_request: dict) -> dict:
    """Translate a raw request into a valid initial agent state.
    Validate here so the agent never reasons over garbage,
    and seed EVERY field the exit logic will later read."""
    user_message = raw_request.get("message")
    if not user_message or not isinstance(user_message, str):
        raise InvalidInput("message is required and must be a string")
    if len(user_message) > MAX_INPUT_CHARS:
        raise InvalidInput(f"message exceeds {MAX_INPUT_CHARS} characters")

    return {
        "messages": [HumanMessage(content=user_message)],
        "goal": user_message,
        # --- bookkeeping the exit logic depends on; seeded, never user-provided ---
        "iterations": 0,
        "tokens_used": 0,             # 3.13 reads this on iteration 1
        "started_at": time.time(),    # 3.13's wall-clock check reads this
        "action_history": [],         # 3.14 deadlock detection
        "seen_signatures": set(),     # 3.14 repetition detection
        "stall_count": 0,             # 3.12 convergence tracking
        # seeded per the injector contract ("seed EVERY field the exit logic reads"):
        "required_keys": [],          # 3.10 acceptance criteria; set per-goal at setup
        "known_facts": {},            # 3.10/3.12 read these on iteration 1
        "exit_reason": None,
        "final_answer": None,
        "status": None,
        # Ch7-added fields (8.3 calls these out): guard verdict + human decision
        "guard_block": None,
        "approval_granted": False,
    }


@dataclass
class RunHandle:
    thread_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    pending: list[str] = field(default_factory=list)


class AgentHarness:
    def __init__(self, compiled_graph, config: "HarnessConfig", sinks):
        self.app = compiled_graph
        self.config = config
        self.sinks = sinks
        # one shared thread pool so each invoke can be time-bounded from outside
        self._pool = ThreadPoolExecutor(max_workers=config.invoke_workers)
        # re-entry handling (3.15)
        self.active_runs = {}
        self.re_entry_policy = config.re_entry_policy

    def _initial_state(self, message: str) -> dict:
        # delegates to the same injector contract from 3.2 — seeds EVERY field
        return inject_input({"message": message})

    def run_safe(self, message: str, thread_id: str | None = None) -> dict:
        """The single execution core every run mode goes through.
        CONTRACT: never raises. All paths return a dict with at least 'status'."""
        started = time.time()
        try:
            # 1. INPUT INJECTION (+ validation)
            try:
                state = self._initial_state(message)
            except InvalidInput as e:
                return {"status": "rejected", "error": str(e), "exit_reason": "invalid_input"}

            # 2. CONTEXT MANAGEMENT: per-run config handed to the graph
            run_config = {
                "configurable": {"thread_id": thread_id or "default"},
                "recursion_limit": self.config.recursion_limit,
            }

            # 3. ENVIRONMENT CONTROL: timeout + retries + error handling
            result = self._execute_with_environment_control(state, run_config)

            # 4. OUTPUT CAPTURE: route metrics; return answer
            result["latency_ms"] = int((time.time() - started) * 1000)
            self._capture(result)
            return result
        except Exception as e:
            # Last-resort net so the CONTRACT holds even if capture/config misbehaves.
            self.sinks.errors.log(f"run_safe unexpected: {e!r}")
            return {"status": "error", "error": str(e),
                    "exit_reason": "harness_fault",
                    "latency_ms": int((time.time() - started) * 1000)}

    def _capture(self, result: dict) -> None:
        try:
            self.sinks.metrics.record(
                iterations=result.get("iterations", 0),
                latency_ms=result.get("latency_ms", 0),
                status=result.get("status"),
                exit_reason=result.get("exit_reason"),
            )
        except Exception as e:
            # capture failures must NEVER fail a run
            self.sinks.errors.log(f"metrics capture failed: {e!r}")

    def _invoke_with_timeout(self, state, run_config) -> dict:
        """REAL wall-clock timeout around the BLOCKING invoke.
        app.invoke() is synchronous; the only way to bound a hung provider call
        is to run it off-thread and reclaim control with .result(timeout=...)."""
        future = self._pool.submit(self.app.invoke, state, config=run_config)
        try:
            return future.result(timeout=self.config.max_seconds)
        except FutureTimeout:
            future.cancel()   # best-effort; the underlying call may still finish
            raise HarnessTimeout(f"invoke exceeded {self.config.max_seconds}s")

    def _execute_with_environment_control(self, state, run_config) -> dict:
        last_error = None
        for attempt in range(self.config.retry.max_retries + 1):
            try:
                final_state = self._invoke_with_timeout(state, run_config)
                return {
                    "status": "ok",
                    "final_answer": final_state.get("final_answer"),
                    "iterations": final_state.get("iterations", 0),
                    "exit_reason": final_state.get("exit_reason", "goal_achieved"),
                }
            except GraphRecursionError:
                # Hit LangGraph's recursion limit — the catastrophe net fired,
                # which means our OWN exit logic failed to stop the loop. Bug.
                self.sinks.errors.log("GraphRecursionError: exit logic failed to stop")
                return {"status": "loop_limit", "final_answer": None,
                        "exit_reason": "recursion_limit"}
            except (HarnessTimeout, *PROVIDER_TRANSIENT_ERRORS) as e:
                last_error = e
                self.sinks.errors.log(f"attempt {attempt} transient: {e!r}")
                time.sleep(self.config.retry.base_backoff_seconds * (2 ** attempt))
                continue
            except Exception as e:
                # Non-transient: don't retry, capture and fail cleanly.
                self.sinks.errors.log(f"fatal: {e!r}")
                return {"status": "error", "error": str(e), "exit_reason": "fatal"}
        return {"status": "error", "exit_reason": "retries_exhausted",
                "error": f"exhausted retries: {last_error!r}"}

    ## --- Run modes (3.7) ---------------------------------------------------

    def run_single(self, message: str) -> dict:
        return self.run_safe(message)

    def run_batch(self, messages: list[str], max_concurrency: int = 8) -> list[dict]:
        out = []
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futs = {pool.submit(self.run_safe, m): m for m in messages}
            for f in as_completed(futs):
                out.append(f.result())     # safe: run_safe honors the never-raise contract
        return out

    def run_scheduled(self, scheduled_goal: str) -> dict:
        # No user; the "input" is a fixed goal the schedule fires with.
        return self.run_safe(scheduled_goal, thread_id="scheduled")

    ## --- Re-entry handling (3.15) -------------------------------------------

    def run_interactive(self, session_id: str, message: str) -> dict:
        # interactive mode now goes THROUGH submit, so re-entry is always handled
        return self.submit(session_id, message)

    def submit(self, session_id: str, message: str) -> dict:
        active = self.active_runs.get(session_id)
        if active is None:
            return self._start_run(session_id, message)

        # A loop is already running for this session — apply the policy.
        if self.re_entry_policy == "reject":
            return {"status": "busy", "message": "Still working on your last request."}
        if self.re_entry_policy == "queue":
            active.pending.append(message)
            return {"status": "queued"}
        if self.re_entry_policy == "interrupt":
            active.cancel_event.set()              # cooperative cancellation
            active.done_event.wait(timeout=self.config.max_seconds)  # let it unwind
            return self._start_run(session_id, message)
        if self.re_entry_policy == "incorporate":
            # append to the running loop's pending injection; the loop reads it at entry
            active.pending.append(message)
            return {"status": "incorporated"}
        return {"status": "error", "error": f"unknown policy {self.re_entry_policy}"}

    def _start_run(self, session_id: str, message: str) -> dict:
        """Register the run, execute via the same run_safe core, drain the queue."""
        handle = RunHandle(thread_id=session_id)
        self.active_runs[session_id] = handle
        try:
            # the cancel flag reaches the graph via config; the loop's entry check reads it
            result = self.run_safe(message, thread_id=session_id)
        finally:
            handle.done_event.set()
            self.active_runs.pop(session_id, None)
        # 'queue' policy: drain one pending message as the next turn
        if handle.pending:
            return self.submit(session_id, handle.pending.pop(0))
        return result

    ## --- Streaming (3.19) ----------------------------------------------------

    def run_streaming(self, session_id: str, message: str):
        """A streaming run mode. Same injection/sink discipline as run_safe,
        but yields progress as the loop turns instead of returning at the end."""
        state = self._initial_state(message)        # same injector contract from 3.2
        config = {"configurable": {"thread_id": session_id},
                  "recursion_limit": self.config.recursion_limit}
        for chunk in self.app.stream(state, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                event = self._to_progress_event(node, update)
                self.sinks.stream.push(session_id, event)   # to the UI
                self.sinks.trace.write(event)               # to observability

    def _to_progress_event(self, node: str, update: dict) -> dict:
        """Repo helper: shape a node's state delta into a small progress event."""
        return {"node": node, "keys": sorted(update.keys()) if update else []}
