## agent/sandbox.py — the isolation primitive referenced by the tool suite (Ch5)
## and the coding case study (Ch13). Referenced in the book as "see repo".
##
## This is a deliberately minimal, dependency-free stand-in: it runs a command
## in a subprocess with a scrubbed environment, a working-directory jail, and a
## hard timeout. In production, replace the subprocess call with a real
## isolation boundary (container, microVM, or remote runner) — the INTERFACE
## is the part the rest of the project depends on.
import os
import subprocess
from dataclasses import dataclass, field


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    # populated by run_in_sandbox for pytest-style runs (Ch13)
    failed_tests: tuple = field(default_factory=tuple)
    passed_tests: tuple = field(default_factory=tuple)


def run_in_sandbox(cmd: list[str], cwd: str, timeout: float = 60.0) -> SandboxResult:
    """Run `cmd` inside `cwd` with no inherited credentials and a hard timeout.
    Never raises on command failure — failure is data the agent observes."""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": cwd}   # no creds, no keys
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return SandboxResult(returncode=-1, stdout=e.stdout or "",
                             stderr="TIMEOUT", timed_out=True)
    result = SandboxResult(proc.returncode, proc.stdout, proc.stderr)
    if cmd and cmd[0] == "pytest":                      # Ch13: parse the outcome
        result.failed_tests = tuple(
            line.split(" ")[0] for line in proc.stdout.splitlines()
            if line.startswith("FAILED"))
        result.passed_tests = tuple(
            line.split(" ")[0] for line in proc.stdout.splitlines()
            if line.endswith("PASSED"))
    return result


def sandbox(cmd: list[str], cwd: str, timeout: float = 60.0) -> str:
    """String-returning convenience wrapper for tool use: stdout on success,
    an explicit ERROR string on failure (failure-as-observation, Ch5)."""
    r = run_in_sandbox(cmd, cwd, timeout)
    if r.timed_out:
        return f"ERROR: command timed out after {timeout}s."
    if r.returncode != 0:
        return f"ERROR (exit {r.returncode}): {r.stderr[:2_000]}"
    return r.stdout[:10_000]
