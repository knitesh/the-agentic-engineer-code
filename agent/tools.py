## agent/tools.py — the real suite, replacing the two 4.5 placeholders.
import os
import httpx
from typing import Literal
from pydantic import BaseModel, Field
## Repo glue: langchain-core 0.3's decorator lacks the `metadata` kwarg the
## contracts below use — this override adds it (see agent/toolkit.py).
from agent.toolkit import tool
from agent.safe import safe_eval            # the sandboxed evaluator from Ch2
from agent.sandbox import sandbox           # isolation primitive (see repo)

WORKSPACE = os.environ.get("AGENT_WORKSPACE", "/srv/agent/workspace")


## --- COMPUTE -------------------------------------------------------------
@tool(metadata={"kind": "compute"})
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact result.

    Use this for ANY arithmetic — do not compute in your head, you make
    mistakes. Handles +, -, *, /, **, parentheses. Returns the numeric
    result, or a clear error if the expression is malformed.

    Args:
        expression: e.g. "0.17 * 4200" or "(3 + 4) ** 2".
    """
    try:
        return str(safe_eval(expression))    # never bare eval (Ch2)
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}': {e}. Check the syntax."


## --- SEARCH --------------------------------------------------------------
class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query — be specific.")
    num_results: int = Field(
        default=5, ge=1, le=10,
        description="How many results to return (1-10).",
    )


@tool(args_schema=WebSearchArgs, metadata={"kind": "search"})
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for current information.

    Use when you need facts you don't have or that may have changed
    (news, prices, recent events). Returns a ranked list of result
    titles + snippets to reason over — treat them as LEADS, not verified
    facts. Returns "No results found." if the search is empty.
    """
    try:
        resp = httpx.get(
            "https://api.example-search.com/v1/search",
            params={"q": query, "n": num_results},
            timeout=8.0,
        )
    except httpx.TimeoutException:
        return f"ERROR: search timed out for '{query}'. Try again or narrow the query."
    except httpx.RequestError as e:
        return f"ERROR: could not reach search service: {e}. Proceed without web data if possible."
    if resp.status_code != 200:
        return f"ERROR: search returned {resp.status_code}. The service may be down."

    results = resp.json().get("results", [])
    if not results:
        return f"No results found for '{query}'. Try broader or different terms."
    return "\n".join(
        f"{i+1}. {r['title']}\n   {r['snippet']}"
        for i, r in enumerate(results[:num_results])
    )


## --- READ ----------------------------------------------------------------
def _safe_path(rel: str) -> str:
    full = os.path.realpath(os.path.join(WORKSPACE, rel))
    if not full.startswith(WORKSPACE + os.sep):
        raise ValueError("escapes workspace")
    return full


@tool(metadata={"kind": "read"})
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the agent workspace (relative path only).

    Use to read files the agent has access to. Returns up to 10,000
    characters of content, or a clear error. Cannot read outside the
    workspace. For writing, use `write_file`.
    """
    try:
        full = _safe_path(path)
    except ValueError:
        return f"ERROR: '{path}' is outside the workspace and cannot be read."
    if not os.path.isfile(full):
        return f"ERROR: no file at '{path}'. Use a path relative to the workspace."
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()[:10_000]
    except UnicodeDecodeError:
        return f"ERROR: '{path}' is not a UTF-8 text file."


## --- WRITE ---------------------------------------------------------------
@tool(metadata={"kind": "write"})
def write_file(path: str, content: str) -> str:
    """Write text to a file in the agent workspace (relative path only).

    This WRITES to disk and OVERWRITES any existing file at the path.
    Cannot write outside the workspace. Returns a confirmation with the
    bytes written, or a clear error. Prefer reading first if unsure
    whether a file already exists.
    """
    try:
        full = _safe_path(path)
    except ValueError:
        return f"ERROR: '{path}' is outside the workspace; refusing to write."
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            n = f.write(content)
        return f"Wrote {n} characters to '{path}'."
    except OSError as e:
        return f"ERROR: could not write '{path}': {e}."


## --- MEMORY (Ch6) --------------------------------------------------------
## agent/tools.py — add to the suite from 5.6.
## `memory` is the shared SemanticMemory from agent/memory_backend.py.
from agent.memory_backend import memory

@tool(metadata={"kind": "write"})
def remember(fact: str) -> str:
    """Save a durable fact to long-term memory for future sessions.

    Use this ONLY for information worth recalling later: user preferences,
    stable facts about the user, important decisions, resolved issues. Do
    NOT use it for transient details, this conversation's working notes, or
    anything you can recompute. One clear fact per call.
    """
    # `memory` is the shared SemanticMemory from agent/memory_backend.py.
    memory.remember(fact, metadata={"source": "agent", "kind": "fact"})
    return f"Remembered: {fact}"

## --- THE REGISTRY --------------------------------------------------------
TOOLS = [calculator, web_search, read_file, write_file, remember]   # + Ch6
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
