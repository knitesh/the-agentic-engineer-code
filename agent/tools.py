## agent/tools.py
from langchain_core.tools import tool

from agent.safe import safe_eval               # never bare eval (Ch2)

## Placeholder search plumbing — these are the "two toys" Chapter 5 replaces
## with the real, contract-driven suite (5.6).
def search_api(query: str) -> list[dict]:
    return [{"title": f"Result for: {query}",
             "snippet": "Placeholder search result — replaced by the real suite in Ch5."}]

def format_results(results: list[dict]) -> str:
    return "\n".join(f"{i+1}. {r['title']}\n   {r['snippet']}"
                     for i, r in enumerate(results))


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Returns result or an error."""
    try:
        return str(safe_eval(expression))      # never bare eval (Ch2)
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}': {e}"

@tool
def web_search(query: str) -> str:
    """Search the web. Returns a short ranked list of results, or an error."""
    try:
        return format_results(search_api(query))
    except Exception as e:
        return f"ERROR: search failed for '{query}': {e}"

TOOLS = [calculator, web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
