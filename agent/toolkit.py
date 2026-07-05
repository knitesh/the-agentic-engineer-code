## agent/toolkit.py — repo glue for the tool decorator.
## langchain-core 0.3.x's @tool decorator does not accept a `metadata` kwarg
## (BaseTool HAS the field; the decorator just lacks the parameter). The book's
## tool contracts rely on @tool(metadata={"kind": ...}) — this thin wrapper
## attaches the metadata after construction so those listings run verbatim
## on the pinned stack.
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as _lc_tool


def tool(*dargs: Any, metadata: dict[str, Any] | None = None, **dkwargs: Any) -> Any:
    # bare usage: @tool
    if dargs and callable(dargs[0]) and metadata is None and not dkwargs:
        return _lc_tool(dargs[0])

    # parameterized usage: @tool(metadata=..., args_schema=..., ...)
    inner: Callable[[Callable[..., Any]], BaseTool] = (
        _lc_tool(*dargs, **dkwargs) if (dargs or dkwargs) else _lc_tool
    )

    def wrap(fn: Callable[..., Any]) -> BaseTool:
        t = inner(fn)
        t.metadata = dict(metadata or {})
        return t

    return wrap
