"""Central registry of tools available to the agent orchestrator."""
from __future__ import annotations

from typing import Any, Callable

from tools import afr_search, asx_tool, rba_tool, sentiment_tool

_TOOLS: dict[str, tuple[dict, Callable[[dict], Any]]] = {
    rba_tool.TOOL_SCHEMA["name"]: (rba_tool.TOOL_SCHEMA, rba_tool.run),
    asx_tool.TOOL_SCHEMA["name"]: (asx_tool.TOOL_SCHEMA, asx_tool.run),
    afr_search.TOOL_SCHEMA["name"]: (afr_search.TOOL_SCHEMA, afr_search.run),
    sentiment_tool.TOOL_SCHEMA["name"]: (sentiment_tool.TOOL_SCHEMA, sentiment_tool.run),
}


def schemas() -> list[dict]:
    return [schema for schema, _ in _TOOLS.values()]


def names() -> list[str]:
    return list(_TOOLS.keys())


def call(name: str, args: dict) -> Any:
    if name not in _TOOLS:
        raise ValueError(f"Unknown tool: {name}. Known tools: {names()}")
    _, fn = _TOOLS[name]
    return fn(args)
