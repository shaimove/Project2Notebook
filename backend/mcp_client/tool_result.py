"""Helpers for interpreting MCP tool call results."""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.exceptions import ToolError


def _label(tool: str, server: str = "") -> str:
    return f"{server}/{tool}" if server else tool


def require_tool_result(
    result: Dict[str, Any],
    *,
    tool: str,
    server: str = "",
) -> Dict[str, Any]:
    """Return *result* or raise ``ToolError`` when the tool reported failure."""
    if not isinstance(result, dict):
        raise ToolError(f"{_label(tool, server)}: unexpected non-dict result")
    if "error" in result:
        raise ToolError(f"{_label(tool, server)}: {result['error']}")
    return result


def optional_tool_result(
    result: Dict[str, Any],
    *,
    tool: str,
    server: str = "",
) -> Optional[Dict[str, Any]]:
    """Return *result* or ``None`` when the tool reported failure (no raise)."""
    if not isinstance(result, dict) or "error" in result:
        return None
    return result
