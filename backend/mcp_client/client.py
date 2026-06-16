"""MCP client abstraction.

Agents call tools *only* through this client — never by importing tool
implementations directly. Every call is timed and logged so the UI can show a
transparent trace of which server/tool was used, with what input and result.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from backend.mcp_client.registry import registry


def _summarize(output: Any) -> str:
    """Create a short, human-readable summary of a tool's output."""
    if isinstance(output, dict):
        if "error" in output:
            return f"error: {output['error']}"
        keys = list(output.keys())
        preview = ", ".join(keys[:6])
        return f"keys: {preview}" + (" …" if len(keys) > 6 else "")
    text = str(output)
    return text[:160]


class MCPClient:
    def __init__(self, on_log: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        # on_log lets the runner collect tool-call logs into the agent state.
        self._on_log = on_log
        self.call_log: List[Dict[str, Any]] = []

    def list_available_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for server_name, server in registry.servers.items():
            for t in server.list_tools():
                tools.append({"server": server_name, **t})
        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        status = "success"
        output: Dict[str, Any]
        try:
            server = registry.get(server_name)
            output = server.call(tool_name, arguments)
            if isinstance(output, dict) and "error" in output:
                status = "error"
        except Exception as exc:  # noqa: BLE001 - report tool failures, don't crash run
            output = {"error": f"{type(exc).__name__}: {exc}"}
            status = "error"
        duration_ms = int((time.time() - start) * 1000)

        log_entry = {
            "server": server_name,
            "tool": tool_name,
            "input": _redact(arguments),
            "output_summary": _summarize(output),
            "status": status,
            "duration_ms": duration_ms,
        }
        self.call_log.append(log_entry)
        if self._on_log:
            self._on_log(log_entry)
        return output


def _redact(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Keep logged inputs small/readable (drop large text/object payloads)."""
    redacted: Dict[str, Any] = {}
    for k, v in (arguments or {}).items():
        if isinstance(v, str) and len(v) > 200:
            redacted[k] = v[:200] + f"… (+{len(v) - 200} chars)"
        elif isinstance(v, (dict, list)) and len(str(v)) > 200:
            redacted[k] = f"<{type(v).__name__} with {len(v)} items>"
        else:
            redacted[k] = v
    return redacted
