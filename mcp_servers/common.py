"""Shared framework for defining MCP tool servers.

Each server registers a set of *tools*. A tool has a name, a human description,
a JSON-ish input schema (for documentation / discovery) and a Python handler
``(args: dict) -> dict``.

The in-process MCP client (``backend/mcp_client``) discovers and calls tools
through these server objects. The same server definitions can be exposed over
the real MCP stdio transport via ``serve_fastmcp`` when the optional ``mcp``
package is installed (see ``__main__`` blocks in each server module). This keeps
reasoning (agents) cleanly separated from execution (tool servers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler


@dataclass
class MCPServer:
    name: str
    description: str = ""
    tools: Dict[str, Tool] = field(default_factory=dict)

    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        self.tools[name] = Tool(name, description, input_schema, handler)

    def tool(self, description: str = "", input_schema: Dict[str, Any] | None = None):
        """Decorator form for registering a tool handler."""

        def deco(fn: ToolHandler) -> ToolHandler:
            self.add_tool(fn.__name__, description or (fn.__doc__ or ""), input_schema or {}, fn)
            return fn

        return deco

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not found on server '{self.name}'")
        return self.tools[tool_name].handler(args or {})


def serve_fastmcp(server: MCPServer) -> None:
    """Expose an MCPServer over real MCP stdio transport (optional).

    Requires the ``mcp`` package (``pip install mcp``). This lets the very same
    tool definitions run as a standalone MCP server process. If the package is
    not installed we print a clear message instead of failing silently.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        print(
            f"[{server.name}] The 'mcp' package is not installed. "
            "Install it (pip install mcp) to serve over real MCP stdio. "
            "The in-process client in backend/mcp_client works without it."
        )
        return

    app = FastMCP(server.name)
    for tool in server.tools.values():
        # Wrap each handler so FastMCP can expose it. We accept a single dict
        # argument named 'args' for simplicity and schema-stability.
        def make_fn(t: Tool):
            def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
                return t.handler(args or {})

            _fn.__name__ = t.name
            _fn.__doc__ = t.description
            return _fn

        app.tool()(make_fn(tool))

    app.run()
