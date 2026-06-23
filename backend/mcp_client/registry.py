"""Registry of available MCP servers.

For the MVP the client talks to the servers *in-process* (each server is a
Python module exposing an ``MCPServer`` object). This keeps the demo robust and
dependency-light while preserving the protocol-style architecture: tools are
owned by independent servers, discovered through a registry, and invoked through
structured schemas via the client.

To switch to *real* MCP stdio transport, replace ``InProcessRegistry`` with one
that launches each ``mcp_servers/*_server.py`` as a subprocess (they already
support ``serve_fastmcp``) and routes ``call_tool`` over the MCP protocol. The
client API does not change.
"""
from __future__ import annotations

from typing import Dict

from mcp_servers import (
    codegen_server,
    data_inspection_server,
    data_quality_server,
    eda_review_server,
    eda_server,
    experiment_server,
    modeling_server,
    notebook_server,
    prior_art_server,
    preprocessing_server,
    project_understanding_server,
)
from mcp_servers.common import MCPServer


class InProcessRegistry:
    """Maps server name -> MCPServer instance."""

    def __init__(self) -> None:
        servers = [
            project_understanding_server.server,
            prior_art_server.server,
            data_quality_server.server,
            data_inspection_server.server,
            eda_server.server,
            eda_review_server.server,
            preprocessing_server.server,
            modeling_server.server,
            experiment_server.server,
            notebook_server.server,
            codegen_server.server,
        ]
        self._servers: Dict[str, MCPServer] = {s.name: s for s in servers}

    def get(self, server_name: str) -> MCPServer:
        if server_name not in self._servers:
            raise KeyError(f"Unknown MCP server: {server_name}")
        return self._servers[server_name]

    @property
    def servers(self) -> Dict[str, MCPServer]:
        return self._servers


registry = InProcessRegistry()
