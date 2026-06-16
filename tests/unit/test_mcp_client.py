"""Unit tests for MCP client error normalization."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.exceptions import ToolError
from backend.mcp_client.client import MCPClient


class _FakeServer:
    name = "fake-server"

    def __init__(self, handler):
        self._handler = handler

    def call(self, tool_name: str, args: dict) -> dict:
        return self._handler(tool_name, args)


def test_call_tool_wraps_tool_error_dict():
    server = _FakeServer(lambda _tool, _args: {"error": "tool failed"})
    client = MCPClient()

    with patch("backend.mcp_client.client.registry.get", return_value=server):
        output = client.call_tool("fake-server", "broken_tool", {})

    assert output == {"error": "tool failed"}
    assert client.call_log[-1]["status"] == "error"
    assert "tool failed" in client.call_log[-1]["output_summary"]


def test_call_tool_wraps_unhandled_exception():
    def boom(_tool, _args):
        raise RuntimeError("unexpected")

    server = _FakeServer(boom)
    client = MCPClient()

    with patch("backend.mcp_client.client.registry.get", return_value=server):
        output = client.call_tool("fake-server", "raise_tool", {})

    assert output["error"].startswith("RuntimeError:")
    assert client.call_log[-1]["status"] == "error"


def test_call_tool_required_raises_tool_error():
    server = _FakeServer(lambda _tool, _args: {"error": "nope"})
    client = MCPClient()

    with patch("backend.mcp_client.client.registry.get", return_value=server):
        with pytest.raises(ToolError, match="nope"):
            client.call_tool_required("fake-server", "broken_tool", {})


def test_call_tool_success_logs_ok_status():
    server = _FakeServer(lambda _tool, _args: {"ok": True, "value": 42})
    client = MCPClient()

    with patch("backend.mcp_client.client.registry.get", return_value=server):
        output = client.call_tool("fake-server", "good_tool", {"x": 1})

    assert output == {"ok": True, "value": 42}
    assert client.call_log[-1]["status"] == "success"
