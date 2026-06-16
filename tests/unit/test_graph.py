"""Unit tests for graph continue-on-failure behavior."""
from __future__ import annotations

from backend.agents import graph as graph_module
from backend.agents.graph import run_graph
from backend.agents.state import new_state
from backend.mcp_client.client import MCPClient


def _ok_node(state, _client):
    state["completed_marker"] = True
    return state


def _fail_node(_state, _client):
    raise ValueError("simulated node failure")


def test_run_graph_records_error_and_continues():
    workflow = [
        ("Failing Step", _fail_node),
        ("Successful Step", _ok_node),
    ]
    original = graph_module.WORKFLOW
    graph_module.WORKFLOW = workflow
    try:
        state = new_state(
            project_id="graph-test",
            project_document_path="",
            csv_paths=[],
            pdf_paths=[],
        )
        result = run_graph(state)
    finally:
        graph_module.WORKFLOW = original

    assert len(result["errors"]) == 1
    assert result["errors"][0]["step"] == "Failing Step"
    assert "ValueError" in result["errors"][0]["error"]
    assert result.get("completed_marker") is True
    assert result["timeline"][0]["status"] == "error"
    assert result["timeline"][1]["status"] == "completed"


def test_run_graph_collects_multiple_errors():
    workflow = [
        ("Fail One", _fail_node),
        ("Fail Two", _fail_node),
    ]
    original = graph_module.WORKFLOW
    graph_module.WORKFLOW = workflow
    try:
        state = new_state("graph-test", "", [], [])
        result = run_graph(state)
    finally:
        graph_module.WORKFLOW = original

    assert len(result["errors"]) == 2
    assert all(item["status"] == "error" for item in result["timeline"])


def test_run_graph_appends_tool_calls_from_mcp_client():
    from unittest.mock import patch

    def tool_node(state, client: MCPClient):
        client.call_tool("fake-server", "noop", {})
        return state

    class FakeServer:
        name = "fake-server"

        def call(self, _tool, _args):
            return {"ok": True}

    workflow = [("Tool Step", tool_node)]
    original = graph_module.WORKFLOW
    graph_module.WORKFLOW = workflow
    try:
        with patch("backend.mcp_client.client.registry.get", return_value=FakeServer()):
            state = new_state("graph-test", "", [], [])
            result = run_graph(state)
    finally:
        graph_module.WORKFLOW = original

    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "noop"
