"""Visualization MCP Server.

Deterministic Plotly HTML charts for EDA, data quality, split diagnostics, and
audit inspection. Agents call these tools instead of importing visualization
code directly — results are saved under ``artifacts/{project_id}/plots/`` and
reused by the dashboard and notebook.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pathlib import Path

from backend.services import artifact_store
from backend.services.plotly_viz import (
    generate_audit_missingness_plotly,
    generate_data_quality_plotly,
    generate_eda_plotly,
    generate_split_target_plotly,
)
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="viz-tools",
    description="Generate interactive Plotly HTML visualizations for EDA and data curation.",
)

_PID = {"project_id": {"type": "string", "required": True}}


@server.tool(
    "Build grouped class EDA Plotly HTML (one row per modeling feature).",
    {
        **_PID,
        "csv_path": {"type": "string", "required": True},
        "features": {"type": "array", "items": {"type": "string"}},
        "target": {"type": "string"},
        "task_type": {"type": "string"},
    },
)
def generate_eda_plotly_html(args: Dict[str, Any]) -> Dict[str, Any]:
    return generate_eda_plotly(
        args["project_id"],
        args["csv_path"],
        list(args.get("features") or []),
        args.get("target"),
        args.get("task_type") or "",
    )


@server.tool(
    "Plot target distribution per train/valid/test split (Plotly HTML).",
    {
        **_PID,
        "csv_path": {"type": "string", "required": True},
        "target": {"type": "string", "required": True},
        "split_meta": {"type": "object"},
        "scaling_method": {"type": "string"},
    },
)
def generate_split_target_plotly_html(args: Dict[str, Any]) -> Dict[str, Any]:
    return generate_split_target_plotly(
        args["project_id"],
        args["csv_path"],
        args["target"],
        args.get("split_meta") or {},
        args.get("scaling_method") or "standard",
    )


@server.tool(
    "Data-quality overview: missingness by column + issue severity (Plotly HTML).",
    {**_PID, "report": {"type": "object", "required": True}},
)
def generate_data_quality_plotly_html(args: Dict[str, Any]) -> Dict[str, Any]:
    return generate_data_quality_plotly(args["project_id"], args.get("report") or {})


@server.tool(
    "Missing-value bar chart from data audit column profiles (Plotly HTML).",
    {**_PID, "audit": {"type": "object", "required": True}},
)
def generate_audit_missingness_plotly_html(args: Dict[str, Any]) -> Dict[str, Any]:
    return generate_audit_missingness_plotly(args["project_id"], args.get("audit") or {})


@server.tool(
    "List Plotly HTML plot files saved for a project.",
    _PID,
)
def list_html_plots(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    names = [Path(p).name for p in artifact_store.list_html_plots(project_id)]
    return {"ok": True, "html_plots": names, "count": len(names)}


if __name__ == "__main__":
    serve_fastmcp(server)
