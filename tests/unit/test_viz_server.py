"""Tests for viz-tools MCP server."""
from __future__ import annotations

import pandas as pd

from backend.services import artifact_store
from mcp_servers import viz_server


def test_generate_data_quality_plotly_html(isolated_storage, tmp_path):
    pid = "viz-dq"
    report = {
        "column_profiles": [
            {"name": "a", "pct_missing": 0.0},
            {"name": "b", "pct_missing": 0.4},
        ],
        "issues": [{"severity": "warning", "description": "high missing"}],
        "modeling_features": ["a"],
        "excluded_columns": ["b"],
    }
    out = viz_server.server.call("generate_data_quality_plotly_html", {
        "project_id": pid,
        "report": report,
    })
    assert out["ok"] is True
    assert out["html_name"] == "data_quality_overview.html"
    assert (artifact_store.plots_dir(pid) / "data_quality_overview.html").is_file()


def test_generate_audit_missingness_plotly_html(isolated_storage):
    pid = "viz-audit"
    audit = {
        "columns": [
            {"name": "x", "missing_frac": 0.1},
            {"name": "y", "missing_frac": 0.3},
        ]
    }
    out = viz_server.server.call("generate_audit_missingness_plotly_html", {
        "project_id": pid,
        "audit": audit,
    })
    assert out["ok"] is True
    assert (artifact_store.plots_dir(pid) / "audit_missingness.html").is_file()


def test_list_html_plots(isolated_storage):
    pid = "viz-list"
    artifact_store.plots_dir(pid).mkdir(parents=True, exist_ok=True)
    (artifact_store.plots_dir(pid) / "eda_features.html").write_text("<html></html>")
    out = viz_server.server.call("list_html_plots", {"project_id": pid})
    assert out["ok"] is True
    assert "eda_features.html" in out["html_plots"]


def test_generate_eda_plotly_html(isolated_storage, tmp_path):
    pid = "viz-eda"
    csv = tmp_path / "data.csv"
    pd.DataFrame({"target": [0, 1, 0, 1], "feat": [1.0, 2.0, 3.0, 4.0]}).to_csv(csv, index=False)
    out = viz_server.server.call("generate_eda_plotly_html", {
        "project_id": pid,
        "csv_path": str(csv),
        "features": ["feat"],
        "target": "target",
        "task_type": "binary_classification",
    })
    assert out["ok"] is True
    assert (artifact_store.plots_dir(pid) / "eda_features.html").is_file()
