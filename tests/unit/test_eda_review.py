"""Tests for EDA review MCP tools and agent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.agents.nodes import eda_review
from backend.services import artifact_store
from mcp_servers.eda_review_server import analyze_eda_tables, list_eda_plot_inventory
from tests.conftest import MockMCPClient, make_state


def _write_eda_tables(project_id: str) -> None:
    tdir = artifact_store.tables_dir(project_id)
    tdir.mkdir(parents=True, exist_ok=True)
    corr = pd.DataFrame({
        "": ["a", "b", "target"],
        "a": [1.0, 0.95, 0.8],
        "b": [0.95, 1.0, 0.1],
        "target": [0.8, 0.1, 1.0],
    })
    corr.to_csv(tdir / "correlation.csv", index=False)
    pd.DataFrame([{"column": "a", "n_outliers": 5, "pct": 0.06}]).to_csv(
        tdir / "outliers.csv", index=False
    )


def test_analyze_eda_tables_correlation(isolated_storage):
    pid = "eda-test-1"
    _write_eda_tables(pid)
    result = analyze_eda_tables({
        "project_id": pid,
        "data_audit": {"target": "target", "leakage_prone_columns": []},
        "spec": {"ml_task_type": "binary_classification"},
    })
    assert "a" in result["important_columns"]
    assert "correlation.csv" in result["analysis_sources"]
    assert any(g for g in result["multicollinear_groups"] if "a" in g and "b" in g)


def test_list_eda_plot_inventory(isolated_storage):
    pid = "eda-test-2"
    pdir = artifact_store.plots_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "correlation.png").write_bytes(b"fake")
    inv = list_eda_plot_inventory({
        "project_id": pid,
        "plot_names": ["correlation.png"],
    })
    assert len(inv["plots"]) == 1
    assert inv["plots"][0]["plot_type"] == "Numeric correlation heatmap"


def test_eda_review_agent_runs(isolated_storage):
    pid = "eda-test-3"
    _write_eda_tables(pid)
    pdir = artifact_store.plots_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "feature_target.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    def responder(server, tool, args):
        if tool == "analyze_eda_tables":
            return analyze_eda_tables(args)
        if tool == "list_eda_plot_inventory":
            return list_eda_plot_inventory(args)
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_id=pid,
        project_spec={"ml_task_type": "binary_classification", "targets": ["target"]},
        data_audit_report={"target": "target", "n_rows": 100, "n_cols": 3},
        eda_artifacts={"plots": [str(pdir / "feature_target.png")], "tables": []},
    )

    with patch("backend.agents.nodes.eda_review.llm") as mock_llm:
        mock_llm.enabled = False
        result = eda_review.run(state, client)

    assert result["eda_findings"]["important_columns"]
    assert artifact_store.read_json(pid, "eda_findings.json") is not None


def test_eda_review_raises_on_table_tool_error(isolated_storage):
    client = MockMCPClient({("eda-review-tools", "analyze_eda_tables"): {"error": "no tables"}})
    state = make_state(eda_artifacts={"plots": [], "tables": []})
    from backend.exceptions import ToolError

    with pytest.raises(ToolError):
        eda_review.run(state, client)
