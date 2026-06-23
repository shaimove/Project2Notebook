"""Tests for data quality MCP tools and agent."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.agents.nodes import data_quality
from backend.services import artifact_store, file_store
from mcp_servers.data_quality_server import apply_remediation, scan_data_quality
from tests.conftest import MockMCPClient, make_state


def _messy_csv(project_id: str) -> Path:
    content = (
        "Unnamed: 0,Target Column,country,mixed_num\n"
        "0,1,USA,12\n"
        "1,1,\" US \",12.0\n"
        "2,0,U.S.A.,twelve\n"
        "2,0,U.S.A.,twelve\n"
    )
    return file_store.save_bytes(project_id, "messy.csv", content.encode("utf-8"))


def test_scan_drops_columns_above_missing_threshold(isolated_storage):
    pid = "dq-miss-threshold"
    content = "target,region,ok\n1,EU,10\n0,,20\n0,,30\n1,APAC,40\n"
    csv_path = file_store.save_bytes(pid, "sparse.csv", content.encode("utf-8"))
    result = scan_data_quality({"csv_path": str(csv_path), "spec": {"targets": ["target"]}})
    by_name = {p["name"]: p for p in result["column_profiles"]}
    assert by_name["region"]["role"] == "drop"
    assert "region" not in result["modeling_features"]
    assert "region" in result["remediation_plan"]["columns_to_drop"] or "region" in (
        result.get("excluded_columns") or []
    )


def test_scan_data_quality_detects_issues(isolated_storage):
    pid = "dq-test-1"
    csv_path = _messy_csv(pid)
    result = scan_data_quality({"csv_path": str(csv_path), "spec": {"targets": ["Target Column"]}})
    issue_types = {i["issue_type"] for i in result["issues"]}
    assert "bad_column_name" in issue_types or "unnamed_column" in issue_types
    assert "inconsistent_categories" in issue_types
    plan = result["remediation_plan"]
    assert plan["columns_to_drop"] or plan["drop_duplicate_rows"] or plan["category_maps"]
    assert len(result["column_profiles"]) == result["n_cols_before"]
    assert "modeling_features" in result
    assert result["target_column"] == "Target Column"


def test_apply_remediation_writes_cleaned_csv(isolated_storage):
    pid = "dq-test-2"
    csv_path = _messy_csv(pid)
    scan = scan_data_quality({"csv_path": str(csv_path), "spec": {"targets": ["Target Column"]}})
    applied = apply_remediation({
        "csv_path": str(csv_path),
        "plan": scan["remediation_plan"],
        "project_id": pid,
    })
    cleaned = Path(applied["cleaned_csv_path"])
    assert cleaned.exists()
    assert applied["n_rows_after"] < scan["n_rows_before"]
    text = cleaned.read_text(encoding="utf-8")
    assert "unnamed" not in text.lower()
    assert "country" in text.lower()


def test_data_quality_agent_runs(isolated_storage):
    pid = "dq-test-3"
    csv_path = _messy_csv(pid)

    def responder(server, tool, args):
        if tool == "scan_data_quality":
            return scan_data_quality(args)
        if tool == "apply_remediation":
            return apply_remediation(args)
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_id=pid,
        csv_paths=[str(csv_path)],
        project_spec={"targets": ["Target Column"], "ml_task_type": "binary_classification"},
    )
    result = data_quality.run(state, client)

    assert result["data_quality_report"]["cleaned_csv_path"]
    assert result["csv_paths"][0].endswith("cleaned_messy.csv")
    assert artifact_store.read_json(pid, "data_quality_report.json") is not None


def test_data_quality_raises_on_scan_error(isolated_storage):
    client = MockMCPClient({("data-quality-tools", "scan_data_quality"): {"error": "bad csv"}})
    state = make_state(csv_paths=["/missing.csv"])
    from backend.exceptions import ToolError

    with pytest.raises(ToolError):
        data_quality.run(state, client)
