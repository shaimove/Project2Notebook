"""EDA Code Agent.

A *code-writing* agent. It reads the EDA plan + audit + working memory, builds a
single consolidated EDA script, and runs it through the **code-tools** MCP server
(write → validate-no-shell → run → debug). It collects the generated plots and
tables, writes a human-readable ``eda_report.md`` and a machine ``eda_artifacts.json``,
and updates the working context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from backend.agents import code_authoring
from backend.agents.state import DataScientist
from backend.exceptions import PipelineStepError
from backend.mcp_client.client import MCPClient
from backend.services import artifact_store, memory


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    csv_path = (state.get("csv_paths") or [None])[0]
    audit = state.get("data_audit_report") or {}
    target = audit.get("target")
    time_cols = audit.get("time_columns") or []
    numeric_cols = [
        c["name"] for c in audit.get("columns", [])
        if str(c.get("dtype", "")).startswith(("int", "float")) and c.get("inferred_role") == "feature"
    ]

    # Read memory before acting (per the artifact-memory design).
    _ = memory.load(project_id)

    code = code_authoring.build_eda_code(csv_path, target, numeric_cols, time_cols[0] if time_cols else None)
    result = code_authoring.run_code_agent(client, project_id, "eda.py", code)
    if not result.get("ok"):
        if result.get("blocked"):
            detail = result.get("stderr") or "blocked by safety validation"
        else:
            detail = result.get("stderr") or "unknown error"
        raise PipelineStepError(f"EDA code execution failed: {str(detail)[:300]}")

    plots = sorted(set(result.get("plots", [])))
    tables = sorted(set(result.get("tables", [])))
    artifacts = {
        "plots": plots,
        "tables": tables,
        "n_plots": len(plots),
        "code_path": result.get("code_path"),
        "log_path": result.get("log_path"),
        "ok": result.get("ok", False),
        "debug_attempts": result.get("debug_attempts", 0),
        "stderr": result.get("stderr", "") if not result.get("ok") else "",
    }
    artifact_store.write_json(project_id, "eda_artifacts.json", artifacts)

    report = _build_report(state, plots, tables, result)
    artifact_store.write_text(project_id, "eda_report.md", report)

    state["eda_artifacts"] = artifacts
    state["eda_report"] = report

    memory.update(
        project_id,
        phase="Executable EDA",
        data_quality_findings=audit.get("notes", []),
        open_questions=[],
    )
    return state


def _build_report(state: DataScientist, plots: List[str], tables: List[str], result: Dict[str, Any]) -> str:
    audit = state.get("data_audit_report") or {}
    spec = state.get("project_spec") or {}
    lines = ["# EDA Report", ""]
    lines.append(f"- Task: **{spec.get('ml_task_type')}**, target: **{audit.get('target')}**")
    lines.append(f"- Shape: **{audit.get('n_rows')} rows × {audit.get('n_cols')} cols**")
    if audit.get("class_imbalance"):
        lines.append(f"- Class imbalance ratio: **{audit.get('class_imbalance')}**")
    if audit.get("near_empty_columns"):
        lines.append(f"- Near-empty columns: {', '.join(audit['near_empty_columns'])}")
    lines.append("")
    lines.append(f"EDA executed via the code runner (`code/eda.py`), status: "
                 f"**{'ok' if result.get('ok') else 'failed'}**"
                 + (f", debug attempts: {result.get('debug_attempts')}" if result.get('debug_attempts') else "") + ".")
    lines.append("")
    lines.append(f"## Plots ({len(plots)})")
    for p in plots:
        lines.append(f"- `{Path(p).name}`")
    lines.append("")
    lines.append(f"## Tables ({len(tables)})")
    for t in tables:
        lines.append(f"- `{Path(t).name}`")
    if not result.get("ok") and result.get("stderr"):
        lines.append("")
        lines.append("## Execution error")
        lines.append("```\n" + result["stderr"][:1500] + "\n```")
    return "\n".join(lines)
