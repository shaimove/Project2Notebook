"""Data Quality Review Agent.

Scans raw CSV for column-name, value, and categorical issues, applies a
deterministic remediation plan, and points downstream agents at ``cleaned_*.csv``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.contracts import NodeContract
from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.schemas.data_quality import DataQualityReport
from backend.schemas.validation import validate_model
from backend.services import artifact_store, memory

DQ = "data-quality-tools"
VIZ = "viz-tools"

CONTRACT = NodeContract(
    requires=("project_spec.json",),
    requires_state=("project_id", "csv_paths", "project_spec"),
    produces=("data_quality_report.json", "data_quality.md"),
    produces_state=("data_quality_report", "modeling_features", "excluded_columns"),
)


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    csv_path = (state.get("csv_paths") or [None])[0]
    spec = state.get("project_spec") or {}

    state["original_csv_paths"] = list(state.get("csv_paths") or [])

    scan = client.call_tool_required(DQ, "scan_data_quality", {
        "csv_path": csv_path,
        "spec": spec,
    })
    plan = scan.get("remediation_plan") or {}
    has_actions = _plan_has_actions(plan)

    applied: Dict[str, Any] = {}
    if has_actions:
        applied = client.call_tool_required(DQ, "apply_remediation", {
            "csv_path": csv_path,
            "plan": plan,
            "project_id": project_id,
        })
        cleaned = applied.get("cleaned_csv_path")
        if cleaned:
            state["csv_paths"] = [cleaned]
            state["cleaned_csv_path"] = cleaned

    merged: Dict[str, Any] = {
        **scan,
        "cleaned_csv_path": applied.get("cleaned_csv_path") or scan.get("original_csv_path"),
        "n_rows_after": applied.get("n_rows_after") or scan.get("n_rows_before"),
        "n_cols_after": applied.get("n_cols_after") or scan.get("n_cols_before"),
        "actions_applied": applied.get("actions_applied") or (
            ["No remediation required; data passed quality scan."] if not has_actions else []
        ),
    }
    report = validate_model(DataQualityReport, merged, context="data_quality_report")
    report_dict = report.model_dump()

    # Correct target if data quality found image column misidentified as target
    suggested_target = report_dict.get("target_column")
    spec_targets = (spec.get("targets") or [])
    if suggested_target and spec_targets and suggested_target != spec_targets[0]:
        spec = dict(spec)
        spec["targets"] = [suggested_target]
        notes = list(spec.get("assumptions") or [])
        notes.append(
            f"Target corrected to '{suggested_target}' by Data Quality "
            f"(excluded image/non-label columns)."
        )
        spec["assumptions"] = notes
        if report_dict.get("task_type_hint"):
            spec["ml_task_type"] = report_dict["task_type_hint"]
        state["project_spec"] = spec
        artifact_store.write_json(project_id, "project_spec.json", spec)

    state["modeling_features"] = report_dict.get("modeling_features") or []
    state["excluded_columns"] = report_dict.get("excluded_columns") or []

    artifact_store.write_json(project_id, "data_quality_report.json", report_dict)
    artifact_store.write_text(project_id, "data_quality.md", _render_md(report_dict))

    dq_viz = client.call_tool(VIZ, "generate_data_quality_plotly_html", {
        "project_id": project_id,
        "report": report_dict,
    })
    if dq_viz.get("ok"):
        report_dict["quality_plotly_html"] = dq_viz.get("html_name")
        state["quality_plotly_html"] = dq_viz.get("html_name")
        artifact_store.write_json(project_id, "data_quality_report.json", report_dict)

    state["data_quality_report"] = report_dict

    memory.update(
        project_id,
        phase="Data Quality Review",
        data_quality_findings=[
            f"[{i['severity']}] {i['description']}" for i in report_dict.get("issues", [])[:15]
        ],
        data_quality_summary=report.summary,
        cleaned_csv_path=report_dict.get("cleaned_csv_path"),
        target_column=report_dict.get("target_column"),
        modeling_features=report_dict.get("modeling_features") or [],
        image_columns=report_dict.get("image_columns") or [],
        column_renames=[
            f"{r['original']} → {r['cleaned']}" for r in report_dict.get("column_renames", [])
        ],
    )
    return state


def _plan_has_actions(plan: Dict[str, Any]) -> bool:
    return bool(
        plan.get("column_renames")
        or plan.get("columns_to_drop")
        or plan.get("category_maps")
        or plan.get("coerce_numeric_columns")
        or plan.get("drop_duplicate_rows")
    )


def _render_md(d: Dict[str, Any]) -> str:
    lines = [
        "# Data Quality Report",
        "",
        d.get("summary", ""),
        "",
        f"- **Original:** `{d.get('original_csv_path', '')}`",
        f"- **Cleaned:** `{d.get('cleaned_csv_path', '')}`",
        f"- **Shape:** {d.get('n_rows_before')}×{d.get('n_cols_before')} → "
        f"{d.get('n_rows_after')}×{d.get('n_cols_after')}",
        "",
    ]
    if d.get("column_renames"):
        lines += ["## Column Renames"]
        for r in d["column_renames"]:
            lines.append(f"- `{r['original']}` → `{r['cleaned']}`")
        lines.append("")
    if d.get("image_columns"):
        lines += ["## Image Columns (Excluded From Tabular Models)"]
        lines += [f"- {c}" for c in d["image_columns"]] + [""]
    if d.get("feature_engineering_notes"):
        lines += ["## Feature Engineering Notes"] + [f"- {n}" for n in d["feature_engineering_notes"]] + [""]
    if d.get("column_profiles"):
        lines += ["## Column Profiles", ""]
        lines += [
            "| Column | Type | Role | Missing | Unique | Description |",
            "|--------|------|------|---------|--------|-------------|",
        ]
        for p in d["column_profiles"]:
            lines.append(
                f"| {p['name']} | {p['column_type']} | {p['role']} | "
                f"{p['pct_missing']:.1%} | {p['n_unique']} | {p['description']} |"
            )
        lines.append("")
    if d.get("modeling_features"):
        lines += ["## Selected Modeling Features", ", ".join(d["modeling_features"]), ""]
    if d.get("bad_row_examples"):
        lines.append("## Bad Row Examples")
        for ex in d["bad_row_examples"]:
            lines.append(f"- **{ex['issue_type']}**: {ex['description']}")
        lines.append("")
    if d.get("actions_applied"):
        lines += ["## Cleaning Actions Applied"] + [f"- {a}" for a in d["actions_applied"]] + [""]
    if d.get("issues"):
        lines.append("## Data Issues")
        for issue in d["issues"][:30]:
            col = issue.get("column")
            prefix = f"**{col}** — " if col else ""
            lines.append(
                f"- [{issue.get('severity', 'info')}] {prefix}{issue.get('description', '')}"
            )
        lines.append("")
    return "\n".join(lines)
