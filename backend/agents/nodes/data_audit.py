"""Data Audit Agent.

Profiles the dataset and compares it against the project understanding. Writes
``data_audit_report.json``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.contracts import NodeContract
from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.schemas.data_audit import ColumnProfile, DataAuditReport, TargetDistribution
from backend.schemas.validation import parse_models, validate_model
from backend.services import artifact_store, memory

DI = "data-inspection-tools"
VIZ = "viz-tools"

CONTRACT = NodeContract(
    requires=("project_spec.json",),
    requires_state=("project_id", "csv_paths", "project_spec"),
    produces=("data_audit_report.json", "data_audit.md"),
    produces_state=("data_audit_report",),
)


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    csv_path = (state.get("csv_paths") or [None])[0]
    spec = state.get("project_spec") or {}
    targets = spec.get("targets") or []
    target = targets[0] if targets else None

    profile = client.call_tool_required(DI, "profile_dataset", {"csv_path": csv_path})
    missing = client.call_tool(DI, "summarize_missing_values", {"csv_path": csv_path})
    invalid = client.call_tool(DI, "detect_invalid_values", {"csv_path": csv_path})
    dups = client.call_tool(DI, "detect_duplicates", {"csv_path": csv_path})
    time_cols = client.call_tool(DI, "detect_time_columns", {"csv_path": csv_path}).get("time_columns", [])
    entity = client.call_tool(DI, "detect_entity_columns", {"csv_path": csv_path})

    columns, skipped = parse_models(
        ColumnProfile,
        profile.get("columns", []),
        context="ColumnProfile",
        on_item=lambda col: {**col, "inferred_role": _infer_role(col, target, time_cols, entity)},
    )
    high_card: List[str] = []
    constant: List[str] = []
    for col in columns:
        if col.is_constant:
            constant.append(col.name)
        if (col.cardinality or 0) > 50:
            high_card.append(col.name)

    target_dist: Dict[str, Any] = {}
    class_imbalance = None
    if target:
        dist = validate_model(
            TargetDistribution,
            client.call_tool_required(
                DI, "inspect_target_distribution", {"csv_path": csv_path, "target": target}
            ),
            context="inspect_target_distribution",
        )
        target_dist = dist.model_dump()
        class_imbalance = dist.imbalance_ratio

    leakage_prone = _leakage_prone_columns(profile, spec)
    notes = _notes(invalid, dups, class_imbalance)
    if skipped:
        notes.append(f"Skipped {len(skipped)} column(s) with invalid profile data: {', '.join(skipped)}.")

    report = DataAuditReport(
        n_rows=profile.get("n_rows", 0),
        n_cols=profile.get("n_cols", 0),
        columns=columns,
        n_duplicate_rows=dups.get("n_duplicate_rows", 0),
        target=target,
        target_distribution=target_dist,
        class_imbalance=class_imbalance,
        time_columns=time_cols,
        entity_columns=entity.get("entity_columns", []),
        leakage_prone_columns=leakage_prone,
        near_empty_columns=missing.get("near_empty_columns", []),
        constant_columns=constant,
        high_cardinality_columns=high_card,
        description_vs_data_mismatches=[],
        notes=notes,
    )
    d = report.model_dump()
    pid = state["project_id"]
    _ = memory.load(pid)
    artifact_store.write_json(pid, "data_audit_report.json", d)
    artifact_store.write_text(pid, "data_audit.md", _render_md(d))
    state["data_audit_report"] = d

    audit_viz = client.call_tool(VIZ, "generate_audit_missingness_plotly_html", {
        "project_id": pid,
        "audit": d,
    })
    if audit_viz.get("ok"):
        state["audit_plotly_html"] = audit_viz.get("html_name")

    findings = list(d.get("notes", []))
    if d.get("leakage_prone_columns"):
        findings.append("Leakage-prone columns: " + ", ".join(d["leakage_prone_columns"]))
    memory.update(
        pid,
        phase="Data Audit",
        data_quality_findings=findings,
        leakage_risks=[f"column '{c}' may carry future/outcome info" for c in d.get("leakage_prone_columns", [])],
    )
    return state


def _render_md(d: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in (items or [])) or "- (none)"

    return "\n".join([
        "# Data Audit",
        "",
        f"- **Shape:** {d.get('n_rows')} rows × {d.get('n_cols')} cols",
        f"- **Duplicate rows:** {d.get('n_duplicate_rows')}",
        f"- **Target:** {d.get('target')}",
        f"- **Class imbalance ratio:** {d.get('class_imbalance')}",
        f"- **Time columns:** {', '.join(d.get('time_columns', [])) or '(none)'}",
        f"- **Entity/ID columns:** {', '.join(d.get('entity_columns', [])) or '(none)'}",
        f"- **Near-empty columns:** {', '.join(d.get('near_empty_columns', [])) or '(none)'}",
        f"- **Constant columns:** {', '.join(d.get('constant_columns', [])) or '(none)'}",
        f"- **High-cardinality columns:** {', '.join(d.get('high_cardinality_columns', [])) or '(none)'}",
        "",
        "## Leakage-prone columns",
        bullets(d.get("leakage_prone_columns")),
        "",
        "## Notes",
        bullets(d.get("notes")),
    ])


def _infer_role(col: Dict[str, Any], target, time_cols, entity) -> str:
    name = col["name"]
    if target and name == target:
        return "target"
    if name in (time_cols or []):
        return "time"
    if name in (entity.get("entity_columns", []) or []) or name in (entity.get("unique_id_like", []) or []):
        return "id"
    if col.get("is_constant"):
        return "drop"
    return "feature"


def _leakage_prone_columns(profile: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    risky: List[str] = []
    for col in profile.get("columns", []):
        c = col["name"].lower()
        if any(t in c for t in ("future", "next", "post_", "_after", "outcome")):
            risky.append(col["name"])
    return risky


def _notes(invalid, dups, imbalance) -> List[str]:
    notes: List[str] = []
    if dups.get("n_duplicate_rows"):
        notes.append(f"{dups['n_duplicate_rows']} duplicate rows detected.")
    issues = invalid.get("invalid_value_issues", {})
    if issues:
        notes.append(f"Invalid-value issues in: {', '.join(issues.keys())}.")
    if imbalance and imbalance > 1.5:
        notes.append(f"Target appears imbalanced (ratio ~{imbalance}).")
    return notes
