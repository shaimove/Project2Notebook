"""EDA Planning Agent.

Decides which plots/tables/checks to run and why, based on the audit + spec.
Writes ``eda_plan.json``.
"""
from __future__ import annotations

from typing import List

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.schemas.data_audit import EDACheck, EDAPlan
from backend.services import artifact_store, memory


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    audit = state.get("data_audit_report") or {}
    spec = state.get("project_spec") or {}
    target = audit.get("target")
    numeric_cols = [c["name"] for c in audit.get("columns", []) if c.get("inferred_role") == "feature" and str(c.get("dtype", "")).startswith(("int", "float"))]

    checks: List[EDACheck] = [
        EDACheck(name="target_distribution", kind="plot",
                 rationale="Understand target balance / spread to choose metrics and split.",
                 columns=[target] if target else []),
        EDACheck(name="missingness", kind="plot",
                 rationale="Quantify missing data to plan imputation."),
        EDACheck(name="feature_distributions", kind="plot",
                 rationale="Spot skew, outliers and scaling needs.", columns=numeric_cols),
    ]
    if target:
        checks.append(EDACheck(name="feature_target", kind="plot",
                               rationale="Assess which features separate/relate to the target.",
                               columns=numeric_cols))
    if len(numeric_cols) >= 2:
        checks.append(EDACheck(name="correlation", kind="plot",
                               rationale="Detect multicollinearity / redundant features."))
    checks.append(EDACheck(name="outliers", kind="table",
                           rationale="Identify outliers that may need clipping/robust models."))
    if spec.get("has_time_component") and audit.get("time_columns"):
        checks.append(EDACheck(name="time_series", kind="plot",
                               rationale="Inspect temporal trend/seasonality and confirm time-based split.",
                               columns=audit.get("time_columns", [])[:1] + ([target] if target else [])))

    plan = EDAPlan(
        checks=checks,
        summary=f"Planned {len(checks)} EDA checks tailored to a {spec.get('ml_task_type')} task.",
    )
    d = plan.model_dump()
    pid = state["project_id"]
    _ = memory.load(pid)
    artifact_store.write_json(pid, "eda_plan.json", d)
    md = ["# EDA Plan", "", d.get("summary", ""), "",
          "_This planner only describes the checks; the EDA Code Agent writes and runs the code._", ""]
    for c in d.get("checks", []):
        md.append(f"- **{c['name']}** ({c['kind']}) — {c['rationale']}")
    artifact_store.write_text(pid, "eda_plan.md", "\n".join(md))
    state["eda_plan"] = d
    memory.update(pid, phase="EDA Planning")
    return state
