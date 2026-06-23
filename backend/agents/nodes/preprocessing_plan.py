"""Preprocessing Planner (+ Preprocessing Code Agent hand-off).

The *planner* (this node) decides the leakage-aware preprocessing/split and
writes ``preprocessing_decisions.md`` + ``preprocessing_plan.json`` +
``split_report.json``. It does NOT write executable code itself.

Canonical, leakage-safe execution (split + fit-on-train + transform → prepared
arrays consumed by the modeling tools) runs through the **preprocessing MCP
tools**. In addition, the **Preprocessing Code Agent** authors and runs a
standalone, validated, reproducible ``preprocessing.py`` (used by the final
notebook and as a sanity check), via the code-tools layer.
"""
from __future__ import annotations

from backend.agents import code_authoring
from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.schemas.experiment import PreprocessingPlan, SplitReport
from backend.schemas.validation import validate_model
from backend.services import artifact_store, memory
from backend.services.plotly_viz import generate_split_target_plotly

PP = "preprocessing-tools"


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    csv_path = (state.get("csv_paths") or [None])[0]
    spec = state.get("project_spec") or {}
    audit = state.get("data_audit_report") or {}
    _ = memory.load(project_id)

    profile = {"columns": audit.get("columns", [])}
    plan_raw = client.call_tool_required(
        PP, "create_preprocessing_plan", {
            "profile": profile,
            "spec": spec,
            "eda_findings": state.get("eda_findings") or {},
        }
    )
    plan = validate_model(PreprocessingPlan, plan_raw, context="preprocessing plan").model_dump()

    dq = state.get("data_quality_report") or {}
    modeling_features = dq.get("modeling_features") or state.get("modeling_features") or []
    if modeling_features:
        target_col = (spec.get("targets") or [None])[0]
        plan["keep_columns"] = [c for c in modeling_features if c != target_col]
        plan.setdefault("notes", []).append(
            f"Keep columns aligned with Data Quality selection ({len(plan['keep_columns'])} features)."
        )

    leak = client.call_tool(PP, "check_preprocessing_leakage", {"plan": plan, "spec": spec})
    plan.setdefault("notes", []).append(
        "Leakage check: " + ("no issues" if leak.get("ok") else "; ".join(leak.get("leakage_warnings", [])))
    )

    split_raw = client.call_tool_required(PP, "build_train_valid_test_split", {
        "csv_path": csv_path, "plan": plan, "spec": spec, "project_id": project_id,
    })
    client.call_tool_required(PP, "fit_preprocessor_on_train", {"project_id": project_id})
    shapes = client.call_tool_required(PP, "transform_valid_test", {"project_id": project_id})
    split_raw["feature_count"] = shapes.get("n_features")
    split_report = validate_model(SplitReport, split_raw, context="split report").model_dump()

    target = (spec.get("targets") or [None])[0] or audit.get("target")
    scaling = plan.get("scaling_strategy") or "standard"
    split_viz = generate_split_target_plotly(
        project_id, csv_path, target, split_report, scaling_method=scaling,
    ) if target else {"ok": False}
    if split_viz.get("ok"):
        state["split_plotly_html"] = split_viz.get("html_name")
        state["split_ratios"] = split_viz.get("ratios")

    artifact_store.write_json(project_id, "preprocessing_plan.json", plan)
    artifact_store.write_json(project_id, "split_report.json", split_report)
    artifact_store.write_text(project_id, "preprocessing_decisions.md", _render_md(plan, split_report))

    code = code_authoring.build_preprocessing_code(csv_path, plan, spec)
    code_authoring.run_code_agent(client, project_id, "preprocessing.py", code)

    state["preprocessing_plan"] = plan
    state["split_report"] = split_report

    memory.update(
        project_id,
        phase="Preprocessing",
        split_strategy=split_report.get("strategy", ""),
        selected_features=plan.get("keep_columns", []),
        dropped_features=plan.get("drop_columns", []),
        preprocessing_decisions=[
            f"encoding={plan.get('encoding_strategy')}, scaling={plan.get('scaling_strategy')}",
        ] + plan.get("leakage_mitigations", []),
        leakage_risks=leak.get("leakage_warnings", []),
    )
    return state


def _render_md(plan: dict, split: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in (items or [])) or "- (none)"

    return "\n".join([
        "# Split Data & Scaling Decisions",
        "",
        f"- **Drop columns:** {', '.join(plan.get('drop_columns', [])) or '(none)'}",
        f"- **Keep columns:** {', '.join(plan.get('keep_columns', [])) or '(none)'}",
        f"- **Numeric:** {', '.join(plan.get('numeric_columns', [])) or '(none)'}",
        f"- **Categorical:** {', '.join(plan.get('categorical_columns', [])) or '(none)'}",
        f"- **Encoding:** {plan.get('encoding_strategy')}  |  **Scaling:** {plan.get('scaling_strategy')}",
        f"- **Missing-value strategy:** {plan.get('missing_value_strategy')}",
        "",
        "## Split",
        f"- **Strategy:** {split.get('strategy')} ({split.get('rationale')})",
        f"- **Sizes:** train={split.get('train_rows')}, valid={split.get('valid_rows')}, test={split.get('test_rows')}",
        f"- **Group column:** {split.get('group_column')}  |  **Time column:** {split.get('time_column')}",
        "",
        "## Leakage mitigations",
        bullets(plan.get("leakage_mitigations")),
        "",
        "## Notes",
        bullets(plan.get("notes")),
    ])
