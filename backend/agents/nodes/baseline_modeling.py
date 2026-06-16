"""Baseline Modeling agent (+ Modeling Code Agent hand-off).

Canonical training/evaluation runs through the **modeling MCP tools** on the
prepared leakage-safe splits. This node also writes a human-readable
``modeling_report.md`` and machine ``model_results.json`` (alias of
``baseline_results.json``), and the **Modeling Code Agent** authors + runs a
standalone reproducible ``modeling.py`` (for the notebook / sanity check).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from backend.agents import code_authoring
from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.mcp_client.tool_result import optional_tool_result
from backend.schemas.experiment import ModelComparisonResult, ModelResult
from backend.schemas.validation import parse_tool_results, validate_model
from backend.services import artifact_store, memory

MD = "modeling-tools"

_TOOLS = [
    ("train_dummy_baseline", "dummy"),
    ("train_linear_model", "linear"),
    ("train_tree_model", "tree"),
    ("train_random_forest", "random_forest"),
    ("train_boosting_model", "boosting"),
    ("train_svm_model", "svm"),
]


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    spec = state.get("project_spec") or {}
    _ = memory.load(project_id)

    n_rows = (state.get("data_audit_report") or {}).get("n_rows", 0)
    skip_svm = n_rows > 20000  # SVM scales poorly; skip on large data

    raw_results: List[Dict[str, Any]] = []
    for tool_name, model_name in _TOOLS:
        if model_name == "svm" and skip_svm:
            continue
        res = client.call_tool(MD, tool_name, {"project_id": project_id, "spec": spec})
        if optional_tool_result(res, tool=tool_name, server=MD):
            raw_results.append(res)

    results, skipped = parse_tool_results(ModelResult, raw_results, context="model training")
    comparison_obj = validate_model(
        ModelComparisonResult,
        client.call_tool_required(MD, "compare_models", {"results": results}),
        context="compare_models",
    )
    rows = [row.model_dump() for row in comparison_obj.rows]

    payload = {"results": results, "comparison": comparison_obj.model_dump()}
    artifact_store.write_json(project_id, "baseline_results.json", payload)
    artifact_store.write_json(project_id, "model_results.json", payload)
    if rows:
        pd.DataFrame(rows).to_csv(
            artifact_store.project_artifact_dir(project_id) / "model_comparison.csv", index=False
        )
        pd.DataFrame(rows).to_csv(artifact_store.tables_dir(project_id) / "model_comparison.csv", index=False)

    metric = results[0].get("primary_metric") if results else comparison_obj.primary_metric
    notes: List[str] = []
    if skipped:
        notes.append(f"Skipped {len(skipped)} invalid model result(s): {', '.join(skipped)}.")
    artifact_store.write_text(
        project_id,
        "modeling_report.md",
        _render_md(rows, metric, notes),
    )

    state["baseline_results"] = {"results": results}
    state["model_comparison"] = rows
    best = comparison_obj.best_model
    if results:
        state["primary_metric"] = results[0].get("primary_metric") or comparison_obj.primary_metric
        state["higher_is_better"] = results[0].get("higher_is_better", True)
        best_row = next((r for r in rows if r["model"] == best), None)
        if best_row:
            state["best_validation_score"] = best_row.get("valid_score")
            state["best_pipeline_id"] = best

    code = code_authoring.build_modeling_code(
        (state.get("csv_paths") or [""])[0], state.get("preprocessing_plan") or {}, spec
    )
    code_authoring.run_code_agent(client, project_id, "modeling.py", code)

    memory.update(
        project_id,
        phase="Baseline Modeling",
        model_results=[
            f"{r['model']}: {metric}={round(r['valid_score'], 4) if r.get('valid_score') is not None else 'n/a'}"
            for r in rows
        ],
        best_pipeline=f"{best} (baseline)" if best else "",
    )
    return state


def _render_md(rows: List[Dict[str, Any]], metric: str, notes: List[str]) -> str:
    lines = ["# Baseline Modeling Report", "", f"Primary metric: **{metric}** (validation).", ""]
    if notes:
        lines.extend(notes)
        lines.append("")
    lines.extend([
        "| Model | Family | Valid | Train-Valid gap | Runtime (s) |",
        "|---|---|---|---|---|",
    ])
    for r in rows:
        vs = round(r["valid_score"], 4) if r.get("valid_score") is not None else "—"
        gap = round(r["train_valid_gap"], 4) if r.get("train_valid_gap") is not None else "—"
        lines.append(f"| {r['model']} | {r.get('family')} | {vs} | {gap} | {r.get('runtime_seconds', '—')} |")
    return "\n".join(lines)
