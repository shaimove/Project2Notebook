"""Assembles a run result payload (artifacts + their contents) for the API/UI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from backend.agents.state import DataScientist
from backend.services import artifact_store, memory


def assemble_artifacts(state: DataScientist) -> Dict[str, Any]:
    project_id = state["project_id"]
    plots = artifact_store.list_plots(project_id)
    tables = artifact_store.list_tables(project_id)

    def rel_plots(paths):
        return [Path(p).name for p in paths]

    code_dir = artifact_store.code_dir(project_id)
    code_files = sorted(p.name for p in code_dir.glob("*.py")) if code_dir.exists() else []

    artifacts: Dict[str, Any] = {
        "project_spec": state.get("project_spec"),
        "prior_art_report": state.get("prior_art_report"),
        "data_audit_report": state.get("data_audit_report"),
        "eda_plan": state.get("eda_plan"),
        "eda_report": state.get("eda_report"),
        "preprocessing_plan": state.get("preprocessing_plan"),
        "split_report": state.get("split_report"),
        "baseline_results": state.get("baseline_results"),
        "model_comparison": state.get("model_comparison"),
        "first_conclusion": state.get("first_conclusion"),
        "iteration_reports": state.get("iteration_reports"),
        "iteration_summary": state.get("iteration_summary"),
        "leakage_review": state.get("leakage_review"),
        "final_test_report": state.get("final_test_report"),
        "final_test_report_obj": state.get("final_test_report_obj"),
        "notebook_path": state.get("notebook_path"),
        "plots": plots,
        "plot_names": rel_plots(plots),
        "tables": [Path(t).name for t in tables],
        "code_files": code_files,
        "working_context": memory.load_markdown(project_id),
        "artifact_files": artifact_store.list_artifacts(project_id),
    }
    return artifacts


def build_summary(state: DataScientist) -> str:
    spec = state.get("project_spec") or {}
    final = state.get("final_test_report_obj") or {}
    metric = state.get("primary_metric") or ""
    test_score = (final.get("test_metrics") or {}).get(metric)
    parts = [
        f"Task: {spec.get('ml_task_type')} on target '{', '.join(spec.get('targets', []))}'.",
        f"Selected model: {state.get('best_pipeline_id')}.",
    ]
    if test_score is not None:
        parts.append(f"Test {metric} = {round(test_score, 4)}.")
    parts.append(f"Goal met: {'yes' if final.get('goal_met') else 'not clearly'}.")
    if state.get("errors"):
        parts.append(f"{len(state['errors'])} step error(s) recorded.")
    return " ".join(parts)
