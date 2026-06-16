"""Final Test Evaluation Agent.

Selects the final pipeline from validation results, then evaluates ONCE on the
held-out test set. Writes ``final_test_report.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.schemas.experiment import FinalTestReport, ModelEvaluationResult
from backend.schemas.validation import validate_model
from backend.services import artifact_store, memory

MD = "modeling-tools"


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    _ = memory.load(project_id)  # read memory before acting
    spec = state.get("project_spec") or {}
    metric = state.get("primary_metric") or ""
    higher = bool(state.get("higher_is_better", True))
    best_model = state.get("best_pipeline_id") or "best"

    test_eval = validate_model(
        ModelEvaluationResult,
        client.call_tool_required(MD, "evaluate_model", {
            "project_id": project_id, "spec": spec, "model_name": "best", "split": "test",
        }),
        context="test evaluation",
    )
    valid_eval = validate_model(
        ModelEvaluationResult,
        client.call_tool_required(MD, "evaluate_model", {
            "project_id": project_id, "spec": spec, "model_name": "best", "split": "valid",
        }),
        context="valid evaluation",
    )

    test_metrics = test_eval.metrics
    valid_metrics = valid_eval.metrics
    test_score = test_eval.score
    valid_score = valid_eval.score

    goal_met = _goal_met(metric, test_score, higher)
    gap_note = _gap_note(metric, valid_score, test_score, higher)

    report_obj = FinalTestReport(
        final_model=best_model,
        test_metrics=test_metrics,
        valid_metrics=valid_metrics,
        goal_met=goal_met,
        generalization_notes=[gap_note],
        limitations=[
            "Single train/valid/test split (no cross-validation) for the MVP.",
            "Curated offline prior-art only (web search disabled by default).",
            "Limited hyperparameter search within the controlled iteration budget.",
        ],
        future_directions=[
            "Add cross-validation and a proper hyperparameter search.",
            "Engineer richer temporal/aggregation features and re-audit for leakage.",
            "Calibrate probabilities and tune the decision threshold to the business cost.",
            "Collect more data for minority classes / rare regimes.",
        ],
    )

    md = _render(spec, metric, best_model, valid_score, test_score, test_metrics, goal_met, gap_note)
    artifact_store.write_text(project_id, "final_test_report.md", md)
    obj = report_obj.model_dump()
    artifact_store.write_json(project_id, "final_test_report.json", obj)
    state["final_test_report"] = md
    state["final_test_report_obj"] = obj

    memory.update(
        project_id,
        phase="Final Test Evaluation",
        model_results=[f"FINAL TEST {best_model}: {metric}={round(test_score, 4) if test_score is not None else 'n/a'}"],
        best_pipeline=f"{best_model} — final (test {metric}="
                      f"{round(test_score, 4) if test_score is not None else 'n/a'})",
        open_questions=report_obj.future_directions[:2],
    )
    return state


def _goal_met(metric: str, score, higher: bool) -> bool:
    if score is None:
        return False
    if metric in ("roc_auc", "pr_auc"):
        return score >= 0.7
    if metric in ("f1", "f1_macro", "accuracy", "balanced_accuracy"):
        return score >= 0.6
    if metric == "r2":
        return score >= 0.3
    return True


def _gap_note(metric: str, valid_score, test_score, higher: bool) -> str:
    if valid_score is None or test_score is None:
        return "Could not compare validation and test scores."
    diff = (valid_score - test_score) if higher else (test_score - valid_score)
    if abs(diff) < 0.03:
        return f"Validation and test {metric} are close ({round(valid_score,4)} vs {round(test_score,4)}) — good generalisation."
    return (f"Validation {metric}={round(valid_score,4)} vs test {round(test_score,4)} "
            f"(gap {round(diff,4)}) — watch for {'overfitting' if diff>0 else 'underestimation'}.")


def _render(spec, metric, best_model, valid_score, test_score, test_metrics, goal_met, gap_note) -> str:
    md = ["# Final Test Evaluation", ""]
    md.append(f"- **Selected model (by validation)**: `{best_model}`")
    md.append(f"- **Primary metric**: {metric}")
    md.append(f"- **Validation {metric}**: {round(valid_score,4) if valid_score is not None else 'n/a'}")
    md.append(f"- **Test {metric}**: {round(test_score,4) if test_score is not None else 'n/a'}")
    md.append("")
    md.append("## Full test metrics")
    for k, v in test_metrics.items():
        md.append(f"- {k}: {round(v,4)}")
    md.append("")
    md.append(f"## Goal assessment\n- Project goal met: **{'YES' if goal_met else 'NOT CLEARLY'}**")
    md.append(f"- Success criteria: {spec.get('success_criteria','')}")
    md.append("")
    md.append("## Generalization\n- " + gap_note)
    md.append("")
    md.append("> The test set was used exactly once, after final model selection on validation.")
    return "\n".join(md)
