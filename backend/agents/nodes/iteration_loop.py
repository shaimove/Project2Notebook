"""Iterative Improvement Loop.

Controlled loop (max 3 iterations). Each iteration:
- asks the experiment server for a concrete, supported next action
- executes it via the modeling tools (validation only — test stays untouched)
- compares to the current best using *relative improvement*
- accepts/rejects and decides whether to stop

Stopping rules: max iterations reached, relative improvement <= threshold,
suspected overfitting, or no supported hypothesis.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.agents import code_authoring
from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.mcp_client.tool_result import optional_tool_result
from backend.schemas.experiment import (
    IterationReport,
    IterationStopDecision,
    IterationSuggestion,
    ModelResult,
)
from backend.schemas.validation import try_validate_model, validate_model
from backend.services import artifact_store, memory
from mcp_servers._mlcommon import prepared_dir

MD = "modeling-tools"
EX = "experiment-tools"

_TOOL_FOR = {
    "dummy": "train_dummy_baseline",
    "linear": "train_linear_model",
    "tree": "train_tree_model",
    "random_forest": "train_random_forest",
    "boosting": "train_boosting_model",
    "svm": "train_svm_model",
}


def _models_dir(project_id: str) -> Path:
    return prepared_dir(project_id) / "models"


def _promote_best(project_id: str, model_name: str) -> None:
    src = _models_dir(project_id) / f"{model_name}.joblib"
    if src.exists():
        shutil.copyfile(src, _models_dir(project_id) / "best.joblib")


def _relative_improvement(new: float, best: float, higher_is_better: bool) -> float:
    if best is None or new is None:
        return 0.0
    denom = abs(best) if abs(best) > 1e-12 else 1e-12
    return (new - best) / denom if higher_is_better else (best - new) / denom


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    _ = memory.load(project_id)  # read memory before acting
    spec = state.get("project_spec") or {}
    prior_art = state.get("prior_art_report") or {}
    audit = state.get("data_audit_report") or {}
    higher = bool(state.get("higher_is_better", True))
    max_iter = min(int(state.get("max_iterations", 3)), 3)
    min_rel = float(state.get("min_relative_improvement", 0.05))

    best_score: Optional[float] = state.get("best_validation_score")
    best_model: Optional[str] = state.get("best_pipeline_id")
    best_family: Optional[str] = state.get("best_pipeline_id")
    best_params: Dict[str, Any] = {}
    if best_model:
        _promote_best(project_id, best_model)  # snapshot baseline best

    reports: List[Dict[str, Any]] = []
    stop_reason = ""

    for i in range(1, max_iter + 1):
        suggestion = validate_model(
            IterationSuggestion,
            client.call_tool(EX, "suggest_next_iteration", {
                "spec": spec, "prior_art": prior_art, "data_audit": audit,
                "best_result": {"model": best_model, "score": best_score}, "iteration": i,
            }),
            context="iteration suggestion",
        )
        if not suggestion.supported or suggestion.action == "stop":
            stop_reason = "No further EDA/prior-art-supported change available."
            break

        model_name = suggestion.model_name
        tool = _TOOL_FOR.get(model_name, "train_boosting_model")
        result = client.call_tool(MD, tool, {
            "project_id": project_id, "spec": spec, "params": suggestion.params,
        })
        if not optional_tool_result(result, tool=tool, server=MD):
            reports.append(IterationReport(
                iteration=i,
                hypothesis=suggestion.hypothesis,
                motivation=suggestion.motivation,
                change_description=f"Trained {model_name} with params {suggestion.params}.",
                model_name=model_name,
                valid_score=None,
                previous_best=best_score,
                relative_improvement=None,
                accepted=False,
                decision_reason="Rejected: model training failed.",
                train_valid_gap=None,
                metrics={},
            ).model_dump())
            continue

        train_result = try_validate_model(ModelResult, result, context=f"iteration {i} training")
        if train_result is None:
            reports.append(IterationReport(
                iteration=i,
                hypothesis=suggestion.hypothesis,
                motivation=suggestion.motivation,
                change_description=f"Trained {model_name} with params {suggestion.params}.",
                model_name=model_name,
                valid_score=None,
                previous_best=best_score,
                relative_improvement=None,
                accepted=False,
                decision_reason="Rejected: invalid model output.",
                train_valid_gap=None,
                metrics={},
            ).model_dump())
            continue

        new_score = train_result.valid_score
        gap = train_result.train_valid_gap
        overfit = bool(gap is not None and abs(gap) > 0.2)
        rel = _relative_improvement(new_score, best_score, higher) if best_score is not None else 1.0
        accepted = (rel > min_rel) and (not overfit)

        report = IterationReport(
            iteration=i,
            hypothesis=suggestion.hypothesis,
            motivation=suggestion.motivation,
            change_description=f"Trained {model_name} with params {suggestion.params}.",
            model_name=model_name,
            valid_score=new_score,
            previous_best=best_score,
            relative_improvement=round(rel, 4) if rel is not None else None,
            accepted=accepted,
            decision_reason=("Accepted: improvement above threshold." if accepted
                             else ("Rejected: suspected overfitting." if overfit
                                   else f"Rejected: improvement {rel:.1%} <= {min_rel:.0%}.")),
            train_valid_gap=gap,
            metrics=train_result.valid_metrics,
        )
        reports.append(report.model_dump())
        artifact_store.write_json(project_id, f"iteration_{i}_report.json", report.model_dump())

        snippet = code_authoring.build_iteration_code(
            i, model_name, suggestion.params, suggestion.hypothesis
        )
        code_authoring.run_code_agent(client, project_id, f"iteration_{i}.py", snippet, max_debug=0)

        if accepted:
            best_score = new_score
            best_model = f"{model_name} (iter {i})"
            best_family = model_name
            best_params = suggestion.params
            _promote_best(project_id, model_name)

        decision = validate_model(
            IterationStopDecision,
            client.call_tool(EX, "stop_iteration_decision", {
                "iteration": i, "max_iterations": max_iter,
                "relative_improvement": rel, "min_relative_improvement": min_rel,
                "overfit_suspected": overfit, "hypothesis_supported": True,
            }),
            context="iteration stop decision",
        )
        if decision.stop:
            stop_reason = decision.reason
            break

    state["iteration_reports"] = reports
    state["best_validation_score"] = best_score
    state["best_pipeline_id"] = best_model
    state["best_model_family"] = best_family
    state["best_params"] = best_params
    summary = _summary(reports, best_model, best_score, state.get("primary_metric"), stop_reason)
    artifact_store.write_text(project_id, "iteration_summary.md", summary)
    artifact_store.write_text(project_id, "iteration_report.md", summary)
    artifact_store.write_json(project_id, "iteration_result.json", {
        "iterations": reports,
        "best_pipeline": best_model,
        "best_validation_score": best_score,
        "stop_reason": stop_reason,
    })
    state["iteration_summary"] = summary

    memory.update(
        project_id,
        phase="Iterative Improvement Loop",
        iteration_history=[
            f"iter {r['iteration']} ({r['model_name']}): "
            f"{'accepted' if r['accepted'] else 'rejected'} — {r['decision_reason']}"
            for r in reports
        ],
        rejected_ideas=[
            f"iter {r['iteration']} {r['model_name']}: {r['hypothesis']}"
            for r in reports if not r["accepted"]
        ],
        best_pipeline=f"{best_model} (valid {state.get('primary_metric')}="
                      f"{round(best_score, 4) if best_score is not None else 'n/a'})",
    )
    return state


def _summary(reports, best_model, best_score, metric, stop_reason) -> str:
    md = ["# Iteration Summary", ""]
    if not reports:
        md.append("No iterations were run (no supported improvement hypothesis).")
    for r in reports:
        md.append(f"## Iteration {r['iteration']}: {r['model_name']}")
        md.append(f"- Hypothesis: {r['hypothesis']}")
        md.append(f"- Motivation: {r['motivation']}")
        md.append(f"- Validation {metric}: {round(r['valid_score'], 4) if r['valid_score'] is not None else 'n/a'} "
                  f"(prev best {round(r['previous_best'], 4) if r['previous_best'] is not None else 'n/a'}, "
                  f"rel. improvement {r['relative_improvement']})")
        md.append(f"- Decision: **{'ACCEPTED' if r['accepted'] else 'REJECTED'}** — {r['decision_reason']}")
        md.append("")
    md.append(f"**Stopped:** {stop_reason or 'completed all planned iterations.'}")
    md.append(f"**Selected pipeline (by validation):** `{best_model}` with {metric} = "
              f"{round(best_score, 4) if best_score is not None else 'n/a'}.")
    return "\n".join(md)
