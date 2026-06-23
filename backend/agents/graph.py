"""Deterministic agent graph orchestrator.

For the MVP we use a small, explicit sequential orchestrator instead of a full
LangGraph runtime. The workflow is deterministic (the only loop is the
controlled iteration loop inside ``iteration_loop``), which keeps runs
reproducible and easy to reason about. The node functions all share the
LangGraph-style ``DataScientist`` state and can be ported to LangGraph directly.

Checkpoints are persisted to SQLite after each step (see ``checkpoint_store``).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.services.checkpoint_store import get_checkpoint_store
from backend.agents.nodes import (
    baseline_modeling,
    data_audit,
    data_quality,
    eda_planning,
    eda_review,
    executable_eda,
    final_test_evaluation,
    first_conclusion,
    iteration_loop,
    leakage_review,
    notebook_author,
    preprocessing_plan,
    prior_art,
    project_understanding,
)

NodeFn = Callable[[DataScientist, MCPClient], DataScientist]

WORKFLOW: List[Tuple[str, NodeFn]] = [
    ("Project Understanding (plan)", project_understanding.run),
    ("Prior Art Search (research)", prior_art.run),
    ("Data Quality Review (review+code)", data_quality.run),
    ("Data Audit (review)", data_audit.run),
    ("EDA Planning (plan)", eda_planning.run),
    ("EDA Code Agent (code)", executable_eda.run),
    ("EDA Review (review)", eda_review.run),
    ("Preprocessing Planner + Code Agent (plan+code)", preprocessing_plan.run),
    ("Modeling + Code Agent (code)", baseline_modeling.run),
    ("First Conclusion (review)", first_conclusion.run),
    ("Iteration + Code Agent (code)", iteration_loop.run),
    ("Leakage Review (review)", leakage_review.run),
    ("Final Test Evaluation (review)", final_test_evaluation.run),
    ("Notebook Author (code)", notebook_author.run),
]

logger = logging.getLogger(__name__)


def run_graph(
    state: DataScientist,
    *,
    run_id: Optional[str] = None,
    start_step: int = 1,
) -> DataScientist:
    store = get_checkpoint_store()
    if run_id is None:
        run_id = store.start_run(
            state["project_id"],
            {
                "enable_prior_art": state.get("enable_prior_art"),
                "max_iterations": state.get("max_iterations"),
                "min_relative_improvement": state.get("min_relative_improvement"),
            },
        )
    state["run_id"] = run_id

    client = MCPClient(on_log=lambda entry: state["tool_calls"].append(entry))

    for idx, (title, fn) in enumerate(WORKFLOW, start=1):
        if idx < start_step:
            continue
        start = time.time()
        logger.info("Pipeline step %d: %s (run %s)", idx, title, run_id)
        try:
            state = fn(state, client)
            detail = _detail_for(title, state)
            _add_timeline(state, idx, title, "completed", detail, start)
            store.save_checkpoint(
                run_id, idx, title, dict(state), status="completed",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - keep the run alive, record the error
            logger.exception("Pipeline step failed: %s", title)
            err = {"step": title, "error": f"{type(exc).__name__}: {exc}"}
            state["errors"].append(err)
            _add_timeline(state, idx, title, "error", err["error"], start)
            store.save_checkpoint(
                run_id, idx, title, dict(state), status="error",
                error=err, duration_ms=int((time.time() - start) * 1000),
            )
    return state


def finish_graph_run(state: DataScientist) -> None:
    run_id = state.get("run_id")
    if not run_id:
        return
    status = "completed_with_errors" if state.get("errors") else "completed"
    get_checkpoint_store().finish_run(run_id, status)


def _add_timeline(state: DataScientist, step: int, title: str, status: str, detail: str, start: float) -> None:
    state["timeline"].append({
        "step": step,
        "title": title,
        "status": status,
        "detail": detail,
        "duration_ms": int((time.time() - start) * 1000),
    })


def _detail_for(title: str, state: DataScientist) -> str:
    spec = state.get("project_spec") or {}
    if title.startswith("Project Understanding"):
        return f"Task: {spec.get('ml_task_type')} | target: {', '.join(spec.get('targets', []))}"
    if title.startswith("Prior Art"):
        pa = state.get("prior_art_report") or {}
        return "enabled" if pa.get("enabled") else (pa.get("message", "disabled"))
    if title.startswith("Data Quality"):
        dq = state.get("data_quality_report") or {}
        n_issues = len(dq.get("issues") or [])
        cleaned = dq.get("cleaned_csv_path") or ""
        suffix = Path(cleaned).name if cleaned else "unchanged"
        return f"{n_issues} issue(s) | output: {suffix}"
    if title.startswith("Data Audit"):
        a = state.get("data_audit_report") or {}
        return f"{a.get('n_rows')} rows x {a.get('n_cols')} cols"
    if title.startswith("EDA Code Agent"):
        ea = state.get("eda_artifacts") or {}
        return f"{len(ea.get('plots', []))} plots, {len(ea.get('tables', []))} tables"
    if title.startswith("EDA Review"):
        ef = state.get("eda_findings") or {}
        return f"{len(ef.get('important_columns', []))} important, {len(ef.get('features_to_drop', []))} drop"
    if title.startswith("Preprocessing"):
        sr = state.get("split_report") or {}
        return f"split={sr.get('strategy')} ({sr.get('train_rows')}/{sr.get('valid_rows')}/{sr.get('test_rows')})"
    if title.startswith("Modeling"):
        cmp = state.get("model_comparison") or []
        return f"{len(cmp)} models compared"
    if title.startswith("Iteration"):
        its = state.get("iteration_reports") or []
        return f"{len(its)} iteration(s)"
    if title.startswith("Final Test"):
        obj = state.get("final_test_report_obj") or {}
        return f"final model: {obj.get('final_model')}"
    if title.startswith("Notebook"):
        return state.get("notebook_path") or ""
    return ""
