"""Deterministic agent graph orchestrator.

For the MVP we use a small, explicit sequential orchestrator instead of a full
LangGraph runtime. The workflow is deterministic (the only loop is the
controlled iteration loop inside ``iteration_loop``), which keeps runs
reproducible and easy to reason about. The node functions all share the
LangGraph-style ``DataScientist`` state and can be ported to LangGraph directly.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Tuple

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.agents.nodes import (
    baseline_modeling,
    data_audit,
    eda_planning,
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

# (timeline title, node function). Order defines the workflow.
# Roles: [plan]/[review] agents describe what's needed and never author code;
# [code] agents author + run Python via the code-tools layer.
WORKFLOW: List[Tuple[str, NodeFn]] = [
    ("Project Understanding (plan)", project_understanding.run),
    ("Prior Art Search (research)", prior_art.run),
    ("Data Audit (review)", data_audit.run),
    ("EDA Planning (plan)", eda_planning.run),
    ("EDA Code Agent (code)", executable_eda.run),
    ("Preprocessing Planner + Code Agent (plan+code)", preprocessing_plan.run),
    ("Modeling + Code Agent (code)", baseline_modeling.run),
    ("First Conclusion (review)", first_conclusion.run),
    ("Iteration + Code Agent (code)", iteration_loop.run),
    ("Leakage Review (review)", leakage_review.run),
    ("Final Test Evaluation (review)", final_test_evaluation.run),
    ("Notebook Author (code)", notebook_author.run),
]

logger = logging.getLogger(__name__)


def run_graph(state: DataScientist) -> DataScientist:
    client = MCPClient(on_log=lambda entry: state["tool_calls"].append(entry))

    for idx, (title, fn) in enumerate(WORKFLOW, start=1):
        start = time.time()
        logger.info("Pipeline step %d: %s", idx, title)
        try:
            state = fn(state, client)
            detail = _detail_for(title, state)
            _add_timeline(state, idx, title, "completed", detail, start)
        except Exception as exc:  # noqa: BLE001 - keep the run alive, record the error
            logger.exception("Pipeline step failed: %s", title)
            state["errors"].append({"step": title, "error": f"{type(exc).__name__}: {exc}"})
            _add_timeline(state, idx, title, "error", f"{type(exc).__name__}: {exc}", start)
    return state


def _add_timeline(state: DataScientist, step: int, title: str, status: str, detail: str, start: float) -> None:
    state["timeline"].append({
        "step": step,
        "title": title,
        "status": status,
        "detail": detail,
        "duration_ms": int((time.time() - start) * 1000),
    })


def _detail_for(title: str, state: DataScientist) -> str:
    """Produce a short human-readable detail for the timeline."""
    spec = state.get("project_spec") or {}
    if title.startswith("Project Understanding"):
        return f"Task: {spec.get('ml_task_type')} | target: {', '.join(spec.get('targets', []))}"
    if title.startswith("Prior Art"):
        pa = state.get("prior_art_report") or {}
        return "enabled" if pa.get("enabled") else (pa.get("message", "disabled"))
    if title.startswith("Data Audit"):
        a = state.get("data_audit_report") or {}
        return f"{a.get('n_rows')} rows x {a.get('n_cols')} cols"
    if title.startswith("EDA Code Agent"):
        ea = state.get("eda_artifacts") or {}
        return f"{len(ea.get('plots', []))} plots, {len(ea.get('tables', []))} tables"
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
