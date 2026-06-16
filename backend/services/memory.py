"""Artifact-based agent memory: ``working_context.md`` (+ JSON sidecar).

A single, evolving project memory that every agent reads before acting and
updates after its phase. It stores *decision summaries, evidence, assumptions,
uncertainties and rejected alternatives* — NOT hidden chain-of-thought.

Storage:
- ``working_context.json`` — structured fields (machine-friendly, merge target).
- ``working_context.md``  — human-readable rendering, regenerated on each update.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.services import artifact_store

_JSON = "working_context.json"
_MD = "working_context.md"

# The canonical set of memory fields. Lists accumulate; scalars get overwritten.
_DEFAULT: Dict[str, Any] = {
    "project_goal": "",
    "ml_task_type": "",
    "target_columns": [],
    "primary_metric": "",
    "split_strategy": "",
    "leakage_risks": [],
    "data_quality_findings": [],
    "selected_features": [],
    "dropped_features": [],
    "preprocessing_decisions": [],
    "model_results": [],          # list of short strings
    "best_pipeline": "",
    "rejected_ideas": [],         # list of short strings
    "iteration_history": [],      # list of short strings
    "open_questions": [],
    "phase_log": [],              # list of "phase: summary"
}

_LIST_FIELDS = {
    "target_columns", "leakage_risks", "data_quality_findings", "selected_features",
    "dropped_features", "preprocessing_decisions", "model_results", "rejected_ideas",
    "iteration_history", "open_questions", "phase_log",
}


def load(project_id: str) -> Dict[str, Any]:
    data = artifact_store.read_json(project_id, _JSON)
    ctx = dict(_DEFAULT)
    if isinstance(data, dict):
        ctx.update(data)
    return ctx


def load_markdown(project_id: str) -> str:
    return artifact_store.read_text(project_id, _MD) or ""


def update(project_id: str, phase: str = "", **fields: Any) -> Dict[str, Any]:
    """Merge ``fields`` into the memory and re-render the Markdown.

    For list fields, items are appended (de-duplicated, order-preserving).
    For scalar fields, values are overwritten when truthy.
    """
    ctx = load(project_id)
    for key, value in fields.items():
        if value is None:
            continue
        if key in _LIST_FIELDS:
            existing = list(ctx.get(key, []))
            new_items = value if isinstance(value, list) else [value]
            for item in new_items:
                if item not in existing:
                    existing.append(item)
            ctx[key] = existing
        else:
            ctx[key] = value
    if phase:
        log = list(ctx.get("phase_log", []))
        entry = f"{phase}"
        if entry not in log:
            log.append(entry)
        ctx["phase_log"] = log

    artifact_store.write_json(project_id, _JSON, ctx)
    artifact_store.write_text(project_id, _MD, _render(ctx))
    return ctx


def _render(ctx: Dict[str, Any]) -> str:
    def bullets(items: List[Any]) -> str:
        items = [str(i) for i in (items or [])]
        return "\n".join(f"- {i}" for i in items) if items else "- _(none yet)_"

    lines = [
        "# Working Context (Agent Memory)",
        "",
        "_Shared, evolving project memory. Updated after each major phase. "
        "Contains decisions, evidence, assumptions, uncertainties and rejected "
        "alternatives — not hidden reasoning._",
        "",
        "## Problem framing",
        f"- **Project goal:** {ctx.get('project_goal') or '_(tbd)_'}",
        f"- **ML task type:** {ctx.get('ml_task_type') or '_(tbd)_'}",
        f"- **Target columns:** {', '.join(ctx.get('target_columns', [])) or '_(tbd)_'}",
        f"- **Primary metric:** {ctx.get('primary_metric') or '_(tbd)_'}",
        f"- **Split strategy:** {ctx.get('split_strategy') or '_(tbd)_'}",
        "",
        "## Leakage risks",
        bullets(ctx.get("leakage_risks", [])),
        "",
        "## Data quality findings",
        bullets(ctx.get("data_quality_findings", [])),
        "",
        "## Features",
        f"**Selected:** {', '.join(ctx.get('selected_features', [])) or '_(tbd)_'}",
        "",
        f"**Dropped:** {', '.join(ctx.get('dropped_features', [])) or '_(tbd)_'}",
        "",
        "## Preprocessing decisions",
        bullets(ctx.get("preprocessing_decisions", [])),
        "",
        "## Model results so far",
        bullets(ctx.get("model_results", [])),
        "",
        f"**Best pipeline so far:** {ctx.get('best_pipeline') or '_(tbd)_'}",
        "",
        "## Iteration history",
        bullets(ctx.get("iteration_history", [])),
        "",
        "## Rejected ideas",
        bullets(ctx.get("rejected_ideas", [])),
        "",
        "## Open questions / uncertainties",
        bullets(ctx.get("open_questions", [])),
        "",
        "## Phase log",
        bullets(ctx.get("phase_log", [])),
        "",
    ]
    return "\n".join(lines)
