"""Project Understanding Agent.

Reads the brief + inspects columns to infer the ML problem structure and writes
``project_spec.json``. Uses MCP tools for all execution; optionally enriches
narrative fields with an LLM when configured.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import ValidationError as PydanticValidationError

from backend.agents.state import DataScientist
from backend.agents.contracts import NodeContract
from backend.mcp_client.client import MCPClient
from backend.schemas.data_audit import DatasetSummary, TargetDistribution
from backend.schemas.project_spec import ProjectSpec
from backend.schemas.validation import validate_model
from backend.services import artifact_store, memory
from backend.services.llm import llm

PU = "project-understanding-tools"
DI = "data-inspection-tools"

logger = logging.getLogger(__name__)

CONTRACT = NodeContract(
    requires_state=("project_id", "csv_paths"),
    produces=("project_spec.json", "project_understanding.md"),
    produces_state=("project_spec", "primary_metric"),
)


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    doc_path = state.get("project_document_path") or ""
    csv_path = (state.get("csv_paths") or [None])[0]

    parsed = client.call_tool(PU, "parse_project_document", {"path": doc_path})
    text = parsed.get("text", "")

    columns: List[str] = []
    target_kind = None
    n_classes = None
    if csv_path:
        ds = validate_model(
            DatasetSummary,
            client.call_tool_required(DI, "load_dataset", {"csv_path": csv_path}),
            context="load_dataset",
        )
        columns = ds.columns

    targets = client.call_tool(PU, "identify_targets", {"text": text, "columns": columns}).get("targets", [])
    target = targets[0] if targets else None

    if csv_path and target:
        dist = validate_model(
            TargetDistribution,
            client.call_tool_required(
                DI, "inspect_target_distribution", {"csv_path": csv_path, "target": target}
            ),
            context="inspect_target_distribution",
        )
        if dist.kind == "continuous":
            target_kind = "continuous"
        elif dist.kind == "categorical":
            target_kind = "categorical"
            n_classes = dist.n_classes

    task = client.call_tool(PU, "infer_ml_task", {
        "text": text, "target_kind": target_kind, "n_classes": n_classes,
    }).get("ml_task_type", "unknown")

    business_goal = client.call_tool(PU, "identify_business_goals", {"text": text}).get("business_goal", "")
    metrics = client.call_tool(PU, "suggest_metrics", {"ml_task_type": task})

    time_cols: List[str] = []
    entity_cols: List[str] = []
    if csv_path:
        time_cols = client.call_tool(DI, "detect_time_columns", {"csv_path": csv_path}).get("time_columns", [])
        entity_cols = client.call_tool(DI, "detect_entity_columns", {"csv_path": csv_path}).get("entity_columns", [])

    has_time = bool(time_cols) or ("time" in text.lower() and "leakage" in text.lower())
    split = client.call_tool(PU, "identify_split_strategy", {
        "text": text, "has_time_component": has_time,
        "has_groups": bool(entity_cols), "ml_task_type": task,
    }).get("recommended_split", "random")

    leakage = client.call_tool(PU, "identify_leakage_risks", {"text": text, "columns": columns}).get("leakage_risks", [])

    spec = ProjectSpec(
        business_goal=business_goal,
        ml_task_type=task,
        targets=targets,
        unit_of_prediction=_guess_unit(entity_cols, target),
        unit_of_analysis=entity_cols[0] if entity_cols else "row",
        aggregation="",
        has_time_component=has_time,
        recommended_split=split,
        leakage_risks=leakage,
        primary_metric=metrics.get("primary_metric", ""),
        secondary_metrics=metrics.get("secondary_metrics", []),
        success_criteria=_guess_success(task, metrics.get("primary_metric", "")),
        assumptions=[
            "The provided CSV reflects the production data distribution.",
            "The target column is correctly identified from the brief.",
        ],
        open_questions=[
            "Is there an explicit prediction time / cutoff for each row?",
            "Are there business costs that should weight false positives vs negatives?",
        ],
    )

    spec = _maybe_llm_enrich(spec, text)
    spec = validate_model(ProjectSpec, spec.model_dump(), context="project_spec")
    spec_dict = spec.model_dump()
    artifact_store.write_json(state["project_id"], "project_spec.json", spec_dict)
    artifact_store.write_text(state["project_id"], "project_understanding.md", _render_md(spec_dict))
    state["project_spec"] = spec_dict
    state["primary_metric"] = spec_dict["primary_metric"]

    memory.update(
        state["project_id"],
        phase="Project Understanding",
        project_goal=spec_dict.get("business_goal", ""),
        ml_task_type=spec_dict.get("ml_task_type", ""),
        target_columns=spec_dict.get("targets", []),
        primary_metric=spec_dict.get("primary_metric", ""),
        split_strategy=spec_dict.get("recommended_split", ""),
        leakage_risks=spec_dict.get("leakage_risks", []),
        open_questions=spec_dict.get("open_questions", []),
    )
    return state


def _render_md(spec: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in (items or [])) or "- (none)"

    return "\n".join([
        "# Project Understanding",
        "",
        f"**Business goal:** {spec.get('business_goal') or '(inferred from brief)'}",
        "",
        f"- **ML task type:** {spec.get('ml_task_type')}",
        f"- **Target(s):** {', '.join(spec.get('targets', [])) or '(tbd)'}",
        f"- **Unit of prediction:** {spec.get('unit_of_prediction')}",
        f"- **Unit of analysis:** {spec.get('unit_of_analysis')}",
        f"- **Has time component:** {spec.get('has_time_component')}",
        f"- **Recommended split:** {spec.get('recommended_split')}",
        f"- **Primary metric:** {spec.get('primary_metric')}",
        f"- **Secondary metrics:** {', '.join(spec.get('secondary_metrics', [])) or '(none)'}",
        f"- **Success criteria:** {spec.get('success_criteria')}",
        "",
        "## Leakage risks",
        bullets(spec.get("leakage_risks")),
        "",
        "## Assumptions",
        bullets(spec.get("assumptions")),
        "",
        "## Open questions",
        bullets(spec.get("open_questions")),
    ])


def _guess_unit(entity_cols: List[str], target: str | None) -> str:
    if entity_cols:
        return f"one prediction per {entity_cols[0]}"
    return "one prediction per row"


def _guess_success(task: str, metric: str) -> str:
    if task in ("binary_classification", "multiclass_classification", "anomaly_detection"):
        return f"Beat the dummy baseline and reach a usefully high {metric or 'ROC-AUC'} on held-out data."
    return f"Beat the dummy baseline and achieve a low {metric or 'RMSE'} that supports the business decision."


def _maybe_llm_enrich(spec: ProjectSpec, text: str) -> ProjectSpec:
    if not llm.enabled or not text:
        return spec
    system = (
        "You are a senior data scientist refining a structured project spec. "
        "Keep the given fields unless clearly wrong; you may improve business_goal, "
        "success_criteria, assumptions and open_questions."
    )
    prompt = (
        f"Project brief:\n{text[:4000]}\n\nCurrent spec JSON:\n{spec.model_dump_json()}\n"
        "Return the improved spec as JSON with the same keys."
    )
    data = llm.complete_json(system, prompt, fallback=spec.model_dump())
    try:
        return ProjectSpec.model_validate({**spec.model_dump(), **data})
    except PydanticValidationError:
        logger.warning("LLM returned invalid project spec; keeping deterministic spec")
        return spec
