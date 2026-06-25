"""First Conclusion Agent.

Produces an intermediate conclusion after baseline modeling and writes
``first_conclusion.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.contracts import NodeContract
from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.services import artifact_store, memory
from backend.services.llm import llm

CONTRACT = NodeContract(
    requires=("model_results.json", "split_report.json"),
    requires_state=("project_id", "model_comparison", "project_spec"),
    produces=("first_conclusion.md",),
    produces_state=("first_conclusion",),
)


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    _ = memory.load(state["project_id"])  # read memory before acting
    spec = state.get("project_spec") or {}
    audit = state.get("data_audit_report") or {}
    rows = state.get("model_comparison") or []
    metric = state.get("primary_metric") or ""
    best = state.get("best_pipeline_id")
    best_score = state.get("best_validation_score")

    dummy = next((r for r in rows if r["model"] == "dummy"), None)
    best_row = next((r for r in rows if r["model"] == best), None)
    overfit = bool(best_row and best_row.get("train_valid_gap") and abs(best_row["train_valid_gap"]) > 0.15)
    imbalance = audit.get("class_imbalance")

    md: List[str] = ["# First Modeling Conclusion", ""]
    md.append(f"- **Task formulation**: {spec.get('ml_task_type')} predicting `{audit.get('target')}`. "
              "This matches the brief.")
    md.append(f"- **Split strategy**: `{(state.get('split_report') or {}).get('strategy')}` — "
              "appropriate given the data structure and leakage considerations.")
    if imbalance and imbalance > 1.5:
        md.append(f"- **Class imbalance** present (ratio ~{imbalance}); consider class weights / threshold tuning.")
    else:
        md.append("- **Class/target balance** looks acceptable for standard training.")
    md.append(f"- **Best baseline**: `{best}` with validation {metric} = "
              f"{round(best_score, 4) if best_score is not None else 'n/a'}.")
    if dummy is not None:
        md.append(f"- **Dummy baseline** {metric} = {round(dummy.get('valid_score'), 4) if dummy.get('valid_score') is not None else 'n/a'} "
                  "(the model must beat this to be useful).")
    md.append(f"- **Overfitting signs**: {'YES — large train/valid gap' if overfit else 'no strong signs'}.")
    md.append("- **Underfitting signs**: " + ("possible — best model barely beats dummy." if _underfit(rows, dummy, best_row) else "no — models clearly beat the dummy."))
    md.append("")
    md.append("## What to change next")
    md.append("- Improve the strongest families (boosting / random forest) via tuning.")
    if imbalance and imbalance > 1.5:
        md.append("- Address class imbalance with class weights and/or threshold tuning.")
    md.append("- Keep the test set untouched; iterate using validation only.")

    text = "\n".join(md)
    text = _maybe_llm(text, spec, rows, metric)
    artifact_store.write_text(state["project_id"], "first_conclusion.md", text)
    state["first_conclusion"] = text
    memory.update(
        state["project_id"],
        phase="First Conclusion",
        open_questions=["Which model family is most worth improving next?"],
    )
    return state


def _underfit(rows: List[Dict[str, Any]], dummy, best_row) -> bool:
    if not dummy or not best_row:
        return False
    d, b = dummy.get("valid_score"), best_row.get("valid_score")
    if d is None or b is None:
        return False
    return abs(b - d) < 0.02


def _maybe_llm(text: str, spec, rows, metric) -> str:
    if not llm.enabled:
        return text
    extra = llm.complete(
        "You are a senior data scientist. Add 2-3 concise, insightful bullet points.",
        f"Task: {spec.get('ml_task_type')}. Metric: {metric}. Model comparison: {rows}. "
        f"Existing conclusion:\n{text}\nReturn only the extra bullet points.",
        max_tokens=300,
    )
    if extra:
        return text + "\n\n## Additional notes (LLM)\n" + extra.strip()
    return text
