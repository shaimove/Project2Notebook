"""EDA Review Agent.

Reads EDA plots (LLM vision when configured) and tables (deterministic MCP tools),
produces ``eda_findings.json`` with feature engineering/selection recommendations,
and feeds preprocessing + working memory.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.schemas.eda_findings import EDAFindingsReport
from backend.schemas.validation import validate_model
from backend.services import artifact_store, memory
from backend.services.llm import llm
from backend.services.plotly_viz import generate_eda_plotly

logger = logging.getLogger(__name__)

ER = "eda-review-tools"


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    project_id = state["project_id"]
    csv_path = (state.get("csv_paths") or [None])[0]
    audit = state.get("data_audit_report") or {}
    spec = state.get("project_spec") or {}
    dq = state.get("data_quality_report") or {}
    features = dq.get("modeling_features") or state.get("modeling_features") or []
    target = (spec.get("targets") or [None])[0] or audit.get("target") or dq.get("target_column")
    eda_artifacts = state.get("eda_artifacts") or {}
    plot_names = [
        __import__("pathlib").Path(p).name
        for p in (eda_artifacts.get("plots") or [])
    ]

    plotly_result = generate_eda_plotly(
        project_id,
        csv_path,
        features,
        target,
        spec.get("ml_task_type", ""),
    )
    if plotly_result.get("ok"):
        state["eda_plotly_html"] = plotly_result.get("html_name")
        state["eda_plotly_conclusions"] = plotly_result.get("conclusions") or []

    plotly_conclusions = plotly_result.get("conclusions") or []
    table_analysis = client.call_tool_required(ER, "analyze_eda_tables", {
        "project_id": project_id,
        "data_audit": audit,
        "spec": spec,
    })
    inventory = client.call_tool(ER, "list_eda_plot_inventory", {
        "project_id": project_id,
        "plot_names": plot_names,
    }).get("plots", [])

    plot_reviews = _review_plots_with_llm(inventory, audit, spec, table_analysis)
    merged = _merge_findings(table_analysis, plot_reviews, audit, spec)

    if llm.enabled and plot_reviews:
        llm_enriched = _llm_synthesize_findings(merged, audit, spec, plotly_conclusions)
        merged = _merge_dicts(merged, llm_enriched)

    report = validate_model(EDAFindingsReport, merged, context="eda_findings")
    report_dict = report.model_dump()

    if plotly_result.get("ok") and plotly_result.get("conclusions"):
        plotly_summary = "Plotly EDA conclusions:\n" + "\n".join(
            f"- {c}" for c in plotly_result["conclusions"][:10]
        )
        report_dict["summary"] = (report_dict.get("summary", "") + " " + plotly_summary).strip()
        open_q = list(report_dict.get("open_questions") or [])
        if dq.get("image_columns"):
            open_q.append(
                "Image columns excluded from tabular EDA — consider dedicated image feature extraction."
            )
        report_dict["open_questions"] = open_q

    artifact_store.write_json(project_id, "eda_findings.json", report_dict)
    artifact_store.write_text(project_id, "eda_findings.md", _render_md(report_dict))

    state["eda_findings"] = report_dict
    state["eda_findings_report"] = report_dict

    memory.update(
        project_id,
        phase="EDA Review",
        eda_findings_summary=report.summary,
        feature_insights=[
            f"{i['feature']}: {i['recommended_action']} — {i['evidence']}"
            for i in report_dict.get("feature_insights", [])[:12]
        ],
        important_columns=report.important_columns,
        selected_features=report.important_columns,
        dropped_features=report.features_to_drop,
        multicollinear_groups=[", ".join(g) for g in report.multicollinear_groups],
        features_to_engineer=[
            f"{e['base_column']} -> {e['transform']}: {e.get('rationale', '')}"
            for e in report_dict.get("features_to_engineer", [])
        ],
        preprocessing_decisions=report.preprocessing_implications,
        open_questions=report.open_questions,
    )
    return state


def _review_plots_with_llm(
    inventory: List[Dict[str, Any]],
    audit: Dict[str, Any],
    spec: Dict[str, Any],
    table_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    if not inventory:
        return reviews

    system = (
        "You are a senior data scientist reviewing EDA plots for a tabular ML pipeline. "
        "Describe patterns relevant to feature engineering, feature selection, leakage, and modeling. "
        "Features were already chosen in Data Quality; scaling and train/val/test split happen later. "
        "Be concise and cite feature names you see."
    )
    for item in inventory:
        path = item.get("path", "")
        plot_name = item.get("plot_name", "")
        plot_type = item.get("plot_type", "")
        prompt = (
            f"Project task: {spec.get('ml_task_type')} | target: {audit.get('target')}\n"
            f"Plot type: {plot_type}\n"
            f"Known table findings (do not contradict): {table_analysis.get('important_columns', [])}\n"
            "Return JSON with keys: summary (string), key_observations (list of strings), "
            "features_mentioned (list of column names), "
            "recommended_actions (list of strings: keep/drop/engineer/watch)."
        )
        if llm.enabled:
            text = llm.complete_with_images(system, prompt, [path], max_tokens=900)
            parsed = _parse_plot_json(text) if text else {}
            reviews.append({
                "plot_name": plot_name,
                "plot_type": plot_type,
                "summary": parsed.get("summary") or (text or "")[:500],
                "key_observations": parsed.get("key_observations") or [],
                "features_mentioned": parsed.get("features_mentioned") or [],
                "reviewed_by": "llm_vision" if text else "skipped",
            })
        else:
            reviews.append({
                "plot_name": plot_name,
                "plot_type": plot_type,
                "summary": item.get("description", ""),
                "key_observations": [],
                "features_mentioned": [],
                "reviewed_by": "heuristic",
            })
    return reviews


def _parse_plot_json(text: str) -> Dict[str, Any]:
    from backend.services.llm import _extract_json

    parsed = _extract_json(text or "")
    return parsed if isinstance(parsed, dict) else {}


def _llm_synthesize_findings(
    merged: Dict[str, Any],
    audit: Dict[str, Any],
    spec: Dict[str, Any],
    plotly_conclusions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    fallback = {
        "summary": merged.get("summary", ""),
        "open_questions": merged.get("open_questions", []),
    }
    plotly_block = ""
    if plotly_conclusions:
        plotly_block = (
            "\nPlotly per-feature conclusions (from interactive EDA HTML):\n"
            + "\n".join(f"- {c}" for c in plotly_conclusions[:15])
        )
    prompt = (
        f"Task: {spec.get('ml_task_type')} target={audit.get('target')}\n"
        f"Table analysis: {merged}\n"
        f"Plot reviews: {merged.get('plot_reviews', [])}\n"
        f"{plotly_block}\n"
        "Produce a consolidated JSON with keys: summary, important_columns (list), "
        "features_to_drop (list), features_to_engineer (list of "
        "{base_column, transform, rationale, priority}), features_to_watch (list), "
        "preprocessing_implications (list), open_questions (list). "
        "Reference Plotly conclusions when they suggest drops or new features. "
        "Never drop the target or leakage-prone columns without noting leakage."
    )
    return llm.complete_json(
        "You consolidate EDA findings for a tabular ML pipeline.",
        prompt,
        fallback=fallback,
    )


def _merge_findings(
    table_analysis: Dict[str, Any],
    plot_reviews: List[Dict[str, Any]],
    audit: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    target = audit.get("target")
    leakage = set(audit.get("leakage_prone_columns") or [])

    important = list(table_analysis.get("important_columns") or [])
    to_drop = list(table_analysis.get("features_to_drop") or [])
    to_watch = list(table_analysis.get("features_to_watch") or [])
    to_engineer = list(table_analysis.get("features_to_engineer") or [])
    open_q: List[str] = []

    for review in plot_reviews:
        for feat in review.get("features_mentioned") or []:
            if feat and feat != target and feat not in important:
                important.append(feat)
        for obs in review.get("key_observations") or []:
            if "drop" in obs.lower() and review.get("features_mentioned"):
                for feat in review["features_mentioned"]:
                    if feat not in leakage and feat != target:
                        to_drop.append(feat)

    # Never drop target or explicit leakage columns without keeping in watch
    to_drop = [c for c in _uniq(to_drop) if c != target]
    for col in leakage:
        if col in to_drop:
            to_watch.append(col)

    summary_parts = [
        f"Reviewed EDA for {spec.get('ml_task_type', 'unknown')} task.",
        f"{len(table_analysis.get('feature_insights', []))} table-derived insights.",
        f"{len(plot_reviews)} plot(s) reviewed.",
    ]
    if important:
        summary_parts.append(f"Important columns: {', '.join(_uniq(important)[:8])}.")

    return {
        "summary": " ".join(summary_parts),
        "target_insights": table_analysis.get("target_insights") or [],
        "feature_insights": table_analysis.get("feature_insights") or [],
        "important_columns": _uniq(important),
        "features_to_drop": _uniq(to_drop),
        "features_to_engineer": to_engineer,
        "features_to_watch": _uniq(to_watch),
        "multicollinear_groups": table_analysis.get("multicollinear_groups") or [],
        "preprocessing_implications": table_analysis.get("preprocessing_implications") or [],
        "plot_reviews": plot_reviews,
        "open_questions": open_q,
        "analysis_sources": (table_analysis.get("analysis_sources") or []) + ["plot_review"],
    }


def _merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value
    return out


def _uniq(items: List[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _render_md(d: Dict[str, Any]) -> str:
    lines = ["# EDA Findings", "", d.get("summary", ""), ""]
    if d.get("target_insights"):
        lines += ["## Target insights"] + [f"- {t}" for t in d["target_insights"]] + [""]
    if d.get("important_columns"):
        lines += ["## Important columns", ", ".join(d["important_columns"]), ""]
    if d.get("features_to_drop"):
        lines += ["## Recommended drops"] + [f"- {c}" for c in d["features_to_drop"]] + [""]
    if d.get("features_to_engineer"):
        lines.append("## Feature engineering ideas")
        for e in d["features_to_engineer"]:
            lines.append(f"- **{e.get('base_column')}** → {e.get('transform')}: {e.get('rationale', '')}")
        lines.append("")
    if d.get("plot_reviews"):
        lines.append("## Plot reviews")
        for p in d["plot_reviews"]:
            lines.append(f"### {p.get('plot_name')} ({p.get('reviewed_by')})")
            if p.get("summary"):
                lines.append(p["summary"])
            for obs in p.get("key_observations") or []:
                lines.append(f"- {obs}")
        lines.append("")
    if d.get("preprocessing_implications"):
        lines += ["## Preprocessing implications"] + [f"- {x}" for x in d["preprocessing_implications"]]
    return "\n".join(lines)
