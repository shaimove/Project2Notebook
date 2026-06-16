"""Prior Art Agent.

Looks up common approaches for the inferred task type. Web search is disabled by
default and clearly marked as such — the curated knowledge base is inspiration
only and must be verified against the real dataset by later agents.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from backend.agents.state import DataScientist
from backend.config import settings
from backend.mcp_client.client import MCPClient
from backend.schemas.project_spec import PriorArtReport
from backend.services import artifact_store, memory

PA = "prior-art-tools"
PU = "project-understanding-tools"


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    spec = state.get("project_spec") or {}
    task = spec.get("ml_task_type", "unknown")

    if not state.get("enable_prior_art", True):
        report = PriorArtReport(enabled=False, message="Prior Art Agent disabled for this run.")
        _persist(state, report)
        return state

    pdf_texts: List[str] = []
    for pdf in state.get("pdf_paths", []) or []:
        parsed = client.call_tool(PU, "parse_project_document", {"path": pdf})
        if parsed.get("text"):
            pdf_texts.append(parsed["text"][:2000])

    search = client.call_tool(PA, "search_prior_art", {
        "spec": spec, "enable_web_search": settings.enable_web_search, "pdf_texts": pdf_texts,
    })
    approaches = client.call_tool(PA, "summarize_common_approaches", {"ml_task_type": task})
    candidates = client.call_tool(PA, "extract_candidate_strategies", {
        "approaches": approaches, "pdf_texts": pdf_texts,
    })

    web_enabled = bool(search.get("enabled"))
    summary = (
        f"Curated offline prior-art for '{task}'. "
        + ("Web search active. " if web_enabled and search.get("sources") else "Web search disabled — curated knowledge only. ")
        + "All ideas must be verified against the actual dataset."
    )
    report = PriorArtReport(
        enabled=True,
        message=search.get("message", ""),
        searched_queries=search.get("searched_queries", []),
        sources=search.get("sources", []) + ([f"uploaded PDF ({len(pdf_texts)})"] if pdf_texts else []),
        common_models=approaches.get("common_models", []),
        common_feature_engineering=approaches.get("common_feature_engineering", []),
        common_preprocessing=approaches.get("common_preprocessing", []),
        common_split_strategies=approaches.get("common_split_strategies", []),
        common_metrics=approaches.get("common_metrics", []),
        leakage_warnings=approaches.get("leakage_warnings", []),
        ideas_to_test=candidates.get("ideas_to_test", []),
        ideas_to_ignore=candidates.get("ideas_to_ignore", []),
        summary=summary,
    )
    _persist(state, report)
    return state


def _persist(state: DataScientist, report: PriorArtReport) -> None:
    d = report.model_dump()
    pid = state["project_id"]
    artifact_store.write_json(pid, "prior_art_report.json", d)
    artifact_store.write_text(pid, "prior_art.md", _render_md(d))
    state["prior_art_report"] = d
    memory.update(
        pid,
        phase="Prior Art",
        rejected_ideas=[f"(prior-art suggests ignoring) {i}" for i in d.get("ideas_to_ignore", [])],
        open_questions=[],
    )


def _render_md(d: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {i}" for i in (items or [])) or "- (none)"

    lines = ["# Prior Art", "", f"_{d.get('summary', '')}_", ""]
    if d.get("message"):
        lines += [f"> {d['message']}", ""]
    if not d.get("enabled"):
        lines += ["Prior Art Agent disabled for this run."]
        return "\n".join(lines)
    lines += [
        "## Common models", bullets(d.get("common_models")), "",
        "## Common feature engineering", bullets(d.get("common_feature_engineering")), "",
        "## Common preprocessing", bullets(d.get("common_preprocessing")), "",
        "## Common split strategies", bullets(d.get("common_split_strategies")), "",
        "## Common metrics", bullets(d.get("common_metrics")), "",
        "## Leakage warnings", bullets(d.get("leakage_warnings")), "",
        "## Ideas to test (verify on the actual data!)", bullets(d.get("ideas_to_test")), "",
        "## Ideas to ignore", bullets(d.get("ideas_to_ignore")), "",
        f"Sources: {', '.join(d.get('sources', [])) or '(curated offline knowledge)'}",
    ]
    return "\n".join(lines)
