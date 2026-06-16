"""Run the agentic pipeline and expose status/artifacts."""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend.agents.graph import run_graph
from backend.agents.state import new_state
from backend.schemas.api import RunRequest, RunResponse, StatusResponse
from backend.services import project_store
from backend.services.run_result import assemble_artifacts, build_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["run"])


@router.post("/run", response_model=RunResponse)
def run_pipeline(req: RunRequest) -> RunResponse:
    record = project_store.get_project(req.project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    if not record.get("csv_paths"):
        raise HTTPException(status_code=400, detail="upload at least one CSV before running")

    logger.info("Starting pipeline run for project %s", req.project_id)
    project_store.update_project(req.project_id, status="running")

    state = new_state(
        project_id=req.project_id,
        project_document_path=record.get("project_document_path") or "",
        csv_paths=record.get("csv_paths", []),
        pdf_paths=record.get("pdf_paths", []),
        enable_prior_art=req.enable_prior_art,
        max_iterations=req.max_iterations,
        min_relative_improvement=req.min_relative_improvement,
    )
    state = run_graph(state)

    status = "completed_with_errors" if state.get("errors") else "completed"
    result: Dict[str, Any] = {
        "project_id": req.project_id,
        "status": status,
        "timeline": state.get("timeline", []),
        "tool_calls": state.get("tool_calls", []),
        "artifacts": assemble_artifacts(state),
        "summary": build_summary(state),
        "errors": state.get("errors", []),
    }
    project_store.set_run_result(req.project_id, result)
    if state.get("errors"):
        logger.warning(
            "Pipeline completed with %d error(s) for project %s",
            len(state["errors"]),
            req.project_id,
        )
    else:
        logger.info("Pipeline completed successfully for project %s", req.project_id)
    return RunResponse(**result)


@router.get("/projects/{project_id}/status", response_model=StatusResponse)
def get_status(project_id: str) -> StatusResponse:
    record = project_store.get_project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    run = record.get("run_result") or {}
    return StatusResponse(
        project_id=project_id,
        status=record.get("status", "created"),
        timeline=run.get("timeline", []),
        errors=run.get("errors", []),
    )


@router.get("/projects/{project_id}/artifacts")
def get_artifacts(project_id: str) -> Dict[str, Any]:
    record = project_store.get_project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    run = record.get("run_result") or {}
    return {
        "project_id": project_id,
        "artifacts": run.get("artifacts", {}),
        "tool_calls": run.get("tool_calls", []),
        "timeline": run.get("timeline", []),
        "summary": run.get("summary", ""),
        "errors": run.get("errors", []),
    }
