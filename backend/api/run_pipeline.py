"""Run the agentic pipeline and expose status/artifacts."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.agents.graph import finish_graph_run, run_graph
from backend.agents.state import new_state
from backend.exceptions import NotFoundError
from backend.schemas.api import RunRequest, RunResponse, RunSessionInfo, StatusResponse
from backend.services import project_store
from backend.services.checkpoint_store import get_checkpoint_store
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

    store = get_checkpoint_store()
    run_id: str
    start_step = 1
    resumed_from: int | None = None

    if req.resume or req.from_step is not None:
        existing = store.get_latest_resumable_run(req.project_id)
        if existing is None:
            raise HTTPException(status_code=400, detail="no resumable run found for this project")
        run_id = existing["run_id"]
        try:
            state, start_step = store.load_resume_state(run_id, from_step=req.from_step)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        resumed_from = start_step
        state["errors"] = []
        state["timeline"] = []
        state["tool_calls"] = []
        logger.info("Resuming run %s for project %s from step %d", run_id, req.project_id, start_step)
    else:
        logger.info("Starting pipeline run for project %s", req.project_id)
        state = new_state(
            project_id=req.project_id,
            project_document_path=record.get("project_document_path") or "",
            csv_paths=record.get("csv_paths", []),
            pdf_paths=record.get("pdf_paths", []),
            enable_prior_art=req.enable_prior_art,
            max_iterations=req.max_iterations,
            min_relative_improvement=req.min_relative_improvement,
        )
        run_id = store.start_run(
            req.project_id,
            {
                "enable_prior_art": req.enable_prior_art,
                "max_iterations": req.max_iterations,
                "min_relative_improvement": req.min_relative_improvement,
            },
        )

    project_store.update_project(req.project_id, status="running")
    state = run_graph(state, run_id=run_id, start_step=start_step)
    finish_graph_run(state)

    status = "completed_with_errors" if state.get("errors") else "completed"
    result: Dict[str, Any] = {
        "project_id": req.project_id,
        "run_id": run_id,
        "resumed_from_step": resumed_from,
        "status": status,
        "timeline": state.get("timeline", []),
        "tool_calls": state.get("tool_calls", []),
        "artifacts": assemble_artifacts(state),
        "summary": build_summary(state),
        "errors": state.get("errors", []),
    }
    project_store.set_run_result(req.project_id, result)
    return RunResponse(**result)


@router.get("/projects/{project_id}/runs", response_model=List[RunSessionInfo])
def list_runs(project_id: str) -> List[RunSessionInfo]:
    record = project_store.get_project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    runs = get_checkpoint_store().list_runs(project_id)
    return [RunSessionInfo(**r) for r in runs]


@router.get("/projects/{project_id}/runs/{run_id}/checkpoints")
def list_checkpoints(project_id: str, run_id: str) -> Dict[str, Any]:
    record = project_store.get_project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    run = get_checkpoint_store().get_run(run_id)
    if run is None or run.get("project_id") != project_id:
        raise NotFoundError("run not found")
    return {
        "run_id": run_id,
        "checkpoints": get_checkpoint_store().list_checkpoints(run_id),
    }


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
        "run_id": run.get("run_id"),
    }
