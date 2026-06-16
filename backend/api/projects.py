"""Project create/read endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from backend.schemas.api import CreateProjectRequest, ProjectInfo
from backend.services import project_store

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("", response_model=ProjectInfo)
def create_project(req: CreateProjectRequest) -> ProjectInfo:
    record = project_store.create_project(req.name, req.description)
    return ProjectInfo(**_to_info(record))


@router.get("", response_model=List[ProjectInfo])
def list_projects() -> List[ProjectInfo]:
    return [ProjectInfo(**_to_info(r)) for r in project_store.list_projects()]


@router.get("/{project_id}", response_model=ProjectInfo)
def get_project(project_id: str) -> ProjectInfo:
    record = project_store.get_project(project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    return ProjectInfo(**_to_info(record))


def _to_info(record: dict) -> dict:
    return {
        "project_id": record["project_id"],
        "name": record["name"],
        "created_at": record["created_at"],
        "project_document_path": record.get("project_document_path"),
        "csv_paths": record.get("csv_paths", []),
        "pdf_paths": record.get("pdf_paths", []),
        "status": record.get("status", "created"),
    }
