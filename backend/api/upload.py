"""File upload endpoints for project document, CSVs and PDFs."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas.api import UploadResponse
from backend.services import file_store, project_store

router = APIRouter(prefix="/api/upload", tags=["upload"])


def _require_project(project_id: str) -> None:
    if project_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")


def _save(project_id: str, file: UploadFile, kind: str) -> UploadResponse:
    _require_project(project_id)
    path = file_store.save_upload(project_id, file.filename, file.file)
    project_store.add_file(project_id, kind, str(path))
    return UploadResponse(project_id=project_id, stored_path=str(path), kind=kind, filename=file.filename)


@router.post("/project-document", response_model=UploadResponse)
def upload_project_document(project_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    return _save(project_id, file, "project_document")


@router.post("/csv", response_model=UploadResponse)
def upload_csv(project_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="expected a .csv file")
    return _save(project_id, file, "csv")


@router.post("/pdf", response_model=UploadResponse)
def upload_pdf(project_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="expected a .pdf file")
    return _save(project_id, file, "pdf")
