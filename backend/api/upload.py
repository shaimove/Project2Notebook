"""File upload endpoints for project document, CSVs and PDFs."""
from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.exceptions import BadRequestError, NotFoundError
from backend.schemas.api import UploadResponse
from backend.services import file_store, project_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])


def _require_project(project_id: str) -> None:
    if project_store.get_project(project_id) is None:
        raise NotFoundError("project not found")


def _read_upload_bytes(file: UploadFile) -> bytes:
    if not file.filename:
        raise BadRequestError("missing filename")
    content = file.file.read()
    file.file.seek(0)
    if len(content) == 0:
        raise BadRequestError("empty file")
    if len(content) > settings.max_upload_bytes:
        raise BadRequestError(
            f"file exceeds maximum size of {settings.max_upload_bytes} bytes"
        )
    return content


def _validate_csv(content: bytes) -> None:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BadRequestError("CSV must be UTF-8 encoded") from exc
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or all(not cell.strip() for cell in header):
        raise BadRequestError("CSV must have a header row with column names")


def _save(project_id: str, file: UploadFile, kind: str, *, validate_csv: bool = False) -> UploadResponse:
    _require_project(project_id)
    content = _read_upload_bytes(file)
    if validate_csv:
        _validate_csv(content)
    path = file_store.save_upload(project_id, file.filename, file.file)
    logger.info("Uploaded %s for project %s (%s bytes)", kind, project_id, len(content))
    project_store.add_file(project_id, kind, str(path))
    return UploadResponse(project_id=project_id, stored_path=str(path), kind=kind, filename=file.filename)


@router.post("/project-document", response_model=UploadResponse)
def upload_project_document(project_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    return _save(project_id, file, "project_document")


@router.post("/csv", response_model=UploadResponse)
def upload_csv(project_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="expected a .csv file")
    return _save(project_id, file, "csv", validate_csv=True)


@router.post("/pdf", response_model=UploadResponse)
def upload_pdf(project_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="expected a .pdf file")
    return _save(project_id, file, "pdf")
