"""Handles persistence of uploaded files (project doc, CSVs, PDFs)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from backend.config import settings


def project_upload_dir(project_id: str) -> Path:
    d = settings.uploads_dir / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(project_id: str, filename: str, fileobj: BinaryIO) -> Path:
    """Persist an uploaded file under the project's upload directory."""
    safe_name = Path(filename).name  # strip any path components
    dest = project_upload_dir(project_id) / safe_name
    with dest.open("wb") as out:
        shutil.copyfileobj(fileobj, out)
    return dest


def save_bytes(project_id: str, filename: str, data: bytes) -> Path:
    safe_name = Path(filename).name
    dest = project_upload_dir(project_id) / safe_name
    dest.write_bytes(data)
    return dest


def copy_into_project(project_id: str, source: Path) -> Path:
    """Copy an existing file (e.g. from the demo folder) into the project."""
    dest = project_upload_dir(project_id) / source.name
    shutil.copyfile(source, dest)
    return dest
