"""In-memory + on-disk registry of projects and their run results.

For the MVP this is a simple JSON-backed store. It is intentionally small and
easy to replace with a real database later.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import settings

_LOCK = threading.Lock()
_REGISTRY_PATH = settings.storage_root / "projects.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> Dict[str, Any]:
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(data: Dict[str, Any]) -> None:
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def create_project(name: str, description: Optional[str] = None) -> Dict[str, Any]:
    with _LOCK:
        data = _load()
        project_id = uuid.uuid4().hex[:12]
        record = {
            "project_id": project_id,
            "name": name or "Untitled Project",
            "description": description,
            "created_at": _now(),
            "project_document_path": None,
            "csv_paths": [],
            "pdf_paths": [],
            "status": "created",
            "run_result": None,
        }
        data[project_id] = record
        _save(data)
        return record


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    return _load().get(project_id)


def list_projects() -> List[Dict[str, Any]]:
    return list(_load().values())


def update_project(project_id: str, **fields: Any) -> Dict[str, Any]:
    with _LOCK:
        data = _load()
        record = data.get(project_id)
        if record is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        record.update(fields)
        data[project_id] = record
        _save(data)
        return record


def add_file(project_id: str, kind: str, path: str) -> Dict[str, Any]:
    """kind in {project_document, csv, pdf}."""
    with _LOCK:
        data = _load()
        record = data.get(project_id)
        if record is None:
            raise KeyError(f"Unknown project_id: {project_id}")
        if kind == "project_document":
            record["project_document_path"] = path
        elif kind == "csv":
            if path not in record["csv_paths"]:
                record["csv_paths"].append(path)
        elif kind == "pdf":
            if path not in record["pdf_paths"]:
                record["pdf_paths"].append(path)
        else:
            raise ValueError(f"Unknown file kind: {kind}")
        data[project_id] = record
        _save(data)
        return record


def set_run_result(project_id: str, result: Dict[str, Any]) -> None:
    update_project(project_id, run_result=result, status=result.get("status", "completed"))
