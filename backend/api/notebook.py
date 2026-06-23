"""Notebook + plot serving endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.services import artifact_store, project_store

router = APIRouter(prefix="/api/projects", tags=["notebook"])


@router.get("/{project_id}/notebook")
def get_notebook(project_id: str) -> Dict[str, Any]:
    if project_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    spec = artifact_store.read_json(project_id, "notebook_spec.json") or {"sections": []}
    run = (project_store.get_project(project_id) or {}).get("run_result") or {}
    nb_path = (run.get("artifacts") or {}).get("notebook_path")
    return {
        "project_id": project_id,
        "notebook_path": nb_path,
        "title": spec.get("title", "Final Report"),
        "sections": spec.get("sections", []),
        "download_url": f"/api/projects/{project_id}/notebook/download",
    }


@router.get("/{project_id}/notebook/download")
def download_notebook(project_id: str) -> FileResponse:
    run = (project_store.get_project(project_id) or {}).get("run_result") or {}
    nb_path = (run.get("artifacts") or {}).get("notebook_path")
    if not nb_path or not Path(nb_path).exists():
        raise HTTPException(status_code=404, detail="notebook not generated yet")
    return FileResponse(nb_path, media_type="application/x-ipynb+json",
                        filename=f"{project_id}_final_notebook.ipynb")


@router.get("/{project_id}/plots/{name}")
def get_plot(project_id: str, name: str) -> FileResponse:
    # Prevent path traversal: only serve files directly in the plots dir.
    safe = Path(name).name
    path = artifact_store.plots_dir(project_id) / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="plot not found")
    if safe.lower().endswith(".html"):
        return FileResponse(path, media_type="text/html")
    return FileResponse(path, media_type="image/png")
