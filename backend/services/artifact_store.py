"""Reads/writes per-phase artifacts under storage/artifacts/{project_id}.

Each agent phase persists a clear artifact file (JSON / Markdown / CSV / plots)
so the run is transparent and reproducible.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.exceptions import ArtifactCorruptError

logger = logging.getLogger(__name__)


def project_artifact_dir(project_id: str) -> Path:
    d = settings.artifacts_dir / project_id
    for sub in ("plots", "code", "tables", "reports"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def plots_dir(project_id: str) -> Path:
    return project_artifact_dir(project_id) / "plots"


def code_dir(project_id: str) -> Path:
    return project_artifact_dir(project_id) / "code"


def tables_dir(project_id: str) -> Path:
    return project_artifact_dir(project_id) / "tables"


def reports_dir(project_id: str) -> Path:
    return project_artifact_dir(project_id) / "reports"


def list_tables(project_id: str) -> list[str]:
    tdir = tables_dir(project_id)
    if not tdir.exists():
        return []
    return sorted(str(p) for p in tdir.glob("*.csv"))


def write_json(project_id: str, name: str, data: Any) -> Path:
    path = project_artifact_dir(project_id) / name
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def read_json(project_id: str, name: str, *, default: Any = None) -> Any:
    path = project_artifact_dir(project_id) / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactCorruptError(
            f"Corrupt JSON artifact '{name}' for project {project_id}"
        ) from exc


def write_text(project_id: str, name: str, text: str) -> Path:
    path = project_artifact_dir(project_id) / name
    path.write_text(text, encoding="utf-8")
    return path


def read_text(project_id: str, name: str) -> str | None:
    path = project_artifact_dir(project_id) / name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def list_plots(project_id: str) -> List[str]:
    pdir = plots_dir(project_id)
    if not pdir.exists():
        return []
    return sorted(str(p) for p in pdir.glob("*.png"))


def list_artifacts(project_id: str) -> Dict[str, Any]:
    """Return a manifest of all artifacts produced for a project."""
    adir = project_artifact_dir(project_id)
    files: Dict[str, Any] = {}
    for p in sorted(adir.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(adir))] = str(p)
    return files
