"""Shared helpers for the data/EDA/preprocessing/modeling MCP servers.

Tools are stateless: they read the project's CSV from disk and persist
intermediate results (prepared splits, fitted preprocessing) under the
project's artifact folder. This keeps tool calls reproducible and avoids
shipping large dataframes across the (in-process) MCP boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.services import artifact_store


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8
    return False


def looks_like_id(name: str, series: pd.Series, n_rows: int) -> bool:
    lname = name.lower()
    name_hint = lname.endswith("_id") or lname == "id" or lname.endswith("id")
    high_unique = series.nunique(dropna=True) >= max(0.9 * n_rows, 1)
    return bool(name_hint and high_unique) or high_unique and name_hint


def infer_numeric_categorical(
    df: pd.DataFrame, exclude: List[str]
) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


# ---------------------------------------------------------------------------
# Prepared-data persistence (numpy arrays + metadata)
# ---------------------------------------------------------------------------
def prepared_dir(project_id: str) -> Path:
    d = artifact_store.project_artifact_dir(project_id) / "prepared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_prepared(
    project_id: str,
    splits: Dict[str, np.ndarray],
    feature_names: List[str],
    meta: Dict[str, Any],
) -> str:
    d = prepared_dir(project_id)
    np.savez_compressed(d / "splits.npz", **splits)
    (d / "feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta, default=str, indent=2), encoding="utf-8")
    return str(d / "splits.npz")


def load_prepared(project_id: str) -> Optional[Dict[str, Any]]:
    d = prepared_dir(project_id)
    npz_path = d / "splits.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    feature_names = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return {
        "splits": {k: data[k] for k in data.files},
        "feature_names": feature_names,
        "meta": meta,
    }


def to_native(obj: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serialisable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj
