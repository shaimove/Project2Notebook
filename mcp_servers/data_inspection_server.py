"""Data Inspection MCP Server.

Tools to load and profile the dataset, detect missingness/invalid values,
duplicates, target distribution, time and entity columns.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from backend.config import settings
from mcp_servers._mlcommon import (
    chunked_duplicate_count,
    chunked_missing_fractions,
    csv_load_plan,
    estimate_row_count,
    is_datetime_like,
    load_csv,
    load_csv_for_analysis,
    looks_like_id,
    to_native,
)
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="data-inspection-tools",
    description="Load and profile datasets, detect data-quality issues.",
)

_S = {"csv_path": {"type": "string", "required": True}}


def _load(args: Dict[str, Any], for_analysis: bool = True) -> pd.DataFrame:
    target = (args.get("spec") or {}).get("targets") or []
    target_col = target[0] if target else args.get("target")
    return load_csv(args["csv_path"], for_analysis=for_analysis, target_column=target_col)


@server.tool("Load a CSV dataset and return basic shape and column names.", _S)
def load_dataset(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = csv_load_plan(args["csv_path"])
    header = pd.read_csv(args["csv_path"], nrows=0)
    return {
        "n_rows": plan["estimated_rows"],
        "n_cols": int(len(header.columns)),
        "columns": list(header.columns),
        "dtypes": {c: str(header[c].dtype) for c in header.columns},
        "file_bytes": plan["file_bytes"],
        "sampled_for_analysis": plan["use_analysis_sample"],
    }


@server.tool("Full per-column profile of the dataset.", _S)
def profile_dataset(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = csv_load_plan(args["csv_path"])
    df = _load(args, for_analysis=True)
    n_full = plan["estimated_rows"] or len(df)
    miss_exact = chunked_missing_fractions(args["csv_path"]) if plan["use_chunked_stats"] else None
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        pct_missing = miss_exact.get(c, float(s.isna().mean())) if miss_exact else float(s.isna().mean())
        n_missing = int(round(pct_missing * n_full))
        n_unique = int(s.nunique(dropna=True))
        profile: Dict[str, Any] = {
            "name": c,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 4),
            "n_unique": n_unique,
            "is_constant": n_unique <= 1,
            "is_near_constant": (n_unique <= 1)
            or (s.value_counts(normalize=True, dropna=True).iloc[0] > 0.98 if n_unique else False),
            "sample_values": [to_native(v) for v in s.dropna().unique()[:5]],
        }
        if pd.api.types.is_numeric_dtype(s):
            profile.update(
                mean=to_native(s.mean()),
                std=to_native(s.std()),
                min=to_native(s.min()),
                max=to_native(s.max()),
            )
        else:
            profile["cardinality"] = n_unique
        cols.append(profile)
    return {
        "n_rows": n_full,
        "n_cols": int(df.shape[1]),
        "analysis_sample_rows": n,
        "sampled_for_analysis": plan["use_analysis_sample"],
        "columns": cols,
    }


@server.tool("Validate the dataset against an expected set of columns.", {
    **_S, "expected_columns": {"type": "array"}})
def validate_schema(args: Dict[str, Any]) -> Dict[str, Any]:
    df = _load(args)
    expected = set(args.get("expected_columns") or [])
    actual = set(df.columns)
    return {
        "missing_from_data": sorted(expected - actual),
        "unexpected_in_data": sorted(actual - expected) if expected else [],
        "matched": sorted(expected & actual),
    }


@server.tool("Summarise missing values per column.", _S)
def summarize_missing_values(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = csv_load_plan(args["csv_path"])
    if plan["use_chunked_stats"]:
        pct = chunked_missing_fractions(args["csv_path"])
        n = plan["estimated_rows"]
        miss = {c: int(round(p * n)) for c, p in pct.items()}
    else:
        df = _load(args, for_analysis=False)
        n = len(df)
        miss = {c: int(df[c].isna().sum()) for c in df.columns}
        pct = {c: (v / n if n else 0.0) for c, v in miss.items()}

    return {
        "per_column": miss,
        "per_column_pct": {c: round(p, 4) for c, p in pct.items()},
        "columns_with_missing": [c for c, v in miss.items() if v > 0],
        "near_empty_columns": [c for c, p in pct.items() if p > settings.missing_drop_threshold],
    }


@server.tool("Detect invalid values (e.g. negative where non-negative expected, infinities).", _S)
def detect_invalid_values(args: Dict[str, Any]) -> Dict[str, Any]:
    df = _load(args)
    issues: Dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            n_inf = int(np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).sum())
            n_neg = int((s < 0).sum())
            col_issues = {}
            if n_inf:
                col_issues["infinities"] = n_inf
            # Heuristic: count-like / duration-like columns should be non-negative.
            lname = c.lower()
            if any(k in lname for k in ("count", "num_", "duration", "age", "amount", "qty")) and n_neg:
                col_issues["unexpected_negatives"] = n_neg
            if col_issues:
                issues[c] = col_issues
    return {"invalid_value_issues": issues}


@server.tool("Count duplicate rows.", _S)
def detect_duplicates(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = csv_load_plan(args["csv_path"])
    if plan["use_chunked_stats"]:
        n_dup = chunked_duplicate_count(args["csv_path"])
    else:
        df = _load(args, for_analysis=False)
        n_dup = int(df.duplicated().sum())
    return {"n_duplicate_rows": n_dup}


@server.tool("Inspect the target distribution.", {**_S, "target": {"type": "string", "required": True}})
def inspect_target_distribution(args: Dict[str, Any]) -> Dict[str, Any]:
    df = _load(args)
    target = args["target"]
    if target not in df.columns:
        return {"error": f"target '{target}' not in columns"}
    s = df[target]
    out: Dict[str, Any] = {"target": target, "dtype": str(s.dtype), "n_missing": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 15:
        out["kind"] = "continuous"
        out["stats"] = {
            "mean": to_native(s.mean()),
            "std": to_native(s.std()),
            "min": to_native(s.min()),
            "max": to_native(s.max()),
            "skew": to_native(s.skew()),
        }
    else:
        vc = s.value_counts(dropna=False)
        counts = {str(k): int(v) for k, v in vc.items()}
        props = {str(k): round(v / len(s), 4) for k, v in vc.items()}
        minority = min(props.values()) if props else None
        out["kind"] = "categorical"
        out["counts"] = counts
        out["proportions"] = props
        out["n_classes"] = int(s.nunique(dropna=True))
        out["minority_class_share"] = minority
        out["imbalance_ratio"] = round(max(props.values()) / minority, 3) if minority else None
    return out


@server.tool("Detect likely time/date columns.", _S)
def detect_time_columns(args: Dict[str, Any]) -> Dict[str, Any]:
    df = _load(args)
    time_cols = [c for c in df.columns if is_datetime_like(df[c]) or "date" in c.lower() or "time" in c.lower()]
    # de-dup while preserving order
    seen, result = set(), []
    for c in time_cols:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return {"time_columns": result}


@server.tool("Detect likely entity/ID columns (for grouped splits).", _S)
def detect_entity_columns(args: Dict[str, Any]) -> Dict[str, Any]:
    df = _load(args)
    n = len(df)
    entity_cols = []
    for c in df.columns:
        lname = c.lower()
        if lname.endswith("_id") or lname == "id" or "customer" in lname or "user" in lname or "patient" in lname or "machine" in lname or "session" in lname:
            entity_cols.append(c)
    id_like = [c for c in df.columns if looks_like_id(c, df[c], n)]
    return {"entity_columns": sorted(set(entity_cols)), "unique_id_like": sorted(set(id_like))}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
