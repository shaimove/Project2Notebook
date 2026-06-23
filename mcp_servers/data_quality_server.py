"""Data Quality MCP Server.

Scans raw CSVs for column-name, value, and categorical consistency issues and
applies a deterministic remediation plan (writes ``cleaned_*.csv``).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.services import file_store
from mcp_servers._mlcommon import (
    csv_load_plan,
    chunked_missing_fractions,
    drop_duplicates_streaming,
    estimate_row_count,
    is_datetime_like,
    load_csv,
    looks_like_id,
    to_native,
    transform_csv_chunked,
)
from backend.config import settings
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="data-quality-tools",
    description="Scan and remediate raw tabular data quality issues.",
)

_S = {"csv_path": {"type": "string", "required": True}}

# Columns with missingness above this fraction are dropped (not imputed) in Data Quality.
MISSING_DROP_THRESHOLD = settings.missing_drop_threshold


def _normalize_col_name(name: str) -> str:
    n = str(name).strip()
    if re.match(r"^unnamed:\s*\d+$", n, re.I):
        return ""
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^\w]", "_", n)
    n = re.sub(r"_+", "_", n).strip("_").lower()
    return n


def _scan_column_names(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    issues: List[Dict[str, Any]] = []
    renames: Dict[str, str] = {}
    drops: List[str] = []
    seen: Dict[str, str] = {}

    for col in df.columns:
        raw = str(col)
        clean = _normalize_col_name(raw)
        if not clean:
            drops.append(raw)
            issues.append({
                "issue_type": "unnamed_column",
                "severity": "warning",
                "column": raw,
                "description": f"Column '{raw}' looks like an index/export artifact.",
                "suggested_fix": "Drop before analysis.",
            })
            continue
        if clean != raw:
            renames[raw] = clean
            issues.append({
                "issue_type": "bad_column_name",
                "severity": "info",
                "column": raw,
                "description": f"Column name '{raw}' should be normalized.",
                "suggested_fix": f"Rename to '{clean}'.",
            })
        if clean in seen and seen[clean] != raw:
            suffix = 2
            candidate = f"{clean}_{suffix}"
            while candidate in seen.values():
                suffix += 1
                candidate = f"{clean}_{suffix}"
            renames[raw] = candidate
            issues.append({
                "issue_type": "duplicate_column_name",
                "severity": "warning",
                "column": raw,
                "description": f"Duplicate normalized name '{clean}'.",
                "suggested_fix": f"Rename to '{candidate}'.",
            })
            clean = candidate
        seen[clean] = raw

    return issues, renames, drops


def _scan_value_consistency(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    coerce: List[str] = []
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
            continue
        non_null = s.dropna().astype(str).str.strip()
        if non_null.empty:
            continue
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = float(numeric.notna().mean())
        if 0.4 < numeric_ratio < 0.95:
            issues.append({
                "issue_type": "mixed_numeric_strings",
                "severity": "warning",
                "column": str(col),
                "description": (
                    f"Column '{col}' mixes numeric and non-numeric string values "
                    f"({numeric_ratio:.0%} parse as numbers)."
                ),
                "suggested_fix": "Coerce to numeric; invalid values become NaN.",
            })
            coerce.append(str(col))
    return issues, coerce


def _canonical_category(value: str) -> str:
    v = value.strip()
    key = re.sub(r"[^a-z0-9]+", "", v.lower())
    aliases = {
        "usa": "usa",
        "us": "usa",
        "u s a": "usa",
        "unitedstates": "usa",
        "uk": "uk",
        "unitedkingdom": "uk",
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
    }
    return aliases.get(key, v)


def _scan_categorical_cardinality(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    issues: List[Dict[str, Any]] = []
    category_maps: Dict[str, Dict[str, str]] = {}
    n = len(df)

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        non_null = s.dropna().astype(str).str.strip()
        if non_null.empty:
            continue
        n_unique = non_null.nunique()
        if n_unique > min(50, max(10, n // 5)):
            continue

        canonical_counts: Counter[str] = Counter()
        mapping: Dict[str, str] = {}
        for raw in non_null.unique():
            canon = _canonical_category(raw)
            canonical_counts[canon] += int((non_null == raw).sum())
            if canon != raw:
                mapping[raw] = canon

        # Collapse alias groups to the most frequent canonical label.
        by_canon: Dict[str, List[str]] = {}
        for raw, canon in mapping.items():
            by_canon.setdefault(canon, []).append(raw)
        for canon, raws in by_canon.items():
            winner = max(raws, key=lambda r: int((non_null == r).sum()))
            for raw in raws:
                mapping[raw] = winner

        if mapping:
            category_maps[str(col)] = mapping
            issues.append({
                "issue_type": "inconsistent_categories",
                "severity": "warning",
                "column": str(col),
                "description": (
                    f"Column '{col}' has {len(mapping)} categorical variant(s) "
                    f"that can be normalized."
                ),
                "suggested_fix": "Map aliases to canonical labels.",
            })
    return issues, category_maps


def _scan_uniqueness(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[Dict[str, Any]] = []
    drops: List[str] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        if looks_like_id(str(col), s, n) and not str(col).lower().endswith("_id"):
            issues.append({
                "issue_type": "id_like_column",
                "severity": "warning",
                "column": str(col),
                "description": f"Column '{col}' is {n_unique}/{n} unique (ID-like).",
                "suggested_fix": "Drop before modeling unless used for grouping.",
            })
        if n_unique == n and n > 1 and str(col).lower() in ("id", "index", "row_id"):
            drops.append(str(col))
            issues.append({
                "issue_type": "row_identifier",
                "severity": "critical",
                "column": str(col),
                "description": f"Column '{col}' is a row identifier.",
                "suggested_fix": "Drop before modeling.",
            })
    return issues, drops


def _scan_missing_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    n = len(df)
    if n == 0:
        return issues
    miss_pct = {str(c): float(df[c].isna().mean()) for c in df.columns}
    heavy = [c for c, p in miss_pct.items() if p > MISSING_DROP_THRESHOLD]
    for col in heavy:
        issues.append({
            "issue_type": "heavy_missingness",
            "severity": "warning",
            "column": col,
            "description": (
                f"Column '{col}' is {miss_pct[col]:.1%} missing "
                f"(>{MISSING_DROP_THRESHOLD:.0%} threshold)."
            ),
            "suggested_fix": "Drop before modeling.",
        })
    return issues


def _scan_duplicates(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], bool]:
    n_dup = int(df.duplicated().sum())
    if not n_dup:
        return [], False
    return [{
        "issue_type": "duplicate_rows",
        "severity": "warning",
        "column": None,
        "description": f"{n_dup} duplicate row(s) detected.",
        "suggested_fix": "Drop duplicate rows.",
    }], True


_IMAGE_NAME_HINTS = ("image", "img", "photo", "picture", "thumbnail", "pixel", "base64")
_PATH_PATTERN = re.compile(
    r"^(https?://|/|\./|data:image|.*\.(jpg|jpeg|png|gif|webp|bmp|tiff))", re.I
)


def _looks_like_image_content(col: str, series: pd.Series) -> bool:
    lname = str(col).lower()
    non_null = series.dropna()
    if non_null.empty:
        return any(h in lname for h in _IMAGE_NAME_HINTS)
    as_str = non_null.astype(str)
    avg_len = float(as_str.str.len().mean())
    path_ratio = float(as_str.str.match(_PATH_PATTERN, na=False).mean())
    if any(h in lname for h in _IMAGE_NAME_HINTS) and (avg_len > 40 or path_ratio > 0.2):
        return True
    if pd.api.types.is_object_dtype(series) and avg_len > 150 and series.nunique(dropna=True) > 50:
        return True
    return False


def _infer_column_type(col: str, series: pd.Series, n_rows: int) -> str:
    if _looks_like_image_content(col, series):
        return "image"
    if series.nunique(dropna=True) <= 1:
        return "constant"
    if looks_like_id(col, series, n_rows) or str(col).lower() in ("id", "index", "row_id"):
        return "id"
    if is_datetime_like(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        n_unique = int(series.nunique(dropna=True))
        if n_unique == 2:
            return "binary"
        return "continuous"
    non_null = series.dropna().astype(str)
    n_unique = int(non_null.nunique())
    if n_unique <= 2:
        return "binary"
    if n_unique <= 50:
        return "categorical"
    avg_len = float(non_null.str.len().mean()) if len(non_null) else 0
    if avg_len > 80:
        return "free_text"
    return "categorical"


def _resolve_target(
    df: pd.DataFrame, spec: Dict[str, Any], image_cols: List[str]
) -> Optional[str]:
    declared = list(spec.get("targets") or [])
    for t in declared:
        if t in df.columns and t not in image_cols:
            return t
    # Heuristic: label-like columns that are NOT image paths
    label_hints = (
        "quality", "target", "label", "class", "outcome", "churn", "defect",
        "grade", "score", "rating", "status", "y",
    )
    candidates: List[str] = []
    for col in df.columns:
        if col in image_cols:
            continue
        lname = str(col).lower()
        if any(h in lname for h in label_hints):
            candidates.append(col)
    if candidates:
        return candidates[0]
    # Fallback: low-cardinality non-image column (not id)
    n = len(df)
    best = None
    best_score = 9999
    for col in df.columns:
        if col in image_cols:
            continue
        s = df[col]
        if looks_like_id(col, s, n):
            continue
        n_unique = int(s.nunique(dropna=True))
        if 2 <= n_unique <= 30:
            if n_unique < best_score:
                best_score = n_unique
                best = col
    return best


def _build_column_profiles(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    image_cols: List[str],
    target: Optional[str],
    entity_cols: Optional[List[str]] = None,
    missing_fractions: Optional[Dict[str, float]] = None,
    n_rows_full: Optional[int] = None,
) -> List[Dict[str, Any]]:
    n = n_rows_full or len(df)
    entity_cols = entity_cols or []
    profiles: List[Dict[str, Any]] = []
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    for col in df.columns:
        s = df[col]
        col_type = _infer_column_type(col, s, len(df))
        pct_miss = float(missing_fractions.get(str(col), s.isna().mean())) if missing_fractions else float(s.isna().mean())
        n_unique = int(s.nunique(dropna=True))
        is_const = n_unique <= 1

        role = "unknown"
        desc_parts: List[str] = []
        include = False

        if col == target:
            role = "target"
            desc_parts.append("Prediction target for modeling.")
            include = False
        elif col in image_cols or col_type == "image":
            role = "image"
            desc_parts.append(
                "Image or path-like data — excluded from tabular models; "
                "requires image feature extraction or a vision model."
            )
            include = False
        elif col in entity_cols or col_type == "id" or looks_like_id(col, s, n):
            role = "id"
            desc_parts.append("Identifier column — excluded from training features.")
            include = False
        elif is_datetime_like(s) or col_type == "datetime":
            role = "time"
            desc_parts.append("Datetime column — use for splitting, not as a raw feature.")
            include = False
        elif is_const or col in constant_cols:
            role = "drop"
            desc_parts.append("Constant or near-constant — dropped.")
            include = False
        elif pct_miss > MISSING_DROP_THRESHOLD:
            role = "drop"
            desc_parts.append(
                f"Missingness {pct_miss:.1%} exceeds {MISSING_DROP_THRESHOLD:.0%} threshold — dropped."
            )
            include = False
        elif col_type == "free_text":
            role = "excluded"
            desc_parts.append("Free text — excluded unless text features are engineered.")
            include = False
        else:
            role = "train_feature"
            desc_parts.append("Selected as a tabular modeling feature.")
            include = True

        if col_type == "constant" and role not in ("target", "image"):
            role = "drop"

        profiles.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "column_type": col_type,
            "role": role,
            "pct_missing": round(pct_miss, 4),
            "n_unique": n_unique,
            "is_constant": is_const,
            "description": " ".join(desc_parts),
            "sample_values": [to_native(v) for v in s.dropna().unique()[:3]],
            "include_in_modeling": include,
        })
    return profiles


def _find_bad_row_examples(df: pd.DataFrame, max_preview: int = 3) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    n = len(df)
    if n == 0:
        return examples

    dup_mask = df.duplicated(keep=False)
    if dup_mask.any():
        idx = df.index[dup_mask].tolist()[:max_preview]
        examples.append({
            "issue_type": "duplicate_rows",
            "description": f"{int(dup_mask.sum())} row(s) involved in duplicates.",
            "row_indices": [int(i) for i in idx],
            "preview": df.loc[idx].head(max_preview).astype(str).to_dict(orient="records"),
        })

    all_null = df.isna().all(axis=1)
    if all_null.any():
        idx = df.index[all_null].tolist()[:max_preview]
        examples.append({
            "issue_type": "empty_rows",
            "description": f"{int(all_null.sum())} completely empty row(s).",
            "row_indices": [int(i) for i in idx],
            "preview": df.loc[idx].head(max_preview).astype(str).to_dict(orient="records"),
        })

    return examples


def _build_remediation_plan(
    renames: Dict[str, str],
    drops: List[str],
    category_maps: Dict[str, Dict[str, str]],
    coerce: List[str],
    drop_duplicates: bool,
    modeling_features: Optional[List[str]] = None,
    feature_engineering_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "column_renames": renames,
        "columns_to_drop": sorted(set(drops)),
        "category_maps": category_maps,
        "drop_duplicate_rows": drop_duplicates,
        "coerce_numeric_columns": coerce,
        "strip_string_columns": True,
        "modeling_features": modeling_features or [],
        "feature_engineering_notes": feature_engineering_notes or [],
    }


@server.tool("Scan a CSV for column, value, and categorical data-quality issues.", _S)
def scan_data_quality(args: Dict[str, Any]) -> Dict[str, Any]:
    csv_path = args["csv_path"]
    spec = args.get("spec") or {}
    targets = set(spec.get("targets") or [])
    target_hint = list(targets)[0] if targets else None
    plan_info = csv_load_plan(csv_path)

    df = load_csv(csv_path, for_analysis=True, target_column=target_hint)
    miss_exact = chunked_missing_fractions(csv_path) if plan_info["use_chunked_stats"] else None
    n_rows_full = plan_info["estimated_rows"] or len(df)

    name_issues, renames, name_drops = _scan_column_names(df)
    value_issues, coerce_cols = _scan_value_consistency(df)
    cat_issues, category_maps = _scan_categorical_cardinality(df)
    uniq_issues, id_drops = _scan_uniqueness(df)
    miss_issues = _scan_missing_patterns(df)
    dup_issues, drop_duplicates = _scan_duplicates(df)

    image_cols = [
        str(c) for c in df.columns if _looks_like_image_content(str(c), df[c])
    ]

    target = _resolve_target(df, spec, image_cols)
    if targets and list(targets)[0] in image_cols:
        wrong = list(targets)[0]
        name_issues.append({
            "issue_type": "image_misidentified_as_target",
            "severity": "critical",
            "column": wrong,
            "description": (
                f"Column '{wrong}' looks like image data, not a classification label. "
                f"Suggested target: '{target}'."
            ),
            "suggested_fix": "Use a label column as target; exclude image column from tabular models.",
        })

    column_profiles = _build_column_profiles(
        df, spec, image_cols, target, missing_fractions=miss_exact, n_rows_full=n_rows_full,
    )
    bad_row_examples = _find_bad_row_examples(df)

    modeling_features = [p["name"] for p in column_profiles if p["include_in_modeling"]]
    excluded = [p["name"] for p in column_profiles if p["role"] in ("image", "id", "drop", "excluded", "time")]
    profile_drops = [p["name"] for p in column_profiles if p["role"] == "drop"]

    fe_notes: List[str] = []
    if image_cols:
        fe_notes.append(
            f"Image column(s) {', '.join(image_cols)} require dedicated image feature extraction "
            "(CNN embeddings, histogram features) or a vision model — excluded from baseline tabular pipeline."
        )

    for p in column_profiles:
        if p["is_constant"]:
            miss_issues.append({
                "issue_type": "constant_column",
                "severity": "warning",
                "column": p["name"],
                "description": f"Column '{p['name']}' is constant.",
                "suggested_fix": "Drop before modeling.",
            })
        elif 0 < p["pct_missing"] <= MISSING_DROP_THRESHOLD:
            miss_issues.append({
                "issue_type": "missing_values",
                "severity": "info" if p["pct_missing"] < 0.1 else "warning",
                "column": p["name"],
                "description": f"Column '{p['name']}' has {p['pct_missing']:.1%} missing values.",
                "suggested_fix": "Impute during preprocessing (below drop threshold).",
            })

    for col in image_cols:
        miss_issues.append({
            "issue_type": "image_column",
            "severity": "critical",
            "column": col,
            "description": f"Column '{col}' stores image/path data — not usable as raw tabular feature.",
            "suggested_fix": "Exclude from baseline models or engineer image features.",
        })

    drops = list(name_drops) + [c for c in id_drops if c not in {target}] + profile_drops
    issues = name_issues + value_issues + cat_issues + uniq_issues + miss_issues + dup_issues

    safe_renames = {}
    for old, new in renames.items():
        if old == target:
            continue
        safe_renames[old] = new
    drops = [c for c in drops if c != target]
    drops = sorted(set(drops))

    plan = _build_remediation_plan(
        safe_renames, drops, category_maps, coerce_cols, drop_duplicates,
        modeling_features=modeling_features, feature_engineering_notes=fe_notes,
    )
    summary_parts = [
        f"Scanned {n_rows_full} rows × {df.shape[1]} columns.",
        f"{len(issues)} issue(s) found.",
        f"Target: {target or 'unknown'}.",
        f"{len(modeling_features)} tabular feature(s) selected.",
    ]
    if plan_info["use_analysis_sample"]:
        summary_parts.append(
            f"Analysis used a stratified sample of {len(df)} rows "
            f"(file {plan_info['file_bytes'] // (1024 * 1024)} MB); "
            f"missingness computed on full file."
        )
    if image_cols:
        summary_parts.append(f"{len(image_cols)} image column(s) excluded.")
    if plan["column_renames"]:
        summary_parts.append(f"{len(plan['column_renames'])} column rename(s) proposed.")
    if plan["columns_to_drop"]:
        summary_parts.append(f"{len(plan['columns_to_drop'])} column(s) to drop.")

    column_renames = [
        {"original": o, "cleaned": n, "reason": "normalize name"}
        for o, n in plan["column_renames"].items()
    ]

    task_hint = spec.get("ml_task_type") or ""
    if target and not task_hint:
        tcol = df[target]
        n_classes = int(tcol.nunique(dropna=True))
        if n_classes == 2:
            task_hint = "binary_classification"
        elif 2 < n_classes <= 50:
            task_hint = "multiclass_classification"
        elif pd.api.types.is_numeric_dtype(tcol):
            task_hint = "regression"

    return {
        "summary": " ".join(summary_parts),
        "original_csv_path": csv_path,
        "n_rows_before": int(n_rows_full),
        "n_cols_before": int(df.shape[1]),
        "target_column": target,
        "task_type_hint": task_hint,
        "image_columns": image_cols,
        "column_profiles": column_profiles,
        "bad_row_examples": bad_row_examples,
        "modeling_features": modeling_features,
        "excluded_columns": excluded,
        "feature_engineering_notes": fe_notes,
        "issues": issues,
        "remediation_plan": plan,
        "column_renames": column_renames,
    }


@server.tool("Apply a remediation plan and write cleaned CSV.", {
    **_S,
    "plan": {"type": "object", "required": True},
    "project_id": {"type": "string", "required": True},
})
def apply_remediation(args: Dict[str, Any]) -> Dict[str, Any]:
    csv_path = args["csv_path"]
    plan = args.get("plan") or {}
    project_id = args["project_id"]
    plan_info = csv_load_plan(csv_path)
    src = Path(csv_path)
    out_name = f"cleaned_{src.name}"
    out_path = file_store.project_upload_dir(project_id) / out_name
    actions: List[str] = []

    def _transform(chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()
        renames = plan.get("column_renames") or {}
        if renames:
            df = df.rename(columns=renames)
        for col in plan.get("columns_to_drop") or []:
            if col in df.columns:
                df = df.drop(columns=[col])
        if plan.get("strip_string_columns", True):
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        for col, mapping in (plan.get("category_maps") or {}).items():
            if col in df.columns:
                df[col] = df[col].astype(str).replace(mapping)
        for col in plan.get("coerce_numeric_columns") or []:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    if plan_info["file_bytes"] >= settings.large_csv_bytes:
        n_before = plan_info["estimated_rows"]
        transform_csv_chunked(csv_path, str(out_path), _transform)
        if plan.get("drop_duplicate_rows"):
            dedup_path = out_path.with_suffix(".dedup.csv")
            n_after = drop_duplicates_streaming(str(out_path), str(dedup_path))
            dedup_path.replace(out_path)
            if n_before > n_after:
                actions.append(f"Dropped {n_before - n_after} duplicate row(s) (streaming).")
        else:
            n_after = estimate_row_count(str(out_path))
        if plan.get("column_renames"):
            actions.append(f"Renamed {len(plan.get('column_renames') or {})} column(s).")
        for col in plan.get("columns_to_drop") or []:
            actions.append(f"Dropped column '{col}'.")
        if plan.get("strip_string_columns", True):
            actions.append("Stripped whitespace on string columns.")
        for col, mapping in (plan.get("category_maps") or {}).items():
            actions.append(f"Normalized categories in '{col}' ({len(mapping)} mapping(s)).")
        for col in plan.get("coerce_numeric_columns") or []:
            actions.append(f"Coerced '{col}' to numeric.")
        df_head = pd.read_csv(out_path, nrows=5)
        return {
            "cleaned_csv_path": str(out_path),
            "n_rows_after": int(n_after),
            "n_cols_after": int(len(df_head.columns)),
            "actions_applied": actions,
            "columns": list(df_head.columns),
        }

    df = load_csv(csv_path, for_analysis=False)

    renames = plan.get("column_renames") or {}
    if renames:
        df = df.rename(columns=renames)
        actions.append(f"Renamed {len(renames)} column(s).")

    for col in plan.get("columns_to_drop") or []:
        if col in df.columns:
            df = df.drop(columns=[col])
            actions.append(f"Dropped column '{col}'.")

    if plan.get("strip_string_columns", True):
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        actions.append("Stripped whitespace on string columns.")

    for col, mapping in (plan.get("category_maps") or {}).items():
        if col in df.columns:
            df[col] = df[col].astype(str).replace(mapping)
            actions.append(f"Normalized categories in '{col}' ({len(mapping)} mapping(s)).")

    for col in plan.get("coerce_numeric_columns") or []:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            actions.append(f"Coerced '{col}' to numeric.")

    n_before = len(df)
    if plan.get("drop_duplicate_rows"):
        df = df.drop_duplicates()
        removed = n_before - len(df)
        if removed:
            actions.append(f"Dropped {removed} duplicate row(s).")

    src = Path(csv_path)
    out_name = f"cleaned_{src.name}"
    out_path = file_store.project_upload_dir(project_id) / out_name
    df.to_csv(out_path, index=False)

    return {
        "cleaned_csv_path": str(out_path),
        "n_rows_after": int(len(df)),
        "n_cols_after": int(df.shape[1]),
        "actions_applied": actions,
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    serve_fastmcp(server)
