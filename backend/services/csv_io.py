"""Efficient CSV loading for datasets up to ~120 MB.

Strategy:
- **Upload**: allow up to ``settings.max_upload_bytes`` (default 150 MB).
- **Analysis** (data quality scan, audit profile, EDA plots): stratified row
  sample (default 100k rows) when the file is large; missingness computed
  exactly via chunked passes when needed.
- **Training / cleaning**: one full in-memory load (acceptable up to the user
  upload cap); remediation can stream chunk-by-chunk for very large files.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np
import pandas as pd

from backend.config import settings


def csv_file_size(csv_path: str) -> int:
    return Path(csv_path).stat().st_size


def estimate_row_count(csv_path: str) -> int:
    """Fast newline count (exact for well-formed CSV without embedded newlines)."""
    path = Path(csv_path)
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as fh:
        for _ in fh:
            count += 1
    return max(0, count - 1)


def csv_load_plan(csv_path: str) -> Dict[str, Any]:
    size = csv_file_size(csv_path)
    rows = estimate_row_count(csv_path)
    large_by_size = size >= settings.large_csv_bytes
    large_by_rows = rows > settings.csv_analysis_max_rows
    use_sample = large_by_size or large_by_rows
    use_chunked_stats = use_sample or size >= settings.large_csv_bytes
    return {
        "file_bytes": size,
        "estimated_rows": rows,
        "large_by_size": large_by_size,
        "large_by_rows": large_by_rows,
        "use_analysis_sample": use_sample,
        "use_chunked_stats": use_chunked_stats,
        "analysis_max_rows": settings.csv_analysis_max_rows,
    }


def iter_csv_chunks(
    csv_path: str,
    chunksize: Optional[int] = None,
    **read_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    size = chunksize or settings.csv_chunk_rows
    return pd.read_csv(csv_path, chunksize=size, low_memory=True, **read_kwargs)


def chunked_missing_fractions(csv_path: str) -> Dict[str, float]:
    """Exact per-column missing rate in a single streaming pass."""
    totals: Dict[str, int] = defaultdict(int)
    missing: Dict[str, int] = defaultdict(int)
    n_rows = 0
    for chunk in iter_csv_chunks(csv_path):
        n_rows += len(chunk)
        for col in chunk.columns:
            missing[col] += int(chunk[col].isna().sum())
            totals[col] += len(chunk)
    if n_rows == 0:
        return {}
    return {col: missing[col] / totals[col] for col in totals}


def chunked_duplicate_count(csv_path: str) -> int:
    """Count duplicate rows across the full file using row hashes."""
    seen: Set[int] = set()
    dupes = 0
    for chunk in iter_csv_chunks(csv_path):
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        for h in hashes:
            hv = int(h)
            if hv in seen:
                dupes += 1
            else:
                seen.add(hv)
    return dupes


def _stratified_sample(df: pd.DataFrame, n: int, target_column: Optional[str], seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    if not target_column or target_column not in df.columns:
        return df.sample(n=n, random_state=seed)
    parts: List[pd.DataFrame] = []
    groups = df[target_column].dropna().unique()
    if len(groups) < 2:
        return df.sample(n=n, random_state=seed)
    per = max(1, n // len(groups))
    for cls in groups:
        sub = df[df[target_column] == cls]
        parts.append(sub.sample(n=min(len(sub), per), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    elif len(out) < n:
        extra = df.drop(out.index, errors="ignore").sample(
            n=min(n - len(out), max(0, len(df) - len(out))),
            random_state=seed,
        )
        out = pd.concat([out, extra], ignore_index=True)
    return out


def reservoir_sample_csv(
    csv_path: str,
    n: int,
    target_column: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Reservoir / stratified sample without loading the full file into memory."""
    rng = np.random.default_rng(seed)
    if target_column:
        buckets: Dict[Any, List[pd.Series]] = defaultdict(list)
        per = max(1, n // 20)
        for chunk in iter_csv_chunks(csv_path):
            if target_column not in chunk.columns:
                break
            for cls, grp in chunk.groupby(target_column, dropna=False):
                rows = [grp.iloc[i] for i in range(len(grp))]
                bucket = buckets[cls]
                bucket.extend(rows)
                if len(bucket) > per * 2:
                    bucket[:] = list(rng.choice(bucket, size=per, replace=False))
        parts = []
        for bucket in buckets.values():
            if bucket:
                parts.append(pd.DataFrame(bucket))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            return _stratified_sample(df, n, target_column, seed)

    reservoir: Optional[pd.DataFrame] = None
    seen = 0
    for chunk in iter_csv_chunks(csv_path):
        for i in range(len(chunk)):
            seen += 1
            if reservoir is None:
                reservoir = chunk.iloc[[i]].copy()
            elif len(reservoir) < n:
                reservoir = pd.concat([reservoir, chunk.iloc[[i]]], ignore_index=True)
            else:
                j = int(rng.integers(0, seen))
                if j < n:
                    reservoir.iloc[j] = chunk.iloc[i].values
    return reservoir if reservoir is not None else pd.read_csv(csv_path, nrows=0)


def load_csv_for_analysis(
    csv_path: str,
    target_column: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a CSV or a representative sample for profiling / EDA."""
    plan = csv_load_plan(csv_path)
    cap = max_rows or settings.csv_analysis_max_rows
    if plan["use_analysis_sample"]:
        return reservoir_sample_csv(csv_path, cap, target_column=target_column)
    df = pd.read_csv(csv_path, low_memory=True)
    if len(df) > cap:
        return _stratified_sample(df, cap, target_column, seed=42)
    return df


def load_csv_full(csv_path: str) -> pd.DataFrame:
    """Load the entire CSV (used for split / cleaning up to the upload cap)."""
    return pd.read_csv(csv_path, low_memory=True)


def load_csv(
    csv_path: str,
    *,
    for_analysis: bool = False,
    target_column: Optional[str] = None,
) -> pd.DataFrame:
    if for_analysis:
        return load_csv_for_analysis(csv_path, target_column=target_column)
    return load_csv_full(csv_path)


def transform_csv_chunked(
    csv_path: str,
    out_path: str,
    transform,
    *,
    chunksize: Optional[int] = None,
) -> int:
    """Apply ``transform(chunk) -> chunk`` streaming to ``out_path``. Returns row count."""
    size = chunksize or settings.csv_chunk_rows
    total = 0
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=size, low_memory=True):
        out = transform(chunk)
        out.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        total += len(out)
    return total


def drop_duplicates_streaming(csv_path: str, out_path: str) -> int:
    """Write deduplicated CSV in a streaming pass."""
    seen: Set[int] = set()
    total = 0
    first = True
    for chunk in iter_csv_chunks(csv_path):
        hashes = pd.util.hash_pandas_object(chunk, index=False)
        keep_mask = []
        for h in hashes:
            hv = int(h)
            if hv in seen:
                keep_mask.append(False)
            else:
                seen.add(hv)
                keep_mask.append(True)
        kept = chunk.loc[keep_mask]
        if len(kept):
            kept.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
            first = False
            total += len(kept)
    return total
