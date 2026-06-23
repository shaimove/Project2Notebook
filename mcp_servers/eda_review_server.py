"""EDA Review MCP Server — deterministic analysis of EDA tables + plot inventory.

The EDA Review *agent* calls these tools, then uses the LLM (vision) on plots
when configured. Tables provide reproducible numeric evidence; plots provide
semantic patterns the LLM reads best.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backend.services import artifact_store
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="eda-review-tools",
    description="Analyze EDA tables and inventory plots for the EDA Review agent.",
)

_PLOT_CATALOG = {
    "target_distribution.png": "Target variable distribution (balance/spread)",
    "missingness.png": "Missing values fraction per column",
    "feature_distributions.png": "Numeric feature histograms (skew/outliers)",
    "feature_target.png": "Feature vs target relationships (boxplots/scatter)",
    "correlation.png": "Numeric correlation heatmap",
    "time_series.png": "Target over time (trend/seasonality)",
}


def _read_table(project_id: str, name: str) -> Optional[pd.DataFrame]:
    path = artifact_store.tables_dir(project_id) / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _target_correlation_insights(
    corr: pd.DataFrame, target: Optional[str], task: str
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    insights: List[Dict[str, Any]] = []
    important: List[str] = []
    to_drop: List[str] = []
    if target is None or target not in corr.columns:
        return insights, important, to_drop

    is_regression = task in ("regression", "multi_output_regression", "forecasting")
    threshold_high = 0.15 if is_regression else 0.08
    threshold_low = 0.02

    for col in corr.columns:
        if col == target:
            continue
        try:
            r = float(corr.loc[col, target])
        except Exception:
            continue
        if abs(r) >= threshold_high:
            insights.append({
                "feature": col,
                "insight_type": "high_target_correlation",
                "evidence": f"corr({col}, {target}) = {round(r, 3)}",
                "evidence_source": "correlation.csv",
                "confidence": min(0.95, 0.5 + abs(r)),
                "recommended_action": "keep",
                "rationale": "Strong linear association with the target.",
            })
            important.append(col)
        elif abs(r) <= threshold_low:
            insights.append({
                "feature": col,
                "insight_type": "low_target_correlation",
                "evidence": f"corr({col}, {target}) = {round(r, 3)}",
                "evidence_source": "correlation.csv",
                "confidence": 0.7,
                "recommended_action": "watch" if is_regression else "drop",
                "rationale": "Very weak linear association; candidate for removal after review.",
            })
            if not is_regression:
                to_drop.append(col)
    return insights, important, to_drop


def _multicollinear_groups(corr: pd.DataFrame, threshold: float = 0.9) -> List[List[str]]:
    cols = list(corr.columns)
    groups: List[List[str]] = []
    seen: Set[str] = set()
    for i, a in enumerate(cols):
        group = [a]
        for b in cols[i + 1 :]:
            try:
                r = abs(float(corr.loc[a, b]))
            except Exception:
                continue
            if r >= threshold:
                group.append(b)
        if len(group) > 1:
            key = tuple(sorted(group))
            if key not in seen:
                seen.add(key)
                groups.append(sorted(group))
    return groups


@server.tool(
    "Deterministic analysis of EDA CSV tables (correlation, outliers, missingness, numeric summary).",
    {
        "project_id": {"type": "string", "required": True},
        "data_audit": {"type": "object"},
        "spec": {"type": "object"},
    },
)
def analyze_eda_tables(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    audit = args.get("data_audit") or {}
    spec = args.get("spec") or {}
    target = audit.get("target")
    task = spec.get("ml_task_type", "unknown")
    leakage_prone = set(audit.get("leakage_prone_columns") or [])

    insights: List[Dict[str, Any]] = []
    important: List[str] = []
    to_drop: List[str] = []
    to_watch: List[str] = []
    to_engineer: List[Dict[str, Any]] = []
    multicollinear: List[List[str]] = []
    preprocessing: List[str] = []
    target_insights: List[str] = []
    sources: List[str] = []

    corr_df = _read_table(project_id, "correlation.csv")
    if corr_df is not None and not corr_df.empty:
        corr_df = corr_df.set_index(corr_df.columns[0])
        sources.append("correlation.csv")
        c_ins, c_imp, c_drop = _target_correlation_insights(corr_df, target, task)
        insights.extend(c_ins)
        important.extend(c_imp)
        to_drop.extend(c_drop)
        multicollinear = _multicollinear_groups(corr_df)
        for grp in multicollinear:
            for col in grp:
                if col == target:
                    continue
                insights.append({
                    "feature": col,
                    "insight_type": "multicollinear",
                    "evidence": f"Group {grp} has |r| >= 0.9",
                    "evidence_source": "correlation.csv",
                    "confidence": 0.85,
                    "recommended_action": "watch",
                    "rationale": "Redundant with other features; keep one representative.",
                })
            if target in corr_df.columns:
                rep = max(
                    (c for c in grp if c != target),
                    key=lambda c: abs(float(corr_df.loc[c, target]))
                    if c in corr_df.index
                    else 0,
                    default=grp[0],
                )
                for c in grp:
                    if c != rep and c != target and c not in leakage_prone:
                        to_drop.append(c)

    outliers_df = _read_table(project_id, "outliers.csv")
    if outliers_df is not None and not outliers_df.empty:
        sources.append("outliers.csv")
        for _, row in outliers_df.iterrows():
            col = str(row.get("column", ""))
            pct = float(row.get("pct", 0) or 0)
            if pct >= 0.05:
                insights.append({
                    "feature": col,
                    "insight_type": "heavy_outliers",
                    "evidence": f"{pct:.1%} rows are IQR outliers",
                    "evidence_source": "outliers.csv",
                    "confidence": 0.8,
                    "recommended_action": "clip_outliers",
                    "rationale": "Heavy tails; consider clipping or robust models.",
                })
                to_watch.append(col)
                preprocessing.append(f"Consider clipping or robust scaling for '{col}'.")

    missing_df = _read_table(project_id, "missingness.csv")
    if missing_df is not None and not missing_df.empty:
        sources.append("missingness.csv")
        col_name = missing_df.columns[0]
        for _, row in missing_df.iterrows():
            col = str(row.get(col_name, row.iloc[0]))
            frac = float(row.iloc[1]) if len(row) > 1 else 0.0
            if frac >= 0.4:
                insights.append({
                    "feature": col,
                    "insight_type": "high_missing",
                    "evidence": f"{frac:.1%} missing",
                    "evidence_source": "missingness.csv",
                    "confidence": 0.9,
                    "recommended_action": "drop",
                    "rationale": "Too sparse to impute reliably.",
                })
                if col not in leakage_prone and col != target:
                    to_drop.append(col)

    summary_df = _read_table(project_id, "numeric_summary.csv")
    if summary_df is not None and not summary_df.empty:
        sources.append("numeric_summary.csv")
        idx_col = summary_df.columns[0]
        for _, row in summary_df.iterrows():
            col = str(row.get(idx_col, row.iloc[0]))
            skew = row.get("skew") if "skew" in summary_df.columns else None
            if skew is not None:
                try:
                    skew_v = float(skew)
                except Exception:
                    continue
                if abs(skew_v) >= 2:
                    insights.append({
                        "feature": col,
                        "insight_type": "skewed",
                        "evidence": f"skew = {round(skew_v, 2)}",
                        "evidence_source": "numeric_summary.csv",
                        "confidence": 0.75,
                        "recommended_action": "log_transform",
                        "rationale": "Strong skew; log1p or yeo-johnson may help linear models.",
                    })
                    to_engineer.append({
                        "base_column": col,
                        "transform": "log1p",
                        "rationale": f"Skew={round(skew_v, 2)} from numeric summary.",
                        "priority": 2,
                    })

    target_df = _read_table(project_id, "target_distribution.csv")
    if target_df is not None and audit.get("class_imbalance"):
        sources.append("target_distribution.csv")
        ratio = audit.get("class_imbalance")
        target_insights.append(f"Class imbalance ratio ~{ratio}; consider class weights.")
        preprocessing.append("Use stratified split and class-weighted models.")

    for col in leakage_prone:
        if col not in to_drop:
            to_drop.append(col)

    # De-duplicate while preserving order
    def _uniq(items: List[str]) -> List[str]:
        out: List[str] = []
        for x in items:
            if x and x not in out and x != target:
                out.append(x)
        return out

    return {
        "feature_insights": insights,
        "important_columns": _uniq(important),
        "features_to_drop": _uniq(to_drop),
        "features_to_watch": _uniq(to_watch),
        "features_to_engineer": to_engineer,
        "multicollinear_groups": multicollinear,
        "preprocessing_implications": preprocessing,
        "target_insights": target_insights,
        "analysis_sources": sources,
    }


@server.tool(
    "List EDA plots available for LLM vision review with semantic descriptions.",
    {"project_id": {"type": "string", "required": True}, "plot_names": {"type": "array"}},
)
def list_eda_plot_inventory(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    names = args.get("plot_names") or []
    plots_dir = artifact_store.plots_dir(project_id)
    inventory: List[Dict[str, Any]] = []
    for name in names:
        path = plots_dir / Path(name).name
        if not path.exists():
            continue
        inventory.append({
            "plot_name": path.name,
            "plot_type": _PLOT_CATALOG.get(path.name, "eda_plot"),
            "path": str(path),
            "description": _PLOT_CATALOG.get(path.name, "EDA visualization"),
        })
    return {"plots": inventory}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
