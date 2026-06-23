"""Preprocessing MCP Server.

Leakage-aware preprocessing:
- The split is created first (random / stratified / grouped / time-based).
- The preprocessor (impute/encode/scale) is fit ONLY on the training split.
- The fitted preprocessor is then applied to validation and test splits.
- The test split is never touched until final evaluation (the runner enforces
  this by only feeding test arrays to the final-evaluation node).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from mcp_servers._mlcommon import (
    infer_numeric_categorical,
    load_csv,
    prepared_dir,
    save_prepared,
)
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="preprocessing-tools",
    description="Leakage-aware split + fit-on-train preprocessing.",
)

CLASSIFICATION_TASKS = {"binary_classification", "multiclass_classification", "anomaly_detection"}


def _ohe() -> OneHotEncoder:
    # sklearn renamed the arg in 1.2; support both.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_names_out(ct: ColumnTransformer, numeric: List[str], categorical: List[str]) -> List[str]:
    """Compute output feature names robustly across sklearn versions.

    sklearn < 1.1 does not implement ``get_feature_names_out`` on SimpleImputer
    inside a Pipeline, so we build the names manually (numeric pass-through +
    one-hot expansion), matching the transformer order used in fit.
    """
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names: List[str] = list(numeric)
        if categorical:
            try:
                ohe = ct.named_transformers_["cat"].named_steps["ohe"]
                try:
                    cat_names = list(ohe.get_feature_names_out(categorical))
                except Exception:
                    cat_names = list(ohe.get_feature_names(categorical))
                names += cat_names
            except Exception:
                names += [f"cat_{i}" for i in range(0)]
        return names


@server.tool("Create a heuristic preprocessing plan from profile + spec + EDA findings.", {
    "profile": {"type": "object"}, "spec": {"type": "object"}, "eda_findings": {"type": "object"}})
def create_preprocessing_plan(args: Dict[str, Any]) -> Dict[str, Any]:
    profile = args.get("profile") or {}
    spec = args.get("spec") or {}
    eda = args.get("eda_findings") or {}
    targets: List[str] = spec.get("targets") or []
    target = targets[0] if targets else None
    leakage_cols = set()
    drop, keep, numeric, categorical, fe, lag = [], [], [], [], [], []
    missing_strategy: Dict[str, str] = {}

    leak_terms = ("future", "next", "post_", "_after", "outcome", "_next")
    risk_text = " ".join(str(r).lower() for r in (spec.get("leakage_risks") or []))

    for col in profile.get("columns", []):
        name = col["name"]
        if name == target:
            continue
        if col.get("is_constant") or col.get("pct_missing", 0) > 0.6:
            drop.append(name)
            continue
        lname = name.lower()
        if lname == "id" or lname.endswith("_id"):
            drop.append(name)
            continue
        if any(t in lname for t in leak_terms) or lname in risk_text:
            drop.append(name)
            leakage_cols.add(f"Dropped leakage-prone column '{name}'.")
            continue
        if any(t in lname for t in ("timestamp", "date")) or lname == "time":
            drop.append(name)
            continue
        keep.append(name)
        is_numeric = col.get("dtype", "").startswith(("int", "float"))
        if is_numeric:
            numeric.append(name)
            if col.get("pct_missing", 0) > 0:
                missing_strategy[name] = "median"
        else:
            categorical.append(name)
            if col.get("pct_missing", 0) > 0:
                missing_strategy[name] = "most_frequent"

    # Merge EDA Review recommendations (never un-drop leakage-prone columns).
    eda_drop = set(eda.get("features_to_drop") or [])
    eda_watch = set(eda.get("features_to_watch") or [])
    eda_important = set(eda.get("important_columns") or [])
    for col in eda_drop:
        if col == target:
            continue
        if col in keep:
            keep.remove(col)
        if col not in drop:
            drop.append(col)
    for col in eda_important:
        if col in drop and col not in eda_drop:
            drop.remove(col)
            if col not in keep:
                keep.append(col)

    for eng in eda.get("features_to_engineer") or []:
        base = eng.get("base_column") or eng.get("base")
        transform = eng.get("transform", "engineer")
        if base:
            fe.append(f"{transform}({base}): {eng.get('rationale', 'from EDA review')}")

    if spec.get("has_time_component"):
        fe.append("Add lag/rolling features of the target where appropriate (y(t-1), rolling medians).")
        lag = ["y_lag_1", "y_rolling_median_5"]

    notes: List[str] = []
    if eda.get("summary"):
        notes.append(f"EDA review: {eda.get('summary', '')[:200]}")
    if eda_watch:
        notes.append("Watch columns: " + ", ".join(sorted(eda_watch)[:10]))
    for imp in eda.get("preprocessing_implications") or []:
        notes.append(str(imp))

    plan = {
        "drop_columns": sorted(set(drop)),
        "keep_columns": sorted(set(keep)),
        "feature_engineering": fe,
        "numeric_columns": [c for c in numeric if c in keep],
        "categorical_columns": [c for c in categorical if c in keep],
        "missing_value_strategy": missing_strategy,
        "encoding_strategy": "one_hot",
        "scaling_strategy": "standard",
        "aggregation": spec.get("aggregation", ""),
        "leakage_mitigations": [
            "Fit imputers/encoders/scalers on the training split only.",
            "Hold out the test set until final evaluation.",
        ]
        + list(leakage_cols),
        "lag_features": lag,
        "notes": notes,
    }
    return plan


def _make_target(y: pd.Series, task: str) -> "tuple[np.ndarray, Dict[str, Any]]":
    if task in CLASSIFICATION_TASKS:
        classes = sorted([str(c) for c in y.dropna().unique()])
        mapping = {c: i for i, c in enumerate(classes)}
        codes = y.astype(str).map(mapping).to_numpy()
        return codes, {"classes": classes, "mapping": mapping}
    return y.to_numpy(dtype="float64"), {}


@server.tool("Create train/valid/test split using the recommended strategy.", {
    "csv_path": {"type": "string", "required": True},
    "plan": {"type": "object", "required": True},
    "spec": {"type": "object", "required": True},
    "project_id": {"type": "string", "required": True}})
def build_train_valid_test_split(args: Dict[str, Any]) -> Dict[str, Any]:
    df = load_csv(args["csv_path"])
    plan = args["plan"]
    spec = args["spec"]
    project_id = args["project_id"]
    task = spec.get("ml_task_type", "unknown")
    strategy = spec.get("recommended_split", "random")
    targets = spec.get("targets") or []
    target = targets[0] if targets else None

    if not target or target not in df.columns:
        return {"error": f"target '{target}' not in dataset"}

    df = df.dropna(subset=[target]).reset_index(drop=True)
    feat_cols = [c for c in plan.get("keep_columns", []) if c in df.columns and c != target]
    X = df[feat_cols].copy()
    y_series = df[target]

    idx = np.arange(len(df))
    seed = 42
    group_col = None
    time_col = None
    stratified = False

    if strategy == "time_based" and spec.get("has_time_component"):
        tcols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if tcols:
            time_col = tcols[0]
            order = pd.to_datetime(df[time_col], errors="coerce").argsort(kind="stable").to_numpy()
            idx = order
            n = len(idx)
            n_test = int(n * 0.2)
            n_valid = int(n * 0.2)
            test_idx = idx[-n_test:]
            valid_idx = idx[-(n_test + n_valid):-n_test]
            train_idx = idx[:-(n_test + n_valid)]
        else:
            strategy = "random"
    if strategy == "grouped":
        gcols = [c for c in df.columns if c.lower().endswith("_id") or "customer" in c.lower() or "user" in c.lower()]
        if gcols:
            group_col = gcols[0]
            groups = df[group_col].to_numpy()
            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            trv_idx, test_idx = next(gss1.split(idx, groups=groups))
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_rel, va_rel = next(gss2.split(trv_idx, groups=groups[trv_idx]))
            train_idx, valid_idx = trv_idx[tr_rel], trv_idx[va_rel]
        else:
            strategy = "random"

    if strategy in ("random", "stratified"):
        strat = None
        if strategy == "stratified" and task in CLASSIFICATION_TASKS:
            strat = y_series.astype(str).to_numpy()
            stratified = True
        trv_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=strat
        )
        strat2 = strat[trv_idx] if strat is not None else None
        train_idx, valid_idx = train_test_split(
            trv_idx, test_size=0.2, random_state=seed, stratify=strat2
        )

    split_report = {
        "strategy": strategy,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
        "group_column": group_col,
        "time_column": time_col,
        "stratified": stratified,
        "rationale": f"Used '{strategy}' split to respect the data structure and avoid leakage.",
    }

    # Persist raw split indices and frames for the fit/transform tools.
    d = prepared_dir(project_id)
    np.savez(d / "split_idx.npz", train=train_idx, valid=valid_idx, test=test_idx)
    df.to_parquet(d / "frame.parquet") if _has_parquet() else df.to_csv(d / "frame.csv", index=False)
    (d / "split_meta.json").write_text(
        json.dumps({
            "feat_cols": feat_cols, "target": target, "task": task,
            "report": split_report,
        }, indent=2), encoding="utf-8")
    return split_report


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def _load_frame(d: Path) -> pd.DataFrame:
    if (d / "frame.parquet").exists():
        return pd.read_parquet(d / "frame.parquet")
    return pd.read_csv(d / "frame.csv")


@server.tool("Fit the preprocessing pipeline on the TRAIN split only.", {
    "project_id": {"type": "string", "required": True}})
def fit_preprocessor_on_train(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    d = prepared_dir(project_id)
    meta = json.loads((d / "split_meta.json").read_text(encoding="utf-8"))
    df = _load_frame(d)
    splits = np.load(d / "split_idx.npz")
    train_idx = splits["train"]

    feat_cols = meta["feat_cols"]
    numeric, categorical = infer_numeric_categorical(df[feat_cols], exclude=[])
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe())]),
            categorical,
        ))
    ct = ColumnTransformer(transformers, remainder="drop")
    ct.fit(df.iloc[train_idx][feat_cols])  # <-- fit on TRAIN ONLY (no leakage)
    joblib.dump(ct, d / "preprocessor.joblib")

    feature_names = _feature_names_out(ct, numeric, categorical)
    (d / "fitted_meta.json").write_text(
        json.dumps({"numeric": numeric, "categorical": categorical,
                    "n_features_out": len(feature_names)}, indent=2),
        encoding="utf-8",
    )
    return {
        "fitted_on": "train_split_only",
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "n_features_out": len(feature_names),
    }


@server.tool("Transform train/valid/test with the train-fitted preprocessor.", {
    "project_id": {"type": "string", "required": True}})
def transform_valid_test(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    d = prepared_dir(project_id)
    meta = json.loads((d / "split_meta.json").read_text(encoding="utf-8"))
    df = _load_frame(d)
    splits = np.load(d / "split_idx.npz")
    ct = joblib.load(d / "preprocessor.joblib")

    feat_cols = meta["feat_cols"]
    target = meta["target"]
    task = meta["task"]
    y_all, target_meta = _make_target(df[target], task)

    def block(name: str):
        i = splits[name]
        X = ct.transform(df.iloc[i][feat_cols])
        return np.asarray(X, dtype="float64"), y_all[i]

    Xtr, ytr = block("train")
    Xva, yva = block("valid")
    Xte, yte = block("test")
    fitted = json.loads((d / "fitted_meta.json").read_text(encoding="utf-8"))
    feature_names = _feature_names_out(ct, fitted.get("numeric", []), fitted.get("categorical", []))

    save_prepared(
        project_id,
        {
            "X_train": Xtr, "y_train": ytr,
            "X_valid": Xva, "y_valid": yva,
            "X_test": Xte, "y_test": yte,
        },
        feature_names,
        {"task": task, "target": target, "target_meta": target_meta,
         "n_features": len(feature_names)},
    )
    return {
        "train_shape": list(Xtr.shape),
        "valid_shape": list(Xva.shape),
        "test_shape": list(Xte.shape),
        "n_features": len(feature_names),
    }


@server.tool("Static leakage check of the preprocessing plan/split.", {
    "plan": {"type": "object"}, "spec": {"type": "object"}})
def check_preprocessing_leakage(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = args.get("plan") or {}
    spec = args.get("spec") or {}
    warnings_list: List[str] = []
    risks = spec.get("leakage_risks", []) or []
    kept = set(plan.get("keep_columns", []))
    for risk in risks:
        for col in kept:
            if col.lower() in str(risk).lower():
                warnings_list.append(f"Kept column '{col}' matches a known leakage risk: {risk}")
    if spec.get("has_time_component") and spec.get("recommended_split") not in ("time_based", "grouped"):
        warnings_list.append(
            "Data has a time component but split is not time-based; future info may leak."
        )
    return {
        "leakage_warnings": warnings_list,
        "fit_on_train_only": True,
        "test_held_out": True,
        "ok": len(warnings_list) == 0,
    }


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
