"""Modeling MCP Server.

Trains baseline model families on the prepared (leakage-safe) splits, evaluates
on train/valid, and compares models. The TEST split is only evaluated via
``evaluate_model`` with split='test', which the runner calls exactly once at the
final-evaluation stage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import time

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import joblib

from mcp_servers._mlcommon import load_prepared, prepared_dir
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(name="modeling-tools", description="Train and evaluate baseline models.")

CLASSIFICATION = {"binary_classification", "multiclass_classification"}

HIGHER_IS_BETTER = {"accuracy", "f1", "f1_macro", "roc_auc", "pr_auc", "r2", "balanced_accuracy"}
LOWER_IS_BETTER = {"rmse", "mae", "mse"}

_METRIC_ALIASES = {
    "auc": "roc_auc",
    "roc-auc": "roc_auc",
    "roc auc": "roc_auc",
    "f1-score": "f1",
    "f1 score": "f1",
    "macro f1": "f1_macro",
    "root mean squared error": "rmse",
    "mean absolute error": "mae",
    "r^2": "r2",
    "r2 score": "r2",
}


def _models_dir(project_id: str) -> Path:
    d = prepared_dir(project_id) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_is_classification(task: str, y: np.ndarray) -> bool:
    if task in CLASSIFICATION:
        return True
    if task == "anomaly_detection":
        return True  # we evaluate against labels when present
    if task == "regression":
        return False
    # fallback inference
    return len(np.unique(y)) <= 15


def primary_metric_for(spec: Dict[str, Any], task: str, y: np.ndarray) -> Tuple[str, bool]:
    raw = (spec.get("primary_metric") or "").strip().lower()
    raw = _METRIC_ALIASES.get(raw, raw)
    known = HIGHER_IS_BETTER | LOWER_IS_BETTER
    if raw in known:
        return raw, raw in HIGHER_IS_BETTER
    if _task_is_classification(task, y):
        n_classes = len(np.unique(y))
        return ("roc_auc", True) if n_classes == 2 else ("f1_macro", True)
    return ("rmse", False)


def _evaluate(model, X, y, task: str) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    metrics: Dict[str, float] = {}
    if _task_is_classification(task, y):
        pred = model.predict(X)
        metrics["accuracy"] = float(accuracy_score(y, pred))
        avg = "binary" if len(np.unique(y)) == 2 else "macro"
        try:
            metrics["f1"] = float(f1_score(y, pred, average="binary")) if len(np.unique(y)) == 2 else float(f1_score(y, pred, average="macro"))
        except Exception:
            metrics["f1"] = 0.0
        metrics["f1_macro"] = float(f1_score(y, pred, average="macro"))
        # ROC-AUC for binary if probabilities available
        if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y, proba))
            except Exception:
                pass
    else:
        pred = model.predict(X)
        mse = float(mean_squared_error(y, pred))
        metrics["rmse"] = float(np.sqrt(mse))
        metrics["mae"] = float(mean_absolute_error(y, pred))
        metrics["r2"] = float(r2_score(y, pred))
    return metrics


def _builder(name: str, task: str, y: np.ndarray):
    is_clf = _task_is_classification(task, y)
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent") if is_clf else DummyRegressor()
    if name == "linear":
        return LogisticRegression(max_iter=1000) if is_clf else Ridge()
    if name == "tree":
        return DecisionTreeClassifier(max_depth=6, random_state=0) if is_clf else DecisionTreeRegressor(max_depth=6, random_state=0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1) if is_clf else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    if name == "boosting":
        try:
            from xgboost import XGBClassifier, XGBRegressor
            if is_clf:
                return XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.9, eval_metric="logloss", verbosity=0,
                    use_label_encoder=False, n_jobs=-1,
                )
            return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, n_jobs=-1, verbosity=0)
        except Exception:
            return HistGradientBoostingClassifier(random_state=0) if is_clf else HistGradientBoostingRegressor(random_state=0)
    if name == "svm":
        return SVC(probability=True) if is_clf else SVR()
    raise ValueError(f"unknown model {name}")


def _train_one(args: Dict[str, Any], name: str, family: str) -> Dict[str, Any]:
    project_id = args["project_id"]
    spec = args.get("spec") or {}
    prepared = load_prepared(project_id)
    if not prepared:
        return {"error": "no prepared data; run preprocessing first"}
    sp = prepared["splits"]
    task = prepared["meta"]["task"]
    Xtr, ytr, Xva, yva = sp["X_train"], sp["y_train"], sp["X_valid"], sp["y_valid"]

    extra = dict(args.get("params") or {})
    model = _builder(name, task, ytr)
    if extra and hasattr(model, "set_params"):
        try:
            model.set_params(**extra)
        except Exception:
            pass

    t0 = time.time()
    model.fit(Xtr, ytr)
    runtime = time.time() - t0

    train_m = _evaluate(model, Xtr, ytr, task)
    valid_m = _evaluate(model, Xva, yva, task)
    metric, higher = primary_metric_for(spec, task, ytr)
    gap = None
    if metric in train_m and metric in valid_m:
        gap = float(train_m[metric] - valid_m[metric])

    joblib.dump(model, _models_dir(project_id) / f"{name}.joblib")

    return {
        "model_name": name,
        "family": family,
        "train_metrics": train_m,
        "valid_metrics": valid_m,
        "primary_metric": metric,
        "higher_is_better": higher,
        "valid_score": valid_m.get(metric),
        "train_valid_gap": gap,
        "runtime_seconds": round(runtime, 3),
    }


_S = {"project_id": {"type": "string", "required": True}, "spec": {"type": "object"}}


@server.tool("Train a dummy baseline.", _S)
def train_dummy_baseline(args: Dict[str, Any]) -> Dict[str, Any]:
    return _train_one(args, "dummy", "baseline")


@server.tool("Train a (regularized) linear/logistic model.", _S)
def train_linear_model(args: Dict[str, Any]) -> Dict[str, Any]:
    return _train_one(args, "linear", "linear")


@server.tool("Train a decision tree model.", _S)
def train_tree_model(args: Dict[str, Any]) -> Dict[str, Any]:
    return _train_one(args, "tree", "tree")


@server.tool("Train a random forest model.", _S)
def train_random_forest(args: Dict[str, Any]) -> Dict[str, Any]:
    return _train_one(args, "random_forest", "ensemble")


@server.tool("Train a gradient boosting model (xgboost or HistGradientBoosting).", _S)
def train_boosting_model(args: Dict[str, Any]) -> Dict[str, Any]:
    return _train_one(args, "boosting", "boosting")


@server.tool("Train an SVM model.", _S)
def train_svm_model(args: Dict[str, Any]) -> Dict[str, Any]:
    return _train_one(args, "svm", "svm")


@server.tool("Evaluate a stored model on a split (train/valid/test).", {
    **_S, "model_name": {"type": "string", "required": True},
    "split": {"type": "string", "required": True}})
def evaluate_model(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    spec = args.get("spec") or {}
    split = args.get("split", "valid")
    name = args["model_name"]
    prepared = load_prepared(project_id)
    if not prepared:
        return {"error": "no prepared data"}
    sp = prepared["splits"]
    task = prepared["meta"]["task"]
    X, y = sp[f"X_{split}"], sp[f"y_{split}"]
    model_path = _models_dir(project_id) / f"{name}.joblib"
    if not model_path.exists():
        return {"error": f"model '{name}' not found"}
    model = joblib.load(model_path)
    metrics = _evaluate(model, X, y, task)
    metric, higher = primary_metric_for(spec, task, y)
    return {
        "model_name": name,
        "split": split,
        "metrics": metrics,
        "primary_metric": metric,
        "score": metrics.get(metric),
        "higher_is_better": higher,
    }


@server.tool("Compare a list of model results into a ranked table.", {
    "results": {"type": "array", "required": True}})
def compare_models(args: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = args.get("results") or []
    if not results:
        return {"rows": [], "best_model": None}
    metric = results[0].get("primary_metric", "")
    higher = results[0].get("higher_is_better", True)
    rows = []
    for r in results:
        rows.append({
            "model": r.get("model_name"),
            "family": r.get("family"),
            "valid_score": r.get("valid_score"),
            "train_valid_gap": r.get("train_valid_gap"),
            "runtime_seconds": r.get("runtime_seconds"),
            "primary_metric": metric,
        })
    rows_sorted = sorted(
        rows,
        key=lambda x: (x["valid_score"] if x["valid_score"] is not None else (-1e9 if higher else 1e9)),
        reverse=higher,
    )
    return {"rows": rows_sorted, "best_model": rows_sorted[0]["model"], "primary_metric": metric}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
