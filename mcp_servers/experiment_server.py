"""Experiment MCP Server.

Bookkeeping + decision logic for the iterative improvement loop:
- register experiments, compare runs
- suggest the next iteration (concrete, executable action proposals)
- run a small ablation test
- decide when to stop (3 iters / <5% relative improvement / overfitting / no
  supported hypothesis)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(name="experiment-tools", description="Iteration loop bookkeeping and decisions.")


@server.tool("Register an experiment run and return a run id.", {"run": {"type": "object"}})
def register_experiment(args: Dict[str, Any]) -> Dict[str, Any]:
    run = args.get("run") or {}
    run_id = run.get("model_name", "run") + "_" + str(run.get("iteration", 0))
    return {"run_id": run_id, "registered": True, "run": run}


@server.tool("Compare experiment runs and rank by validation score.", {"runs": {"type": "array"}})
def compare_experiment_runs(args: Dict[str, Any]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = args.get("runs") or []
    if not runs:
        return {"ranked": [], "best": None}
    higher = runs[0].get("higher_is_better", True)
    ranked = sorted(
        runs,
        key=lambda r: (r.get("valid_score") if r.get("valid_score") is not None else (-1e9 if higher else 1e9)),
        reverse=higher,
    )
    return {"ranked": ranked, "best": ranked[0]}


@server.tool("Propose the next iteration as a concrete, executable action.", {
    "spec": {"type": "object"}, "prior_art": {"type": "object"},
    "data_audit": {"type": "object"}, "best_result": {"type": "object"},
    "iteration": {"type": "integer"}})
def suggest_next_iteration(args: Dict[str, Any]) -> Dict[str, Any]:
    spec = args.get("spec") or {}
    prior_art = args.get("prior_art") or {}
    audit = args.get("data_audit") or {}
    best = args.get("best_result") or {}
    iteration = int(args.get("iteration", 1))
    task = spec.get("ml_task_type", "unknown")
    is_clf = task in {"binary_classification", "multiclass_classification", "anomaly_detection"}

    imbalance = audit.get("class_imbalance")
    imbalanced = bool(imbalance and imbalance > 1.5)
    prior_ideas = prior_art.get("ideas_to_test", []) if prior_art.get("enabled") else []

    # Ordered candidate actions; pick by iteration index, skipping N/A ones.
    candidates: List[Dict[str, Any]] = []
    if is_clf and imbalanced:
        candidates.append({
            "action": "class_weight",
            "model_name": "random_forest",
            "params": {"class_weight": "balanced"},
            "hypothesis": "Balancing class weights will improve minority-class detection.",
            "motivation": f"EDA found class imbalance (ratio ~{imbalance}).",
        })
    candidates.append({
        "action": "tune_boosting",
        "model_name": "boosting",
        "params": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
        "hypothesis": "A deeper, more thoroughly trained boosting model will fit non-linear structure.",
        "motivation": "Boosting is a strong baseline for tabular data (prior art) — tune capacity.",
    })
    candidates.append({
        "action": "tune_forest",
        "model_name": "random_forest",
        "params": {"n_estimators": 400, "min_samples_leaf": 2},
        "hypothesis": "A larger random forest reduces variance and may generalise better.",
        "motivation": "Ensemble averaging tends to reduce overfitting on tabular data.",
    })
    if is_clf:
        candidates.append({
            "action": "tune_linear",
            "model_name": "linear",
            "params": {"C": 0.5, "class_weight": "balanced"},
            "hypothesis": "Stronger regularisation + balancing improves a linear baseline.",
            "motivation": "Regularised linear models are robust, interpretable references.",
        })
    else:
        candidates.append({
            "action": "tune_ridge",
            "model_name": "linear",
            "params": {"alpha": 10.0},
            "hypothesis": "Stronger ridge regularisation reduces variance.",
            "motivation": "Regularisation helps when features are correlated.",
        })

    if iteration - 1 < len(candidates):
        chosen = candidates[iteration - 1]
        if prior_ideas:
            chosen = {**chosen, "prior_art_support": prior_ideas[: 2]}
        chosen["supported"] = True
        return chosen
    return {
        "action": "stop",
        "supported": False,
        "hypothesis": "",
        "motivation": "No further EDA/prior-art-supported change available.",
    }


@server.tool("Run a small ablation: retrain a model with overridden params and report delta.", {
    "project_id": {"type": "string", "required": True}, "spec": {"type": "object"},
    "model_name": {"type": "string", "required": True}, "params": {"type": "object"}})
def run_ablation_test(args: Dict[str, Any]) -> Dict[str, Any]:
    from mcp_servers import modeling_server

    base_name = args["model_name"]
    spec = args.get("spec") or {}
    project_id = args["project_id"]
    trainer = {
        "dummy": modeling_server.train_dummy_baseline,
        "linear": modeling_server.train_linear_model,
        "tree": modeling_server.train_tree_model,
        "random_forest": modeling_server.train_random_forest,
        "boosting": modeling_server.train_boosting_model,
        "svm": modeling_server.train_svm_model,
    }.get(base_name)
    if trainer is None:
        return {"error": f"unknown model {base_name}"}
    baseline = trainer({"project_id": project_id, "spec": spec})
    ablated = trainer({"project_id": project_id, "spec": spec, "params": args.get("params") or {}})
    return {
        "model_name": base_name,
        "baseline_valid_score": baseline.get("valid_score"),
        "ablated_valid_score": ablated.get("valid_score"),
        "delta": (ablated.get("valid_score") or 0) - (baseline.get("valid_score") or 0),
        "params": args.get("params") or {},
    }


@server.tool("Decide whether to stop the iteration loop.", {
    "iteration": {"type": "integer"}, "max_iterations": {"type": "integer"},
    "relative_improvement": {"type": "number"}, "min_relative_improvement": {"type": "number"},
    "overfit_suspected": {"type": "boolean"}, "hypothesis_supported": {"type": "boolean"}})
def stop_iteration_decision(args: Dict[str, Any]) -> Dict[str, Any]:
    iteration = int(args.get("iteration", 0))
    max_iter = int(args.get("max_iterations", 3))
    rel = args.get("relative_improvement")
    min_rel = float(args.get("min_relative_improvement", 0.05))
    overfit = bool(args.get("overfit_suspected", False))
    supported = bool(args.get("hypothesis_supported", True))

    if iteration >= max_iter:
        return {"stop": True, "reason": f"Reached the maximum of {max_iter} iterations."}
    if not supported:
        return {"stop": True, "reason": "Next change is not supported by EDA/prior-art/a clear hypothesis."}
    if overfit:
        return {"stop": True, "reason": "Suspected validation overfitting (large train-valid gap)."}
    if rel is not None and rel <= min_rel:
        return {"stop": True, "reason": f"Validation improvement {rel:.1%} <= {min_rel:.0%} threshold."}
    return {"stop": False, "reason": "Improvement is sufficient; continue iterating."}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
