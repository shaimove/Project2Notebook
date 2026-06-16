"""Prior Art MCP Server.

Provides domain-knowledge / prior-art lookup for a given ML problem type.

Honesty rules (per spec):
- If ``ENABLE_WEB_SEARCH`` is false (default), ``search_prior_art`` returns a
  clearly-marked placeholder. It NEVER pretends a web search happened.
- ``summarize_common_approaches`` returns a *curated offline knowledge base* of
  common approaches by task type. This is explicitly labelled as offline
  heuristics, not web results. All suggestions are inspiration only and must be
  verified against the real dataset by the downstream agents.
"""
from __future__ import annotations

from typing import Any, Dict, List

from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(name="prior-art-tools", description="Domain knowledge / prior-art lookup.")

# Curated, offline knowledge base keyed by task type. Inspiration only.
_KNOWLEDGE: Dict[str, Dict[str, List[str]]] = {
    "binary_classification": {
        "common_models": ["Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "SVM"],
        "common_feature_engineering": [
            "Aggregate behavioural counts over time windows",
            "Ratios between recent and long-term activity",
            "Recency/frequency features",
            "Target/mean encoding for high-cardinality categoricals (fit on train only)",
        ],
        "common_preprocessing": ["Median imputation", "One-hot / target encoding", "Standard scaling for linear/SVM"],
        "common_split_strategies": ["Stratified split", "Time-based split when there is a time component", "Grouped split by entity id"],
        "common_metrics": ["ROC-AUC", "PR-AUC", "F1", "Recall at fixed precision"],
        "leakage_warnings": [
            "Do not use post-outcome activity windows as features",
            "Do not leak the label via aggregates computed over the full period",
        ],
        "ideas_to_test": [
            "Handle class imbalance via class weights or resampling",
            "Tune gradient boosting capacity",
            "Tune the decision threshold for the business metric",
        ],
        "ideas_to_ignore": ["Deep learning on small tabular data (usually unnecessary)"],
    },
    "multiclass_classification": {
        "common_models": ["Logistic Regression (multinomial)", "Random Forest", "Gradient Boosting"],
        "common_feature_engineering": ["Class-aware aggregates", "Categorical encodings (fit on train)"],
        "common_preprocessing": ["Imputation", "One-hot encoding", "Scaling for linear models"],
        "common_split_strategies": ["Stratified split"],
        "common_metrics": ["Macro F1", "Balanced accuracy", "Accuracy"],
        "leakage_warnings": ["Avoid label-derived features"],
        "ideas_to_test": ["Tune boosting", "Address class imbalance"],
        "ideas_to_ignore": [],
    },
    "regression": {
        "common_models": ["Ridge/Lasso", "Random Forest", "Gradient Boosting"],
        "common_feature_engineering": ["Log-transform skewed targets/features", "Interaction terms", "Lag/rolling features for temporal data"],
        "common_preprocessing": ["Median imputation", "Scaling for linear models", "Outlier handling"],
        "common_split_strategies": ["Random split", "Time-based split for temporal data"],
        "common_metrics": ["RMSE", "MAE", "R^2"],
        "leakage_warnings": ["Avoid using future values as features"],
        "ideas_to_test": ["Tune boosting", "Stronger regularisation", "Target transformation"],
        "ideas_to_ignore": [],
    },
    "forecasting": {
        "common_models": ["Gradient Boosting on lag features", "Linear models on lags", "Classical (ARIMA/ETS) baselines"],
        "common_feature_engineering": ["Lag features y(t-1..t-k)", "Rolling mean/median/std", "Calendar features"],
        "common_preprocessing": ["Stationarity checks", "Scaling"],
        "common_split_strategies": ["Time-based split", "Rolling/expanding window CV"],
        "common_metrics": ["RMSE", "MAE", "MAPE"],
        "leakage_warnings": ["Never shuffle time", "Compute lags without using the future"],
        "ideas_to_test": ["Add more lag/rolling features", "Try different horizons"],
        "ideas_to_ignore": [],
    },
    "anomaly_detection": {
        "common_models": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Supervised model if labels exist"],
        "common_feature_engineering": ["Robust scaling", "Domain-specific deviation features"],
        "common_preprocessing": ["Robust scaling", "Handle heavy tails"],
        "common_split_strategies": ["Time-based for streaming data", "Stratified if rare labels exist"],
        "common_metrics": ["ROC-AUC", "PR-AUC (rare positives)"],
        "leakage_warnings": ["Avoid contaminating training with known anomalies if doing unsupervised"],
        "ideas_to_test": ["Compare unsupervised vs supervised when labels exist"],
        "ideas_to_ignore": [],
    },
}


@server.tool("Search prior art for a problem (web search if enabled, else placeholder).", {
    "spec": {"type": "object"}, "enable_web_search": {"type": "boolean"},
    "pdf_texts": {"type": "array"}})
def search_prior_art(args: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(args.get("enable_web_search"))
    spec = args.get("spec") or {}
    task = spec.get("ml_task_type", "unknown")
    queries = [
        f"common approaches for {task.replace('_', ' ')}",
        f"feature engineering for {spec.get('business_goal', task)[:60]}",
        "leakage pitfalls and validation strategy",
    ]
    if not enabled:
        return {
            "enabled": False,
            "message": "Prior-art web search is disabled. Enable ENABLE_WEB_SEARCH=true and configure a search provider.",
            "searched_queries": queries,
            "sources": [],
        }
    # Web search provider not implemented in this MVP. Be explicit rather than fake.
    return {
        "enabled": True,
        "message": "ENABLE_WEB_SEARCH=true but no web-search provider is wired in this MVP. Returning curated offline knowledge instead. Implement a provider in prior_art_server.search_prior_art.",
        "searched_queries": queries,
        "sources": [],
    }


@server.tool("Summarise common approaches for a task (curated offline knowledge).", {
    "ml_task_type": {"type": "string"}})
def summarize_common_approaches(args: Dict[str, Any]) -> Dict[str, Any]:
    task = args.get("ml_task_type", "unknown")
    kb = _KNOWLEDGE.get(task) or _KNOWLEDGE.get("binary_classification")
    return {"source": "curated_offline_knowledge_base", **kb}


@server.tool("Extract candidate strategies (ideas to test / ignore) from approaches.", {
    "approaches": {"type": "object"}, "pdf_texts": {"type": "array"}})
def extract_candidate_strategies(args: Dict[str, Any]) -> Dict[str, Any]:
    approaches = args.get("approaches") or {}
    return {
        "ideas_to_test": approaches.get("ideas_to_test", []),
        "ideas_to_ignore": approaches.get("ideas_to_ignore", []),
    }


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
