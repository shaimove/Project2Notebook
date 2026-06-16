"""Unit tests for Pydantic validation helpers and schemas."""
from __future__ import annotations

import pytest

from backend.exceptions import DomainValidationError
from backend.schemas.data_audit import DatasetSummary, TargetDistribution
from backend.schemas.experiment import (
    ModelComparisonResult,
    ModelEvaluationResult,
    ModelResult,
)
from backend.schemas.validation import (
    parse_tool_results,
    try_validate_model,
    validate_model,
)


def test_validate_model_accepts_model_result():
    result = validate_model(
        ModelResult,
        {
            "model_name": "boosting",
            "family": "boosting",
            "valid_score": 0.91,
            "primary_metric": "roc_auc",
        },
        context="train",
    )
    assert result.model_name == "boosting"
    assert result.valid_score == 0.91


def test_validate_model_raises_domain_validation_error():
    with pytest.raises(DomainValidationError, match="compare_models"):
        validate_model(ModelComparisonResult, {"rows": [{"model": 123}]}, context="compare_models")


def test_try_validate_model_returns_none_for_invalid_payload():
    assert try_validate_model(ModelResult, {"family": "tree"}, context="train") is None


def test_parse_tool_results_skips_invalid_training_outputs():
    valid, skipped = parse_tool_results(
        ModelResult,
        [
            {"model_name": "dummy", "family": "dummy", "valid_score": 0.5},
            {"family": "broken"},
        ],
        context="model training",
    )
    assert len(valid) == 1
    assert valid[0]["model_name"] == "dummy"
    assert skipped == ["?"]


def test_dataset_summary_validation():
    summary = validate_model(
        DatasetSummary,
        {"n_rows": 10, "n_cols": 2, "columns": ["a", "b"], "dtypes": {"a": "float64"}},
    )
    assert summary.columns == ["a", "b"]


def test_model_evaluation_result_validation():
    evaluation = validate_model(
        ModelEvaluationResult,
        {
            "model_name": "best",
            "split": "test",
            "metrics": {"roc_auc": 0.8},
            "primary_metric": "roc_auc",
            "score": 0.8,
        },
        context="test evaluation",
    )
    assert evaluation.score == 0.8


def test_target_distribution_validation():
    dist = validate_model(TargetDistribution, {"kind": "categorical", "n_classes": 2})
    assert dist.kind == "categorical"
