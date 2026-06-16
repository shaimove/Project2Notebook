"""Schemas for preprocessing, modeling and the iteration loop."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PreprocessingPlan(BaseModel):
    drop_columns: List[str] = Field(default_factory=list)
    keep_columns: List[str] = Field(default_factory=list)
    feature_engineering: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    missing_value_strategy: Dict[str, str] = Field(default_factory=dict)
    encoding_strategy: str = "one_hot"
    scaling_strategy: str = "standard"
    aggregation: str = ""
    leakage_mitigations: List[str] = Field(default_factory=list)
    lag_features: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class SplitReport(BaseModel):
    strategy: str = "random"
    train_rows: int = 0
    valid_rows: int = 0
    test_rows: int = 0
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    stratified: bool = False
    rationale: str = ""
    feature_count: Optional[int] = None


class ModelResult(BaseModel):
    """Output of a modeling-tools train_* call."""

    model_name: str
    family: str = ""
    train_metrics: Dict[str, float] = Field(default_factory=dict)
    valid_metrics: Dict[str, float] = Field(default_factory=dict)
    primary_metric: str = ""
    higher_is_better: bool = True
    valid_score: Optional[float] = None
    train_valid_gap: Optional[float] = None
    runtime_seconds: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class ModelComparisonRow(BaseModel):
    model: str
    family: Optional[str] = None
    valid_score: Optional[float] = None
    train_valid_gap: Optional[float] = None
    runtime_seconds: Optional[float] = None
    primary_metric: str = ""


class ModelComparisonResult(BaseModel):
    rows: List[ModelComparisonRow] = Field(default_factory=list)
    best_model: Optional[str] = None
    primary_metric: str = ""


class ModelEvaluationResult(BaseModel):
    """Output of modeling-tools evaluate_model."""

    model_name: str = ""
    split: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    primary_metric: str = ""
    score: Optional[float] = None
    higher_is_better: bool = True


class IterationSuggestion(BaseModel):
    """Output of experiment-tools suggest_next_iteration."""

    supported: bool = False
    action: str = "stop"
    model_name: str = "boosting"
    params: Dict[str, Any] = Field(default_factory=dict)
    hypothesis: str = ""
    motivation: str = ""


class IterationStopDecision(BaseModel):
    stop: bool = False
    reason: str = ""


class IterationReport(BaseModel):
    iteration: int
    hypothesis: str = ""
    motivation: str = ""
    change_description: str = ""
    model_name: str = ""
    valid_score: Optional[float] = None
    previous_best: Optional[float] = None
    relative_improvement: Optional[float] = None
    accepted: bool = False
    decision_reason: str = ""
    train_valid_gap: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class FinalTestReport(BaseModel):
    final_model: str = ""
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    valid_metrics: Dict[str, float] = Field(default_factory=dict)
    goal_met: Optional[bool] = None
    generalization_notes: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)
