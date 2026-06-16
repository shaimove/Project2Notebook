"""Schema for the Project Understanding output (project_spec.json)."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

MLTaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "multi_output_regression",
    "anomaly_detection",
    "forecasting",
    "ranking",
    "clustering",
    "unknown",
]

SplitStrategy = Literal["random", "stratified", "grouped", "time_based", "unknown"]


class ProjectSpec(BaseModel):
    """Structured interpretation of the project brief."""

    business_goal: str = ""
    ml_task_type: MLTaskType = "unknown"
    targets: List[str] = Field(default_factory=list)
    unit_of_prediction: str = ""
    unit_of_analysis: str = ""
    aggregation: str = ""
    has_time_component: bool = False
    recommended_split: SplitStrategy = "unknown"
    leakage_risks: List[str] = Field(default_factory=list)
    primary_metric: str = ""
    secondary_metrics: List[str] = Field(default_factory=list)
    success_criteria: str = ""
    assumptions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


class PriorArtReport(BaseModel):
    """Output of the Prior Art Agent (prior_art_report.json)."""

    enabled: bool = False
    message: str = ""
    searched_queries: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    common_models: List[str] = Field(default_factory=list)
    common_feature_engineering: List[str] = Field(default_factory=list)
    common_preprocessing: List[str] = Field(default_factory=list)
    common_split_strategies: List[str] = Field(default_factory=list)
    common_metrics: List[str] = Field(default_factory=list)
    leakage_warnings: List[str] = Field(default_factory=list)
    ideas_to_test: List[str] = Field(default_factory=list)
    ideas_to_ignore: List[str] = Field(default_factory=list)
    summary: str = ""
