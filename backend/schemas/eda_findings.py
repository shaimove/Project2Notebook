"""Schemas for the EDA Review agent output (eda_findings.json)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

InsightType = Literal[
    "high_target_correlation",
    "low_target_correlation",
    "multicollinear",
    "skewed",
    "heavy_outliers",
    "high_missing",
    "high_cardinality",
    "leakage_suspect",
    "time_trend",
    "non_linear_pattern",
    "class_separation",
    "distribution_shape",
    "other",
]

RecommendedAction = Literal[
    "keep",
    "drop",
    "engineer",
    "bin",
    "log_transform",
    "clip_outliers",
    "watch",
    "investigate",
]


class FeatureInsight(BaseModel):
    feature: str
    insight_type: InsightType = "other"
    evidence: str = ""
    evidence_source: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    recommended_action: RecommendedAction = "watch"
    rationale: str = ""


class FeatureEngineeringIdea(BaseModel):
    base_column: str
    transform: str
    rationale: str = ""
    priority: int = Field(default=1, ge=1, le=5)


class PlotReview(BaseModel):
    plot_name: str
    plot_type: str = ""
    summary: str = ""
    key_observations: List[str] = Field(default_factory=list)
    features_mentioned: List[str] = Field(default_factory=list)
    reviewed_by: Literal["llm_vision", "heuristic", "skipped"] = "heuristic"


class EDAFindingsReport(BaseModel):
    summary: str = ""
    target_insights: List[str] = Field(default_factory=list)
    feature_insights: List[FeatureInsight] = Field(default_factory=list)
    important_columns: List[str] = Field(default_factory=list)
    features_to_drop: List[str] = Field(default_factory=list)
    features_to_engineer: List[FeatureEngineeringIdea] = Field(default_factory=list)
    features_to_watch: List[str] = Field(default_factory=list)
    multicollinear_groups: List[List[str]] = Field(default_factory=list)
    preprocessing_implications: List[str] = Field(default_factory=list)
    plot_reviews: List[PlotReview] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    analysis_sources: List[str] = Field(default_factory=list)
