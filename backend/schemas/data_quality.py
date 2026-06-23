"""Schemas for the Data Quality Review agent (data_quality_report.json)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["critical", "warning", "info"]
ColumnType = Literal[
    "continuous",
    "binary",
    "categorical",
    "free_text",
    "datetime",
    "image",
    "id",
    "constant",
    "unknown",
]
ColumnRole = Literal[
    "target",
    "train_feature",
    "id",
    "time",
    "image",
    "redundant",
    "drop",
    "excluded",
    "unknown",
]


class QualityIssue(BaseModel):
    issue_type: str
    severity: Severity = "info"
    column: Optional[str] = None
    description: str = ""
    suggested_fix: str = ""


class ColumnRename(BaseModel):
    original: str
    cleaned: str
    reason: str = ""


class ColumnProfileRow(BaseModel):
    name: str
    dtype: str = ""
    column_type: ColumnType = "unknown"
    role: ColumnRole = "unknown"
    pct_missing: float = 0.0
    n_unique: int = 0
    is_constant: bool = False
    description: str = ""
    sample_values: List[Any] = Field(default_factory=list)
    include_in_modeling: bool = False


class BadRowExample(BaseModel):
    issue_type: str
    description: str = ""
    row_indices: List[int] = Field(default_factory=list)
    preview: List[Dict[str, Any]] = Field(default_factory=list)


class RemediationPlan(BaseModel):
    column_renames: Dict[str, str] = Field(default_factory=dict)
    columns_to_drop: List[str] = Field(default_factory=list)
    category_maps: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    drop_duplicate_rows: bool = False
    coerce_numeric_columns: List[str] = Field(default_factory=list)
    strip_string_columns: bool = True
    modeling_features: List[str] = Field(default_factory=list)
    feature_engineering_notes: List[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    summary: str = ""
    original_csv_path: str = ""
    cleaned_csv_path: Optional[str] = None
    n_rows_before: int = 0
    n_rows_after: Optional[int] = None
    n_cols_before: int = 0
    n_cols_after: Optional[int] = None
    target_column: Optional[str] = None
    task_type_hint: str = ""
    image_columns: List[str] = Field(default_factory=list)
    column_profiles: List[ColumnProfileRow] = Field(default_factory=list)
    bad_row_examples: List[BadRowExample] = Field(default_factory=list)
    modeling_features: List[str] = Field(default_factory=list)
    excluded_columns: List[str] = Field(default_factory=list)
    issues: List[QualityIssue] = Field(default_factory=list)
    remediation_plan: RemediationPlan = Field(default_factory=RemediationPlan)
    column_renames: List[ColumnRename] = Field(default_factory=list)
    actions_applied: List[str] = Field(default_factory=list)
    feature_engineering_notes: List[str] = Field(default_factory=list)
