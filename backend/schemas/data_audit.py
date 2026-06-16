"""Schemas for the data audit and EDA planning phases."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    n_missing: int = 0
    pct_missing: float = 0.0
    n_unique: int = 0
    is_constant: bool = False
    is_near_constant: bool = False
    sample_values: List[Any] = Field(default_factory=list)
    # numeric stats (optional)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    # categorical
    cardinality: Optional[int] = None
    inferred_role: str = "feature"  # feature | target | id | time | drop


class DataAuditReport(BaseModel):
    n_rows: int = 0
    n_cols: int = 0
    columns: List[ColumnProfile] = Field(default_factory=list)
    n_duplicate_rows: int = 0
    target: Optional[str] = None
    target_distribution: Dict[str, Any] = Field(default_factory=dict)
    class_imbalance: Optional[float] = None
    time_columns: List[str] = Field(default_factory=list)
    entity_columns: List[str] = Field(default_factory=list)
    leakage_prone_columns: List[str] = Field(default_factory=list)
    near_empty_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    high_cardinality_columns: List[str] = Field(default_factory=list)
    description_vs_data_mismatches: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class EDACheck(BaseModel):
    name: str
    kind: str  # plot | table | check
    rationale: str
    columns: List[str] = Field(default_factory=list)


class EDAPlan(BaseModel):
    checks: List[EDACheck] = Field(default_factory=list)
    summary: str = ""
