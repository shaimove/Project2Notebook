"""Shared agent state for the Project2Notebook workflow.

This mirrors a LangGraph-style state object. The in-repo orchestrator
(``graph.py``) passes a single ``DataScientist`` dict between nodes. Each node
reads what it needs and writes its own outputs back into the state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class DataScientist(TypedDict, total=False):
    project_id: str
    project_document_path: str
    csv_paths: List[str]
    original_csv_paths: List[str]
    cleaned_csv_path: Optional[str]
    pdf_paths: List[str]

    # Run configuration
    enable_prior_art: bool
    max_iterations: int
    min_relative_improvement: float

    # Phase artifacts
    project_spec: Optional[Dict[str, Any]]
    prior_art_report: Optional[Dict[str, Any]]
    data_quality_report: Optional[Dict[str, Any]]
    data_audit_report: Optional[Dict[str, Any]]
    eda_plan: Optional[Dict[str, Any]]
    eda_report: Optional[str]
    eda_artifacts: Optional[Dict[str, Any]]
    eda_findings: Optional[Dict[str, Any]]
    eda_findings_report: Optional[Dict[str, Any]]
    eda_plotly_html: Optional[str]
    eda_plotly_conclusions: Optional[List[str]]
    quality_plotly_html: Optional[str]
    audit_plotly_html: Optional[str]
    split_plotly_html: Optional[str]
    split_ratios: Optional[Dict[str, Any]]
    modeling_features: List[str]
    excluded_columns: List[str]
    preprocessing_plan: Optional[Dict[str, Any]]
    split_report: Optional[Dict[str, Any]]

    baseline_results: Optional[Dict[str, Any]]
    model_comparison: Optional[List[Dict[str, Any]]]
    first_conclusion: Optional[str]

    iteration_reports: List[Dict[str, Any]]
    iteration_summary: Optional[str]
    best_validation_score: Optional[float]
    best_pipeline_id: Optional[str]
    best_model_family: Optional[str]
    best_params: Optional[Dict[str, Any]]
    primary_metric: Optional[str]
    higher_is_better: Optional[bool]

    leakage_review: Optional[Dict[str, Any]]
    final_test_report: Optional[str]
    final_test_report_obj: Optional[Dict[str, Any]]
    notebook_path: Optional[str]

    # Transparency
    run_id: Optional[str]
    tool_calls: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


def new_state(
    project_id: str,
    project_document_path: str,
    csv_paths: List[str],
    pdf_paths: List[str],
    enable_prior_art: bool = True,
    max_iterations: int = 3,
    min_relative_improvement: float = 0.05,
) -> DataScientist:
    return DataScientist(
        project_id=project_id,
        project_document_path=project_document_path,
        csv_paths=csv_paths,
        original_csv_paths=[],
        cleaned_csv_path=None,
        pdf_paths=pdf_paths,
        enable_prior_art=enable_prior_art,
        max_iterations=max_iterations,
        min_relative_improvement=min_relative_improvement,
        project_spec=None,
        prior_art_report=None,
        data_quality_report=None,
        data_audit_report=None,
        eda_plan=None,
        eda_report=None,
        eda_artifacts=None,
        eda_findings=None,
        eda_findings_report=None,
        eda_plotly_html=None,
        eda_plotly_conclusions=None,
        quality_plotly_html=None,
        audit_plotly_html=None,
        split_plotly_html=None,
        split_ratios=None,
        modeling_features=[],
        excluded_columns=[],
        preprocessing_plan=None,
        split_report=None,
        baseline_results=None,
        model_comparison=None,
        first_conclusion=None,
        iteration_reports=[],
        iteration_summary=None,
        best_validation_score=None,
        best_pipeline_id=None,
        primary_metric=None,
        higher_is_better=None,
        final_test_report=None,
        final_test_report_obj=None,
        notebook_path=None,
        run_id=None,
        tool_calls=[],
        timeline=[],
        errors=[],
    )
