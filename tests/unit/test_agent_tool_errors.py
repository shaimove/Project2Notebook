"""Per-agent tests verifying tool ``{"error": ...}`` handling."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.agents.nodes import (
    baseline_modeling,
    data_audit,
    eda_planning,
    executable_eda,
    final_test_evaluation,
    first_conclusion,
    iteration_loop,
    leakage_review,
    notebook_author,
    preprocessing_plan,
    prior_art,
    project_understanding,
)
from backend.exceptions import PipelineStepError, ToolError
from backend.services import artifact_store
from tests.conftest import MockMCPClient, make_state


def test_project_understanding_raises_on_load_dataset_error(isolated_storage):
    client = MockMCPClient(
        {
            ("project-understanding-tools", "parse_project_document"): {"text": ""},
            ("data-inspection-tools", "load_dataset"): {"error": "csv missing"},
        }
    )
    state = make_state(csv_paths=["/missing.csv"])

    with pytest.raises(ToolError, match="load_dataset"):
        project_understanding.run(state, client)


def test_prior_art_tolerates_search_tool_error(isolated_storage):
    client = MockMCPClient(
        {
            ("prior-art-tools", "search_prior_art"): {"error": "search unavailable"},
            ("prior-art-tools", "summarize_common_approaches"): {
                "common_models": ["logistic"],
                "common_feature_engineering": [],
                "common_preprocessing": [],
                "common_split_strategies": [],
                "common_metrics": [],
                "leakage_warnings": [],
            },
            ("prior-art-tools", "extract_candidate_strategies"): {
                "ideas_to_test": ["try boosting"],
                "ideas_to_ignore": [],
            },
        }
    )
    state = make_state(project_spec={"ml_task_type": "binary_classification"})

    result = prior_art.run(state, client)

    assert result["prior_art_report"]["enabled"] is True
    assert "prior_art_report.json" in artifact_store.list_artifacts(state["project_id"])


def test_data_audit_raises_on_profile_dataset_error(isolated_storage):
    client = MockMCPClient(
        {("data-inspection-tools", "profile_dataset"): {"error": "bad csv"}}
    )
    state = make_state()

    with pytest.raises(ToolError, match="profile_dataset"):
        data_audit.run(state, client)


def test_eda_planning_runs_without_mcp_tools(isolated_storage):
    client = MockMCPClient(default={"error": "should not be called"})
    state = make_state(
        project_spec={"ml_task_type": "regression", "has_time_component": False},
        data_audit_report={
            "target": "y",
            "columns": [
                {"name": "x", "dtype": "float64", "inferred_role": "feature"},
            ],
        },
    )

    result = eda_planning.run(state, client)

    assert client.call_log == []
    assert result["eda_plan"]["checks"]


def test_executable_eda_raises_on_code_execution_failure(isolated_storage):
    client = MockMCPClient(default={})
    state = make_state(
        data_audit_report={
            "target": "y",
            "n_rows": 10,
            "n_cols": 2,
            "columns": [{"name": "x", "dtype": "float64", "inferred_role": "feature"}],
            "time_columns": [],
        }
    )

    with patch(
        "backend.agents.nodes.executable_eda.code_authoring.run_code_agent",
        return_value={"ok": False, "stderr": "NameError: x", "plots": [], "tables": []},
    ):
        with pytest.raises(PipelineStepError, match="EDA code execution failed"):
            executable_eda.run(state, client)


def test_preprocessing_plan_raises_on_split_tool_error(isolated_storage):
    def responder(server, tool, _args):
        if tool == "create_preprocessing_plan":
            return {
                "drop_columns": [],
                "keep_columns": ["feature"],
                "numeric_columns": ["feature"],
                "categorical_columns": [],
            }
        if tool == "check_preprocessing_leakage":
            return {"ok": True, "leakage_warnings": []}
        if tool == "build_train_valid_test_split":
            return {"error": "target not in dataset"}
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_spec={"targets": ["y"], "ml_task_type": "regression"},
        data_audit_report={"columns": [{"name": "feature"}]},
    )

    with pytest.raises(ToolError, match="build_train_valid_test_split"):
        preprocessing_plan.run(state, client)


def test_baseline_modeling_skips_failed_models(isolated_storage):
    def responder(server, tool, _args):
        if tool.startswith("train_"):
            return {"error": "training failed"}
        if tool == "compare_models":
            return {"rows": [], "best_model": None, "primary_metric": ""}
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_spec={"ml_task_type": "regression", "targets": ["y"]},
        data_audit_report={"n_rows": 100},
    )

    with patch(
        "backend.agents.nodes.baseline_modeling.code_authoring.run_code_agent",
        return_value={"ok": True},
    ):
        result = baseline_modeling.run(state, client)

    assert result["baseline_results"]["results"] == []
    assert result["model_comparison"] == []


def test_baseline_modeling_skips_invalid_model_payload(isolated_storage):
    def responder(server, tool, _args):
        if tool == "train_dummy_baseline":
            return {"family": "dummy"}
        if tool == "compare_models":
            return {"rows": [], "best_model": None, "primary_metric": ""}
        if tool.startswith("train_"):
            return {"error": "training failed"}
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_spec={"ml_task_type": "regression", "targets": ["y"]},
        data_audit_report={"n_rows": 100},
    )

    with patch(
        "backend.agents.nodes.baseline_modeling.code_authoring.run_code_agent",
        return_value={"ok": True},
    ):
        result = baseline_modeling.run(state, client)

    assert result["baseline_results"]["results"] == []


def test_first_conclusion_runs_without_mcp_tools(isolated_storage):
    client = MockMCPClient(default={"error": "should not be called"})
    state = make_state(
        project_spec={"ml_task_type": "regression"},
        data_audit_report={"target": "y"},
        model_comparison=[{"model": "dummy", "valid_score": 0.5}],
        split_report={"strategy": "random"},
        best_pipeline_id="dummy",
        best_validation_score=0.5,
        primary_metric="rmse",
    )

    result = first_conclusion.run(state, client)

    assert client.call_log == []
    assert result["first_conclusion"]
    assert artifact_store.read_text(state["project_id"], "first_conclusion.md")


def test_iteration_loop_rejects_failed_training(isolated_storage):
    def responder(server, tool, _args):
        if tool == "suggest_next_iteration":
            return {
                "supported": True,
                "action": "tune_boosting",
                "model_name": "boosting",
                "params": {},
                "hypothesis": "tune boosting",
                "motivation": "prior art",
            }
        if tool == "train_boosting_model":
            return {"error": "insufficient data"}
        if tool == "stop_iteration_decision":
            return {"stop": True, "reason": "done"}
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_spec={"ml_task_type": "regression"},
        prior_art_report={"ideas_to_test": []},
        data_audit_report={},
        best_validation_score=0.5,
        best_pipeline_id="dummy",
        higher_is_better=False,
        primary_metric="rmse",
        max_iterations=1,
    )

    with patch(
        "backend.agents.nodes.iteration_loop.code_authoring.run_code_agent",
        return_value={"ok": True},
    ):
        result = iteration_loop.run(state, client)

    assert len(result["iteration_reports"]) == 1
    assert result["iteration_reports"][0]["accepted"] is False
    assert "training failed" in result["iteration_reports"][0]["decision_reason"].lower()


def test_leakage_review_tolerates_tool_error_dict(isolated_storage):
    client = MockMCPClient(
        {
            ("preprocessing-tools", "check_preprocessing_leakage"): {
                "error": "plan incomplete",
            }
        }
    )
    state = make_state(
        project_spec={},
        preprocessing_plan={},
        data_audit_report={},
        split_report={"strategy": "random"},
    )

    result = leakage_review.run(state, client)

    assert result["leakage_review"]["ok"] is True
    assert artifact_store.read_json(state["project_id"], "leakage_review.json")


def test_final_test_evaluation_raises_on_eval_error(isolated_storage):
    client = MockMCPClient(
        default={"error": "model not found"},
    )
    state = make_state(
        project_spec={"ml_task_type": "regression", "primary_metric": "rmse"},
        best_pipeline_id="best",
        primary_metric="rmse",
        higher_is_better=False,
    )

    with pytest.raises(ToolError, match="evaluate_model"):
        final_test_evaluation.run(state, client)


def test_notebook_author_raises_on_export_error(isolated_storage):
    def responder(server, tool, _args):
        if tool == "update_notebook":
            return {"ok": True}
        if tool == "export_final_notebook":
            return {"error": "export failed"}
        return {}

    client = MockMCPClient(responder=responder)
    state = make_state(
        project_spec={"ml_task_type": "regression", "targets": ["y"], "primary_metric": "rmse"},
        data_audit_report={"target": "y", "n_rows": 10, "n_cols": 2},
        prior_art_report={"enabled": False},
        preprocessing_plan={},
        split_report={"strategy": "random"},
        model_comparison=[],
        csv_paths=["/tmp/data.csv"],
    )

    with pytest.raises(ToolError, match="export_final_notebook"):
        notebook_author.run(state, client)
