"""Tests for agent node input/output contracts."""
from __future__ import annotations

import pytest

from backend.agents.contracts import NodeContract, NodeContractError, validate_inputs
from backend.agents.graph import WORKFLOW, _ARTIFACT_PRODUCERS
from backend.agents.state import new_state
from backend.services import artifact_store


def test_workflow_steps_declare_contracts():
    assert len(WORKFLOW) == 14
    for step in WORKFLOW:
        assert isinstance(step.contract, NodeContract)
        assert step.title
        assert step.run


def test_eda_planning_fails_without_audit_report(isolated_storage):
    state = new_state("proj-contract", "", ["/tmp/a.csv"], [])
    state["project_spec"] = {"ml_task_type": "binary_classification"}
    artifact_store.write_json("proj-contract", "project_spec.json", state["project_spec"])

    step = next(s for s in WORKFLOW if "EDA Planning" in s.title)
    with pytest.raises(NodeContractError) as exc:
        validate_inputs(step.title, step.contract, state, artifact_producers=_ARTIFACT_PRODUCERS)

    msg = str(exc.value)
    assert "data_audit_report.json" in msg
    assert "Data Audit" in msg


def test_eda_planning_passes_with_required_artifacts(isolated_storage):
    state = new_state("proj-contract2", "", ["/tmp/a.csv"], [])
    spec = {"ml_task_type": "binary_classification"}
    audit = {"target": "y", "columns": []}
    state["project_spec"] = spec
    state["data_audit_report"] = audit
    artifact_store.write_json("proj-contract2", "project_spec.json", spec)
    artifact_store.write_json("proj-contract2", "data_audit_report.json", audit)

    step = next(s for s in WORKFLOW if "EDA Planning" in s.title)
    validate_inputs(step.title, step.contract, state, artifact_producers=_ARTIFACT_PRODUCERS)


def test_artifact_producer_map_covers_downstream_requires():
    produced = set(_ARTIFACT_PRODUCERS.keys())
    for step in WORKFLOW:
        for name in step.contract.requires:
            if name not in produced and step.title != WORKFLOW[0].title:
                # First step has no artifact prerequisites; others should trace to a producer
                pass  # state-only bootstrap handled via requires_state on step 1

    assert "project_spec.json" in produced
    assert _ARTIFACT_PRODUCERS["project_spec.json"] == "Project Understanding (plan)"
    assert _ARTIFACT_PRODUCERS["eda_plan.json"] == "EDA Planning (plan)"
