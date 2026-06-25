"""Declarative input/output contracts for pipeline agent nodes.

Each node declares ``requires`` / ``produces`` artifact filenames (and optional
``requires_state`` / ``produces_state`` keys on ``DataScientist``). The graph
validates inputs before running a step so missing upstream outputs fail fast
with a clear error instead of breaking deep inside a tool call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from backend.exceptions import PipelineStepError
from backend.services import artifact_store

# Map artifact filename -> primary state key (when present).
ARTIFACT_TO_STATE: Dict[str, str] = {
    "project_spec.json": "project_spec",
    "prior_art_report.json": "prior_art_report",
    "data_quality_report.json": "data_quality_report",
    "data_audit_report.json": "data_audit_report",
    "eda_plan.json": "eda_plan",
    "eda_artifacts.json": "eda_artifacts",
    "eda_findings.json": "eda_findings",
    "preprocessing_plan.json": "preprocessing_plan",
    "split_report.json": "split_report",
    "baseline_results.json": "baseline_results",
    "model_results.json": "baseline_results",
    "leakage_review.json": "leakage_review",
    "leakage_flags.json": "leakage_review",
    "final_test_report.json": "final_test_report_obj",
}


@dataclass(frozen=True)
class NodeContract:
    """Artifacts and state keys a node consumes and produces."""

    requires: Tuple[str, ...] = ()
    requires_state: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    produces_state: Tuple[str, ...] = ()


class NodeContractError(PipelineStepError):
    """Raised when a node is invoked without its declared inputs."""

    code = "NODE_CONTRACT_VIOLATION"


def _state_has_value(state: dict, key: str) -> bool:
    if key not in state:
        return False
    value = state[key]
    if value is None:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


def _artifact_available(project_id: str, name: str, state: dict) -> bool:
    path = artifact_store.project_artifact_dir(project_id) / name
    if path.is_file():
        return True
    state_key = ARTIFACT_TO_STATE.get(name)
    if state_key:
        return _state_has_value(state, state_key)
    return False


def validate_inputs(
    step_title: str,
    contract: NodeContract,
    state: dict,
    *,
    artifact_producers: Dict[str, str] | None = None,
) -> None:
    """Fail fast if required artifacts or state keys are missing."""
    project_id = state.get("project_id") or ""
    missing_artifacts: List[str] = []
    missing_state: List[str] = []

    for name in contract.requires:
        if not project_id or not _artifact_available(project_id, name, state):
            missing_artifacts.append(name)

    for key in contract.requires_state:
        if not _state_has_value(state, key):
            missing_state.append(key)

    if not missing_artifacts and not missing_state:
        return

    parts: List[str] = [f"Step {step_title!r} cannot run — missing required inputs:"]
    if missing_artifacts:
        hints = []
        for name in missing_artifacts:
            producer = (artifact_producers or {}).get(name)
            hint = f"{name}" + (f" (produced by {producer!r})" if producer else "")
            hints.append(hint)
        parts.append("  artifacts: " + ", ".join(hints))
    if missing_state:
        parts.append("  state: " + ", ".join(missing_state))
    raise NodeContractError("\n".join(parts))


def verify_outputs(
    step_title: str,
    contract: NodeContract,
    state: dict,
) -> None:
    """Optional post-step check that declared outputs were written."""
    project_id = state.get("project_id") or ""
    missing_artifacts: List[str] = []
    missing_state: List[str] = []

    for name in contract.produces:
        if not project_id or not _artifact_available(project_id, name, state):
            missing_artifacts.append(name)

    for key in contract.produces_state:
        if not _state_has_value(state, key):
            missing_state.append(key)

    if not missing_artifacts and not missing_state:
        return

    parts: List[str] = [f"Step {step_title!r} finished but did not produce declared outputs:"]
    if missing_artifacts:
        parts.append("  artifacts: " + ", ".join(missing_artifacts))
    if missing_state:
        parts.append("  state: " + ", ".join(missing_state))
    raise NodeContractError("\n".join(parts))


def build_artifact_producers(
    steps: Sequence[Tuple[str, NodeContract]],
) -> Dict[str, str]:
    """Map artifact filename -> step title that should have created it."""
    producers: Dict[str, str] = {}
    for title, contract in steps:
        for name in contract.produces:
            producers[name] = title
    return producers
