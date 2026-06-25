"""Tests for SQLite checkpoint store and graph resume."""
from __future__ import annotations

from backend.agents import graph as graph_module
from backend.agents.graph import run_graph
from backend.agents.state import new_state
from backend.services.checkpoint_store import CheckpointStore


def test_checkpoint_save_and_resume(isolated_storage, tmp_path):
    db = tmp_path / "cp.db"
    store = CheckpointStore(db)

    run_id = store.start_run("proj1", {"max_iterations": 3})
    state = new_state("proj1", "", ["/tmp/a.csv"], [])
    store.save_checkpoint(run_id, 1, "Step One", dict(state), status="completed")

    loaded, start = store.load_resume_state(run_id)
    assert start == 2
    assert loaded["project_id"] == "proj1"


def test_checkpoint_resume_retries_failed_step(isolated_storage, tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    run_id = store.start_run("proj1", {})
    state = new_state("proj1", "", [], [])
    store.save_checkpoint(run_id, 3, "Failed", dict(state), status="error", error={"step": "x"})

    _, start = store.load_resume_state(run_id)
    assert start == 3


def test_run_graph_persists_checkpoints(isolated_storage, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.graph.get_checkpoint_store",
        lambda: CheckpointStore(tmp_path / "cp.db"),
    )

    from backend.agents.contracts import NodeContract
    from backend.agents.graph import WorkflowStep

    def ok(state, _client):
        state["marker"] = True
        return state

    original = graph_module.WORKFLOW
    graph_module.WORKFLOW = [
        WorkflowStep(
            "Ok Step",
            ok,
            NodeContract(requires_state=("project_id",), produces_state=("marker",)),
        )
    ]
    try:
        state = new_state("cp-graph", "", [], [])
        result = run_graph(state)
        store = CheckpointStore(tmp_path / "cp.db")
        assert result.get("run_id")
        cps = store.list_checkpoints(result["run_id"])
        assert len(cps) == 1
        assert cps[0]["status"] == "completed"
    finally:
        graph_module.WORKFLOW = original
