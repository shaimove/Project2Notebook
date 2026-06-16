"""Integration tests for API error paths and storage edge cases."""
from __future__ import annotations

import json

import pytest

from backend.exceptions import ArtifactCorruptError, NotFoundError
from backend.services import artifact_store, project_store
from tests.conftest import csv_upload


def test_get_project_returns_404(client):
    response = client.get("/api/projects/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_run_pipeline_returns_404_for_unknown_project(client):
    response = client.post(
        "/api/run",
        json={
            "project_id": "missing-project",
            "enable_prior_art": True,
            "max_iterations": 3,
            "min_relative_improvement": 0.05,
        },
    )
    assert response.status_code == 404


def test_run_pipeline_returns_400_without_csv(client, project_id):
    response = client.post(
        "/api/run",
        json={
            "project_id": project_id,
            "enable_prior_art": True,
            "max_iterations": 3,
            "min_relative_improvement": 0.05,
        },
    )
    assert response.status_code == 400
    assert "upload at least one CSV" in response.json()["detail"]


def test_upload_csv_returns_404_for_unknown_project(client):
    files = csv_upload("a,b\n1,2\n")
    files["project_id"] = "unknown"
    response = client.post("/api/upload/csv", data={"project_id": files["project_id"]}, files={"file": files["file"]})
    assert response.status_code == 404


def test_upload_rejects_empty_csv(client, project_id):
    files = csv_upload("")
    response = client.post(
        "/api/upload/csv",
        data={"project_id": project_id},
        files={"file": files["file"]},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "BAD_REQUEST"
    assert "empty" in response.json()["message"].lower()


def test_upload_rejects_csv_without_header(client, project_id):
    files = csv_upload(" , \n1,2\n")
    response = client.post(
        "/api/upload/csv",
        data={"project_id": project_id},
        files={"file": files["file"]},
    )
    assert response.status_code == 400
    assert "header" in response.json()["message"].lower()


def test_upload_rejects_non_csv_extension(client, project_id):
    files = csv_upload("a,b\n1,2\n", filename="data.txt")
    response = client.post(
        "/api/upload/csv",
        data={"project_id": project_id},
        files={"file": files["file"]},
    )
    assert response.status_code == 400


def test_upload_accepts_valid_csv(client, project_id):
    files = csv_upload("feature,target\n1.0,0\n2.0,1\n")
    response = client.post(
        "/api/upload/csv",
        data={"project_id": project_id},
        files={"file": files["file"]},
    )
    assert response.status_code == 200
    record = project_store.get_project(project_id)
    assert len(record["csv_paths"]) == 1


def test_corrupt_projects_registry_is_reset_on_write(client, isolated_storage):
    project_store._REGISTRY_PATH.write_text("{not valid json", encoding="utf-8")

    created = project_store.create_project("Recovery Project")
    assert created["name"] == "Recovery Project"

    loaded = json.loads(project_store._REGISTRY_PATH.read_text(encoding="utf-8"))
    assert created["project_id"] in loaded


def test_update_missing_project_raises_not_found(isolated_storage):
    with pytest.raises(NotFoundError):
        project_store.update_project("missing", status="running")


def test_read_json_raises_on_corrupt_artifact(isolated_storage):
    project_id = project_store.create_project("Artifact Project")["project_id"]
    path = artifact_store.project_artifact_dir(project_id) / "broken.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError):
        artifact_store.read_json(project_id, "broken.json")


def test_request_validation_returns_structured_422(client):
    response = client.post("/api/run", json={"project_id": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert "details" in body
