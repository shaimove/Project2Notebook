"""Shared pytest fixtures for Project2Notebook tests."""
from __future__ import annotations

import io
from typing import Any, Callable, Dict, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from backend.agents.state import DataScientist, new_state
from backend.config import get_settings
from backend.mcp_client.client import MCPClient
from backend.services import project_store


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Route all storage to a temp directory and disable the LLM."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    settings = get_settings()
    root = tmp_path / "storage"
    settings.storage_root = root
    settings.uploads_dir = root / "uploads"
    settings.artifacts_dir = root / "artifacts"
    settings.notebooks_dir = root / "notebooks"
    settings.reports_dir = root / "reports"
    for directory in (
        settings.uploads_dir,
        settings.artifacts_dir,
        settings.notebooks_dir,
        settings.reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(project_store, "_REGISTRY_PATH", root / "projects.json")
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def client(isolated_storage):
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def project_id(isolated_storage):
    return project_store.create_project("Test Project")["project_id"]


def make_state(**overrides) -> DataScientist:
    state = new_state(
        project_id=overrides.pop("project_id", "testproj01"),
        project_document_path=overrides.pop("project_document_path", ""),
        csv_paths=overrides.pop("csv_paths", ["/tmp/data.csv"]),
        pdf_paths=overrides.pop("pdf_paths", []),
    )
    state.update(overrides)
    return state


class MockMCPClient(MCPClient):
    """In-memory MCP client for agent unit tests."""

    def __init__(
        self,
        responses: Optional[Dict[Tuple[str, str], Any]] = None,
        default: Any = None,
        responder: Optional[
            Callable[[str, str, Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__()
        self.responses = responses or {}
        self.default = default if default is not None else {}
        self.responder = responder

    def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        key = (server_name, tool_name)
        if self.responder is not None:
            output = self.responder(server_name, tool_name, arguments)
        elif key in self.responses:
            output = self.responses[key]
        elif callable(self.default):
            output = self.default(server_name, tool_name, arguments)
        else:
            output = self.default

        status = "error" if isinstance(output, dict) and "error" in output else "success"
        self.call_log.append(
            {
                "server": server_name,
                "tool": tool_name,
                "input": arguments,
                "output_summary": str(output)[:160],
                "status": status,
                "duration_ms": 0,
            }
        )
        return output


@pytest.fixture
def mock_mcp_factory():
    return MockMCPClient


def csv_upload(content: str, filename: str = "data.csv") -> dict:
    return {
        "project_id": "",
        "file": (filename, io.BytesIO(content.encode("utf-8")), "text/csv"),
    }
