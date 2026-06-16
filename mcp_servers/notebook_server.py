"""Notebook MCP Server.

Builds the notebook spec incrementally and exports a real .ipynb via the
notebook_builder service (nbformat).
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.schemas.notebook import NotebookCell, NotebookSection, NotebookSpec
from backend.services import artifact_store, notebook_builder
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(name="notebook-tools", description="Assemble and export the final notebook.")

_SPEC_FILE = "notebook_spec.json"


def _load_spec(project_id: str) -> Dict[str, Any]:
    data = artifact_store.read_json(project_id, _SPEC_FILE)
    return data or {"title": "Project2Notebook — Final Report", "sections": []}


def _save_spec(project_id: str, spec: Dict[str, Any]) -> None:
    artifact_store.write_json(project_id, _SPEC_FILE, spec)


def _find_section(spec: Dict[str, Any], title: str) -> Dict[str, Any]:
    for s in spec["sections"]:
        if s["title"] == title:
            return s
    sec = {"title": title, "cells": []}
    spec["sections"].append(sec)
    return sec


@server.tool("Create (or get) a notebook section.", {
    "project_id": {"type": "string", "required": True}, "title": {"type": "string", "required": True}})
def create_notebook_section(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    spec = _load_spec(project_id)
    _find_section(spec, args["title"])
    _save_spec(project_id, spec)
    return {"sections": [s["title"] for s in spec["sections"]]}


@server.tool("Add a markdown cell to a section.", {
    "project_id": {"type": "string", "required": True},
    "section_title": {"type": "string", "required": True}, "source": {"type": "string"}})
def add_markdown_cell(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    spec = _load_spec(project_id)
    sec = _find_section(spec, args["section_title"])
    sec["cells"].append({"cell_type": "markdown", "source": args.get("source", "")})
    _save_spec(project_id, spec)
    return {"ok": True}


@server.tool("Add a code cell to a section.", {
    "project_id": {"type": "string", "required": True},
    "section_title": {"type": "string", "required": True}, "source": {"type": "string"}})
def add_code_cell(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    spec = _load_spec(project_id)
    sec = _find_section(spec, args["section_title"])
    sec["cells"].append({"cell_type": "code", "source": args.get("source", "")})
    _save_spec(project_id, spec)
    return {"ok": True}


@server.tool("Replace the whole notebook spec.", {
    "project_id": {"type": "string", "required": True}, "spec": {"type": "object", "required": True}})
def update_notebook(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    _save_spec(project_id, args["spec"])
    return {"n_sections": len(args["spec"].get("sections", []))}


@server.tool("Export the assembled notebook to .ipynb and return its path.", {
    "project_id": {"type": "string", "required": True}})
def export_final_notebook(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    raw = _load_spec(project_id)
    spec = NotebookSpec(
        title=raw.get("title", "Project2Notebook — Final Report"),
        sections=[
            NotebookSection(
                title=s["title"],
                cells=[NotebookCell(**c) for c in s.get("cells", [])],
            )
            for s in raw.get("sections", [])
        ],
    )
    path = notebook_builder.build_notebook(project_id, spec)
    return {"notebook_path": str(path), "n_sections": len(spec.sections)}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
