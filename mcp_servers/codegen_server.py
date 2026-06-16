"""Code-tools MCP Server.

The safe code-authoring + execution layer used by the *code agents*. Planning
and review agents do not use these tools; they describe what code is needed and
the code agents call these tools to write/run it.

Tools:
- write_python_file        : persist generated Python under code/ (no execution)
- validate_no_shell_commands : static check for shell/command-execution
- run_python_file          : execute a saved Python file via the controlled runner
- read_artifact            : read a text/JSON artifact
- list_artifacts           : manifest of all artifacts
- capture_generated_plots  : list PNGs produced under plots/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from backend.services import artifact_store, code_runner
from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(name="code-tools", description="Safe code authoring + execution.")


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    if not name.endswith(".py"):
        name += ".py"
    return name


@server.tool("Statically check code for shell/command-execution patterns.", {
    "code": {"type": "string", "required": True}})
def validate_no_shell_commands(args: Dict[str, Any]) -> Dict[str, Any]:
    violations = code_runner.validate_no_shell(args.get("code", ""))
    return {"ok": len(violations) == 0, "violations": violations}


@server.tool("Write generated Python to code/ (validates, does not execute).", {
    "project_id": {"type": "string", "required": True},
    "filename": {"type": "string", "required": True},
    "code": {"type": "string", "required": True}})
def write_python_file(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    filename = _safe_name(args["filename"])
    code = args.get("code", "")
    violations = code_runner.validate_no_shell(code)
    path = artifact_store.code_dir(project_id) / filename
    path.write_text(code, encoding="utf-8")
    return {
        "path": str(path),
        "filename": filename,
        "written": True,
        "validation": {"ok": len(violations) == 0, "violations": violations},
    }


@server.tool("Execute a saved Python file via the controlled runner.", {
    "project_id": {"type": "string", "required": True},
    "filename": {"type": "string", "required": True}})
def run_python_file(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    filename = _safe_name(args["filename"])
    path = artifact_store.code_dir(project_id) / filename
    if not path.exists():
        return {"error": f"file '{filename}' not found; write it first"}
    code = path.read_text(encoding="utf-8")
    result = code_runner.run_python(project_id, code, filename=filename)
    return result.to_dict()


@server.tool("Read a text/JSON artifact by name.", {
    "project_id": {"type": "string", "required": True},
    "name": {"type": "string", "required": True}})
def read_artifact(args: Dict[str, Any]) -> Dict[str, Any]:
    project_id = args["project_id"]
    name = Path(args["name"]).name
    if name.endswith(".json"):
        data = artifact_store.read_json(project_id, name)
        return {"name": name, "kind": "json", "content": data}
    text = artifact_store.read_text(project_id, name)
    if text is None:
        # try code/ and reports/ subfolders
        for sub in ("code", "reports", "tables"):
            p = artifact_store.project_artifact_dir(project_id) / sub / name
            if p.exists():
                text = p.read_text(encoding="utf-8")
                break
    return {"name": name, "kind": "text", "text": text}


@server.tool("List all artifacts produced for a project.", {
    "project_id": {"type": "string", "required": True}})
def list_artifacts(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"artifacts": artifact_store.list_artifacts(args["project_id"])}


@server.tool("List PNG plots generated under plots/.", {
    "project_id": {"type": "string", "required": True}})
def capture_generated_plots(args: Dict[str, Any]) -> Dict[str, Any]:
    plots = artifact_store.list_plots(args["project_id"])
    return {"plots": plots, "n": len(plots)}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
