"""Project2Notebook FastAPI application entrypoint.

Run with:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import chat, notebook, projects, run_pipeline, upload
from backend.config import settings

app = FastAPI(
    title="Project2Notebook",
    description="Agentic ML engineering system: brief + data -> reproducible notebook.",
    version="0.1.0",
)

# Allow the Next.js dev server (and others) to call the API during the MVP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(upload.router)
app.include_router(run_pipeline.router)
app.include_router(notebook.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict:
    return {
        "name": "Project2Notebook",
        "status": "ok",
        "llm_enabled": settings.llm_enabled,
        "web_search_enabled": settings.enable_web_search,
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_enabled": settings.llm_enabled}


@app.get("/api/tools")
def list_tools() -> dict:
    """Expose the MCP tool catalogue for transparency in the UI."""
    from backend.mcp_client.client import MCPClient

    return {"tools": MCPClient().list_available_tools()}
