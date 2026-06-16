"""Project2Notebook FastAPI application entrypoint.

Run with:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import chat, notebook, projects, run_pipeline, upload
from backend.config import settings
from backend.exceptions import Project2NotebookError
from backend.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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


@app.exception_handler(Project2NotebookError)
async def project2notebook_error_handler(
    _request: Request, exc: Project2NotebookError
) -> JSONResponse:
    logger.warning("Request failed (%s): %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "An internal error occurred"},
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
