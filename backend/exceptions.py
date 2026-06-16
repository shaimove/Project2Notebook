"""Typed exceptions for Project2Notebook.

API routes and services raise these; FastAPI maps them to HTTP responses in
``backend.main``.
"""
from __future__ import annotations


class Project2NotebookError(Exception):
    """Base error with a stable machine-readable code."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(Project2NotebookError):
    code = "NOT_FOUND"
    status_code = 404


class BadRequestError(Project2NotebookError):
    code = "BAD_REQUEST"
    status_code = 400


class DomainValidationError(Project2NotebookError):
    code = "VALIDATION_ERROR"
    status_code = 422


class ToolError(Project2NotebookError):
    """An MCP tool returned ``{"error": ...}`` or failed unexpectedly."""

    code = "TOOL_FAILED"
    status_code = 500


class PipelineStepError(Project2NotebookError):
    """A pipeline node failed in a way that should be recorded on the step."""

    code = "PIPELINE_STEP_FAILED"
    status_code = 500


class ArtifactCorruptError(Project2NotebookError):
    code = "ARTIFACT_CORRUPT"
    status_code = 500
