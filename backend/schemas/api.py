"""API request/response schemas (Pydantic models for FastAPI)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CreateProjectRequest(BaseModel):
    name: str = Field(default="Untitled Project", min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=5000)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be empty")
        return stripped


class ProjectInfo(BaseModel):
    project_id: str
    name: str
    created_at: str
    project_document_path: Optional[str] = None
    csv_paths: List[str] = Field(default_factory=list)
    pdf_paths: List[str] = Field(default_factory=list)
    status: str = "created"


class UploadResponse(BaseModel):
    project_id: str
    stored_path: str
    kind: str
    filename: str


class RunRequest(BaseModel):
    project_id: str = Field(min_length=1)
    enable_prior_art: bool = True
    max_iterations: int = Field(default=3, ge=0, le=10)
    min_relative_improvement: float = Field(default=0.05, ge=0.0, le=1.0)
    resume: bool = False
    from_step: Optional[int] = Field(default=None, ge=1, le=20)


class RunSessionInfo(BaseModel):
    run_id: str
    project_id: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    config_json: str = ""
    resume_from_step: Optional[int] = None


class ToolCallLog(BaseModel):
    server: str
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    status: str = "success"
    duration_ms: int = 0


class TimelineItem(BaseModel):
    step: int
    title: str
    status: str = "completed"
    detail: str = ""


class PipelineError(BaseModel):
    step: str
    error: str


class RunResponse(BaseModel):
    project_id: str
    run_id: str = ""
    resumed_from_step: Optional[int] = None
    status: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    errors: List[PipelineError] = Field(default_factory=list)


class StatusResponse(BaseModel):
    project_id: str
    status: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[PipelineError] = Field(default_factory=list)


class ChatRequest(BaseModel):
    project_id: str
    message: str


class ChatResponse(BaseModel):
    project_id: str
    reply: str
