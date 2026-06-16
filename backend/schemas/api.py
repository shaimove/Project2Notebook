"""API request/response schemas (Pydantic models for FastAPI)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateProjectRequest(BaseModel):
    name: str = Field(default="Untitled Project")
    description: Optional[str] = None


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
    project_id: str
    enable_prior_art: bool = True
    max_iterations: int = 3
    min_relative_improvement: float = 0.05


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


class RunResponse(BaseModel):
    project_id: str
    status: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class StatusResponse(BaseModel):
    project_id: str
    status: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    project_id: str
    message: str


class ChatResponse(BaseModel):
    project_id: str
    reply: str
