"""Lightweight chat endpoint.

Answers questions about a completed run using the stored artifacts. When an LLM
is configured it is used; otherwise a deterministic summary is returned.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.api import ChatRequest, ChatResponse
from backend.services import project_store
from backend.services.llm import llm

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    record = project_store.get_project(req.project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="project not found")
    run = record.get("run_result") or {}
    artifacts = run.get("artifacts") or {}
    summary = run.get("summary", "")

    if not run:
        return ChatResponse(project_id=req.project_id, reply="No run has completed yet. Upload data and run the pipeline first.")

    if llm.enabled:
        context = (
            f"Run summary: {summary}\n"
            f"Project spec: {artifacts.get('project_spec')}\n"
            f"Final test report: {artifacts.get('final_test_report')}\n"
        )
        reply = llm.complete(
            "You are answering questions about a completed ML run. Be concise and factual.",
            f"{context}\nUser question: {req.message}",
        )
        if reply:
            return ChatResponse(project_id=req.project_id, reply=reply.strip())

    spec = artifacts.get("project_spec") or {}
    reply = (
        f"{summary}\n\n"
        f"Task: {spec.get('ml_task_type')} | metric: {spec.get('primary_metric')} | "
        f"split: {spec.get('recommended_split')}.\n"
        "(LLM disabled — this is a deterministic summary. Set OPENAI_API_KEY for interactive Q&A.)"
    )
    return ChatResponse(project_id=req.project_id, reply=reply)
