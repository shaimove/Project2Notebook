"""Project Understanding MCP Server.

Deterministic, offline heuristics that read the project brief and infer the ML
problem structure. The Project Understanding *agent* (node) may further refine
these with an LLM when one is configured, but these tools guarantee a sensible
result with no API key.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from mcp_servers.common import MCPServer, serve_fastmcp

server = MCPServer(
    name="project-understanding-tools",
    description="Parse the brief and infer the ML problem structure.",
)


def _read_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    ext = p.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(p))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""
    if ext == ".docx":
        return _read_docx(p)
    if ext == ".doc":
        return _read_doc_legacy(p)
    return p.read_text(encoding="utf-8", errors="ignore")


def _read_docx(path: Path) -> str:
    try:
        import zipfile
        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
        root = ET.fromstring(xml)
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        parts = [node.text for node in root.iter(f"{ns}t") if node.text]
        return "\n".join(parts)
    except Exception:
        return ""


def _read_doc_legacy(path: Path) -> str:
    """Best-effort .doc extraction (macOS textutil, else empty)."""
    import shutil
    import subprocess

    textutil = shutil.which("textutil")
    if textutil:
        try:
            proc = subprocess.run(
                [textutil, "-stdout", "-convert", "txt", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.stdout.strip():
                return proc.stdout
        except Exception:
            pass
    return ""


@server.tool("Read and return the text of the project document (md/txt/pdf/doc/docx).", {
    "path": {"type": "string", "required": True}})
def parse_project_document(args: Dict[str, Any]) -> Dict[str, Any]:
    text = _read_document(args["path"])
    headings = re.findall(r"^#+\s*(.+)$", text, flags=re.MULTILINE)
    return {"text": text, "n_chars": len(text), "headings": headings}


@server.tool("Infer the ML task type from the brief (and optional target info).", {
    "text": {"type": "string"}, "target_kind": {"type": "string"}, "n_classes": {"type": "integer"}})
def infer_ml_task(args: Dict[str, Any]) -> Dict[str, Any]:
    text = (args.get("text") or "").lower()
    target_kind = args.get("target_kind")
    n_classes = args.get("n_classes")

    def has(*words: str) -> bool:
        return any(w in text for w in words)

    task = "unknown"
    if has("anomaly", "outlier", "novelty", "fraud detection"):
        task = "anomaly_detection"
    elif has("forecast", "time series", "next day", "next week", "future value"):
        task = "forecasting"
    elif has("cluster", "segment customers", "unsupervised group"):
        task = "clustering"
    elif has("churn", "classif", "predict whether", "predict which", "spam",
             "default", "convert", "fraud", "label"):
        task = "classification"
    elif has("learning to rank", "ranking task", "rank order", "rank the"):
        task = "ranking"
    elif has("regress", "predict the price", "estimate the amount", "predict value", "predict the number", "sales", "revenue", "demand"):
        task = "regression"

    if task == "classification" or (task == "unknown" and target_kind == "categorical"):
        if n_classes is not None and n_classes > 2:
            task = "multiclass_classification"
        else:
            task = "binary_classification"
    if task == "unknown" and target_kind == "continuous":
        task = "regression"
    return {"ml_task_type": task}


@server.tool("Identify the business goal sentence(s) from the brief.", {"text": {"type": "string"}})
def identify_business_goals(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text") or ""
    goal = ""
    for kw in ("business goal", "goal is", "objective", "we want to", "aim to", "predict"):
        m = re.search(rf"([^.\n]*{kw}[^.\n]*\.)", text, flags=re.IGNORECASE)
        if m:
            goal = m.group(1).strip()
            break
    return {"business_goal": goal[:400]}


@server.tool("Identify likely target columns from the brief and available columns.", {
    "text": {"type": "string"}, "columns": {"type": "array"}})
def identify_targets(args: Dict[str, Any]) -> Dict[str, Any]:
    text = (args.get("text") or "").lower()
    columns: List[str] = args.get("columns") or []
    targets: List[str] = []
    # Explicit mention "target variable: X" or "predict X"
    for col in columns:
        c = col.lower()
        if re.search(rf"target[^a-z0-9]+{re.escape(c)}", text) or re.search(rf"predict[^.\n]*\b{re.escape(c)}\b", text):
            targets.append(col)
    # Common target names
    if not targets:
        for cand in ("churn", "churned", "target", "label", "y", "default", "fraud", "converted"):
            for col in columns:
                if col.lower() == cand:
                    targets.append(col)
    # Last resort: last column
    if not targets and columns:
        targets.append(columns[-1])
    # de-dup
    seen, out = set(), []
    for t in targets:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return {"targets": out}


@server.tool("Suggest primary/secondary metrics for a task type.", {"ml_task_type": {"type": "string"}})
def suggest_metrics(args: Dict[str, Any]) -> Dict[str, Any]:
    task = args.get("ml_task_type", "unknown")
    table = {
        "binary_classification": ("roc_auc", ["f1", "accuracy", "pr_auc"]),
        "multiclass_classification": ("f1_macro", ["accuracy", "balanced_accuracy"]),
        "multilabel_classification": ("f1_macro", ["accuracy"]),
        "regression": ("rmse", ["mae", "r2"]),
        "multi_output_regression": ("rmse", ["mae", "r2"]),
        "forecasting": ("rmse", ["mae"]),
        "anomaly_detection": ("roc_auc", ["pr_auc", "f1"]),
        "ranking": ("roc_auc", ["pr_auc"]),
        "clustering": ("silhouette", []),
    }
    primary, secondary = table.get(task, ("", []))
    return {"primary_metric": primary, "secondary_metrics": secondary}


@server.tool("Recommend a split strategy.", {
    "text": {"type": "string"}, "has_time_component": {"type": "boolean"},
    "has_groups": {"type": "boolean"}, "ml_task_type": {"type": "string"}})
def identify_split_strategy(args: Dict[str, Any]) -> Dict[str, Any]:
    text = (args.get("text") or "").lower()
    has_time = bool(args.get("has_time_component"))
    has_groups = bool(args.get("has_groups"))
    task = args.get("ml_task_type", "")
    if has_time or "time-based" in text or "avoid leakage from future" in text or "future activity" in text:
        return {"recommended_split": "time_based"}
    if has_groups or "same customer" in text or "per user" in text or "group" in text:
        return {"recommended_split": "grouped"}
    if task in ("binary_classification", "multiclass_classification", "anomaly_detection"):
        return {"recommended_split": "stratified"}
    return {"recommended_split": "random"}


@server.tool("Identify potential leakage risks from the brief and columns.", {
    "text": {"type": "string"}, "columns": {"type": "array"}})
def identify_leakage_risks(args: Dict[str, Any]) -> Dict[str, Any]:
    text = (args.get("text") or "").lower()
    columns: List[str] = args.get("columns") or []
    risks: List[str] = []
    leak_terms = ("future", "after", "post", "outcome", "label", "result", "_30d", "_next", "subsequent")
    for col in columns:
        c = col.lower()
        if any(t in c for t in ("future", "next", "post_", "_after", "outcome")):
            risks.append(f"Column '{col}' may encode information from after the prediction point.")
    if "future activity" in text or "avoid leakage" in text or "data leakage" in text:
        risks.append("Brief explicitly warns about leakage from future activity windows.")
    if not risks:
        risks.append("Verify that no feature is computed using information unavailable at prediction time.")
    return {"leakage_risks": risks}


if __name__ == "__main__":  # pragma: no cover
    serve_fastmcp(server)
