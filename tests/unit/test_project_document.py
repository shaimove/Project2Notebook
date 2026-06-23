"""Tests for project document parsing (doc/docx)."""
from __future__ import annotations

import zipfile
from pathlib import Path

from mcp_servers.project_understanding_server import _read_docx, parse_project_document


def test_read_docx_extracts_text(tmp_path):
    docx = tmp_path / "brief.docx"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Predict churn from usage data.</w:t></w:r></w:p></w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", xml)
    text = _read_docx(docx)
    assert "Predict churn" in text


def test_parse_project_document_docx(tmp_path):
    docx = tmp_path / "task.docx"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Business goal: reduce churn.</w:t></w:r></w:p></w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", xml)
    result = parse_project_document({"path": str(docx)})
    assert "reduce churn" in result["text"]
