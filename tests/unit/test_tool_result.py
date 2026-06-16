"""Unit tests for MCP tool result helpers."""
from __future__ import annotations

import pytest

from backend.exceptions import ToolError
from backend.mcp_client.tool_result import optional_tool_result, require_tool_result


def test_require_tool_result_returns_success_payload():
    payload = {"rows": 1}
    assert require_tool_result(payload, tool="profile_dataset", server="data") == payload


def test_require_tool_result_raises_on_error_dict():
    with pytest.raises(ToolError, match="data/profile_dataset: missing file"):
        require_tool_result(
            {"error": "missing file"},
            tool="profile_dataset",
            server="data",
        )


def test_require_tool_result_raises_on_non_dict():
    with pytest.raises(ToolError, match="unexpected non-dict"):
        require_tool_result("bad", tool="t")  # type: ignore[arg-type]


def test_optional_tool_result_returns_none_on_error():
    assert optional_tool_result({"error": "fail"}, tool="train") is None


def test_optional_tool_result_returns_payload_on_success():
    payload = {"valid_score": 0.9}
    assert optional_tool_result(payload, tool="train") == payload
