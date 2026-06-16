"""Unit tests for LLM fallback behavior."""
from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from backend.services.llm import LLM, _extract_json


@pytest.fixture
def llm_settings(isolated_storage, monkeypatch):
    """Point the LLM module at the isolated settings object."""
    import backend.services.llm as llm_module

    monkeypatch.setattr(llm_module, "settings", isolated_storage)
    return isolated_storage


def test_complete_returns_none_when_disabled(llm_settings):
    llm = LLM()
    assert llm.enabled is False
    assert llm.complete("system", "prompt") is None


def test_complete_json_returns_fallback_when_disabled(llm_settings):
    llm = LLM()
    fallback = {"business_goal": "deterministic"}
    assert llm.complete_json("system", "prompt", fallback=fallback) == fallback


def test_complete_json_returns_fallback_on_http_error(llm_settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from backend.config import get_settings

    get_settings.cache_clear()
    import backend.services.llm as llm_module

    monkeypatch.setattr(llm_module, "settings", get_settings())

    llm = LLM()
    assert llm.enabled is True

    fallback = {"status": "fallback"}
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("network down"),
    ):
        assert llm.complete_json("system", "prompt", fallback=fallback) == fallback

    get_settings.cache_clear()


def test_extract_json_parses_embedded_object():
    text = 'Here is JSON:\n```json\n{"a": 1, "b": "two"}\n```'
    assert _extract_json(text) == {"a": 1, "b": "two"}


def test_extract_json_returns_none_for_invalid_payload():
    assert _extract_json("no json here") is None
