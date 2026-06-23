"""Abstract LLM provider.

Design goals:
- Keep the provider swappable (only this module knows about OpenAI).
- Work *offline*: when no API key is configured, the system runs in
  ``heuristic`` mode and the agents rely on deterministic, data-derived logic.
  This is clearly surfaced via ``LLM.enabled`` so the UI never claims the LLM
  did something it didn't.

The agents in this MVP compute their core decisions deterministically from the
real dataset (via MCP tools). The LLM, when available, is used to *enrich*
narrative summaries. This makes runs reproducible and demonstrable without a
key, while still benefiting from an LLM when one is configured.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings


class LLM:
    """Thin LLM wrapper with an OpenAI-compatible HTTP backend.

    Falls back to ``None`` results when disabled or on any error, so callers
    must always provide a deterministic fallback.
    """

    def __init__(self) -> None:
        self.enabled: bool = settings.llm_enabled
        self.model: str = settings.openai_model
        self._api_key: str = settings.openai_api_key

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Return text completion or None if disabled/unavailable."""
        if not self.enabled:
            return None
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
            return None

    def complete_with_images(
        self,
        system: str,
        prompt: str,
        image_paths: List[str],
        *,
        max_tokens: int = 1200,
    ) -> Optional[str]:
        """Vision-capable completion using base64-encoded local PNG/JPEG files."""
        if not self.enabled or not image_paths:
            return None
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths[:6]:
            p = Path(path)
            if not p.exists():
                continue
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        if len(content) == 1:
            return None
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
            return None

    def complete_json(
        self, system: str, prompt: str, fallback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return parsed JSON from the LLM, or the deterministic fallback."""
        text = self.complete(
            system + "\nRespond ONLY with valid minified JSON.", prompt
        )
        if not text:
            return fallback
        parsed = _extract_json(text)
        return parsed if isinstance(parsed, dict) else fallback


def _extract_json(text: str) -> Optional[Any]:
    text = text.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# Singleton used across agents.
llm = LLM()
