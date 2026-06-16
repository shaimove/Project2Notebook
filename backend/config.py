"""Central configuration loaded from environment / .env.

Keep configuration in one place so services and agents can import a single
``settings`` object. Values come from environment variables (optionally loaded
from a local ``.env`` file).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repository root if present (no-op if missing).
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime settings for Project2Notebook."""

    def __init__(self) -> None:
        self.repo_root: Path = _REPO_ROOT
        self.storage_root: Path = _REPO_ROOT / "backend" / "storage"
        self.uploads_dir: Path = self.storage_root / "uploads"
        self.artifacts_dir: Path = self.storage_root / "artifacts"
        self.notebooks_dir: Path = self.storage_root / "notebooks"
        self.reports_dir: Path = self.storage_root / "reports"

        # LLM
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()

        # Server
        self.backend_host: str = os.getenv("BACKEND_HOST", "localhost")
        self.backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))

        # Agentic workflow
        # Web search defaults to ON. NOTE: no live web-search provider is wired in
        # this MVP, so the Prior Art Agent falls back to its curated offline
        # knowledge and clearly says so (it never fakes web results).
        self.enable_web_search: bool = _as_bool(os.getenv("ENABLE_WEB_SEARCH"), True)
        self.max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
        self.min_relative_improvement: float = float(
            os.getenv("MIN_RELATIVE_IMPROVEMENT", "0.05")
        )

        # Code runner
        self.code_timeout_seconds: int = int(os.getenv("CODE_TIMEOUT_SECONDS", "120"))

        self._ensure_dirs()

    @property
    def llm_enabled(self) -> bool:
        """True when a real LLM provider is configured.

        When False, the system falls back to deterministic heuristics so the
        whole pipeline still runs offline (clearly marked in outputs/UI).
        """
        return bool(self.openai_api_key)

    def _ensure_dirs(self) -> None:
        for d in (
            self.uploads_dir,
            self.artifacts_dir,
            self.notebooks_dir,
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
