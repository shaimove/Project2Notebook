#!/usr/bin/env python3
"""Single-command launcher for the Project2Notebook dashboard.

Runs the FastAPI backend which also serves the self-contained web dashboard,
so the whole app (UI + API + in-process MCP tools) starts from one terminal:

    python run.py

Then open the printed URL (default http://localhost:8000).
"""
from __future__ import annotations

import os
import webbrowser

import uvicorn

from backend.config import settings


def main() -> None:
    host = os.getenv("BACKEND_HOST", settings.backend_host)
    port = int(os.getenv("BACKEND_PORT", settings.backend_port))
    url = f"http://{host}:{port}"

    print("=" * 60)
    print("  Project2Notebook dashboard")
    print(f"  Open: {url}")
    print(f"  LLM enabled: {settings.llm_enabled}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    if os.getenv("OPEN_BROWSER", "1").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
