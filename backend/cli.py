"""Command-line entrypoint to run the full pipeline (handy for the demo / CI).

Usage:
    python -m backend.cli demo
    python -m backend.cli run --doc path/to/brief.md --csv path/to/data.csv [--pdf ref.pdf]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.agents.graph import run_graph
from backend.agents.state import new_state
from backend.config import settings
from backend.services import file_store, project_store
from backend.services.run_result import build_summary


def _run(doc: str, csvs: list[str], pdfs: list[str], name: str) -> None:
    record = project_store.create_project(name=name)
    pid = record["project_id"]
    if doc:
        p = file_store.copy_into_project(pid, Path(doc))
        project_store.add_file(pid, "project_document", str(p))
    for c in csvs:
        p = file_store.copy_into_project(pid, Path(c))
        project_store.add_file(pid, "csv", str(p))
    for d in pdfs:
        p = file_store.copy_into_project(pid, Path(d))
        project_store.add_file(pid, "pdf", str(p))

    record = project_store.get_project(pid)
    state = new_state(
        project_id=pid,
        project_document_path=record.get("project_document_path") or "",
        csv_paths=record.get("csv_paths", []),
        pdf_paths=record.get("pdf_paths", []),
        max_iterations=settings.max_iterations,
        min_relative_improvement=settings.min_relative_improvement,
    )
    state = run_graph(state)

    print("\n=== TIMELINE ===")
    for item in state["timeline"]:
        flag = "OK " if item["status"] == "completed" else "ERR"
        print(f"[{flag}] {item['step']:>2}. {item['title']} — {item['detail']}")
    print(f"\nTool calls: {len(state['tool_calls'])}")
    if state["errors"]:
        print("\n=== ERRORS ===")
        for e in state["errors"]:
            print(f"- {e['step']}: {e['error']}")
    print("\n=== SUMMARY ===")
    print(build_summary(state))
    print(f"\nNotebook: {state.get('notebook_path')}")
    print(f"Artifacts: {settings.artifacts_dir / pid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project2Notebook CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Run the bundled demo project")
    p_demo.add_argument("--name", default="Churn Demo")

    p_run = sub.add_parser("run", help="Run on your own files")
    p_run.add_argument("--doc", default="")
    p_run.add_argument("--csv", action="append", default=[])
    p_run.add_argument("--pdf", action="append", default=[])
    p_run.add_argument("--name", default="CLI Project")

    args = parser.parse_args()
    if args.cmd == "demo":
        demo_dir = settings.repo_root / "demo"
        _run(
            doc=str(demo_dir / "project_description.md"),
            csvs=[str(demo_dir / "sample_dataset.csv")],
            pdfs=[],
            name=args.name,
        )
    else:
        _run(doc=args.doc, csvs=args.csv, pdfs=args.pdf, name=args.name)


if __name__ == "__main__":
    main()
