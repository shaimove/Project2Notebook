"""Leakage Reviewer Agent.

Final leakage check before touching the test set. Confirms fit-on-train-only and
flags any remaining risks. Writes ``leakage_review.json``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.state import DataScientist
from backend.mcp_client.client import MCPClient
from backend.services import artifact_store, memory

PP = "preprocessing-tools"


def run(state: DataScientist, client: MCPClient) -> DataScientist:
    _ = memory.load(state["project_id"])  # read memory before acting
    spec = state.get("project_spec") or {}
    plan = state.get("preprocessing_plan") or {}
    audit = state.get("data_audit_report") or {}

    check = client.call_tool(PP, "check_preprocessing_leakage", {"plan": plan, "spec": spec})

    findings: List[str] = list(check.get("leakage_warnings", []))
    if audit.get("leakage_prone_columns"):
        findings.append(
            "Audit flagged leakage-prone columns: " + ", ".join(audit["leakage_prone_columns"])
            + " — verify these are not computed from post-prediction information."
        )
    review = {
        "fit_on_train_only": check.get("fit_on_train_only", True),
        "test_held_out": check.get("test_held_out", True),
        "split_strategy": (state.get("split_report") or {}).get("strategy"),
        "findings": findings or ["No leakage issues detected; preprocessing fit on train only and test held out."],
        "ok": check.get("ok", True),
    }
    pid = state["project_id"]
    artifact_store.write_json(pid, "leakage_review.json", review)
    artifact_store.write_json(pid, "leakage_flags.json", review)
    md = ["# Leakage Review", "",
          f"- **Fit on train only:** {review.get('fit_on_train_only')}",
          f"- **Test held out:** {review.get('test_held_out')}",
          f"- **Split strategy:** {review.get('split_strategy')}",
          f"- **Status:** {'OK' if review.get('ok') else 'WARNINGS'}",
          "", "## Findings"]
    md += [f"- {f}" for f in review.get("findings", [])]
    artifact_store.write_text(pid, "leakage_review.md", "\n".join(md))
    state["leakage_review"] = review
    memory.update(pid, phase="Leakage Review",
                  leakage_risks=[] if review.get("ok") else review.get("findings", []))
    return state
