"""Controlled Python execution service.

Security model for the MVP:
- Only executes *generated Python code* (no arbitrary shell commands).
- Runs as an isolated subprocess (``python -I``) with a wall-clock timeout.
- Working directory is fixed to the project's artifact folder.
- Captures stdout, stderr, exit code and any newly created files.

This is not a hardened sandbox; it is a controlled runner appropriate for a
local MVP. For production you would add resource limits / containerisation.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from backend.config import settings
from backend.services import artifact_store


# Patterns that indicate an attempt to run shell commands or escape the
# Python-only sandbox. Generated code is rejected if any are present.
_SHELL_PATTERNS = [
    "subprocess", "os.system", "os.popen", "os.exec", "pty.spawn",
    "commands.getoutput", "sh.", "pexpect", "!pip", "!python", "!ls",
    "%%bash", "%%sh", "system(", "Popen",
]


def validate_no_shell(code: str) -> List[str]:
    """Return a list of shell/command-execution violations found in code."""
    violations: List[str] = []
    for pat in _SHELL_PATTERNS:
        if pat in code:
            violations.append(pat)
    # Naked shell escapes at the start of a line (notebook-style).
    for line in code.splitlines():
        if line.lstrip().startswith("!"):
            violations.append(line.strip()[:40])
    return sorted(set(violations))


@dataclass
class CodeRunResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int
    code_path: str
    new_files: List[str] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    log_path: str = ""
    blocked: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "returncode": self.returncode,
            "duration_ms": self.duration_ms,
            "code_path": self.code_path,
            "new_files": self.new_files,
            "plots": self.plots,
            "tables": self.tables,
            "log_path": self.log_path,
            "blocked": self.blocked,
        }


def _write_log(project_id: str, filename: str, result: "CodeRunResult") -> str:
    log_dir = artifact_store.reports_dir(project_id)
    log_path = log_dir / f"{Path(filename).stem}.log"
    content = (
        f"# Execution log for {filename}\n"
        f"returncode={result.returncode} ok={result.ok} duration_ms={result.duration_ms}\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n"
        f"--- NEW FILES ---\n" + "\n".join(result.new_files) + "\n"
    )
    log_path.write_text(content, encoding="utf-8")
    return str(log_path)


def run_python(project_id: str, code: str, filename: str = "eda.py") -> CodeRunResult:
    """Persist and execute generated Python code inside the project folder.

    Safety: rejects code containing shell/command-execution patterns; runs an
    isolated subprocess (``python -I``) with the project artifact dir as the cwd
    and a wall-clock timeout; captures stdout/stderr; categorises new files into
    plots/tables; and saves an execution log under ``reports/``.
    """
    art_dir = artifact_store.project_artifact_dir(project_id)
    code_path = artifact_store.code_dir(project_id) / filename
    code_path.write_text(code, encoding="utf-8")

    violations = validate_no_shell(code)
    if violations:
        result = CodeRunResult(
            ok=False,
            stdout="",
            stderr="BLOCKED: shell/command-execution not allowed: " + ", ".join(violations),
            returncode=-2,
            duration_ms=0,
            code_path=str(code_path),
            blocked=True,
        )
        result.log_path = _write_log(project_id, filename, result)
        return result

    # Snapshot existing files so we can report newly created ones.
    before = {p for p in art_dir.rglob("*") if p.is_file()}

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(code_path)],
            cwd=str(art_dir),
            capture_output=True,
            text=True,
            timeout=settings.code_timeout_seconds,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        duration = int((time.time() - start) * 1000)
        result = CodeRunResult(
            ok=False,
            stdout=exc.stdout or "",
            stderr=f"TIMEOUT after {settings.code_timeout_seconds}s",
            returncode=-1,
            duration_ms=duration,
            code_path=str(code_path),
        )
        result.log_path = _write_log(project_id, filename, result)
        return result

    duration = int((time.time() - start) * 1000)
    after = {p for p in art_dir.rglob("*") if p.is_file()}
    new_files = sorted(str(p) for p in (after - before))
    plots = [f for f in new_files if f.endswith(".png")]
    tables = [f for f in new_files if f.endswith(".csv")]

    result = CodeRunResult(
        ok=(rc == 0),
        stdout=stdout,
        stderr=stderr,
        returncode=rc,
        duration_ms=duration,
        code_path=str(code_path),
        new_files=new_files,
        plots=plots,
        tables=tables,
    )
    result.log_path = _write_log(project_id, filename, result)
    return result
