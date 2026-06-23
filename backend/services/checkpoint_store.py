"""SQLite checkpoint store for pipeline runs (A1/A2).

Persists ``DataScientist`` state after each workflow step so a run can resume
from the last successful checkpoint (or retry a failed step).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or (settings.storage_root / "checkpoints.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_sessions (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    config_json TEXT NOT NULL,
                    resume_from_step INTEGER
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    error_json TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, step_index)
                );
                CREATE INDEX IF NOT EXISTS idx_run_sessions_project
                    ON run_sessions(project_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run
                    ON checkpoints(run_id, step_index DESC);
                """
            )

    def start_run(self, project_id: str, config: Dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:16]
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_sessions (run_id, project_id, status, started_at, config_json)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, project_id, _now(), json.dumps(config, default=str)),
            )
        return run_id

    def save_checkpoint(
        self,
        run_id: str,
        step_index: int,
        step_title: str,
        state: Dict[str, Any],
        *,
        status: str,
        error: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
    ) -> None:
        checkpoint_id = f"{run_id}:{step_index}"
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, run_id, step_index, step_title, status,
                    state_json, error_json, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step_index) DO UPDATE SET
                    step_title=excluded.step_title,
                    status=excluded.status,
                    state_json=excluded.state_json,
                    error_json=excluded.error_json,
                    duration_ms=excluded.duration_ms,
                    created_at=excluded.created_at
                """,
                (
                    checkpoint_id,
                    run_id,
                    step_index,
                    step_title,
                    status,
                    json.dumps(state, default=str),
                    json.dumps(error) if error else None,
                    duration_ms,
                    _now(),
                ),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                UPDATE run_sessions SET status = ?, finished_at = ? WHERE run_id = ?
                """,
                (status, _now(), run_id),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM run_sessions WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self, project_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM run_sessions
                WHERE project_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_resumable_run(self, project_id: str) -> Optional[Dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM run_sessions
                WHERE project_id = ?
                  AND status IN ('running', 'completed_with_errors')
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ?
                ORDER BY step_index DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def load_resume_state(
        self, run_id: str, *, from_step: Optional[int] = None
    ) -> Tuple[Dict[str, Any], int]:
        """Return (state, start_step_index).

        If *from_step* is set, load the checkpoint before that step (or step 0).
        Otherwise resume from last checkpoint: retry failed step or continue after success.
        """
        with _LOCK, self._connect() as conn:
            if from_step is not None and from_step > 1:
                row = conn.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE run_id = ? AND step_index = ?
                    """,
                    (run_id, from_step - 1),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        """
                        SELECT * FROM checkpoints
                        WHERE run_id = ? AND step_index < ?
                        ORDER BY step_index DESC LIMIT 1
                        """,
                        (run_id, from_step),
                    ).fetchone()
                if row is None:
                    raise KeyError(f"No checkpoint before step {from_step} for run {run_id}")
                state = json.loads(row["state_json"])
                return state, from_step

            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ?
                ORDER BY step_index DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No checkpoints for run {run_id}")

            state = json.loads(row["state_json"])
            if row["status"] == "completed":
                return state, int(row["step_index"]) + 1
            return state, int(row["step_index"])

    def list_checkpoints(self, run_id: str) -> List[Dict[str, Any]]:
        with _LOCK, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT checkpoint_id, step_index, step_title, status, duration_ms, created_at
                FROM checkpoints WHERE run_id = ?
                ORDER BY step_index ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]


_store: Optional[CheckpointStore] = None


def get_checkpoint_store() -> CheckpointStore:
    global _store
    if _store is None:
        _store = CheckpointStore()
    return _store
