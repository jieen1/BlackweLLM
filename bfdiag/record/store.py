"""SQLite-backed storage for RunRecords.

Layout, shared across all bfdiag agents::

    ${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/
        runs.sqlite
        runs/<run_id>/record.json
        runs/<run_id>/trace.jsonl        # written by the flight recorder
        runs/<run_id>/artifacts/
        oracle_cache/<prompt_hash>/      # written by the oracle differ

``runs.sqlite`` is opened in WAL mode with a busy timeout so that concurrent
processes (a script recording a run, ``bf ls`` reading it, another script's
subprocess recording a nested run) can all touch the database without
corrupting it. Every write to a run happens inside a single transaction that
also rewrites ``record.json`` -- callers see either the old state or the new
state, never a torn mix of the two.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from bfdiag.record.schema import RunRecord

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    script TEXT,
    status TEXT,
    error TEXT,
    record_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (name);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs (started_at);
"""


def _repo_root() -> Path:
    # bfdiag/record/store.py -> bfdiag/record -> bfdiag -> repo root
    return Path(__file__).resolve().parents[2]


def default_bfdiag_dir() -> Path:
    return _repo_root() / ".bfdiag"


def bfdiag_dir() -> Path:
    """``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}``."""
    override = os.environ.get("QSR_BFDIAG_DIR")
    return Path(override) if override else default_bfdiag_dir()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class RunStore:
    """Reads and writes RunRecords under a bfdiag storage root."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else bfdiag_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.sqlite"
        conn = _connect(self.db_path)
        try:
            conn.executescript(_SCHEMA_SQL)
        finally:
            conn.close()

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"

    def record_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "record.json"

    def save(self, record: RunRecord) -> None:
        """Persist ``record`` atomically: record.json plus the runs/metrics
        rows are written as one unit (the sqlite transaction commits only
        after record.json has already landed on disk).
        """
        run_dir = self.run_dir(record.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir(record.run_id).mkdir(parents=True, exist_ok=True)

        payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        _atomic_write_text(self.record_path(record.run_id), payload)

        conn = _connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, schema_version, started_at, finished_at,
                         script, status, error, record_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        schema_version = excluded.schema_version,
                        finished_at = excluded.finished_at,
                        status = excluded.status,
                        error = excluded.error,
                        record_json = excluded.record_json
                    """,
                    (
                        record.run_id,
                        record.schema_version,
                        record.started_at,
                        record.finished_at,
                        record.script,
                        record.status,
                        record.error,
                        payload,
                    ),
                )
                conn.execute("DELETE FROM metrics WHERE run_id = ?", (record.run_id,))
                if record.metrics:
                    conn.executemany(
                        "INSERT INTO metrics (run_id, name, value) VALUES (?, ?, ?)",
                        [(record.run_id, name, float(v)) for name, v in record.metrics.items()],
                    )
        finally:
            conn.close()

    def load(self, run_id: str) -> RunRecord:
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT record_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"no such run: {run_id!r}")
        return RunRecord.from_dict(json.loads(row[0]))

    def resolve_run_id(self, ref: str) -> str:
        """Resolve a run_id or unique prefix to a full run_id.

        Raises KeyError if nothing matches, ValueError if the prefix is
        ambiguous (more than one run_id starts with it).
        """
        conn = _connect(self.db_path)
        try:
            exact = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (ref,)).fetchone()
            if exact:
                return exact[0]
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE run_id LIKE ? ORDER BY run_id", (ref + "%",)
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            raise KeyError(f"no run matches id/prefix: {ref!r}")
        if len(rows) > 1:
            matches = ", ".join(r[0] for r in rows)
            raise ValueError(f"ambiguous run prefix {ref!r}, matches: {matches}")
        return rows[0][0]

    def list_runs(self, limit: int | None = None) -> list[RunRecord]:
        """Most recently started run first."""
        conn = _connect(self.db_path)
        try:
            if limit is None:
                rows = conn.execute(
                    "SELECT record_json FROM runs ORDER BY started_at DESC, run_id DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT record_json FROM runs "
                    "ORDER BY started_at DESC, run_id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        finally:
            conn.close()
        return [RunRecord.from_dict(json.loads(r[0])) for r in rows]

    def query_metric(self, name: str) -> list[tuple[str, float]]:
        """``(run_id, value)`` pairs for every run that recorded this metric."""
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT run_id, value FROM metrics WHERE name = ? ORDER BY run_id", (name,)
            ).fetchall()
        finally:
            conn.close()
        return [(r[0], r[1]) for r in rows]


def default_store() -> RunStore:
    """A store rooted at ``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}`` (re-evaluated
    each call, so tests that monkeypatch the env var see it take effect).
    """
    return RunStore(bfdiag_dir())
