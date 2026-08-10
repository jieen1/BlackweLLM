"""Structured run records with environment fingerprints.

Two ways to instrument a script, both persisting a
:class:`~bfdiag.record.schema.RunRecord` to the sqlite store under
``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}`` -- including when the script crashes.
A crashed experiment is still data; see
``notes/2026-07-27-bfdiag-run-records.md``.

Explicit, context-manager style::

    from bfdiag.record import run_record
    with run_record(script=__file__, workload={"prompt_hash": h, "k": 15}) as rec:
        rec.metric("acceptance_rate", 0.687)
        rec.artifact("profile", path)

Zero-invasion, for scripts you don't want to restructure::

    from bfdiag.record import auto_record
    auto_record()

Both set ``QSR_BFDIAG_RUN_ID`` (and ``QSR_RUN_RECORD``, ``QSR_BFDIAG_DIR``)
in the environment for the duration of the run, so other bfdiag tooling
(e.g. the flight recorder) can associate itself with the current run purely
by reading an environment variable -- no import of this package required.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from bfdiag.record import fingerprint as _fingerprint
from bfdiag.record.adopt import auto_record
from bfdiag.record.schema import RunRecord, new_run_id, utc_now_iso
from bfdiag.record.store import RunStore, default_store

__all__ = ["RunHandle", "run_record", "auto_record", "RunRecord", "RunStore", "default_store"]

_ENV_RUN_ID = "QSR_BFDIAG_RUN_ID"
_ENV_RUN_RECORD = "QSR_RUN_RECORD"
_ENV_BFDIAG_DIR = "QSR_BFDIAG_DIR"
_ENV_KEYS = (_ENV_RUN_ID, _ENV_RUN_RECORD, _ENV_BFDIAG_DIR)


class RunHandle:
    """Mutable handle for one in-flight run, returned by :func:`run_record`
    and :func:`~bfdiag.record.adopt.auto_record`.
    """

    def __init__(self, record: RunRecord, store: RunStore) -> None:
        self._record = record
        self._store = store

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def record(self) -> RunRecord:
        return self._record

    def metric(self, name: str, value: float) -> None:
        self._record.metrics[name] = float(value)

    def artifact(self, name: str, path: str | Path) -> None:
        """Register a file under this run's ``artifacts/`` directory.

        If ``path`` doesn't already live there, it's copied in (best
        effort -- a missing/unreadable source file never raises here, since
        losing an artifact shouldn't cost you the rest of the run record).
        """
        src = Path(path)
        artifacts_dir = self._store.artifacts_dir(self._record.run_id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        dest = src
        if src.parent.resolve() != artifacts_dir.resolve():
            dest = artifacts_dir / src.name
            with contextlib.suppress(OSError):
                shutil.copy2(src, dest)
        self._record.artifacts[name] = str(dest.relative_to(self._store.root))

    def save(self) -> None:
        self._store.save(self._record)


def _export_run_env(store: RunStore, run_id: str) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    os.environ[_ENV_RUN_ID] = run_id
    os.environ[_ENV_RUN_RECORD] = str(store.record_path(run_id))
    os.environ[_ENV_BFDIAG_DIR] = str(store.root)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _reset_trace_for_record() -> Any | None:
    """Start a fresh flight-recorder window when tracing is enabled.

    The warm daemon stays alive across many ``run_record`` scopes, while the
    trace ring is process-global.  Resetting here makes the trace belong to
    this record rather than to an arbitrary mixture of a prior canary and
    prior experiments.  The import remains deferred so ordinary CPU-only
    records never import the trace machinery.
    """
    from bfdiag.trace import ring

    if not ring.TRACE_ENABLED:
        return None
    ring.reset()
    return ring


def _write_record_trace(trace_ring: Any, store: RunStore, run_id: str) -> Path | None:
    """Flush a trace-enabled record before a long-lived daemon continues."""
    from bfdiag.trace import dump

    ring = trace_ring.get_ring()
    if ring is not None:
        return dump.write_trace(ring, store.run_dir(run_id) / "trace.jsonl")
    return None


@contextlib.contextmanager
def run_record(
    *,
    script: str | None = None,
    argv: list[str] | None = None,
    model: dict[str, Any] | None = None,
    workload: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    store: RunStore | None = None,
):
    """Capture a fingerprint, run the block, persist a RunRecord.

    Persists even when the block raises: status becomes ``"failed"`` with
    the full traceback in ``error``, and the exception is re-raised
    unchanged -- this context manager never swallows an error, it only
    makes sure one gets recorded before it propagates.
    """
    active_store = store or default_store()
    record = RunRecord(
        run_id=new_run_id(),
        started_at=utc_now_iso(),
        script=script or (sys.argv[0] if sys.argv else ""),
        argv=list(argv if argv is not None else sys.argv[1:]),
        status="ok",
        fingerprint=_fingerprint.capture(model=model, workload=workload, extra=extra),
    )
    handle = RunHandle(record, active_store)
    env_tokens = _export_run_env(active_store, record.run_id)
    trace_ring = _reset_trace_for_record()
    try:
        active_store.save(record)  # write a live record immediately, before the block runs
        yield handle
    except SystemExit as exc:
        if exc.code not in (None, 0):
            record.status = "failed"
            record.error = f"SystemExit({exc.code!r})"
        record.finished_at = utc_now_iso()
        active_store.save(record)
        raise
    except BaseException as exc:
        record.status = "failed"
        record.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        record.finished_at = utc_now_iso()
        active_store.save(record)
        raise
    else:
        record.finished_at = utc_now_iso()
        active_store.save(record)
    finally:
        if trace_ring is not None:
            trace_path = _write_record_trace(trace_ring, active_store, record.run_id)
            if trace_path is not None:
                record.trace_path = str(trace_path.relative_to(active_store.root))
                active_store.save(record)
        _restore_env(env_tokens)
