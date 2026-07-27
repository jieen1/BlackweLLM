"""Zero-invasion instrumentation for scripts you don't want to restructure.

::

    from bfdiag.record import auto_record
    auto_record()

That's the whole integration: no ``with`` block, no re-indenting the rest
of the script. ``auto_record()`` captures a fingerprint immediately, saves a
"live" RunRecord, and installs an ``atexit`` hook (plus a ``sys.excepthook``
chain) so the record is finalized -- ``status="ok"`` on a normal exit,
``status="failed"`` with a full traceback on an uncaught exception -- no
matter how the process ends. ``os._exit()`` and being killed by a signal are
the only ways to skip the atexit hook; there is no way around that from
pure Python, and it's the same limitation every ``atexit``-based tool has.

Note: this module intentionally does not import from ``bfdiag.record``
(the package ``__init__``) at module load time -- that module imports
``auto_record`` from here, so a top-level import back would be circular.
The imports below are deferred into the function body instead, which is
safe because by the time a caller can actually reach ``auto_record()`` the
``bfdiag.record`` package has already finished initializing.
"""

from __future__ import annotations

import atexit
import sys
import traceback
from typing import Any

_active_handle: Any = None  # at most one auto_record() per process


def auto_record(
    *,
    script: str | None = None,
    argv: list[str] | None = None,
    model: dict[str, Any] | None = None,
    workload: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
):
    """Start recording the current process as a run. Idempotent: calling
    this more than once in the same process returns the same handle.

    Returns a ``RunHandle`` so a script that wants to record a metric or two
    can still do so (``auto_record().metric("acceptance_rate", 0.687)``),
    but touching the handle again is entirely optional -- the record
    persists with just script/argv/fingerprint/status even if the caller
    never calls anything on it.
    """
    global _active_handle
    if _active_handle is not None:
        return _active_handle

    # Deferred to avoid the module-load-time circular import described above.
    from bfdiag.record import RunHandle, _export_run_env
    from bfdiag.record import fingerprint as _fingerprint
    from bfdiag.record.schema import RunRecord, new_run_id, utc_now_iso
    from bfdiag.record.store import default_store

    store = default_store()
    record = RunRecord(
        run_id=new_run_id(),
        started_at=utc_now_iso(),
        script=script or (sys.argv[0] if sys.argv else ""),
        argv=list(argv if argv is not None else sys.argv[1:]),
        status="ok",
        fingerprint=_fingerprint.capture(model=model, workload=workload, extra=extra),
    )
    handle = RunHandle(record, store)
    _export_run_env(store, record.run_id)
    store.save(record)

    previous_excepthook = sys.excepthook

    def _on_uncaught(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        record.status = "failed"
        record.error = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        previous_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _on_uncaught

    def _on_exit() -> None:
        if record.finished_at is None:
            record.finished_at = utc_now_iso()
        store.save(record)

    atexit.register(_on_exit)
    _active_handle = handle
    return handle


def _reset_for_tests() -> None:
    """Test-only: clear the module-level idempotency guard so a test process
    can call ``auto_record()`` more than once. Not part of the public API.
    """
    global _active_handle
    _active_handle = None
