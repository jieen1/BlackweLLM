"""``check(level, name, cond, **ctx)``: the one API every concrete invariant
in ``bfdiag.invariants.checks`` calls.

``QSR_ASSERT_LEVEL`` (read once at import time, like ``QSR_TRACE``):
  0 (default) -- every check is a no-op. This is the deployed-by-default
     state; nothing here should cost anything at level 0 beyond evaluating
     the already-cheap boolean condition the caller computed (this module
     never does GPU work itself).
  1 -- cheap, host-side-only checks (int/list comparisons -- see
     ``checks.py``'s level assignments).
  2 -- also runs the pricier checks (still host-side; genuinely expensive
     checks would need to read GPU buffer state back to host, which this
     runtime doesn't do without real GPU access -- see
     notes/2026-07-27-bfdiag-flight-recorder.md's GPU-verification TODO).

On violation, raises ``InvariantViolation`` with a message carrying the
invariant name, the failing context, AND the most recent trace events (when
``QSR_TRACE=1``) -- so the exception message alone is a small incident
report, not just "assertion failed".
"""

from __future__ import annotations

import os

ASSERT_LEVEL: int = int(os.environ.get("QSR_ASSERT_LEVEL", "0"))

_RECENT_EVENTS_IN_MESSAGE = 10


class InvariantViolation(RuntimeError):
    """Raised by ``check()`` when a gated invariant fails."""


def _recent_trace_context() -> list[str]:
    """Best-effort: pull the last few trace rows to embed in a violation
    message. Never raises (a broken trace read must not mask the real
    invariant failure); returns ``[]`` if tracing is off or unavailable."""
    try:
        from bfdiag.trace.ring import TRACE_ENABLED, get_ring

        if not TRACE_ENABLED:
            return []
        ring = get_ring()
        if ring is None:
            return []
        rows = ring.snapshot()[-_RECENT_EVENTS_IN_MESSAGE:]
        return [repr(r) for r in rows]
    except Exception as exc:  # pragma: no cover - defensive, not expected to trigger
        return [f"<failed to read recent trace events: {exc!r}>"]


def check(level: int, name: str, cond: bool, **ctx: object) -> None:
    """Assert ``cond`` when ``level <= QSR_ASSERT_LEVEL``. No-op otherwise.

    ``cond`` must already be evaluated by the caller (this keeps ``check``
    itself trivial and keeps every concrete invariant's actual comparison
    logic visible at its call site in ``checks.py`` rather than hidden
    behind a generic predicate callback).
    """
    if level > ASSERT_LEVEL:
        return
    if cond:
        return
    lines = [f"invariant violated: {name} (level={level})", f"context: {ctx!r}"]
    recent = _recent_trace_context()
    if recent:
        lines.append(f"last {len(recent)} trace event(s):")
        lines.extend(f"  {line}" for line in recent)
    raise InvariantViolation("\n".join(lines))


if __name__ == "__main__":
    # Self-test: level 0 (default) never raises even for a false condition;
    # forcing ASSERT_LEVEL up makes the same call raise.
    check(1, "self_test_noop_at_level_0", False, sample=1)
    ASSERT_LEVEL = 1
    try:
        check(1, "self_test_raises_at_level_1", False, sample=2)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation as e:
        assert "self_test_raises_at_level_1" in str(e)
        print("registry.py self-test OK")
