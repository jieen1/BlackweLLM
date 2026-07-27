"""Minimal local stand-in for the (not-yet-landed) ``bfprobe.bus`` module.

The real bus is owned by the P1 agent (see
``notes/2026-07-27-probe-system-design-and-plan.md`` section 7, "P1 -- 探针
总线 + T0 正式接入生产路径"). Its public contract is:

    TIER_EVENT, TIER_SIGNATURE, TIER_TENSOR = 0, 1, 2
    PROBE_ENABLED: bool
    def emit_tensor(site_id: int, tensor) -> None: ...     # T2
    def emit_signature(site_id: int, tensor) -> None: ...  # T1
    def emit_scalar(site_id: int, **fields) -> None: ...   # T0

This module mirrors that contract exactly (same names, same signatures) so
``bfprobe/routing.py`` can do::

    try:
        from bfprobe.bus import PROBE_ENABLED, emit_tensor
    except ImportError:
        from bfprobe._bus_stub import PROBE_ENABLED, emit_tensor

and keep working, unmodified, once the real bus lands -- the ``except
ImportError`` branch simply stops being taken. Do not add behavior here that
the real bus doesn't also have; this file exists solely so bfprobe's own unit
tests don't need the real bus to exist yet.

Unlike the eventual production bus (GPU-resident rings, drain threads,
dropped-record counters), this stub just appends every emitted record to an
in-process list, gated by ``PROBE_ENABLED``, so tests can assert on exactly
what was emitted. ``reset()`` clears that state between tests.
"""

from __future__ import annotations

from typing import Any

TIER_EVENT = 0
TIER_SIGNATURE = 1
TIER_TENSOR = 2

# Module-level flag, checked once per call site -- matches the real bus's
# zero-overhead-when-disabled contract (design doc section 2, "写侧极笨").
PROBE_ENABLED = False

# Recorded calls, most-recent-last. Test-only introspection; the real bus has
# no equivalent public API (consumers read from the drained ring instead).
recorded_tensors: list[tuple[int, Any]] = []
recorded_signatures: list[tuple[int, Any]] = []
recorded_scalars: list[tuple[int, dict[str, Any]]] = []


def emit_tensor(site_id: int, tensor: Any) -> None:
    """T2: record a full tensor for the given site id."""
    if PROBE_ENABLED:
        recorded_tensors.append((site_id, tensor))


def emit_signature(site_id: int, tensor: Any) -> None:
    """T1: record a tensor signature (absmax/L2/mean/NaN·Inf count, ...)."""
    if PROBE_ENABLED:
        recorded_signatures.append((site_id, tensor))


def emit_scalar(site_id: int, **fields: Any) -> None:
    """T0: record host-side scalar fields for the given site id."""
    if PROBE_ENABLED:
        recorded_scalars.append((site_id, fields))


def reset() -> None:
    """Test helper: clear all recorded calls. Not part of the bus contract."""
    recorded_tensors.clear()
    recorded_signatures.clear()
    recorded_scalars.clear()
