"""Lightweight env-gated per-round phase profiler for the MTP decode loop.

Enabled by ``QSR_PROFILE_ROUNDS=1``. Each round logs one JSON line with the
wall time of every instrumented phase, so a full run's per-phase distribution
can be aggregated without nsys (whose per-kernel tracing inflates the very
host-side costs this profiler exists to measure).

``QSR_PROFILE_ROUNDS=1`` is wall-clock-only: it never calls ``synchronize``,
so a profiled serving run is byte-identical in GPU behaviour to an
unprofiled one.  ``QSR_PROFILE_ROUNDS=2`` additionally enables the CUDA-event
GPU spans in ``qwen36_mtp.py`` (``verify_gpu_ms``/``sync_gpu_ms``/
``draft_gpu_ms``); those record events on the stream and must drain it to
read ``elapsed_time``, so mode 2 IS a benchmark perturbation and is for
diagnosis only.

Design constraints:
- zero import-time cost and zero allocations on the hot path when disabled;
- no third-party dependencies (stdlib only);
- phases are a flat list of ``(name, ms)`` pairs, never nested.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("qwen_sm120.round_profile")


class RoundProfile:
    """Accumulates one round's phases and emits a single JSON line at end."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("QSR_PROFILE_ROUNDS") in ("1", "2")
        self.cuda_events = os.environ.get("QSR_PROFILE_ROUNDS") == "2"
        if self.enabled:
            # The server's logging config only wires its own app logger, so
            # an INFO record on this logger would otherwise be dropped at the
            # root WARNING default. Give the profiler its own stderr handler
            # so ``QSR_PROFILE_ROUNDS=1`` output is always visible.
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
                logger.propagate = False
        self._round_start = 0.0
        self._phase_start = 0.0
        self._phases: list[tuple[str, float]] = []
        self._notes: dict[str, float] = {}

    def begin_round(self) -> None:
        if not self.enabled:
            return
        self._round_start = time.perf_counter()
        self._phase_start = self._round_start
        self._phases = []
        self._notes = {"t_begin": round(self._round_start, 6)}

    def note(self, name: str, value: float) -> None:
        """Attach a named scalar (e.g. CUDA-event GPU ms) to this round."""
        if not self.enabled:
            return
        self._notes[name] = round(float(value), 3)

    def phase(self, name: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._phases.append((name, (now - self._phase_start) * 1000.0))
        self._phase_start = now

    def end_round(self, *, label: str = "") -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._phases.append(("total", (now - self._round_start) * 1000.0))
        self._notes["t_end"] = round(now, 6)
        record: dict[str, object] = {"label": label, "phases": self._phases}
        if self._notes:
            record["notes"] = self._notes
        logger.info(json.dumps(record))

    def engine_step(self, round_batch_ms: float, bookkeep_ms: float) -> None:
        """Record the engine-side split around the MTP round call."""
        if not self.enabled:
            return
        logger.info(
            json.dumps(
                {
                    "label": "engine_step",
                    "round_batch_ms": round(round_batch_ms, 2),
                    "bookkeep_ms": round(bookkeep_ms, 2),
                }
            )
        )


#: Module-level singleton; the engine and the MTP engine share one object so a
#: round's phases stay in a single record even when both files instrument it.
round_profile = RoundProfile()
