"""Zero-sync timing primitives for the flight recorder.

Hot-path contract (this is the whole point of this module): ``begin()`` and
``mark()`` only ever call ``cudaEventRecord`` (via ``torch.cuda.Event.record``)
or, on the CPU fallback, ``time.perf_counter()`` -- both are cheap,
non-blocking captures. Nothing in this module calls ``torch.cuda.synchronize``
or ``Event.elapsed_time`` outside of ``resolve_row``, which
``bfdiag.trace.dump`` only calls once, at dump time -- never from the
per-round hot path. Synchronizing per-round would serialize the GPU on the
exact pipeline this diagnostic exists to observe, silently corrupting the
very timings it records.

Falls back to ``time.perf_counter()`` automatically when torch isn't
installed or CUDA isn't available, so this module -- and everything built on
it -- imports and runs on a plain CPU box with no torch at all.
"""

from __future__ import annotations

import time
from array import array

try:
    import torch
except ImportError:  # pragma: no cover - exercised on CPU-only boxes
    torch = None  # type: ignore[assignment]


class Timeline:
    """Preallocated pool of ``marks_per_round`` timestamp slots for each of
    ``capacity`` rows, reused ring-style (row ``i`` is reused verbatim by
    round ``i + capacity``, ``i + 2*capacity``, ...).

    ``use_cuda`` defaults to auto-detection (torch installed AND CUDA
    available); pass it explicitly to force a backend, which every unit test
    in this repo does -- tests must never trigger a real CUDA probe (see
    ``tests/test_bfdiag_ring.py``), so they always pass ``use_cuda=False``.
    """

    def __init__(
        self,
        capacity: int,
        marks_per_round: int,
        *,
        use_cuda: bool | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if marks_per_round <= 0:
            raise ValueError(f"marks_per_round must be positive, got {marks_per_round}")
        self.capacity = capacity
        self.marks_per_round = marks_per_round
        self._use_cuda = (
            (torch is not None and torch.cuda.is_available()) if use_cuda is None else use_cuda
        )
        if self._use_cuda:
            self._events: list[list[object]] | None = [
                [torch.cuda.Event(enable_timing=True) for _ in range(marks_per_round)]
                for _ in range(capacity)
            ]
            self._times: array | None = None
        else:
            self._events = None
            # Flat float64 column-major-free layout: row * marks_per_round + col.
            self._times = array("d", [0.0]) * (capacity * marks_per_round)

    def _index(self, row: int, col: int) -> int:
        return row * self.marks_per_round + col

    def record(self, row: int, col: int) -> None:
        """O(1), no allocation, no sync: capture 'now' at (row, col)."""
        if self._use_cuda:
            self._events[row][col].record()  # type: ignore[index]
        else:
            self._times[self._index(row, col)] = time.perf_counter()  # type: ignore[index]

    # `begin`/`mark` are the same primitive under two names for readability
    # at call sites (``timeline.begin(row)`` vs. ``timeline.mark(row, col)``).
    def begin(self, row: int) -> None:
        self.record(row, 0)

    def mark(self, row: int, col: int) -> None:
        self.record(row, col)

    def resolve_deltas_ms(self, row: int, count: int) -> list[float]:
        """Dump-time only: return the ``count - 1`` consecutive deltas (in
        milliseconds) between marks ``0..count-1`` of ``row``. May
        synchronize (CUDA backend); never call this from the hot path."""
        if count < 2:
            return []
        if self._use_cuda:
            events = self._events[row]  # type: ignore[index]
            events[count - 1].synchronize()
            return [float(events[i].elapsed_time(events[i + 1])) for i in range(count - 1)]
        base = self._index(row, 0)
        times = self._times
        return [(times[base + i + 1] - times[base + i]) * 1000.0 for i in range(count - 1)]  # type: ignore[index]


if __name__ == "__main__":
    # Self-test: CPU fallback only (never probes real CUDA).
    tl = Timeline(4, 3, use_cuda=False)
    tl.begin(0)
    time.sleep(0.001)
    tl.mark(0, 1)
    time.sleep(0.001)
    tl.mark(0, 2)
    deltas = tl.resolve_deltas_ms(0, 3)
    assert len(deltas) == 2
    assert all(d >= 0.0 for d in deltas)
    print("timing.py self-test OK:", deltas)
