"""Fixed-capacity ring buffer for per-round trace events -- the flight
recorder's hot-path core.

Design constraints (see notes/2026-07-27-bfdiag-flight-recorder.md for the
full rationale):

- Preallocated, fixed capacity (``QSR_TRACE_RING_SIZE``, default 8192).
  Wraparound silently overwrites the oldest row -- this is a ring buffer,
  not an unbounded log.
- Hot path (``begin_round``/``mark``/``finish_round``) does zero heap
  allocation, zero GPU sync, zero f-strings, zero dict construction: every
  field is a scalar write into a preallocated ``array.array`` column (int64
  via ``'q'``, int8 via ``'b'``). Dicts/strings/JSON only get built in
  ``snapshot()``, which is dump-time only.
  The one deliberate exception is ``QSR_FORCE_SYNC`` (see
  ``bfdiag/determinism.py``): when that debug-only switch is on,
  ``begin_round``/``mark`` (and therefore ``finish_round``/
  ``finish_dflash_round``, which call ``mark`` internally) each do one
  ``torch.cuda.synchronize()`` -- an intentional, opt-in, default-off
  reintroduction of exactly the GPU sync this module otherwise avoids, for
  debugging only. Off by default, it costs the same single boolean check as
  the ``TRACE_ENABLED`` guard below; on, it is loud about the tradeoff (see
  that module).
- No numpy: this repo's own dev environment doesn't install numpy (it's
  gated behind the ``cuda`` extra, see pyproject.toml), and
  ``runtime/block_pool.py`` already sets the precedent of using stdlib
  ``array`` for exactly this reason -- this module follows suit instead of
  adding a hard numpy dependency this package doesn't need.
- ``QSR_TRACE=0`` (the default) must cost effectively nothing: every
  integration call site guards its *own* work with
  ``if ring.TRACE_ENABLED:`` (a single cheap global-bool check) BEFORE
  calling into this module, so when tracing is off, nothing here executes
  at all -- see ``tests/test_bfdiag_ring.py``'s microbenchmark.
- Never blocks: no locks, no I/O, no waiting on the hot path -- every write
  is an unconditional array-index store. Wraparound is the *only* way data
  is ever lost, and it's never silent: ``round_idx`` is a monotonic counter
  that keeps incrementing across wraparound, so a reader can always compute
  how many earlier rounds were overwritten (``dropped = the earliest
  surviving row's round_idx`` -- see ``bfdiag.trace.panel.compute_stats``'s
  ``dropped`` field, reported in both ``bf trace show`` and its ``--json``
  output). There is no separate "arm/trigger" step: the ring is always
  recording, so whatever's in it at inspection time already **is** the
  pre-trigger window an anomaly needs (the logic-analyzer/flight-recorder
  framing) -- freezing on detection, not opening a fresh capture after the
  fact, which would be too late for a cause that happened rounds earlier.
- Storage is a plain ``name -> array-like`` mapping (``self._cols``) rather
  than a hardcoded set of positional columns, specifically so a future
  device-resident tier (T2 in ``bfdiag.trace.events``'s roadmap -- full
  tensors via a GPU-side ring, not host round-trips) can add columns backed
  by a different array-like (e.g. a pinned/GPU buffer) without changing the
  ``begin_round``/``mark``/``finish_round`` call surface any integration
  hook uses today.
"""

from __future__ import annotations

import atexit
import os
import time
from array import array
from pathlib import Path
from typing import Any

import bfdiag.determinism as determinism
from bfdiag.trace import events
from bfdiag.trace.timing import Timeline

# --------------------------------------------------------------------------
# Environment / storage layout (env-var coupling only -- see module docs in
# bfdiag/__init__.py's sibling agents: we never import another agent's
# not-yet-merged package, so run-id/dir plumbing is env-var based).
# --------------------------------------------------------------------------


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


def _repo_root() -> Path:
    # bfdiag/trace/ring.py -> bfdiag/trace -> bfdiag -> <repo root>
    return Path(__file__).resolve().parents[2]


TRACE_ENABLED: bool = _env_flag("QSR_TRACE", "0")
RING_SIZE: int = int(os.environ.get("QSR_TRACE_RING_SIZE", "8192"))
RUN_ID: str = os.environ.get("QSR_BFDIAG_RUN_ID") or f"local-{os.getpid()}-{int(time.time())}"
BFDIAG_DIR: Path = Path(os.environ.get("QSR_BFDIAG_DIR", str(_repo_root() / ".bfdiag")))
RUN_DIR: Path = BFDIAG_DIR / "runs" / RUN_ID
TRACE_PATH: Path = RUN_DIR / "trace.jsonl"

_INT64_FIELDS = (
    "round_idx",
    "slot",
    "kv_len_before",
    "position",
    "row_count",
    "compressor_ratio",
    "window_entries",
    "ratio4_entries",
    "ratio128_entries",
    "draft_tokens_n",
    "accepted_n",
    "reject_position",
    "bonus_token",
    "mem_allocated",
)
_INT8_FIELDS = ("event_kind", "path", "cg_miss_reason", "valid")


class RoundRing:
    """One preallocated ring buffer of ``capacity`` rounds. See module
    docstring for the hot-path contract."""

    def __init__(self, capacity: int, *, use_cuda: bool | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._cols: dict[str, array] = {
            name: array("q", [0]) * capacity for name in _INT64_FIELDS
        }
        for name in _INT8_FIELDS:
            self._cols[name] = array("b", [0]) * capacity
        self._phase_codes = array("b", [0]) * (capacity * events.MAX_MARKS_PER_ROUND)
        self._mark_count = array("b", [0]) * capacity
        self._timer = Timeline(capacity, events.MAX_MARKS_PER_ROUND, use_cuda=use_cuda)
        self._cursor = 0
        self._written = 0  # total rounds ever begun; doubles as round_idx source

    # -- hot path -----------------------------------------------------------

    def begin_round(self, slot: int, kv_len_before: int) -> int:
        # QSR_FORCE_SYNC (see bfdiag/determinism.py): sync *before* stamping
        # this round's start time, so the previous round's async GPU work
        # (and any pending error from it) is fully resolved before this
        # round's clock starts -- default-off, single boolean check.
        if determinism.FORCE_SYNC:
            determinism.maybe_sync()
        row = self._cursor
        self._cursor += 1
        if self._cursor == self.capacity:
            self._cursor = 0
        self._cols["round_idx"][row] = self._written
        self._cols["slot"][row] = slot
        self._cols["kv_len_before"][row] = kv_len_before
        self._cols["position"][row] = -1
        self._cols["row_count"][row] = 0
        self._cols["compressor_ratio"][row] = -1
        self._cols["window_entries"][row] = 0
        self._cols["ratio4_entries"][row] = 0
        self._cols["ratio128_entries"][row] = 0
        self._cols["event_kind"][row] = int(events.EventKind.DECODE_ROUND)
        self._cols["valid"][row] = 0
        self._timer.begin(row)
        self._phase_codes[row * events.MAX_MARKS_PER_ROUND] = events.PHASE_START
        self._mark_count[row] = 1
        self._written += 1
        return row

    def mark(self, row: int, phase: int) -> None:
        count = self._mark_count[row]
        if count >= events.MAX_MARKS_PER_ROUND:
            return
        # QSR_FORCE_SYNC: sync *before* capturing this mark's timestamp, so
        # (a) the timestamp only lands once every kernel up to this phase
        # boundary has actually finished, and (b) any pending async CUDA
        # error from those kernels raises right here instead of at some
        # later, unrelated synchronizing call. Default-off (single boolean
        # check); auto-degrades to a no-op with no CUDA (see maybe_sync()).
        if determinism.FORCE_SYNC:
            determinism.maybe_sync()
        self._timer.mark(row, count)
        self._phase_codes[row * events.MAX_MARKS_PER_ROUND + count] = phase
        self._mark_count[row] = count + 1

    def finish_round(
        self,
        row: int,
        phase: int,
        *,
        path: int,
        cg_miss_reason: int,
        draft_tokens_n: int,
        accepted_n: int,
        reject_position: int,
        bonus_token: int,
        event_kind: int = events.EventKind.DECODE_ROUND,
        position: int = -1,
        row_count: int = 0,
        compressor_ratio: int = -1,
        window_entries: int = 0,
        ratio4_entries: int = 0,
        ratio128_entries: int = 0,
        mem_allocated: int = 0,
    ) -> None:
        self.mark(row, phase)  # inherits the QSR_FORCE_SYNC check above
        self._cols["event_kind"][row] = int(event_kind)
        self._cols["position"][row] = position
        self._cols["row_count"][row] = row_count
        self._cols["compressor_ratio"][row] = compressor_ratio
        self._cols["window_entries"][row] = window_entries
        self._cols["ratio4_entries"][row] = ratio4_entries
        self._cols["ratio128_entries"][row] = ratio128_entries
        self._cols["path"][row] = int(path)
        self._cols["cg_miss_reason"][row] = int(cg_miss_reason)
        self._cols["draft_tokens_n"][row] = draft_tokens_n
        self._cols["accepted_n"][row] = accepted_n
        self._cols["reject_position"][row] = reject_position
        self._cols["bonus_token"][row] = bonus_token
        self._cols["mem_allocated"][row] = mem_allocated
        self._cols["valid"][row] = 1

    # -- dump-time only -------------------------------------------------

    def snapshot(self) -> list[events.RoundEvent]:
        """Resolve every valid row into a ``RoundEvent``, oldest first. May
        synchronize (CUDA backend) -- dump-time only, never the hot path."""
        rows: list[tuple[int, int]] = []  # (round_idx, physical row) for sorting
        for row in range(self.capacity):
            if self._cols["valid"][row]:
                rows.append((self._cols["round_idx"][row], row))
        rows.sort(key=lambda pair: pair[0])

        out: list[events.RoundEvent] = []
        for _, row in rows:
            out.append(self._resolve_row(row))
        return out

    def _resolve_row(self, row: int) -> events.RoundEvent:
        count = self._mark_count[row]
        deltas = self._timer.resolve_deltas_ms(row, count)
        base_col = row * events.MAX_MARKS_PER_ROUND
        durations = dict.fromkeys(events.TIMING_FIELDS, 0.0)
        total_ms = 0.0
        for i, delta in enumerate(deltas):
            phase = self._phase_codes[base_col + i + 1]
            name = events.PHASE_NAMES.get(phase)
            field = f"t_{name}" if name else None
            if field in durations:
                durations[field] += delta
            total_ms += delta
        durations["t_round"] = total_ms

        return events.RoundEvent(
            round_idx=self._cols["round_idx"][row],
            slot=self._cols["slot"][row],
            kv_len_before=self._cols["kv_len_before"][row],
            path=events.path_label(self._cols["path"][row]),
            cg_miss_reason=events.reason_label(self._cols["cg_miss_reason"][row]),
            draft_tokens_n=self._cols["draft_tokens_n"][row],
            accepted_n=self._cols["accepted_n"][row],
            reject_position=self._cols["reject_position"][row],
            bonus_token=self._cols["bonus_token"][row],
            mem_allocated=self._cols["mem_allocated"][row],
            t_main_forward=durations["t_main_forward"],
            t_draft=durations["t_draft"],
            t_verify=durations["t_verify"],
            t_commit=durations["t_commit"],
            t_round=durations["t_round"],
            event_kind=events.event_kind_label(self._cols["event_kind"][row]),
            position=self._cols["position"][row],
            row_count=self._cols["row_count"][row],
            compressor_ratio=self._cols["compressor_ratio"][row],
            window_entries=self._cols["window_entries"][row],
            ratio4_entries=self._cols["ratio4_entries"][row],
            ratio128_entries=self._cols["ratio128_entries"][row],
        )


# --------------------------------------------------------------------------
# Module-level singleton + free-function API used by integration hooks.
# ``_ring`` is only constructed when tracing is enabled -- constructing it
# eagerly regardless would probe CUDA (via Timeline's auto-detect) even when
# the operator asked for tracing to be off, which is exactly the "don't even
# check" posture the runtime is required to keep.
# --------------------------------------------------------------------------

_ring: RoundRing | None = RoundRing(RING_SIZE) if TRACE_ENABLED else None


def get_ring() -> RoundRing | None:
    """Returns the process-wide ring buffer, or ``None`` if tracing is off."""
    return _ring


def begin_round(slot: int, kv_len_before: int) -> int:
    """Call only when ``TRACE_ENABLED`` is true (callers guard this
    themselves -- see the ``if bfdiag_trace.TRACE_ENABLED:`` hooks in
    ``runtime/backends/laguna_dflash.py``)."""
    return _ring.begin_round(slot, kv_len_before)  # type: ignore[union-attr]


def mark(row: int, phase: int) -> None:
    _ring.mark(row, phase)  # type: ignore[union-attr]


def finish_round(row: int, phase: int, **kwargs: Any) -> None:
    _ring.finish_round(row, phase, **kwargs)  # type: ignore[union-attr]


def _split_dsv4_compressed_entries(
    compressor_ratio: int,
    compressed_entries: int,
) -> tuple[int, int]:
    if compressor_ratio == 4:
        return compressed_entries, 0
    if compressor_ratio == 128:
        return 0, compressed_entries
    return 0, 0


def finish_dflash_round(
    row: int,
    verify_cg_hit: bool,
    cuda_graph_enabled: bool,
    draft_tokens_n: int,
    decision: dict,
    bonus_token: int,
) -> None:
    """Convenience wrapper for ``dflash_round``'s single finish-of-round call
    site: translates the real ``_verify_only_accept_reject`` decision dict
    (``num_accepted``/``rejected_at``) into ring-buffer fields, so the
    integration hook in ``runtime/backends/laguna_dflash.py`` stays a single
    call instead of duplicating this translation inline.

    Also inherits the ``QSR_FORCE_SYNC`` check (see ``RoundRing.mark``): this
    calls ``finish_round``, which calls ``self.mark(row, phase)`` as its
    first action -- no separate sync call is needed here."""
    if verify_cg_hit:
        path = events.Path.CG_REPLAY
        reason = events.CgMissReason.NONE
    else:
        path = events.Path.EAGER
        reason = (
            events.CgMissReason.CUDA_GRAPH_DISABLED
            if not cuda_graph_enabled
            else events.CgMissReason.CG_UNAVAILABLE
        )
    rejected_at = decision.get("rejected_at")
    finish_round(
        row,
        events.PHASE_DRAFT,
        path=path,
        cg_miss_reason=reason,
        draft_tokens_n=draft_tokens_n,
        accepted_n=decision["num_accepted"],
        reject_position=-1 if rejected_at is None else rejected_at,
        bonus_token=bonus_token,
    )


def record_dsv4_prefill_chunk(
    slot: int,
    kv_len_before: int,
    *,
    position: int,
    row_count: int,
    compressor_ratio: int,
    window_entries: int,
    compressed_entries: int,
) -> None:
    """One-shot DSV4 prefill-chunk record.

    Prefill may emit layer-specific chunk events, so ``compressor_ratio`` is
    recorded directly and the compressed-entry count is stored in the ratio-4
    or ratio-128 column according to that ratio.
    """
    ratio4_entries, ratio128_entries = _split_dsv4_compressed_entries(
        compressor_ratio, compressed_entries
    )
    row = begin_round(slot, kv_len_before)
    finish_dsv4_prefill_chunk(
        row,
        position=position,
        row_count=row_count,
        compressor_ratio=compressor_ratio,
        window_entries=window_entries,
        ratio4_entries=ratio4_entries,
        ratio128_entries=ratio128_entries,
    )


def finish_dsv4_prefill_chunk(
    row: int,
    *,
    position: int,
    row_count: int,
    compressor_ratio: int,
    window_entries: int,
    ratio4_entries: int,
    ratio128_entries: int,
) -> None:
    """Finish a prefill row begun before the real forward executes."""
    finish_round(
        row,
        events.PHASE_VERIFY,
        event_kind=events.EventKind.PREFILL_CHUNK,
        position=position,
        row_count=row_count,
        compressor_ratio=compressor_ratio,
        window_entries=window_entries,
        ratio4_entries=ratio4_entries,
        ratio128_entries=ratio128_entries,
        path=events.Path.EAGER,
        cg_miss_reason=events.CgMissReason.NONE,
        draft_tokens_n=0,
        accepted_n=1,
        reject_position=-1,
        bonus_token=-1,
    )


def record_dsv4_decode_round(
    slot: int,
    kv_len_before: int,
    *,
    position: int,
    row_count: int,
    path: int,
    cg_miss_reason: int,
    window_entries: int,
    ratio4_entries: int,
    ratio128_entries: int,
) -> None:
    """One-shot DSV4 decode-round record.

    A single decode round stays a single row even when it touches both the
    ratio-4 and ratio-128 compressed regions; this preserves round counts and
    CUDA-Graph hit-rate accounting.
    """
    row = begin_round(slot, kv_len_before)
    finish_dsv4_decode_round(
        row,
        position=position,
        row_count=row_count,
        path=path,
        cg_miss_reason=cg_miss_reason,
        window_entries=window_entries,
        ratio4_entries=ratio4_entries,
        ratio128_entries=ratio128_entries,
    )


def finish_dsv4_decode_round(
    row: int,
    *,
    position: int,
    row_count: int,
    path: int,
    cg_miss_reason: int,
    window_entries: int,
    ratio4_entries: int,
    ratio128_entries: int,
) -> None:
    """Finish a decode row begun before graph replay/eager execution."""
    finish_round(
        row,
        events.PHASE_VERIFY,
        event_kind=events.EventKind.DECODE_ROUND,
        position=position,
        row_count=row_count,
        compressor_ratio=-1,
        window_entries=window_entries,
        ratio4_entries=ratio4_entries,
        ratio128_entries=ratio128_entries,
        path=path,
        cg_miss_reason=cg_miss_reason,
        draft_tokens_n=0,
        accepted_n=1,
        reject_position=-1,
        bonus_token=-1,
    )


def record_decode_batch_path(
    slot_ids: list[int],
    kv_lengths: list[int],
    decode_cg: object | None,
    cg_eligible: bool,
    return_logprobs: bool,
    params_list: list,
) -> None:
    """One-shot instrumentation for ``LagunaBackend.decode_batch_sampled``'s
    CG-eligibility branch (the exact "capacity>1 batch-size mismatch ->
    silent eager fallback" bug class from
    notes/2026-07-27-dflash-concurrency-handoff.md). Emits one lightweight
    row per slot in the batch -- draft/accept fields are N/A (this is not a
    DFlash round) so they're recorded as sentinels."""
    if cg_eligible:
        path = events.Path.CG_REPLAY
        reason = events.CgMissReason.NONE
    elif decode_cg is None:
        path = events.Path.EAGER
        reason = events.CgMissReason.CG_UNAVAILABLE
    else:
        path = events.Path.CG_MISS
        if return_logprobs:
            reason = events.CgMissReason.LOGPROBS_REQUESTED
        elif not all(p.is_greedy for p in params_list):
            reason = events.CgMissReason.NON_GREEDY
        else:
            reason = events.CgMissReason.BATCH_SIZE_MISMATCH
    for slot, kv_len in zip(slot_ids, kv_lengths):
        record_simple(slot, kv_len, path, reason)


def record_simple(slot: int, kv_len_before: int, path: int, cg_miss_reason: int) -> None:
    """One-shot record for call sites with no separate draft/verify/commit
    phases to mark (e.g. plain decode steps)."""
    row = begin_round(slot, kv_len_before)
    finish_round(
        row,
        events.PHASE_VERIFY,
        event_kind=events.EventKind.DECODE_ROUND,
        path=path,
        cg_miss_reason=cg_miss_reason,
        draft_tokens_n=0,
        accepted_n=1,
        reject_position=-1,
        bonus_token=-1,
    )


def reset(capacity: int | None = None, *, use_cuda: bool | None = None) -> None:
    """Test/debug helper: replace the module-wide ring with a fresh one.
    Not used by the hot path."""
    global _ring
    _ring = RoundRing(capacity or RING_SIZE, use_cuda=use_cuda)


def _atexit_flush() -> None:
    if _ring is None:
        return
    from bfdiag.trace import dump

    dump.write_trace(_ring, TRACE_PATH)


if TRACE_ENABLED:
    atexit.register(_atexit_flush)


if __name__ == "__main__":
    # Self-test: force the CPU backend explicitly (no real CUDA probe).
    ring = RoundRing(4, use_cuda=False)
    row = ring.begin_round(slot=0, kv_len_before=100)
    ring.mark(row, events.PHASE_VERIFY)
    ring.mark(row, events.PHASE_COMMIT)
    ring.finish_round(
        row,
        events.PHASE_DRAFT,
        path=events.Path.CG_REPLAY,
        cg_miss_reason=events.CgMissReason.NONE,
        draft_tokens_n=15,
        accepted_n=15,
        reject_position=-1,
        bonus_token=42,
    )
    snap = ring.snapshot()
    assert len(snap) == 1
    assert snap[0].path == "cg_replay"
    assert snap[0].accepted_n == 15
    print("ring.py self-test OK:", snap[0])
