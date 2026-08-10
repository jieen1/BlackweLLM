"""Trace event schema: the single source of truth for field names, phase
codes, and the ``path``/``cg_miss_reason`` vocabularies used by
``bfdiag.trace.ring`` (hot-path storage) and ``bfdiag.trace.dump``/``panel``
(dump-time rendering).

Field provenance (real code, not invented -- see
``notes/2026-07-27-bfdiag-flight-recorder.md`` for the line-numbered
citations):

- ``round_idx``: assigned by the ring buffer itself (a monotonic counter),
  not by the engine -- neither ``DFlashEngine`` nor ``LagunaBackend`` tracks
  a per-round sequence number today.
- ``slot``, ``kv_len_before``: ``runtime/backends/laguna_dflash.py``'s
  ``dflash_round`` reads ``backend.slot_kv_len[slot]`` at round entry.
- ``path`` / ``cg_miss_reason``: DFlash's verify/draft CUDA Graphs
  (``self._verify_cg`` / ``self._draft_cg`` in ``DFlashEngine``) are captured
  once at engine construction (``_init_cuda_graph``) and either used every
  round (``cg_replay``) or never (``eager``, because capture failed or
  ``QSR_VERIFY_CUDA_GRAPH=0``/``QSR_DECODE_CUDA_GRAPH`` disabled it) --
  DFlash rounds never see the *dynamic* per-call ineligibility case.  That
  case is real, though: ``LagunaBackend._decode_cg_batch_eligible`` (used by
  ``decode_batch_sampled``) rejects a batch whose size doesn't exactly match
  the captured decode graph's batch size, or that needs logprobs, or that
  has a non-greedy request -- for those calls the recorded path is
  ``cg_miss`` (a graph existed but this specific call couldn't use it), with
  a reason drawn from that same eligibility check.
- ``draft_tokens_n`` / ``accepted_n`` / ``reject_position`` / ``bonus_token``:
  ``runtime/mtp_accept.py::determine_accept_reject_from_predictions`` returns
  ``num_accepted`` (0..K, matched draft tokens only) and ``rejected_at``
  (``None`` when every draft token was accepted).  ``reject_position`` maps
  ``rejected_at`` to ``-1`` on full acceptance so the field is always a
  plain int (friendlier for histogramming than an int-or-None).
- ``t_main_forward``/``t_draft``/``t_verify``/``t_commit``/``t_round``:
  resolved from ``bfdiag.trace.timing.Timeline`` marks recorded at the real
  phase boundaries inside ``dflash_round``/``speculative_decode_step``.
  ``dflash_round`` (the production per-step path driven by
  ``ServerEngine._step_sync``) has no standalone main-forward phase --
  verify's parallel forward already yields next round's bonus token as
  ``next_anchor`` -- so ``t_main_forward`` is always 0.0 there; only the
  legacy whole-generation loop (``speculative_decode_step``, used by
  ``generate_verify_only`` and the ``benchmarks/`` profiling scripts) has a
  distinct main-forward phase.
- ``mem_allocated``: ``torch.cuda.memory_allocated()`` at round end, when
  torch+CUDA are available; 0 otherwise (CPU tests, or torch not installed).

Layered probe roadmap (T0/T1/T2) -- this module only implements T0; ``tier``/
``site_id``/``payload_ref`` below are reserved socket fields so T1/T2 don't
require a schema rewrite:

- T0 (this module, implemented): scalar per-round events, host-side ring.
  What ``RoundEvent`` carries today.
- T1 (not implemented): per-tensor reduction signatures -- absmax/L2/mean/
  NaN+Inf count, ~32 bytes/tensor. At 48 layers that's ~1.5KB/round.
- T2 (not implemented): full tensors (hidden states, logits, top-10 MoE
  routing ids) via a GPU-resident ring (device-to-device memcpy, not a
  host round-trip). Measured against this runtime's real shape (48 layers,
  hidden=3072, bf16, 256 experts/top-10): one round's full-layer hidden
  state is ~4.6MB; a D2D copy of that is ~3us, ~0.007% of a 44.16ms round
  -- cheap enough to be "always on" if/when it's built.

``tier`` (0/1/2) says which of the above produced this record. ``site_id``
identifies *which probe* recorded it (an index into an offline, versioned
probe table -- see ``SCHEMA_VERSION`` below -- so the hot path never
formats a probe name into a string; NanoLog's approach). ``payload_ref`` is
an opaque "where's the rest of this record" pointer (e.g. a length+offset
into a T1/T2 backend's own storage) for records whose payload doesn't fit
in this module's fixed scalar columns. All three default to 0/0/"" for
every T0 record emitted today.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum

# --------------------------------------------------------------------------
# Path / cg_miss_reason vocabularies
# --------------------------------------------------------------------------


class Path(IntEnum):
    """How this round's dominant CUDA-Graph-eligible work actually ran."""

    CG_REPLAY = 0  # captured graph replayed normally
    EAGER = 1  # no graph exists for this call at all (never captured / disabled)
    CG_MISS = 2  # a graph exists but THIS call was dynamically ineligible


class CgMissReason(IntEnum):
    """Why ``path`` isn't ``CG_REPLAY``. ``NONE`` when it is."""

    NONE = 0
    CG_UNAVAILABLE = 1  # capture failed or was never attempted (static, DFlash)
    CUDA_GRAPH_DISABLED = 2  # engine-wide --enable-cudagraph off / QSR_*_CUDA_GRAPH=0
    BATCH_SIZE_MISMATCH = 3  # len(slot_ids) != captured decode CG batch_size
    NON_GREEDY = 4  # a request in the batch is not greedy sampling
    LOGPROBS_REQUESTED = 5  # caller asked for logprobs (CG replay can't produce them)
    CAPTURE_FAILED = 6  # capture was attempted and failed
    NOT_CAPTURED = 7  # capture has not been attempted for this graph yet


_PATH_LABELS = {int(p): p.name.lower() for p in Path}
_REASON_LABELS = {int(r): r.name.lower() for r in CgMissReason}


class EventKind(IntEnum):
    """High-level round type. Legacy rows default to ``DECODE_ROUND``."""

    DECODE_ROUND = 0
    PREFILL_CHUNK = 1


_EVENT_KIND_LABELS = {int(kind): kind.name.lower() for kind in EventKind}


def path_label(value: int) -> str:
    return _PATH_LABELS.get(int(value), f"unknown({value})")


def reason_label(value: int) -> str:
    return _REASON_LABELS.get(int(value), f"unknown({value})")


def event_kind_label(value: int) -> str:
    return _EVENT_KIND_LABELS.get(int(value), f"unknown({value})")


# --------------------------------------------------------------------------
# Phase marks (Timeline column indices, in the order a round's phases are
# *observed*, not necessarily the order they occur in wall-clock time --
# ``dflash_round`` marks VERIFY -> COMMIT -> DRAFT while the legacy
# ``speculative_decode_step`` marks MAIN_FORWARD -> DRAFT -> VERIFY -> COMMIT.
# Duration attribution in ``ring.RoundRing.snapshot`` is based on the actual
# chronological sequence of marks recorded for a given row, not on these
# constants' numeric order, so both call orders are handled correctly.
# --------------------------------------------------------------------------

PHASE_START = 0
PHASE_MAIN_FORWARD = 1
PHASE_DRAFT = 2
PHASE_VERIFY = 3
PHASE_COMMIT = 4

PHASE_NAMES = {
    PHASE_START: "start",
    PHASE_MAIN_FORWARD: "main_forward",
    PHASE_DRAFT: "draft",
    PHASE_VERIFY: "verify",
    PHASE_COMMIT: "commit",
}

# Maximum marks recorded per round: START + the 4 named phases above.
MAX_MARKS_PER_ROUND = 5

# Fixed column layout for the numeric (non-timing) ring buffer. Order here
# is the on-disk/JSONL field order too.
NUMERIC_FIELDS: tuple[str, ...] = (
    "round_idx",
    "slot",
    "kv_len_before",
    "position",
    "row_count",
    "compressor_ratio",
    "window_entries",
    "ratio4_entries",
    "ratio128_entries",
    "path",
    "cg_miss_reason",
    "draft_tokens_n",
    "accepted_n",
    "reject_position",
    "bonus_token",
    "mem_allocated",
)

TIMING_FIELDS: tuple[str, ...] = (
    "t_main_forward",
    "t_draft",
    "t_verify",
    "t_commit",
    "t_round",
)

ALL_FIELDS: tuple[str, ...] = NUMERIC_FIELDS + TIMING_FIELDS

# Bumped whenever a field is added/removed/repurposed. Readers (``bfdiag.
# trace.dump``/``panel``) tolerate old records missing newer optional
# fields (see ``RoundEvent.from_dict``) rather than requiring a lockstep
# writer/reader upgrade -- the "offline decode dictionary" this schema
# module IS should evolve without invalidating trace files already on disk.
SCHEMA_VERSION = 2


@dataclass
class RoundEvent:
    """One fully-resolved round, as read back from ``trace.jsonl`` or a
    dump-time ring snapshot. Timing fields are milliseconds; ``path`` /
    ``cg_miss_reason`` are the human-readable string labels (not the packed
    int codes the ring buffer stores).

    ``site_id``/``tier``/``payload_ref`` are T1/T2 reserved fields (see the
    module docstring) -- every T0 record leaves them at their defaults."""

    round_idx: int
    slot: int
    kv_len_before: int
    path: str
    cg_miss_reason: str
    draft_tokens_n: int
    accepted_n: int
    reject_position: int
    bonus_token: int
    mem_allocated: int
    t_main_forward: float
    t_draft: float
    t_verify: float
    t_commit: float
    t_round: float
    event_kind: str = "decode_round"
    position: int = -1
    row_count: int = 0
    compressor_ratio: int = -1
    window_entries: int = 0
    ratio4_entries: int = 0
    ratio128_entries: int = 0
    site_id: int = 0
    tier: int = 0
    payload_ref: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> RoundEvent:
        # Only require fields the writer always emits; a record from an
        # older SCHEMA_VERSION missing a newer optional field (e.g.
        # site_id/tier/payload_ref before they existed) still parses --
        # the dataclass default fills the gap instead of a KeyError.
        return cls(**{field: data[field] for field in _EVENT_FIELD_NAMES if field in data})

    @classmethod
    def from_json(cls, line: str) -> RoundEvent:
        return cls.from_dict(json.loads(line))


_EVENT_FIELD_NAMES = tuple(RoundEvent.__dataclass_fields__.keys())


if __name__ == "__main__":
    # Self-test: round-trip a RoundEvent through JSON, and sanity-check the
    # path/reason label tables cover every enum member.
    ev = RoundEvent(
        round_idx=0,
        slot=1,
        kv_len_before=100,
        path=path_label(Path.CG_REPLAY),
        cg_miss_reason=reason_label(CgMissReason.NONE),
        draft_tokens_n=15,
        accepted_n=15,
        reject_position=-1,
        bonus_token=42,
        mem_allocated=0,
        t_main_forward=0.0,
        t_draft=1.0,
        t_verify=1.0,
        t_commit=0.5,
        t_round=2.5,
    )
    round_tripped = RoundEvent.from_json(ev.to_json())
    assert round_tripped == ev
    assert ev.site_id == 0 and ev.tier == 0 and ev.payload_ref == ""
    for p in Path:
        assert path_label(int(p)) == p.name.lower()
    for r in CgMissReason:
        assert reason_label(int(r)) == r.name.lower()
    for kind in EventKind:
        assert event_kind_label(int(kind)) == kind.name.lower()

    # Forward-compat: a record from before site_id/tier/payload_ref existed
    # (no such keys in the dict) still parses, filling in the defaults.
    _reserved_keys = (
        "event_kind",
        "position",
        "row_count",
        "compressor_ratio",
        "window_entries",
        "ratio4_entries",
        "ratio128_entries",
        "site_id",
        "tier",
        "payload_ref",
    )
    legacy_data = {k: v for k, v in asdict(ev).items() if k not in _reserved_keys}
    legacy_ev = RoundEvent.from_dict(legacy_data)
    assert legacy_ev == ev

    print("events.py self-test OK:", ev)
