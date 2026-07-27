"""Write a saved checkpoint back into a live engine's target slot, then
hand off to :mod:`bfdiag.checkpoint.verify`'s safety valve before returning
anything to the caller.

No ``runtime.*`` import here either -- ``backend``/``engine`` are
duck-typed, same convention as ``store.py``/``verify.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bfdiag.checkpoint import store, verify
from bfdiag.checkpoint.state import (
    draft_ring_block_range,
    full_block_range,
    slot_geometry,
    swa_ring_block_range,
)


@dataclass
class RestoreResult:
    """What a caller gets back from :func:`restore_checkpoint`. ``anchor``/
    ``draft_tokens`` are ready to feed straight into the next
    ``engine.dflash_round(slot, anchor, draft_tokens)`` call -- the same
    shape ``dflash_prefill_bootstrap`` returns, so restoring a checkpoint
    is a drop-in replacement for "prefill this prompt" in a decode loop.
    """

    name: str
    slot: int
    verified: bool
    anchor: int | None
    draft_tokens: list[int]
    verified_tokens: list[int]
    manifest: store.CheckpointManifest
    soft_fingerprint_diff: list[str] = field(default_factory=list)


def restore_checkpoint(
    engine: Any,
    slot: int,
    name: str,
    *,
    root: Path | str | None = None,
    verify_after: bool = True,
    derive_next_round: bool = True,
    require_clean_fingerprint: bool = False,
    model_revision: str | None = None,
    repo_paths: dict[str, str] | None = None,
) -> RestoreResult:
    """Restore checkpoint ``name`` into ``engine``'s slot ``slot``.

    Order of operations (each step cites the ``state.py`` item it
    implements):

    1. Load the manifest; check the fingerprint's HARD_FINGERPRINT_KEYS
       (block_size, blocks_per_slot, ring_blocks_per_slot, swa_window,
       draft_blocks_per_slot, kv_dtype, num_slots, model_revision) against
       the live engine's own geometry. ANY mismatch raises
       :class:`store.FingerprintMismatchError` naming every mismatched
       field -- this is checked BEFORE touching any tensor.
    2. Defensively reset ONLY the target slot (``backend.reset_slot(slot)``
       + zero this slot's draft-KV ring range) -- not the whole engine, so
       restoring into slot N of a warm multi-slot daemon does not disturb
       whatever is already live in other slots. See state.py's "CUDA Graph
       capture-time warmup residue" item for why this must happen before
       writing checkpointed tensors in.
    3. Write the full-attention / SWA-ring / draft-ring tensors back into
       their exact block ranges (recomputed from the manifest's own
       ``slot_kv_len``, not the live engine's, since the slot was just
       reset to kv_len=0).
    4. Restore ``slot_kv_len``/``slot_committed_tokens`` verbatim.
    5. If ``verify_after`` (default True): run
       :func:`bfdiag.checkpoint.verify.verify_restored_slot`, which raises
       :class:`bfdiag.checkpoint.verify.CheckpointVerificationError` on any
       divergence -- restore does NOT return a usable result in that case.
       Otherwise, if ``derive_next_round`` (default True): call
       :func:`bfdiag.checkpoint.verify.next_round_inputs` once to populate
       the returned ``anchor``/``draft_tokens``.

    IMPORTANT: both ``verify_after=True`` and (when it's False)
    ``derive_next_round=True`` call into ``engine._draft_forward``/
    ``_draft_cg.replay`` at least once, which -- matching real production
    semantics (see state.py's "next-round (anchor, draft_tokens) pair"
    item) -- WRITES new speculative mask-token bytes into the draft-KV
    ring for the upcoming round as a side effect. So after steps 1-4 finish,
    the restored tensors are byte-for-byte identical to what was saved, but
    by the time this function RETURNS (with either flag left at its
    default), the draft-KV ring has already been advanced by one round's
    worth of speculative writes -- inspecting "pure, untouched" restored
    bytes requires ``verify_after=False, derive_next_round=False``.

    Returns a :class:`RestoreResult` whose ``anchor``/``draft_tokens`` are
    ready for the caller's next ``dflash_round`` call (``anchor`` is
    ``None`` and ``draft_tokens`` is empty only when both
    ``verify_after=False`` and ``derive_next_round=False``); if
    verification ran, those are for the round AFTER the verification probe
    (whose own output is real, deterministic generation, returned in
    ``verified_tokens`` -- not thrown away).
    """
    backend = engine.backend
    manifest = store.load_manifest(name, root=root)

    geom = slot_geometry(backend, engine, slot)
    current_fp = store.capture_fingerprint(
        backend, engine, geom, model_revision=model_revision, repo_paths=repo_paths
    )

    hard_mismatches = store.check_fingerprint_compatible(manifest.fingerprint, current_fp)
    if hard_mismatches:
        raise store.FingerprintMismatchError(
            f"refusing to restore checkpoint {name!r}: incompatible with the live engine "
            f"on {len(hard_mismatches)} field(s): " + "; ".join(hard_mismatches)
        )

    soft_diffs = store.soft_fingerprint_diff(manifest.fingerprint, current_fp)
    if soft_diffs and require_clean_fingerprint:
        raise store.FingerprintMismatchError(
            f"refusing to restore checkpoint {name!r} (require_clean_fingerprint=True): "
            + "; ".join(soft_diffs)
        )

    # Step 2: defensive, target-slot-only reset (see docstring point 2).
    backend.reset_slot(slot)
    draft_start, draft_end = draft_ring_block_range(geom)
    for layer_name in geom.draft_layer_names:
        engine._draft_kv_caches[layer_name][:, draft_start:draft_end].zero_()

    # Step 3: write tensors back, recomputing ranges from the MANIFEST's
    # kv_len (the live slot is at kv_len=0 right after reset_slot above).
    tensors = store.load_tensors(name, root=root)
    full_start, full_end = full_block_range(geom, manifest.slot_kv_len)
    ring_start, ring_end = swa_ring_block_range(geom)
    for entry in manifest.tensors:
        key = entry["key"]
        category = entry["category"]
        layer_name = entry["layer_name"]
        payload = tensors[key]
        if category == "full":
            dest = backend.kv_caches[layer_name]
            dest[:, full_start:full_end] = payload.to(dtype=dest.dtype, device=dest.device)
        elif category == "swa":
            dest = backend.kv_caches[layer_name]
            dest[:, ring_start:ring_end] = payload.to(dtype=dest.dtype, device=dest.device)
        elif category == "draft":
            dest = engine._draft_kv_caches[layer_name]
            dest[:, draft_start:draft_end] = payload.to(dtype=dest.dtype, device=dest.device)
        else:
            raise ValueError(f"unknown checkpoint tensor category {category!r} for key {key!r}")

    # Step 4: bookkeeping, verbatim.
    backend.slot_kv_len[slot] = manifest.slot_kv_len
    backend.slot_committed_tokens[slot] = list(manifest.slot_committed_tokens)

    # Step 5: the safety valve.
    if verify_after:
        replay = verify.verify_restored_slot(engine, slot, manifest)
        return RestoreResult(
            name=name,
            slot=slot,
            verified=True,
            anchor=replay.final_anchor,
            draft_tokens=list(replay.final_draft_tokens),
            verified_tokens=list(replay.committed_tokens),
            manifest=manifest,
            soft_fingerprint_diff=soft_diffs,
        )

    if not derive_next_round:
        return RestoreResult(
            name=name,
            slot=slot,
            verified=False,
            anchor=None,
            draft_tokens=[],
            verified_tokens=[],
            manifest=manifest,
            soft_fingerprint_diff=soft_diffs,
        )

    anchor, draft_tokens = verify.next_round_inputs(engine, slot)
    return RestoreResult(
        name=name,
        slot=slot,
        verified=False,
        anchor=anchor,
        draft_tokens=draft_tokens,
        verified_tokens=[],
        manifest=manifest,
        soft_fingerprint_diff=soft_diffs,
    )
