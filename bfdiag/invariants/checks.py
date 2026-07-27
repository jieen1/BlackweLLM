"""Concrete invariants for the Laguna/DFlash decode loop, each grounded in a
specific real code fact (see the docstring on each function -- no invented
invariants). Every function is a thin wrapper over
``bfdiag.invariants.registry.check`` so the actual comparison is visible
right where it's used, and every call is a no-op unless
``QSR_ASSERT_LEVEL`` is high enough.

Level assignment:
  1 (cheap): plain int/list comparisons already available at the call site.
  2 (also-expensive): same cost today (this runtime has no GPU access to do
     anything pricier), but flagged as "paranoid / less frequently needed"
     -- see each function's docstring for what a real level-2 GPU-side
     version of the check would additionally verify.
"""

from __future__ import annotations

from bfdiag.invariants.registry import check


def check_kv_len_matches_committed(slot: int, kv_len: int, committed_len: int) -> None:
    """``LagunaBackend.slot_kv_len[slot]`` and
    ``len(LagunaBackend.slot_committed_tokens[slot])`` are two independently
    updated pieces of per-slot state (``runtime/backends/laguna.py``) that
    every call site advances together (e.g. ``dflash_round``:
    ``backend.slot_kv_len[slot] += context_count`` immediately followed by
    appending ``context_count`` tokens to ``slot_committed_tokens[slot]``).
    A future edit that advances one without the other would silently
    desync the KV cache's logical length from the token history used for
    detokenization/logging."""
    check(
        1,
        "slot_kv_len_matches_committed_tokens",
        kv_len == committed_len,
        slot=slot,
        kv_len=kv_len,
        committed_len=committed_len,
    )


def check_accepted_bound(slot: int, committed_n: int, k: int) -> None:
    """One DFlash round can never commit more than ``K + 1`` tokens (``K``
    matched draft tokens plus exactly one recovery/bonus token) --
    ``runtime/mtp_accept.py::determine_accept_reject_from_predictions``
    bounds ``num_accepted`` to ``0..k`` by construction (the loop only runs
    ``range(k)`` iterations), so ``len(committed) == num_accepted + 1`` can
    never exceed ``k + 1``. A violation here means the accept/reject
    bookkeeping itself is broken, not just slow."""
    check(
        1,
        "accepted_n_le_k_plus_1",
        committed_n <= k + 1,
        slot=slot,
        committed_n=committed_n,
        k=k,
    )


def check_no_duplicate_ids(name: str, ids: list[int]) -> None:
    """``BlockPool.allocate`` hands out ``n`` blocks by popping the free
    queue ``n`` times (``runtime/block_pool.py``); each pop removes that
    block from the queue before the next pop, so the returned ids should
    never repeat. A duplicate here means a block was hand out twice in the
    same batch -- two live slots would alias the same physical KV block."""
    check(
        1,
        "no_duplicate_ids",
        len(ids) == len(set(ids)),
        context=name,
        ids=ids,
    )


def check_kv_len_monotonic(slot: int, prev_kv_len: int, new_kv_len: int) -> None:
    """A committed round only ever appends tokens: every ``dflash_round``
    call does ``backend.slot_kv_len[slot] += context_count`` where
    ``context_count = 1 + num_accepted >= 1`` (see
    ``_verify_only_accept_reject``), so ``kv_len`` must strictly increase
    across one round (it only resets to 0 via the separate, explicit
    ``reset_slot`` path, never inside a round)."""
    check(
        1,
        "kv_len_monotonic_nondecreasing",
        new_kv_len >= prev_kv_len,
        slot=slot,
        prev_kv_len=prev_kv_len,
        new_kv_len=new_kv_len,
    )


def check_aux_hidden_alignment(slot: int, prompt_len: int, aux_len: int, aux_offset: int) -> None:
    """``DFlashEngine.dflash_prefill_bootstrap`` computes
    ``aux_offset = prompt_len - aux_len`` (``runtime/backends/
    laguna_dflash.py``) before using ``aux_offset`` as the absolute start
    position for ``_bulk_precompute_context_kv``. ``aux_len`` is the number
    of rows vLLM's ``aux_hidden_states`` capture returned for this prefill;
    if it ever exceeded ``prompt_len`` (e.g. an off-by-one in which layers
    get captured), ``aux_offset`` would go negative and every subsequent
    draft-context KV write would land at the wrong ring position."""
    check(
        2,
        "aux_hidden_offset_alignment",
        aux_offset >= 0 and aux_offset + aux_len == prompt_len,
        slot=slot,
        prompt_len=prompt_len,
        aux_len=aux_len,
        aux_offset=aux_offset,
    )


def check_cg_replay_slot_consistency(slot: int, replay_slot: int) -> None:
    """DFlash's verify/draft CUDA Graphs recompute their physical KV address
    fresh from the ``slot`` argument on every ``replay()`` call
    (``runtime/backends/laguna_dflash_cudagraph.py::_fill_buffers``,
    ``phys = _physical_slot(slot)`` -- see commit 30675d2, "Fix CG binding
    address caching: 64K DFlash acceptance 19%->86%", which fixed a real
    stale-address bug of exactly this shape). This check verifies the
    caller-side half of that contract: the ``slot`` a round believes it is
    operating on must be the same ``slot`` passed into ``replay()``.

    NOTE: this only catches a caller-side mixup (e.g. a future edit that
    reads the wrong local variable) -- it cannot see whether
    ``_fill_buffers`` itself wrote to the physical address ``_physical_slot
    (replay_slot)`` actually resolves to, since that lives inside
    ``laguna_dflash_cudagraph.py`` (not part of this task's file list) and
    would need a real GPU run reading the CUDA Graph's own buffers back to
    host to verify end-to-end. Tracked as a GPU-verification TODO in
    notes/2026-07-27-bfdiag-flight-recorder.md."""
    check(
        2,
        "cg_replay_slot_matches_round_slot",
        replay_slot == slot,
        slot=slot,
        replay_slot=replay_slot,
    )


if __name__ == "__main__":
    import bfdiag.invariants.registry as registry
    from bfdiag.invariants.registry import InvariantViolation

    registry.ASSERT_LEVEL = 2
    try:
        check_accepted_bound(slot=0, committed_n=17, k=15)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass
    check_accepted_bound(slot=0, committed_n=16, k=15)  # should not raise
    check_no_duplicate_ids("self_test", [1, 2, 3])
    try:
        check_no_duplicate_ids("self_test", [1, 2, 2])
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass
    registry.ASSERT_LEVEL = 0
    print("checks.py self-test OK")
