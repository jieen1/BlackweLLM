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


def check_committed_ahead_of_kv_by_one(slot: int, kv_len: int, committed_len: int) -> None:
    """``committed_len == kv_len + 1`` -- NOT equality. This was originally
    written (wrongly) as ``kv_len == committed_len`` and would have fired on
    every single DFlash round in production; a coordinator review caught it
    before it shipped (see notes/2026-07-27-bfdiag-flight-recorder.md's
    architecture-coupling addendum for the incident).

    Where the ``+ 1`` comes from -- every prefill variant in
    ``runtime/backends/laguna.py`` ends by sampling ``first_token`` and
    doing (grep the pattern, it repeats 5x: ``prefill``, ``prefill_sampled``,
    ``prefill_with_aux``, the incremental-resume path, and the chunked
    variant):

        self.slot_kv_len[slot] = len(prompt_ids)              # e.g. line 1123
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]  # line 1124

    ``first_token`` was just sampled from the prompt's forward pass -- its
    KV entry doesn't exist yet (it becomes an *input* only on the next
    round). So the instant prefill finishes, ``committed_len == kv_len + 1``,
    and every later update preserves that gap by construction, never closes
    it: ``dflash_round`` (``runtime/backends/laguna_dflash.py``) does
    ``slot_kv_len[slot] += context_count`` where
    ``context_count = 1 + num_accepted`` (``_verify_only_accept_reject``,
    whose own comment says it plainly: "The old anchor and matching drafts
    ... now have valid target/draft context KV. The recovery/bonus remains
    pending.") immediately followed by appending exactly ``context_count``
    tokens (``len(committed) == context_count``, same value) to
    ``slot_committed_tokens[slot]`` -- same increment on both sides, so the
    ``+ 1`` offset is invariant, not just a one-time artifact of prefill.
    ``decode_batch_sampled`` does the same with a 1-token increment on both
    sides. A violation here means the two pieces of per-slot state
    (``runtime/backends/laguna.py``'s ``slot_kv_len``/
    ``slot_committed_tokens``) advanced by different amounts in the same
    update -- a real desync between the KV cache's logical length and the
    token history used for detokenization/logging."""
    check(
        1,
        "committed_len_is_kv_len_plus_one",
        committed_len == kv_len + 1,
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
    """Within-this-call uniqueness only -- NOT a global/cross-call claim.
    ``BlockPool.allocate(n)`` hands out ``n`` blocks by popping the free
    queue ``n`` times in one Python loop (``runtime/block_pool.py``): each
    ``_evict_one()`` pops the queue's front block object and removes it from
    the intrusive linked list before the next pop runs, so within THIS
    ``ids`` list (one call's return value) the same ``block_id`` cannot be
    popped twice -- that's what this checks.

    What this does NOT check: whether a block in ``ids`` is already live in
    some OTHER slot's block table from a DIFFERENT, earlier ``allocate``
    call (i.e. true cross-slot aliasing). That's a different claim, already
    guarded by ``allocate``'s own hard (always-on, not level-gated)
    ``RuntimeError`` checks in ``runtime/block_pool.py`` -- a popped block
    with ``ref_cnt != 0`` (meaning some slot still holds it) raises there
    directly, before this check ever runs. A duplicate in ``ids`` itself
    would mean the free-queue/eviction bookkeeping is broken (the same
    physical block hand out twice in one grow -- one slot's own block table
    would alias the same physical KV block against itself)."""
    check(
        1,
        "no_duplicate_ids_within_call",
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
    draft-context KV write would land at the wrong ring position.

    Only asserts ``aux_offset >= 0`` (equivalently ``aux_len <= prompt_len``)
    -- that's the one relationship a real bug could actually violate. An
    earlier version of this check also asserted
    ``aux_offset + aux_len == prompt_len``, which is a re-derivation of
    ``aux_offset``'s own defining formula computed at the call site
    (``aux_offset = prompt_len - aux_len``): for ANY integers, plugging that
    definition back in gives ``(prompt_len - aux_len) + aux_len ==
    prompt_len`` identically, so that clause could never be false and
    provided zero additional protection -- caught during a from-scratch
    re-derivation of every invariant here (see notes/2026-07-27-bfdiag-
    flight-recorder.md's architecture-coupling addendum); removed rather
    than left as dead weight that looks like it's checking something."""
    check(
        2,
        "aux_offset_nonnegative",
        aux_offset >= 0,
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

    # The real post-prefill/post-round relationship is +1, not equality.
    check_committed_ahead_of_kv_by_one(slot=0, kv_len=100, committed_len=101)
    try:
        check_committed_ahead_of_kv_by_one(slot=0, kv_len=100, committed_len=100)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    check_kv_len_monotonic(slot=0, prev_kv_len=100, new_kv_len=101)
    try:
        check_kv_len_monotonic(slot=0, prev_kv_len=100, new_kv_len=99)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    check_aux_hidden_alignment(slot=0, prompt_len=100, aux_len=40, aux_offset=60)
    try:
        check_aux_hidden_alignment(slot=0, prompt_len=100, aux_len=140, aux_offset=-40)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    check_cg_replay_slot_consistency(slot=2, replay_slot=2)
    try:
        check_cg_replay_slot_consistency(slot=2, replay_slot=3)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    registry.ASSERT_LEVEL = 0
    print("checks.py self-test OK")
