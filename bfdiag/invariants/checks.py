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


def check_page_table_covers_seqlen(
    group_key: object,
    cache_seqlens: int,
    n_filled_pages: int,
    page_size: int,
) -> None:
    """``cdiv(cache_seqlens, page_size) <= n_filled_pages`` -- the number of
    page_table entries actually populated must be enough to cover the
    sequence length the same replay call declares via ``cache_seqlens``.
    Violating this means the attention kernel will read ``page_table``
    entries beyond ``n_filled_pages`` that were never written for this
    slot/round -- leftover page ids from a previous CUDA-Graph capture or a
    DIFFERENT slot's replay, so the kernel silently attends to someone
    else's KV. Symptom in production: output looks structurally fine but
    predictions quietly degrade -- there is no crash to grep for.

    This is a real, currently-latent bug, found by a sub-agent auditing the
    block_size=64->128 migration (not the cause of the acceptance-rate
    regression that investigation was chasing, but independently real):
    ``runtime/backends/laguna_cuda_graph.py``'s
    ``LagunaCudaGraphVerify._fill_buffers`` (the M=16 main-model verify
    CUDA Graph), SWA branch::

        n_ring = min(-(-aligned_len // bs), self._ring_blocks_per_slot)  # line 648, CLIPPED
        pt[0, :n_ring] = (ring_base + (block_starts % ring_slots) // bs).to(pt.dtype)  # line 654
        self._cache_seqlens[group_key][0] = aligned_len  # line 655, UNCLIPPED

    ``n_ring`` is capped at ``self._ring_blocks_per_slot`` (the ring's
    physical capacity) before it's used as the fill count, but
    ``cache_seqlens`` is set to the raw, uncapped ``aligned_len``. Whenever
    ``cdiv(aligned_len, bs) > ring_blocks_per_slot`` (the window's real span
    needs more pages than the ring physically has), the ``min()`` silently
    truncates the fill while the kernel is still told the longer length.
    Today's production block_size=64 and block_size=128 configs both land
    ``cdiv(aligned_len, bs)`` EXACTLY equal to ``ring_blocks_per_slot``
    (``_ring_blocks_for_window``'s ``+ 1`` fudge term is sized so the
    ``min()`` is a no-op at the currently-used block sizes) -- that's why
    this hasn't manifested yet; it is one alignment-granularity change away
    from firing for real.

    Contrast with the two branches that do NOT have this bug (both provably
    exact, by construction, with today's code -- no clip exists in either):

    - Full-attention, both ``LagunaCudaGraphDecode._fill_buffers``/
      ``_fill_buffers_b1`` (lines 161-174, 216-224) and
      ``LagunaCudaGraphVerify._fill_buffers`` (lines 660-666): ``n_blocks =
      cdiv(new_kv, ps)`` with NO cap, and ``cache_seqlens = new_kv`` -- the
      two are always equal.
    - SWA in ``LagunaCudaGraphDecode._fill_buffers``/``_fill_buffers_b1``
      (lines 175-198, 225-247): ``n_ring = cdiv(aligned_len, ps)`` with NO
      cap either (unlike the Verify class above) -- ``cache_seqlens =
      aligned_len`` always equals ``cdiv(cache_seqlens, ps)`` pages needed.

    ``n_filled_pages`` means the CURRENT valid entry count in ``page_table``
    after this call returns, NOT "how many entries this specific call
    physically rewrote". ``LagunaCudaGraphDecode`` only rewrites
    ``page_table`` when ``n_ring`` (or ``n_blocks``) differs from the
    previous call's value (``if n_ring != self._swa_prev_n_blocks[i]:``,
    lines 188/237) -- on a cache-hit call that skips the rewrite,
    ``n_filled_pages`` is still ``n_ring`` (this call's own requirement),
    because the skip is only correct precisely because the previously
    written entries already numbered ``n_ring`` (the count didn't change,
    by the ``if``'s own condition). Pass this call's computed page count
    (``n_ring``/``n_blocks``), not a rewrite-happened flag."""
    pages_needed = -(-cache_seqlens // page_size)  # cdiv
    deficit = max(0, pages_needed - n_filled_pages)
    check(
        1,
        "page_table_covers_seqlen",
        pages_needed <= n_filled_pages,
        group_key=group_key,
        cache_seqlens=cache_seqlens,
        page_size=page_size,
        pages_needed=pages_needed,
        n_filled_pages=n_filled_pages,
        deficit=deficit,
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


# ---------------------------------------------------------------------------
# A3 (docs/a3-cache-coordinator-design.md §2/§7 step 7-e): the four
# invariants that table marks as having NO automated assertion today
# (INV-A3-2/3/4/8). Wired here, not into a real call site yet -- the same
# shape as check_cg_replay_slot_consistency/check_aux_hidden_alignment
# above, which name their intended call site in their own docstrings but
# are not called from any production code either. Step 7-g (out of scope
# for this task) is where a real caller would start passing these live
# values; until then, these exist tested and ready.
# ---------------------------------------------------------------------------


def check_prefix_hit_state_le_kv(slot: int, kv_hit: int, state_hit: int) -> None:
    """INV-A3-2: ``state_hit <= kv_hit`` always holds -- the region
    ``[state_hit, kv_hit)`` has KV physically resident but no matching
    recurrent-state checkpoint, so treating it as safe to skip would start a
    recurrent layer's forward from a state that is stale for those
    positions (docs/a3-cache-coordinator-design.md §2/§3).

    ``runtime.backends.protocol.PrefixHit.__post_init__`` already enforces
    this unconditionally (always-on, raises ``ValueError``, cannot be
    constructed otherwise) -- this bfdiag check is deliberate defense in
    depth, not a substitute: it exists for a call site that wants the
    trace-ring-annotated ``InvariantViolation`` message this module's
    ``check()`` produces (recent trace events embedded, per
    ``bfdiag.invariants.registry``'s own docstring), or that is checking the
    relationship against values read back out of some OTHER representation
    (e.g. a logged/serialized hit) rather than a live ``PrefixHit`` instance
    the dataclass guard already protected. A violation here that the
    dataclass did NOT already catch would mean something is bypassing
    ``PrefixHit`` construction entirely -- worth knowing on its own."""
    check(
        1,
        "prefix_hit_state_hit_le_kv_hit",
        state_hit <= kv_hit,
        slot=slot,
        kv_hit=kv_hit,
        state_hit=state_hit,
    )


def check_lockstep_eviction(
    key: object,
    *,
    kv_hash_dropped: bool,
    checkpoint_dropped: bool,
    kv_ref_cnt: int,
) -> None:
    """INV-A3-3: bidirectional, asymmetric lockstep eviction between the KV
    and recurrent-state resources (docs/a3-cache-coordinator-design.md
    §2/§4; ``oracle/qwen36_vllm/gdn_state.py:183-209``'s comment block is
    the real-code fact this is grounded in, reimplemented in
    ``runtime.recurrent_state_pool.RecurrentStatePool.evict`` -- A3 step
    7-c).

    Two sub-conditions checked together, since they are the same named
    invariant in docs/a3-cache-coordinator-design.md §2's table:

    * Forward (KV-triggered): if the KV resource's hash was dropped
      (``kv_hash_dropped``), the co-keyed checkpoint MUST also have been
      dropped (``checkpoint_dropped``) -- a KV block is never evicted while
      its checkpoint survives, which would let a future reconcile find a
      ghost attention hit with no matching state.
    * Reverse (checkpoint-triggered), the asymmetric half: if the co-keyed
      KV resource is still referenced (``kv_ref_cnt > 0``) at the moment the
      checkpoint is evicted, its hash MUST NOT have been dropped
      (``kv_hash_dropped`` must be ``False``) -- losing only the checkpoint
      is a safe compute miss (``L = G <= A`` still holds); reclaiming a
      still-referenced KV block's memory instead would be a
      use-after-free-shaped correctness bug, not a performance one."""
    forward_ok = (not kv_hash_dropped) or checkpoint_dropped
    reverse_ok = (kv_ref_cnt <= 0) or (not kv_hash_dropped)
    check(
        1,
        "lockstep_eviction_bidirectional_asymmetric",
        forward_ok and reverse_ok,
        key=key,
        kv_hash_dropped=kv_hash_dropped,
        checkpoint_dropped=checkpoint_dropped,
        kv_ref_cnt=kv_ref_cnt,
    )


def check_referenced_resource_never_evicted(
    resource_id: object, ref_cnt: int, was_evicted: bool
) -> None:
    """INV-A3-4: a resource with a live reference (``ref_cnt > 0``, or
    equivalently an actively-occupied slot) is never evicted by either
    allocator (docs/a3-cache-coordinator-design.md §2; mirrors
    ``runtime.block_pool.BlockPool.allocate``'s own hard, always-on
    ``RuntimeError`` on true exhaustion, and
    ``runtime.recurrent_state_pool.RecurrentStatePool.evict``'s hard
    ``RuntimeError`` on a pinned key -- A3 step 7-c). This is the
    resource-agnostic, level-gated form usable at a call site that does not
    hold a reference to either concrete allocator, e.g. a coordinator
    (A3 step 7-d) auditing an eviction decision it did not itself make."""
    check(
        1,
        "referenced_resource_never_evicted",
        not (ref_cnt > 0 and was_evicted),
        resource_id=resource_id,
        ref_cnt=ref_cnt,
        was_evicted=was_evicted,
    )


def check_reserved_physical_slots_agree(
    context: str, state_pool_reserved: int, backend_reserved: int
) -> None:
    """INV-A3-8: a state allocator's ``RESERVED_PHYSICAL_SLOTS`` convention
    must match the KV-side runtime it is paired with -- NOT be assumed
    equal by default (docs/a3-cache-coordinator-design.md §1.8: this
    project already has one real, documented divergence,
    ``runtime.block_pool.RESERVED_PHYSICAL_SLOTS == 1`` vs
    ``runtime.backends.laguna.RESERVED_PHYSICAL_SLOTS == 0``, and one prior
    100%-deterministic wrong-output incident traced to physical index 0's
    addressing assumptions, per ``block_pool.py``'s own
    ``RESERVED_PHYSICAL_SLOTS`` comment). A mismatch here means a state
    pool and the backend it is meant to serve disagree about which physical
    address is off-limits -- exactly the class of bug that has already
    happened once in this project, this time caught by an assertion instead
    of by a debugging session."""
    check(
        1,
        "reserved_physical_slots_agree",
        state_pool_reserved == backend_reserved,
        context=context,
        state_pool_reserved=state_pool_reserved,
        backend_reserved=backend_reserved,
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

    # Real production boundary (block_size=64, ring_blocks_per_slot=10 from
    # _ring_blocks_for_window(512, 64)): the min() clip is a no-op here.
    check_page_table_covers_seqlen("swa", cache_seqlens=590, n_filled_pages=10, page_size=64)
    # A widened alignment granularity that needs 7 pages but the ring was
    # only ever sized/clipped to 6: exactly the latent bug.
    try:
        check_page_table_covers_seqlen("swa", cache_seqlens=800, n_filled_pages=6, page_size=128)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    # A3 step 7-e: INV-A3-2/3/4/8.
    check_prefix_hit_state_le_kv(slot=0, kv_hit=900, state_hit=400)
    try:
        check_prefix_hit_state_le_kv(slot=0, kv_hit=100, state_hit=101)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    check_lockstep_eviction(
        key=1, kv_hash_dropped=True, checkpoint_dropped=True, kv_ref_cnt=0
    )
    check_lockstep_eviction(
        key=1, kv_hash_dropped=False, checkpoint_dropped=True, kv_ref_cnt=5
    )
    try:
        # Forward direction violated: KV hash dropped but checkpoint survived.
        check_lockstep_eviction(
            key=1, kv_hash_dropped=True, checkpoint_dropped=False, kv_ref_cnt=0
        )
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass
    try:
        # Reverse direction violated: a still-referenced KV block's hash was
        # dropped anyway (the asymmetry INV-A3-3 exists to prevent).
        check_lockstep_eviction(
            key=1, kv_hash_dropped=True, checkpoint_dropped=True, kv_ref_cnt=3
        )
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    check_referenced_resource_never_evicted(resource_id=1, ref_cnt=1, was_evicted=False)
    check_referenced_resource_never_evicted(resource_id=1, ref_cnt=0, was_evicted=True)
    try:
        check_referenced_resource_never_evicted(resource_id=1, ref_cnt=1, was_evicted=True)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    check_reserved_physical_slots_agree("laguna", state_pool_reserved=0, backend_reserved=0)
    try:
        # The real, documented §1.8 divergence: block_pool.py=1 vs laguna.py=0.
        check_reserved_physical_slots_agree("laguna", state_pool_reserved=1, backend_reserved=0)
        raise AssertionError("expected InvariantViolation")
    except InvariantViolation:
        pass

    registry.ASSERT_LEVEL = 0
    print("checks.py self-test OK")
