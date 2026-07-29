"""B5: GDN state management extracted from DirectModelRunner.

GDN (Gated Delta Network) recurrent state operations: checkpoint
allocation/materialization/eviction, slot reset, and state
snapshot/restore for prefix cache warm-continue.

``self._r`` is the owning runner instance.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner

from runtime.block_pool import _physical_slot


class GdnStateManager:
    """GDN recurrent state lifecycle management.

    Mechanically extracted from DirectModelRunner.
    ``self._r`` is the owning runner instance.
    """

    def __init__(self, runner: DirectModelRunner) -> None:
        self._r = runner

    def _allocate_gdn_checkpoint_pool(self) -> None:
        # Persistent full-stack GDN checkpoint pool (P3.1, notes/2026-07-19-p3-
        # implementation-plan.md step 3; R8-aware from day one). SEPARATE from
        # the live per-slot gdn_snapshot_* buffers above (those keep their
        # existing MTP role, untouched). Each checkpoint is a full 48-layer
        # (conv_state, ssm_state) snapshot at an exact prefix boundary -- the
        # recurrent state, so its size is INDEPENDENT of prefix length (~151 MB
        # measured, the ~604 MB/4-slot figure). Fixed-address discipline: a pool
        # slot's per-layer tensors are allocated once (lazily on first use) and
        # never reallocated; the pool is bounded by max_checkpoints =
        # byte_budget // per_checkpoint_bytes (default 8 GB => ~53 slots). Only
        # called when enable_persistent_prefix_cache is on, so the default-off
        # production path allocates nothing here (byte-for-byte P2).
        self._r._gdn_ckpt_conv_shape: dict[str, tuple] = {}
        self._r._gdn_ckpt_ssm_shape: dict[str, tuple] = {}
        self._r._gdn_ckpt_conv_dtype: dict[str, torch.dtype] = {}
        self._r._gdn_ckpt_ssm_dtype: dict[str, torch.dtype] = {}
        per_checkpoint_bytes = 0
        for name in self._r.gdn_layer_names:
            conv_state, ssm_state = self._r.kv_caches[name]
            # Column-0 row shapes (what snapshot_gdn_state captures): one row
            # per layer, shape shape[1:]. The K spec rows are per-slot scratch,
            # never cached (INV4 / MambaSpec.supports_eagle_cache_peek=False).
            self._r._gdn_ckpt_conv_shape[name] = tuple(conv_state.shape[1:])
            self._r._gdn_ckpt_ssm_shape[name] = tuple(ssm_state.shape[1:])
            self._r._gdn_ckpt_conv_dtype[name] = conv_state.dtype
            self._r._gdn_ckpt_ssm_dtype[name] = ssm_state.dtype
            conv_elems = 1
            for d in conv_state.shape[1:]:
                conv_elems *= int(d)
            ssm_elems = 1
            for d in ssm_state.shape[1:]:
                ssm_elems *= int(d)
            per_checkpoint_bytes += (
                conv_elems * conv_state.element_size() + ssm_elems * ssm_state.element_size()
            )
        self._r.gdn_ckpt_per_checkpoint_bytes = per_checkpoint_bytes
        self._r.gdn_ckpt_max_checkpoints = max(
            1, self._r.gdn_checkpoint_byte_budget // max(1, per_checkpoint_bytes)
        )
        # Per-layer pool-slot tensor lists, lazily allocated (None until first
        # materialize into that slot), bounded by gdn_ckpt_max_checkpoints.
        self._r.gdn_ckpt_conv: dict[str, list[torch.Tensor | None]] = {
            name: [None] * self._r.gdn_ckpt_max_checkpoints for name in self._r.gdn_layer_names
        }
        self._r.gdn_ckpt_ssm: dict[str, list[torch.Tensor | None]] = {
            name: [None] * self._r.gdn_ckpt_max_checkpoints for name in self._r.gdn_layer_names
        }
        # Meta keyed by the boundary tail block id ("key"): each entry records
        # {key, hash_value, num_tokens, pool_slot, bytes, __slot__}. The
        # hash_value tag is what makes a wrong-prefix restore REJECTED, not used
        # (R1). _gdn_ckpt_by_hash is the reverse index reconcile_prefix_hit
        # probes (sec 3.4 GDN boundary G). _gdn_ckpt_free is the free pool-slot
        # stack; _gdn_ckpt_lru (OrderedDict, oldest-first) is maintained now and
        # hardened into byte-budget eviction in P3.2 (here it is only the
        # bounded-pool safety valve when the pool is full).
        self._r.gdn_ckpt_meta: dict[int, dict] = {}
        self._r._gdn_ckpt_by_hash: dict[int, int] = {}
        self._r._gdn_ckpt_free: list[int] = list(range(self._r.gdn_ckpt_max_checkpoints))
        self._r._gdn_ckpt_lru: OrderedDict[int, None] = OrderedDict()

    def _gdn_ckpt_alloc_slot(self) -> int:
        # Pop a free pool slot, or -- only if the bounded pool is full -- evict
        # the LRU checkpoint to reclaim one (safety valve keeping the pool
        # bounded; P3.2 replaces this with real byte-budget LRU eviction in
        # lockstep with the attention index).
        if self._r._gdn_ckpt_free:
            return self._r._gdn_ckpt_free.pop()
        lru_key = next(iter(self._r._gdn_ckpt_lru))
        evicted_slot = self._r.gdn_ckpt_meta[lru_key]["pool_slot"]
        self.evict_gdn_checkpoint(lru_key)
        return evicted_slot

    def materialize_gdn_checkpoint(
        self, slot: int, key: int, hash_value: int, num_tokens: int
    ) -> None:
        # foreach_copy the 48-layer live state at _physical_slot(slot) (the
        # column-0 conv/ssm rows the just-completed forward wrote) INTO a free
        # pool slot, tagged with hash_value (R1's checkpoint-hash tag). The
        # source is read-only. Idempotent on (key, hash_value): re-materializing
        # the same boundary is a no-op. Mirrors snapshot_gdn_state's foreach_copy
        # (same column-0 rows), but into the PERSISTENT pool instead of the live
        # per-slot snapshot buffer.
        existing = self._r.gdn_ckpt_meta.get(key)
        if existing is not None:
            if existing["hash_value"] == hash_value:
                self._r._gdn_ckpt_lru.move_to_end(key)
                return
            # Same block id reused for a different prefix (post-eviction): drop
            # the stale entry first.
            self.evict_gdn_checkpoint(key)
        # P3.2 byte-budget LRU (R8): if adding this checkpoint would exceed
        # gdn_checkpoint_byte_budget, evict LRU checkpoints (lockstep with their
        # keyed attention blocks) until it fits. Checkpoints exist only at
        # chunk + completion boundaries, so this is a bounded, rare operation.
        self._evict_gdn_checkpoints_for_budget(self._r.gdn_ckpt_per_checkpoint_bytes)
        pool_slot = self._gdn_ckpt_alloc_slot()
        physical = _physical_slot(slot)
        conv_dsts, ssm_dsts, conv_srcs, ssm_srcs = [], [], [], []
        for name in self._r.gdn_layer_names:
            if self._r.gdn_ckpt_conv[name][pool_slot] is None:
                self._r.gdn_ckpt_conv[name][pool_slot] = torch.zeros(
                    self._r._gdn_ckpt_conv_shape[name],
                    dtype=self._r._gdn_ckpt_conv_dtype[name],
                    device=self._r.device,
                )
                self._r.gdn_ckpt_ssm[name][pool_slot] = torch.zeros(
                    self._r._gdn_ckpt_ssm_shape[name],
                    dtype=self._r._gdn_ckpt_ssm_dtype[name],
                    device=self._r.device,
                )
            conv_state, ssm_state = self._r.kv_caches[name]
            conv_dsts.append(self._r.gdn_ckpt_conv[name][pool_slot])
            ssm_dsts.append(self._r.gdn_ckpt_ssm[name][pool_slot])
            conv_srcs.append(conv_state[physical])
            ssm_srcs.append(ssm_state[physical])
        torch._foreach_copy_(conv_dsts, conv_srcs)
        torch._foreach_copy_(ssm_dsts, ssm_srcs)
        self._r.gdn_ckpt_meta[key] = {
            "key": key,
            "hash_value": hash_value,
            "num_tokens": num_tokens,
            "pool_slot": pool_slot,
            "bytes": self._r.gdn_ckpt_per_checkpoint_bytes,
            "__slot__": slot,
        }
        self._r._gdn_ckpt_by_hash[hash_value] = key
        self._r._gdn_ckpt_lru[key] = None
        self._r._gdn_ckpt_lru.move_to_end(key)

    def checkpoint_view(self, key: int) -> dict | None:
        # Return a snapshot-shaped dict for the checkpoint at boundary block
        # "key", consumable UNCHANGED by the EXISTING restore_gdn_state(dest,
        # view, allow_cross_slot=True) (P3 writes no second restore). The
        # __slot__ tag is the SOURCE slot whose state was checkpointed (the
        # cross-slot path only requires it to be non-None). Returns None if no
        # checkpoint exists for key. Revives the entry in the LRU.
        meta = self._r.gdn_ckpt_meta.get(key)
        if meta is None:
            return None
        pool_slot = meta["pool_slot"]
        view: dict = {"__slot__": meta["__slot__"]}
        for name in self._r.gdn_layer_names:
            view[name] = (
                self._r.gdn_ckpt_conv[name][pool_slot],
                self._r.gdn_ckpt_ssm[name][pool_slot],
            )
        self._r._gdn_ckpt_lru.move_to_end(key)
        return view

    def evict_gdn_checkpoint(self, key: int) -> None:
        # Drop the checkpoint at boundary block "key": remove its meta + hash
        # index + LRU entry and return its pool slot to the free stack (the
        # pool-slot tensors stay allocated for reuse).
        #
        # LOCKSTEP, reverse direction (INV3/R5): the checkpoint is keyed by the
        # attention tail block id == key, so dropping the checkpoint ALSO drops
        # that block's hash -- but ONLY if the block is free (ref_cnt == 0). The
        # two halves then never disagree about what is cached: a future
        # reconcile finds A shrunk below this boundary (compute miss L=0), never
        # a ghost attention hit with no GDN state. If the block is ref_cnt > 0
        # (an active slot still references it), its hash stays -- losing only the
        # checkpoint, which merely turns a future would-be hit into a safe
        # compute miss (L = G <= A still holds). The forward direction
        # (BlockPool._evict_one reclaiming the attention block) clears block_hash
        # BEFORE calling here, so this reverse step is a no-op there.
        meta = self._r.gdn_ckpt_meta.pop(key, None)
        if meta is None:
            return
        self._r._gdn_ckpt_by_hash.pop(meta["hash_value"], None)
        self._r._gdn_ckpt_lru.pop(key, None)
        self._r._gdn_ckpt_free.append(meta["pool_slot"])
        if 0 <= key < self._r.block_pool.num_blocks:
            block = self._r.block_pool.blocks[key]
            if block.ref_cnt == 0 and block.block_hash is not None:
                self._r.block_pool.hash_to_block.pop(block.block_hash.value, None)
                block.block_hash = None

    def _evict_gdn_checkpoints_for_budget(self, incoming_bytes: int) -> None:
        # P3.2 byte-budget LRU (R8): evict LRU checkpoints (oldest-first per
        # _gdn_ckpt_lru) until adding ``incoming_bytes`` fits within
        # gdn_checkpoint_byte_budget. Each eviction is lockstep (evict_gdn_
        # checkpoint drops the co-keyed attention block's hash if free). Pure
        # bookkeeping (no tensor ops), so it is unit-testable without a GPU.
        # Never evicts the entry about to be (re-)materialized: callers handle
        # the idempotent/stale-key cases before invoking this.
        total_bytes = sum(meta["bytes"] for meta in self._r.gdn_ckpt_meta.values())
        while (
            self._r.gdn_ckpt_meta
            and total_bytes + incoming_bytes > self._r.gdn_checkpoint_byte_budget
        ):
            lru_key = next(iter(self._r._gdn_ckpt_lru))
            total_bytes -= self._r.gdn_ckpt_meta[lru_key]["bytes"]
            self.evict_gdn_checkpoint(lru_key)

    def reset_slot(self, slot: int) -> None:
        """Release a slot for reuse by a new logical request. Does not zero
        the underlying tensors -- the next prefill's has_initial_state=False
        and kv_len bookkeeping starting from 0 is what makes reuse correct,
        matching this project's established fixed-slot-generation design.

        **2026-07-17 fix** (Codex-sol review, confirmed real): this used to
        leave ``slot_draft_sync_len``/``slot_pending_draft_tokens`` at
        whatever stale value the PREVIOUS logical request left behind. A
        fresh ``mtp_prefill()`` on this slot starts its real target KV at
        position 0, but its draft-sync step-0 call reads
        ``self._r.slot_draft_sync_len[slot]`` as ``prior_kv_len`` -- if that
        was never reset, the very first MTP cycle for the NEW request
        would build attention metadata against the OLD request's leftover
        history length, an immediate correctness bug for any slot that is
        ever reused (which is this project's whole fixed-slot-generation
        premise). Now cleared alongside the pre-existing fields, matching
        the same "every persistent per-slot MTP field must be reset on
        reuse" discipline.

        **P1 (2026-07-19, notes/prefix-cache-design.md sec 5, design doc's
        risk R10)**: also releases this slot's own physical attention
        blocks back to ``self._r.block_pool`` (``ref_cnt -= 1``, re-enters the
        free queue at 0) and clears ``self._r.block_table[slot]`` to ``[]`` --
        without this, P1's on-demand allocator would leak a block every
        time a slot is reused (``_ensure_blocks`` only ever grows, it never
        shrinks). Driven by ``self._r.block_table[slot]``'s own CONTENTS, not
        ``self._r.enable_block_table``'s current value -- correct regardless
        of whether the flag was on when these blocks were allocated (a
        slot that never grew any blocks, because the flag was off the
        whole time, has an empty list here and this is a no-op)."""
        if self._r.block_table[slot]:
            # P3.2 (design doc sec 3.2/3.9): free in REVERSE logical order so a
            # slot's deep-prefix (tail) blocks are enqueued ahead of its shallow
            # ones and die first under eviction -- keeping shallow, more-shared
            # prefixes cached longer. (Among hashed blocks, free appends to the
            # LRU tail in call order, so the first-freed deep tail lands closest
            # to the evict-next front.)
            self._r.block_pool.free(list(reversed(self._r.block_table[slot])))
            self._r.block_table[slot] = []
        self._r.slot_kv_len[slot] = 0
        self._r.slot_gdn_initialized[slot] = False
        self._r.slot_draft_sync_len[slot] = 0
        self._r.slot_pending_draft_tokens[slot] = None
        # Phase 2 (2026-07-18): bootstrap value for the spec-decode GDN
        # mechanism -- see __init__'s field comment.
        self._r.slot_num_accepted_tokens[slot] = 1
        # P3 (notes/2026-07-19-p3-implementation-plan.md step 4): reset this
        # slot's LOCAL hash-chain view. The published blocks themselves stay in
        # the global content index at ref_cnt == 0 (freed above, hash retained)
        # so they remain hit-able across this reset (R10) -- only the slot's own
        # cursor/chain is cleared for reuse by a new logical request.
        self._r.slot_block_hashes[slot] = []
        self._r.slot_published_blocks[slot] = 0
        # P3.2 decode-position populate: clear this slot's committed-token
        # record (the published blocks themselves stay in the global index at
        # ref_cnt == 0, hash retained -- only the slot-local sequence is reset).
        self._r.slot_committed_tokens[slot] = []

    def snapshot_gdn_state(self, slot: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Copy out this slot's ``(conv_state, ssm_state)`` for every GDN
        layer, keyed by layer name. Building block for MTP verify's GDN
        state commit/rollback (2026-07-17 round): unlike attention's paged
        KV cache (content-addressed by position, safe to just stop
        advancing ``slot_kv_len`` past a rejected boundary), GDN's
        recurrent/chunked state has no position index to truncate to -- it
        is a single accumulated value per slot that a verify call updates
        in place. Snapshotting before a verify call and restoring here on
        partial rejection (this class's chosen strategy -- "Option A" in
        notes/direct-model-runner-design.md's MTP-semantics design
        section) is the correctness-first approach: simple to reason about
        and to verify independently of the rest of MTP (see
        ``benchmarks/mtp_gdn_rollback_check.py``), at the cost of an extra
        state copy per verify call and a recompute forward pass on
        rejection.

        **2026-07-17, Phase 1 (GPU-resident double buffer)**: returns
        GPU-resident VIEWS into a preallocated, fixed-address per-slot
        buffer (``self._r.gdn_snapshot_conv``/``self._r.gdn_snapshot_ssm``, see
        ``_allocate_gdn_snapshot_buffers``) instead of fresh CPU clones --
        the data is copied via a single D2D ``copy_`` per layer (~0.4ms at
        HBM rates, measured; see notes/2026-07-17-post-ragged-round-next-
        steps.md's section 8) instead of a blocking pageable D2H memcpy
        (89-117ms/round, per that doc's section 7). API/return shape is
        UNCHANGED (same dict keys, same per-layer ``(conv, ssm)`` tuple
        shape) -- callers (``restore_gdn_state``, and, at the time, both
        ``mtp_verify_and_commit``/``_batch``) did not need to change.

        **2026-07-18, Phase B**: neither production verify path
        (``mtp_verify_and_commit``/``_batch``) calls this method any more
        -- both migrated to the real spec-decode GDN mechanism (see
        ``mtp_verify_and_commit``'s docstring), under which state commit is
        acceptance-aware and no snapshot/restore is ever needed. This
        method is retained as a tested, standalone primitive: a falsifier
        check (before Phase B's migration) confirmed
        ``benchmarks/mtp_gdn_rollback_check.py`` tests it directly
        (independent of any MTP verify call), and several other
        diagnostics (``mtp_real_draft_check.py``, ``mtp_trace_driven_probe.py``,
        ``mtp_slot_identity_pinpoint_diag.py``, ``mtp_batch_divergence_diag.py``,
        ``phase0_nsys_gap_ledger_diag.py``) call it directly too.

        Tags the snapshot with the SOURCE slot id and this slot's current
        generation counter (``self._r.slot_gdn_snapshot_gen``, bumped on
        every snapshot) -- 2026-07-17 addition per Codex-sol's explicit
        ask for explicit per-slot state so a STALE snapshot (e.g. a caller
        accidentally holding on to one from two rounds ago) can never be
        restored by mistake; ``restore_gdn_state`` rejects a generation
        mismatch. The slot-id tag was added in a follow-up fix the same
        day: without it, a caller mistakenly restoring slot A's snapshot
        into slot B could still pass the generation check (both slots
        typically climb their OWN counters in lockstep in a symmetric
        multi-slot workload, so equal generation numbers say nothing about
        SLOT identity) -- ``restore_gdn_state`` now also rejects a
        slot-id mismatch. Also marks the snapshot ``__consumed__`` on a
        successful restore -- restoring the SAME snapshot object a second
        time now raises instead of silently succeeding (idempotent in
        this specific case since both restores would write the same
        bytes, but a caller path that restores twice by mistake is exactly
        the kind of latent bug this project's "no silent passes" standard
        exists to catch). These three invariants are unchanged by the
        Phase 1 storage-medium change -- they are checked in
        ``restore_gdn_state`` BEFORE any tensor data is read, so a stale
        snapshot is still rejected even though the underlying GPU buffer
        may since have been overwritten by a newer generation's data (see
        ``_allocate_gdn_snapshot_buffers``'s docstring for why that's
        safe)."""
        physical = _physical_slot(slot)
        self._r.slot_gdn_snapshot_gen[slot] += 1
        snapshot: dict = {
            "__slot__": slot,
            "__generation__": self._r.slot_gdn_snapshot_gen[slot],
            "__consumed__": False,
        }
        # 2026-07-17, Phase 3 (round 2, coordinator-directed fast-iteration
        # pass): replaced the per-layer Python loop's 2*len(gdn_layer_names)
        # individual ``.copy_()`` kernel launches (96 for 48 layers, x4
        # slots/round = 384 -- Phase 0's ledger figure) with TWO
        # ``torch._foreach_copy_`` calls (one for all conv tensors, one for
        # all ssm tensors) -- PyTorch's multi-tensor-apply fuses the whole
        # list into a small constant number of kernel launches regardless
        # of layer count, cutting per-round host dispatch for this phase by
        # roughly 48x. Same D2D copy semantics as before (still fixed-address
        # buffers, no reallocation, no host round-trip) -- purely a launch-
        # count reduction, not a new mechanism.
        conv_dsts, ssm_dsts, conv_srcs, ssm_srcs = [], [], [], []
        for name in self._r.gdn_layer_names:
            conv_state, ssm_state = self._r.kv_caches[name]
            conv_dsts.append(self._r.gdn_snapshot_conv[name][slot])
            ssm_dsts.append(self._r.gdn_snapshot_ssm[name][slot])
            conv_srcs.append(conv_state[physical])
            ssm_srcs.append(ssm_state[physical])
        torch._foreach_copy_(conv_dsts, conv_srcs)
        torch._foreach_copy_(ssm_dsts, ssm_srcs)
        for name, snap_conv, snap_ssm in zip(self._r.gdn_layer_names, conv_dsts, ssm_dsts):
            snapshot[name] = (snap_conv, snap_ssm)
        return snapshot

    def restore_gdn_state(
        self,
        slot: int,
        snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        allow_cross_slot: bool = False,
    ) -> None:
        """Restore this slot's GDN state from a prior
        ``snapshot_gdn_state()`` call -- writes IN PLACE into the same
        persistent ``kv_caches`` tensors (never reallocates them), so this
        is safe to call between real forward passes without disturbing any
        other slot or any fixed-address buffer a CUDA-graph-captured call
        might depend on. Rejects a stale snapshot (generation counter
        mismatch), a snapshot taken for a DIFFERENT slot, or a snapshot
        that has already been consumed by a prior restore -- see
        ``snapshot_gdn_state``'s docstring for why each of these was
        added (2026-07-17, Codex-sol review), and (2026-07-17, Phase 1)
        for why they still hold with GPU-resident snapshot storage.

        **2026-07-17, Phase 1**: ``snapshot[name]`` is now already a
        GPU-resident tensor (a view into the fixed-address per-slot
        buffer), so the restore is a single D2D ``copy_`` per layer with
        no host round-trip and no ``.to(self._r.device)`` staging step -- the
        old CPU-clone path did both a D2H (in ``snapshot_gdn_state``) and
        an H2D (here) blocking pageable-memory copy per layer per slot."""
        if allow_cross_slot:
            # P2 fan-out fork (notes/prefix-cache-design.md sec 5, "P2 --
            # Fan-out fork", and sec 3.5 step 2): restore the LEADER's
            # snapshot into a SIBLING slot. The snapshot's ``__slot__`` is
            # the leader (the SOURCE of the recurrent state), which
            # legitimately differs from the destination ``slot`` here, so
            # the same-slot guard is relaxed. The generation counter is a
            # SAME-slot staleness guard (it catches restoring a slot's own
            # long-ago snapshot after that slot re-snapshotted itself); it
            # is meaningless across slots, where freshness is instead
            # guaranteed by the caller's synchronous structure --
            # ``mtp_prefill_fanout_batch`` snapshots the leader and restores
            # every sibling within ONE atomic admission tick, with no
            # intervening re-snapshot of the leader (and the MTP verify path
            # no longer calls snapshot_gdn_state at all, so the leader's
            # snapshot buffer cannot be clobbered mid-fork). ``__consumed__``
            # is deliberately NOT set below in this mode, so the ONE leader
            # snapshot can seed all N siblings: each restore is a read-only
            # D2D copy FROM the leader's fixed-address snapshot buffer INTO
            # this slot's own kv_caches row, and the source buffer stays
            # stable for the whole fork. R1 (GDN corruption on restore) is
            # still guarded structurally -- the foreach_copy below reads the
            # same 48-layer snapshot[name] tensors the leader populated.
            if snapshot.get("__slot__") is None:
                raise RuntimeError(
                    "cross-slot GDN restore requires a real snapshot (missing __slot__ tag)"
                )
        else:
            if snapshot.get("__slot__") != slot:
                raise RuntimeError(
                    f"GDN snapshot was taken for slot {snapshot.get('__slot__')}, "
                    f"not slot {slot} -- refusing a cross-slot restore"
                )
            if snapshot.get("__consumed__"):
                raise RuntimeError(f"GDN snapshot for slot {slot} was already restored once")
            gen = snapshot.get("__generation__")
            if gen != self._r.slot_gdn_snapshot_gen[slot]:
                raise RuntimeError(
                    f"stale GDN snapshot for slot {slot}: snapshot generation {gen} != "
                    f"current {self._r.slot_gdn_snapshot_gen[slot]}"
                )
        physical = _physical_slot(slot)
        # 2026-07-17, Phase 3 (round 2): same torch._foreach_copy_
        # launch-count reduction as snapshot_gdn_state's mirror-image
        # change above.
        conv_dsts, ssm_dsts, conv_srcs, ssm_srcs = [], [], [], []
        for name in self._r.gdn_layer_names:
            conv_state, ssm_state = self._r.kv_caches[name]
            snap_conv, snap_ssm = snapshot[name]
            conv_dsts.append(conv_state[physical])
            ssm_dsts.append(ssm_state[physical])
            conv_srcs.append(snap_conv)
            ssm_srcs.append(snap_ssm)
        torch._foreach_copy_(conv_dsts, conv_srcs)
        torch._foreach_copy_(ssm_dsts, ssm_srcs)
        if not allow_cross_slot:
            snapshot["__consumed__"] = True
