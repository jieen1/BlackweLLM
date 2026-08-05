"""Track B / B2: :class:`runtime.model.qwen36_slots.Qwen36SlotPool`.

Runs against a **stub model** rather than a real Qwen3.6 checkpoint. The
pool reads exactly six geometry attributes off each layer and nothing
else, so a stub exercises every line of its addressing and lifecycle
logic while staying runnable on any machine with torch -- the 27B
checkpoint is 50+ GiB and needs a GPU, which would turn the properties
below into a GPU-only claim for no gain. What genuinely needs the real
model (bit-exactness against B1's eager path, batched-vs-serial decode,
CUDA Graph replay) is verified by ``scripts/b2_verify_*.py`` on GPU and
cannot be faked here; this file covers the half that can be pinned down
deterministically.

The property that matters most below is the one B0-5 singled out: a
fresh slot's recurrent state must be **zeroed**, not merely marked
unused. A stale KV byte past ``kv_len`` is never read; a stale recurrent
state is read on the very first step of the next sequence and produces a
plausible, non-crashing, wrong continuation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")

from runtime.model.qwen36_slots import Qwen36SlotPool  # noqa: E402

_CONV_DIM = 8
_CONV_K = 4
_V_HEADS = 2
_HEAD_DIM = 4
_KV_HEADS = 2
_Q_HEADS = 4


def _stub_model(layer_types: list[str]):
    """Minimal stand-in exposing only what Qwen36SlotPool reads."""
    layers = []
    for i, kind in enumerate(layer_types):
        if kind == "linear_attention":
            linear_attn = SimpleNamespace(
                conv_dim=_CONV_DIM,
                conv_kernel_size=_CONV_K,
                num_v_heads=_V_HEADS,
                head_k_dim=_HEAD_DIM,
                head_v_dim=_HEAD_DIM,
            )
            self_attn = None
        else:
            linear_attn = None
            self_attn = SimpleNamespace(
                num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, num_heads=_Q_HEADS
            )
        layers.append(
            SimpleNamespace(
                layer_idx=i, layer_type=kind, linear_attn=linear_attn, self_attn=self_attn
            )
        )
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def _pool(num_slots: int = 3, max_seq_len: int = 256) -> Qwen36SlotPool:
    return Qwen36SlotPool(
        _stub_model(["full_attention", "linear_attention", "linear_attention"]),
        num_slots=num_slots,
        max_seq_len=max_seq_len,
        device="cpu",
        dtype=torch.float32,
    )


class TestGeometry:
    def test_reports_what_it_allocated_rather_than_what_was_asked_for(self) -> None:
        # max_seq_len rounds UP to a whole number of pages; reporting the
        # requested value instead of the allocated one is how a capacity
        # check and the buffer it guards stop agreeing.
        pool = _pool(max_seq_len=200)
        assert pool.geometry.page_size == 128
        assert pool.geometry.pages_per_slot == 2
        assert pool.geometry.max_seq_len == 256
        assert pool.max_seq_len == 256

    def test_counts_both_cache_families(self) -> None:
        pool = _pool()
        assert pool.geometry.num_recurrent_layers == 2
        assert pool.geometry.num_paged_kv_layers == 1

    def test_recurrent_bytes_per_slot_is_measured_not_inherited(self) -> None:
        pool = _pool()
        per_layer = (_CONV_DIM * _CONV_K + _V_HEADS * _HEAD_DIM * _HEAD_DIM) * 4  # fp32
        assert pool.geometry.recurrent_bytes_per_slot == 2 * per_layer
        assert pool.recurrent_checkpoint_nbytes() == pool.geometry.recurrent_bytes_per_slot

    def test_allocates_one_scratch_row_past_the_last_real_slot(self) -> None:
        pool = _pool(num_slots=3)
        assert pool.scratch_row == 3
        assert pool.conv_pools[1].shape[0] == 4


class TestSlotViewsAliasThePool:
    def test_per_slot_state_is_a_view_not_a_copy(self) -> None:
        pool = _pool()
        state = pool.slot_state(1)
        state.gdn_states[1].recurrent_state.fill_(7.0)
        assert torch.all(pool.recurrent_pools[1][1] == 7.0)
        # ... and touching slot 1 must not touch slot 0 or 2.
        assert torch.all(pool.recurrent_pools[1][0] == 0.0)
        assert torch.all(pool.recurrent_pools[1][2] == 0.0)

    def test_slot_state_object_identity_is_stable_across_resets(self) -> None:
        # Callers may hold the state object across requests; reset_slot
        # clears it in place rather than replacing it.
        pool = _pool()
        before = pool.slot_state(0)
        pool.reset_slot(0)
        assert pool.slot_state(0) is before

    def test_kv_page_table_maps_a_slot_to_its_initial_contiguous_range(self) -> None:
        pool = _pool()
        cache = pool.slot_state(2).attn_caches[0]
        assert cache.num_pages == pool.pages_per_slot
        assert cache.physical_num_pages == pool.k_pools[0].shape[0]
        cache.k_cache[cache.page_table[0]] = 3.0
        lo = 2 * pool.pages_per_slot
        hi = lo + pool.pages_per_slot
        assert torch.all(pool.k_pools[0][lo:hi] == 3.0)
        assert torch.all(pool.k_pools[0][:lo] == 0.0)
        assert torch.all(pool.k_pools[0][hi:] == 0.0)

    def test_local_and_global_addressing_reach_the_same_bytes(self) -> None:
        # The whole point of one allocation with two addressings: a write
        # through the per-slot view must be visible at the global page id
        # the batched path would compute for it.
        pool = _pool()
        slot, local_page = 2, 1
        cache = pool.slot_state(slot).attn_caches[0]
        cache.k_cache[cache.page_table[0, local_page]].fill_(5.0)
        global_page = slot * pool.pages_per_slot + local_page
        assert torch.all(pool.k_pools[0][global_page] == 5.0)

    def test_single_slot_append_obeys_the_same_logical_page_table_as_batching(self) -> None:
        pool = _pool(num_slots=2)
        slot = 0
        cache = pool.slot_state(slot).attn_caches[0]
        # Deliberately non-contiguous physical pages: this is the prefix-cache
        # allocation shape the old local-slice implementation could not
        # express.  `cache.page_table` is a stable view of the pool row.
        pool.set_page_table_row(slot, [1, 0])
        values = torch.arange(129 * _KV_HEADS * _HEAD_DIM, dtype=torch.float32).view(
            129, _KV_HEADS, _HEAD_DIM
        )
        cache.append(values, values + 1)
        assert torch.equal(pool.k_pools[0][1, 0], values[0])
        assert torch.equal(pool.k_pools[0][0, 0], values[128])
        assert torch.equal(pool.v_pools[0][1, 0], values[0] + 1)
        assert torch.equal(pool.v_pools[0][0, 0], values[128] + 1)

    def test_cross_slot_prefix_copy_follows_non_contiguous_page_tables(self) -> None:
        pool = _pool(num_slots=2)
        # Source and destination deliberately interleave physical pages.
        # A contiguous-slice copy would pass the ordinary layout but corrupt
        # this allocator shape as soon as page recycling is introduced.
        pool.set_page_table_row(0, [1, 0])
        pool.set_page_table_row(1, [4, 2])
        source_pages = pool._global_page_table[0]  # noqa: SLF001
        target_pages = pool._global_page_table[1]  # noqa: SLF001
        pool.k_pools[0][source_pages[0]].fill_(7.0)
        pool.k_pools[0][source_pages[1]].fill_(8.0)
        pool.v_pools[0][source_pages[0]].fill_(9.0)
        pool.v_pools[0][source_pages[1]].fill_(10.0)

        pool.copy_prefix_kv(0, 1, 129)

        assert torch.all(pool.k_pools[0][target_pages[0]] == 7.0)
        assert torch.all(pool.k_pools[0][target_pages[1]] == 8.0)
        assert torch.all(pool.v_pools[0][target_pages[0]] == 9.0)
        assert torch.all(pool.v_pools[0][target_pages[1]] == 10.0)
        pool.k_pools[0][target_pages[0]].fill_(11.0)
        assert torch.all(pool.k_pools[0][source_pages[0]] == 7.0)

    def test_shared_prefix_page_detaches_before_a_suffix_write(self) -> None:
        pool = _pool(num_slots=2, max_seq_len=256)
        source_page = int(pool._global_page_table[0, 0])  # noqa: SLF001
        target_original_page = int(pool._global_page_table[1, 0])  # noqa: SLF001
        pool.k_pools[0][source_page].fill_(7.0)
        pool.v_pools[0][source_page].fill_(9.0)

        # A 64-token GDN checkpoint is inside the 128-token attention page,
        # so sharing includes the page and COW must preserve its first half.
        pool.share_prefix_kv(0, 1, 64)
        assert int(pool._global_page_table[1, 0]) == source_page  # noqa: SLF001
        assert pool._page_refcounts[source_page] == 2  # noqa: SLF001
        assert target_original_page in pool._free_physical_pages  # noqa: SLF001

        pool.prepare_kv_writes(1, 64, 1)
        target_page = int(pool._global_page_table[1, 0])  # noqa: SLF001
        assert target_page == target_original_page
        assert pool._page_refcounts[source_page] == 1  # noqa: SLF001
        assert torch.all(pool.k_pools[0][target_page] == 7.0)
        assert torch.all(pool.v_pools[0][target_page] == 9.0)

        pool.k_pools[0][target_page].fill_(11.0)
        assert torch.all(pool.k_pools[0][source_page] == 7.0)

    def test_scratch_prefix_can_be_shared_then_detached_without_new_pages(self) -> None:
        pool = _pool(num_slots=2, max_seq_len=256)
        source_page = int(pool._global_page_table[0, 0])  # noqa: SLF001
        target_page = int(pool._global_page_table[1, 0])  # noqa: SLF001
        scratch_page = int(pool._global_page_table[pool.scratch_row, 0])  # noqa: SLF001
        pool.k_pools[0][source_page].fill_(7.0)
        pool.v_pools[0][source_page].fill_(9.0)

        pool.copy_prefix_to_scratch(0, 64)
        assert torch.all(pool.k_pools[0][scratch_page] == 7.0)
        assert torch.all(pool.v_pools[0][scratch_page] == 9.0)

        pool.share_scratch_prefix(1, 64)
        assert int(pool._global_page_table[1, 0]) == scratch_page  # noqa: SLF001
        pool.prepare_kv_writes(1, 64, 1)
        assert int(pool._global_page_table[1, 0]) == target_page  # noqa: SLF001
        assert torch.all(pool.k_pools[0][target_page] == 7.0)
        pool.k_pools[0][target_page].fill_(11.0)
        assert torch.all(pool.k_pools[0][scratch_page] == 7.0)


class TestResetZeroesRecurrentStateButNotKV:
    """B0-5's one operational requirement, as an executable assertion."""

    def test_reset_zeroes_conv_and_recurrent_state(self) -> None:
        pool = _pool()
        state = pool.slot_state(0)
        for gdn in state.gdn_states:
            if gdn is not None:
                gdn.conv_state.fill_(1.0)
                gdn.recurrent_state.fill_(1.0)
                gdn.has_previous_state = True
        pool.reset_slot(0)
        for gdn in state.gdn_states:
            if gdn is not None:
                assert torch.all(gdn.conv_state == 0.0)
                assert torch.all(gdn.recurrent_state == 0.0)
                assert gdn.has_previous_state is False

    def test_reset_does_not_zero_kv(self) -> None:
        # KV bytes are the prefix cache; zeroing them would destroy the
        # thing reset_slot exists to preserve (LagunaBackend.reset_slot's
        # own contract, kept here).
        pool = _pool()
        pool.slot_state(0).attn_caches[0].k_cache.fill_(2.0)
        pool.reset_slot(0)
        assert torch.all(pool.slot_state(0).attn_caches[0].k_cache == 2.0)
        assert pool.slot_state(0).attn_caches[0].seq_len == 0

    def test_reset_of_one_slot_leaves_every_other_slots_state_intact(self) -> None:
        pool = _pool()
        for slot in range(3):
            for gdn in pool.slot_state(slot).gdn_states:
                if gdn is not None:
                    gdn.recurrent_state.fill_(float(slot + 1))
        pool.reset_slot(1)
        assert torch.all(pool.recurrent_pools[1][0] == 1.0)
        assert torch.all(pool.recurrent_pools[1][1] == 0.0)
        assert torch.all(pool.recurrent_pools[1][2] == 3.0)

    def test_reset_clears_length_bookkeeping(self) -> None:
        pool = _pool()
        pool.slot_kv_len[0] = 17
        pool.slot_committed_tokens[0] = [1, 2, 3]
        pool.reset_slot(0)
        assert pool.slot_kv_len[0] == 0
        assert pool.slot_committed_tokens[0] == []
        assert pool.slot_state(0).num_tokens_seen == 0


class TestCheckpointCaptureRestore:
    def test_capture_is_a_clone_that_survives_a_reset(self) -> None:
        pool = _pool()
        for gdn in pool.slot_state(0).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(4.0)
        ckpt = pool.capture_recurrent_state(0)
        pool.reset_slot(0)
        assert all(torch.all(t == 4.0) for t in ckpt[1::2])

    def test_restore_puts_the_bytes_back_and_marks_state_present(self) -> None:
        pool = _pool()
        for gdn in pool.slot_state(0).gdn_states:
            if gdn is not None:
                gdn.conv_state.fill_(1.5)
                gdn.recurrent_state.fill_(2.5)
        ckpt = pool.capture_recurrent_state(0)
        pool.reset_slot(0)
        pool.restore_recurrent_state(0, ckpt)
        for gdn in pool.slot_state(0).gdn_states:
            if gdn is not None:
                assert torch.all(gdn.conv_state == 1.5)
                assert torch.all(gdn.recurrent_state == 2.5)
                assert gdn.has_previous_state is True

    def test_restore_into_a_second_slot_does_not_alias_the_first(self) -> None:
        # docs/a3-cache-coordinator-design.md §6 pitfall 4: two requests
        # restoring from one checkpoint in the same round is safe *because*
        # restore copies. If it ever aliased, this test goes red.
        pool = _pool()
        for gdn in pool.slot_state(0).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(9.0)
        ckpt = pool.capture_recurrent_state(0)
        pool.restore_recurrent_state(1, ckpt)
        pool.restore_recurrent_state(2, ckpt)
        pool.slot_state(1).gdn_states[1].recurrent_state.fill_(0.0)
        assert torch.all(pool.slot_state(2).gdn_states[1].recurrent_state == 9.0)
        assert all(torch.all(t == 9.0) for t in ckpt[1::2])

    def test_restore_rejects_a_checkpoint_of_the_wrong_shape(self) -> None:
        pool = _pool()
        with pytest.raises(ValueError, match="checkpoint has"):
            pool.restore_recurrent_state(0, [])


class TestDecodeBatchAddressing:
    def test_write_index_lands_in_the_slots_own_page_range(self) -> None:
        pool = _pool(num_slots=3, max_seq_len=256)
        pool.slot_kv_len[0] = 0
        pool.slot_kv_len[2] = 130  # second page of slot 2, offset 2
        batch, b = pool.build_decode_batch([0, 2], [11, 22])
        assert b == 2
        # slot 0, position 0 -> global page 0, offset 0 -> row 0
        assert int(batch.write_index[0]) == 0
        # slot 2, position 130 -> local page 1 -> global page 5, offset 2
        assert int(batch.write_index[1]) == 5 * 128 + 2

    def test_cache_seqlens_include_the_token_written_this_step(self) -> None:
        pool = _pool()
        pool.slot_kv_len[1] = 7
        batch, _ = pool.build_decode_batch([1], [42])
        assert int(batch.cache_seqlens[0]) == 8
        assert int(batch.positions[0]) == 7  # position of the new token
        assert pool.slot_kv_len[1] == 8

    def test_page_table_rows_are_the_slots_global_pages(self) -> None:
        pool = _pool(num_slots=3, max_seq_len=256)
        batch, _ = pool.build_decode_batch([2, 0], [1, 2])
        assert batch.page_table[0].tolist() == [4, 5]
        assert batch.page_table[1].tolist() == [0, 1]

    def test_write_index_follows_a_remapped_page_table_row(self) -> None:
        pool = _pool(num_slots=2, max_seq_len=256)
        assert pool.page_table_version(0) == 0
        pool.set_page_table_row(0, [1, 0])
        assert pool.page_table_version(0) == 1

        first, _ = pool.build_decode_batch([0], [11])
        assert first.write_index.tolist() == [128]
        assert first.page_table.tolist() == [[1, 0]]

        pool.slot_kv_len[0] = 128
        second, _ = pool.build_decode_batch([0], [12])
        assert second.write_index.tolist() == [0]

    def test_remap_rejects_non_bijective_or_wrong_sized_rows(self) -> None:
        pool = _pool(max_seq_len=256)
        with pytest.raises(ValueError, match="needs 2"):
            pool.set_page_table_row(0, [0])
        with pytest.raises(ValueError, match="distinct"):
            pool.set_page_table_row(0, [0, 0])

    def test_batch_views_are_narrowed_from_one_persistent_buffer(self) -> None:
        # CUDA Graph capture bakes addresses in. If build_decode_batch ever
        # allocated fresh tensors, replay would read a dangling buffer -- a
        # failure that shows up as wrong output, not as a crash.
        pool = _pool()
        first, _ = pool.build_decode_batch([0], [1])
        second, _ = pool.build_decode_batch([0], [2])
        assert first.input_ids.data_ptr() == second.input_ids.data_ptr()
        assert first.page_table.data_ptr() == second.page_table.data_ptr()
        assert first.cache_seqlens.data_ptr() == second.cache_seqlens.data_ptr()

    def test_slot_state_bookkeeping_advances_with_the_batch(self) -> None:
        pool = _pool()
        pool.build_decode_batch([0, 1], [5, 6])
        for slot in (0, 1):
            state = pool.slot_state(slot)
            assert state.num_tokens_seen == 1
            assert state.attn_caches[0].seq_len == 1
        assert pool.slot_committed_tokens[0] == [5]
        assert pool.slot_committed_tokens[1] == [6]

    def test_refuses_to_decode_past_capacity(self) -> None:
        pool = _pool(max_seq_len=256)
        pool.slot_kv_len[0] = 256
        with pytest.raises(RuntimeError, match="at capacity"):
            pool.build_decode_batch([0], [1])

    def test_refuses_an_empty_batch(self) -> None:
        with pytest.raises(ValueError):
            _pool().build_decode_batch([], [])


class TestUniformPrefillBatchAddressing:
    """CPU proof for the BxQ descriptor; attention math is GPU-gated."""

    @staticmethod
    def _install_driver_stub(monkeypatch, pool: Qwen36SlotPool) -> None:
        def build(batch: int, tokens_per_slot: int):
            return SimpleNamespace(
                batch=batch,
                tokens_per_slot=tokens_per_slot,
                page_table=torch.zeros(
                    batch, pool.pages_per_slot, dtype=torch.int32, device=pool.device
                ),
                cache_seqlens=torch.zeros(batch, dtype=torch.int32, device=pool.device),
                output=torch.empty(
                    batch * tokens_per_slot,
                    _Q_HEADS,
                    _HEAD_DIM,
                    dtype=pool.dtype,
                    device=pool.device,
                ),
            )

        monkeypatch.setattr(pool, "prefill_attention_driver", build)

    def test_request_major_positions_and_global_kv_rows(self, monkeypatch) -> None:
        pool = _pool(num_slots=3, max_seq_len=256)
        self._install_driver_stub(monkeypatch, pool)
        pool.slot_kv_len[0] = pool.slot_kv_len[2] = 126
        for slot in (0, 2):
            state = pool.slot_state(slot)
            state.num_tokens_seen = 126
            for cache in state.attn_caches:
                if cache is not None:
                    cache.seq_len = 126
            for gdn in state.gdn_states:
                if gdn is not None:
                    gdn.has_previous_state = True

        batch = pool.build_prefill_batch([2, 0], [[11, 12, 13], [21, 22, 23]])

        assert batch.input_ids.tolist() == [[11, 12, 13], [21, 22, 23]]
        assert batch.positions.tolist() == [126, 127, 128, 126, 127, 128]
        # slot 2 pages start at 4; the third token crosses into page 5.
        assert batch.write_index.tolist() == [4 * 128 + 126, 4 * 128 + 127, 5 * 128, 126, 127, 128]
        assert batch.attn.page_table.tolist() == [[4, 5], [0, 1]]
        assert batch.attn.cache_seqlens.tolist() == [129, 129]
        assert batch.has_previous_state is True
        full_attention_outputs = [out for out in batch.attn_outputs if out is not None]
        assert len({out.data_ptr() for out in full_attention_outputs}) == 1

    def test_write_rows_follow_remapped_page_tables(self, monkeypatch) -> None:
        pool = _pool(num_slots=2, max_seq_len=256)
        self._install_driver_stub(monkeypatch, pool)
        pool.set_page_table_row(0, [1, 0])
        pool.set_page_table_row(1, [3, 2])
        pool.slot_kv_len[0] = pool.slot_kv_len[1] = 127
        for slot in (0, 1):
            state = pool.slot_state(slot)
            state.num_tokens_seen = 127

        batch = pool.build_prefill_batch([0, 1], [[11, 12], [21, 22]])

        assert batch.write_index.tolist() == [2 * 128 - 1, 0, 4 * 128 - 1, 2 * 128]
        assert batch.attn.page_table.tolist() == [[1, 0], [3, 2]]

    def test_commit_updates_host_lengths_only_after_forward(self, monkeypatch) -> None:
        pool = _pool(num_slots=2)
        self._install_driver_stub(monkeypatch, pool)
        pool.build_prefill_batch([0, 1], [[1, 2], [3, 4]])
        # Descriptor construction is side-effect-free with respect to
        # sequence lengths; an exception in the model must not commit KV.
        assert pool.slot_kv_len[:2] == [0, 0]
        pool.commit_prefill_batch([0, 1], [[1, 2], [3, 4]])
        assert pool.slot_kv_len[:2] == [2, 2]
        for slot in (0, 1):
            state = pool.slot_state(slot)
            assert state.num_tokens_seen == 2
            assert state.attn_caches[0].seq_len == 2
            assert all(gdn.has_previous_state for gdn in state.gdn_states if gdn is not None)

    def test_refuses_mixed_prior_lengths(self, monkeypatch) -> None:
        pool = _pool(num_slots=2)
        self._install_driver_stub(monkeypatch, pool)
        pool.slot_kv_len[1] = 1
        with pytest.raises(ValueError, match="equal prior KV lengths"):
            pool.build_prefill_batch([0, 1], [[1], [2]])


class TestRewind:
    def test_rewind_moves_the_kv_cursor_without_touching_recurrent_state(self) -> None:
        # rewind_slot is only ever called after a checkpoint restore; it must
        # not itself pretend to know what the recurrent state at that
        # boundary was.
        pool = _pool()
        for gdn in pool.slot_state(0).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(6.0)
        pool.slot_kv_len[0] = 200
        pool.rewind_slot(0, 128)
        assert pool.slot_kv_len[0] == 128
        assert pool.slot_state(0).num_tokens_seen == 128
        assert pool.slot_state(0).attn_caches[0].seq_len == 128
        assert torch.all(pool.slot_state(0).gdn_states[1].recurrent_state == 6.0)

    def test_rewind_past_capacity_is_refused(self) -> None:
        pool = _pool(max_seq_len=256)
        with pytest.raises(ValueError):
            pool.rewind_slot(0, 257)
