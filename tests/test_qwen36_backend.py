"""Track B / B2: :class:`runtime.backends.qwen36.Qwen36Backend`.

Two kinds of claim live in this file, and they are kept apart on purpose:

* **Contract shape** -- ``check_conformance`` against ``ModelBackend``.
  Mechanical, no model needed.
* **Slot / prefix-cache / checkpoint bookkeeping** -- run against a stub
  model on CPU. This is where the two-cache-family invariants live, and
  every one of them fails *silently* when broken (INV-A3-1/2/3: "不是崩溃
  ——是某个请求的输出因为另一个请求的写入而改变"). Pinning them to a
  deterministic fake is the only way to get a red light out of them at
  all; a GPU run of a 27B checkpoint would report the same bug as slightly
  worse output quality, if at all.

What is deliberately NOT here: anything about numerics. "Batched decode is
bit-exact against B1's eager path", "the CUDA Graph replays what eager
computes", "concurrency >= 2 actually runs" are claims about a real
checkpoint on a real GPU and are made by ``scripts/b2_verify_serving.py``,
not faked here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")

from runtime.backends.protocol import (  # noqa: E402
    BackendCapabilities,
    PrefixHit,
    check_conformance,
)
from runtime.backends.qwen36 import Qwen36Backend, _prefix_hash  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

_VOCAB = 32
_CONV_DIM = 8
_CONV_K = 4
_V_HEADS = 2
_HEAD_DIM = 4


class _StubModel:
    """A model-shaped object that advances state the way the real graph does.

    It reproduces the two side effects ``Qwen36Backend`` depends on and
    would otherwise be silently assuming: ``state.num_tokens_seen`` and
    each attention cache's ``seq_len`` advance by the number of tokens
    forwarded. Logits are a deterministic function of the last input token
    so a test can assert *which* token was sampled without pretending to
    model anything.
    """

    def __init__(self, layer_types: list[str]) -> None:
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
                self_attn = SimpleNamespace(num_kv_heads=2, head_dim=_HEAD_DIM, num_heads=4)
            layers.append(
                SimpleNamespace(
                    layer_idx=i, layer_type=kind, linear_attn=linear_attn, self_attn=self_attn
                )
            )
        self.model = SimpleNamespace(layers=layers)
        self.forward_lengths: list[int] = []
        self.prefill_batches: list[list[list[int]]] = []
        self.logit_sequence_lengths: list[int] = []

    def __call__(self, input_ids, state):
        seq_len = int(input_ids.shape[1])
        self.forward_lengths.append(seq_len)
        state.num_tokens_seen += seq_len
        for cache in state.attn_caches:
            if cache is not None:
                cache.seq_len += seq_len
        return input_ids.to(torch.float32).unsqueeze(-1)  # [1, seq, 1]

    def decode_batch(self, batch):
        # The batched path's bookkeeping is advanced by the pool before the
        # forward, so this only has to produce logits -- deliberately by the
        # same rule as compute_logits, so a test can compare the two decode
        # paths' *token* choices without the stub encoding an opinion.
        self.forward_lengths.append(int(batch.input_ids.shape[0]))
        out = torch.zeros(batch.input_ids.shape[0], _VOCAB)
        for i, tok in enumerate(batch.input_ids[:, 0].tolist()):
            out[i, (int(tok) + 1) % _VOCAB] = 1.0
        return out

    def prefill_batch(self, batch):
        self.prefill_batches.append(batch.input_ids.tolist())
        return batch.input_ids.to(torch.float32).unsqueeze(-1)

    def compute_logits(self, hidden):
        # hidden is [seq, 1]; produce a one-hot-ish row per position whose
        # argmax is (last_token + 1) % vocab -- deterministic and distinct.
        seq = hidden.shape[0]
        self.logit_sequence_lengths.append(seq)
        out = torch.zeros(seq, _VOCAB)
        for i in range(seq):
            out[i, (int(hidden[i, 0]) + 1) % _VOCAB] = 1.0
        return out


class _PrefixMTPState:
    """Minimal MTP-prefix facade for exercising the backend ownership hook.

    MTP's cache implementation is covered by ``test_qwen36_mtp_engine``;
    this deliberately small fake pins the separate backend responsibility:
    it may restore target/GDN prefix state only when MTP can restore the
    identically-sized causal prefix too.
    """

    def __init__(self, *, restorable: bool) -> None:
        self.restorable = restorable
        self.restore_calls: list[tuple[int, int]] = []
        self.copy_calls: list[tuple[int, int, int]] = []

    def can_restore_prefix(self, slot: int, kv_len: int) -> bool:
        del slot, kv_len
        return self.restorable

    def restore_prefix(self, slot: int, kv_len: int) -> None:
        self.restore_calls.append((slot, kv_len))

    def copy_prefix(self, source_slot: int, target_slot: int, kv_len: int) -> None:
        self.copy_calls.append((source_slot, target_slot, kv_len))

    def drop_prefix(self, slot: int) -> None:
        del slot


def _backend(num_slots: int = 3, block_size: int = 64, **kw) -> Qwen36Backend:
    model = _StubModel(["full_attention", "linear_attention"])
    return Qwen36Backend(
        model,
        num_slots=num_slots,
        max_seq_len=512,
        block_size=block_size,
        device="cpu",
        dtype=torch.float32,
        **kw,
    )


def _run(backend: Qwen36Backend, slot: int, prompt: list[int], steps: int) -> list[int]:
    """Prefill + ``steps`` greedy decode tokens through the public API."""
    params = SamplingParams()
    state = backend.prefill_chunked_begin([slot], [prompt], params_per_slot={})
    token = state.result[slot]["anchor"]
    out = [token]
    for _ in range(steps):
        token = backend.decode_batch_sampled(
            [slot], [token], [backend.slot_state(slot).kv_len], [params]
        )[0]
        out.append(token)
    return out


class TestContractShape:
    def test_conforms_to_the_model_backend_protocol(self) -> None:
        # B3 (2026-08-03): speculative_decode=True now -- was False before
        # MTP was wired into the serving path. Using the REAL capabilities
        # value (not a hand-picked hypothetical) is the strongest form of
        # this check: it requires has_speculative_decode/
        # mtp_verify_and_commit_batch to actually exist with the protocol's
        # exact signatures, which is exactly the drift a real caller
        # (server/engine.py's classify_decode_slots + _step_sync) would hit.
        #
        # Unlike LagunaBackend.capabilities, this property reads
        # self.enable_prefix_cache -- ``.fget(None)`` is not available here,
        # so a real (stub-model) instance is used instead.
        assert check_conformance(Qwen36Backend, _backend().capabilities) == []

    def test_conforms_when_declared_without_speculative_decode_too(self) -> None:
        # The weaker hypothetical still holds (a caller that doesn't know
        # about MTP yet is not required to see it).
        caps = BackendCapabilities(
            speculative_decode=False,
            prefix_cache=True,
            cuda_graph=True,
            chunked_prefill=True,
            warm_continue=False,
        )
        assert check_conformance(Qwen36Backend, caps) == []

    def test_capabilities_are_honest_about_what_is_not_implemented(self) -> None:
        backend = _backend()
        caps = backend.capabilities
        # B3 (2026-08-03): speculative_decode flipped True -- the backend
        # CAN do MTP now (enable_mtp exists and works). has_speculative_decode
        # stays False on a fresh instance that never called enable_mtp(),
        # same split LagunaBackend's own capabilities/has_speculative_decode
        # docstring documents for DFlash. warm_continue is still honestly
        # False -- protocol.py's own docstring (N8) is about a capability
        # claimed by silence and swallowed by try/except for three years.
        assert caps.speculative_decode is True
        assert backend.has_speculative_decode is False
        assert caps.warm_continue is False

    def test_page_size_must_be_a_multiple_of_block_size(self) -> None:
        # §1.7: the divisibility that holds today holds by coincidence of two
        # independently chosen defaults, and must be checked rather than
        # assumed the moment a checkpoint-boundary policy depends on it.
        with pytest.raises(ValueError, match="multiple of"):
            _backend(block_size=48)


class TestSlotLifecycle:
    def test_fresh_backend_reports_every_slot_fresh(self) -> None:
        backend = _backend()
        assert all(backend.slot_state(s).is_fresh for s in range(3))
        assert all(snap.is_fresh for snap in backend.snapshot().slots)

    def test_prefill_then_decode_advances_kv_len_by_one_per_token(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3, 4], steps=3)
        assert backend.slot_state(0).kv_len == 4 + 3

    def test_reset_zeroes_recurrent_state_of_that_slot_only(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=1)
        _run(backend, 1, [9, 9, 9], steps=1)
        for gdn in backend.pool.slot_state(0).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(3.0)
        for gdn in backend.pool.slot_state(1).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(4.0)
        backend.reset_slot(0)
        assert torch.all(backend.pool.recurrent_pools[1][0] == 0.0)
        assert torch.all(backend.pool.recurrent_pools[1][1] == 4.0)

    def test_reset_preserves_the_prefix_cache_and_double_reset_does_not_clear_it(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=1)
        backend.reset_slot(0)
        saved = list(backend._prefix_cache_tokens[0] or [])
        assert saved
        backend.reset_slot(0)  # admission-time second reset
        assert backend._prefix_cache_tokens[0] == saved

    def test_decode_refuses_a_scheduler_kv_length_it_disagrees_with(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=0)
        with pytest.raises(RuntimeError, match="scheduler says"):
            backend.decode_batch_sampled([0], [7], [999], [SamplingParams()])

    def test_empty_decode_round_is_a_no_op(self) -> None:
        assert _backend().decode_batch_sampled([], [], [], []) == []

    def test_prefill_into_a_dirty_slot_is_refused(self) -> None:
        # Without this guard the GDN layers would continue from the previous
        # sequence's recurrent state: no exception, no NaN, just a wrong
        # continuation. That is INV-A3-1's symptom and the reason reset_slot
        # zeroes rather than marks.
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=1)
        with pytest.raises(RuntimeError, match="must reset_slot first"):
            backend.prefill_chunked_begin([0], [[4, 5, 6]])

    def test_equal_length_multi_slot_prefill_uses_the_batched_model_entry(
        self, monkeypatch
    ) -> None:
        backend = _backend(num_slots=2)
        built: list[tuple[list[int], list[list[int]]]] = []

        def build(slots: list[int], tokens: list[list[int]]):
            built.append((list(slots), [list(row) for row in tokens]))
            return SimpleNamespace(
                input_ids=torch.tensor(tokens, dtype=torch.long),
            )

        monkeypatch.setattr(backend.pool, "build_prefill_batch", build)
        state = backend.prefill_chunked_begin([0, 1], [[1, 2, 3], [9, 8, 7]])

        assert built == [([0, 1], [[1, 2, 3], [9, 8, 7]])]
        assert backend.model.prefill_batches == [[[1, 2, 3], [9, 8, 7]]]
        assert backend.model.forward_lengths == []
        assert state.done is True
        assert backend.pool.slot_kv_len[:2] == [3, 3]
        assert backend.stats["prefill_batched_forwards"] == 1
        # Two anchors are projected together; the rest of the prompt remains
        # available as hidden state for MTP sync but never reaches lm_head.
        assert backend.model.logit_sequence_lengths == [2]

    def test_serial_prefill_projects_only_the_final_hidden_position(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3, 4], steps=0)
        assert backend.model.logit_sequence_lengths == [1]


class TestPrefixCacheTwoFamilies:
    def test_cold_backend_reports_no_hit(self) -> None:
        assert _backend().reconcile_prefix_hit([1, 2, 3]) == PrefixHit(kv_hit=0, state_hit=0)

    def test_disabled_prefix_cache_reports_no_hit_and_takes_no_checkpoint(self) -> None:
        backend = _backend(enable_prefix_cache=False)
        _run(backend, 0, list(range(70)), steps=0)
        backend.reset_slot(0)
        assert backend.reconcile_prefix_hit(list(range(70))) == PrefixHit(kv_hit=0, state_hit=0)
        assert backend.stats["checkpoints_taken"] == 0
        assert backend.capabilities.prefix_cache is False

    def test_kv_hits_without_a_checkpoint_are_a_compute_miss_not_a_partial_hit(self) -> None:
        # The oracle's own rule (oracle/qwen36_vllm/prefix_cache.py:135-139):
        # A>0, G=0 is a miss. Using kv_hit here is the INV-A3-2 violation --
        # it does not crash, it makes the GDN layers resume from a state that
        # is stale for [state_hit, kv_hit).
        backend = _backend(block_size=64)
        prompt = list(range(100))
        _run(backend, 0, prompt, steps=0)  # kv_len=100, no 64-boundary crossed after prefill?
        backend._evict_checkpoint(0)  # force the "KV is there, state is not" case
        backend.reset_slot(0)
        hit = backend.reconcile_prefix_hit(prompt + [999])
        assert hit.kv_hit == 64
        assert hit.state_hit == 0
        assert hit.effective == 0
        assert backend.stats["prefix_hit_split_events"] == 1

    def test_a_checkpointed_boundary_becomes_a_real_state_hit(self) -> None:
        backend = _backend(block_size=64)
        prompt = list(range(64))  # prefill lands exactly on a boundary
        _run(backend, 0, prompt, steps=0)
        assert backend.stats["checkpoints_taken"] == 1
        backend.reset_slot(0)
        hit = backend.reconcile_prefix_hit(prompt + [777, 778])
        assert hit.kv_hit == 64
        assert hit.state_hit == 64
        assert hit.effective == 64

    def test_a_hit_actually_shortens_the_forward(self) -> None:
        backend = _backend(block_size=64)
        prompt = list(range(64))
        _run(backend, 0, prompt, steps=0)
        backend.reset_slot(0)
        follow_up = prompt + [777, 778]
        backend.reconcile_prefix_hit(follow_up)  # populates the pending side table
        backend.model.forward_lengths.clear()
        backend.prefill_chunked_begin([0], [follow_up])
        # Only the two novel tokens are forwarded, not all 66.
        assert backend.model.forward_lengths == [2]
        assert backend.slot_state(0).kv_len == 66

    def test_mtp_prefix_state_restores_in_lockstep_with_target_and_gdn(self) -> None:
        backend = _backend(block_size=64)
        prompt = list(range(64))
        _run(backend, 0, prompt, steps=0)
        backend.reset_slot(0)
        mtp = _PrefixMTPState(restorable=True)
        backend._mtp = mtp  # noqa: SLF001 - pin the backend/MTP boundary directly

        follow_up = prompt + [777]
        backend.reconcile_prefix_hit(follow_up)
        assert backend._apply_prefix_hit(0, follow_up) == 64  # noqa: SLF001
        assert mtp.restore_calls == [(0, 64)]

    def test_mtp_prefix_miss_forces_a_safe_target_recompute(self) -> None:
        backend = _backend(block_size=64)
        prompt = list(range(64))
        _run(backend, 0, prompt, steps=0)
        backend.reset_slot(0)
        mtp = _PrefixMTPState(restorable=False)
        backend._mtp = mtp  # noqa: SLF001 - pin the backend/MTP boundary directly

        follow_up = prompt + [777]
        backend.reconcile_prefix_hit(follow_up)
        assert backend._apply_prefix_hit(0, follow_up) == 0  # noqa: SLF001
        assert mtp.restore_calls == []

    def test_cross_slot_hit_copies_kv_and_checkpoint_before_only_forwarding_suffix(self) -> None:
        backend = _backend(num_slots=2, block_size=64)
        prefix = list(range(64))
        _run(backend, 0, prefix, steps=0)
        backend.reset_slot(0)

        source_page = backend.pool._global_page_table[0, 0]  # noqa: SLF001
        target_page = backend.pool._global_page_table[1, 0]  # noqa: SLF001
        backend.pool.k_pools[0][source_page].fill_(3.0)
        backend.pool.v_pools[0][source_page].fill_(4.0)
        source_checkpoint = [tensor.clone() for tensor in backend._checkpoint_tensors[0]]  # noqa: SLF001

        follow_up = prefix + [777]
        backend.model.forward_lengths.clear()
        # No source-slot side channel: the target discovers the retained
        # source during prefill and must clone every causal cache family.
        backend.prefill_chunked_begin([1], [follow_up])

        assert backend.model.forward_lengths == [1]
        assert backend.slot_state(1).kv_len == 65
        # The 64-token checkpoint shares an attention page, so the one-token
        # suffix must have detached target page 0 before its write.
        assert int(backend.pool._global_page_table[1, 0]) == int(target_page)  # noqa: SLF001
        assert torch.all(backend.pool.k_pools[0][target_page] == 3.0)
        assert torch.all(backend.pool.v_pools[0][target_page] == 4.0)
        assert torch.all(backend.pool.k_pools[0][source_page] == 3.0)
        assert backend.stats["prefix_cross_slot_restores"] == 1
        restored = backend.pool.capture_recurrent_state(1)
        assert all(
            torch.equal(actual, expected) for actual, expected in zip(restored, source_checkpoint)
        )

    def test_cross_slot_hit_copies_mtp_context_in_lockstep(self) -> None:
        backend = _backend(num_slots=2, block_size=64)
        prefix = list(range(64))
        _run(backend, 0, prefix, steps=0)
        backend.reset_slot(0)
        mtp = _PrefixMTPState(restorable=True)
        backend._mtp = mtp  # noqa: SLF001 - exercise cross-cache ownership directly

        assert backend._apply_prefix_hit(1, prefix + [777]) == 64  # noqa: SLF001
        assert mtp.restore_calls == []
        assert mtp.copy_calls == [(0, 1, 64)]

    def test_scratch_prefix_survives_source_slot_reuse(self) -> None:
        """The persistent arena is not an idle-slot affinity cache.

        After slot 0 has been admitted for an unrelated prompt, its old page
        row is no longer a legal prefix source.  The retained scratch snapshot
        must nevertheless restore the old prefix into slot 1 and forward only
        its novel suffix.
        """
        backend = _backend(
            num_slots=2,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        prefix = list(range(64))
        _run(backend, 0, prefix, steps=0)
        backend.reset_slot(0)
        assert backend.stats["prefix_persistent_stores"] == 1

        # First restore into its original slot.  This used to remove the
        # persistent hash index when it discarded the superseded slot-local
        # checkpoint, making every later cross-slot lookup a false miss.
        backend.prefill_chunked_begin([0], [prefix + [123]])
        backend.reset_slot(0)
        backend.prefill_chunked_begin([0], [[999]])
        assert backend.pool.slot_kv_len[0] == 1
        backend.model.forward_lengths.clear()
        backend.prefill_chunked_begin([1], [prefix + [777]])

        assert backend.model.forward_lengths == [1]
        assert backend.pool.slot_kv_len[1] == 65
        assert backend.stats["prefix_persistent_restores"] == 2

    def test_same_prompt_repeat_restores_the_prompt_boundary_without_forward(self) -> None:
        """A full-prompt hit must skip every forward, anchor included.

        The rolling checkpoint drifts past the prompt end as soon as decode
        starts, so the exact-length entry is published at prefill commit with
        its anchor-row hidden.  A same-prompt repeat then restores KV/GDN
        state and reproduces the anchor logits from that row alone.
        """
        backend = _backend(
            num_slots=2,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        prompt = list(range(64))
        first = _run(backend, 0, prompt, steps=2)
        backend.reset_slot(0)

        backend.model.forward_lengths.clear()
        state = backend.prefill_chunked_begin([1], [prompt])

        assert backend.model.forward_lengths == []
        assert backend.stats["prefix_persistent_restores"] == 1
        assert state.result[1]["anchor"] == first[0]
        assert backend.pool.slot_kv_len[1] == len(prompt)

    def test_repeated_full_prompt_hits_stay_persistent_across_generations(self) -> None:
        """A full-prompt repeat must not orphan the persistent hash index.

        The repeat path re-publishes the same boundary as the slot's own
        rolling checkpoint immediately after restoring it from the scratch
        arena.  The checkpoint pool's hash index is one-to-one, so that
        registration used to overwrite the persistent key and every later
        repeat silently fell back to a full compute -- the alternating-hit
        corruption seen on 2026-08-05.
        """
        backend = _backend(
            num_slots=2,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        prompt = list(range(64))
        first = _run(backend, 0, prompt, steps=2)
        backend.reset_slot(0)
        assert backend.stats["prefix_persistent_stores"] == 1

        for slot, expected_restores in ((1, 1), (0, 2)):
            backend.model.forward_lengths.clear()
            state = backend.prefill_chunked_begin([slot], [prompt])
            assert backend.model.forward_lengths == []
            assert backend.stats["prefix_persistent_restores"] == expected_restores
            assert state.result[slot]["anchor"] == first[0]
            backend.reset_slot(slot)

        (entry,) = backend._persistent_prefixes.values()  # noqa: SLF001
        assert entry.checkpoint_key in backend.checkpoint_pool
        assert backend.checkpoint_pool.get_by_hash(entry.hash_value) == entry.checkpoint_key

    def test_scratch_arena_retains_multiple_lru_entries_within_checkpoint_budget(self) -> None:
        backend = _backend(
            num_slots=3,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        first = list(range(64))
        second = list(range(100, 164))
        _run(backend, 0, first, steps=0)
        backend.reset_slot(0)
        _run(backend, 1, second, steps=0)
        backend.reset_slot(1)

        # The default two-checkpoint budget holds both independent contents;
        # keeping them in different scratch pages is what makes this a cache,
        # rather than one global "last prompt" slot.
        assert len(backend._persistent_prefixes) == 2  # noqa: SLF001
        assert len(backend._persistent_free_scratch_pages) == 2  # noqa: SLF001

        backend.prefill_chunked_begin([0], [[999]])
        backend.prefill_chunked_begin([1], [[998]])
        backend.prefill_chunked_begin([2], [first + [777]])
        assert backend.pool.slot_kv_len[2] == 65
        backend.reset_slot(2)
        backend.prefill_chunked_begin([2], [second + [776]])
        assert backend.pool.slot_kv_len[2] == 65
        assert backend.stats["prefix_persistent_restores"] == 2

    def test_per_slot_chunked_prefill_cow_detaches_aliased_page(self) -> None:
        """A single-slot prefill must not write through a shared scratch page.

        The per-slot chunked path forwards through ``Qwen36GenerationState``
        attention caches whose page table is a view of the pool row.  After a
        persistent restore that row aliases the scratch arena, so the forward
        used to scatter its KV straight into the shared pages -- silently
        corrupting the persistent entry's bytes while keeping the alias
        (measured 2026-08-05: the 250K c=1 COLD overwrote the 128K entry's
        KV in place).  The batched path already COW-detaches inside
        ``build_prefill_batch``; the per-slot path must do the same before
        the model writes.
        """
        backend = _backend(
            num_slots=2,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        entry = list(range(64))
        _run(backend, 0, entry, steps=0)
        backend.reset_slot(0)
        e1 = next(iter(backend._persistent_prefixes.values()))  # noqa: SLF001
        assert e1.kv_len == 64
        scratch_page = (
            backend.pool.scratch_row * backend.pool.pages_per_slot + e1.scratch_page_offsets[0]
        )
        kv_before = (
            [t[scratch_page].clone() for t in backend.pool.k_pools if t is not None],
            [t[scratch_page].clone() for t in backend.pool.v_pools if t is not None],
        )

        # Restore into both slots so an idle alias and a live alias both
        # exist, then prefill a longer prompt whose suffix writes the aliased
        # page (hit=64 is not page-aligned, so the suffix starts mid-page).
        backend.prefill_chunked_begin([1], [entry + [777]])
        backend.reset_slot(1)
        prompt = entry + list(range(100, 292))  # 256 tokens, hit=64, suffix 192
        _run(backend, 0, prompt, steps=0)

        # The live slot's row must no longer point at the entry's scratch
        # page, and the scratch bytes must be untouched.
        row0 = list(backend.pool._page_table_host[0])  # noqa: SLF001
        assert row0[0] != scratch_page
        kv_after = (
            [t[scratch_page].clone() for t in backend.pool.k_pools if t is not None],
            [t[scratch_page].clone() for t in backend.pool.v_pools if t is not None],
        )
        assert all(torch.equal(a, b) for a, b in zip(kv_before[0], kv_after[0], strict=True))
        assert all(torch.equal(a, b) for a, b in zip(kv_before[1], kv_after[1], strict=True))
        # The new prompt's own entry still publishes at its block boundary.
        assert backend.stats["prefix_persistent_stores"] == 2
        assert any(e.kv_len == 256 for e in backend._persistent_prefixes.values())  # noqa: SLF001

    def test_prefill_commit_store_evicts_entry_aliased_by_live_slot(self) -> None:
        """A live slot's committed alias must not deadlock the scratch arena.

        A partial-hit prefill whose hit is page-aligned never writes the
        aliased pages, so at prefill-commit the live slot still pins the
        restored entry's scratch pages (refcount > 1).  Evicting that entry
        to make room for the new prompt used to fail: the detach loop only
        considered idle slots, so the store silently returned and the next
        identical request re-ran cold (measured 2026-08-05: the 250K store
        after a 128K-page-aligned context).  Detaching a live slot's
        committed (read-only) alias range unblocks it.
        """
        backend = _backend(
            num_slots=2,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        e1 = list(range(128))
        e2 = list(range(500, 628))
        _run(backend, 0, e1, steps=0)
        backend.reset_slot(0)
        _run(backend, 1, e2, steps=0)
        backend.reset_slot(1)
        assert backend.stats["prefix_persistent_stores"] == 2

        # Warm-restore e1 into slot 0, then prefill e1 + 256 suffix tokens:
        # hit=128 is page-aligned, so the suffix writes page 1 only and page
        # 0 stays aliased while slot 0 is live at the prefill-commit store.
        backend.prefill_chunked_begin([0], [e1 + [999]])
        backend.reset_slot(0)
        prompt = e1 + list(range(700, 1084))  # 512 tokens -> 4 scratch pages
        _run(backend, 0, prompt, steps=0)

        assert backend.stats["prefix_persistent_stores"] == 3
        assert backend.stats["prefix_persistent_evictions"] == 2
        (entry,) = backend._persistent_prefixes.values()  # noqa: SLF001
        assert entry.kv_len == 512
        # The evicted entries must have returned their pages to the arena.
        assert len(backend._persistent_free_scratch_pages) == backend.pool.pages_per_slot - 4

    def test_graph_recapture_invalidation_drops_every_persistent_identity(self) -> None:
        backend = _backend(
            num_slots=2,
            block_size=64,
            enable_persistent_prefix_cache=True,
        )
        _run(backend, 0, list(range(64)), steps=0)
        backend.reset_slot(0)
        assert backend._persistent_prefixes  # noqa: SLF001

        backend._clear_persistent_prefixes()  # noqa: SLF001 - capture lifecycle hook

        assert not backend._persistent_prefixes  # noqa: SLF001
        assert len(backend._persistent_free_scratch_pages) == backend.pool.pages_per_slot  # noqa: SLF001
        assert len(backend.checkpoint_pool) == 0

    def test_a_checkpoint_from_a_different_prefix_of_the_same_length_is_rejected(self) -> None:
        # Length agreement is not identity. A checkpoint produced by other
        # tokens resumes from a state that is wrong in a way nothing
        # downstream can detect.
        backend = _backend(block_size=64)
        _run(backend, 0, list(range(64)), steps=0)
        backend.reset_slot(0)
        impostor = [5] * 64 + [1]
        hit = backend.reconcile_prefix_hit(impostor)
        assert hit.state_hit == 0

    def test_find_best_slot_prefers_the_resumable_slot_over_the_deeper_kv_one(self) -> None:
        backend = _backend(num_slots=3, block_size=64)
        # slot 0: long KV match, checkpoint deliberately dropped.
        long_prompt = list(range(200))
        _run(backend, 0, long_prompt, steps=0)
        backend.reset_slot(0)
        backend._evict_checkpoint(0)
        # slot 1: shorter match, checkpoint intact.
        _run(backend, 1, long_prompt[:64], steps=0)
        backend.reset_slot(1)
        slot, depth = backend.find_best_slot_for_prompt(long_prompt + [1], [0, 1, 2])
        assert (slot, depth) == (1, 64)

    def test_reconcile_records_the_split_signal_the_design_asks_for(self) -> None:
        backend = _backend(block_size=64)
        _run(backend, 0, list(range(200)), steps=0)
        backend.reset_slot(0)
        backend._evict_checkpoint(0)
        before = backend.stats["prefix_hit_split_events"]
        backend.reconcile_prefix_hit(list(range(200)) + [1])
        assert backend.stats["prefix_hit_split_events"] == before + 1
        assert backend.stats["prefix_kv_hit_tokens"] > backend.stats["prefix_state_hit_tokens"]


class TestCheckpointLockstep:
    def test_dropping_the_kv_prefix_cascades_into_the_checkpoint(self) -> None:
        # INV-A3-3 forward direction, unconditional: the KV side has decided
        # those bytes no longer describe the tokens it thought they did, so a
        # checkpoint keyed to them can only produce a wrong resume.
        backend = _backend(block_size=64)
        prompt = list(range(64))
        _run(backend, 0, prompt, steps=0)
        backend.reset_slot(0)
        assert (0, 64) in backend.checkpoint_pool
        backend.drop_prefix_cache(0)
        assert (0, 64) not in backend.checkpoint_pool
        assert backend.reconcile_prefix_hit(prompt + [1]) == PrefixHit(kv_hit=0, state_hit=0)
        assert backend.stats["checkpoints_evicted_by_kv"] == 1

    def test_budget_pressure_evicts_the_oldest_checkpoint_of_an_idle_slot(self) -> None:
        # Reverse direction of INV-A3-3, idle case: the co-keyed KV carries
        # no live reference, so dropping its hash alongside the checkpoint is
        # allowed (oracle/qwen36_vllm/gdn_state.py:205-209's `ref_cnt == 0`
        # branch). What is NOT allowed is reclaiming *live* KV -- the next
        # test covers that side.
        backend = _backend(num_slots=3, block_size=64)
        prompts = [list(range(64)), list(range(100, 164)), list(range(200, 264))]
        for slot, prompt in enumerate(prompts):
            _run(backend, slot, prompt, steps=0)
            backend.reset_slot(slot)
        # Budget is 2 checkpoints (DEFAULT_CHECKPOINT_BUDGET_MULTIPLE); the
        # third registration must have pushed the first one out.
        assert len(backend.checkpoint_pool) == 2
        assert (0, 64) not in backend.checkpoint_pool
        assert backend.stats["checkpoints_evicted_by_budget"] == 1
        assert backend.reconcile_prefix_hit(prompts[0] + [1]) == PrefixHit(kv_hit=0, state_hit=0)
        # The two younger slots are untouched.
        assert backend.reconcile_prefix_hit(prompts[2] + [1]).state_hit == 64

    def test_budget_pressure_never_touches_a_live_slots_kv(self) -> None:
        # Reverse direction, live case: "losing only the checkpoint ... merely
        # turns a future would-be hit into a safe compute miss (L = G <= A
        # still holds)" -- gdn_state.py:196. Slot 0 is mid-generation when its
        # checkpoint is evicted, so its KV must survive and later show up as
        # the kv_hit > state_hit split the design asks to be observable.
        backend = _backend(num_slots=3, block_size=64)
        prompts = [list(range(64)), list(range(100, 164)), list(range(200, 264))]
        for slot, prompt in enumerate(prompts):
            _run(backend, slot, prompt, steps=0)  # every slot stays live
        assert (0, 64) not in backend.checkpoint_pool
        assert backend.slot_state(0).kv_len == 64
        backend.reset_slot(0)
        hit = backend.reconcile_prefix_hit(prompts[0] + [1])
        assert hit.kv_hit == 64
        assert hit.state_hit == 0

    def test_a_live_slots_checkpoint_is_not_chosen_by_budget_eviction(self) -> None:
        # INV-A3-4: a resource with a live reference is never evicted by
        # either allocator. Slot 0 is mid-generation here.
        backend = _backend(num_slots=3, block_size=64)
        _run(backend, 0, list(range(64)), steps=0)
        assert backend.checkpoint_pool.is_pinned((0, 64)) is False
        # Being live is expressed as kv_len > 0; the reverse-lockstep
        # predicate must refuse to touch such a slot's KV.
        assert backend._checkpoint_kv_is_free((0, 64)) is False
        backend.reset_slot(0)
        assert backend._checkpoint_kv_is_free((0, 64)) is True

    def test_a_slots_checkpoint_rolls_forward_rather_than_accumulating(self) -> None:
        backend = _backend(num_slots=1, block_size=64)
        _run(backend, 0, list(range(64)), steps=64)
        # Two boundaries crossed (64 and 128), one checkpoint retained.
        assert backend.stats["checkpoints_taken"] == 2
        assert len(backend.checkpoint_pool) == 1
        assert backend._checkpoint_len[0] == 128


class TestObservability:
    def test_snapshot_covers_every_slot_and_holds_values_not_references(self) -> None:
        backend = _backend(num_slots=3)
        _run(backend, 1, [1, 2, 3], steps=1)
        snap = backend.snapshot()
        assert len(snap.slots) == 3
        assert len(snap.prefix) == 3
        assert snap.dflash_cg_status == ()
        assert snap.slots[1].kv_len == 4
        # Frozen values: mutating the backend afterwards must not change it.
        _run(backend, 2, [7, 7], steps=0)
        assert snap.slots[2].kv_len == 0


def test_rolling_prefix_hash_matches_fresh_hash_across_boundaries() -> None:
    """The incremental checkpoint hash is bit-identical to a fresh full hash.

    ``_rolling_prefix_hash`` feeds only the delta since the last block
    boundary into a cached blake2b context.  If the delta bookkeeping ever
    diverges from ``_prefix_hash`` (wrong slice, stale context, byte order),
    prefix dedupe against the persistent family silently misses or, worse,
    matches the wrong boundary.  Pin the equivalence with block-aligned and
    ragged lengths, plus every context-invalidation path.
    """
    backend = Qwen36Backend.__new__(Qwen36Backend)
    backend.block_size = 16
    backend._prefix_hash_ctx = {}
    backend._prefix_hash_len = {}

    rng = __import__("random").Random(0xB10C)

    # One slot, one growing prefix: the incremental chain must agree with a
    # fresh full hash at every boundary and at ragged (non-boundary) points.
    tokens = [rng.randrange(152064) for _ in range(8192)]
    for length in (16, 32, 48, 64, 80, 100, 4096, 8192):
        assert backend._rolling_prefix_hash(0, tokens, length) == _prefix_hash(
            tokens, length
        )

    # A second slot hashes the same content independently and agrees.
    for length in (16, 32, 47, 63, 128, 255, 512):
        assert backend._rolling_prefix_hash(1, tokens, length) == _prefix_hash(
            tokens, length
        )

    # Content replacement is always preceded by a context invalidation
    # (reset_slot / _commit_prefill); simulate it and re-check from scratch.
    backend._prefix_hash_ctx.pop(0, None)
    backend._prefix_hash_len.pop(0, None)
    new_tokens = [rng.randrange(152064) for _ in range(600)]
    for length in (16, 32, 47, 63, 128, 255, 512):
        assert backend._rolling_prefix_hash(0, new_tokens, length) == _prefix_hash(
            new_tokens, length
        )

    # Shorter length after a longer one: stale context must be rebuilt.
    backend._prefix_hash_ctx.pop(0, None)
    backend._prefix_hash_len.pop(0, None)
    assert backend._rolling_prefix_hash(0, new_tokens, 64) == _prefix_hash(
        new_tokens, 64
    )
    assert backend._rolling_prefix_hash(0, new_tokens, 128) == _prefix_hash(
        new_tokens, 128
    )

    # A jump larger than one block falls back to a fresh hash and rebuilds.
    backend._prefix_hash_ctx.pop(0, None)
    backend._prefix_hash_len.pop(0, None)
    assert backend._rolling_prefix_hash(0, new_tokens, 32) == _prefix_hash(
        new_tokens, 32
    )
    assert backend._rolling_prefix_hash(0, new_tokens, 512) == _prefix_hash(
        new_tokens, 512
    )
