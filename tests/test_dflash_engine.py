"""CPU-only tests for DFlash engine structure and logic.

Tests the speculative decode accept/reject logic, buffer management,
and configuration constants without requiring GPU or model weights.
"""

import pytest

from runtime.backends.dflash_constants import (
    AUX_LAYER_IDS,
    DRAFT_HEAD_DIM,
    DRAFT_NUM_KV_HEADS,
    DRAFT_NUM_LAYERS,
    DRAFT_NUM_QO_HEADS,
    DRAFT_WINDOW,
    MASK_TOKEN_ID,
    NUM_QUERY_PER_REQ,
    NUM_SPECULATIVE_TOKENS,
)


class TestDFlashConstants:
    """Verify DFlash configuration constants match model config."""

    def test_speculative_tokens(self):
        assert NUM_SPECULATIVE_TOKENS == 15

    def test_query_per_req(self):
        assert NUM_QUERY_PER_REQ == 16  # 1 bonus + 15 mask

    def test_aux_layer_ids(self):
        # vLLM post-layer indexing, matches dflash_config.target_layer_ids
        assert AUX_LAYER_IDS == (2, 11, 20, 30, 39, 48)
        assert len(AUX_LAYER_IDS) == 6

    def test_mask_token_id(self):
        assert MASK_TOKEN_ID == 12

    def test_draft_architecture(self):
        assert DRAFT_NUM_LAYERS == 6
        assert DRAFT_WINDOW == 512
        assert DRAFT_NUM_QO_HEADS == 72
        assert DRAFT_NUM_KV_HEADS == 8
        assert DRAFT_HEAD_DIM == 128


class TestRingPrefixReuse:
    """Prefix rewind is valid only while every ring retains the old window."""

    def _is_safe(self, cached_kv_len, prefix_len, ring_specs):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _ring_prefix_reuse_is_safe

        return _ring_prefix_reuse_is_safe(cached_kv_len, prefix_len, ring_specs)

    def test_exact_boundary_is_safe(self):
        assert self._is_safe(4096, 4096, ((640, 512), (640, 512)))

    def test_rewind_within_spare_capacity_is_safe(self):
        assert self._is_safe(4224, 4096, ((640, 512), (640, 512)))

    def test_rewind_past_spare_capacity_is_unsafe(self):
        assert not self._is_safe(4225, 4096, ((640, 512), (640, 512)))

    def test_tightest_ring_controls_reuse(self):
        assert not self._is_safe(4160, 4096, ((640, 512), (544, 512)))

    def test_invalid_prefix_is_unsafe(self):
        assert not self._is_safe(4096, 0, ((640, 512),))
        assert not self._is_safe(4096, 4097, ((640, 512),))


class TestGreedyAcceptReject:
    """Exercise the REAL accept/reject function shared by _accept_reject
    (CUDA Graph path), _verify (eager path), and generate_verify_only
    (production path) -- runtime.backends.laguna_dflash._greedy_accept_reject
    -- rather than a hand-copied reimplementation, so a change to the real
    logic can't silently drift out of sync with what these tests check."""

    def _accept_reject(self, verify_argmax, draft_tokens, bonus_token):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _greedy_accept_reject

        return _greedy_accept_reject(verify_argmax, draft_tokens, bonus_token)

    def test_all_accepted(self):
        """All 15 draft tokens match verify → 16 tokens accepted."""
        bonus = 42
        draft = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        verify = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        accepted, n = self._accept_reject(verify, draft, bonus)
        assert n == 15
        assert len(accepted) == 16
        assert accepted[0] == bonus
        assert accepted[1:] == draft

    def test_first_rejected(self):
        """First draft token rejected → bonus + correction = 2 tokens."""
        bonus = 42
        draft = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        verify = [99, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        accepted, n = self._accept_reject(verify, draft, bonus)
        assert n == 1
        assert len(accepted) == 2
        assert accepted == [42, 99]

    def test_partial_accept(self):
        """5 accepted then rejection → 7 tokens (bonus + 5 + correction)."""
        bonus = 42
        draft = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        verify = [10, 20, 30, 40, 50, 99, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        accepted, n = self._accept_reject(verify, draft, bonus)
        assert n == 6  # 5 matches + 1 correction
        assert len(accepted) == 7
        assert accepted == [42, 10, 20, 30, 40, 50, 99]

    def test_empty_draft(self):
        """Edge case: no draft tokens."""
        accepted, n = self._accept_reject([], [], 42)
        assert n == 0
        assert accepted == [42]

    def test_last_rejected(self):
        """Only the final draft token mismatches → 14 matches + correction."""
        bonus = 1
        draft = list(range(10, 25))  # 15 tokens
        verify = list(range(10, 24)) + [999]
        accepted, n = self._accept_reject(verify, draft, bonus)
        assert n == 15
        assert accepted == [1] + list(range(10, 24)) + [999]

    def test_verify_argmax_from_logits_tensor(self):
        """End-to-end shape check: verify_argmax as produced by the real
        callers (Tensor.argmax(dim=-1).tolist()), not a hand-built list."""
        torch = pytest.importorskip("torch")

        draft = [2, 4, 6]
        bonus = 0
        # 3 positions, vocab size 8; force argmax to equal draft everywhere
        # except position 1, where it should pick index 5 instead of 4.
        logits = torch.full((3, 8), -10.0)
        for i, tok in enumerate(draft):
            logits[i, tok] = 10.0
        logits[1, 4] = -10.0
        logits[1, 5] = 10.0
        verify_argmax = logits.argmax(dim=-1).tolist()

        accepted, n = self._accept_reject(verify_argmax, draft, bonus)
        assert n == 2
        assert accepted == [0, 2, 5]


class TestVerifyOnlyAcceptReject:
    """Lock the production verify-only state transition separately from legacy."""

    def _decide(self, all_argmax, draft_tokens, bonus_token=10):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _verify_only_accept_reject

        return _verify_only_accept_reject(all_argmax, draft_tokens, bonus_token)

    def test_first_reject(self):
        decision = self._decide([99, 30, 40, 50], [20, 30, 40])
        assert decision == {
            "num_accepted": 0,
            "committed": [99],
            "rejected_at": 0,
            "context_count": 1,
            "next_anchor": 99,
        }

    def test_middle_reject(self):
        decision = self._decide([20, 99, 40, 50], [20, 30, 40])
        assert decision == {
            "num_accepted": 1,
            "committed": [20, 99],
            "rejected_at": 1,
            "context_count": 2,
            "next_anchor": 99,
        }

    def test_full_accept_emits_target_bonus(self):
        decision = self._decide([20, 30, 40, 50], [20, 30, 40])
        assert decision == {
            "num_accepted": 3,
            "committed": [20, 30, 40, 50],
            "rejected_at": None,
            "context_count": 4,
            "next_anchor": 50,
        }


class TestSampleVerifyOnlyAcceptReject:
    """E2-b (docs/e2e-and-quality-plan.md §2.2): the non-greedy sibling of
    TestVerifyOnlyAcceptReject above, exercising the REAL
    ``runtime.backends.laguna_dflash._sample_verify_only_accept_reject`` --
    same ``context_count``/``next_anchor`` bookkeeping contract, but the
    accept/reject decision itself comes from rejection sampling
    (``runtime.mtp_accept.sample_accept_reject``, already proven correct in
    tests/test_mtp_accept_sampling.py) instead of an argmax comparison.
    These tests use deterministic (RNG-independent) accept/reject edge
    cases -- q==p forces certain acceptance, disjoint support forces
    certain rejection -- so the CONTRACT (context_count/next_anchor
    derivation, dict shape) can be locked down here without re-deriving
    E2-a's statistical proof.
    """

    def _decide(self, draft_tokens, draft_probs, target_probs, generator=None):
        pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _sample_verify_only_accept_reject

        return _sample_verify_only_accept_reject(
            draft_tokens,
            draft_probs,
            target_probs,
            generator=generator or torch.Generator().manual_seed(0),
        )

    def test_certain_accept_matches_greedy_shape(self):
        """q == p at every position: certain full accept, same dict shape
        TestVerifyOnlyAcceptReject.test_full_accept_emits_target_bonus
        locks down for the greedy path (context_count = K+1, next_anchor =
        the bonus token)."""
        torch = pytest.importorskip("torch")

        uniform = torch.full((5,), 0.2, dtype=torch.float64)
        draft_probs = torch.stack([uniform, uniform, uniform])
        bonus_row = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2], dtype=torch.float64)
        target_probs = torch.stack([uniform, uniform, uniform, bonus_row])

        decision = self._decide([0, 1, 2], draft_probs, target_probs)
        assert decision["num_accepted"] == 3
        assert decision["rejected_at"] is None
        assert decision["committed"][:3] == [0, 1, 2]
        assert decision["context_count"] == 4
        assert decision["next_anchor"] == decision["committed"][3]

    def test_certain_reject_matches_greedy_shape(self):
        """Disjoint support: certain rejection at position 0, same dict
        shape TestVerifyOnlyAcceptReject.test_first_reject locks down for
        the greedy path (context_count = 1, next_anchor = the recovery
        token)."""
        torch = pytest.importorskip("torch")

        q = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64)
        p = torch.tensor([0.0, 0.0, 0.6, 0.4], dtype=torch.float64)
        draft_probs = q.unsqueeze(0)
        target_probs = torch.stack([p, p])

        decision = self._decide([0], draft_probs, target_probs)
        assert decision["num_accepted"] == 0
        assert decision["rejected_at"] == 0
        assert len(decision["committed"]) == 1
        assert decision["committed"][0] in (2, 3)  # p's support, never q's
        assert decision["context_count"] == 1
        assert decision["next_anchor"] == decision["committed"][0]

    def test_middle_reject_matches_greedy_shape(self):
        """First position certain-accepts (q==p), second certain-rejects
        (disjoint support) -- exercises the early-return-after-K=1-of-2
        path, mirroring TestVerifyOnlyAcceptReject.test_middle_reject's
        shape (context_count = 2, next_anchor = the recovery token)."""
        torch = pytest.importorskip("torch")

        uniform = torch.full((4,), 0.25, dtype=torch.float64)
        q1 = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64)
        p1 = torch.tensor([0.0, 0.0, 0.6, 0.4], dtype=torch.float64)
        draft_probs = torch.stack([uniform, q1])
        target_probs = torch.stack([uniform, p1, p1])  # 3rd row unused (rejects at 1)

        decision = self._decide([0, 0], draft_probs, target_probs)
        assert decision["num_accepted"] == 1
        assert decision["rejected_at"] == 1
        assert decision["committed"][0] == 0
        assert decision["committed"][1] in (2, 3)
        assert decision["context_count"] == 2
        assert decision["next_anchor"] == decision["committed"][1]


class TestRingBlocksForDraft:
    """Verify draft KV cache sizing against the REAL ring-blocks formula
    (runtime.backends.laguna._ring_blocks_for_window), not a hand-copied
    inline formula that could silently diverge from it."""

    def test_draft_ring_blocks(self):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna import _ring_blocks_for_window

        # Draft needs: window-1 + qo_max positions + 1 extra block
        # = 511 + 16 = 527 positions → cdiv(527, 16) + 1 = 33 + 1 = 34
        block_size = 16
        blocks = _ring_blocks_for_window(DRAFT_WINDOW, block_size, qo_max=NUM_QUERY_PER_REQ)
        assert blocks == 34

    def test_draft_kv_memory(self):
        """Draft KV cache should be small (~10-20 MB per slot)."""
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna import _ring_blocks_for_window

        block_size = 16
        blocks = _ring_blocks_for_window(DRAFT_WINDOW, block_size, qo_max=NUM_QUERY_PER_REQ)
        # KV cache shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        # dtype: bf16 (2 bytes)
        bytes_per_block = 2 * block_size * DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM * 2
        total_per_slot = blocks * bytes_per_block * DRAFT_NUM_LAYERS
        mb_per_slot = total_per_slot / (1024 * 1024)
        assert 10 < mb_per_slot < 20, f"Draft KV per slot: {mb_per_slot:.1f} MB"


class TestPrefillChunkRanges:
    """The final aux chunk must cover the complete draft SWA window."""

    def _ranges(self, prompt_len, chunk_tokens=8192, window=DRAFT_WINDOW):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna import _prefill_chunk_ranges

        return _prefill_chunk_ranges(
            0,
            prompt_len,
            chunk_tokens,
            min_final_tokens=window,
        )

    def _replay_boundary(self, *, prefix_len, prompt_len, snapshot_boundary):
        pytest.importorskip("numpy")
        pytest.importorskip("torch")
        from runtime.backends.laguna import _exact_prefix_replay_boundary

        return _exact_prefix_replay_boundary(
            prefix_len=prefix_len,
            prompt_len=prompt_len,
            chunk_tokens=8192,
            min_final_tokens=DRAFT_WINDOW,
            snapshot_boundary=snapshot_boundary,
        )

    def test_aligned_prompt_keeps_full_final_chunk(self):
        ranges = self._ranges(65536)
        assert ranges[-1] == (57344, 65536)

    def test_short_remainder_is_borrowed_from_previous_chunk(self):
        ranges = self._ranges(65568)
        assert ranges[-2] == (57344, 65056)
        assert ranges[-1] == (65056, 65568)
        assert ranges[-1][1] - ranges[-1][0] == DRAFT_WINDOW

    def test_remainder_at_window_boundary_is_unchanged(self):
        ranges = self._ranges(65536 + DRAFT_WINDOW)
        assert ranges[-1] == (65536, 65536 + DRAFT_WINDOW)

    def test_ranges_are_contiguous_and_bounded(self):
        ranges = self._ranges(65537)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 65537
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        assert all(0 < end - start <= 8192 for start, end in ranges)

    def test_prefix_replay_uses_last_shared_cold_chunk_boundary(self):
        assert (
            self._replay_boundary(
                prefix_len=55488,
                prompt_len=65536,
                snapshot_boundary=49152,
            )
            == 49152
        )

    def test_prefix_replay_rejects_a_boundary_inside_new_cold_chunk(self):
        assert (
            self._replay_boundary(
                prefix_len=55488,
                prompt_len=65536,
                snapshot_boundary=55488,
            )
            is None
        )

    def test_prefix_replay_rejects_snapshot_beyond_textual_match(self):
        assert (
            self._replay_boundary(
                prefix_len=55488,
                prompt_len=65536,
                snapshot_boundary=57344,
            )
            is None
        )


class TestPrefixChunkSnapshot:
    def test_snapshot_restores_swa_ring_at_shared_cold_boundary(self):
        torch = pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend

        class Backend:
            _prefill_chunk_tokens = 8
            _swa_window = 4
            _ring_blocks_per_slot = 2
            _swa_layer_names = ["swa"]
            slot_kv_len = [12]
            slot_committed_tokens = [[1] * 12]
            _prefix_chunk_snapshots = [None]
            kv_caches = {"swa": torch.arange(16, dtype=torch.uint8).reshape(2, 2, 4, 1, 1)}

        backend = Backend()
        LagunaBackend._capture_prefix_chunk_snapshot(backend, 0, 8)
        expected = backend.kv_caches["swa"].clone()
        backend.kv_caches["swa"].zero_()

        start = LagunaBackend.prepare_exact_prefix_replay(backend, 0, [2] * 16, 12)

        assert start == 8
        assert backend.slot_kv_len == [8]
        assert backend.slot_committed_tokens == [[2] * 8]
        assert torch.equal(backend.kv_caches["swa"], expected)

    def test_snapshot_does_not_restore_when_no_shared_boundary_exists(self):
        torch = pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend

        class Backend:
            _prefill_chunk_tokens = 8
            _swa_window = 4
            _ring_blocks_per_slot = 1
            _swa_layer_names = ["swa"]
            slot_kv_len = [12]
            slot_committed_tokens = [[1] * 12]
            _prefix_chunk_snapshots = [None]
            kv_caches = {"swa": torch.ones(2, 1, 4, 1, 1, dtype=torch.uint8)}

        backend = Backend()
        LagunaBackend._capture_prefix_chunk_snapshot(backend, 0, 8)

        assert LagunaBackend.prepare_exact_prefix_replay(backend, 0, [2] * 16, 7) is None


class TestCgCaptureFailureHandling:
    """Regression tests for the C-1 second bug: a CUDA Graph capture
    failure must never be silently swallowed into an unobservable, and
    (before the SparkinferPrefillWorkspace capacity fix) actively broken,
    eager fallback. See notes/2026-08-01-c1-c2-gpu-investigation.md and
    _attempt_cg_capture's docstring in runtime/backends/laguna_dflash.py.

    _attempt_cg_capture is a plain function (no torch/CUDA/model needed),
    so these run in every lane, including the no-torch `.[dev]` venv --
    laguna_dflash.py itself imports torch at module scope, so the import is
    still guarded by importorskip("torch") to skip cleanly there rather
    than erroring at collection.
    """

    def test_returns_captured_and_does_not_log_on_success(self, caplog):
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _attempt_cg_capture

        calls: list[str] = []
        status = _attempt_cg_capture("verify", lambda: calls.append("ran"), strict=False)

        assert status == "captured"
        assert calls == ["ran"]
        assert not any(record.levelname == "ERROR" for record in caplog.records)

    def test_swallows_and_returns_failed_when_not_strict(self, caplog):
        pytest.importorskip("torch")
        import logging

        from runtime.backends.laguna_dflash import _attempt_cg_capture

        def _boom():
            raise ValueError("fixed-capacity paged workspace exceeded")

        with caplog.at_level(logging.ERROR, logger="qwen_sm120_runtime.dflash"):
            status = _attempt_cg_capture("verify", _boom, strict=False)

        assert status == "failed"
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) == 1
        assert "verify" in error_records[0].message
        # exc_info=True must carry the real traceback, not just the message
        # -- that is the whole point of upgrading past the old bare
        # `logger.warning("...: %s", e)`.
        assert error_records[0].exc_info is not None

    def test_reraises_when_strict(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _attempt_cg_capture

        def _boom():
            raise ValueError("fixed-capacity paged workspace exceeded")

        with pytest.raises(ValueError, match="fixed-capacity paged workspace exceeded"):
            _attempt_cg_capture("verify", _boom, strict=True)

    def test_strict_capture_failure_is_not_silently_recorded_as_failed(self):
        """Strict mode's contract is "never returns failed" -- it either
        returns captured or the caller's exception propagates. A caller that
        forgot this and still recorded cg_status on the (unreachable) return
        path would be dead code, not a bug, but assert the actual contract
        rather than assume it."""
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import _attempt_cg_capture

        def _boom():
            raise RuntimeError("boom")

        try:
            _attempt_cg_capture("draft", _boom, strict=True)
        except RuntimeError:
            pass
        else:
            pytest.fail("expected the RuntimeError to propagate in strict mode")


class TestCudaGraphsHealthy:
    """DFlashEngine.cuda_graphs_healthy() is a pure function of cg_status --
    test it against a bare stand-in object (no real engine/GPU/model
    needed), matching this file's existing "fake object, real method" style.
    """

    def test_healthy_when_all_attempted_graphs_captured(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import DFlashEngine

        class Fake:
            cg_status = {"verify": "captured", "draft": "captured"}

        assert DFlashEngine.cuda_graphs_healthy(Fake()) is True

    def test_unhealthy_when_any_attempted_graph_failed(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import DFlashEngine

        class Fake:
            cg_status = {"verify": "captured", "draft": "failed"}

        assert DFlashEngine.cuda_graphs_healthy(Fake()) is False

    def test_vacuously_healthy_before_anything_is_attempted(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna_dflash import DFlashEngine

        class Fake:
            cg_status: dict[str, str] = {}

        assert DFlashEngine.cuda_graphs_healthy(Fake()) is True
