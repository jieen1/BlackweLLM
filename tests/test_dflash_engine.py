"""CPU-only tests for DFlash engine structure and logic.

Tests the speculative decode accept/reject logic, buffer management,
and configuration constants without requiring GPU or model weights.
"""

import sys
import types

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

# runtime.backends.laguna_dflash imports runtime.backends.laguna, which
# imports runtime.compat_vllm at module level -- the sole module in that
# chain that actually needs real vllm. Stub just that one (see
# test_swa_ring_kv.py for the same pattern/rationale), bracketed to this
# file's own test window so it doesn't leak into other test files sharing
# this pytest process.
_MODULES_BEFORE: frozenset[str] = frozenset()


def setup_module() -> None:
    global _MODULES_BEFORE
    _MODULES_BEFORE = frozenset(sys.modules)
    if "runtime.compat_vllm" not in sys.modules:
        stub = types.ModuleType("runtime.compat_vllm")
        for attr in [
            "VllmConfig",
            "bind_kv_cache",
            "get_distributed_init_method",
            "get_model",
            "get_open_port",
            "init_worker_distributed_environment",
            "set_current_vllm_config",
            "set_forward_context",
            "get_flashinfer_metadata_builder",
            "get_common_attn_metadata_cls",
            "init_flashinfer_workspace",
        ]:
            setattr(stub, attr, None)
        sys.modules["runtime.compat_vllm"] = stub


def teardown_module() -> None:
    for mod_name in list(sys.modules):
        if mod_name not in _MODULES_BEFORE and mod_name.startswith("runtime."):
            sys.modules.pop(mod_name, None)


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
