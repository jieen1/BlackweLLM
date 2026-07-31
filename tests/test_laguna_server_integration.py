"""E1: LagunaBackend <-> ServerEngine wiring (CPU-only, no GPU/model load).

Covers:
- classify_decode_slots: the pure predicate that routes a decode round's
  active slots to the MTP path vs. the plain sampled path used by Laguna.
- ServerEngine backend selection: real (non-GPU) Laguna-only construction,
  verifying MODEL/K/eos_token_ids are set correctly.
- LagunaBackend's new E1 surface (reconcile_prefix_hit, prefill_chunked_begin/
  step, decode_batch_sampled's signature) via __new__ bypass -- these methods
  either don't touch GPU state at all, or only do so past an early return we
  never reach in these tests.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("torch")

from server.engine import ServerEngine, classify_decode_slots


class TestClassifyDecodeSlots:
    def test_mtp_capable_reproduces_original_split(self):
        active = {
            1: {"sampled": False},
            2: {"sampled": True},
            3: {"sampled": False},
        }
        greedy, sampled = classify_decode_slots(
            [1, 2, 3], active, grammar_slots=[], mtp_capable=True
        )
        assert greedy == [1, 3]
        assert sampled == [2]

    def test_mtp_capable_grammar_slots_forced_to_sampled(self):
        active = {1: {"sampled": False}, 2: {"sampled": False}}
        greedy, sampled = classify_decode_slots([1, 2], active, grammar_slots=[2], mtp_capable=True)
        assert greedy == [1]
        assert sampled == [2]

    def test_non_mtp_backend_routes_everything_to_sampled(self):
        """Laguna (mtp_capable=False): even a 'greedy' slot skips MTP."""
        active = {1: {"sampled": False}, 2: {"sampled": True}}
        greedy, sampled = classify_decode_slots([1, 2], active, grammar_slots=[], mtp_capable=False)
        assert greedy == []
        assert sampled == [1, 2]

    def test_empty_active_slots(self):
        greedy, sampled = classify_decode_slots([], {}, grammar_slots=[], mtp_capable=True)
        assert greedy == []
        assert sampled == []


class TestServerEngineBackendSelection:
    """Real (GPU-free) ServerEngine construction -- __init__ never touches
    the GPU; model loading happens later, only on the engine thread via
    start(), which these tests never call."""

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="backend"):
            ServerEngine(backend="not-a-real-backend", capacity=1, num_slots=1)

    def test_laguna_is_the_default_backend(self):
        """vLLM removal: ServerEngine is now Laguna-only (BACKEND='laguna')."""
        engine = ServerEngine(capacity=1, num_slots=1, enable_cudagraph=False, production=True)
        assert engine.backend_name == "laguna"

    def test_laguna_backend_overrides_model_and_k(self):
        engine = ServerEngine(
            backend="laguna", capacity=1, num_slots=1, enable_cudagraph=False, production=True
        )
        assert engine.backend_name == "laguna"
        assert engine.MODEL == "poolside/Laguna-S-2.1-NVFP4"
        assert engine.K == 0
        # Laguna's generation_config.json declares eos_token_id: [2, 24] --
        # both must be in the live stop set, not just the tokenizer's single
        # eos_token (id 2).
        assert 2 in engine.eos_token_ids
        assert 24 in engine.eos_token_ids


class TestLagunaBackendE1Surface:
    """Exercises the new methods added to LagunaBackend without constructing
    a real instance (which requires a GPU + loaded model). __new__ bypasses
    __init__; every method under test either never touches `self` at all, or
    returns before reaching any GPU-backed attribute.

    Laguna imports are vLLM-free, so these tests intentionally import the
    real backend rather than masking its import closure with a test stub.
    """

    def _bare_backend(self):
        from runtime.backends.laguna import LagunaBackend

        return LagunaBackend.__new__(LagunaBackend)

    def test_rejects_unsupported_sparkinfer_page_size_before_model_load(self):
        """A stale 16-token launcher must fail before allocating GPU state."""
        from runtime.backends.laguna import LagunaBackend

        with pytest.raises(ValueError, match=r"block_size in \(64, 128\)"):
            LagunaBackend(None, block_size=16)

    def test_reconcile_prefix_hit_cold_miss(self):
        """No warm cache → always returns 0 (cold miss)."""
        backend = self._bare_backend()
        backend._prefix_cache_tokens = [None, None]
        backend._prefix_cache_kv_len = [0, 0]
        backend._pending_prefix_hits = {}
        backend.block_size = 64
        assert backend.reconcile_prefix_hit([1, 2, 3]) == 0
        assert backend.reconcile_prefix_hit([]) == 0

    def test_reconcile_prefix_hit_warm_match(self):
        """Warm cache with matching prefix → returns block-aligned hit depth."""
        backend = self._bare_backend()
        backend._prefix_cache_tokens = [[10, 20, 30, 40, 50] * 20, None]  # 100 tokens
        backend._prefix_cache_kv_len = [100, 0]
        backend._pending_prefix_hits = {}
        backend.block_size = 64
        # Prompt shares first 100 tokens → hit = 64 (block-aligned)
        prompt = [10, 20, 30, 40, 50] * 20 + [99, 98, 97]
        hit = backend.reconcile_prefix_hit(prompt)
        assert hit == 64  # block-aligned down from 100
        assert backend._pending_prefix_hits.get(0) == 64

    def test_slot_state_is_immutable_scheduler_snapshot(self):
        backend = self._bare_backend()
        backend.slot_kv_len = [3]
        backend.slot_committed_tokens = [[11, 12, 13, 14]]

        state = backend.slot_state(0)

        assert state.kv_len == 3
        assert state.committed_tokens == (11, 12, 13, 14)
        assert state.is_fresh is False
        backend.slot_committed_tokens[0].append(15)
        assert state.committed_tokens == (11, 12, 13, 14)


def test_server_engine_uses_laguna_owned_runtime_surface():
    """Keep Qwen compatibility fields out of the Laguna server path."""
    source = inspect.getsource(ServerEngine)
    for private_member in (
        "_decode_cg",
        "_dflash",
        "block_table",
        "slot_kv_len",
        "slot_committed_tokens",
    ):
        assert f"runner.{private_member}" not in source


def test_laguna_chat_response_preserves_generated_think_tags(monkeypatch):
    """Laguna-only serving must not apply Qwen's thinking-token stripping."""
    from server import app as server_app

    class _FakeEngine:
        MODEL = "laguna-test"
        capacity_tokens_per_slot = 128

        def capacity_ok(self, prompt_tokens, max_tokens):
            return True

        async def submit(self, *_args, **_kwargs):
            return {
                "committed_token_ids": [1],
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": "stop",
            }

    async def _tokenize_chat(*_args, **_kwargs):
        return [0]

    async def _tokenize_decode(*_args, **_kwargs):
        return "<think>generated by Laguna</think>answer"

    async def _noop(*_args, **_kwargs):
        return None

    def _sync_noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_app, "engine", _FakeEngine())
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    monkeypatch.setattr(server_app, "_tokenize_decode", _tokenize_decode)
    monkeypatch.setattr(server_app, "_debug_log_input", _noop)
    monkeypatch.setattr(server_app, "_debug_log_output", _sync_noop)

    response = asyncio.run(
        server_app.chat_completions(
            server_app.ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
            request=None,
        )
    )

    assert (
        response["choices"][0]["message"]["content"]
        == "<think>generated by Laguna</think>answer"
    )
