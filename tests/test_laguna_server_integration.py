"""E1: LagunaBackend <-> ServerEngine wiring (CPU-only, no GPU/model load).

Covers:
- classify_decode_slots: the pure predicate that routes a decode round's
  active slots to the MTP path (Qwen) vs. the plain sampled path (Laguna,
  which has no MTP yet).
- ServerEngine backend selection: real (non-GPU) construction with
  backend="laguna" vs. the default "qwen36", verifying MODEL/K/eos_token_ids
  are set correctly and the original Qwen path is untouched.
- LagunaBackend's new E1 surface (reconcile_prefix_hit, prefill_chunked_begin/
  step, decode_batch_sampled's signature) via __new__ bypass -- these methods
  either don't touch GPU state at all, or only do so past an early return we
  never reach in these tests.
"""

from __future__ import annotations

import inspect
import sys
import types

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

    def test_qwen36_default_is_unchanged(self):
        engine = ServerEngine(capacity=1, num_slots=1, enable_cudagraph=False, production=True)
        assert engine.backend_name == "qwen36"
        assert engine.MODEL == "unsloth/Qwen3.6-27B-NVFP4"
        assert engine.K == 3
        assert engine.eos_token_ids == frozenset({engine.eos_token_id})

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

    runtime.backends.laguna imports runtime.compat_vllm at module level,
    which needs real vllm -- not installed in the CPU-only test venv. Since
    __new__ skips __init__, none of these methods actually need a working
    compat_vllm (its names are only used inside __init__/other GPU-backed
    methods), so a stub with the needed names bound to None is enough to let
    the import succeed. Installed/removed per-class (not module top level)
    so it doesn't leak into other test files sharing this pytest process.
    """

    _MODULES_BEFORE: frozenset = frozenset()

    @classmethod
    def setup_class(cls) -> None:
        cls._MODULES_BEFORE = frozenset(sys.modules)
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

    @classmethod
    def teardown_class(cls) -> None:
        for mod_name in list(sys.modules):
            if mod_name not in cls._MODULES_BEFORE and mod_name.startswith("runtime."):
                sys.modules.pop(mod_name, None)

    def _bare_backend(self):
        from runtime.backends.laguna import LagunaBackend

        return LagunaBackend.__new__(LagunaBackend)

    def test_rejects_unsupported_sparkinfer_page_size_before_model_load(self):
        """A stale 16-token launcher must fail before allocating GPU state."""
        from runtime.backends.laguna import LagunaBackend

        with pytest.raises(ValueError, match=r"block_size in \(64, 128\)"):
            LagunaBackend(None, block_size=16)

    def test_reconcile_prefix_hit_is_permanent_miss(self):
        backend = self._bare_backend()
        assert backend.reconcile_prefix_hit([1, 2, 3]) == 0
        assert backend.reconcile_prefix_hit([]) == 0

    def test_prefill_chunked_begin_rejects_mismatched_lengths(self):
        backend = self._bare_backend()
        with pytest.raises(ValueError, match="equal length"):
            backend.prefill_chunked_begin([0, 1], [[1, 2]])

    def test_prefill_chunked_begin_empty_is_immediately_done(self):
        backend = self._bare_backend()
        state = backend.prefill_chunked_begin([], [])
        assert state.done is True
        assert state.result == {}

    def test_prefill_chunked_step_reflects_state_done_flag(self):
        from runtime.block_pool import ChunkedPrefillState

        backend = self._bare_backend()
        assert backend.prefill_chunked_step(ChunkedPrefillState(done=True, result={})) is True

    def test_decode_batch_sampled_signature_matches_direct_model_runner(self):
        """E1: both backends must accept the exact same positional/keyword
        shape so ServerEngine's call site works unmodified against either."""
        pytest.importorskip("vllm")
        # Unlike the __new__-bypass tests above, this one needs the REAL
        # runtime.compat_vllm (DirectModelRunner's import chain needs names
        # -- AttentionBackendEnum et al. -- the class-level stub in
        # setup_class() doesn't provide, since those tests never construct
        # a real instance). Evict the stub (and anything cached against it)
        # so this import resolves against genuine vllm, now confirmed
        # available by the importorskip above.
        for mod_name in list(sys.modules):
            if mod_name == "runtime.compat_vllm" or mod_name.startswith("runtime.backends."):
                if getattr(sys.modules[mod_name], "__file__", None) is None:
                    sys.modules.pop(mod_name, None)

        from runtime.backends.laguna import LagunaBackend
        from runtime.direct_model_runner import DirectModelRunner

        laguna_sig = inspect.signature(LagunaBackend.decode_batch_sampled)
        qwen_sig = inspect.signature(DirectModelRunner.decode_batch_sampled)
        assert list(laguna_sig.parameters) == list(qwen_sig.parameters)
        for name in laguna_sig.parameters:
            assert laguna_sig.parameters[name].kind == qwen_sig.parameters[name].kind
