"""B3/serving: ``ServerEngine``'s ``enable_mtp`` construction-time contract.

Torch-free by construction: every assertion here fires before
``ServerEngine.__init__`` ever touches ``AutoTokenizer.from_pretrained`` or
any GPU state (the checks all sit ahead of that call, same as
``enable_session_affinity``'s own N8 guard -- see
``tests/test_engine_session_affinity.py::TestSessionAffinityRejectedAtStartup``
for the established pattern this file follows). Landing MTP without a
backend guard would mean ``ServerEngine(backend="laguna", enable_mtp=True)``
either silently no-ops (an operator thinks MTP is on; it never runs) or
crashes deep inside ``_load_laguna_model`` with a confusing
``AttributeError`` the first time a request is served -- this is the same
"fail loud at construction, before any GPU work" discipline N8 established
for the identical ``enable_session_affinity``/``warm_continue`` mismatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.engine import ServerEngine, _cuda_graph_extra_slots, _qwen_kv_bundle_bytes


def test_qwen_bundle_budget_includes_backbone_and_mtp_storage_dtypes() -> None:
    torch = pytest.importorskip("torch")
    backbone_attn = SimpleNamespace(
        num_kv_heads=2,
        head_dim=4,
        kv_cache_dtype=torch.float8_e4m3fn,
    )
    mtp_attn = SimpleNamespace(
        num_kv_heads=1,
        head_dim=4,
        kv_cache_dtype=torch.bfloat16,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(layer_type="full_attention", self_attn=backbone_attn),
                SimpleNamespace(layer_type="linear_attention"),
            ]
        ),
        mtp=SimpleNamespace(layers=[SimpleNamespace(self_attn=mtp_attn)]),
    )

    assert _qwen_kv_bundle_bytes(model, include_mtp=False) == 2 * 128 * 2 * 4
    assert _qwen_kv_bundle_bytes(model, include_mtp=True) == (2 * 128 * 2 * 4 + 2 * 128 * 1 * 4 * 2)


class TestMtpRejectedForWrongBackend:
    def test_rejects_mtp_for_laguna_backend(self) -> None:
        # Raises before ServerEngine.__init__ reaches AutoTokenizer.from_
        # pretrained (this module's own docstring) -- genuinely torch- and
        # transformers-free, unlike the two tests below.
        with pytest.raises(ValueError, match="enable_mtp requires a Qwen-family backend"):
            ServerEngine(
                backend="laguna",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                enable_mtp=True,
            )

    def test_rejects_dynamic_qwen_mode_for_other_backends(self) -> None:
        with pytest.raises(ValueError, match="requires backend='qwen36'"):
            ServerEngine(
                backend="laguna",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                qwen_kv_mode="strict",
            )

    def test_elastic_mode_requires_an_explicit_byte_budget(self) -> None:
        with pytest.raises(ValueError, match="requires qwen_kv_pool_bytes > 0"):
            ServerEngine(
                backend="qwen36",
                capacity=1,
                num_slots=1,
                enable_cudagraph=True,
                qwen_kv_mode="elastic",
            )

    def test_dynamic_mtp_rejects_fixed_row_eager_cache_path(self) -> None:
        with pytest.raises(ValueError, match="requires CUDA Graph pooled MTP caches"):
            ServerEngine(
                backend="qwen36",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                enable_mtp=True,
                qwen_kv_mode="strict",
            )

    def test_dynamic_mode_refuses_unsafe_chunk_only_reservation(self) -> None:
        with pytest.raises(ValueError, match="requires full-sequence reservation"):
            ServerEngine(
                backend="qwen36",
                capacity=1,
                num_slots=1,
                enable_cudagraph=True,
                qwen_kv_mode="strict",
                qwen_kv_full_sequence_must_fit=False,
            )

    def test_default_backend_with_mtp_off_is_unaffected(self) -> None:
        # The default (and only shipped-by-default) configuration must still
        # construct without needing any qwen36-specific state at all. Needs
        # a real tokenizer load (Laguna's), unlike the rejection test above.
        pytest.importorskip("transformers")
        engine = ServerEngine(backend="laguna", capacity=1, num_slots=1, enable_cudagraph=False)
        assert engine.enable_mtp is False
        assert engine.K == 0

    def test_qwen_cuda_graph_does_not_require_a_duplicate_live_slot(self) -> None:
        assert (
            _cuda_graph_extra_slots(backend="qwen36", enable_cudagraph=True, enable_dflash=False)
            == 0
        )

    def test_laguna_decode_graph_keeps_its_dedicated_capture_slot(self) -> None:
        assert (
            _cuda_graph_extra_slots(backend="laguna", enable_cudagraph=True, enable_dflash=False)
            == 1
        )

    def test_qwen_still_rejects_fewer_slots_than_capacity(self) -> None:
        with pytest.raises(ValueError, match="num_slots=3 must be >= 4"):
            ServerEngine(
                backend="qwen36",
                capacity=4,
                num_slots=3,
                enable_cudagraph=True,
                enable_mtp=True,
                production=True,
            )

    def test_mtp_k_is_recorded_even_when_disabled(self) -> None:
        # mtp_num_speculative_tokens is stored regardless of enable_mtp, but
        # self.K (the capacity headroom the admission path reserves) must
        # stay 0 unless MTP is actually on -- a non-MTP request must not pay
        # capacity_ok()'s headroom for a feature it never uses.
        pytest.importorskip("transformers")
        engine = ServerEngine(
            backend="laguna",
            capacity=1,
            num_slots=1,
            enable_cudagraph=False,
            mtp_num_speculative_tokens=8,
        )
        assert engine.mtp_num_speculative_tokens == 8
        assert engine.K == 0


class TestExtensibleKvRejectedWhenMisconfigured:
    def test_extensible_requires_dynamic_mode(self) -> None:
        with pytest.raises(ValueError, match="requires qwen_kv_mode"):
            ServerEngine(
                backend="qwen36",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                qwen_kv_extensible=True,
            )

    def test_extensible_requires_positive_commit_buffer(self) -> None:
        with pytest.raises(ValueError, match="qwen_kv_commit_buffer_gb"):
            ServerEngine(
                backend="qwen36",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                qwen_kv_mode="strict",
                qwen_kv_extensible=True,
                qwen_kv_commit_buffer_gb=-1,
            )

    def test_extensible_records_config(self) -> None:
        engine = ServerEngine(
            backend="qwen36",
            capacity=1,
            num_slots=1,
            enable_cudagraph=False,
            qwen_kv_mode="strict",
            qwen_kv_extensible=True,
        )
        assert engine.qwen_kv_extensible is True
        assert engine.qwen_kv_commit_buffer_gb == 10.0
