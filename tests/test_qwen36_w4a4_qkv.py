"""Construction and routing guards for the incremental ModelOpt W4A4 QKV path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("b12x")

from runtime.model.modelopt_linear import (  # noqa: E402
    ModelOptFP8Linear,
    ModelOptNVFP4W4A4Linear,
)
from runtime.model.qwen36_model import Qwen36Attention  # noqa: E402
from tests.test_qwen36_mtp_head import _tiny_config  # noqa: E402


def _attention(quantized: dict[str, str]) -> Qwen36Attention:
    return Qwen36Attention(_tiny_config(), 1, quantized, max_seq_len=64)


def _w4a4_quantized() -> dict[str, str]:
    return {
        "model.language_model.layers.1.self_attn.q_proj": "W4A4_NVFP4",
        "model.language_model.layers.1.self_attn.k_proj": "W4A4_NVFP4",
        "model.language_model.layers.1.self_attn.v_proj": "W4A4_NVFP4",
    }


class TestIncrementalW4A4QKVRouting:
    def test_gittensor_qkv_uses_modelopt_w4a4_linears_only(self) -> None:
        attention = _attention(_w4a4_quantized())

        assert isinstance(attention.q_proj, ModelOptNVFP4W4A4Linear)
        assert isinstance(attention.k_proj, ModelOptNVFP4W4A4Linear)
        assert isinstance(attention.v_proj, ModelOptNVFP4W4A4Linear)
        assert attention._fused_w4a4_qkv is None
        assert not attention._fused_w4a4_qkv_checked

    def test_unsloth_fp8_qkv_never_enters_w4a4_fusion(self) -> None:
        quantized = {
            "model.language_model.layers.1.self_attn.q_proj": "FP8",
            "model.language_model.layers.1.self_attn.k_proj": "FP8",
            "model.language_model.layers.1.self_attn.v_proj": "FP8",
        }
        attention = _attention(quantized)

        assert isinstance(attention.q_proj, ModelOptFP8Linear)
        assert isinstance(attention.k_proj, ModelOptFP8Linear)
        assert isinstance(attention.v_proj, ModelOptFP8Linear)
        assert attention._fused_w4a4_qkv is None
        assert not attention._fused_w4a4_qkv_checked

    def test_w4a4_cpu_projection_keeps_existing_reference_route(self) -> None:
        attention = _attention(_w4a4_quantized())
        hidden = torch.zeros(2, 32, dtype=torch.bfloat16)

        query, key, value = attention._qkv_proj(hidden)

        assert query.shape == (2, 64)
        assert key.shape == (2, 16)
        assert value.shape == (2, 16)
        assert not attention._fused_w4a4_qkv_checked
