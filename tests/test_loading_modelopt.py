"""Tests for the modelopt loader adapter (Track B / B1,
``runtime/loading/modelopt.py``). Needs torch (dequantization is real
tensor arithmetic, unlike ``compressed_tensors.py``'s pure string/tuple
logic) -- guarded with ``pytest.importorskip("torch")`` so this skips
cleanly under the CPU-only ci-sim job rather than erroring at collection,
matching this repo's existing convention (see ``pyproject.toml``'s note on
``pytest.importorskip("torch")``-guarded modules).

``scripts/b1_verify_nvfp4_dequant.py`` tried to cross-check the E2M1 LUT
against torch's own native ``float4_e2m1fn_x2`` cast on GPU -- that cast
turned out to be non-functional on this torch build in both directions
(2026-08-02 finding, see that script and ``runtime/loading/modelopt.py``'s
module docstring). The LUT values asserted below are instead the E2M1
format's mathematically-determined table (see that docstring); the
packing order is this module's one genuinely unverified assumption.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from runtime.loading.modelopt import (  # noqa: E402
    NVFP4_GROUP_SIZE,
    QUANT_ALGO_FP8,
    QUANT_ALGO_NVFP4,
    QUANT_ALGO_NVFP4_W4A4,
    QUANT_ALGO_UNQUANTIZED,
    classify_module,
    dequantize_fp8,
    dequantize_nvfp4,
    quantized_layers_map,
    unpack_nvfp4_to_fp32,
)
from runtime.model.fp8_lm_head import NativeFP8LMHead, native_fp8_lm_head_enabled
from runtime.model.modelopt_linear import (
    QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV,
    FusedModelOptNVFP4W4A4QKV,
    ModelOptNVFP4Linear,
    ModelOptNVFP4W4A4Linear,
    _modelopt_flashinfer_fp4_quant_enabled,
)
from runtime.model.plain_linear import PlainLinear


class TestQuantizedLayersMap:
    def test_empty_when_no_quantization_config(self):
        assert quantized_layers_map({}) == {}
        assert quantized_layers_map({"quantization_config": None}) == {}

    def test_reads_quantized_layers_dict(self):
        config = {
            "quantization_config": {
                "quantized_layers": {
                    "model.language_model.layers.0.mlp.gate_proj": {
                        "quant_algo": "W4A16_NVFP4",
                        "group_size": 16,
                    },
                    "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                }
            }
        }
        result = quantized_layers_map(config)
        assert result == {
            "model.language_model.layers.0.mlp.gate_proj": "W4A16_NVFP4",
            "model.language_model.layers.0.self_attn.q_proj": "FP8",
        }

    def test_expands_static_qwen_w4a4_config_group(self):
        config = {
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "quantization_config": {
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                            "group_size": 16,
                            "dynamic": False,
                        },
                        "input_activations": {
                            "type": "float",
                            "num_bits": 4,
                            "group_size": 16,
                            "dynamic": False,
                        },
                    }
                },
            },
        }
        result = quantized_layers_map(config)
        assert len(result) == 13
        assert result["model.language_model.layers.0.linear_attn.in_proj_qkv"] == (
            QUANT_ALGO_NVFP4_W4A4
        )
        assert result["model.language_model.layers.1.self_attn.o_proj"] == (QUANT_ALGO_NVFP4_W4A4)
        assert "model.language_model.layers.0.linear_attn.in_proj_a" not in result


class TestClassifyModule:
    def test_fp8(self):
        assert classify_module("x", {"x": "FP8"}) == QUANT_ALGO_FP8

    def test_nvfp4(self):
        assert classify_module("x", {"x": "W4A16_NVFP4"}) == QUANT_ALGO_NVFP4

    def test_nvfp4_w4a4(self):
        assert classify_module("x", {"x": "W4A4_NVFP4"}) == QUANT_ALGO_NVFP4_W4A4

    def test_absent_is_unquantized(self):
        assert classify_module("x", {}) == QUANT_ALGO_UNQUANTIZED
        assert classify_module("x", {"y": "FP8"}) == QUANT_ALGO_UNQUANTIZED

    def test_unknown_algo_raises(self):
        with pytest.raises(ValueError, match="quant_algo"):
            classify_module("x", {"x": "SOMETHING_NEW"})


class TestDequantizeFp8:
    def test_matches_hand_computed_value(self):
        # 2.0 (E4M3-representable exactly) * scale 0.5 -> 1.0
        weight = torch.tensor([[2.0, -4.0], [1.0, 0.0]], dtype=torch.float8_e4m3fn)
        scale = torch.tensor(0.5, dtype=torch.float32)
        out = dequantize_fp8(weight, scale)
        assert out.dtype == torch.bfloat16
        expected = torch.tensor([[1.0, -2.0], [0.5, 0.0]], dtype=torch.bfloat16)
        assert torch.equal(out, expected)

    def test_scalar_scale_shape_is_flexible(self):
        weight = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
        # A shape-(1,) scale (as loaded straight from a safetensors scalar
        # tensor) must work the same as a true 0-d scalar.
        scale_1d = torch.tensor([2.0], dtype=torch.float32)
        out = dequantize_fp8(weight, scale_1d)
        assert out.item() == 2.0


class TestUnpackNvfp4:
    # Textbook OCP E2M1 table (index = 4-bit nibble):
    # 0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:3, 6:4, 7:6, 8:-0, 9:-0.5, 10:-1,
    # 11:-1.5, 12:-2, 13:-3, 14:-4, 15:-6.
    EXPECTED_LUT = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]

    def test_lut_matches_textbook_e2m1_table(self):
        for nibble, expected in enumerate(self.EXPECTED_LUT):
            byte = torch.tensor([[nibble]], dtype=torch.uint8)  # high nibble 0
            out = unpack_nvfp4_to_fp32(byte)
            assert out.shape == (1, 2)
            assert out[0, 0].item() == expected  # low nibble
            assert out[0, 1].item() == self.EXPECTED_LUT[0]  # high nibble = 0 -> value 0.0

    def test_packing_order_low_nibble_is_even_element(self):
        # byte 0x21: low nibble=1 (-> 0.5), high nibble=2 (-> 1.0).
        byte = torch.tensor([[0x21]], dtype=torch.uint8)
        out = unpack_nvfp4_to_fp32(byte)
        assert out[0, 0].item() == 0.5  # even element = low nibble
        assert out[0, 1].item() == 1.0  # odd element = high nibble

    def test_rejects_non_uint8(self):
        with pytest.raises(ValueError, match="uint8"):
            unpack_nvfp4_to_fp32(torch.zeros(2, 2, dtype=torch.float32))


class TestDequantizeNvfp4:
    def test_matches_hand_computed_value(self):
        # in_dim=32, group_size=16 -> 2 blocks. out_dim=1.
        # First 16 packed bytes (32 elements) all nibble=2 (value 1.0);
        # scale for block 0 = 2.0 (fp8), block 1 = 4.0 (fp8); global scale=0.5.
        # Expected: block0 elements = 1.0 * 2.0 * 0.5 = 1.0
        #           block1 elements = 1.0 * 4.0 * 0.5 = 2.0
        weight_u8 = torch.full((1, 16), 0x22, dtype=torch.uint8)  # both nibbles = 2 -> 1.0 each
        weight_scale = torch.tensor([[2.0, 4.0]], dtype=torch.float8_e4m3fn)
        weight_scale_2 = torch.tensor(0.5, dtype=torch.float32)
        out = dequantize_nvfp4(weight_u8, weight_scale, weight_scale_2, group_size=16)
        assert out.shape == (1, 32)
        assert out.dtype == torch.bfloat16
        assert torch.allclose(out[0, :16], torch.full((16,), 1.0, dtype=torch.bfloat16))
        assert torch.allclose(out[0, 16:], torch.full((16,), 2.0, dtype=torch.bfloat16))

    def test_rejects_bad_block_shape(self):
        weight_u8 = torch.zeros(1, 16, dtype=torch.uint8)  # in_dim=32
        bad_scale = torch.zeros(1, 3, dtype=torch.float8_e4m3fn)  # wrong block count
        with pytest.raises(ValueError, match="weight_scale shape"):
            dequantize_nvfp4(weight_u8, bad_scale, torch.tensor(1.0), group_size=16)

    def test_rejects_non_divisible_group_size(self):
        weight_u8 = torch.zeros(1, 15, dtype=torch.uint8)  # in_dim=30, not a multiple of 16
        scale = torch.zeros(1, 2, dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="not a multiple"):
            dequantize_nvfp4(weight_u8, scale, torch.tensor(1.0), group_size=16)

    def test_default_group_size_matches_checkpoint(self):
        # down_proj real shape: [5120, 17408] weight -> [5120, 8704] packed,
        # weight_scale [5120, 1088] (17408 // 16 == 1088, B0-2 verified).
        assert NVFP4_GROUP_SIZE == 16


class TestModelOptNvfp4W4A4Linear:
    def test_native_w4a16_head_is_a_cuda_only_optional_path(self):
        lin = ModelOptNVFP4Linear(32, 1, native_w4a16=True)
        lin.weight.data.fill_(0x22)
        lin.weight_scale.data.fill_(1.0)
        lin.weight_scale_2.data.fill_(0.25)

        assert not lin.prepare_native_w4a16()
        out = lin(torch.ones(1, 32, dtype=torch.bfloat16))
        assert out.item() == 8.0

    def test_activation_quantizer_switch_defaults_to_flashinfer_and_is_reversible(
        self, monkeypatch
    ):
        monkeypatch.delenv(QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV, raising=False)
        assert _modelopt_flashinfer_fp4_quant_enabled()

        monkeypatch.setenv(QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV, "flashinfer")
        assert _modelopt_flashinfer_fp4_quant_enabled()

        monkeypatch.setenv(QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV, "local")
        assert not _modelopt_flashinfer_fp4_quant_enabled()

    def test_loads_input_scale_and_normalizes_modelopt_scales_for_b12x(self):
        lin = ModelOptNVFP4W4A4Linear(32, 2)
        lin.weight.data.zero_()
        lin.weight_scale.data.fill_(1.0)
        lin.weight_scale_2.data.fill_(0.25)
        lin.input_scale.data.fill_(0.5)

        _, _, weight_gs, activation_gs = lin.nvfp4_w4a4_components_for_fuse()

        assert "input_scale" in dict(lin.named_parameters())
        assert weight_gs.item() == 4.0
        assert activation_gs.item() == 2.0

    def test_cpu_forward_keeps_exact_modelopt_w4a16_reference(self):
        lin = ModelOptNVFP4W4A4Linear(32, 1)
        lin.weight.data.fill_(0x22)  # E2M1 code 1.0 in both nibbles
        lin.weight_scale.data.fill_(1.0)
        lin.weight_scale_2.data.fill_(0.25)
        lin.input_scale.data.fill_(0.5)

        out = lin(torch.ones(1, 32, dtype=torch.bfloat16))
        assert out.item() == 8.0


class TestNativeQwen38LmHead:
    def test_modelopt_is_default_and_compressed_tensors_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("QSR_NATIVE_QWEN38_LM_HEAD_FP8", raising=False)

        assert native_fp8_lm_head_enabled(
            {"quantization_config": {"quant_method": "modelopt"}}
        )
        assert not native_fp8_lm_head_enabled(
            {"quantization_config": {"quant_method": "compressed-tensors"}}
        )

    def test_explicit_setting_overrides_checkpoint_default(self, monkeypatch):
        monkeypatch.setenv("QSR_NATIVE_QWEN38_LM_HEAD_FP8", "0")
        assert not native_fp8_lm_head_enabled(
            {"quantization_config": {"quant_method": "modelopt"}}
        )

        monkeypatch.setenv("QSR_NATIVE_QWEN38_LM_HEAD_FP8", "1")
        assert native_fp8_lm_head_enabled({})

    def test_conversion_requires_cuda_resident_weights(self):
        linear = PlainLinear(32, 4)

        with pytest.raises(RuntimeError, match="requires CUDA"):
            NativeFP8LMHead.from_plain_linear(linear)


class TestFusedModelOptNvfp4W4A4QKV:
    @staticmethod
    def _projections() -> tuple[
        ModelOptNVFP4W4A4Linear,
        ModelOptNVFP4W4A4Linear,
        ModelOptNVFP4W4A4Linear,
    ]:
        projections = (
            ModelOptNVFP4W4A4Linear(32, 8),
            ModelOptNVFP4W4A4Linear(32, 4),
            ModelOptNVFP4W4A4Linear(32, 4),
        )
        for projection in projections:
            projection.weight.data.zero_()
            projection.weight_scale.data.fill_(1.0)
            projection.weight_scale_2.data.fill_(0.25)
            projection.input_scale.data.fill_(0.5)
        return projections

    def test_matching_calibration_is_accepted_without_touching_parameters(self):
        q_proj, k_proj, v_proj = self._projections()
        fused = FusedModelOptNVFP4W4A4QKV(q_proj, k_proj, v_proj)

        raw = fused._validate_raw_parameters()

        assert len(raw) == 8
        assert not fused.ready
        assert q_proj.weight.numel() == 8 * 16
        assert k_proj.weight.numel() == 4 * 16
        assert v_proj.weight.numel() == 4 * 16

    def test_mismatched_input_calibration_rejects_shared_quantization(self):
        q_proj, k_proj, v_proj = self._projections()
        v_proj.input_scale.data.fill_(0.75)
        fused = FusedModelOptNVFP4W4A4QKV(q_proj, k_proj, v_proj)

        with pytest.raises(ValueError, match="matching input_scale"):
            fused._validate_raw_parameters()

    def test_mismatched_weight_calibration_rejects_shared_alpha(self):
        q_proj, k_proj, v_proj = self._projections()
        k_proj.weight_scale_2.data.fill_(0.5)
        fused = FusedModelOptNVFP4W4A4QKV(q_proj, k_proj, v_proj)

        with pytest.raises(ValueError, match="matching weight_scale_2"):
            fused._validate_raw_parameters()
