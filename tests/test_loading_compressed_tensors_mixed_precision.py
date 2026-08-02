"""Tensor-arithmetic tests for the compressed-tensors "mixed-precision"
adapter (Track B, unsloth's ``unsloth/Qwen3.6-27B-NVFP4``). Needs torch --
guarded with ``pytest.importorskip("torch")`` so this skips cleanly under the
CPU-only ci-sim job, matching ``tests/test_loading_modelopt.py``'s existing
convention (and keeping ``tests/test_loading_compressed_tensors.py`` itself
torch-free, see that file's own docstring).

Two things are tested here that a real GPU cannot add anything to:
synthetic-tensor round trips (:func:`dequantize_fp8_channel`, and the
NVFP4 pack/unpack math reused unchanged from
``runtime.loading.modelopt`` -- already covered by
``tests/test_loading_modelopt.py``, exercised again here only through the
new :class:`~runtime.model.compressed_tensors_linear.CompressedTensorsNVFP4Linear`
wiring), and the new Linear classes' Parameter names/shapes, which is exactly
what decides whether a real checkpoint's tensors land anywhere at all. What
this file does NOT and cannot claim: that dequantizing a REAL checkpoint's
weights this way reproduces the checkpoint's own intended numerics end to
end -- that needs a GPU forward against the real model, out of scope here
(see the adapter's PR/commit description for what remains unverified).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from runtime.loading.compressed_tensors import dequantize_fp8_channel  # noqa: E402
from runtime.model.compressed_tensors_linear import (  # noqa: E402
    CompressedTensorsFP8ChannelLinear,
    CompressedTensorsNVFP4Linear,
)


class TestDequantizeFp8Channel:
    def test_matches_hand_computed_value_per_row(self):
        # Row 0 scale 0.5, row 1 scale 2.0 -- distinct per output channel,
        # unlike modelopt's single scalar (this is the whole point of the
        # function: a scalar-shaped call would apply only one row's scale
        # to every row).
        weight = torch.tensor([[2.0, -4.0], [1.0, 0.0]], dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[0.5], [2.0]], dtype=torch.bfloat16)
        out = dequantize_fp8_channel(weight, scale)
        assert out.dtype == torch.bfloat16
        expected = torch.tensor([[1.0, -2.0], [2.0, 0.0]], dtype=torch.bfloat16)
        assert torch.equal(out, expected)

    def test_rejects_non_fp8_weight(self):
        weight = torch.zeros(2, 2, dtype=torch.float32)
        scale = torch.zeros(2, 1, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="float8_e4m3fn"):
            dequantize_fp8_channel(weight, scale)

    def test_rejects_scale_with_wrong_element_count(self):
        weight = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(3, 1, dtype=torch.bfloat16)  # should be 4 elements, one per row
        with pytest.raises(ValueError, match="one per output channel"):
            dequantize_fp8_channel(weight, scale)

    def test_flat_shape_1d_scale_also_works(self):
        # As loaded straight from a safetensors [out, 1] tensor, this is
        # already 2-D -- but the function should not care about a stray
        # extra/missing singleton dim as long as the element count matches.
        weight = torch.full((3, 2), 1.0, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16)
        out = dequantize_fp8_channel(weight, scale)
        assert torch.equal(out[:, 0], torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16))


class TestCompressedTensorsFP8ChannelLinear:
    def test_parameter_names_match_the_real_checkpoint_suffixes(self):
        # Must be exactly {"weight", "weight_scale"} -- Qwen36ForCausalLMSelfBuilt
        # .load_weights does no per-tensor remapping for the backbone, only a
        # fixed top-level prefix strip, so these names have to equal the
        # checkpoint's own tensor suffixes verbatim.
        lin = CompressedTensorsFP8ChannelLinear(8, 4, bias=False)
        names = {name for name, _ in lin.named_parameters()}
        assert names == {"weight", "weight_scale"}
        assert lin.weight.shape == (4, 8)
        assert lin.weight.dtype == torch.float8_e4m3fn
        assert lin.weight_scale.shape == (4, 1)
        assert lin.weight_scale.dtype == torch.bfloat16

    def test_forward_matches_hand_computed_value(self):
        lin = CompressedTensorsFP8ChannelLinear(4, 2, bias=False)
        lin.weight.data.copy_(torch.full((2, 4), 2.0, dtype=torch.float8_e4m3fn))
        lin.weight_scale.data.copy_(torch.tensor([[1.0], [0.5]], dtype=torch.bfloat16))
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        out = lin(x)
        # row 0: (2.0 * 1.0) summed over 4 inputs of 1.0 = 8.0
        # row 1: (2.0 * 0.5) summed over 4 inputs of 1.0 = 4.0
        assert torch.equal(out, torch.tensor([[8.0, 4.0]], dtype=torch.bfloat16))

    def test_dequant_cache_is_lazy_and_reused(self):
        lin = CompressedTensorsFP8ChannelLinear(4, 2, bias=False)
        lin.weight.data.copy_(torch.ones(2, 4, dtype=torch.float8_e4m3fn))
        lin.weight_scale.data.copy_(torch.ones(2, 1, dtype=torch.bfloat16))
        assert lin._weight_bf16 is None
        lin(torch.ones(1, 4, dtype=torch.bfloat16))
        cached = lin._weight_bf16
        assert cached is not None
        lin(torch.ones(1, 4, dtype=torch.bfloat16))
        assert lin._weight_bf16 is cached  # not recomputed


class TestCompressedTensorsNVFP4Linear:
    def test_parameter_names_match_the_real_checkpoint_suffixes(self):
        # Must be exactly these three -- .input_global_scale (the fourth
        # real checkpoint tensor per module) is deliberately never given a
        # Parameter here (see class docstring); Qwen36ForCausalLMSelfBuilt's
        # _IGNORED_WEIGHT_SUFFIXES lets it fall on the floor without being
        # mistaken for a missing-parameter bug.
        lin = CompressedTensorsNVFP4Linear(32, 4, bias=False)
        names = {name for name, _ in lin.named_parameters()}
        assert names == {"weight_packed", "weight_scale", "weight_global_scale"}
        assert lin.weight_packed.shape == (4, 16)
        assert lin.weight_packed.dtype == torch.uint8
        assert lin.weight_scale.shape == (4, 2)  # 32 // group_size(16)
        assert lin.weight_scale.dtype == torch.float8_e4m3fn
        assert lin.weight_global_scale.shape == ()
        assert lin.weight_global_scale.dtype == torch.float32

    def test_forward_matches_hand_computed_value(self):
        # Same worked example as tests/test_loading_modelopt.py's
        # TestDequantizeNvfp4.test_matches_hand_computed_value, run through
        # the Linear wrapper instead of the bare function: in_dim=32,
        # group_size=16 -> 2 blocks, both nibbles=2 (code value 1.0), block
        # scales 2.0/4.0. checkpoint-side weight_global_scale is 2.0 here --
        # _ensure_ready reciprocates it to the effective 0.5 the dequant
        # math actually uses (see class docstring: unsloth's
        # weight_global_scale is the reciprocal of modelopt's
        # weight_scale_2, measured off real checkpoint headers, 2026-08-03)
        # -> per-element values 1.0/2.0, same as the modelopt-side example.
        lin = CompressedTensorsNVFP4Linear(32, 1, bias=False)
        lin.weight_packed.data.copy_(torch.full((1, 16), 0x22, dtype=torch.uint8))
        lin.weight_scale.data.copy_(torch.tensor([[2.0, 4.0]], dtype=torch.float8_e4m3fn))
        lin.weight_global_scale.data.copy_(torch.tensor(2.0))
        x = torch.ones(1, 32, dtype=torch.bfloat16)
        out = lin(x)
        # weight row = [1.0]*16 + [2.0]*16 -> dot with all-ones input =
        # 16*1.0 + 16*2.0 = 48.0
        assert out.item() == 48.0

    def test_weight_global_scale_is_reciprocated_not_used_directly(self):
        # The regression this class exists to guard against (2026-08-03):
        # a real GPU run that passed weight_global_scale straight into
        # dequantize_nvfp4, unreciprocated, produced a layer-0 MLP weight
        # with mean=247.8/std=426744 (nonsense for a neural net weight) and
        # cascaded into "!!!!!!!!!!!!!!!!!!!!" degenerate model output.
        # Pinned here with the real checkpoint's own measured value
        # (layers.0.mlp.gate_proj.weight_global_scale == 6624.0) so a
        # future edit that drops the reciprocal fails a fast CPU test
        # instead of needing a full GPU generation run to notice.
        lin = CompressedTensorsNVFP4Linear(32, 1, bias=False)
        lin.weight_packed.data.copy_(torch.full((1, 16), 0x22, dtype=torch.uint8))  # code 1.0
        lin.weight_scale.data.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float8_e4m3fn))
        lin.weight_global_scale.data.copy_(torch.tensor(6624.0))
        x = torch.ones(1, 32, dtype=torch.bfloat16)
        out = lin(x)
        naive_wrong = 32 * 1.0 * 6624.0  # what "no reciprocal" would produce
        correct = 32 * 1.0 * (1.0 / 6624.0)
        assert abs(out.item() - correct) < 1e-3
        assert abs(out.item() - naive_wrong) > 1.0

    def test_rejects_odd_input_size(self):
        with pytest.raises(ValueError, match="must be even"):
            CompressedTensorsNVFP4Linear(31, 4, bias=False)

    def test_rejects_input_size_not_a_multiple_of_group_size(self):
        with pytest.raises(ValueError, match="multiple of group_size"):
            CompressedTensorsNVFP4Linear(20, 4, bias=False, group_size=16)

    def test_dequant_cache_is_lazy_and_reused(self):
        lin = CompressedTensorsNVFP4Linear(32, 2, bias=False)
        lin.weight_packed.data.copy_(torch.zeros(2, 16, dtype=torch.uint8))
        lin.weight_scale.data.copy_(torch.ones(2, 2, dtype=torch.float8_e4m3fn))
        lin.weight_global_scale.data.copy_(torch.tensor(1.0))
        assert lin._weight_bf16 is None
        lin(torch.ones(1, 32, dtype=torch.bfloat16))
        cached = lin._weight_bf16
        assert cached is not None
        lin(torch.ones(1, 32, dtype=torch.bfloat16))
        assert lin._weight_bf16 is cached
