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
        # Must be exactly these four -- .input_global_scale (the checkpoint's
        # activation-side static scale for this format's genuine W4A4
        # scheme, `config_groups.group_1.input_activations`) now has a
        # matching Parameter too (2026-08-03 W4A4 follow-up: it used to fall
        # on the floor via Qwen36ForCausalLMSelfBuilt's default "no matching
        # Parameter -> skip" behavior, deliberately, back when only the
        # W4A16 dequant-to-BF16 path existed and never read it -- see
        # nvfp4_w4a4_components_for_fuse's docstring for the consumer that
        # now needs it loaded).
        lin = CompressedTensorsNVFP4Linear(32, 4, bias=False)
        names = {name for name, _ in lin.named_parameters()}
        assert names == {
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        }
        assert lin.weight_packed.shape == (4, 16)
        assert lin.weight_packed.dtype == torch.uint8
        assert lin.weight_scale.shape == (4, 2)  # 32 // group_size(16)
        assert lin.weight_scale.dtype == torch.float8_e4m3fn
        assert lin.weight_global_scale.shape == ()
        assert lin.weight_global_scale.dtype == torch.float32
        assert lin.input_global_scale.shape == ()
        assert lin.input_global_scale.dtype == torch.float32

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


class TestNvfp4W4A4ComponentsForFuse:
    """``nvfp4_w4a4_components_for_fuse`` (2026-08-03, W4A4-GEMM
    investigation, ``work/w4a4-20260803``) uses the OPPOSITE global-scale
    convention from ``nvfp4_components_for_fuse`` right above it: both
    ``weight_global_scale`` and ``input_global_scale`` come back verbatim,
    NOT reciprocated -- because they feed ``sparkinfer.gemm.blockscaled.mm``'s
    own ``alpha = 1 / (weight_gs * activation_gs)`` convention, not
    ``dequantize_nvfp4``'s direct-multiplier one.

    This is exactly the kind of convention that produces a plausible-looking
    but wrong model when it flips silently (the same class's own
    ``test_weight_global_scale_is_reciprocated_not_used_directly`` pins the
    OTHER method's convention for the same reason) -- measured on a real GPU
    with real checkpoint weights, not assumed: reciprocating either scale
    here collapses the output to a flat 0 (activation reciprocated -- the
    per-block scale saturates below the smallest representable e4m3 value)
    or blows it up past 1e10 (weight reciprocated -- see
    ``scripts/verify_nvfp4_w4a4_gemm_single_layer.py``'s 4-way convention
    sweep, run 2026-08-03). Only the "both direct" combination this test
    pins landed within 1% of the BF16 reference's magnitude.
    """

    def test_both_global_scales_come_back_unreciprocated(self):
        lin = CompressedTensorsNVFP4Linear(32, 4, bias=False)
        lin.weight_packed.data.copy_(torch.zeros(4, 16, dtype=torch.uint8))
        lin.weight_scale.data.copy_(torch.ones(4, 2, dtype=torch.float8_e4m3fn))
        lin.weight_global_scale.data.copy_(torch.tensor(18432.0))
        lin.input_global_scale.data.copy_(torch.tensor(376.0))

        packed, scale, weight_gs, input_gs = lin.nvfp4_w4a4_components_for_fuse()

        assert packed.data_ptr() == lin.weight_packed.data.data_ptr()
        assert scale.data_ptr() == lin.weight_scale.data.data_ptr()
        assert weight_gs.item() == 18432.0, (
            "weight_global_scale must pass through verbatim for the "
            "blockscaled.mm convention -- if this is 1/18432 instead, "
            "nvfp4_w4a4_components_for_fuse started reusing "
            "nvfp4_components_for_fuse's reciprocal by mistake"
        )
        assert input_gs.item() == 376.0, (
            "input_global_scale must pass through verbatim too -- reciprocating "
            "it (the bug this test guards) collapses every block scale below "
            "the smallest representable e4m3 value, silently zeroing the "
            "quantized activation instead of raising"
        )

    def test_differs_from_the_w4a16_reciprocating_convention(self):
        """The two methods must disagree on the same checkpoint tensor --
        if a future refactor makes them share one code path, this fails
        instead of silently applying the W4A16 kernel's convention to the
        W4A4 blockscaled kernel (or vice versa)."""
        lin = CompressedTensorsNVFP4Linear(32, 4, bias=False)
        lin.weight_packed.data.copy_(torch.zeros(4, 16, dtype=torch.uint8))
        lin.weight_scale.data.copy_(torch.ones(4, 2, dtype=torch.float8_e4m3fn))
        lin.weight_global_scale.data.copy_(torch.tensor(6624.0))
        lin.input_global_scale.data.copy_(torch.tensor(776.0))

        _, _, w4a16_gs = lin.nvfp4_components_for_fuse()
        _, _, w4a4_weight_gs, _ = lin.nvfp4_w4a4_components_for_fuse()

        assert abs(w4a16_gs.item() - 1.0 / 6624.0) < 1e-9
        assert w4a4_weight_gs.item() == 6624.0
        assert w4a16_gs.item() != w4a4_weight_gs.item()
