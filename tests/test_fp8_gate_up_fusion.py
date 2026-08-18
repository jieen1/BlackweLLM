"""CPU coverage for the vLLM-style FP8 gate/up weight packing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.compressed_tensors_linear import (  # noqa: E402
    CompressedTensorsFP8ChannelLinear,
    FusedFP8ChannelGateUp,
)


def test_gate_up_fusion_concatenates_weights_and_channel_scales() -> None:
    gate = CompressedTensorsFP8ChannelLinear(4, 3)
    up = CompressedTensorsFP8ChannelLinear(4, 3)
    gate.weight.data.copy_(torch.arange(12, dtype=torch.float32).view(3, 4))
    up.weight.data.copy_(torch.arange(12, 24, dtype=torch.float32).view(3, 4))
    gate.weight_scale.data.copy_(torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16))
    up.weight_scale.data.copy_(torch.tensor([[4.0], [5.0], [6.0]], dtype=torch.bfloat16))

    fused = FusedFP8ChannelGateUp(gate, up)
    fused._ensure()

    assert fused.ready
    assert fused._weight is not None
    assert fused._weight_scale is not None
    assert fused._weight.shape == (6, 4)
    assert fused._weight_scale.shape == (6,)
    assert gate.weight.data.data_ptr() == fused._weight.data_ptr()
    assert up.weight.data.data_ptr() == fused._weight[3:].data_ptr()
    torch.testing.assert_close(fused._weight[:3], gate.weight)
    torch.testing.assert_close(fused._weight[3:], up.weight)
    torch.testing.assert_close(
        fused._weight_scale,
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def test_gate_up_fusion_rejects_bias() -> None:
    gate = CompressedTensorsFP8ChannelLinear(4, 3, bias=True)
    up = CompressedTensorsFP8ChannelLinear(4, 3)

    with pytest.raises(ValueError, match="bias-less"):
        FusedFP8ChannelGateUp(gate, up)
