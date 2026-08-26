"""Bit-exactness tests for the fused GDN causal conv+SiLU kernel.

The eager reference dispatches ``F.conv1d`` to
``conv_depthwise2d_forward_kernel_generic`` on the production (transposed,
non-contiguous) layout, which accumulates each output in FP32 in tap order
and rounds once to BF16. The fused kernel must reproduce that bit-for-bit on
every element; anything less changes greedy token streams downstream.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.kernels.gdn_conv import fused_causal_conv_silu  # noqa: E402

F = torch.nn.functional


def _reference(x: torch.Tensor, weight: torch.Tensor, padding: int) -> torch.Tensor:
    return F.silu(F.conv1d(x, weight, bias=None, padding=padding, groups=x.shape[1]))


def _prod_view(b: int, c: int, seq: int, dtype) -> torch.Tensor:
    """The production input: mixed_qkvz[..., :c].transpose(1, 2)."""
    return torch.randn(b, seq, c, dtype=dtype).transpose(1, 2)


def _cases():
    yield (
        "prefill_shape",
        _prod_view(1, 10240, 8195, torch.bfloat16),
        torch.randn(10240, 1, 4, dtype=torch.bfloat16),
    )
    yield (
        "with_state_prefix",
        torch.randn(2, 2560, 131, dtype=torch.bfloat16),
        torch.randn(2560, 1, 4, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("name,x,w", _cases(), ids=[c[0] for c in _cases()])
def test_fused_conv_silu_bit_exact(name, x, w):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dev = torch.device("cuda")
    x = x.to(dev)
    w = w.to(dev)
    ref = _reference(x, w, padding=3)
    got = fused_causal_conv_silu(x, w, padding=3)
    assert got is not None
    # The caller truncates [:input_len]; compare over that region only.
    ref = ref[:, :, : got.shape[2]]
    assert got.dtype == ref.dtype
    assert got.shape == ref.shape
    # Byte-for-byte equality against the eager pair on every element.
    assert torch.equal(got, ref), (got != ref).float().mean().item()


def test_decode_window_bit_exact():
    """The single-token decode branch: cat([state(3), x(1)]) with padding=0.

    The caller slices ``[:, :, -1:]``; the fused output must equal the eager
    pair's final column bit-for-bit.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dev = torch.device("cuda")
    for b, c in ((2, 2560), (4, 10240)):
        x = torch.randn(b, c, 4, dtype=torch.bfloat16, device=dev)
        w = ((torch.randn(c, 1, 4, device=dev)) * 0.1).to(torch.bfloat16)
        ref = _reference(x, w, padding=0)[:, :, -1:]
        got = fused_causal_conv_silu(x, w, padding=0)
        assert got is not None
        assert torch.equal(got, ref)


def test_fp32_gguf_path_not_fused():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.randn(1, 130, 64, dtype=torch.float32, device="cuda").transpose(1, 2)
    w = torch.randn(64, 1, 4, dtype=torch.float32, device="cuda")
    # FP32 (GGUF) eager conv contracts FMAs differently; fusion stays off.
    assert fused_causal_conv_silu(x, w, padding=3) is None


def test_non_divisible_channels_bit_exact():
    """C % BLOCK_C != 0 exercises the step kernel's channel mask."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dev = torch.device("cuda")
    for c in (2560, 3000, 10240):
        x = torch.randn(2, c, 4, dtype=torch.bfloat16, device=dev)
        w = ((torch.randn(c, 1, 4, device=dev)) * 0.1).to(torch.bfloat16)
        ref = _reference(x, w, padding=0)[:, :, -1:]
        got = fused_causal_conv_silu(x, w, padding=0)
        assert got is not None
        assert torch.equal(got, ref), f"c={c}"
