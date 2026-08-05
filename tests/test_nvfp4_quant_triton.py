"""Bit-parity gate: Triton NVFP4 activation quantizer vs sparkinfer's own
oracle quantizer (quantize_grouped_nvfp4_torch). Both must produce the SAME
packed codes and the SAME swizzled scale bytes, or the W4A4 prefill path
silently changes arithmetic. GPU-only; self-skips without CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path  # noqa: E402

ensure_sparkinfer_path()

from sparkinfer._lib.intrinsics import (  # noqa: E402
    as_grouped_scale_view,
    quantize_grouped_nvfp4_torch,
    swizzle_block_scale,
)

from runtime.kernels.nvfp4_quant import quantize_nvfp4_activation  # noqa: E402


def _oracle_operand(x2d: torch.Tensor, gs: torch.Tensor):
    m, k = x2d.shape
    row_counts = torch.full((1,), m, dtype=torch.int32, device=x2d.device)
    packed, _sf_view = quantize_grouped_nvfp4_torch(x2d.unsqueeze(0), row_counts, gs.reshape(1))
    # rebuild the linear scale from the same recipe to compare bytes: the
    # oracle returns only a swizzled view, so recompute via its internals.
    return packed


def _triton_operand(x2d: torch.Tensor, gs: torch.Tensor):
    packed, sf_linear = quantize_nvfp4_activation(x2d, gs)
    sw = swizzle_block_scale(sf_linear.view(torch.float8_e4m3fn).unsqueeze(0))
    view = as_grouped_scale_view(sw.view(torch.uint8), x2d.shape[0], x2d.shape[1])
    return packed, sw, view


@pytest.mark.parametrize(
    "shape,gs",
    [
        ((16, 5120), 376.0),
        ((256, 5120), 376.0),
        ((4096, 5120), 376.0),
        ((333, 17408), 120.0),  # ragged rows, wide-K (down_proj shape)
        ((128, 20480), 96.5),
    ],
)
def test_bit_parity_codes_and_scales(shape, gs):
    torch.manual_seed(7)
    m, k = shape
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    # a few exact zeros and one huge outlier exercise the edge branches
    x[0, :16] = 0.0
    x[3 % m, 33] = 3000.0
    gs_t = torch.tensor(gs, dtype=torch.float32, device="cuda")

    packed_ref, _ = quantize_grouped_nvfp4_torch(
        x.unsqueeze(0),
        torch.full((1,), m, dtype=torch.int32, device="cuda"),
        gs_t.reshape(1),
    )
    # reference scale bytes: rerun the oracle recipe in torch to get linear
    # scales, then swizzle identically to the production operand builder.
    xf = x.float().view(m, k // 16, 16)
    bmax = xf.abs().amax(dim=-1)
    sf = ((gs * (bmax / 6.0)).to(torch.float8_e4m3fn)).float()
    sf_linear_ref = sf.to(torch.float8_e4m3fn)

    packed_tri, sw_tri, view_tri = _triton_operand(x, gs_t)

    assert torch.equal(packed_ref[:, :, 0], packed_tri), (
        f"packed codes differ: {(packed_ref[:, :, 0] != packed_tri).sum().item()} bytes"
    )
    # compare scale bytes: swizzle BOTH the reference (recomputed recipe,
    # unpadded rows) and the Triton output (pre-padded) -- swizzle_block_scale
    # zero-pads rows/cols itself, so both land on the same padded shape.
    sw_ref = swizzle_block_scale(sf_linear_ref.unsqueeze(0))
    assert tuple(sw_tri[0].shape) == tuple(sw_ref[0].shape), (
        tuple(sw_tri[0].shape),
        tuple(sw_ref[0].shape),
    )
    ndiff = (sw_tri[0].view(torch.uint8) != sw_ref[0].view(torch.uint8)).sum().item()
    assert ndiff == 0, f"swizzled scale bytes differ: {ndiff}"
