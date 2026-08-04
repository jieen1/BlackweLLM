"""Bit-parity gate: the fused Triton per-token E4M3 quantizer must be
BIT-IDENTICAL to the pure-torch chain it replaces in
``compressed_tensors_linear._quantize_fp8_activation_for_torch_scaled_mm``
-- any drift moves the greedy acceptance anchor. GPU-only; self-skips
without CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from runtime.kernels.fp8_per_token_quant import fp8_per_token_quantize  # noqa: E402


def _torch_chain(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x32 = x_2d.to(torch.float32)
    amax = x32.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 448.0).clamp_min(1.0 / (448.0 * 512.0))
    x_fp8 = (x32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale


@pytest.mark.parametrize(
    "shape",
    [(16, 5120), (16, 20480), (128, 5120), (4096, 5120), (16384, 5120), (4096, 6144)],
)
def test_bit_parity_codes_and_scales(shape):
    torch.manual_seed(5)
    m, k = shape
    x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.7).to(
        torch.bfloat16
    )
    # exact zeros, a saturating outlier, and a sub-scale-floor row exercise
    # the subnormal-midpoint rounding decisions that div-mode bugs flip.
    x[0, :32] = 0.0
    x[1 % m, 7] = 3000.0
    x[2 % m, :] = x[2 % m] * 1e-6
    ref_c, ref_s = _torch_chain(x)
    tri_c, tri_s = fp8_per_token_quantize(x)
    assert torch.equal(ref_c.view(torch.uint8), tri_c.view(torch.uint8)), (
        "codes differ: "
        f"{(ref_c.view(torch.uint8) != tri_c.view(torch.uint8)).sum().item()}"
    )
    assert torch.equal(ref_s, tri_s), "scales differ"
