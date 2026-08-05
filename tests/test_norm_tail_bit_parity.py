"""Bit-parity gate for the fused RMSNorm tail: must be BIT-IDENTICAL to the
torch tail (two fp32 multiplies + bf16 round) -- variance order stays in
torch, so the acceptance anchor cannot move. GPU-only; self-skips on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from runtime.kernels.fused_rms_norm import rms_norm_tail  # noqa: E402


@pytest.mark.parametrize("shape", [(16, 5120), (128, 5120), (4096, 5120), (16384, 5120)])
def test_bit_parity(shape):
    torch.manual_seed(7)
    m, n = shape
    x = (torch.randn(m, n, device="cuda") * 0.8).to(torch.bfloat16)
    w = torch.randn(n, device="cuda") * 0.1
    eps = 1e-6
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    ref = (xf * rstd * (1.0 + w)).to(torch.bfloat16)
    out = rms_norm_tail(xf, rstd.squeeze(-1), (1.0 + w).contiguous())
    assert torch.equal(ref, out), f"diff elements: {(ref != out).sum().item()} of {ref.numel()}"
