"""Bit-parity gate for the fused GDN gated-norm tail: must be BIT-IDENTICAL
to the torch chain (variance order stays in torch; silu uses the div form
measured bit-equal to torch silu). GPU-only; self-skips without CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from runtime.kernels.fused_rms_norm import gated_norm_tail  # noqa: E402


def _torch_chain(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = w * x.to(input_dtype)
    x = x * torch.nn.functional.silu(gate.to(torch.float32))
    return x.to(input_dtype)


@pytest.mark.parametrize("shape", [(16, 5120), (128, 5120), (4096, 5120), (16384, 5120)])
def test_bit_parity(shape):
    torch.manual_seed(9)
    m, n = shape
    x = (torch.randn(m, n, device="cuda") * 0.6).to(torch.bfloat16)
    gate = (torch.randn(m, n, device="cuda") * 1.5).to(torch.bfloat16)
    w = torch.randn(n, device="cuda") * 0.2
    eps = 1e-6
    ref = _torch_chain(x, gate, w, eps)
    xf = x.to(torch.float32)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = gated_norm_tail(xf, rstd.reshape(-1), w, gate)
    assert torch.equal(ref, out), (
        f"diff elements: {(ref != out).sum().item()} of {ref.numel()}"
    )
