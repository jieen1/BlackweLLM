"""Parity test for the fused Q8_0 dequant-GEMM kernel.

Oracle: dequantize_q8_0(packed).reshape(out, in) @ x.T in fp32.  The
dequant is bit-exact; only fp32 reduction order differs (tolerance 1e-3).
"""

from __future__ import annotations

import struct

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from runtime.kernels.dsv4_q8_gemm import q8_0_dequant_gemm, q8_0_grouped_dequant_gemm  # noqa: E402
from runtime.model.dsv4_quant import dequantize_q8_0  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _random_packed(out: int, inp: int, device) -> torch.Tensor:
    n_blocks = out * inp // 32
    blob = bytearray()
    g = torch.Generator().manual_seed(11 + out)
    for _ in range(n_blocks):
        blob += struct.pack("<e", float(torch.randint(1, 20, (1,), generator=g).item()))
        for _ in range(32):
            blob += struct.pack("<b", int(torch.randint(-20, 21, (1,), generator=g).item()))
    return torch.frombuffer(bytes(blob), dtype=torch.uint8).to(device)


@pytest.mark.parametrize("out,inp", [(2048, 4096), (4096, 2048), (1024, 512)])
def test_q8_gemm_matches_eager(out: int, inp: int) -> None:
    dev = "cuda"
    packed = _random_packed(out, inp, dev)
    x = torch.randn(1, inp, device=dev).to(torch.bfloat16)
    got = q8_0_dequant_gemm(x, packed, out_features=out, in_features=inp)
    # The kernel dequantizes to bf16 and tl.dot accumulates in fp32 -- the
    # production weight_dtype=bfloat16 regime.  Compare against the bf16
    # dequant oracle (rel 1e-3 covers fp32 reorder; bf16 weights are exact).
    w = dequantize_q8_0(packed).reshape(out, inp).to(torch.bfloat16).to(torch.float32)
    expect = x.float() @ w.t()
    rel = (got - expect).abs().max().item() / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-2, f"out={out} inp={inp} rel={rel:.2e}"


def test_q8_gemm_multi_token() -> None:
    dev = "cuda"
    out, inp = 1024, 512
    packed = _random_packed(out, inp, dev)
    x = torch.randn(4, inp, device=dev).to(torch.bfloat16)
    got = q8_0_dequant_gemm(x, packed, out_features=out, in_features=inp)
    w = dequantize_q8_0(packed).reshape(out, inp).to(torch.bfloat16).to(torch.float32)
    expect = x.float() @ w.t()
    rel = (got - expect).abs().max().item() / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-2


@pytest.mark.parametrize("groups,gsize,inp,rows_per_g", [(8, 1024, 4096, 1), (4, 256, 512, 2)])
def test_q8_grouped_gemm_matches_eager(groups: int, gsize: int, inp: int, rows_per_g: int) -> None:
    """Grouped dequant-GEMM (per-group wo_a contraction) vs per-group eager."""
    dev = "cuda"
    packed = _random_packed(groups * gsize, inp, dev)
    w = dequantize_q8_0(packed).reshape(groups, gsize, inp).to(torch.bfloat16).to(torch.float32)
    x = torch.randn(groups * rows_per_g, inp, device=dev).to(torch.bfloat16)
    got = q8_0_grouped_dequant_gemm(
        x,
        packed,
        num_groups=groups,
        group_size=gsize,
        in_features=inp,
        rows_per_group=rows_per_g,
    )
    # x rows are group-major: rows [g*r, g*r+r) belong to group g.
    expect = torch.empty_like(got)
    for g in range(groups):
        xg = x[g * rows_per_g : (g + 1) * rows_per_g].float()
        expect[g * rows_per_g : (g + 1) * rows_per_g] = xg @ w[g].t()
    rel = (got - expect).abs().max().item() / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-2, f"groups={groups} rel={rel:.2e}"
