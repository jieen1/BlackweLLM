"""Parity test for the fused Q8_0 dequant-GEMM kernel.

Oracle: dequantize_q8_0(packed).reshape(out, in) @ x.T in fp32.  The
dequant is bit-exact; only fp32 reduction order differs (tolerance 1e-3).
"""

from __future__ import annotations

import struct

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from runtime.kernels.dsv4_q8_gemm import (  # noqa: E402
    _select_q8_0_block_m,
    _select_q8_0_block_n,
    _select_q8_0_grouped_block_m,
    _select_q8_0_grouped_block_n,
    q8_0_dequant_gemm,
    q8_0_dequant_gemv_fp32,
    q8_0_grouped_dequant_gemm,
)
from runtime.model.dsv4_model import PackedQ8_0Linear  # noqa: E402
from runtime.model.dsv4_quant import dequantize_q8_0  # noqa: E402

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.mark.parametrize(
    "rows,inp,out,expected_m,expected_n",
    [
        (1, 4096, 512, 8, 16),
        (4, 8192, 4096, 8, 16),
        (5, 4096, 512, 16, 32),
        (32, 8192, 4096, 16, 32),
        (1, 4096, 1024, 8, 8),
        (1, 1024, 32768, 16, 64),
        (4, 4096, 129280, 16, 32),
        (32, 4096, 129280, 16, 64),
    ],
)
def test_q8_decode_projection_tile_selection(
    rows: int,
    inp: int,
    out: int,
    expected_m: int,
    expected_n: int,
) -> None:
    assert _select_q8_0_block_m(rows, inp, out) == expected_m
    assert _select_q8_0_block_n(rows, inp, out) == expected_n


@pytest.mark.parametrize(
    "rows_per_group,expected_m,expected_n",
    [(1, 8, 16), (2, 8, 16), (4, 8, 16), (5, 16, 64), (32, 16, 64)],
)
def test_q8_grouped_projection_tile_selection(
    rows_per_group: int,
    expected_m: int,
    expected_n: int,
) -> None:
    assert _select_q8_0_grouped_block_m(rows_per_group) == expected_m
    assert _select_q8_0_grouped_block_n(rows_per_group) == expected_n


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("rows", "dtype"),
    [(5, torch.bfloat16), (1, torch.float32)],
    ids=["prefill_rows", "fp32_input"],
)
def test_fused_q8_fp32_is_decode_bf16_only(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    dtype: torch.dtype,
) -> None:
    linear = PackedQ8_0Linear(64, 64, device="cuda")
    linear.packed.zero_()
    linear.fused_q8_fp32 = True

    def fail_if_called(*args, **kwargs):
        raise AssertionError("decode-only packed Q8 kernel must not run")

    monkeypatch.setattr(
        "runtime.kernels.dsv4_q8_gemm.q8_0_dequant_gemv_fp32",
        fail_if_called,
    )
    out = linear(torch.zeros((rows, 64), device="cuda", dtype=dtype))

    assert out.shape == (rows, 64)
    assert out.dtype == torch.float32
    assert torch.count_nonzero(out).item() == 0


def _random_packed(out: int, inp: int, device) -> torch.Tensor:
    n_blocks = out * inp // 32
    blob = bytearray()
    g = torch.Generator().manual_seed(11 + out)
    for _ in range(n_blocks):
        blob += struct.pack("<e", float(torch.randint(1, 20, (1,), generator=g).item()))
        for _ in range(32):
            blob += struct.pack("<b", int(torch.randint(-20, 21, (1,), generator=g).item()))
    return torch.frombuffer(bytes(blob), dtype=torch.uint8).to(device)


@CUDA_REQUIRED
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


@CUDA_REQUIRED
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


@CUDA_REQUIRED
@pytest.mark.parametrize("out,inp", [(1024, 512), (1030, 512), (4096, 2048)])
def test_q8_gemv_fp32_matches_eager(out: int, inp: int) -> None:
    dev = "cuda"
    packed = _random_packed(out, inp, dev)
    x = torch.randn(1, inp, device=dev).to(torch.bfloat16)
    got = q8_0_dequant_gemv_fp32(x, packed, out_features=out, in_features=inp)
    weight = dequantize_q8_0(packed).reshape(out, inp)
    expect = x.float() @ weight.t()
    rel = (got - expect).abs().max().item() / (expect.abs().max().item() + 1e-9)
    assert rel < 2e-5, f"out={out} inp={inp} rel={rel:.2e}"


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "groups,gsize,inp,rows_per_g",
    [
        (8, 1024, 4096, 1),
        (4, 256, 512, 2),
        (2, 64, 64, 3),
        (2, 64, 64, 5),
        (2, 64, 64, 9),
        (2, 64, 64, 17),
        (2, 64, 64, 32),
        (2, 70, 64, 3),
    ],
)
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
