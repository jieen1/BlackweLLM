"""Parity test for the fused IQ2_XS dequant-GEMM kernel.

The eager MoE dequantizes a routed expert to fp32 then matmuls; this
kernel dequantizes in-register.  The parity oracle is
``dequantize_iq2_xs(packed).reshape(rows, cols) @ x.T`` in fp32 -- the
dequant is bit-exact by construction (same grid/ksigns/scale math), so
the only difference is fp32 reduction order in the GEMM (tolerance
1e-3, far above fp32 matmul reorder noise for K=4096).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from loader.gguf_quant_tables import IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS  # noqa: E402
from runtime.kernels.dsv4_iq2xs_gemm import iq2xs_dequant_gemm  # noqa: E402
from runtime.model.dsv4_quant import dequantize_iq2_xs  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _tables(device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(IQ2XS_GRID, dtype=torch.int64, device=device),
        torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device=device),
        torch.tensor(KMASK_IQ2XS, dtype=torch.int32, device=device),
    )


def _random_packed(rows: int, cols: int, device) -> torch.Tensor:
    """Deterministic VALID IQ2_XS fixture: small codes (grid-valid), scales
    0, d ~ 0.1-2.0.  Random bytes can encode out-of-range grid indices that
    the eager dequant ALSO turns into inf/NaN, which would make the parity
    comparison meaningless (both NaN, rel_err=NaN)."""
    import struct

    n_blocks = rows * cols // 256
    out = bytearray()
    g = torch.Generator().manual_seed(12345 + n_blocks)
    for _ in range(n_blocks):
        d = struct.pack("<e", float(torch.randint(100, 2000, (1,), generator=g).item() / 1000.0))
        out += d
        for _ in range(32):
            code = int(torch.randint(0, 512, (1,), generator=g).item())
            out += struct.pack("<h", code)
        out += bytes(8)  # scales 0
    return torch.frombuffer(bytes(out), dtype=torch.uint8).to(device)


@pytest.mark.parametrize("rows,cols", [(256, 512), (2048, 4096), (512, 1024)])
def test_iq2xs_gemm_matches_eager_dequant(rows: int, cols: int) -> None:
    dev = "cuda"
    packed = _random_packed(rows, cols, dev)
    tables = _tables(dev)
    x = torch.randn(1, cols, device=dev).to(torch.bfloat16)

    got = iq2xs_dequant_gemm(
        x, packed, rows=rows, cols=cols, grid_tables=tables
    )
    w = dequantize_iq2_xs(packed).reshape(rows, cols).to(torch.float32)
    expect = (x.float() @ w.t())

    max_abs = (got - expect).abs().max().item()
    rel = max_abs / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-3, f"rows={rows} cols={cols} rel_err={rel:.2e} max_abs={max_abs:.2e}"


def test_iq2xs_gemm_multi_token() -> None:
    dev = "cuda"
    rows, cols = 512, 1024
    packed = _random_packed(rows, cols, dev)
    tables = _tables(dev)
    x = torch.randn(3, cols, device=dev).to(torch.bfloat16)
    got = iq2xs_dequant_gemm(x, packed, rows=rows, cols=cols, grid_tables=tables)
    w = dequantize_iq2_xs(packed).reshape(rows, cols).to(torch.float32)
    expect = (x.float() @ w.t())
    rel = (got - expect).abs().max().item() / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-3


def test_iq2xs_gemm_block_alignment() -> None:
    """cols must be a multiple of 256 (one IQ2_XS block per 256 values)."""
    dev = "cuda"
    rows, cols = 128, 300  # not block-aligned
    packed = _random_packed(rows, 256, dev)
    with pytest.raises(Exception):
        iq2xs_dequant_gemm(
            torch.randn(1, cols, device=dev).to(torch.bfloat16),
            packed,
            rows=rows,
            cols=cols,
            grid_tables=_tables(dev),
        )
