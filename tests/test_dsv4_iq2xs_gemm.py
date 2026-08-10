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
from runtime.kernels.dsv4_iq2xs_gemm import (  # noqa: E402
    iq2xs_dequant_gemm,
    iq2xs_dequant_gemm_batch_indexed,
    iq2xs_dequant_gemm_batch_indexed_dual,
    iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1,
    swiglu_bf16,
)
from runtime.model.dsv4_quant import dequantize_iq2_xs  # noqa: E402

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _random_packed_cpu(rows: int, cols: int) -> torch.Tensor:
    return _random_packed(rows, cols, "cpu")


def _tables(device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(IQ2XS_GRID, dtype=torch.int64, device=device),
        torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device=device),
        torch.tensor(KMASK_IQ2XS, dtype=torch.int32, device=device),
    )


def _random_packed(rows: int, cols: int, device, *, seed_offset: int = 0) -> torch.Tensor:
    """Deterministic valid IQ2_XS fixture with non-zero group scales.

    Codes stay in the first 512 grid-valid entries and d is ~0.1-2.0. Random
    bytes can encode invalid grid indices that also make eager dequant return
    inf/NaN, rendering parity meaningless.
    """
    import struct

    n_blocks = rows * cols // 256
    out = bytearray()
    g = torch.Generator().manual_seed(12345 + n_blocks + seed_offset)
    for _ in range(n_blocks):
        d = struct.pack("<e", float(torch.randint(100, 2000, (1,), generator=g).item() / 1000.0))
        out += d
        for _ in range(32):
            code = int(torch.randint(0, 512, (1,), generator=g).item())
            out += struct.pack("<h", code)
        # Non-zero scales are essential: an all-zero fixture cannot detect
        # swapping the code-group and within-code dimensions in the kernel.
        out += bytes(int(v) for v in torch.randint(0, 256, (8,), generator=g, dtype=torch.int64))
    return torch.frombuffer(bytes(out), dtype=torch.uint8).to(device)


@CUDA_REQUIRED
@pytest.mark.parametrize("rows,cols", [(256, 512), (2048, 4096), (512, 1024)])
def test_iq2xs_gemm_matches_eager_dequant(rows: int, cols: int) -> None:
    dev = "cuda"
    packed = _random_packed(rows, cols, dev)
    tables = _tables(dev)
    x = torch.randn(1, cols, device=dev).to(torch.bfloat16)

    got = iq2xs_dequant_gemm(x, packed, rows=rows, cols=cols, grid_tables=tables)
    w = dequantize_iq2_xs(packed).reshape(rows, cols).to(torch.float32)
    expect = x.float() @ w.t()

    max_abs = (got - expect).abs().max().item()
    rel = max_abs / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-3, f"rows={rows} cols={cols} rel_err={rel:.2e} max_abs={max_abs:.2e}"


@CUDA_REQUIRED
def test_iq2xs_gemm_multi_token() -> None:
    dev = "cuda"
    rows, cols = 512, 1024
    packed = _random_packed(rows, cols, dev)
    tables = _tables(dev)
    x = torch.randn(3, cols, device=dev).to(torch.bfloat16)
    got = iq2xs_dequant_gemm(x, packed, rows=rows, cols=cols, grid_tables=tables)
    w = dequantize_iq2_xs(packed).reshape(rows, cols).to(torch.float32)
    expect = x.float() @ w.t()
    rel = (got - expect).abs().max().item() / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-3


@CUDA_REQUIRED
def test_iq2xs_gemm_batch_indexed_matches_eager() -> None:
    dev = "cuda"
    rows, cols = 512, 1024
    num_experts = 5
    packed_chunks = [_random_packed(rows, cols, dev) for _ in range(num_experts)]
    packed_all = torch.cat(packed_chunks, dim=0)
    expert_ids = torch.tensor([4, 1, 4, 0], device=dev, dtype=torch.int64)
    x = torch.randn(expert_ids.numel(), 2, cols, device=dev).to(torch.bfloat16)
    tables = _tables(dev)

    got = iq2xs_dequant_gemm_batch_indexed(
        x,
        packed_all,
        expert_ids,
        rows=rows,
        cols=cols,
        grid_tables=tables,
    )

    expect_rows = []
    expert_stride = rows * (cols // 256) * 74
    for i, eid in enumerate(expert_ids.tolist()):
        start = eid * expert_stride
        end = start + expert_stride
        w = dequantize_iq2_xs(packed_all[start:end]).reshape(rows, cols).to(torch.float32)
        expect_rows.append(x[i].float() @ w.t())
    expect = torch.stack(expect_rows, dim=0)

    max_abs = (got - expect).abs().max().item()
    rel = max_abs / (expect.abs().max().item() + 1e-9)
    assert rel < 1e-3, f"indexed rel_err={rel:.2e} max_abs={max_abs:.2e}"


@CUDA_REQUIRED
def test_iq2xs_gemm_batch_indexed_dual_matches_two_single_calls_exactly() -> None:
    dev = "cuda"
    rows, cols = 512, 1024
    num_experts = 6
    packed_gate_all = torch.cat(
        [_random_packed(rows, cols, dev) for _ in range(num_experts)],
        dim=0,
    )
    packed_up_all = torch.cat(
        [_random_packed(rows, cols, dev) for _ in range(num_experts)],
        dim=0,
    )
    expert_ids = torch.tensor([5, 2, 5, 1], device=dev, dtype=torch.int64)
    x = torch.randn(expert_ids.numel(), 3, cols, device=dev).to(torch.bfloat16)
    tables = _tables(dev)

    gate_got, up_got = iq2xs_dequant_gemm_batch_indexed_dual(
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        rows=rows,
        cols=cols,
        grid_tables=tables,
    )
    gate_expect = iq2xs_dequant_gemm_batch_indexed(
        x,
        packed_gate_all,
        expert_ids,
        rows=rows,
        cols=cols,
        grid_tables=tables,
    )
    up_expect = iq2xs_dequant_gemm_batch_indexed(
        x,
        packed_up_all,
        expert_ids,
        rows=rows,
        cols=cols,
        grid_tables=tables,
    )

    assert torch.equal(gate_got, gate_expect)
    assert torch.equal(up_got, up_expect)


@CUDA_REQUIRED
def test_iq2xs_gemm_batch_indexed_dual_swiglu_b1_matches_split_exactly() -> None:
    dev = "cuda"
    rows, cols = 512, 1024
    num_experts = 6
    packed_gate_all = torch.cat(
        [_random_packed(rows, cols, dev, seed_offset=100 + i) for i in range(num_experts)],
        dim=0,
    )
    packed_up_all = torch.cat(
        [_random_packed(rows, cols, dev, seed_offset=200 + i) for i in range(num_experts)],
        dim=0,
    )
    expert_ids = torch.tensor([5, 2, 5, 1], device=dev, dtype=torch.int64)
    generator = torch.Generator(device=dev).manual_seed(20260903)
    x = torch.randn(
        expert_ids.numel(), 1, cols, generator=generator, device=dev, dtype=torch.bfloat16
    )
    tables = _tables(dev)
    limit = 10.0

    got = iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1(
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        rows=rows,
        cols=cols,
        grid_tables=tables,
        limit=limit,
    )
    gate, up = iq2xs_dequant_gemm_batch_indexed_dual(
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        rows=rows,
        cols=cols,
        grid_tables=tables,
    )
    expected = swiglu_bf16(gate, up, limit)

    assert got.shape == (expert_ids.numel(), 1, rows)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


@CUDA_REQUIRED
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


def test_iq2xs_gemm_batch_indexed_dual_requires_cuda_x() -> None:
    rows, cols = 256, 512
    x = torch.randn(2, 1, cols)
    packed = _random_packed_cpu(rows, cols)
    expert_ids = torch.tensor([0, 0], dtype=torch.int64)

    with pytest.raises(ValueError, match="require x on CUDA"):
        iq2xs_dequant_gemm_batch_indexed_dual(
            x,
            packed,
            packed,
            expert_ids,
            rows=rows,
            cols=cols,
            grid_tables=_tables("cpu"),
        )


def test_iq2xs_gemm_batch_indexed_dual_rejects_misaligned_rows() -> None:
    rows, cols = 10, 512
    x = torch.randn(1, 1, cols)
    packed = _random_packed_cpu(16, cols)
    expert_ids = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(ValueError, match="rows 10 must be divisible by BLOCK_COLS 8"):
        iq2xs_dequant_gemm_batch_indexed_dual(
            x,
            packed,
            packed,
            expert_ids,
            rows=rows,
            cols=cols,
            grid_tables=_tables("cpu"),
        )


def test_iq2xs_gemm_batch_indexed_dual_swiglu_b1_rejects_multi_token() -> None:
    rows, cols = 256, 512
    x = torch.randn(2, 2, cols)
    packed = _random_packed_cpu(rows, cols)
    expert_ids = torch.tensor([0, 0], dtype=torch.int64)

    with pytest.raises(ValueError, match="requires exactly one token"):
        iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1(
            x,
            packed,
            packed,
            expert_ids,
            rows=rows,
            cols=cols,
            grid_tables=_tables("cpu"),
            limit=10.0,
        )


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_iq2xs_gemm_batch_indexed_dual_swiglu_b1_rejects_invalid_limit(limit: float) -> None:
    rows, cols = 256, 512
    x = torch.randn(2, 1, cols)
    packed = _random_packed_cpu(rows, cols)
    expert_ids = torch.tensor([0, 0], dtype=torch.int64)

    with pytest.raises(ValueError, match="limit must be finite and > 0"):
        iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1(
            x,
            packed,
            packed,
            expert_ids,
            rows=rows,
            cols=cols,
            grid_tables=_tables("cpu"),
            limit=limit,
        )


@CUDA_REQUIRED
def test_swiglu_bf16_matches_decode_chain_exactly() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260902)
    gate = torch.randn(6, 1, 2048, generator=generator, device="cuda") * 8
    up = torch.randn(6, 1, 2048, generator=generator, device="cuda") * 8
    limit = 7.0
    expected = (
        torch.nn.functional.silu(torch.clamp(gate, max=limit))
        * torch.clamp(up, min=-limit, max=limit)
    ).to(torch.bfloat16)
    got = swiglu_bf16(gate, up, limit)
    max_abs = (got.float() - expected.float()).abs().max().item()
    assert max_abs <= 5e-4, max_abs
