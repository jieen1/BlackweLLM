"""Unit tests for the scale-amortized IQ2 MMA16 TC kernel (Phase 2B-0).

These are integration tests requiring the built ``iq2_mma16_tc.so`` artifact
and a CUDA device; they self-skip when torch or the artifact is unavailable.
"""
import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from loader.gguf_quant_tables import IQ2XS_GRID, KSIGNS_IQ2XS  # noqa: E402
from runtime.kernels.iq2_mma16_tc import (  # noqa: E402
    _MANIFEST_PATH,
    IQ2MMA16TCError,
    NativeIQ2MMA16TCLibrary,
)
from runtime.model.dsv4_quant import dequantize_iq2_xs  # noqa: E402

IQ2 = 256


def _make_packed(rows: int, cols: int, generator) -> torch.Tensor:
    n = rows * (cols // IQ2)
    p = torch.zeros(n * 74, dtype=torch.uint8)
    d = (torch.rand(n, generator=generator) * 2 - 1).to(torch.float16)
    c = torch.randint(0, 16384, (n, 32), dtype=torch.int32, generator=generator)
    s = torch.randint(0, 256, (n, 8), dtype=torch.uint8, generator=generator)
    b = p.view(n, 74)
    b[:, :2] = d.view(torch.uint8).reshape(n, 2)
    b[:, 2:66] = c.to(torch.int16).view(torch.uint8).reshape(n, 64)
    b[:, 66:74] = s
    return p


@pytest.fixture(scope="module")
def library():
    return NativeIQ2MMA16TCLibrary.load()


def test_stale_artifact_guard():
    """load() must reject a manifest whose source_sha256 differs from the .cu."""
    kernel_dir = Path(__file__).resolve().parent.parent / "runtime" / "kernels"
    source = kernel_dir / "iq2_mma16_tc.cu"
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    if "source_sha256" not in manifest:
        pytest.skip("manifest has no source_sha256")
    current = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest["source_sha256"] != current:
        # artifact is genuinely stale; load must fail loudly
        with pytest.raises(IQ2MMA16TCError):
            NativeIQ2MMA16TCLibrary.load()
    else:
        # artifact matches source; load must succeed and be loadable
        lib = NativeIQ2MMA16TCLibrary.load()
        assert lib is not None


def test_matches_dequant_e2_m32(library):
    """K-group folding must stay within cos>=0.99 of the exact oracle."""
    E, ROWS, COLS, M_PAD = 2, 2048, 4096, 32
    STRIDE = (COLS // IQ2) * 74
    gen = torch.Generator().manual_seed(77)
    pg = _make_packed(E * ROWS, COLS, gen).cuda()
    pu = _make_packed(E * ROWS, COLS, gen).cuda()
    eids = torch.arange(E, dtype=torch.int64, device="cuda")
    x = (torch.randn(E, M_PAD, COLS, generator=gen) * 0.1).cuda()
    xr = x.reshape(E, M_PAD, COLS // 32, 32)
    xs = (xr.abs().max(-1, keepdim=True).values / 127.0).clamp(min=1e-8)
    xq = (xr / xs).round().clamp(-128, 127).to(torch.int8).reshape(E, M_PAD, COLS)
    xs = xs.reshape(E, M_PAD, COLS // 32)
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")
    g, u = library.grouped_gate_up(xq, xs, pg, pu, eids, grid, ksigns,
                                   rows=ROWS, cols=COLS, stride=STRIDE, m_pad=M_PAD)
    Wg = dequantize_iq2_xs(pg).reshape(E, ROWS, COLS)
    Wu = dequantize_iq2_xs(pu).reshape(E, ROWS, COLS)
    xdec = xq.float() * xs.repeat_interleave(32, dim=-1)
    ref_g = torch.einsum("emk,enk->emn", xdec, Wg)
    ref_u = torch.einsum("emk,enk->emn", xdec, Wu)
    cos_g = (g * ref_g).sum() / (g.norm() * ref_g.norm() + 1e-9)
    cos_u = (u * ref_u).sum() / (u.norm() * ref_u.norm() + 1e-9)
    assert cos_g.item() >= 0.99, f"gate cos {cos_g.item()} < 0.99"
    assert cos_u.item() >= 0.99, f"up cos {cos_u.item()} < 0.99"


def test_rejects_noncontiguous(library):
    xq = torch.zeros(2, 64, 4096, dtype=torch.int8, device="cuda")[:, ::2]
    xs = torch.zeros(2, 32, 128, dtype=torch.float32, device="cuda")
    pg = torch.zeros(2 * 2048 * 1184, dtype=torch.uint8, device="cuda")
    eids = torch.arange(2, dtype=torch.int64, device="cuda")
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")
    with pytest.raises(IQ2MMA16TCError):
        library.grouped_gate_up(xq, xs, pg, pg, eids, grid, ksigns,
                                rows=2048, cols=4096, stride=1184, m_pad=32)


def test_single_down_matches_dual_first(library):
    """single_down output must equal grouped_gate_up's first output."""
    E, ROWS, COLS, M_PAD = 1, 2048, 4096, 32
    STRIDE = (COLS // IQ2) * 74
    gen = torch.Generator().manual_seed(88)
    pg = _make_packed(E * ROWS, COLS, gen).cuda()
    eids = torch.arange(E, dtype=torch.int64, device="cuda")
    x = (torch.randn(E, M_PAD, COLS, generator=gen) * 0.1).cuda()
    xr = x.reshape(E, M_PAD, COLS // 32, 32)
    xs = (xr.abs().max(-1, keepdim=True).values / 127.0).clamp(min=1e-8)
    xq = (xr / xs).round().clamp(-128, 127).to(torch.int8).reshape(E, M_PAD, COLS)
    xs = xs.reshape(E, M_PAD, COLS // 32)
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")
    # dual: gate=up=pg, take first output
    g, _ = library.grouped_gate_up(xq, xs, pg, pg, eids, grid, ksigns,
                                   rows=ROWS, cols=COLS, stride=STRIDE, m_pad=M_PAD)
    # single: same packed
    d = library.single_down(xq, xs, pg, eids, grid, ksigns,
                            rows=ROWS, cols=COLS, stride=STRIDE, m_pad=M_PAD)
    cos = (g * d).sum() / (g.norm() * d.norm() + 1e-9)
    assert cos.item() >= 0.9999, f"single vs dual first cos {cos.item()} < 0.9999"


def test_graph_complete_moe_matches_eager(library):
    """The graph-capturable complete K32 MoE must equal the eager split-batch
    path bit-exactly on the same 64-row chunk (real DSV4 route distribution:
    max routes/expert well under the graph's fixed bucket=64)."""
    from runtime.kernels.iq2_mma16_tc import (
        Dsv4PrefillMoEWorkspace,
        grouped_moe_prefill_k32,
        grouped_moe_prefill_k32_graph,
    )

    E, H, INTER, M, TOPK = 64, 256, 256, 64, 2
    gen = torch.Generator().manual_seed(7)
    # gate/up per-expert weights are [INTER, H]; down per-expert is [H, INTER]
    pg = _make_packed(E * INTER, H, gen).cuda()
    pu = _make_packed(E * INTER, H, gen).cuda()
    pd = _make_packed(E * H, INTER, gen).cuda()
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")

    gen = torch.Generator(device="cuda").manual_seed(1)
    scores = torch.rand(M, E, generator=gen, device="cuda")
    indices = scores.topk(TOPK, dim=-1)[1]
    weights = torch.softmax(scores, dim=-1).gather(1, indices)
    flat = (torch.randn(M, H, generator=gen, device="cuda") * 0.1).bfloat16()
    assert torch.bincount(indices.reshape(-1), minlength=E).max().item() <= 64

    eager = grouped_moe_prefill_k32(
        flat, weights, indices, pg, pu, pd, grid, ksigns,
        inter=INTER, hidden=H, swiglu_limit=10.0, bucket=32, library=library,
    )
    ws = Dsv4PrefillMoEWorkspace(
        device="cuda", hidden=H, inter=INTER, m=M, top_k=TOPK, n_experts=E, bucket=64
    )
    ws.flat.copy_(flat)
    ws.indices.copy_(indices)
    ws.weights.copy_(weights)
    graph = grouped_moe_prefill_k32_graph(
        ws, flat, indices, weights, pg, pu, pd, grid, ksigns,
        inter=INTER, hidden=H, swiglu_limit=10.0, library=library,
    )
    cos = (eager * graph).sum() / (eager.norm() * graph.norm() + 1e-9)
    assert cos.item() == 1.0, f"graph vs eager cos {cos.item()}"
    assert torch.equal(eager, graph)


def test_dynamic_moe_tiles_expert_routes_over_64(library):
    """The compact path must write every route when one expert exceeds a tile.

    Expert 245 receives 81 routes after routes for lower-numbered experts,
    matching both the non-zero compact offset and the overflow shape from
    production.  Six routes per token also exercises the top-k=6 grouping
    specialization used by DSV4.
    The split-batch implementation is the established bit-exact reference for
    the same K32 gate/up and down kernels.
    """
    from runtime.kernels.iq2_mma16_tc import (
        DynamicMoEWorkspace,
        grouped_moe_prefill_k32,
        grouped_moe_prefill_k32_dynamic,
    )

    E, H, INTER, M = 256, 256, 256, 89
    gen = torch.Generator().manual_seed(81)
    pg = _make_packed(E * INTER, H, gen).cuda()
    pu = _make_packed(E * INTER, H, gen).cuda()
    pd = _make_packed(E * H, INTER, gen).cuda()
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")

    cuda_gen = torch.Generator(device="cuda").manual_seed(81)
    flat = (torch.randn(M, H, generator=cuda_gen, device="cuda") * 0.1).bfloat16()
    indices = torch.arange(6, dtype=torch.int64, device="cuda").repeat(M, 1)
    indices[8:, 0] = 245
    weights = torch.full((M, 6), 1.0 / 6.0, dtype=torch.float32, device="cuda")

    reference = grouped_moe_prefill_k32(
        flat,
        weights,
        indices,
        pg,
        pu,
        pd,
        grid,
        ksigns,
        inter=INTER,
        hidden=H,
        swiglu_limit=10.0,
        bucket=64,
        library=library,
    )
    workspace = DynamicMoEWorkspace.create(M, 6, H, INTER, flat.device)
    dynamic = grouped_moe_prefill_k32_dynamic(
        flat,
        weights,
        indices,
        pg,
        pu,
        pd,
        grid,
        ksigns,
        inter=INTER,
        hidden=H,
        swiglu_limit=10.0,
        library=library,
        workspace=workspace,
    )

    assert torch.isfinite(dynamic).all()
    assert torch.equal(reference, dynamic)


def test_dynamic_moe_chunks_bit_exact(library):
    """Chunking a large M over a small workspace must match the single shot.

    The chunked path (workspace.m < M) processes consecutive row slices with
    the same grouping/expert-GEMM/combine kernels; every step is row-local, so
    the result must be bit-exact with one workspace-sized-for-M call.  This
    guards the bounded-workspace prefill path used to fit a 128K-token DSV4
    prompt on a 96 GiB card (a full-M dynamic workspace is ~31 GiB at M=131072).
    """
    from runtime.kernels.iq2_mma16_tc import (
        DynamicMoEWorkspace,
        grouped_moe_prefill_k32_dynamic,
    )

    E, H, INTER, M = 256, 256, 256, 267
    gen = torch.Generator().manual_seed(82)
    pg = _make_packed(E * INTER, H, gen).cuda()
    pu = _make_packed(E * INTER, H, gen).cuda()
    pd = _make_packed(E * H, INTER, gen).cuda()
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")

    cuda_gen = torch.Generator(device="cuda").manual_seed(82)
    flat = (torch.randn(M, H, generator=cuda_gen, device="cuda") * 0.1).bfloat16()
    scores = torch.rand(M, E, generator=cuda_gen, device="cuda")
    indices = scores.topk(6, dim=-1)[1]
    weights = torch.softmax(scores, dim=-1).gather(1, indices)

    full_ws = DynamicMoEWorkspace.create(M, 6, H, INTER, flat.device)
    single = grouped_moe_prefill_k32_dynamic(
        flat, weights, indices, pg, pu, pd, grid, ksigns,
        inter=INTER, hidden=H, swiglu_limit=10.0,
        library=library, workspace=full_ws,
    )
    chunk_ws = DynamicMoEWorkspace.create(89, 6, H, INTER, flat.device)
    chunked = grouped_moe_prefill_k32_dynamic(
        flat, weights, indices, pg, pu, pd, grid, ksigns,
        inter=INTER, hidden=H, swiglu_limit=10.0,
        library=library, workspace=chunk_ws,
    )

    assert torch.isfinite(chunked).all()
    assert torch.equal(single, chunked), (
        "chunked dynamic MoE diverges from the single shot"
    )
