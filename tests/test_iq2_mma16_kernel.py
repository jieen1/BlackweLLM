"""Unit tests for the native IQ2_XS -> INT8 MMA grouped-MoE kernel adapter.

These are integration tests requiring the built ``iq2_mma16.so`` artifact and
a CUDA device; they self-skip when torch or the artifact is unavailable so
the torch-free CI job stays green.
"""
import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from loader.gguf_quant_tables import IQ2XS_GRID, KSIGNS_IQ2XS  # noqa: E402
from runtime.kernels.iq2_mma16 import IQ2MMA16Error, NativeIQ2MMA16Library  # noqa: E402
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
    return NativeIQ2MMA16Library.load()


def _check(library, E, ROWS, COLS, M_PAD, *, pattern="rand"):
    STRIDE = (COLS // IQ2) * 74
    gen = torch.Generator().manual_seed(E * 1000 + M_PAD)
    pg = _make_packed(E * ROWS, COLS, gen).cuda()
    pu = _make_packed(E * ROWS, COLS, gen).cuda()
    eids = torch.arange(E, dtype=torch.int64, device="cuda")
    x = (torch.randn(E, M_PAD, COLS, generator=gen) * 0.1).cuda()
    if pattern == "ones":
        x = torch.ones_like(x)
    xq = x.div(0.5).round().clamp(-128, 127).to(torch.int8)
    xs = xq.float().abs().reshape(E, M_PAD, COLS // 32, 32).max(dim=3).values
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")
    g, u = library.grouped_gate_up(xq, xs, pg, pu, eids, grid, ksigns,
                                   rows=ROWS, cols=COLS, stride=STRIDE, m_pad=M_PAD)
    Wg = dequantize_iq2_xs(pg).reshape(E, ROWS, COLS)
    Wu = dequantize_iq2_xs(pu).reshape(E, ROWS, COLS)
    xs_full = xs.repeat_interleave(32, dim=2)
    ref_g = torch.einsum("emk,enk->emn", xq.float() * xs_full, Wg)
    ref_u = torch.einsum("emk,enk->emn", xq.float() * xs_full, Wu)
    tol = 5e-3 * ref_g.abs().max().item() + 1e-3
    assert (g - ref_g).abs().max().item() < tol, "gate mismatch"
    assert (u - ref_u).abs().max().item() < tol, "up mismatch"


def test_matches_dequant_e1_m16(library):
    _check(library, 1, 2048, 4096, 16)


def test_matches_dequant_e2_m32(library):
    _check(library, 2, 2048, 4096, 32)


def test_matches_dequant_e3_m64(library):
    _check(library, 3, 2048, 4096, 64)


def test_matches_dequant_ones(library):
    _check(library, 1, 2048, 4096, 16, pattern="ones")


def test_rejects_noncontiguous(library):
    torch = pytest.importorskip("torch")
    xq = torch.zeros(2, 64, 4096, dtype=torch.int8, device="cuda")[:, ::2]  # non-contiguous
    xs = torch.zeros(2, 32, 128, dtype=torch.float32, device="cuda")
    pg = torch.zeros(2 * 2048 * 1184, dtype=torch.uint8, device="cuda")
    eids = torch.arange(2, dtype=torch.int64, device="cuda")
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")
    with pytest.raises(IQ2MMA16Error):
        library.grouped_gate_up(xq, xs, pg, pg, eids, grid, ksigns,
                                rows=2048, cols=4096, stride=1184, m_pad=32)


def test_grouped_moe_pipeline_matches_reference(library):
    """End-to-end grouped MoE (group -> gate/up -> SwiGLU -> down -> reduce)."""
    M, top_k, hidden, inter, n_exp = 16, 6, 4096, 2048, 16
    gen = torch.Generator().manual_seed(11)
    pg = _make_packed(n_exp * inter, hidden, gen).cuda()
    pu = _make_packed(n_exp * inter, hidden, gen).cuda()
    pd = _make_packed(n_exp * hidden, inter, gen).cuda()
    flat = (torch.randn(M, hidden, generator=gen) * 0.1).cuda()
    scores = torch.randn(M, n_exp, generator=gen).cuda()
    weights, indices = torch.softmax(scores, dim=-1).topk(top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True) * 0.5)
    indices = indices.to(torch.int64)
    grid = torch.tensor(IQ2XS_GRID, dtype=torch.int64, device="cuda")
    ksigns = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device="cuda")
    from runtime.kernels.iq2_mma16 import grouped_moe_prefill

    got = grouped_moe_prefill(flat, weights, indices, pg, pu, pd, grid, ksigns,
                              inter=inter, hidden=hidden, swiglu_limit=10.0,
                              m_pad=32, library=library)
    assert got.shape == (M, hidden)
    assert torch.isfinite(got).all().item()
    assert got.abs().max().item() > 0.0  # routed contributions are nonzero
