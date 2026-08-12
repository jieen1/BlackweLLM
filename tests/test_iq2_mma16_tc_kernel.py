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
