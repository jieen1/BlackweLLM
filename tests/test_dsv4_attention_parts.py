"""Parity for the functional attention parts against the official reference.

References compared against (notes/dsv4flash-ref/inference/):
- precompute_freqs_cis / apply_rotary_emb (pure torch -> bit-exact expected)
- act_quant / fp4_act_quant tilelang kernels, inplace=True (QAT simulation)
- sparse_attn tilelang kernel (online softmax + bf16 weight rounding ->
  tolerance)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.model.dsv4_attention import (  # noqa: E402
    act_quant_simulate,
    apply_rotary_emb,
    fp4_act_quant_simulate,
    hadamard_transform,
    precompute_freqs_cis,
    sparse_attention_eager,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"

NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.fixture(scope="module")
def ref():
    if not REFERENCE_DIR.exists():
        pytest.skip("reference drop not present")
    if "kernel" not in sys.modules:
        kernel_spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(kernel_spec)
        sys.modules["kernel"] = kernel
        kernel_spec.loader.exec_module(kernel)
    spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rope_freqs_bit_exact(ref) -> None:
    ours = precompute_freqs_cis(
        64,
        100,
        original_seq_len=65536,
        base=10000.0,
        factor=16.0,
        beta_fast=32,
        beta_slow=1,
    )
    theirs = ref.precompute_freqs_cis(64, 100, 65536, 10000.0, 16.0, 32, 1)
    assert torch.equal(ours, theirs)
    # and the no-YaRN branch (window-only layers)
    ours_plain = precompute_freqs_cis(
        64,
        100,
        original_seq_len=0,
        base=10000.0,
        factor=16.0,
        beta_fast=32,
        beta_slow=1,
    )
    theirs_plain = ref.precompute_freqs_cis(64, 100, 0, 10000.0, 16.0, 32, 1)
    assert torch.equal(ours_plain, theirs_plain)


def test_apply_rotary_bit_exact(ref) -> None:
    gen = torch.Generator().manual_seed(3)
    x1 = torch.randn(2, 10, 64, generator=gen)
    x2 = x1.clone()
    freqs = ref.precompute_freqs_cis(64, 10, 65536, 10000.0, 16.0, 32, 1)
    ref.apply_rotary_emb(x1, freqs)
    apply_rotary_emb(x2, freqs)
    assert torch.equal(x1, x2)
    # inverse branch (the attention output de-rotation)
    ref.apply_rotary_emb(x1, freqs, True)
    apply_rotary_emb(x2, freqs, inverse=True)
    assert torch.equal(x1, x2)


@NEEDS_GPU
@pytest.mark.parametrize("ue8m0", [True, False])
def test_act_quant_matches_reference_kernel(ref, ue8m0: bool) -> None:
    gen = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(37, 448, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.2
    ours = act_quant_simulate(x, 64, ue8m0=ue8m0)
    theirs_in = x.clone()
    theirs = ref.act_quant(
        theirs_in,
        64,
        scale_fmt="ue8m0" if ue8m0 else None,
        scale_dtype=torch.float8_e8m0fnu if ue8m0 else torch.float32,
        inplace=True,
    )
    assert torch.equal(ours, theirs)


@NEEDS_GPU
def test_fp4_act_quant_matches_reference_kernel(ref) -> None:
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(21, 128, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.3
    ours = fp4_act_quant_simulate(x, 32)
    theirs_in = x.clone()
    theirs = ref.fp4_act_quant(theirs_in, 32, inplace=True)
    assert torch.equal(ours, theirs)


@NEEDS_GPU
def test_sparse_attention_matches_reference_kernel(ref) -> None:
    b, n, h, d, m, topk = 2, 64, 16, 64, 5, 12
    gen = torch.Generator(device="cuda").manual_seed(6)
    q = torch.randn(b, m, h, d, generator=gen, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, generator=gen, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(h, generator=gen, device="cuda")
    idxs = torch.randint(-1, n, (b, m, topk), generator=gen, device="cuda", dtype=torch.int32)
    # guarantee at least one valid entry per query (kernel assumes it)
    idxs[:, :, 0] = torch.randint(0, n, (b, m), generator=gen, device="cuda", dtype=torch.int32)
    scale = d**-0.5
    ours = sparse_attention_eager(q, kv, sink, idxs, scale)
    theirs = ref.sparse_attn(q, kv, sink, idxs, scale)
    assert torch.allclose(ours, theirs, rtol=2e-3, atol=2e-4)


def test_hadamard_properties() -> None:
    x = torch.randn(3, 128)
    scale = 128**-0.5
    once = hadamard_transform(x, scale)
    twice = hadamard_transform(once, scale)
    # normalized Sylvester Hadamard is an involution
    assert torch.allclose(twice, x, rtol=1e-4, atol=1e-5)
    # norm preservation
    assert torch.allclose(once.norm(dim=-1), x.norm(dim=-1), rtol=1e-4)
