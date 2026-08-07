"""Part-level parity between our re-implementations and DeepSeek's reference.

Phase 1 gate: before writing the DSV4 model graph, prove our reading of the
reference semantics is exact. Each test rebuilds the official reference module
(notes/dsv4flash-ref/inference/model.py) with real GGUF weights and compares
against our independently written formulation.

The reference module is loaded via importlib under a private name: a bare
`import model` would collide with this repo's top-level model/ package inside
a shared pytest process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"
REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)


def _load_reference_model_module():
    # Load under private names WITHOUT touching sys.path: inserting the
    # reference dir would shadow this repo's top-level model/ package.
    if "kernel" not in sys.modules:
        kernel_spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(kernel_spec)
        sys.modules["kernel"] = kernel
        kernel_spec.loader.exec_module(kernel)
    spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ref_args(ref):
    return ref.ModelArgs(
        vocab_size=129280,
        dim=4096,
        moe_inter_dim=2048,
        n_layers=43,
        n_hash_layers=3,
        n_mtp_layers=3,
        n_heads=64,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=6,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        compress_ratios=tuple([0, 0] + [4, 128] * 20 + [4]),
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=(40, 41, 42),
        dspark_markov_rank=256,
    )


@pytest.fixture(scope="module")
def ref():
    if not REFERENCE_DIR.exists():
        pytest.skip("reference drop not present")
    return _load_reference_model_module()


@pytest.fixture(scope="module")
def gguf_weights():
    pytest.importorskip("numpy")
    if not REAL_GGUF.exists():
        pytest.skip("GGUF download not present")
    from runtime.loading.gguf import load_gguf_tensors

    return load_gguf_tensors(
        REAL_GGUF,
        {
            "blk.3.ffn_gate_inp.weight",
            "blk.3.exp_probs_b.bias",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_tid2eid.weight",
            "blk.0.hc_attn_fn.weight",
            "blk.0.hc_attn_base.weight",
            "blk.0.hc_attn_scale.weight",
        },
        device="cpu",
    )


def our_gate(x, weight, bias, input_ids=None, tid2eid=None, topk=6, route_scale=1.5):
    """Our reading of reference Gate.forward (model.py:552-585).

    sqrtsoftplus scores; bias shifts SELECTION only; weights come from the
    unbiased scores gathered at the chosen indices, renormalized, then scaled.
    Hash layers skip the top-k selection but NOT the gate logits.
    """
    scores = torch.nn.functional.softplus(x.float() @ weight.float().t()).sqrt()
    original = scores
    selection = scores + bias if bias is not None else scores
    if tid2eid is not None:
        indices = tid2eid[input_ids]
    else:
        indices = selection.topk(topk, dim=-1)[1]
    weights = original.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * route_scale, indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_score_gate_parity(ref, gguf_weights) -> None:
    gate = ref.Gate(layer_id=3, args=_ref_args(ref))
    weight = gguf_weights["blk.3.ffn_gate_inp.weight"].data.to(torch.bfloat16)
    bias = gguf_weights["blk.3.exp_probs_b.bias"].data.to(torch.float32)
    assert weight.shape == (256, 4096) and bias.shape == (256,)
    gate.weight = torch.nn.Parameter(weight)
    gate.bias = torch.nn.Parameter(bias)

    gen = torch.Generator().manual_seed(20260807)
    x = torch.randn(17, 4096, generator=gen) * 0.05

    ref_weights, ref_indices = gate(x, input_ids=None)
    our_weights, our_indices = our_gate(x, weight, bias)
    assert torch.equal(ref_indices, our_indices)
    assert torch.equal(ref_weights, our_weights)
    # semantic invariants of the noaux_tc contract
    assert ref_weights.shape == (17, 6)
    assert torch.allclose(ref_weights.sum(dim=-1), torch.full((17,), 1.5), atol=1e-5)
    assert ref_indices.min() >= 0 and ref_indices.max() < 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_hash_gate_parity(ref, gguf_weights) -> None:
    gate = ref.Gate(layer_id=0, args=_ref_args(ref))
    weight = gguf_weights["blk.0.ffn_gate_inp.weight"].data.to(torch.bfloat16)
    tid2eid = gguf_weights["blk.0.ffn_gate_tid2eid.weight"].data.to(torch.int64)
    assert weight.shape == (256, 4096) and tid2eid.shape == (129280, 6)
    gate.weight = torch.nn.Parameter(weight)
    gate.tid2eid = torch.nn.Parameter(tid2eid.to(torch.int32), requires_grad=False)

    gen = torch.Generator().manual_seed(20260808)
    x = torch.randn(9, 4096, generator=gen) * 0.05
    input_ids = torch.randint(0, 129280, (9,), generator=gen)

    ref_weights, ref_indices = gate(x, input_ids=input_ids)
    our_weights, our_indices = our_gate(x, weight, bias=None, input_ids=input_ids, tid2eid=tid2eid)
    assert torch.equal(ref_indices, our_indices)
    assert torch.equal(ref_weights, our_weights)
    assert torch.equal(ref_indices, tid2eid[input_ids])
    assert torch.allclose(ref_weights.sum(dim=-1), torch.full((9,), 1.5), atol=1e-5)


def our_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=4, iters=20, eps=1e-6):
    """Our reading of kernel.py hc_split_sinkhorn_kernel (lines 372-427).

    mixes layout: [pre(4) | post(4) | comb(16)]; pre adds eps after sigmoid,
    post carries the factor 2, comb starts as row-softmax + eps, then one
    column normalization, then (iters-1) rounds of row+column normalization.
    """
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(-1, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_hc_split_sinkhorn_parity(ref, gguf_weights) -> None:
    from runtime.loading.gguf import dequantize_q8_0_packed

    hc_fn = gguf_weights["blk.0.hc_attn_fn.weight"]
    hc_fn_f32 = dequantize_q8_0_packed(hc_fn.data, hc_fn.shape)
    assert hc_fn_f32.shape == (24, 16384)
    hc_base = gguf_weights["blk.0.hc_attn_base.weight"].data.to(torch.float32)
    hc_scale = gguf_weights["blk.0.hc_attn_scale.weight"].data.to(torch.float32)
    assert hc_base.shape == (24,) and hc_scale.shape == (3,)

    gen = torch.Generator(device="cuda").manual_seed(20260809)
    mixes = (torch.randn(128, 24, generator=gen, device="cuda") * 2).float()
    hc_base_c, hc_scale_c = hc_base.cuda(), hc_scale.cuda()

    ref_pre, ref_post, ref_comb = ref.hc_split_sinkhorn(
        mixes.unsqueeze(0),
        hc_scale_c,
        hc_base_c,
        hc_mult=4,
        sinkhorn_iters=20,
        eps=1e-6,
    )
    ref_pre, ref_post = ref_pre.squeeze(0), ref_post.squeeze(0)
    ref_comb = ref_comb.squeeze(0)
    our_pre, our_post, our_comb = our_hc_split_sinkhorn(mixes, hc_scale_c, hc_base_c)
    # tolerance: elementwise ops are identical; only tree-reduction order in
    # softmax/sums may differ between tilelang and torch
    assert torch.allclose(ref_pre, our_pre, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ref_post, our_post, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ref_comb, our_comb, rtol=1e-4, atol=1e-5)
    # verified semantic detail: the Sinkhorn loop ENDS on a column
    # normalization, so column sums ~= 1 while row sums drift (measured
    # 0.92..1.08 on this input) -- the matrix is NOT fully doubly stochastic.
    # Do not "fix" this to symmetric normalization; the reference relies on it.
    assert torch.allclose(our_comb.sum(dim=-2), torch.ones(128, 1, device="cuda"), rtol=1e-3)
    assert (our_comb.sum(dim=-1) > 0).all()
    assert (our_pre > 0).all() and (our_post > 0).all()
