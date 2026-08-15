"""Dsv4Attention parity vs the reference Attention module on real weights.

Covers all three layer flavors: blk.2 (ratio-4 CSA: window + compressor +
indexer), blk.3 (ratio-128 HCA: window + compressor + all-compressed
gather), blk.0 (ratio-0: window only, plain rope). Window/compressed cache
writes must be bit-exact; the attention output carries the sparse-kernel
tolerance (bf16 weight rounding + online-vs-two-pass softmax).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

from loader.gguf_header import read_gguf_header  # noqa: E402
from runtime.loading.gguf import load_gguf_tensors  # noqa: E402
from runtime.model.dsv4_config import config_from_gguf_kv  # noqa: E402
from runtime.model.dsv4_model import Dsv4Attention  # noqa: E402
from runtime.model.dsv4_quant import dequantize_q8_0  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"
REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
if not REAL_GGUF.exists():
    pytest.skip("GGUF download not present", allow_module_level=True)
if not REFERENCE_DIR.exists():
    pytest.skip("reference drop not present", allow_module_level=True)

MAX_SEQ = 4096


def _load_ref():
    if "fast_hadamard_transform" not in sys.modules:
        import types

        from runtime.model.dsv4_attention import hadamard_transform as ours

        shim = types.ModuleType("fast_hadamard_transform")
        shim.hadamard_transform = lambda x, scale=1.0: ours(x, scale)
        sys.modules["fast_hadamard_transform"] = shim
    if "kernel" not in sys.modules:
        spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(spec)
        sys.modules["kernel"] = kernel
        spec.loader.exec_module(kernel)
    spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.scale_fmt = "ue8m0"
    module.scale_dtype = torch.float8_e8m0fnu
    # The reference sparse_attn tilelang kernel needs ~141 KB dynamic shared
    # memory at production shape (h=64, d=512, block=64): 64 KB q_shared +
    # 64 KB kv_shared + 8 KB acc_s_cast + margin. SM120's opt-in cap is
    # 99 KB (101376 B), so the kernel cannot run here at all -- it targets
    # datacenter Blackwell (228 KB smem). Substitute the semantics-verified
    # eager equivalent for the aggregation step (kernel semantics themselves
    # are parity-proven at small shape in test_dsv4_attention_parts.py);
    # everything else in the reference Attention stays reference code.
    from runtime.model.dsv4_attention import sparse_attention_eager

    module.sparse_attn = lambda q, kv, sink, idxs, scale: sparse_attention_eager(
        q, kv, sink, idxs, scale
    )
    return module


def _ref_args(ref):
    return ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=MAX_SEQ,
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
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype=None,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=(40, 41, 42),
        dspark_markov_rank=256,
    )


def _build_pair(layer_id: int):
    ref = _load_ref()
    config = config_from_gguf_kv(read_gguf_header(REAL_GGUF).kv)
    ratio = config.layer_ratio(layer_id)
    p = f"blk.{layer_id}."
    names = {
        p + "attn_sinks.weight",
        p + "attn_q_a.weight",
        p + "attn_q_a_norm.weight",
        p + "attn_q_b.weight",
        p + "attn_kv.weight",
        p + "attn_kv_a_norm.weight",
        p + "attn_output_a.weight",
        p + "attn_output_b.weight",
    }
    if ratio:
        names |= {
            p + "attn_compressor_kv.weight",
            p + "attn_compressor_gate.weight",
            p + "attn_compressor_ape.weight",
            p + "attn_compressor_norm.weight",
        }
    if ratio == 4:
        names |= {
            p + "indexer.attn_q_b.weight",
            p + "indexer.proj.weight",
            p + "indexer_compressor_kv.weight",
            p + "indexer_compressor_gate.weight",
            p + "indexer_compressor_ape.weight",
            p + "indexer_compressor_norm.weight",
        }
    t = load_gguf_tensors(REAL_GGUF, names, device="cpu")

    def dq(name, shape):
        return dequantize_q8_0(t[name].data).reshape(shape)

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    ref_attn = ref.Attention(layer_id, _ref_args(ref))
    torch.set_default_dtype(prev)
    ref_attn = ref_attn.cuda()
    ref_attn.attn_sink = torch.nn.Parameter(t[p + "attn_sinks.weight"].data.float().cuda())
    ref_attn.wq_a.weight = torch.nn.Parameter(
        dq(p + "attn_q_a.weight", (1024, 4096)).bfloat16().cuda()
    )
    ref_attn.q_norm.weight = torch.nn.Parameter(t[p + "attn_q_a_norm.weight"].data.float().cuda())
    ref_attn.wq_b.weight = torch.nn.Parameter(
        dq(p + "attn_q_b.weight", (32768, 1024)).bfloat16().cuda()
    )
    ref_attn.wkv.weight = torch.nn.Parameter(
        dq(p + "attn_kv.weight", (512, 4096)).bfloat16().cuda()
    )
    ref_attn.kv_norm.weight = torch.nn.Parameter(t[p + "attn_kv_a_norm.weight"].data.float().cuda())
    ref_attn.wo_a.weight = torch.nn.Parameter(
        dq(p + "attn_output_a.weight", (8192, 4096)).bfloat16().cuda()
    )
    ref_attn.wo_b.weight = torch.nn.Parameter(
        dq(p + "attn_output_b.weight", (4096, 8192)).bfloat16().cuda()
    )
    if ratio:
        comp = ref_attn.compressor
        coff = 2 if ratio == 4 else 1
        comp.wkv.weight = torch.nn.Parameter(
            dq(p + "attn_compressor_kv.weight", (coff * 512, 4096)).cuda()
        )
        comp.wgate.weight = torch.nn.Parameter(
            dq(p + "attn_compressor_gate.weight", (coff * 512, 4096)).cuda()
        )
        comp.ape = torch.nn.Parameter(
            dq(p + "attn_compressor_ape.weight", (ratio, coff * 512)).cuda()
        )
        comp.norm.weight = torch.nn.Parameter(
            t[p + "attn_compressor_norm.weight"].data.float().cuda()
        )
    if ratio == 4:
        idx = ref_attn.indexer
        idx.wq_b.weight = torch.nn.Parameter(
            dq(p + "indexer.attn_q_b.weight", (8192, 1024)).bfloat16().cuda()
        )
        idx.weights_proj.weight = torch.nn.Parameter(
            dq(p + "indexer.proj.weight", (64, 4096)).bfloat16().cuda()
        )
        icomp = idx.compressor
        icomp.wkv.weight = torch.nn.Parameter(
            dq(p + "indexer_compressor_kv.weight", (256, 4096)).cuda()
        )
        icomp.wgate.weight = torch.nn.Parameter(
            dq(p + "indexer_compressor_gate.weight", (256, 4096)).cuda()
        )
        icomp.ape = torch.nn.Parameter(dq(p + "indexer_compressor_ape.weight", (4, 256)).cuda())
        icomp.norm.weight = torch.nn.Parameter(
            t[p + "indexer_compressor_norm.weight"].data.float().cuda()
        )

    ours = Dsv4Attention(config, layer_id, max_seq_len=MAX_SEQ).cuda()
    ours.attn_sink.copy_(t[p + "attn_sinks.weight"].data.float().cuda())
    ours.wq_a.load_packed(t[p + "attn_q_a.weight"])
    ours.q_norm_weight.copy_(t[p + "attn_q_a_norm.weight"].data.float().cuda())
    ours.wq_b.load_packed(t[p + "attn_q_b.weight"])
    ours.wkv.load_packed(t[p + "attn_kv.weight"])
    ours.kv_norm_weight.copy_(t[p + "attn_kv_a_norm.weight"].data.float().cuda())
    ours.wo_a.load_packed(t[p + "attn_output_a.weight"])
    ours.wo_b.load_packed(t[p + "attn_output_b.weight"])
    if ratio:
        ours.compressor.wkv.load_packed(t[p + "attn_compressor_kv.weight"])
        ours.compressor.wgate.load_packed(t[p + "attn_compressor_gate.weight"])
        ours.compressor.ape.copy_(dq(p + "attn_compressor_ape.weight", (ratio, coff * 512)).cuda())
        ours.compressor.norm_weight.copy_(t[p + "attn_compressor_norm.weight"].data.float().cuda())
    if ratio == 4:
        ours.indexer.wq_b.load_packed(t[p + "indexer.attn_q_b.weight"])
        ours.indexer.weights_proj.load_packed(t[p + "indexer.proj.weight"])
        ours.indexer.compressor.wkv.load_packed(t[p + "indexer_compressor_kv.weight"])
        ours.indexer.compressor.wgate.load_packed(t[p + "indexer_compressor_gate.weight"])
        ours.indexer.compressor.ape.copy_(dq(p + "indexer_compressor_ape.weight", (4, 256)).cuda())
        ours.indexer.compressor.norm_weight.copy_(
            t[p + "indexer_compressor_norm.weight"].data.float().cuda()
        )
    return ref, ref_attn, ours


def _run(ref_attn, ours, ratio, seed, prefill_len=40, decode_steps=20) -> None:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(1, prefill_len, 4096, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05

    prev_device = torch.device("cpu")
    torch.set_default_device("cuda")
    try:
        r0 = ref_attn(x, 0)
        o0 = ours(x, 0)
    finally:
        torch.set_default_device(prev_device)
    assert torch.allclose(o0, r0, rtol=2e-3, atol=2e-4), (
        f"prefill out max diff {(o0.float() - r0.float()).abs().max()}"
    )
    assert torch.equal(ours.kv_cache, ref_attn.kv_cache)
    if ratio == 4:
        assert torch.equal(ours.indexer.kv_cache, ref_attn.indexer.kv_cache)

    for step in range(decode_steps):
        pos = prefill_len + step
        x1 = torch.randn(1, 1, 4096, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
        prev_device = torch.device("cpu")
        torch.set_default_device("cuda")
        try:
            r = ref_attn(x1, pos)
            o = ours(x1, pos)
        finally:
            torch.set_default_device(prev_device)
        # While every live compressed entry fits within index_topk, the
        # runtime uses canonical physical order and the reference uses its
        # score-sorted permutation. The attended set is identical, but the
        # BF16 reduction order can differ by one output ULP.
        assert torch.allclose(o, r, rtol=2e-3, atol=8e-3), (
            f"decode out at {pos} max diff {(o.float() - r.float()).abs().max()}"
        )
        assert torch.equal(ours.kv_cache, ref_attn.kv_cache)
        if ratio == 4:
            assert torch.equal(ours.indexer.kv_cache, ref_attn.indexer.kv_cache)


def test_attention_ratio4_csa() -> None:
    _, ref_attn, ours = _build_pair(2)
    _run(ref_attn, ours, ratio=4, seed=31)


def test_attention_ratio128_hca() -> None:
    _, ref_attn, ours = _build_pair(3)
    _run(ref_attn, ours, ratio=128, seed=32)


def test_attention_ratio0_window_only() -> None:
    _, ref_attn, ours = _build_pair(0)
    _run(ref_attn, ours, ratio=0, seed=33)
