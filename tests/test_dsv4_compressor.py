"""Compressor parity: our Dsv4Compressor vs the reference module, driven by
real GGUF weights (blk.3 ratio-128, blk.2 ratio-4 overlap) on GPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

from runtime.loading.gguf import load_gguf_tensors  # noqa: E402
from runtime.model.dsv4_config import config_from_gguf_kv  # noqa: E402
from runtime.model.dsv4_model import Dsv4Compressor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"
REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
if not REAL_GGUF.exists():
    pytest.skip("GGUF download not present", allow_module_level=True)
if not REFERENCE_DIR.exists():
    pytest.skip("reference drop not present", allow_module_level=True)


def _load_ref():
    if "kernel" not in sys.modules:
        spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(spec)
        sys.modules["kernel"] = kernel
        spec.loader.exec_module(kernel)
    spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # production QAT config: fp8-origin weights -> ue8m0 scales
    module.scale_fmt = "ue8m0"
    module.scale_dtype = torch.float8_e8m0fnu
    return module


def _ref_args(ref):
    return ref.ModelArgs(
        max_batch_size=1,
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


def _config():
    from loader.gguf_header import read_gguf_header

    header = read_gguf_header(REAL_GGUF)
    return config_from_gguf_kv(header.kv)


def _build_pair(ref, config, layer_id: int, device="cuda"):
    """Reference + ours compressor sharing the same GGUF weights."""
    prefix = f"blk.{layer_id}."
    tensors = load_gguf_tensors(
        REAL_GGUF,
        {
            prefix + "attn_compressor_kv.weight",
            prefix + "attn_compressor_gate.weight",
            prefix + "attn_compressor_ape.weight",
            prefix + "attn_compressor_norm.weight",
        },
        device="cpu",
    )
    ratio = config.layer_ratio(layer_id)
    coff = 2 if ratio == 4 else 1

    ref_comp = ref.Compressor(_ref_args(ref), compress_ratio=ratio, head_dim=config.head_dim)
    wkv = tensors[prefix + "attn_compressor_kv.weight"]
    wgate = tensors[prefix + "attn_compressor_gate.weight"]
    from runtime.model.dsv4_quant import dequantize_q8_0

    ref_comp.wkv.weight = torch.nn.Parameter(
        dequantize_q8_0(wkv.data).reshape(coff * config.head_dim, config.hidden_size)
    )
    ref_comp.wgate.weight = torch.nn.Parameter(
        dequantize_q8_0(wgate.data).reshape(coff * config.head_dim, config.hidden_size)
    )
    # ape is Q8_0-quantized in this GGUF (norm stays F32)
    ape = dequantize_q8_0(tensors[prefix + "attn_compressor_ape.weight"].data)
    ape = ape.reshape(ratio, coff * config.head_dim)
    ref_comp.ape = torch.nn.Parameter(ape)
    norm_w = tensors[prefix + "attn_compressor_norm.weight"].data.to(torch.float32)
    ref_comp.norm.weight = torch.nn.Parameter(norm_w)
    ref_comp = ref_comp.to(device)

    ours = Dsv4Compressor(config, layer_id).to(device)
    ours.wkv.load_packed(wkv)
    ours.wgate.load_packed(wgate)
    ours.ape.copy_(ape.to(device))
    ours.norm_weight.copy_(norm_w.to(device))

    freqs = ref.precompute_freqs_cis(
        config.rope_head_dim,
        4096,
        config.rope_original_seq_len,
        config.compress_rope_theta,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    ).to(device)
    ref_comp.freqs_cis = freqs
    ours.freqs_cis = freqs
    max_entries = 4096 // ratio + 8
    ref_comp.kv_cache = torch.zeros(
        1, max_entries, config.head_dim, dtype=torch.bfloat16, device=device
    )
    ours.kv_cache = torch.zeros(
        1, max_entries, config.head_dim, dtype=torch.bfloat16, device=device
    )
    return ref_comp, ours


def _run_parity(layer_id: int, prefill_len: int, decode_steps: int) -> None:
    ref = _load_ref()
    config = _config()
    ref_comp, ours = _build_pair(ref, config, layer_id)
    ratio = config.layer_ratio(layer_id)

    gen = torch.Generator(device="cuda").manual_seed(100 + layer_id)
    seq = (
        torch.randn(1, prefill_len, 4096, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
    )

    ref_out = ref_comp(seq, 0)
    our_out = ours(seq, 0)
    if ref_out is None:
        assert our_out is None
    else:
        assert our_out is not None
        assert torch.allclose(our_out, ref_out, rtol=0, atol=0), (
            f"prefill emission differs: max {(our_out - ref_out).abs().max()}"
        )
    assert torch.equal(ours.kv_cache, ref_comp.kv_cache)
    assert torch.equal(ours.kv_state, ref_comp.kv_state)
    assert torch.equal(ours.score_state, ref_comp.score_state)

    for step in range(decode_steps):
        pos = prefill_len + step
        x = torch.randn(1, 1, 4096, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
        r = ref_comp(x, pos)
        o = ours(x, pos)
        assert (r is None) == (o is None)
        if r is not None:
            assert torch.allclose(o, r, rtol=0, atol=0), (
                f"decode emission at {pos} differs: max {(o - r).abs().max()}"
            )
        assert torch.equal(ours.kv_cache, ref_comp.kv_cache)
        assert torch.equal(ours.kv_state, ref_comp.kv_state)
        assert torch.equal(ours.score_state, ref_comp.score_state)
    # sanity: the expected number of compressed entries landed in the cache
    total = prefill_len + decode_steps
    assert int(torch.count_nonzero(ref_comp.kv_cache.abs().sum(dim=-1))) == total // ratio


def test_compressor_ratio128_parity() -> None:
    # 300-token prefill -> 2 emissions + remainder 44; decode across two more
    # emission boundaries
    _run_parity(layer_id=3, prefill_len=300, decode_steps=300)


def test_compressor_ratio4_overlap_parity() -> None:
    # overlap path: carry-over window state, 7 emissions from a 31-token
    # prefill (31 // 4 = 7, remainder 3), then decode through boundaries
    _run_parity(layer_id=2, prefill_len=31, decode_steps=40)
