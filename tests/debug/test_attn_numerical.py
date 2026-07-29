"""Numerical closed-loop test for BFAttention + sparkinfer.

Captures real Q/K/V from model forward, then:
1. FP8 round-trip: write → read back → compare with original
2. Sparkinfer vs SDPA: same Q/K/V, compare attention outputs
3. Checks both layer 0 (full attn) and first SWA layer

Single model load (~2.5min), all tests run in one pass.
"""

import os, sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
sys.path.insert(0, "/home/bot/project/sparkinfer")

import torch
import torch.nn.functional as F

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

from oracle.qwen36_vllm.vllm_compat import EngineArgs

vc = EngineArgs(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.88,
    enforce_eager=True,
    trust_remote_code=True,
).create_engine_config()

from runtime.backends.laguna import LagunaBackend

backend = LagunaBackend(vc, num_slots=1, block_size=64, blocks_per_slot=64)

# ── Capture real Q/K/V and sparkinfer output from layer 0 ──
from runtime.backends.bf_attention import BFAttention, get_bf_attn_context

captured = {}  # layer_name -> {q, k, v, output, k_scale, v_scale, meta}

_orig_fwd = BFAttention.forward


def _capture_fwd(self, query, key, value, output_shape=None, output_dtype=None):
    result = _orig_fwd(self, query, key, value, output_shape, output_dtype)
    name = self.layer_name
    # Capture layer 0 and layer 1 (first SWA)
    if name in ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"):
        ctx = get_bf_attn_context()
        captured[name] = {
            "q": query.detach().clone(),
            "k": key.detach().clone() if key is not None else None,
            "v": value.detach().clone() if value is not None else None,
            "output": result.detach().clone(),
            "k_scale": self._k_scale.detach().clone(),
            "v_scale": self._v_scale.detach().clone(),
            "kv_cache": self.kv_cache,
            "meta": ctx.attn_metadata.get(name),
            "slot_mapping": ctx.slot_mapping.get(name),
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
        }
    return result


BFAttention.forward = _capture_fwd

# Run prefill
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
token_ids = tok.encode("The capital of France is")
assert token_ids[0] == tok.bos_token_id, (
    f"Missing BOS: first token is {token_ids[0]}, expected {tok.bos_token_id}"
)
backend.reset_slot(0)
first = backend.prefill(0, token_ids)
print(f"First token: {first} = {tok.decode([first])!r}")

# ── Analyze captured data ──
for name in sorted(captured.keys()):
    d = captured[name]
    q_raw = d["q"]  # [M, num_heads*head_dim] (2D from LagunaAttention)
    k_raw = d["k"]
    v_raw = d["v"]
    nh = d["num_heads"]
    nkv = d["num_kv_heads"]
    hd = d["head_size"]
    ks = d["k_scale"].item()
    vs = d["v_scale"].item()
    meta = d["meta"]
    sm = d["slot_mapping"]
    kv_cache = d["kv_cache"]
    si_out = d["output"]  # [M, nh*hd]

    M = q_raw.shape[0]
    q = q_raw.view(M, nh, hd)
    k = k_raw.view(M, nkv, hd) if k_raw is not None else None
    v = v_raw.view(M, nkv, hd) if v_raw is not None else None

    print(f"\n{'=' * 60}")
    print(f"Layer: {name}")
    print(f"  M={M} nh={nh} nkv={nkv} hd={hd} k_scale={ks:.6f} v_scale={vs:.6f}")
    print(
        f"  q norm={q.float().norm():.2f} k norm={k.float().norm():.2f} v norm={v.float().norm():.2f}"
    )

    # ── Test 1: FP8 round-trip ──
    k_cache_fp8 = kv_cache[:, 0].view(torch.float8_e4m3fn)
    v_cache_fp8 = kv_cache[:, 1].view(torch.float8_e4m3fn)
    bs = k_cache_fp8.shape[1]
    bi = (sm // bs).tolist()
    bo = (sm % bs).tolist()

    k_dequant = torch.stack([k_cache_fp8[bi[i], bo[i]].float() * ks for i in range(M)])
    v_dequant = torch.stack([v_cache_fp8[bi[i], bo[i]].float() * vs for i in range(M)])
    k_err = (k_dequant - k.float()).norm() / k.float().norm()
    v_err = (v_dequant - v.float()).norm() / v.float().norm()
    print("\n  [FP8 round-trip]")
    print(f"    K rel error: {k_err:.6f}")
    print(f"    V rel error: {v_err:.6f}")
    # Check for overflow (FP8 e4m3 max = 448)
    k_fp8_vals = torch.stack([k_cache_fp8[bi[i], bo[i]].float() for i in range(M)])
    v_fp8_vals = torch.stack([v_cache_fp8[bi[i], bo[i]].float() for i in range(M)])
    print(f"    K FP8 range: [{k_fp8_vals.min():.1f}, {k_fp8_vals.max():.1f}] (max=448)")
    print(f"    V FP8 range: [{v_fp8_vals.min():.1f}, {v_fp8_vals.max():.1f}] (max=448)")

    # ── Test 2: SDPA reference (bf16, no quantization) ──
    seq_len = meta.cache_seqlens[0].item()
    k_exp = k.unsqueeze(2).expand(-1, -1, nh // nkv, -1).reshape(M, nh, hd)
    v_exp = v.unsqueeze(2).expand(-1, -1, nh // nkv, -1).reshape(M, nh, hd)
    ref = (
        F.scaled_dot_product_attention(
            q.float().transpose(0, 1).unsqueeze(0),
            k_exp.float().transpose(0, 1).unsqueeze(0),
            v_exp.float().transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )  # [M, nh, hd]
    ref_flat = ref.reshape(M, nh * hd).to(torch.bfloat16)

    si_flat = si_out  # [M, nh*hd]
    cos = F.cosine_similarity(
        si_flat.float().reshape(1, -1), ref_flat.float().reshape(1, -1)
    ).item()
    rel = (si_flat.float() - ref_flat.float()).norm() / ref_flat.float().norm()
    print("\n  [Sparkinfer vs SDPA (bf16 ref)]")
    print(f"    cosine: {cos:.6f}")
    print(f"    rel_err: {rel.item():.6f}")
    print(
        f"    si norm: {si_flat.float().norm():.2f}  ref norm: {ref_flat.float().norm():.2f}  ratio: {ref_flat.float().norm() / si_flat.float().norm():.4f}"
    )

    # Per-position
    for i in range(min(M, 5)):
        pc = F.cosine_similarity(
            si_flat[i].float().reshape(1, -1), ref_flat[i].float().reshape(1, -1)
        ).item()
        print(
            f"    pos{i}: cos={pc:.4f} si={si_flat[i].float().norm():.3f} ref={ref_flat[i].float().norm():.3f}"
        )

    # ── Test 3: SDPA with dequantized FP8 K/V (isolates quantization) ──
    kd_exp = k_dequant.unsqueeze(2).expand(-1, -1, nh // nkv, -1).reshape(M, nh, hd)
    vd_exp = v_dequant.unsqueeze(2).expand(-1, -1, nh // nkv, -1).reshape(M, nh, hd)
    ref_fp8 = (
        F.scaled_dot_product_attention(
            q.float().transpose(0, 1).unsqueeze(0),
            kd_exp.transpose(0, 1).unsqueeze(0),
            vd_exp.transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    ref_fp8_flat = ref_fp8.reshape(M, nh * hd).to(torch.bfloat16)
    cos_fp8 = F.cosine_similarity(
        si_flat.float().reshape(1, -1), ref_fp8_flat.float().reshape(1, -1)
    ).item()
    cos_quant = F.cosine_similarity(
        ref_fp8_flat.float().reshape(1, -1), ref_flat.float().reshape(1, -1)
    ).item()
    print("\n  [Isolation]")
    print(f"    sparkinfer vs SDPA(fp8-dequant): cos={cos_fp8:.6f}  ← kernel correctness")
    print(f"    SDPA(fp8-dequant) vs SDPA(bf16): cos={cos_quant:.6f}  ← quantization loss")

print(f"\n{'=' * 60}")
print("DONE")
