"""Standalone sparkinfer vs PyTorch SDPA comparison.

Creates known Q/K/V, writes KV to paged cache, runs sparkinfer attention,
compares with torch SDPA. No model loading needed — fast.
"""
import os, sys
os.environ["USE_LIBUV"] = "0"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
sys.path.insert(0, "/home/bot/project/sparkinfer")

import torch
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda"

# Laguna layer 0 shape: 48 Q heads, 8 KV heads, head_dim=128
NUM_Q_HEADADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 5
PAGE_SIZE = 64
NUM_PAGES = 4
SCALE = HEAD_DIM ** -0.5

# FP8 KV scales (typical values from the model)
K_SCALE = 0.032
V_SCALE = 0.001

print("=== Sparkinfer vs PyTorch SDPA ===")
print(f"Shape: {NUM_Q_HEADADS}Q/{NUM_KV_HEADS}KV heads, dim={HEAD_DIM}, seq={SEQ_LEN}")

# Create random Q, K, V in bf16
q_bf16 = torch.randn(SEQ_LEN, NUM_Q_HEADADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
k_bf16 = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
v_bf16 = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

# ── Reference: PyTorch SDPA (bf16, no quantization) ──
# Expand KV heads for GQA
k_expanded = k_bf16.unsqueeze(2).expand(-1, -1, NUM_Q_HEADADS // NUM_KV_HEADS, -1).reshape(SEQ_LEN, NUM_Q_HEADADS, HEAD_DIM)
v_expanded = v_bf16.unsqueeze(2).expand(-1, -1, NUM_Q_HEADADS // NUM_KV_HEADS, -1).reshape(SEQ_LEN, NUM_Q_HEADADS, HEAD_DIM)

# SDPA expects [batch, heads, seq, dim]
q_sdpa = q_bf16.transpose(0, 1).unsqueeze(0)  # [1, 48, 5, 128]
k_sdpa = k_expanded.transpose(0, 1).unsqueeze(0)
v_sdpa = v_expanded.transpose(0, 1).unsqueeze(0)

ref_output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
ref_output = ref_output.squeeze(0).transpose(0, 1)  # [5, 48, 128]
print(f"\nReference (SDPA bf16):")
print(f"  output norm: {ref_output.float().norm().item():.4f}")
print(f"  output[0,0,:4]: {ref_output[0,0,:4].tolist()}")

# ── Sparkinfer path: FP8 paged KV cache ──
# Create paged KV cache [num_pages, 2, page_size, num_kv_heads, head_dim]
kv_cache = torch.zeros(NUM_PAGES, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                        dtype=torch.uint8, device=device)

# Write KV to page 0, offsets 0..4 (with FP8 quantization)
k_cache = kv_cache[:, 0].view(torch.float8_e4m3fn)
v_cache = kv_cache[:, 1].view(torch.float8_e4m3fn)
k_cache[0, :SEQ_LEN] = (k_bf16 / K_SCALE).to(torch.float8_e4m3fn)
v_cache[0, :SEQ_LEN] = (v_bf16 / V_SCALE).to(torch.float8_e4m3fn)

# Read back and verify
k_readback = k_cache[0, :SEQ_LEN].float() * K_SCALE
v_readback = v_cache[0, :SEQ_LEN].float() * V_SCALE
k_err = (k_readback - k_bf16.float()).norm() / k_bf16.float().norm()
v_err = (v_readback - v_bf16.float()).norm() / v_bf16.float().norm()
print(f"\nFP8 quantization error:")
print(f"  K relative error: {k_err.item():.6f}")
print(f"  V relative error: {v_err.item():.6f}")

# Run sparkinfer attention
from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
from sparkinfer.attention.paged.planner import create_paged_plan
from sparkinfer.attention.paged._forward import paged_attention_forward
from sparkinfer.attention.paged._scratch import build_paged_attention_binding

# Metadata
page_table = torch.tensor([[0]], dtype=torch.int32, device=device)  # page 0
cache_seqlens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=device)
cu_seqlens_q = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=device)

# Prepare caches for sparkinfer: [num_pages, page_size, num_kv_heads, head_dim]
si_k_cache = kv_cache[:, 0].view(torch.float8_e4m3fn)
si_v_cache = kv_cache[:, 1].view(torch.float8_e4m3fn)

ws = PagedAttentionWorkspace.for_tensors(
    mode="extend", q=q_bf16, k_cache=si_k_cache, v_cache=si_v_cache,
    use_cuda_graph=False)

plan = create_paged_plan(
    q_bf16, si_k_cache, si_v_cache, page_table, cache_seqlens, cu_seqlens_q,
    mode="extend", enable_cuda_graph=False, window_left=-1)
ws._ensure_capacity(plan)
ws._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
ws._copy_plan_metadata(plan)
ws._plan = plan

si_output = torch.zeros_like(q_bf16)
k_descale = torch.tensor([K_SCALE], dtype=torch.float32, device=device)
v_descale = torch.tensor([V_SCALE], dtype=torch.float32, device=device)

binding = build_paged_attention_binding(
    scratch=ws, q=q_bf16, k_cache=si_k_cache, v_cache=si_v_cache,
    output=si_output, k_descale=k_descale, v_descale=v_descale)
paged_attention_forward(binding=binding)

print(f"\nSparkinfer output:")
print(f"  output norm: {si_output.float().norm().item():.4f}")
print(f"  output[0,0,:4]: {si_output[0,0,:4].tolist()}")

# Compare
cos_sim = F.cosine_similarity(
    si_output.float().reshape(1, -1),
    ref_output.float().reshape(1, -1)
).item()
rel_err = (si_output.float() - ref_output.float()).norm() / ref_output.float().norm()

print(f"\n=== Comparison ===")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Relative error: {rel_err.item():.6f}")

if cos_sim > 0.99:
    print("  ✅ Sparkinfer matches SDPA reference")
else:
    print(f"  ❌ MISMATCH — cos={cos_sim:.4f}")
    # Debug: check per-position
    for i in range(SEQ_LEN):
        pos_cos = F.cosine_similarity(
            si_output[i].float().reshape(1, -1),
            ref_output[i].float().reshape(1, -1)
        ).item()
        print(f"    pos {i}: cos={pos_cos:.4f} si_norm={si_output[i].float().norm():.2f} ref_norm={ref_output[i].float().norm():.2f}")
