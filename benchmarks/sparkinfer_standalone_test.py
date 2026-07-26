#!/usr/bin/env python3
"""sparkinfer standalone MoE: correctness + perf test.

Usage:
    /home/bot/.venvs/vllm/bin/python benchmarks/sparkinfer_standalone_test.py
"""
from __future__ import annotations
import os, sys, time, statistics
os.environ["USE_LIBUV"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from runtime.backends.laguna_sparkinfer_moe import (
    SparkinferMoEModel, load_moe_layer_weights, prepare_sparkinfer_layer,
    SparkinferMoELayer, _find_checkpoint,
    NUM_EXPERTS, TOP_K, HIDDEN_SIZE,
)
from sparkinfer.moe.fused_moe._impl import allocate_tp_moe_workspace_pool

FP4_LUT = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)

def dequant_fp4(packed, scale, global_scale):
    rows, hc = packed.shape
    lo = (packed & 0xF).long(); hi = ((packed >> 4) & 0xF).long()
    lut = FP4_LUT.to(packed.device)
    vals = torch.stack([lut[lo], lut[hi]], -1).reshape(rows, hc * 2)
    return vals * scale.float().repeat_interleave(16, dim=-1) / global_scale

def reference_moe(raw, hidden, topk_ids, topk_weights, device="cuda"):
    M = hidden.shape[0]
    ref = torch.zeros(M, HIDDEN_SIZE, dtype=torch.float32, device=device)
    active = topk_ids.unique().cpu()
    cache = {}
    for eid_t in active:
        eid = eid_t.item()
        cache[eid] = (
            dequant_fp4(raw["gate_w"][eid], raw["gate_sf"][eid], raw["gate_gs"][eid]),
            dequant_fp4(raw["up_w"][eid], raw["up_sf"][eid], raw["up_gs"][eid]),
            dequant_fp4(raw["down_w"][eid], raw["down_sf"][eid], raw["down_gs"][eid]),
        )
    hf = hidden.float()
    for m in range(M):
        for ki in range(TOP_K):
            eid = topk_ids[m, ki].item()
            w = topk_weights[m, ki].float()
            g, u, d = cache[eid]
            go = hf[m] @ g.T; uo = hf[m] @ u.T
            ref[m] += w * (F.silu(go) * uo @ d.T)
    return ref


def main():
    device = "cuda"
    ckpt = _find_checkpoint()
    print(f"Checkpoint: {ckpt}")

    # ── 1. Single-layer correctness ──
    print("\n=== 1. Single-layer correctness (layer 1) ===")
    raw = load_moe_layer_weights(ckpt, 1, device)
    experts = prepare_sparkinfer_layer(raw, device)
    ws = allocate_tp_moe_workspace_pool()
    layer = SparkinferMoELayer(experts, ws, device)

    M = 4
    torch.manual_seed(42)
    hidden = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.1
    router_logits = torch.randn(M, NUM_EXPERTS, dtype=torch.float32, device=device)
    topk_w_raw, topk_ids = torch.topk(router_logits, TOP_K, dim=-1)
    topk_weights = F.softmax(topk_w_raw, dim=-1).to(torch.bfloat16)

    result = layer.forward(hidden, topk_ids, topk_weights)
    ref = reference_moe(raw, hidden, topk_ids, topk_weights, device)
    cos = F.cosine_similarity(result.float().flatten(), ref.flatten(), dim=0)
    rel = (result.float() - ref).norm() / ref.norm()
    print(f"  sparkinfer norm: {result.float().norm():.4f}")
    print(f"  reference norm:  {ref.norm():.4f}")
    print(f"  cosine: {cos.item():.6f}  rel_err: {rel.item():.4f}")
    print(f"  {'✓ PASS' if cos.item() > 0.93 else '✗ FAIL'} (FP4 quantization error expected)")
    del raw; torch.cuda.empty_cache()

    # ── 2. Single-layer perf ──
    print("\n=== 2. Single-layer perf ===")
    for M in [1, 16]:
        hidden = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.1
        topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=device)
        topk_weights = F.softmax(torch.randn(M, TOP_K, device=device), dim=-1).to(torch.bfloat16)

        # Warmup
        for _ in range(10):
            layer.forward(hidden, topk_ids, topk_weights)
        torch.cuda.synchronize()

        # CUDA event timing
        N = 50
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        for i in range(N):
            starts[i].record()
            layer.forward(hidden, topk_ids, topk_weights)
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
        fast_pct = sum(1 for t in times if t < 100) / N * 100
        print(f"  M={M:2d}: p10={times[N//10]:.0f}us median={times[N//2]:.0f}us "
              f"min={times[0]:.0f}us  fast(<100us)={fast_pct:.0f}%  "
              f"47L@min={times[0]*47/1000:.1f}ms")

    # ── 3. CUDA graph capture ──
    print("\n=== 3. CUDA graph capture (M=1) ===")
    M = 1
    hidden = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.1
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=device)
    topk_weights = F.softmax(torch.randn(M, TOP_K, device=device), dim=-1).to(torch.bfloat16)
    out_buf = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(5):
        layer.forward(hidden, topk_ids, topk_weights)
    torch.cuda.synchronize()

    try:
        from sparkinfer.moe.fused_moe._impl import build_tp_moe_fp4_binding, sparkinfer_moe_fp4
        binding = build_tp_moe_fp4_binding(
            scratch=ws, a=hidden, experts=experts,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_mode="nvfp4", input_scales_static=True, output=out_buf,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sparkinfer_moe_fp4(binding=binding)
        torch.cuda.synchronize()

        # Time graph replay
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        N = 50
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        for i in range(N):
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        times_g = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
        print(f"  Graph M=1: median={times_g[N//2]:.0f}us min={times_g[0]:.0f}us "
              f"47L@median={times_g[N//2]*47/1000:.1f}ms")
        print("  ✓ CUDA graph capture works")
    except Exception as e:
        print(f"  ✗ CUDA graph failed: {e}")

    # ── 4. Multi-layer load + chain timing ──
    print("\n=== 4. Loading all 47 MoE layers ===")
    model = SparkinferMoEModel(ckpt, device=device)
    t0 = time.time()
    model.load_all()
    load_time = time.time() - t0
    print(f"  Total load: {load_time:.1f}s ({load_time/47:.1f}s/layer)")

    # Chain timing: run all 47 layers sequentially
    print("\n=== 5. 47-layer chain timing (M=1) ===")
    M = 1
    hidden = torch.randn(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device) * 0.1
    topk_ids = torch.randint(0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device=device)
    topk_weights = F.softmax(torch.randn(M, TOP_K, device=device), dim=-1).to(torch.bfloat16)

    # Warmup
    for lid in model.layer_ids:
        model.forward_layer(lid, hidden, topk_ids, topk_weights)
    torch.cuda.synchronize()

    N = 10
    chain_times = []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.time()
        for lid in model.layer_ids:
            model.forward_layer(lid, hidden, topk_ids, topk_weights)
        torch.cuda.synchronize()
        chain_times.append((time.time() - t0) * 1000)
    med = statistics.median(chain_times)
    print(f"  47-layer chain: median={med:.1f}ms min={min(chain_times):.1f}ms")
    print(f"  Per-layer avg: {med/47:.1f}ms")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
