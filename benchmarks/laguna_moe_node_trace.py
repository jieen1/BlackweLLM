#!/usr/bin/env python3
"""Laguna MoE node trace — 单次加载跑全矩阵的 kernel ledger。

证据驱动门禁脚本（用户裁定 2026-07-23）：
  - 模型加载昂贵（~65s/次），故一次加载尽可能多取数；
  - 单进程内遍历 (batch × context) 组合，每个组合抓 torch.profiler kernel 分解；
  - 按 ledger 分类法归桶 + GEMM 家族（nvjet/cutlass/triton/cublas）归属；
  - 物理论据：MoE GEMM kernel 时间是 eager/compile 模式不变量（compile 只融合
    elementwise + 消 launch gap，不碰 GEMM）→ eager 测得的 GEMM 即生产残留。

PROFILING-ONLY：只调用既有 backend 接口，不改 runtime/。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_moe_node_trace \
        --mode eager --batches 1 4 --ctx 1024 16384
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QSR_A2_CUSTOM_GEMM", "0")

MODEL = "poolside/Laguna-S-2.1-NVFP4"

CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("moe_routing", re.compile(r"topk|gating|softmax|expand|permute|finalize|unpermute|scatter|gather|moe_align|sort", re.I)),
    ("attention", re.compile(r"flash|fmha|attention|rope|rotary|reshape_cache|copy_kernel|fill_kernel|gemvx|qkv|cascade", re.I)),
    ("lm_head", re.compile(r"lm_head|logits", re.I)),
    ("gemm", re.compile(r"cutlass|gemm|nvjet|sm120|splitk|split_k|sgemm|hgemm|wgrad|matmul|scaled_mm|fp4|nvfp4|blockwise", re.I)),
    ("norm_quant", re.compile(r"norm|rms|layernorm|quant|dequant|scale|cast|elementwise|reduce|sum|mean|silu|gelu|activation|fused|vectorized|unrolled", re.I)),
]

GEMM_FAMILY_RULES: list[tuple[str, re.Pattern]] = [
    ("nvjet", re.compile(r"nvjet", re.I)),
    ("cutlass", re.compile(r"cutlass|GemmUniv|sm120|splitk|split_k", re.I)),
    ("triton", re.compile(r"triton", re.I)),
    ("cublas", re.compile(r"cublas|gemvx|ampere|volta|sm[89]0_xmma", re.I)),
]


def categorize(name: str) -> str:
    for cat, pat in CATEGORY_RULES:
        if pat.search(name):
            return cat
    return "other"


def gemm_family(name: str) -> str:
    for fam, pat in GEMM_FAMILY_RULES:
        if pat.search(name):
            return fam
    return "other_gemm"


def build_config(mode: str, max_model_len: int):
    from oracle.qwen36_vllm.vllm_compat import EngineArgs

    if mode == "eager":
        args = EngineArgs(
            model=MODEL, max_model_len=max_model_len, gpu_memory_utilization=0.85,
            enforce_eager=True, dtype="bfloat16", disable_log_stats=True,
            async_scheduling=False,
        )
        return args.create_engine_config()
    from oracle.qwen36_vllm.vllm_compat import CUDAGraphMode
    args = EngineArgs(
        model=MODEL, max_model_len=max_model_len, gpu_memory_utilization=0.85,
        dtype="bfloat16", disable_log_stats=True, async_scheduling=False,
    )
    config = args.create_engine_config()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    return config


def profile_combo(backend, prompt_ids: list[int], batch: int, steps: int, warmup: int, shapes: bool = False) -> dict:
    """Profile one (batch) combo: prefill batch slots, then profile batched decode."""
    import torch
    from torch.profiler import profile, ProfilerActivity

    slots = list(range(batch))
    for s in slots:
        backend.reset_slot(s)
        backend.prefill(s, prompt_ids)

    def step_batch() -> None:
        kv = [backend.slot_kv_len[s] for s in slots]
        backend._forward(slots, [1] * batch, kv, qo_len=1, is_decode=True)
        for s in slots:
            backend.slot_kv_len[s] += 1

    for _ in range(warmup):
        step_batch()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        step_batch()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000
    wall_per_step = wall_ms / steps

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=shapes, with_flops=False) as prof:
        for _ in range(steps):
            step_batch()
        torch.cuda.synchronize()

    cat_ms: dict[str, float] = defaultdict(float)
    fam_ms: dict[str, float] = defaultdict(float)
    detail: list[tuple[str, float, int, str, str]] = []
    total = 0.0
    for e in prof.key_averages():
        if e.device_time_total <= 0:
            continue
        ms = e.device_time_total / 1000.0 / steps
        cat = categorize(e.key)
        cat_ms[cat] += ms
        total += ms
        fam = gemm_family(e.key) if cat == "gemm" else "-"
        if cat == "gemm":
            fam_ms[fam] += ms
        detail.append((e.key, ms, e.count // steps, cat, fam))
    detail.sort(key=lambda r: -r[1])

    shape_rows = []
    if shapes:
        agg = {}
        for e in prof.key_averages(group_by_input_shape=True):
            if e.device_time_total <= 0:
                continue
            sh = str(e.input_shapes)
            key = (e.key, sh)
            agg[key] = agg.get(key, 0.0) + e.device_time_total / 1000.0 / steps
        for (name, sh), ms in sorted(agg.items(), key=lambda x: -x[1]):
            if categorize(name) == "gemm":
                shape_rows.append({"name": name[:60], "shapes": sh, "ms": round(ms, 4),
                                   "fam": gemm_family(name)})

    for s in slots:
        backend.reset_slot(s)

    return {
        "wall_per_step_ms": round(wall_per_step, 3),
        "kernel_per_step_ms": round(total, 3),
        "gap_ms": round(wall_per_step - total, 3),
        "category_ms": {c: round(cat_ms.get(c, 0.0), 3) for c in
                        ["gemm", "norm_quant", "moe_routing", "attention", "lm_head", "other"]},
        "gemm_family_ms": {f: round(m, 3) for f, m in fam_ms.items()},
        "top_kernels": [
            {"name": n, "ms": round(m, 4), "per_step": c, "cat": cat, "fam": fam}
            for n, m, c, cat, fam in detail[:60]
        ],
        "gemm_shapes": shape_rows,
    }


def print_ledger(tag: str, r: dict) -> None:
    total = r["kernel_per_step_ms"]
    labels = {
        "gemm": "GEMM (MoE+dense)", "norm_quant": "Norm/quant/elem",
        "moe_routing": "MoE routing/perm/fin", "attention": "Attention(+RoPE+KV)",
        "lm_head": "lm_head", "other": "other",
    }
    print(f"\n{'='*70}\n{tag}", flush=True)
    print(f"  wall={r['wall_per_step_ms']:.2f}ms  kernel={total:.2f}ms  gap={r['gap_ms']:.2f}ms", flush=True)
    print(f"  {'类别':<26}{'ms/step':>10}{'占比':>8}", flush=True)
    for c in ["gemm", "norm_quant", "moe_routing", "attention", "lm_head", "other"]:
        ms = r["category_ms"].get(c, 0.0)
        pct = 100 * ms / total if total else 0
        print(f"  {labels[c]:<26}{ms:>9.2f}m{pct:>7.1f}%", flush=True)
    gemm_total = r["category_ms"].get("gemm", 0.0)
    print(f"  {'GEMM 家族':<26}{'ms/step':>10}{'占GEMM':>8}", flush=True)
    for f, ms in sorted(r["gemm_family_ms"].items(), key=lambda x: -x[1]):
        pct = 100 * ms / gemm_total if gemm_total else 0
        print(f"  {f:<26}{ms:>9.2f}m{pct:>7.1f}%", flush=True)
    print("  Top 12 kernels:", flush=True)
    for k in r["top_kernels"][:12]:
        nm = k["name"][:46]
        print(f"    {nm:<46}{k['ms']:>7.3f}m ×{k['per_step']:<4} {k['cat']:<11}{k['fam']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "compile"], default="eager")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024])
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--shapes", action="store_true", help="record_shapes for GEMM attribution")
    args = ap.parse_args()

    import torch
    from runtime.backends.laguna import LagunaBackend
    from transformers import AutoTokenizer

    max_ctx = max(args.ctx)
    mml = max(max_ctx + 2048, 4096)
    print(f"GPU: {torch.cuda.get_device_name(0)}  mode={args.mode}", flush=True)
    print(f"batches={args.batches} ctx={args.ctx} max_model_len={mml}", flush=True)

    t0 = time.perf_counter()
    config = build_config(args.mode, mml)
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=(mml + 15) // 16)
    print(f"Backend loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = tok.encode("Write a detailed explanation of quantum computing:")

    all_results = {"gpu": torch.cuda.get_device_name(0), "mode": args.mode,
                   "date": datetime.now().isoformat(timespec="seconds"), "combos": {}}

    for ctx in args.ctx:
        # Build a prompt of ~ctx tokens by repeating base
        reps = max(1, ctx // len(base))
        prompt_ids = (base * reps)[:ctx]
        actual_ctx = len(prompt_ids)
        for batch in args.batches:
            tag = f"mode={args.mode} ctx={actual_ctx} batch={batch}"
            print(f"\n>>> profiling {tag} ...", flush=True)
            r = profile_combo(backend, prompt_ids, batch, args.steps, args.warmup, args.shapes)
            print_ledger(tag, r)
            all_results["combos"][f"ctx{actual_ctx}_b{batch}"] = r

    outpath = Path(f"benchmarks/fixtures/laguna_moe_node_trace_{args.mode}.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {outpath}", flush=True)


if __name__ == "__main__":
    main()
