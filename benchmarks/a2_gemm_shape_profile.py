#!/usr/bin/env python3
"""A2: 精确 GEMM shape 采集 — hook 进每个 linear 层捕获实际 M/N/K。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_gemm_shape_profile \
        [--concurrency 1] [--rounds 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    import torch

    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    prompt_len = args.prompt_len
    num_slots = max(args.concurrency, 1)
    blocks_per_slot = -(-(prompt_len + 8192) // 16)

    print(f"Loading model (prompt_len={prompt_len}, slots={num_slots})...")
    config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=prompt_len + 8192,
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": K,
            "attention_backend": "CUSTOM",
        },
    )
    runner = DirectModelRunner(
        vllm_config=config,
        num_slots=num_slots,
        blocks_per_slot=blocks_per_slot,
        enable_cudagraph=False,
        enable_prefix_cache=False,
    )
    print("Model loaded.")

    # Hook into the model's linear layers to capture GEMM shapes
    model = runner.model
    gemm_shapes: dict[str, list[tuple]] = defaultdict(list)
    hooks = []

    def make_hook(layer_name: str):
        def hook_fn(module, input_args, output):
            # Capture input shape (M, K_in) and weight shape (N, K_in)
            if isinstance(input_args, tuple) and len(input_args) > 0:
                inp = input_args[0]
                if hasattr(inp, 'shape') and inp.ndim == 2:
                    m, k_in = inp.shape
                    if hasattr(module, 'weight'):
                        w = module.weight
                        if hasattr(w, 'shape'):
                            n = w.shape[0]
                            gemm_shapes[layer_name].append((m, n, k_in))
                    elif hasattr(module, 'weight_packed'):
                        w = module.weight_packed
                        if hasattr(w, 'shape'):
                            n = w.shape[0]
                            k_packed = w.shape[1]
                            gemm_shapes[layer_name].append((m, n, k_packed * 2))  # FP4: 2 vals per byte
        return hook_fn

    # Register hooks on all linear-like modules
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if any(kw in cls_name.lower() for kw in ['linear', 'proj', 'mlp', 'gate', 'quant']):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Also hook at the functional level via torch.mm/matmul
    orig_mm = torch.mm
    orig_matmul = torch.matmul
    mm_shapes = []

    def hooked_mm(a, b, **kw):
        mm_shapes.append(('mm', tuple(a.shape), tuple(b.shape)))
        return orig_mm(a, b, **kw)

    def hooked_matmul(a, b, **kw):
        mm_shapes.append(('matmul', tuple(a.shape), tuple(b.shape)))
        return orig_matmul(a, b, **kw)

    # Run decode steps
    prompt = list(range(1000, 1000 + prompt_len))
    slots = list(range(num_slots))
    prompts = [prompt] * num_slots

    print(f"Prefilling {num_slots} slot(s)...")
    result = runner.mtp_prefill_batch(slots, prompts)
    anchors = {s: result[s]["anchor"] for s in slots}
    drafts = {s: result[s]["draft_tokens"] for s in slots}
    torch.cuda.synchronize()

    # Clear and start capturing
    gemm_shapes.clear()
    mm_shapes.clear()
    torch.mm = hooked_mm
    torch.matmul = hooked_matmul

    print(f"Running {args.rounds} MTP verify rounds with shape capture...")
    for r in range(args.rounds):
        decisions = runner.mtp_verify_and_commit_batch(
            slots,
            {s: anchors[s] for s in slots},
            {s: drafts[s] for s in slots},
        )
        for s in slots:
            anchors[s] = decisions[s]["next_anchor"]
            drafts[s] = decisions[s]["next_draft_tokens"]
    torch.cuda.synchronize()

    # Restore
    torch.mm = orig_mm
    torch.matmul = orig_matmul
    for h in hooks:
        h.remove()

    # Analyze
    print(f"\n{'='*80}")
    print(f"A2: GEMM SHAPE PROFILE (decode, c={num_slots}, K={K})")
    print(f"{'='*80}")

    # Aggregate by (M, N, K) shape
    shape_counts: dict[tuple, int] = defaultdict(int)
    shape_names: dict[tuple, list[str]] = defaultdict(list)

    for layer_name, shapes in gemm_shapes.items():
        for shape in shapes:
            shape_counts[shape] += 1
            if layer_name not in shape_names[shape]:
                shape_names[shape].append(layer_name)

    # Also count torch.mm/matmul calls
    mm_shape_counts: dict[tuple, int] = defaultdict(int)
    for op, a_shape, b_shape in mm_shapes:
        key = (op, a_shape, b_shape)
        mm_shape_counts[key] += 1

    print(f"\nModule-level hooks ({len(shape_counts)} unique shapes):")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'count':>6}  layers")
    print("-" * 70)
    for (m, n, k), count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        names = shape_names[(m, n, k)][:3]
        names_str = ", ".join(n.split(".")[-1] for n in names)
        print(f"{m:>6} {n:>6} {k:>6} {count:>6}  {names_str}")

    if mm_shape_counts:
        print(f"\ntorch.mm/matmul calls ({len(mm_shape_counts)} unique):")
        for (op, a_shape, b_shape), count in sorted(mm_shape_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {op}{a_shape} × {b_shape}: {count}×")

    # Compute theoretical FLOPs per shape
    print("\nTheoretical FLOPs per verify round (2*M*N*K per GEMM):")
    total_flops = 0
    for (m, n, k), count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        flops_per_call = 2 * m * n * k
        flops_per_round = flops_per_call * (count // args.rounds)
        total_flops += flops_per_round
        gflops = flops_per_round / 1e9
        print(f"  [{m:>4}×{n:>5}×{k:>5}] ×{count//args.rounds:>3}/round = {gflops:>8.2f} GFLOP/round")
    print(f"  TOTAL: {total_flops/1e9:.1f} GFLOP/round")

    if args.json:
        result = {
            "concurrency": num_slots,
            "mtp_k": K,
            "prompt_len": prompt_len,
            "rounds": args.rounds,
            "shapes": [
                {"M": m, "N": n, "K": k, "count_per_round": count // args.rounds,
                 "layers": shape_names[(m, n, k)][:5]}
                for (m, n, k), count in sorted(shape_counts.items(), key=lambda x: -x[1])
            ],
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()
