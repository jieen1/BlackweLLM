#!/usr/bin/env python3
"""A2: Autotune — find optimal tile config per (M, N, K) shape.

Sweeps all 4 configs × 3 M values × 6 shapes = 72 combinations.
Outputs a JSON autotune table for the runtime to use.
"""
from __future__ import annotations
import ctypes
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SO = Path(_REPO) / "runtime/kernels/nvfp4_gemm_sm120.so"
WARMUP, ITERS = 30, 200

SHAPES = [
    # Laguna-S-2.1 shapes (N, K, num_heads_or_experts, label)
    (11264, 3072, 72, "laguna_qkv_swa"),      # 47 MoE layers
    (3072, 9216, 72, "laguna_o_proj_swa"),      # 47 MoE layers
    (8192, 3072, 48, "laguna_qkv_full"),        # 12 full-attn layers
    (3072, 6144, 48, "laguna_o_proj_full"),     # 12 full-attn layers
    (12288, 3072, 1, "laguna_dense_gate_up"),   # layer 0 dense MLP
    (3072, 12288, 1, "laguna_dense_down"),      # layer 0 dense MLP
    (1024, 3072, 1, "laguna_shared_gate_up"),   # shared expert
    (3072, 1024, 1, "laguna_shared_down"),      # shared expert
    (100352, 3072, 1, "laguna_lm_head"),        # lm_head
]
M_VALUES = [1, 4, 16]
CONFIGS = {0: "128x128", 1: "128x128p", 2: "256x128p", 3: "128x256p"}


def load_lib():
    lib = ctypes.CDLL(str(SO))
    lib.qsr_nvfp4_gemm.restype = ctypes.c_int
    lib.qsr_nvfp4_gemm.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    return lib


def prepare(M, N, K):
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale, pad_nvfp4_weight_for_cutlass, pad_nvfp4_activation_for_cutlass,
    )
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    w_fp4, w_bs = scaled_fp4_quant(w, alpha, is_sf_swizzled_layout=True, backend="cutlass")
    w_bs = swizzle_blockscale(w_bs)
    w_fp4, pad = pad_nvfp4_weight_for_cutlass(w_fp4)
    x_fp4, x_bs = scaled_fp4_quant(x, alpha, is_sf_swizzled_layout=True, backend="cutlass")
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, pad)
    return x_fp4, w_fp4, x_bs, w_bs, alpha


def bench(lib, cfg, x, w, xbs, wbs, alpha, M, N, K):
    Np = w.shape[0]
    out = torch.empty(M, Np, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.current_stream().cuda_stream
    def run():
        return lib.qsr_nvfp4_gemm(cfg, out.data_ptr(), x.data_ptr(), w.data_ptr(),
                                   xbs.data_ptr(), wbs.data_ptr(), alpha.data_ptr(),
                                   M, Np, K, s)
    for _ in range(WARMUP):
        rc = run()
        if rc != 0: return None, None
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return out, float(np.percentile(ts, 50))


def main():
    lib = load_lib()
    print(f"GPU: {torch.cuda.get_device_name()}")
    table = {}

    for N, K, count, name in SHAPES:
        for M in M_VALUES:
            key = f"M{M}_{name}"
            x, w, xbs, wbs, alpha = prepare(M, N, K)
            results = {}
            ref_out = None
            for cfg_id, cfg_name in CONFIGS.items():
                out, p50 = bench(lib, cfg_id, x, w, xbs, wbs, alpha, M, N, K)
                if out is None:
                    results[cfg_name] = {"error": True}
                    continue
                if ref_out is None:
                    ref_out = out
                exact = torch.equal(out[:, :N], ref_out[:, :N])
                results[cfg_name] = {"p50_ms": round(p50, 4), "exact": exact}
            
            # Find best config
            best_cfg = min(
                ((c, r["p50_ms"]) for c, r in results.items() if "p50_ms" in r),
                key=lambda x: x[1], default=(None, None)
            )
            all_exact = all(r.get("exact", False) for r in results.values() if "p50_ms" in r)
            
            table[key] = {
                "M": M, "N": N, "K": K, "count": count,
                "best_config": best_cfg[0],
                "best_p50_ms": best_cfg[1],
                "all_exact": all_exact,
                "configs": results,
            }
            print(f"  {key}: best={best_cfg[0]} ({best_cfg[1]:.4f}ms) exact={all_exact}")

    # Save
    out_path = Path("benchmarks/fixtures/a2_autotune_table.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"),
                    "gpu": torch.cuda.get_device_name(), "table": table}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("AUTOTUNE SUMMARY")
    print(f"{'='*60}")
    for M in M_VALUES:
        total = {c: 0.0 for c in CONFIGS.values()}
        for N, K, count, name in SHAPES:
            key = f"M{M}_{name}"
            for cfg_name, r in table[key]["configs"].items():
                if "p50_ms" in r:
                    total[cfg_name] += r["p50_ms"] * count
        best = min(total, key=total.get)
        print(f"  M={M}: best overall = {best} ({total[best]:.1f} ms/round)")
        for c, t in sorted(total.items(), key=lambda x: x[1]):
            print(f"    {c}: {t:.1f} ms")


if __name__ == "__main__":
    main()
