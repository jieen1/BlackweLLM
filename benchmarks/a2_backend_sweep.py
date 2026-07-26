"""A2: Sweep all FlashInfer NVFP4 backends on real model shapes.

Tests: cutlass, b12x, cute-dsl, trtllm, cudnn
Checks: bit-exactness vs cutlass, latency, bandwidth utilization.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# ── shapes from the model (decode M=4, c=1) ──
SHAPES = [
    # (M, N, K, count_per_round, name)
    (4, 34816, 5120, 56, "gate_up_proj"),
    (4, 5120, 17408, 56, "down_proj"),
    (4, 17408, 17408, 8, "down_proj_attn"),
    (4, 6144, 6144, 64, "out_proj"),
    (4, 5120, 5120, 72, "in_proj_qkvz"),
    (4, 96, 5120, 48, "in_proj_ba"),
]

BACKENDS = ["cutlass", "b12x", "cute-dsl", "trtllm", "cudnn"]
WARMUP = 50
ITERS = 300
PEAK_BW_GBPS = 1338.8


def prepare_nvfp4_inputs(M, N, K, device="cuda"):
    """Create NVFP4-quantized inputs matching vLLM's format."""
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
        pad_nvfp4_weight_for_cutlass,
    )

    # Random bf16 activation
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    # Random bf16 weight (N x K for column-major)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    # Global scale
    alpha = torch.ones(1, dtype=torch.float32, device=device)

    # Quantize activation
    x_fp4, x_bs = scaled_fp4_quant(x_bf16, alpha, is_sf_swizzled_layout=True, backend="cutlass")

    # Quantize weight
    w_fp4, w_bs = scaled_fp4_quant(w_bf16, alpha, is_sf_swizzled_layout=True, backend="cutlass")

    # Swizzle + pad weight (as done in process_weights_after_loading)
    w_bs_swizzled = swizzle_blockscale(w_bs)
    w_fp4_padded, pad_cols = pad_nvfp4_weight_for_cutlass(w_fp4)

    return x_fp4, w_fp4_padded, x_bs, w_bs_swizzled, alpha, pad_cols


def run_backend(backend, x_fp4, w_fp4, x_bs, w_bs, alpha, pad_cols, M, N, K):
    """Run a single backend and return (output, p50_ms, p10_ms, p90_ms)."""
    from flashinfer import mm_fp4

    # Prepare inputs per backend requirements
    a = x_fp4.clone()
    b = w_fp4.clone()
    bs_a = x_bs.clone()
    bs_b = w_bs.clone()

    if backend in ("cutlass", "cudnn"):
        bs_a_u8 = bs_a.view(torch.uint8)
        bs_b_u8 = bs_b.view(torch.uint8)
    else:
        bs_a_u8 = bs_a
        bs_b_u8 = bs_b

    use_8x4 = True if backend == "trtllm" and M <= 32 else False

    def run_once():
        return mm_fp4(
            a,
            b.t(),
            bs_a_u8,
            bs_b_u8.t(),
            alpha,
            torch.bfloat16,
            use_8x4_sf_layout=use_8x4,
            backend=backend,
            block_size=64,
            use_nvfp4=True,
        )

    # Warmup
    try:
        for _ in range(WARMUP):
            out = run_once()
        torch.cuda.synchronize()
    except Exception as e:
        return None, None, None, None, str(e)

    # Benchmark
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return (
        out,
        float(np.percentile(times, 50)),
        float(np.percentile(times, 10)),
        float(np.percentile(times, 90)),
        None,
    )


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")
    print()

    results = {}

    for M, N, K, count, name in SHAPES:
        print(f"{'=' * 70}")
        print(f"Shape: {name} ({M}×{N}×{K}) ×{count}/round")
        print(f"{'=' * 70}")

        x_fp4, w_fp4, x_bs, w_bs, alpha, pad_cols = prepare_nvfp4_inputs(M, N, K, device)
        weight_mb = (N * K * 0.5) / 1e6  # FP4 = 0.5 bytes per element

        shape_results = {}
        ref_output = None

        for backend in BACKENDS:
            out, p50, p10, p90, err = run_backend(
                backend, x_fp4, w_fp4, x_bs, w_bs, alpha, pad_cols, M, N, K
            )

            if err:
                print(f"  {backend:12s}: FAILED - {err[:80]}")
                shape_results[backend] = {"error": err[:200]}
                continue

            bw = weight_mb / (p50 / 1000) / 1e3 if p50 > 0 else 0
            bw_pct = bw / PEAK_BW_GBPS * 100

            # Bit-exact check vs cutlass
            bit_exact = None
            max_diff = None
            if ref_output is not None:
                # Slice to actual N (remove padding)
                out_sliced = out[:, :N]
                ref_sliced = ref_output[:, :N]
                bit_exact = torch.equal(out_sliced, ref_sliced)
                max_diff = (out_sliced.float() - ref_sliced.float()).abs().max().item()

            if backend == "cutlass":
                ref_output = out

            status = ""
            if bit_exact is True:
                status = "✅ EXACT"
            elif bit_exact is False:
                status = f"❌ DIFF (max={max_diff:.6f})"
            else:
                status = "📐 REF"

            print(f"  {backend:12s}: p50={p50:.4f}ms  BW={bw:.0f}GB/s ({bw_pct:.1f}%)  {status}")
            shape_results[backend] = {
                "p50_ms": round(p50, 4),
                "p10_ms": round(p10, 4),
                "p90_ms": round(p90, 4),
                "bw_gbps": round(bw, 1),
                "bw_pct_peak": round(bw_pct, 1),
                "bit_exact_vs_cutlass": bit_exact,
                "max_diff": max_diff,
            }

        shape_results["_meta"] = {
            "M": M,
            "N": N,
            "K": K,
            "count": count,
            "name": name,
            "weight_mb": round(weight_mb, 1),
        }
        results[name] = shape_results
        print()

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Weighted ms/round (lower is better)")
    print(f"{'=' * 70}")
    header = f"{'Shape':<20s}"
    for b in BACKENDS:
        header += f" {b:>10s}"
    print(header)
    print("-" * (20 + 11 * len(BACKENDS)))

    totals = {b: 0.0 for b in BACKENDS}
    for M, N, K, count, name in SHAPES:
        row = f"{name:<20s}"
        for b in BACKENDS:
            r = results[name].get(b, {})
            if "p50_ms" in r:
                weighted = r["p50_ms"] * count
                totals[b] += weighted
                row += f" {weighted:>10.3f}"
            else:
                row += f" {'FAIL':>10s}"
        print(row)

    print("-" * (20 + 11 * len(BACKENDS)))
    row = f"{'TOTAL':<20s}"
    for b in BACKENDS:
        row += f" {totals[b]:>10.3f}"
    print(row)

    # Bit-exact summary
    print(f"\n{'=' * 70}")
    print("BIT-EXACT vs CUTLASS")
    print(f"{'=' * 70}")
    for M, N, K, count, name in SHAPES:
        row = f"{name:<20s}"
        for b in BACKENDS:
            r = results[name].get(b, {})
            be = r.get("bit_exact_vs_cutlass")
            if be is True:
                row += f" {'✅':>10s}"
            elif be is False:
                row += f" {'❌':>10s}"
            elif b == "cutlass":
                row += f" {'REF':>10s}"
            else:
                row += f" {'FAIL':>10s}"
        print(row)

    # Save
    out_path = Path("benchmarks/fixtures/a2_backend_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "date": datetime.now().isoformat(timespec="seconds"),
                "gpu": torch.cuda.get_device_name(),
                "warmup": WARMUP,
                "iters": ITERS,
                "peak_bw_gbps": PEAK_BW_GBPS,
                "results": results,
                "totals_weighted_ms": {b: round(totals[b], 3) for b in BACKENDS},
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
