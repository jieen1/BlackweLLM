#!/usr/bin/env python3
"""Laguna MoE backend 扫描 — 三级火箭①。

对每个候选 NvFP4 MoE backend（flashinfer_cutlass 基线 / marlin / flashinfer_trtllm /
vllm_cutlass / flashinfer_cutedsl / humming）测 decode kernel 时间 + GEMM 家族归属，
判定哪个 backend 在 SM120 上把 routed MoE grouped GEMM（现 cutlass 2.63ms @ 63% peak）
打得最狠。不支持的 backend 在加载时报错，被捕获记为 unsupported。

每个 backend 一次加载（~65s），跑 batch 1+4。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_moe_backend_scan \
        --backends flashinfer_cutlass marlin flashinfer_trtllm vllm_cutlass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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

GEMM_RE = re.compile(r"cutlass|gemm|nvjet|sm120|splitk|matmul|scaled_mm|fp4|nvfp4|marlin|Kernel2", re.I)
FAM = [("nvjet", re.compile(r"nvjet", re.I)),
       ("marlin", re.compile(r"marlin", re.I)),
       ("cutlass", re.compile(r"cutlass|GemmUniv|splitKreduce|Kernel2|tensorrt_llm", re.I)),
       ("triton", re.compile(r"triton", re.I)),
       ("cublas", re.compile(r"cublas|gemvx", re.I))]


def fam(name: str) -> str:
    for f, p in FAM:
        if p.search(name):
            return f
    return "other_gemm"


def run_one(backend_name: str, batches: list[int], steps: int, warmup: int) -> dict:
    """Load model with given moe_backend, profile decode for each batch."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    from runtime.compat_vllm import EngineArgs
    from runtime.backends.laguna import LagunaBackend
    from transformers import AutoTokenizer

    args = EngineArgs(
        model=MODEL, max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=True, dtype="bfloat16", disable_log_stats=True,
        async_scheduling=False, moe_backend=backend_name,
    )
    config = args.create_engine_config()

    t0 = time.perf_counter()
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=64)
    load_s = time.perf_counter() - t0

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = tok.encode("Write a detailed explanation of quantum computing:")

    out = {"backend": backend_name, "load_s": round(load_s, 1), "combos": {}}
    for batch in batches:
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
        t1 = time.perf_counter()
        for _ in range(steps):
            step_batch()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t1) * 1000 / steps

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                step_batch()
            torch.cuda.synchronize()
        gemm_ms = 0.0
        fm = defaultdict(float)
        for e in prof.key_averages():
            if e.device_time_total <= 0:
                continue
            if GEMM_RE.search(e.key):
                ms = e.device_time_total / 1000.0 / steps
                gemm_ms += ms
                fm[fam(e.key)] += ms
        out["combos"][f"b{batch}"] = {
            "wall_ms": round(wall, 3),
            "gemm_ms": round(gemm_ms, 3),
            "gemm_family": {f: round(m, 3) for f, m in fm.items()},
        }
        for s in slots:
            backend.reset_slot(s)
        print(f"  [{backend_name}] b{batch}: wall={wall:.2f}ms gemm={gemm_ms:.2f}ms {dict(fm)}", flush=True)
    return out


def _worker(backend_name: str, batches: list[int], steps: int, warmup: int, out_json: str) -> None:
    """Single-backend worker (runs in its own process for clean GPU state)."""
    try:
        r = run_one(backend_name, batches, steps, warmup)
    except Exception as exc:  # noqa: BLE001
        r = {"backend": backend_name, "error": str(exc).split(chr(10))[0][:300]}
    Path(out_json).write_text(json.dumps(r, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+",
                    default=["flashinfer_cutlass", "marlin", "flashinfer_trtllm", "vllm_cutlass"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_be", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--_out", type=str, default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        _worker(args._be, args.batches, args.steps, args.warmup, args._out)
        return

    tmp = Path("benchmarks/fixtures/_scan_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    results = {"date": datetime.now().isoformat(timespec="seconds"), "backends": {}}
    for be in args.backends:
        print(f"\n>>> backend={be} (subprocess)", flush=True)
        out_json = str(tmp / f"{be}.json")
        cmd = [sys.executable, "-m", "benchmarks.laguna_moe_backend_scan",
               "--_worker", "--_be", be, "--_out", out_json,
               "--batches", *[str(b) for b in args.batches],
               "--steps", str(args.steps), "--warmup", str(args.warmup)]
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600)
        tail = "\n".join(proc.stdout.strip().splitlines()[-4:])
        print(tail, flush=True)
        if Path(out_json).exists():
            results["backends"][be] = json.loads(Path(out_json).read_text())
        else:
            err = (proc.stderr.strip().splitlines() or ["no output"])[-1][:200]
            results["backends"][be] = {"error": f"no result; stderr tail: {err}"}
            print(f"  [{be}] FAILED: {err}", flush=True)

    out = Path("benchmarks/fixtures/laguna_moe_backend_scan.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}", flush=True)
    print("\n=== SUMMARY (b1 / b4 gemm_ms) ===", flush=True)
    for be, r in results["backends"].items():
        if "error" in r:
            print(f"  {be:<22} FAIL: {r['error'][:70]}", flush=True)
        else:
            b1 = r["combos"].get("b1", {})
            b4 = r["combos"].get("b4", {})
            print(f"  {be:<22} b1 gemm={b1.get('gemm_ms')}ms wall={b1.get('wall_ms')}ms | b4 gemm={b4.get('gemm_ms')}ms")
            print(f"  {'':<22} b1 fam={b1.get('gemm_family')}")


if __name__ == "__main__":
    main()
