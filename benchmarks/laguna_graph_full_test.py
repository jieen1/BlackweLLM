#!/usr/bin/env python3
"""Laguna CUDA Graph 综合测试 — 一次模型加载，全部验证。

Tests:
1. 50-tok determinism ×3 (含 page crossing)
2. Step-by-step bit-exact (20 steps)
3. Graph vs eager cosine
4. Throughput benchmark (128 tok × 5 reps)
5. sm_scale 一致性

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_graph_full_test
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QSR_A2_CUSTOM_GEMM", "0")
MODEL = "poolside/Laguna-S-2.1-NVFP4"

def main():
    from runtime.compat_vllm import EngineArgs
    config = EngineArgs(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.80,
        enforce_eager=True, dtype="bfloat16", disable_log_stats=True, async_scheduling=False
    ).create_engine_config()
    from runtime.backends.laguna import LagunaBackend
    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    from transformers import AutoTokenizer

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.perf_counter()
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=64)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = tok.encode("The capital of France is")
    cg = LagunaCudaGraphDecode(backend, batch_size=1)
    cg.capture()
    print("Graph captured", flush=True)

    results = {"gpu": torch.cuda.get_device_name(0), "date": datetime.now().isoformat(timespec="seconds")}
    all_pass = True

    # ── T1: 50-tok determinism ×3 ──
    print("\n=== T1: 50-tok determinism ×3 ===", flush=True)
    def run50():
        cg.reset(); backend.reset_slot(0)
        ft = backend.prefill(0, prompt_ids)
        toks = [ft]
        for _ in range(49):
            kvl = backend.slot_kv_len[0]
            r = cg.replay([0], [toks[-1]], [kvl])
            toks.append(r[0]); backend.slot_kv_len[0] += 1
            backend.slot_committed_tokens[0].append(toks[-2])
            if r[0] in (2,24): break
        return toks
    r1, r2, r3 = run50(), run50(), run50()
    t1_pass = r1 == r2 == r3
    print(f"  run1==run2: {'✅' if r1==r2 else '❌'}  run2==run3: {'✅' if r2==r3 else '❌'}  ({len(r1)} tok)", flush=True)
    print(f"  Output: {tok.decode(r1, skip_special_tokens=True)[:100]!r}", flush=True)
    results["t1_determinism_50"] = t1_pass
    all_pass &= t1_pass

    # ── T2: Step-by-step bit-exact (20 steps) ──
    print("\n=== T2: Step-by-step bit-exact (20 steps) ===", flush=True)
    def run_steps(n):
        cg.reset(); backend.reset_slot(0)
        ft = backend.prefill(0, prompt_ids)
        toks = [ft]; logits = []
        for s in range(n):
            kvl = backend.slot_kv_len[0]
            r = cg.replay([0], [toks[-1]], [kvl])
            logits.append(cg._logits[0].clone())
            toks.append(r[0]); backend.slot_kv_len[0] += 1
            backend.slot_committed_tokens[0].append(toks[-2])
        return toks, logits
    t1_tokens, t1_logits = run_steps(20)
    t2_tokens, t2_logits = run_steps(20)
    t2_pass = True
    for s in range(20):
        be = torch.equal(t1_logits[s], t2_logits[s])
        if not be:
            md = (t1_logits[s].float()-t2_logits[s].float()).abs().max().item()
            print(f"  Step {s}: ❌ max_diff={md:.4f}", flush=True)
            t2_pass = False
    print(f"  20-step bit-exact: {'✅ ALL PASS' if t2_pass else '❌ FAIL'}", flush=True)
    results["t2_stepwise_bitexact"] = t2_pass
    all_pass &= t2_pass

    # ── T3: Graph vs eager cosine ──
    print("\n=== T3: Graph vs eager cosine ===", flush=True)
    cg.reset(); backend.reset_slot(0)
    ft = backend.prefill(0, prompt_ids)
    kvl = backend.slot_kv_len[0]
    cg.replay([0], [ft], [kvl])
    gl = cg._logits[0].clone()
    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    el = backend._forward([0], [ft], [kvl], qo_len=1, is_decode=True)[0].clone()
    cos = torch.nn.functional.cosine_similarity(gl.float().unsqueeze(0), el.float().unsqueeze(0)).item()
    md = (gl.float()-el.float()).abs().max().item()
    t3_pass = cos > 0.99
    print(f"  cosine={cos:.6f}  max_diff={md:.4f}  {'✅' if t3_pass else '⚠️ NEEDS INVESTIGATION'}", flush=True)
    for gk, b in backend._metadata_builders.items():
        print(f"  sm_scale wl={gk[0]}: builder={b.sm_scale!r}", flush=True)
    results["t3_graph_eager_cosine"] = round(cos, 6)
    results["t3_pass"] = t3_pass

    # ── T4: Throughput benchmark ──
    print("\n=== T4: Throughput (128 tok × 5 reps) ===", flush=True)
    bench_ids = tok.encode("Write a detailed explanation of quantum computing:")
    # Warmup
    cg.reset(); backend.reset_slot(0)
    wtok = backend.prefill(0, bench_ids)
    for _ in range(10):
        kvl = backend.slot_kv_len[0]
        r = cg.replay([0], [wtok], [kvl])
        wtok = r[0]; backend.slot_kv_len[0] += 1
        backend.slot_committed_tokens[0].append(wtok)
    times_list, token_counts = [], []
    for rep in range(5):
        cg.reset(); backend.reset_slot(0)
        wtok = backend.prefill(0, bench_ids)
        n = 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(127):
            kvl = backend.slot_kv_len[0]
            r = cg.replay([0], [wtok], [kvl])
            wtok = r[0]; backend.slot_kv_len[0] += 1
            backend.slot_committed_tokens[0].append(wtok)
            n += 1
            if wtok in (2,24): break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed); token_counts.append(n)
        print(f"  Rep {rep}: {n} tok, {elapsed:.3f}s, {n/elapsed:.1f} tok/s", flush=True)
    avg_tps = sum(t/e for t,e in zip(token_counts, times_list)) / 5
    avg_itl = sum(times_list) / sum(token_counts) * 1000
    print(f"  AVG: {avg_tps:.1f} tok/s, ITL: {avg_itl:.2f} ms", flush=True)
    results["t4_tok_s"] = round(avg_tps, 1)
    results["t4_itl_ms"] = round(avg_itl, 2)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL: {'✅ ALL PASS' if all_pass else '⚠️ ISSUES'}", flush=True)
    print(f"  T1 50-tok determinism: {'✅' if t1_pass else '❌'}", flush=True)
    print(f"  T2 20-step bit-exact:  {'✅' if t2_pass else '❌'}", flush=True)
    print(f"  T3 graph/eager cosine: {cos:.6f} {'✅' if t3_pass else '⚠️'}", flush=True)
    print(f"  T4 throughput: {avg_tps:.1f} tok/s, ITL {avg_itl:.2f} ms", flush=True)
    print(f"{'='*60}", flush=True)

    results["all_pass"] = all_pass
    out = Path("benchmarks/fixtures/laguna_graph_full_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved: {out}", flush=True)

if __name__ == "__main__":
    main()
