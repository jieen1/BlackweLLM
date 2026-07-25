#!/usr/bin/env python3
"""Unified benchmark: BlackweLLM vs vLLM — prefix cache + DFlash + CG.

Test pattern (per context length):
  1. Warmup: send base prompt (e.g. 64K) → full prefill, populates prefix cache
  2. Test:   send base prompt + 10K suffix → prefix cache HIT, only prefill 10K

Each runtime runs in a separate subprocess to avoid GPU memory leaks.
All parameters identical for fair comparison.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.unified_comparison
"""
from __future__ import annotations
import gc, json, os, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/")
DRAFT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/"
    "snapshots/723794750422b3efbf3a7b3af76dffb4ba035943/")
OUT = Path(_REPO) / "benchmarks" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared parameters ──
CTX_LIST    = [65536, 131072, 200000]
SUFFIX_LEN  = 10240   # 10K suffix for prefix cache test
MAX_NEW     = 128
GPU_UTIL    = 0.92
DTYPE       = "bfloat16"
SPEC_TOKENS = 15
TEMP        = 0.0
PYTHON      = "/home/bot/.venvs/vllm/bin/python"

def gpu_mem_gib():
    try:
        o = subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
            text=True)
        return float(o.strip().split("\n")[0]) / 1024
    except: return -1.0

def make_prompt(tok, n):
    base = ("The quick brown fox jumps over the lazy dog. "
            "In a world of artificial intelligence and machine learning, "
            "the importance of efficient inference cannot be overstated. "
            "Modern language models require careful optimization of memory "
            "bandwidth and compute utilization to achieve real-time performance. "
            "Speculative decoding offers a promising approach by using a smaller "
            "draft model to propose multiple tokens in parallel, which are then "
            "verified by the larger target model in a single forward pass. ")
    chunk = tok.encode(base, add_special_tokens=False)
    ids = []
    while len(ids) < n:
        ids.extend(chunk)
    return ids[:n]

# ── BlackweLLM (runs in-process) ─────────────────────────────────────────
def bench_ours(base_len, suffix_len, base_ids, suffix_ids, tok):
    import torch; torch.set_grad_enabled(False)
    os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
    from runtime.compat_vllm import EngineArgs
    total_len = base_len + suffix_len
    max_len = max(total_len + 1024, 262144)
    bps = (total_len + 15) // 16 + 512
    vc = EngineArgs(model=MODEL, dtype=DTYPE, max_model_len=max_len,
        gpu_memory_utilization=GPU_UTIL, enforce_eager=True,
        trust_remote_code=True).create_engine_config()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vc, num_slots=1, block_size=16, blocks_per_slot=bps)
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)

    # Small warmup for JIT
    backend.reset_slot(0)
    engine.generate(base_ids[:256], max_tokens=5, temperature=0.0,
                    enable_prefix_cache=False)
    torch.cuda.synchronize()

    full_ids = base_ids + suffix_ids
    results = {}

    # Phase 1: Warmup — full prefill of base prompt (populates prefix cache)
    print(f"  [warmup] Full prefill {base_len} tokens...")
    backend.reset_slot(0)
    t0 = time.perf_counter()
    tok_w, st_w = engine.generate(base_ids, max_tokens=MAX_NEW, temperature=TEMP,
                                   slot=0, enable_prefix_cache=False)
    torch.cuda.synchronize()
    t_warmup = time.perf_counter() - t0
    mem_w = gpu_mem_gib()
    results["warmup"] = dict(
        phase="warmup", prompt_len=base_len,
        n_tokens=len(tok_w), total_s=round(t_warmup,2),
        prefill_ms=round(st_w.get("prefill_ms",0),1),
        decode_ms=round(st_w.get("decode_ms",0),1),
        accept=round(st_w.get("acceptance_rate",0),3),
        tok_step=round(st_w.get("tokens_per_step",0),2),
        tok_s=round(st_w.get("tok_per_s",0),1),
        mem_gib=round(mem_w,1))
    print(f"    {t_warmup:.1f}s  prefill={st_w['prefill_ms']/1000:.1f}s  "
          f"decode={st_w['tok_per_s']:.0f}tok/s  accept={st_w['acceptance_rate']:.0%}  "
          f"mem={mem_w:.1f}GiB")

    # Phase 2: Prefix cache test — same base + 10K suffix
    print(f"  [test] Prefix-cached: {base_len} cached + {suffix_len} suffix...")
    t0 = time.perf_counter()
    tok_t, st_t = engine.generate(full_ids, max_tokens=MAX_NEW, temperature=TEMP,
                                   slot=0, enable_prefix_cache=True)
    torch.cuda.synchronize()
    t_test = time.perf_counter() - t0
    mem_t = gpu_mem_gib()
    results["prefix_cached"] = dict(
        phase="prefix_cached", prompt_len=len(full_ids),
        cached_len=base_len, suffix_len=suffix_len,
        n_tokens=len(tok_t), total_s=round(t_test,2),
        prefill_ms=round(st_t.get("prefill_ms",0),1),
        decode_ms=round(st_t.get("decode_ms",0),1),
        accept=round(st_t.get("acceptance_rate",0),3),
        tok_step=round(st_t.get("tokens_per_step",0),2),
        tok_s=round(st_t.get("tok_per_s",0),1),
        mem_gib=round(mem_t,1),
        preview=tok.decode(tok_t[:50])[:80])
    print(f"    {t_test:.1f}s  prefill={st_t['prefill_ms']/1000:.1f}s  "
          f"decode={st_t['tok_per_s']:.0f}tok/s  accept={st_t['acceptance_rate']:.0%}  "
          f"mem={mem_t:.1f}GiB")

    # Phase 3: Cold baseline — full prefill of full prompt (no cache)
    print(f"  [cold] Full prefill {len(full_ids)} tokens (no cache)...")
    backend.reset_slot(0)
    t0 = time.perf_counter()
    tok_c, st_c = engine.generate(full_ids, max_tokens=MAX_NEW, temperature=TEMP,
                                   slot=0, enable_prefix_cache=False)
    torch.cuda.synchronize()
    t_cold = time.perf_counter() - t0
    mem_c = gpu_mem_gib()
    results["cold"] = dict(
        phase="cold", prompt_len=len(full_ids),
        n_tokens=len(tok_c), total_s=round(t_cold,2),
        prefill_ms=round(st_c.get("prefill_ms",0),1),
        decode_ms=round(st_c.get("decode_ms",0),1),
        accept=round(st_c.get("acceptance_rate",0),3),
        tok_step=round(st_c.get("tokens_per_step",0),2),
        tok_s=round(st_c.get("tok_per_s",0),1),
        mem_gib=round(mem_c,1))
    print(f"    {t_cold:.1f}s  prefill={st_c['prefill_ms']/1000:.1f}s  "
          f"decode={st_c['tok_per_s']:.0f}tok/s  accept={st_c['acceptance_rate']:.0%}  "
          f"mem={mem_c:.1f}GiB")

    del engine, backend; gc.collect(); torch.cuda.empty_cache()
    return dict(runtime="blackwellm", base_ctx=base_len, suffix=suffix_len, **results)

# ── vLLM (separate subprocess) ───────────────────────────────────────────
VLLM_SCRIPT = r'''
import json, os, sys, time, gc
os.environ["USE_LIBUV"]="0"; os.environ["HF_HUB_OFFLINE"]="1"
import torch; torch.set_grad_enabled(False)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = sys.argv[1]; DRAFT = sys.argv[2]
base_len = int(sys.argv[3]); suffix_len = int(sys.argv[4])
max_new = int(sys.argv[5]); gpu_util = float(sys.argv[6])

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_text = ("The quick brown fox jumps over the lazy dog. "
    "In a world of artificial intelligence and machine learning, "
    "the importance of efficient inference cannot be overstated. "
    "Modern language models require careful optimization of memory "
    "bandwidth and compute utilization to achieve real-time performance. "
    "Speculative decoding offers a promising approach by using a smaller "
    "draft model to propose multiple tokens in parallel, which are then "
    "verified by the larger target model in a single forward pass. ")
chunk = tok.encode(base_text, add_special_tokens=False)
base_ids = []
while len(base_ids) < base_len: base_ids.extend(chunk)
base_ids = base_ids[:base_len]
suffix_ids = []
while len(suffix_ids) < suffix_len: suffix_ids.extend(chunk)
suffix_ids = suffix_ids[:suffix_len]
full_ids = base_ids + suffix_ids
base_prompt = tok.decode(base_ids)
full_prompt = tok.decode(full_ids)

max_len = max(len(full_ids) + 1024, 262144)
t0 = time.perf_counter()
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=max_len,
          gpu_memory_utilization=gpu_util, trust_remote_code=True,
          enforce_eager=False, enable_prefix_caching=True,
          speculative_config=dict(method="dflash",
              num_speculative_tokens=15, model=DRAFT))
init_s = time.perf_counter() - t0
print(f"  vLLM init: {init_s:.0f}s", file=sys.stderr)
sp = SamplingParams(temperature=0.0, max_tokens=max_new)

# Warmup
llm.generate(["Hello"], sp)

import subprocess
def mem():
    try:
        o = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], text=True)
        return round(float(o.strip().split("\n")[0])/1024, 1)
    except: return -1

# Phase 1: warmup with base prompt
t0 = time.perf_counter()
o1 = llm.generate([base_prompt], sp)
torch.cuda.synchronize()
t1 = time.perf_counter() - t0
n1 = len(o1[0].outputs[0].token_ids)

# Phase 2: prefix-cached (same base + suffix)
t0 = time.perf_counter()
o2 = llm.generate([full_prompt], sp)
torch.cuda.synchronize()
t2 = time.perf_counter() - t0
n2 = len(o2[0].outputs[0].token_ids)

# Phase 3: cold (reset prefix cache by generating something else first)
llm.generate(["Completely different prompt to evict cache. " * 100], sp)
t0 = time.perf_counter()
o3 = llm.generate([full_prompt], sp)
torch.cuda.synchronize()
t3 = time.perf_counter() - t0
n3 = len(o3[0].outputs[0].token_ids)

result = dict(runtime="vllm", base_ctx=base_len, suffix=suffix_len,
    init_s=round(init_s,1),
    warmup=dict(phase="warmup", prompt_len=base_len, n_tokens=n1,
        total_s=round(t1,2), tok_s=round(n1/max(t1,0.01),1), mem_gib=mem()),
    prefix_cached=dict(phase="prefix_cached", prompt_len=len(full_ids),
        cached_len=base_len, suffix_len=suffix_len, n_tokens=n2,
        total_s=round(t2,2), tok_s=round(n2/max(t2,0.01),1), mem_gib=mem(),
        preview=o2[0].outputs[0].text[:80]),
    cold=dict(phase="cold", prompt_len=len(full_ids), n_tokens=n3,
        total_s=round(t3,2), tok_s=round(n3/max(t3,0.01),1), mem_gib=mem()))
print(json.dumps(result))
'''

def bench_vllm_subprocess(base_len, suffix_len):
    """Run vLLM benchmark in a separate process."""
    script = Path("/tmp/_vllm_bench.py")
    script.write_text(VLLM_SCRIPT)
    cmd = [PYTHON, str(script), MODEL, DRAFT,
           str(base_len), str(suffix_len), str(MAX_NEW), str(GPU_UTIL)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                          env={**os.environ, "USE_LIBUV": "0", "HF_HUB_OFFLINE": "1"})
    if proc.returncode != 0:
        print(f"  vLLM stderr (last 500): {proc.stderr[-500:]}")
        raise RuntimeError(f"vLLM exited {proc.returncode}")
    # Last line of stdout is JSON
    lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
    return json.loads(lines[-1])

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    all_results = dict(
        timestamp=datetime.now().isoformat(),
        gpu="RTX PRO 6000 Blackwell 96GB (SM120)",
        model="poolside/Laguna-S-2.1-NVFP4",
        draft="poolside/Laguna-S-2.1-DFlash-NVFP4 (K=15)",
        params=dict(max_new=MAX_NEW, suffix_len=SUFFIX_LEN, gpu_util=GPU_UTIL,
                    dtype=DTYPE, spec_tokens=SPEC_TOKENS, temperature=TEMP),
        benchmarks=[])

    for ctx in CTX_LIST:
        print(f"\n{'='*70}")
        print(f"  Base context: {ctx//1024}K + {SUFFIX_LEN//1024}K suffix")
        print(f"{'='*70}")

        base_ids = make_prompt(tok, ctx)
        suffix_ids = make_prompt(tok, SUFFIX_LEN)
        # Make suffix different from base
        suffix_ids = [(t + 50000) % 100352 for t in suffix_ids]
        print(f"  Base: {len(base_ids)} tok, Suffix: {len(suffix_ids)} tok")

        # BlackweLLM
        print(f"\n[BlackweLLM] DFlash + CG + prefix cache")
        try:
            r = bench_ours(len(base_ids), len(suffix_ids), base_ids, suffix_ids, tok)
            all_results["benchmarks"].append(r)
        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc()
            all_results["benchmarks"].append(
                dict(runtime="blackwellm", base_ctx=ctx, error=str(e)))
        gc.collect()
        import torch; torch.cuda.empty_cache()
        time.sleep(5)

        # vLLM (separate process)
        print(f"\n[vLLM] DFlash + CG + auto kernel + prefix caching")
        try:
            r = bench_vllm_subprocess(len(base_ids), len(suffix_ids))
            all_results["benchmarks"].append(r)
            for phase in ["warmup", "prefix_cached", "cold"]:
                s = r.get(phase, {})
                print(f"  {phase}: {s.get('total_s',0):.1f}s  "
                      f"{s.get('tok_s',0):.0f}tok/s  mem={s.get('mem_gib',0):.0f}GiB")
        except Exception as e:
            print(f"  FAILED: {e}"); traceback.print_exc()
            all_results["benchmarks"].append(
                dict(runtime="vllm", base_ctx=ctx, error=str(e)))
        time.sleep(5)

    # Save
    p = OUT / "unified_comparison.json"
    with open(p, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {p}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'SUMMARY — Prefix Cache + DFlash + CUDA Graph':^100}")
    print(f"{'='*100}")
    print(f"{'Base':>6} | {'Runtime':>12} | {'Phase':>14} | {'Prompt':>7} | "
          f"{'Total':>7} | {'Prefill':>8} | {'Tok/s':>7} | {'Accept':>6} | {'Mem':>7}")
    print("-"*100)
    for b in all_results["benchmarks"]:
        if "error" in b:
            print(f"{b['base_ctx']//1024:>5}K | {b['runtime']:>12} | "
                  f"{'ERROR':>14} | {b['error'][:50]}")
            continue
        for phase in ["warmup", "prefix_cached", "cold"]:
            s = b.get(phase, {})
            if not s: continue
            base_k = f"{b['base_ctx']//1024}K"
            rt = b["runtime"][:12]
            pl = f"{s.get('prompt_len',0)//1024}K"
            tot = f"{s.get('total_s',0):.1f}s"
            pf = f"{s.get('prefill_ms',0)/1000:.1f}s" if "prefill_ms" in s else "-"
            tps = f"{s.get('tok_s',0):.0f}"
            acc = f"{s.get('accept',0):.0%}" if "accept" in s else "-"
            mem = f"{s.get('mem_gib',0):.0f}GiB"
            print(f"{base_k:>6} | {rt:>12} | {phase:>14} | {pl:>7} | "
                  f"{tot:>7} | {pf:>8} | {tps:>7} | {acc:>6} | {mem:>7}")
        print("-"*100)

if __name__ == "__main__":
    main()
