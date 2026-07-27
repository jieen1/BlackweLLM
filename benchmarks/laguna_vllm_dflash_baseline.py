#!/usr/bin/env python3
"""Laguna-S-2.1-NVFP4 · vLLM 生产基线（DFlash + CUDA Graph + 最优 kernel）。

阶段一（用户裁定 2026-07-23）：在 stock vLLM 里把 DFlash 投机 + CUDA Graph +
最优 MoE kernel 的服务跑起来，测出基准速度——这是后续自研 kernel 的对标尺，
也是发布门禁②「vs stock vLLM 显著优势」的对照基线。

配置（model card 推荐）：
  - target: poolside/Laguna-S-2.1-NVFP4
  - draft:  poolside/Laguna-S-2.1-DFlash-NVFP4 (method=dflash, K=15)
  - CUDA Graph: enforce_eager=False（默认开）
  - 官方 DFlash 接受率 2.9-3.1 tok/step（GB10）

测 accepted tokens/s（含投机）+ ITL，greedy 保证可复现。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_vllm_dflash_baseline \
        --moe-backends auto --ctx 4096 --num-seqs 1 4
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MAX_JOBS", "4")

MODEL = "poolside/Laguna-S-2.1-NVFP4"
DRAFT = "poolside/Laguna-S-2.1-DFlash-NVFP4"


def gpu_mem_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return float(out.strip().split("\n")[0])
    except Exception:
        return -1.0


def make_prompt(tokenizer, target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog near the river bank. "
    words = (base * (target_tokens // 10 + 2)).split()
    lo, hi, best = 1, len(words), len(words)
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(tokenizer.encode(" ".join(words[:mid]))) < target_tokens:
            lo = mid + 1
        else:
            best = mid
            hi = mid - 1
    result = " ".join(words[:best])
    while len(tokenizer.encode(result)) < target_tokens and best < len(words):
        result += " " + words[best]
        best += 1
    return result


def build_llm(moe_backend: str, max_model_len: int, num_seqs: int, k: int,
              dflash: bool = True, cuda_graph: bool = True,
              gpu_mem_util: float = 0.85, log_stats: bool = False):
    from vllm import LLM
    kwargs = dict(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=not cuda_graph,
        dtype="bfloat16",
        disable_log_stats=not log_stats,
        max_num_seqs=num_seqs,
    )
    if dflash:
        kwargs["speculative_config"] = {
            "method": "dflash", "model": DRAFT, "num_speculative_tokens": k}
    if moe_backend != "auto":
        kwargs["moe_backend"] = moe_backend
    return LLM(**kwargs)


def _spec_decode_counter(llm, substr: str) -> float:
    """Sum a Prometheus Counter metric (matched by substring) across labels.

    Counters accumulate every engine step regardless of the periodic
    logging interval (that interval only throttles the pretty-printed
    log line), so this is accurate even for sub-second generate() calls.
    """
    total = 0.0
    for m in llm.llm_engine.get_metrics():
        if substr in m.name and hasattr(m, "value"):
            total += m.value
    return total


def measure(llm, prompts: list[str], max_tokens: int, num_seqs: int,
            log_stats: bool = False) -> dict:
    """Measure accepted tok/s for a batch of prompts (concurrent=num_seqs)."""
    from vllm import SamplingParams
    import torch

    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    # warmup
    llm.generate(prompts[:1], params)

    if log_stats:
        draft_before = _spec_decode_counter(llm, "spec_decode_num_draft_tokens")
        accepted_before = _spec_decode_counter(llm, "spec_decode_num_accepted_tokens")

    reps = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        reps.append((elapsed, n_out))

    elapsed = min(r[0] for r in reps)  # best of 3
    n_out = reps[[r[0] for r in reps].index(elapsed)][1]
    tps = n_out / elapsed
    itl_ms = elapsed / max(n_out // len(prompts), 1) * 1000
    result = {
        "wall_s": round(elapsed, 3),
        "total_out_tokens": n_out,
        "accepted_tok_s": round(tps, 1),
        "itl_ms_per_seq": round(itl_ms, 2),
        "num_seqs": num_seqs,
    }
    if log_stats:
        draft_after = _spec_decode_counter(llm, "spec_decode_num_draft_tokens")
        accepted_after = _spec_decode_counter(llm, "spec_decode_num_accepted_tokens")
        d_draft = draft_after - draft_before
        d_accept = accepted_after - accepted_before
        result["spec_decode_draft_tokens_delta"] = d_draft
        result["spec_decode_accepted_tokens_delta"] = d_accept
        result["acceptance_rate"] = round(d_accept / d_draft, 4) if d_draft else None
    return result


def run_config(moe_backend: str, ctx_list: list[int], num_seqs_list: list[int],
               max_tokens: int, k: int, max_model_len: int,
               dflash: bool = True, cuda_graph: bool = True,
               gpu_mem_util: float = 0.85, log_stats: bool = False) -> dict:
    print(f"\n{'='*70}\n>>> moe_backend={moe_backend}  "
          f"(DFlash={'K'+str(k) if dflash else 'OFF'}, CUDA Graph={'ON' if cuda_graph else 'OFF'})",
          flush=True)
    t0 = time.perf_counter()
    try:
        llm = build_llm(moe_backend, max_model_len, max(num_seqs_list), k, dflash,
                        cuda_graph, gpu_mem_util, log_stats=log_stats)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).split("\n")[0][:250]
        print(f"  LOAD FAILED: {msg}", flush=True)
        return {"moe_backend": moe_backend, "error": msg}
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.1f}s, mem={gpu_mem_mib():.0f}MiB", flush=True)

    tok = llm.get_tokenizer()
    res = {"moe_backend": moe_backend, "dflash": dflash, "cuda_graph": cuda_graph,
           "load_s": round(load_s, 1), "ctx": {}}
    for ctx in ctx_list:
        prompt = make_prompt(tok, ctx)
        actual_ctx = len(tok.encode(prompt))
        print(f"  --- ctx actual={actual_ctx} tokens ---", flush=True)
        ctx_res = {}
        for ns in num_seqs_list:
            prompts = [prompt] * ns
            r = measure(llm, prompts, max_tokens, ns, log_stats=log_stats)
            ctx_res[f"seqs{ns}"] = r
            extra = f"  accept_rate={r['acceptance_rate']}" if log_stats else ""
            print(f"  ctx{actual_ctx} seqs={ns}: accepted={r['accepted_tok_s']} tok/s  "
                  f"ITL={r['itl_ms_per_seq']}ms  out={r['total_out_tokens']}tok/{r['wall_s']}s{extra}", flush=True)
        res["ctx"][f"ctx{actual_ctx}"] = ctx_res

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    time.sleep(3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-backends", nargs="+", default=["auto"])
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096])
    ap.add_argument("--num-seqs", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--no-dflash", action="store_true", help="disable DFlash speculation")
    ap.add_argument("--enforce-eager", action="store_true", help="disable CUDA Graph")
    ap.add_argument("--log-stats", action="store_true",
                    help="enable vLLM stats and report spec-decode acceptance_rate "
                         "(reads Prometheus counters, unaffected by log print interval)")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--out", type=str,
                    default="benchmarks/fixtures/laguna_vllm_dflash_baseline.json")
    args = ap.parse_args()

    results = {"date": datetime.now().isoformat(timespec="seconds"),
               "model": MODEL, "draft": DRAFT, "k": args.k,
               "label": args.label, "dflash": not args.no_dflash,
               "cuda_graph": not args.enforce_eager, "configs": []}
    for be in args.moe_backends:
        r = run_config(be, args.ctx, args.num_seqs, args.max_tokens, args.k, args.max_model_len,
                   dflash=not args.no_dflash, cuda_graph=not args.enforce_eager,
                   gpu_mem_util=args.gpu_mem_util, log_stats=args.log_stats)
        results["configs"].append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*70}\nSaved: {out}", flush=True)
    print("\n=== SUMMARY (accepted tok/s) ===", flush=True)
    for c in results["configs"]:
        if "error" in c:
            print(f"  {c['moe_backend']:<20} FAIL: {c['error'][:60]}", flush=True)
        else:
            for ctx_key, ctx_res in c["ctx"].items():
                for key, r in ctx_res.items():
                    print(f"  {c['moe_backend']:<16} {ctx_key} {key}: "
                          f"{r['accepted_tok_s']} tok/s  ITL={r['itl_ms_per_seq']}ms", flush=True)


if __name__ == "__main__":
    main()
