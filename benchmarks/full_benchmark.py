#!/usr/bin/env python3
"""Full benchmark: BlackForge runtime — 64K/128K/200K × prefix-cache + CG × 3 rounds.

Each context runs in a separate subprocess to avoid GPU memory leaks.
"""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = Path(_REPO) / "benchmarks" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)
PYTHON = "/home/bot/.venvs/vllm/bin/python"

CTX_LIST = [65536, 131072, 200000]
N_ROUNDS = 3

# Worker script (runs in subprocess)
WORKER = '''
import gc, json, os, sys, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE", "1")
sys.path.insert(0, "{repo}")
import torch; torch.set_grad_enabled(False)

CTX = int(sys.argv[1])
N_ROUNDS = int(sys.argv[2])
SUFFIX_LEN = 10240
MAX_NEW = 128
GPU_UTIL = 0.92
BLOCK_SIZE = 64

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/")

def make_prompt(n):
    base = ("The quick brown fox jumps over the lazy dog. "
            "In a world of artificial intelligence and machine learning, "
            "the importance of efficient inference cannot be overstated. "
            "Modern language models require careful optimization of memory "
            "bandwidth and compute utilization to achieve real-time performance. "
            "Speculative decoding offers a promising approach by using a smaller "
            "draft model to propose multiple tokens in parallel, which are then "
            "verified by the larger target model in a single forward pass. ")
    # Simple tokenization: use repeating int IDs (deterministic, no tokenizer needed)
    chunk = list(range(1000, 1100))  # 100 token chunk
    ids = []
    while len(ids) < n:
        ids.extend(chunk)
    return ids[:n]

def gpu_mem_gib():
    import subprocess as sp
    try:
        o = sp.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"], text=True)
        return float(o.strip().split("\\n")[0]) / 1024
    except: return -1.0

from runtime.compat_vllm import EngineArgs
from runtime.backends.laguna import LagunaBackend

base_len = CTX
total_len = base_len + SUFFIX_LEN
max_model_len = total_len + 2048
blocks_per_slot = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 64

engine_args = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=max_model_len,
    gpu_memory_utilization=GPU_UTIL, enforce_eager=True, trust_remote_code=True)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(vllm_config, num_slots=1, block_size=BLOCK_SIZE, blocks_per_slot=blocks_per_slot)

mem_load = gpu_mem_gib()
base_ids = make_prompt(base_len)
suffix_ids = [x + 50000 for x in make_prompt(SUFFIX_LEN)]

# Warmup prefill (base only)
t0 = time.time()
first = backend.prefill(0, base_ids)
warmup_s = time.time() - t0

# Capture CG after warmup
backend._ensure_decode_cg()
cg = backend._decode_cg
kv = backend.slot_kv_len[0]
tok_id = first
for _ in range(20):
    r = cg.replay([0], [tok_id], [kv]); tok_id = r[0]; kv += 1

# Test rounds
rounds = []
for ri in range(N_ROUNDS):
    backend.reset_slot(0)
    # Unpatch CG impls for prefill
    backend._unpatch_impls_for_prefill()

    full_prompt = base_ids + suffix_ids
    torch.cuda.synchronize()
    t0 = time.time()
    first_token = backend.prefill(0, full_prompt)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0

    # Re-patch for decode (CG already captured, just need impls)
    backend._repatch_impls_for_cg()

    kv = backend.slot_kv_len[0]
    tok_id = first_token
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(MAX_NEW - 1):
        r = cg.replay([0], [tok_id], [kv]); tok_id = r[0]; kv += 1
    torch.cuda.synchronize()
    decode_s = time.time() - t0

    step_ms = decode_s / (MAX_NEW - 1) * 1000
    tok_s = (MAX_NEW - 1) / decode_s
    rounds.append({{"round": ri+1, "prefill_s": round(prefill_s,3),
        "step_ms": round(step_ms,2), "tok_s": round(tok_s,1), "mem_gib": round(gpu_mem_gib(),1)}})

result = {{"status": "OK", "ctx": CTX, "total_len": total_len,
    "warmup_prefill_s": round(warmup_s,2), "mem_load_gib": round(mem_load,1), "rounds": rounds}}
print("RESULT_JSON:" + json.dumps(result))
'''

def main():
    results = {
        "date": datetime.now().isoformat(),
        "runtime": "blackforge",
        "config": {"block_size": 64, "gpu_util": 0.92, "prefix_cache": True,
                   "cuda_graph": True, "dflash": False, "max_new": 128,
                   "suffix_len": 10240, "n_rounds": N_ROUNDS},
        "contexts": {}
    }

    for ctx in CTX_LIST:
        print(f"\n{'='*60}")
        print(f"Context: {ctx//1024}K")
        print(f"{'='*60}")

        worker_code = WORKER.format(repo=_REPO)
        try:
            proc = subprocess.run(
                [PYTHON, "-c", worker_code, str(ctx), str(N_ROUNDS)],
                capture_output=True, text=True, timeout=600)

            # Parse result
            for line in proc.stdout.split("\n"):
                if line.startswith("RESULT_JSON:"):
                    data = json.loads(line[len("RESULT_JSON:"):])
                    results["contexts"][str(ctx)] = data
                    for r in data["rounds"]:
                        print(f"  Round {r['round']}: prefill={r['prefill_s']:.2f}s, "
                              f"{r['step_ms']:.2f}ms/step ({r['tok_s']:.1f} tok/s), "
                              f"mem={r['mem_gib']:.1f}GiB")
                    break
            else:
                # No result found
                err = proc.stderr[-500:] if proc.stderr else "no output"
                print(f"  FAILED: {err[:200]}")
                results["contexts"][str(ctx)] = {"status": "FAILED", "error": err[:300]}
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            results["contexts"][str(ctx)] = {"status": "TIMEOUT"}
        except Exception as e:
            print(f"  ERROR: {e}")
            results["contexts"][str(ctx)] = {"status": "ERROR", "error": str(e)[:200]}

    # Save
    out_path = OUT / "full_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY (saved to {out_path})")
    print(f"{'='*60}")
    print(f"{'Ctx':>6} {'Prefill':>8} {'Step':>7} {'Tok/s':>7} {'Mem':>6}")
    for ctx_str, data in results["contexts"].items():
        if data.get("status") == "OK":
            avg_step = sum(r["step_ms"] for r in data["rounds"]) / len(data["rounds"])
            avg_tps = sum(r["tok_s"] for r in data["rounds"]) / len(data["rounds"])
            avg_pf = sum(r["prefill_s"] for r in data["rounds"]) / len(data["rounds"])
            mem = data["rounds"][-1]["mem_gib"]
            print(f"{int(ctx_str)//1024:>5}K {avg_pf:>7.2f}s {avg_step:>6.2f}ms {avg_tps:>6.1f} {mem:>5.1f}G")
        else:
            print(f"{int(ctx_str)//1024:>5}K {data.get('status','?'):>8}")

if __name__ == "__main__":
    main()
