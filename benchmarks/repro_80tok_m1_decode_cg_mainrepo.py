"""M=1 decode CG benchmark at 64K — replicating commit 66d5913 methodology."""
import gc, json, os, sys, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE", "1")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch; torch.set_grad_enabled(False)

CTX = 65536
SUFFIX_LEN = 10240
MAX_NEW = 128
GPU_UTIL = 0.92
BLOCK_SIZE = 64

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/")

def make_prompt(n):
    chunk = list(range(1000, 1100))
    ids = []
    while len(ids) < n:
        ids.extend(chunk)
    return ids[:n]

def gpu_mem_gib():
    import subprocess as sp
    try:
        o = sp.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"], text=True)
        return float(o.strip().split("\n")[0]) / 1024
    except: return -1.0

from runtime.compat_vllm import EngineArgs
from runtime.backends.laguna import LagunaBackend

base_len = CTX
total_len = base_len + SUFFIX_LEN
max_model_len = total_len + 2048
blocks_per_slot = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 64

print(f"Config: ctx={CTX}, total={total_len}, max_model_len={max_model_len}, bps={blocks_per_slot}", file=sys.stderr)

engine_args = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=max_model_len,
    gpu_memory_utilization=GPU_UTIL, enforce_eager=True, trust_remote_code=True)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(vllm_config, num_slots=1, block_size=BLOCK_SIZE, blocks_per_slot=blocks_per_slot)

mem_load = gpu_mem_gib()
print(f"Model loaded: {mem_load:.1f} GiB", file=sys.stderr)

base_ids = make_prompt(base_len)
suffix_ids = [x + 50000 for x in make_prompt(SUFFIX_LEN)]

# Warmup prefill (base only)
t0 = time.time()
first = backend.prefill(0, base_ids)
warmup_s = time.time() - t0
print(f"Warmup prefill: {warmup_s:.1f}s", file=sys.stderr)

# Capture CG after warmup
backend._ensure_decode_cg()
cg = backend._decode_cg
kv = backend.slot_kv_len[0]
tok_id = first
for _ in range(20):
    r = cg.replay([0], [tok_id], [kv]); tok_id = r[0]; kv += 1
print(f"CG captured and warmed up", file=sys.stderr)

# Test rounds
N_ROUNDS = 3
rounds = []
for ri in range(N_ROUNDS):
    backend.reset_slot(0)
    backend._unpatch_impls_for_prefill()

    full_prompt = base_ids + suffix_ids
    torch.cuda.synchronize()
    t0 = time.time()
    first_token = backend.prefill(0, full_prompt)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0

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
    mem = gpu_mem_gib()
    rounds.append({"round": ri+1, "prefill_s": round(prefill_s,3),
        "step_ms": round(step_ms,2), "tok_s": round(tok_s,1), "mem_gib": round(mem,1)})
    print(f"  Round {ri+1}: {step_ms:.2f}ms/step, {tok_s:.1f} tok/s, prefill={prefill_s:.1f}s, mem={mem:.1f}GiB", file=sys.stderr)

result = {
    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "test": "M=1 decode CG (NO DFlash)",
    "config": {"ctx": CTX, "block_size": BLOCK_SIZE, "gpu_util": GPU_UTIL,
               "max_new": MAX_NEW, "suffix_len": SUFFIX_LEN},
    "mem_load_gib": round(mem_load, 1),
    "warmup_prefill_s": round(warmup_s, 1),
    "rounds": rounds,
    "avg_tok_s": round(sum(r["tok_s"] for r in rounds) / len(rounds), 1),
    "avg_step_ms": round(sum(r["step_ms"] for r in rounds) / len(rounds), 2),
}
print(json.dumps(result, indent=2))
