"""Throughput: decode_batch_sampled eager vs CG-routed, server-like single-slot calls."""
import os, sys, time, json
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch
torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

from oracle.qwen36_vllm.vllm_compat import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.sampling import SamplingParams

CTX = 4096
MAX_MODEL_LEN = CTX + 4096
BPS = (MAX_MODEL_LEN + 63) // 64 + 8
N_STEPS = 200

engine_args = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=True)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(vllm_config, num_slots=2, block_size=64, blocks_per_slot=BPS)
prompt = (list(range(1000, 1100)) * ((CTX // 100) + 1))[:CTX]
params_greedy = SamplingParams(temperature=0.0)

def run_eager(n):
    backend.reset_slot(0)
    tok = backend.prefill(0, prompt)
    kv = backend.slot_kv_len[0]
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = backend.decode_batch_sampled([0], [tok], [kv], [params_greedy])
        tok = out[0]
        kv = backend.slot_kv_len[0]
    torch.cuda.synchronize()
    return time.time() - t0

def run_cg(n):
    backend.reset_slot(0)
    tok = backend.prefill(0, prompt)
    kv = backend.slot_kv_len[0]
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = backend.decode_batch_sampled([0], [tok], [kv], [params_greedy])
        tok = out[0]
        kv = backend.slot_kv_len[0]
    torch.cuda.synchronize()
    return time.time() - t0

# Force _decode_cg = None to measure genuinely eager-only path (simulates
# QSR_DECODE_CUDA_GRAPH=0 / enable_cudagraph=False, i.e. today's server default).
eager_s = run_eager(N_STEPS)
eager_tok_s = N_STEPS / eager_s
print(f"eager-only (no CG, matches today's server default): {eager_tok_s:.1f} tok/s ({eager_s*1000/N_STEPS:.2f} ms/step)", file=sys.stderr)

backend._ensure_decode_cg()
assert backend._decode_cg is not None
cg_s = run_cg(N_STEPS)
cg_tok_s = N_STEPS / cg_s
print(f"CG-routed (this change, enable_cudagraph=True): {cg_tok_s:.1f} tok/s ({cg_s*1000/N_STEPS:.2f} ms/step)", file=sys.stderr)

print(json.dumps({
    "ctx": CTX, "n_steps": N_STEPS,
    "eager_tok_s": round(eager_tok_s, 1), "eager_ms_step": round(eager_s*1000/N_STEPS, 2),
    "cg_tok_s": round(cg_tok_s, 1), "cg_ms_step": round(cg_s*1000/N_STEPS, 2),
    "speedup": round(cg_tok_s / eager_tok_s, 3),
}, indent=2))
