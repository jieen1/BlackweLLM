"""Kernel-level torch.profiler breakdown of the M=16 verify CG replay."""
import os, sys
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_DFLASH_CUDA_GRAPH", "1")
os.environ["QSR_VERIFY_CUDA_GRAPH"] = "1"
os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch
from torch.profiler import profile, ProfilerActivity
torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
CTX = 65536

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
BASE_TEXT = "The quick brown fox jumps over the lazy dog. "
CHUNK_IDS = tok.encode(BASE_TEXT, add_special_tokens=False)


def make_ids(n):
    ids = []
    while len(ids) < n:
        ids.extend(CHUNK_IDS)
    return ids[:n]


from oracle.qwen36_vllm.vllm_compat import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.backends.laguna_dflash import DFlashEngine
from runtime.backends.dflash_constants import NUM_QUERY_PER_REQ
from runtime.backends.laguna_dflash import _verify_only_accept_reject, _physical_slot

prompt = make_ids(CTX)
max_model_len = CTX + 256 + 2048
bps = (max_model_len + 63) // 64 + 64

engine_args = EngineArgs(
    model=MODEL, dtype="bfloat16", max_model_len=max_model_len,
    gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(vllm_config, num_slots=1, block_size=64, blocks_per_slot=bps)
engine = DFlashEngine(backend)

slot = 0
print("prefilling...", file=sys.stderr)
first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt)
if aux_hidden_states is not None:
    aux_len = aux_hidden_states[0].shape[0]
    aux_offset = len(prompt) - aux_len
    engine._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)
del aux_hidden_states
torch.cuda.synchronize()

if engine._use_cuda_graph and not engine._cg_captured:
    engine._lazy_capture_cg()
torch.cuda.synchronize()
print(f"CG captured: draft={engine._draft_cg is not None} verify={engine._verify_cg is not None}", file=sys.stderr)

bonus_token = first_token
kv_len = backend.slot_kv_len[slot]
draft_tokens = engine._draft_cg.replay(slot, bonus_token, kv_len)
torch.cuda.synchronize()

# Warmup the verify replay path itself (not just capture) before profiling.
for _ in range(8):
    verify_tokens = [bonus_token] + draft_tokens
    verify_logits, verify_aux = engine._verify_cg.replay_with_aux(slot, verify_tokens, kv_len)
    all_argmax = verify_logits[:NUM_QUERY_PER_REQ].argmax(dim=-1).tolist()
    decision = _verify_only_accept_reject(all_argmax, draft_tokens, bonus_token)
    context_count = decision["context_count"]
    if verify_aux is not None:
        aux_slice = [a[:context_count] for a in verify_aux]
        combined_input = torch.cat(aux_slice, dim=-1)
        combined = engine.draft_model.combine_hidden_states(combined_input)
        bs = engine.block_size
        phys = _physical_slot(slot)
        draft_base = phys * engine._draft_blocks_per_slot
        ring_slots = engine._draft_blocks_per_slot * bs
        context_positions = torch.arange(kv_len, kv_len + context_count, dtype=torch.long, device=engine.device)
        ring_blocks = (context_positions % ring_slots) // bs
        ring_offs = context_positions % bs
        slot_mappings = (draft_base + ring_blocks) * bs + ring_offs
        engine.draft_model.precompute_and_store_context_kv(combined, context_positions, slot_mappings)
    backend.slot_kv_len[slot] += context_count
    bonus_token = decision["next_anchor"]
    kv_len = backend.slot_kv_len[slot]
    draft_tokens = engine._draft_cg.replay(slot, bonus_token, kv_len)
torch.cuda.synchronize()
print("warmup done, profiling...", file=sys.stderr)

N_PROF = 10
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(N_PROF):
        verify_tokens = [bonus_token] + draft_tokens
        verify_logits, verify_aux = engine._verify_cg.replay_with_aux(slot, verify_tokens, kv_len)
        torch.cuda.synchronize()

print(f"\n===== TOP 40 CUDA KERNELS BY TOTAL TIME (over {N_PROF} replays) =====")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

# Bucket by name heuristics for a category rollup.
events = prof.key_averages()
buckets = {"attention": 0.0, "moe": 0.0, "gemm_dense": 0.0, "norm": 0.0, "memcpy_memset": 0.0, "other": 0.0}
total_cuda_us = 0.0
for ev in events:
    cuda_us = ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
    total_cuda_us += cuda_us
    name = ev.key.lower()
    if any(s in name for s in ["attn", "attention", "paged", "flash"]):
        buckets["attention"] += cuda_us
    elif any(s in name for s in ["moe", "expert", "topk", "gating", "scatter", "gather"]):
        buckets["moe"] += cuda_us
    elif any(s in name for s in ["gemm", "cutlass", "wmma", "sgemm", "gemv", "cublas", "linear"]):
        buckets["gemm_dense"] += cuda_us
    elif any(s in name for s in ["norm", "rms"]):
        buckets["norm"] += cuda_us
    elif any(s in name for s in ["memcpy", "memset", "fill"]):
        buckets["memcpy_memset"] += cuda_us
    else:
        buckets["other"] += cuda_us

print("\n===== CATEGORY ROLLUP (per single verify replay, ms) =====")
for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
    per_replay_ms = v / N_PROF / 1000.0
    pct = 100.0 * v / total_cuda_us if total_cuda_us else 0
    print(f"{k:15s}: {per_replay_ms:8.3f} ms/replay  ({pct:5.1f}%)")
print(f"{'TOTAL':15s}: {total_cuda_us/N_PROF/1000.0:8.3f} ms/replay")
