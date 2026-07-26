"""Stage-level CUDA-event profiling of one DFlash speculative-decode round."""
import os, sys, time, json
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_DFLASH_CUDA_GRAPH", "1")
os.environ["QSR_VERIFY_CUDA_GRAPH"] = "1"
os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch
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


from runtime.compat_vllm import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.backends.laguna_dflash import DFlashEngine
from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS, NUM_QUERY_PER_REQ
from runtime.backends.laguna_dflash import (
    _verify_only_accept_reject,
    _physical_slot,
)

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
print("prefill done, capturing CG...", file=sys.stderr)

if engine._use_cuda_graph and not engine._cg_captured:
    engine._lazy_capture_cg()
torch.cuda.synchronize()
print(f"CG captured: draft={engine._draft_cg is not None} verify={engine._verify_cg is not None}", file=sys.stderr)

bonus_token = first_token
kv_len = backend.slot_kv_len[slot]
draft_tokens = engine._draft_cg.replay(slot, bonus_token, kv_len)
torch.cuda.synchronize()

N_WARMUP = 10
N_MEASURE = 30

stage_times = {"draft_replay": [], "verify_replay": [], "accept_reject_sync": [],
               "draft_kv_precompute": [], "bookkeeping": [], "round_total": []}

e0, e1, e2, e3, e4, e5 = (torch.cuda.Event(enable_timing=True) for _ in range(6))

for i in range(N_WARMUP + N_MEASURE):
    kv_len = backend.slot_kv_len[slot]
    verify_tokens = [bonus_token] + draft_tokens

    torch.cuda.synchronize()
    t_round0 = time.perf_counter()

    e0.record()
    verify_logits, verify_aux = engine._verify_cg.replay_with_aux(slot, verify_tokens, kv_len)
    e1.record()

    all_argmax = verify_logits[:NUM_QUERY_PER_REQ].argmax(dim=-1).tolist()
    e2.record()
    torch.cuda.synchronize()  # need this to read decision/timings safely before GPU work below

    decision = _verify_only_accept_reject(all_argmax, draft_tokens, bonus_token)
    num_accepted = decision["num_accepted"]
    new_tokens = decision["committed"]
    new_bonus = decision["next_anchor"]
    context_count = decision["context_count"]

    e3.record()
    if verify_aux is not None:
        aux_slice = [a[:context_count] for a in verify_aux]
        combined_input = torch.cat(aux_slice, dim=-1)
        combined = engine.draft_model.combine_hidden_states(combined_input)
        bs = engine.block_size
        phys = _physical_slot(slot)
        draft_base = phys * engine._draft_blocks_per_slot
        ring_slots = engine._draft_blocks_per_slot * bs
        context_positions = torch.arange(
            kv_len, kv_len + context_count, dtype=torch.long, device=engine.device
        )
        ring_blocks = (context_positions % ring_slots) // bs
        ring_offs = context_positions % bs
        slot_mappings = (draft_base + ring_blocks) * bs + ring_offs
        engine.draft_model.precompute_and_store_context_kv(
            combined, context_positions, slot_mappings
        )
    e4.record()

    backend.slot_kv_len[slot] += context_count
    for t in new_tokens:
        backend.slot_committed_tokens[slot].append(t)

    bonus_token = new_bonus
    new_kv_len = backend.slot_kv_len[slot]
    draft_tokens = engine._draft_cg.replay(slot, bonus_token, new_kv_len)
    e5.record()
    torch.cuda.synchronize()
    t_round1 = time.perf_counter()

    if i >= N_WARMUP:
        stage_times["verify_replay"].append(e0.elapsed_time(e1))
        stage_times["accept_reject_sync"].append(e1.elapsed_time(e2))
        stage_times["draft_kv_precompute"].append(e3.elapsed_time(e4))
        stage_times["draft_replay"].append(e4.elapsed_time(e5))
        stage_times["bookkeeping"].append(e2.elapsed_time(e3))
        stage_times["round_total"].append((t_round1 - t_round0) * 1000)

result = {}
for k, v in stage_times.items():
    result[k] = {"mean_ms": sum(v) / len(v), "min_ms": min(v), "max_ms": max(v), "n": len(v)}

print(json.dumps(result, indent=2))
print(f"\nacceptance this run: {sum(1 for _ in range(1))}", file=sys.stderr)
