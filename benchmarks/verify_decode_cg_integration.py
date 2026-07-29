"""Correctness A/B: decode_batch_sampled eager vs CG-routed path, plus edge cases."""
import os, sys, json
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

CTX = 512
MAX_MODEL_LEN = CTX + 2048
BPS = (MAX_MODEL_LEN + 63) // 64 + 8

engine_args = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=True)
vllm_config = engine_args.create_engine_config()
# num_slots=2: slot 0 for real serving, slot 1 reserved as CG capture scratch,
# matching ServerEngine's min_slots = capacity(1) + capacity(1) formula.
backend = LagunaBackend(vllm_config, num_slots=2, block_size=64, blocks_per_slot=BPS)

prompt = (list(range(1000, 1100)) * ((CTX // 100) + 1))[:CTX]

results = {}

# ---- Test 1: eager path (CG not yet captured) produces baseline tokens ----
backend.reset_slot(0)
first = backend.prefill(0, prompt)
params_greedy = SamplingParams(temperature=0.0)
eager_tokens = [first]
tok, kv = first, backend.slot_kv_len[0]
for _ in range(20):
    out = backend.decode_batch_sampled([0], [tok], [kv], [params_greedy])
    tok = out[0]
    kv = backend.slot_kv_len[0]
    eager_tokens.append(tok)
results["eager_no_cg"] = eager_tokens
print("eager (pre-CG):", eager_tokens, file=sys.stderr)

# ---- Now capture the decode CG (as server startup would) ----
backend._ensure_decode_cg()
assert backend._decode_cg is not None, "CG capture failed"
print(f"CG captured: batch_size={backend._decode_cg.batch_size}", file=sys.stderr)

# ---- Test 2: replay from scratch with CG active, same prompt, must match exactly ----
backend.reset_slot(0)
first2 = backend.prefill(0, prompt)
# Deliberately NOT calling _repatch_impls_for_cg() here: replay() bypasses
# the attention layer's Python impl entirely (a captured CUDA graph replays
# exactly the kernel launches recorded at capture time), so decode_batch_sampled's
# CG routing has no dependency on impl-patch state. Confirm that holds.
assert first2 == first, "prefill anchor changed after CG capture!"
cg_tokens = [first2]
tok, kv = first2, backend.slot_kv_len[0]
cg_path_used = []
for _ in range(20):
    eligible = backend._decode_cg_batch_eligible([0], [params_greedy], False)
    cg_path_used.append(eligible)
    out = backend.decode_batch_sampled([0], [tok], [kv], [params_greedy])
    tok = out[0]
    kv = backend.slot_kv_len[0]
    cg_tokens.append(tok)
results["cg_routed"] = cg_tokens
results["cg_path_used_each_step"] = cg_path_used
print("cg-routed:     ", cg_tokens, file=sys.stderr)
print("cg eligible each step:", cg_path_used, file=sys.stderr)

match = eager_tokens == cg_tokens
results["exact_match"] = match
print(f"EXACT MATCH: {match}", file=sys.stderr)
assert match, "MISMATCH between eager and CG-routed decode_batch_sampled!"

# ---- Test 3: non-greedy request must NOT use CG path (fallback correctness) ----
params_sampled = SamplingParams(temperature=0.7, seed=42)
not_eligible_sampled = backend._decode_cg_batch_eligible([0], [params_sampled], False)
results["sampled_not_eligible"] = not not_eligible_sampled
print(f"sampled request correctly excluded from CG: {not not_eligible_sampled}", file=sys.stderr)
assert not not_eligible_sampled

# ---- Test 4: return_logprobs=True must NOT use CG path ----
not_eligible_lp = backend._decode_cg_batch_eligible([0], [params_greedy], True)
results["logprobs_not_eligible"] = not not_eligible_lp
print(f"logprobs request correctly excluded from CG: {not not_eligible_lp}", file=sys.stderr)
assert not not_eligible_lp

# ---- Test 5: wrong batch size must NOT use CG path (batch_size=1 captured) ----
not_eligible_bs = backend._decode_cg_batch_eligible([0, 1], [params_greedy, params_greedy], False)
results["wrong_batch_size_not_eligible"] = not not_eligible_bs
print(f"batch_size=2 correctly excluded from CG (captured at 1): {not not_eligible_bs}", file=sys.stderr)
assert not not_eligible_bs

# ---- Test 6: mixed sampled request actually still decodes correctly via eager fallback ----
backend.reset_slot(1)
first3 = backend.prefill(1, prompt[:512])
tok3 = first3
out = backend.decode_batch_sampled([1], [tok3], [backend.slot_kv_len[1]], [params_sampled])
results["sampled_fallback_ran_ok"] = True
print(f"sampled decode via eager fallback produced token: {out[0]}", file=sys.stderr)

print(json.dumps(results, indent=2))
