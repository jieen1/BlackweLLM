"""Quick correctness test: 'The capital of France is' → should produce ' Paris'"""

import os, sys, time

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

from transformers import AutoTokenizer

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)


def build_vllm_config():
    from runtime.compat_vllm import EngineArgs

    engine_args = EngineArgs(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        trust_remote_code=True,
    )
    return engine_args.create_engine_config()


print("Building config...")
vllm_config = build_vllm_config()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("Loading backend...")
t0 = time.time()
from runtime.backends.laguna import LagunaBackend

backend = LagunaBackend(vllm_config, num_slots=1, block_size=64, blocks_per_slot=64)
print(f"Backend loaded in {time.time() - t0:.1f}s")

# Test 1: greedy generation
prompt = "The capital of France is"
print(f"\nPrompt: {prompt!r}")
slot = 0
backend.reset_slot(slot)
token_ids = tok.encode(prompt)
assert token_ids[0] == tok.bos_token_id, (
    f"Missing BOS: first token is {token_ids[0]}, expected {tok.bos_token_id}"
)
first = backend.prefill(slot, token_ids)
generated = [first]
for i in range(15):
    nxt = backend.decode(slot, generated[-1])
    generated.append(nxt)

text = tok.decode(generated)
print(f"Generated: {text!r}")
print(f"Full: {prompt + text!r}")

if "Paris" in text or "paris" in text:
    print("\n✅ CORRECT")
else:
    print("\n❌ WRONG — expected 'Paris'")

# Test 2: another prompt
prompt2 = "2 + 2 ="
print(f"\nPrompt: {prompt2!r}")
backend.reset_slot(slot)
token_ids2 = tok.encode(prompt2)
first2 = backend.prefill(slot, token_ids2)
gen2 = [first2]
for i in range(10):
    nxt = backend.decode(slot, gen2[-1])
    gen2.append(nxt)
text2 = tok.decode(gen2)
print(f"Generated: {text2!r}")

backend.shutdown()
