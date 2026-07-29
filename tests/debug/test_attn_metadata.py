import os, sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
from oracle.qwen36_vllm.vllm_compat import EngineArgs

engine_args = EngineArgs(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.88,
    enforce_eager=True,
    trust_remote_code=True,
)
vllm_config = engine_args.create_engine_config()

from runtime.backends.laguna import LagunaBackend

backend = LagunaBackend(vllm_config, num_slots=1, block_size=64, blocks_per_slot=64)

from runtime.backends.bf_attention import BFAttention, get_bf_attn_context
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Instrument layer 0
_orig_fwd = BFAttention.forward
_done = [False]


def _check(self, query, key, value, output_shape=None, output_dtype=None):
    if not _done[0] and self.layer_name == "model.layers.0.self_attn.attn":
        _done[0] = True
        ctx = get_bf_attn_context()
        meta = ctx.attn_metadata.get(self.layer_name)
        sm = ctx.slot_mapping.get(self.layer_name)
        print("=== Layer 0 metadata ===")
        print(f"  slot_mapping: {sm.tolist()}")
        print(f"  num_actual_tokens: {meta.num_actual_tokens}")
        print(f"  page_table shape: {meta.page_table.shape}")
        print(f"  page_table[0,:10]: {meta.page_table[0, :10].tolist()}")
        print(f"  cache_seqlens: {meta.cache_seqlens.tolist()}")
        print(f"  cu_seqlens_q: {meta.cu_seqlens_q.tolist()}")
        print(f"  window_left: {meta.window_left}")
        seq_len = meta.cache_seqlens[0].item()
        num_pages = (seq_len + 63) // 64
        print(f"  seq_len={seq_len}, pages_needed={num_pages}")
        print(f"  page_table entries: {meta.page_table[0, :num_pages].tolist()}")
        print(f"  slot_mapping blocks: {(sm // 64).tolist()}")
        # Check: do page_table entries match slot_mapping blocks?
        pt_blocks = set(meta.page_table[0, :num_pages].tolist())
        sm_blocks = set((sm // 64).tolist())
        print(f"  page_table blocks: {sorted(pt_blocks)}")
        print(f"  slot_mapping blocks: {sorted(sm_blocks)}")
        print(f"  match: {pt_blocks == sm_blocks}")
    return _orig_fwd(self, query, key, value, output_shape, output_dtype)


BFAttention.forward = _check

prompt = "The capital of France is"
token_ids = tok.encode(prompt)
assert token_ids[0] == tok.bos_token_id, (
    f"Missing BOS: first token is {token_ids[0]}, expected {tok.bos_token_id}"
)
print(f"Prompt: {len(token_ids)} tokens: {token_ids}")

slot = 0
backend.reset_slot(slot)
first = backend.prefill(slot, token_ids)
print(f"\nFirst token: {first} = {tok.decode([first])!r}")
