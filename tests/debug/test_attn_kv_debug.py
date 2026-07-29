import os, sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
from runtime.legacy_qwen36_vllm import EngineArgs

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

# Check BFAttention config
from runtime.backends.bf_attention import BFAttention

sfc = backend.static_forward_context
for name in list(sfc.keys())[:3]:
    layer = sfc[name]
    if isinstance(layer, BFAttention):
        print(
            f"{name}: num_heads={layer.num_heads} head_size={layer.head_size} "
            f"num_kv_heads={layer.num_kv_heads} scale={layer.scale:.6f} "
            f"window_left={layer.window_left} "
            f"k_scale={layer._k_scale.item():.6f} v_scale={layer._v_scale.item():.6f} "
            f"kv_cache={layer.kv_cache.shape}"
        )

# Check: read back KV cache after prefill
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompt = "The capital of France is"
token_ids = tok.encode(prompt)
assert token_ids[0] == tok.bos_token_id, (
    f"Missing BOS: first token is {token_ids[0]}, expected {tok.bos_token_id}"
)

slot = 0
backend.reset_slot(slot)

# Instrument to check KV cache after write
_orig_fwd = BFAttention.forward
_check_done = [False]


def _check_forward(self, query, key, value, output_shape=None, output_dtype=None):
    result = _orig_fwd(self, query, key, value, output_shape, output_dtype)
    if not _check_done[0] and self.layer_name == "model.layers.0.self_attn.attn":
        _check_done[0] = True
        # Read back KV cache at the written positions
        from runtime.backends.bf_attention import get_bf_attn_context

        ctx = get_bf_attn_context()
        sm = ctx.slot_mapping.get(self.layer_name)
        if sm is not None and self.kv_cache is not None:
            k_cache = self.kv_cache[:, 0].view(torch.float8_e4m3fn)
            block_size = k_cache.shape[1]
            for i in range(min(3, len(sm))):
                bi = sm[i].item() // block_size
                bo = sm[i].item() % block_size
                kv = k_cache[bi, bo, 0, :4].float()  # first kv_head, first 4 dims
                print(f"  KV cache[{bi},{bo},0,:4] = {kv.tolist()} (should be nonzero)")
        # Check output
        print(f"  output norm={result.float().norm().item():.2f}")
    return result


BFAttention.forward = _check_forward

first = backend.prefill(slot, token_ids)
print(f"\nFirst token: {first} = {tok.decode([first])!r}")
