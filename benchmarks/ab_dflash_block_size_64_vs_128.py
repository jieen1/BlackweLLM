"""A/B: DFlash verify at 64K, block_size=64 vs block_size=128.

Same methodology as ab_dflash_verify_cg_vs_eager.py (same repeated-phrase
prompt, same CTX/K/greedy), but sweeps block_size to validate the L-P0-style
migration unlocking sparkinfer's Laguna kernel traits (which require
page_size==128). Correctness bar: acceptance_rate should match (or be
extremely close -- split-KV/page-size changes floating point reduction
order, see notes/2026-07-27-verify-cg-mode-fix-and-block-size-eval.md for
precedent) between the two block sizes; a large divergence in accepted
tokens would indicate a real bug, not just FP noise.
"""
import gc, json, os, sys, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_DFLASH_CUDA_GRAPH", "1")
os.environ.setdefault("QSR_VERIFY_CUDA_GRAPH", "1")
os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

BLOCK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 65536

import torch
torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
MAX_TOKENS = 256

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

prompt = make_ids(CTX)
max_model_len = CTX + MAX_TOKENS + 2048
margin_tokens = 4096
bps = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + (
    (margin_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
)

engine_args = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=max_model_len,
    gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(vllm_config, num_slots=1, block_size=BLOCK_SIZE, blocks_per_slot=bps)
engine = DFlashEngine(backend)

print(f"block_size={BLOCK_SIZE} blocks_per_slot={bps} verify_cg={engine._verify_cg is not None} "
      f"draft_cg={engine._draft_cg is not None}", file=sys.stderr)

rounds = []
for ri in range(2):
    t0 = time.time()
    tokens, stats = engine.generate(prompt, max_tokens=MAX_TOKENS)
    wall = time.time() - t0
    print(f"Round {ri}: tok_per_s={stats['tok_per_s']:.2f} accept={stats['acceptance_rate']:.6f} "
          f"wall={wall:.1f}s", file=sys.stderr)
    rounds.append({"round": ri, "tok_per_s": stats["tok_per_s"],
                    "acceptance_rate": stats["acceptance_rate"], "wall_s": round(wall, 1),
                    "tokens": tokens})

result = {
    "block_size": BLOCK_SIZE, "blocks_per_slot": bps,
    "ctx": CTX, "max_tokens": MAX_TOKENS,
    "verify_cg_active": engine._verify_cg is not None,
    "draft_cg_active": engine._draft_cg is not None,
    "rounds": rounds,
}
print(json.dumps(result, indent=2))
