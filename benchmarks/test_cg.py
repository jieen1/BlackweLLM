"""Minimal CG crash reproduction."""
import os, sys, time, logging
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch; torch.set_grad_enabled(False)

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/07614121b31898586430f189d27a25a0be310843/")
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 65536

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base_text = ("The quick brown fox jumps over the lazy dog. "
    "In a world of artificial intelligence and machine learning, "
    "the importance of efficient inference cannot be overstated. ")
chunk = tok.encode(base_text, add_special_tokens=False)
ids = []
while len(ids) < CTX: ids.extend(chunk)
ids = ids[:CTX]

from runtime.legacy_qwen36_vllm import EngineArgs
max_len = max(CTX + 1024, 262144)
bps = (CTX + 15) // 16 + 512
print(f"Config: ctx={CTX}, bps={bps}, max_len={max_len}", flush=True)
vc = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=max_len,
    gpu_memory_utilization=0.92, enforce_eager=True, trust_remote_code=True).create_engine_config()
from runtime.backends.laguna import LagunaBackend
backend = LagunaBackend(vc, num_slots=1, block_size=64, blocks_per_slot=bps)
from runtime.backends.laguna_dflash import DFlashEngine
engine = DFlashEngine(backend)

# Phase 1: short warmup to trigger lazy CG capture
print("Phase 1: short warmup (256 tokens)...", flush=True)
backend.reset_slot(0)
tw, sw = engine.generate(ids[:256], max_tokens=5, temperature=0.0, slot=0, enable_prefix_cache=False)
torch.cuda.synchronize()
print(f"  warmup done, cg_captured={engine._cg_captured}", flush=True)
print(f"  verify_cg={engine._verify_cg is not None}, draft_cg={engine._draft_cg is not None}", flush=True)

# Phase 2: full context with CG
print(f"Phase 2: {CTX} context with CG...", flush=True)
backend.reset_slot(0)
for kv_tensor in engine._draft_kv_caches.values():
    kv_tensor.zero_()
t0 = time.perf_counter()
try:
    tokens, stats = engine.generate(ids, max_tokens=32, temperature=0.0, slot=0, enable_prefix_cache=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  SUCCESS: {len(tokens)} tokens, {stats['tok_per_s']:.1f} tok/s, "
          f"accept={stats['acceptance_rate']:.0%}, pf={stats['prefill_ms']/1000:.1f}s", flush=True)
except Exception as e:
    print(f"  CRASH: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
