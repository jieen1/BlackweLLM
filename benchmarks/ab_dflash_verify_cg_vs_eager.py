"""A/B: DFlash verify CG vs eager at 64K, matching cac38ab methodology."""
import json, os, sys, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_DFLASH_CUDA_GRAPH", "1")
os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

VERIFY_CG = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["QSR_VERIFY_CUDA_GRAPH"] = VERIFY_CG

import torch
torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
CTX = 65536
MAX_TOKENS = 256

from bfdiag.record import auto_record  # demo: zero-invasion bf run-record integration
_bf = auto_record(script=__file__, workload={"prompt_len": CTX, "k": 15, "greedy": True, "block_size": 64})

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
BASE_TEXT = "The quick brown fox jumps over the lazy dog. "
CHUNK_IDS = tok.encode(BASE_TEXT, add_special_tokens=False)

def make_ids(n):
    ids = []
    while len(ids) < n:
        ids.extend(CHUNK_IDS)
    return ids[:n]

from runtime.legacy_qwen36_vllm import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.backends.laguna_dflash import DFlashEngine

prompt = make_ids(CTX)
max_model_len = CTX + MAX_TOKENS + 2048
bps = (max_model_len + 63) // 64 + 64

engine_args = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=max_model_len,
    gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(vllm_config, num_slots=1, block_size=64, blocks_per_slot=bps)
engine = DFlashEngine(backend)

print(f"QSR_VERIFY_CUDA_GRAPH={VERIFY_CG}", file=sys.stderr)

rounds = []
for ri in range(2):
    t0 = time.time()
    tokens, stats = engine.generate(prompt, max_tokens=MAX_TOKENS)
    wall = time.time() - t0
    print(f"Round {ri}: tok_per_s={stats['tok_per_s']:.2f} accept={stats['acceptance_rate']:.3f} "
          f"wall={wall:.1f}s verify_cg={engine._verify_cg is not None}", file=sys.stderr)
    rounds.append({"round": ri, "tok_per_s": stats["tok_per_s"],
                    "acceptance_rate": stats["acceptance_rate"], "wall_s": round(wall, 1)})

result = {
    "config": "QSR_VERIFY_CUDA_GRAPH=" + VERIFY_CG,
    "ctx": CTX, "max_tokens": MAX_TOKENS,
    "verify_cg_active": engine._verify_cg is not None,
    "rounds": rounds,
}
_bf.metric("acceptance_rate", rounds[-1]["acceptance_rate"])
print(json.dumps(result, indent=2))
