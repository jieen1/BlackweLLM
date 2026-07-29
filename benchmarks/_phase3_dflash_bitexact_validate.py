"""Phase 3 bit-exact validation: vLLM's load_dflash_model() vs
runtime.model_loading.load_laguna_dflash_draft_model().

任务#46 note: the QSR_DFLASH_MODEL_LOADER=vllm arm below no longer runs
against real vLLM (runtime/backends/laguna_dflash.py's loader branch was
removed once 任务#45 confirmed both paths bit-exact -- and separately
found this mixed combination had accumulated a real, unfixed lm_head/
quant_method tying regression, see notes doc 任务#45/#46). Setting this
env var to "vllm" now silently falls through to the self-built draft
loader instead of erroring. Historical record only.

Same rationale/method as _phase1_bitexact_validate.py, applied to the
DFlash draft model instead of the main model. Runs the FULL production
DFlashEngine.generate_verify_only() (real weights, real KV caches for
both main and draft models, real bf_attention/sparkinfer patching) with
a real, coherent, non-repetitive English prompt and greedy decode, then
compares the exact token sequence between the two draft-model-loader
paths. Main model loader is fixed at QSR_LAGUNA_MODEL_LOADER=selfbuilt
(阶段1/2, already the default) throughout -- only the draft-model loader
axis (QSR_DFLASH_MODEL_LOADER) is under test here.

Usage:
  QSR_DFLASH_MODEL_LOADER=vllm      python _phase3_dflash_bitexact_validate.py vllm.pt
  QSR_DFLASH_MODEL_LOADER=selfbuilt python _phase3_dflash_bitexact_validate.py selfbuilt.pt
"""
import os
import sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_LAGUNA_MODEL_LOADER", "selfbuilt")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime/.claude/worktrees/vllm-removal-phase1")

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase3_dflash_bitexact.pt"
LOADER = os.environ.get("QSR_DFLASH_MODEL_LOADER", "vllm")
print(f"dflash_loader={LOADER} out={OUT_PATH}", file=sys.stderr)

import torch

torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

REAL_TEXT = (
    "The history of computing is often told as a series of breakthroughs, "
    "but it is really a story of accumulated small improvements punctuated "
    "by moments of genuine reinvention. Early mechanical calculators solved "
    "narrow arithmetic problems for accountants and astronomers, while the "
    "first electronic machines were built to break codes and calculate "
    "artillery trajectories during wartime. It took decades before anyone "
    "seriously imagined a computer sitting on a desk, let alone in a "
    "pocket. Each generation of engineers inherited constraints from the "
    "last -- limited memory, slow storage, unreliable components -- and "
    "spent their careers working around them, often producing elegant "
    "solutions that outlived the hardware they were designed for. The "
    "operating systems, programming languages, and network protocols "
    "still in daily use today were mostly designed under assumptions that "
    "no longer hold, yet they persist because rewriting foundational "
    "infrastructure is far riskier than living with its quirks."
)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompt_ids = tok.encode(REAL_TEXT, add_special_tokens=False)
print(f"prompt tokens: {len(prompt_ids)}", file=sys.stderr)

from runtime.legacy_qwen36_vllm import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.backends.laguna_dflash import DFlashEngine

CTX = len(prompt_ids)
MAX_TOKENS = 64
BLOCK_SIZE = 64
max_model_len = CTX + MAX_TOKENS + 2048
bps = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 64

engine_args = EngineArgs(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=max_model_len,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(
    vllm_config, num_slots=1, block_size=BLOCK_SIZE, blocks_per_slot=bps
)
engine = DFlashEngine(backend)

tokens, stats = engine.generate_verify_only(
    prompt_ids, max_tokens=MAX_TOKENS, temperature=0.0, enable_prefix_cache=False
)

print(f"generated {len(tokens)} tokens", file=sys.stderr)
print(f"decoded: {tok.decode(tokens)!r}", file=sys.stderr)
print(f"acceptance_rate={stats['acceptance_rate']:.6f}", file=sys.stderr)

torch.save(
    {
        "loader": LOADER,
        "prompt_len": CTX,
        "tokens": tokens,
        "acceptance_rate": stats["acceptance_rate"],
    },
    OUT_PATH,
)
print(f"saved to {OUT_PATH}", file=sys.stderr)
