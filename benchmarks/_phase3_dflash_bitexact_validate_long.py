"""Phase 3 bit-exact validation, extended coverage: long context + chunked
prefill + many DFlash rounds, forcing wraparound in BOTH the main model's
SWA ring (few thousand tokens) AND the draft model's much smaller
DRAFT_WINDOW=512 ring (runtime/backends/dflash_constants.py). The draft
ring is the higher-risk one for this specific change -- it's what
LagunaDraftModelSelfBuilt.precompute_and_store_context_kv actually writes
into every round, and it wraps roughly every ~35-50 rounds instead of
every few thousand tokens like the main model's ring.

Same real-text-corpus rationale as _phase1_bitexact_validate_long.py
(several distinct real paragraphs, cycled -- not synthetic/random data,
not a single degenerate repeated phrase).

任务#46 note: the QSR_DFLASH_MODEL_LOADER=vllm arm below no longer runs
against real vLLM (runtime/backends/laguna_dflash.py's loader branch was
removed once 任务#45 confirmed both paths bit-exact -- and separately
found this mixed combination had accumulated a real, unfixed lm_head/
quant_method tying regression, see notes doc 任务#45/#46). Setting this
env var to "vllm" now silently falls through to the self-built draft
loader instead of erroring. Historical record only.

Usage:
  QSR_DFLASH_MODEL_LOADER=vllm      python _phase3_dflash_bitexact_validate_long.py vllm.pt
  QSR_DFLASH_MODEL_LOADER=selfbuilt python _phase3_dflash_bitexact_validate_long.py selfbuilt.pt
"""
import os
import sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_LAGUNA_MODEL_LOADER", "selfbuilt")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime/.claude/worktrees/vllm-removal-phase1")

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase3_dflash_bitexact_long.pt"
LOADER = os.environ.get("QSR_DFLASH_MODEL_LOADER", "vllm")
print(f"dflash_loader={LOADER} out={OUT_PATH}", file=sys.stderr)

import torch

torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

PASSAGES = [
    "The history of computing is often told as a series of breakthroughs, "
    "but it is really a story of accumulated small improvements punctuated "
    "by moments of genuine reinvention. Early mechanical calculators solved "
    "narrow arithmetic problems for accountants and astronomers, while the "
    "first electronic machines were built to break codes and calculate "
    "artillery trajectories during wartime.",
    "Coral reefs cover less than one percent of the ocean floor, yet they "
    "support roughly a quarter of all known marine species. This "
    "disproportionate richness comes from the reef's physical complexity: "
    "the branching, cratered structure built up over centuries by countless "
    "tiny polyps creates an enormous surface area of hiding places, feeding "
    "grounds, and nursery habitats packed into a small footprint.",
    "Contract law rests on a deceptively simple idea -- that a promise, "
    "once certain conditions are met, becomes enforceable -- but the "
    "conditions themselves have accumulated centuries of nuance. Courts "
    "distinguish between an offer and an invitation to treat, between "
    "consideration that is merely nominal and consideration that is legally "
    "sufficient, and between terms that were genuinely negotiated and terms "
    "buried in fine print nobody read.",
    "A sourdough starter is, at its core, a small and durable ecosystem: "
    "wild yeast and lactic acid bacteria living in a paste of flour and "
    "water, competing and cooperating in roughly stable proportions once "
    "the culture matures. The sour flavor comes primarily from the "
    "bacteria's acid production, while the yeast is mostly responsible for "
    "the carbon dioxide that leavens the dough.",
    "Migratory birds navigate distances that would be extraordinary for any "
    "animal, let alone one weighing a few dozen grams, using a combination "
    "of senses that took decades of careful experiment to even partially "
    "untangle. They appear to read the sun's position, the pattern of "
    "polarized light it produces, the stars, and the Earth's magnetic "
    "field, cross-checking one against another when conditions make any "
    "single cue unreliable.",
]

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

block_text = " ".join(PASSAGES)
block_ids = tok.encode(block_text, add_special_tokens=False)
print(f"real-text block: {len(block_ids)} tokens", file=sys.stderr)

TARGET_CTX = 10240
prompt_ids: list[int] = []
while len(prompt_ids) < TARGET_CTX:
    prompt_ids.extend(block_ids)
prompt_ids = prompt_ids[:TARGET_CTX]
print(f"prompt tokens: {len(prompt_ids)} (target {TARGET_CTX})", file=sys.stderr)

from oracle.qwen36_vllm.vllm_compat import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.backends.laguna_dflash import DFlashEngine

CTX = len(prompt_ids)
# DRAFT_WINDOW=512, ~15 tokens/round best-case accept -> >=40 rounds needed
# to guarantee wraparound even at low real-world accept rates (each round
# writes at least 1 context-KV position regardless of accept count).
# 640 gives comfortable margin above the 512 capacity.
MAX_TOKENS = 640
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
print(f"prefill chunk size: {backend._prefill_chunk_tokens}", file=sys.stderr)
engine = DFlashEngine(backend)

tokens, stats = engine.generate_verify_only(
    prompt_ids, max_tokens=MAX_TOKENS, temperature=0.0, enable_prefix_cache=False
)

print(f"generated {len(tokens)} tokens", file=sys.stderr)
print(f"decoded: {tok.decode(tokens)!r}", file=sys.stderr)
print(f"acceptance_rate={stats['acceptance_rate']:.6f}", file=sys.stderr)
print(f"num_steps={stats['num_steps']}", file=sys.stderr)

torch.save(
    {
        "loader": LOADER,
        "prompt_len": CTX,
        "tokens": tokens,
        "acceptance_rate": stats["acceptance_rate"],
        "num_steps": stats["num_steps"],
    },
    OUT_PATH,
)
print(f"saved to {OUT_PATH}", file=sys.stderr)
