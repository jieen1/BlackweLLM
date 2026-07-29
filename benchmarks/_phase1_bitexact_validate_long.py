"""Phase 1 bit-exact validation, extended coverage: long context + chunked
prefill + many decode rounds (ring-buffer/block-boundary exercise).

任务#46 note: the QSR_LAGUNA_MODEL_LOADER=vllm arm below no longer runs
against real vLLM (runtime/backends/laguna.py's loader branch was
removed once 任务#45 confirmed both paths bit-exact); setting this env
var to "vllm" now silently falls through to the self-built path instead
of erroring. Historical record only -- see _phase5_e2e_bitexact_validate.py
for the current single-path regression snapshot tool.

Same rationale and method as _phase1_bitexact_validate.py, extended per
coordinator direction after the first (167-token, 32-step) validation
passed: today's two big investigations (block_size accept-rate regression,
fused_kv_scatter.py stride bug) both surfaced at KV-cache/ring-buffer/
chunked-prefill boundaries, not at short/simple inputs. This run targets
those same boundaries for the model-loading change specifically:

- CTX ~10240 tokens: crosses QSR_PREFILL_CHUNK's default 8192-token
  threshold, so prefill goes through the multi-chunk path
  (_prefill_chunk_ranges), not the single-shot path the first validation
  exercised.
- 128 decode rounds: SWA ring capacity is a handful of 64-token blocks
  (_ring_blocks_for_window), so 128 rounds forces multiple ring
  wrap-arounds, not just steady-state single-page decode.
- Real, non-repetitive text throughout (several distinct real paragraphs
  concatenated, not a single repeated phrase) -- still avoiding both
  synthetic/random data (today's fused_kv_scatter lesson) and the
  degenerate repeated-single-phrase pattern used for DFlash accept-rate
  stress tests (today's block_size investigation showed that pattern
  creates unusually dense near-tie decision points, which is a different
  kind of test than what this phase needs).
"""
import os
import sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime/.claude/worktrees/vllm-removal-phase1")

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase1_bitexact_long.pt"
LOADER = os.environ.get("QSR_LAGUNA_MODEL_LOADER", "vllm")
print(f"loader={LOADER} out={OUT_PATH}", file=sys.stderr)

import torch

torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

# Several distinct real, coherent passages (not one repeated phrase) --
# concatenated and, if needed to reach the target length, the whole
# concatenated block is cycled. Cycling a multi-paragraph block spanning
# thousands of tokens is a fundamentally different (much less degenerate)
# input than repeating a single short phrase every few tokens.
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
from runtime.sampling import SamplingParams

CTX = len(prompt_ids)
MAX_TOKENS = 128
max_model_len = CTX + MAX_TOKENS + 2048
BLOCK_SIZE = 64
bps = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 64

engine_args = EngineArgs(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=max_model_len,
    gpu_memory_utilization=0.88,
    enforce_eager=True,
    trust_remote_code=True,
)
vllm_config = engine_args.create_engine_config()
backend = LagunaBackend(
    vllm_config, num_slots=1, block_size=BLOCK_SIZE, blocks_per_slot=bps
)
print(f"prefill chunk size: {backend._prefill_chunk_tokens}", file=sys.stderr)

first_token = backend.prefill(0, prompt_ids)
print(f"first_token={first_token}", file=sys.stderr)

params = SamplingParams(temperature=0.0)
tokens = [first_token]
tok_id = first_token
for step in range(MAX_TOKENS - 1):
    (tok_id,) = backend.decode_batch_sampled(
        [0], [tok_id], [backend.slot_kv_len[0]], [params]
    )
    tokens.append(tok_id)

print(f"generated {len(tokens)} tokens", file=sys.stderr)
print(f"decoded: {tok.decode(tokens)!r}", file=sys.stderr)

torch.save(
    {
        "loader": LOADER,
        "prompt_len": CTX,
        "first_token": first_token,
        "tokens": tokens,
    },
    OUT_PATH,
)
print(f"saved to {OUT_PATH}", file=sys.stderr)
