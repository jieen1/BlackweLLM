"""Phase 5 end-to-end bit-exact validation: real weights, real (long,
chunk-prefill-crossing) text, through the full self-built production
pipeline (main model + DFlash draft model), captured as a regression
snapshot.

Originally an A/B comparison against the real vLLM pipeline via
QSR_LAGUNA_MODEL_LOADER/QSR_DFLASH_MODEL_LOADER (see git history before
任务#46 for that version) -- both loader env vars and the real-vLLM
escape hatch they selected were removed from runtime/backends/laguna.py
and laguna_dflash.py once 任务#45's own validation confirmed the
self-built path bit-exact, so that comparison is no longer possible to
run. This script now just exercises the one production path directly;
kept as a standing bit-exact regression snapshot tool (compare a new
run's saved tokens/acceptance_rate/num_steps against a prior run's).

Uses a fresh, genuinely different real passage from every prior
validation script in this repo (not reused content, not a repeated
phrase -- per the block_size investigation's lesson that repeated/
degenerate text creates unusually dense near-tie decision points that
are a different, more adversarial kind of test than representative
production traffic). Sized to cross the main model's chunk-prefill
threshold and force multiple wraps of the draft model's DRAFT_WINDOW=512
ring, exactly like the phase 1/3 stress tests.

Usage:
  python _phase5_e2e_bitexact_validate.py out.pt
"""
import os
import sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime/.claude/worktrees/vllm-removal-phase1")

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase5_e2e_bitexact.pt"
print(f"out={OUT_PATH}", file=sys.stderr)

import torch

torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

# Fresh content -- distinct topics/register from every prior validation
# script's corpus, still real coherent prose (not synthetic, not a
# repeated phrase).
PASSAGES = [
    "Debugging a production incident under time pressure rewards a "
    "specific discipline: form a hypothesis, find the cheapest possible "
    "way to falsify it, and only then reach for the expensive diagnostic. "
    "Engineers who skip straight to adding logging everywhere often drown "
    "the signal they were looking for in noise from a dozen other "
    "subsystems, and the incident drags on not because the root cause was "
    "obscure but because the investigation itself became unmanageable.",
    "Municipal water systems in older cities frequently predate accurate "
    "record-keeping, so utilities sometimes discover the true layout of "
    "buried pipe networks only when something breaks. Replacing a single "
    "corroded section can require first mapping decades of undocumented "
    "patches, rerouted mains, and abandoned junctions that never made it "
    "into any surviving blueprint.",
    "A well-constructed sourdough loaf and a well-constructed legal brief "
    "share an odd structural property: both rely on a long, mostly "
    "invisible process of development that the final product only hints "
    "at. Readers and eaters alike judge the finished result, but the "
    "quality is set almost entirely by choices made well before either one "
    "was ever presented.",
    "Coastal erosion rarely proceeds at a steady rate; instead, decades of "
    "apparent stability can be undone by a single severe storm season. "
    "This makes long-term shoreline planning unusually difficult, since "
    "the historical average conceals exactly the kind of rare, high-impact "
    "event that actually determines where the coastline ends up.",
    "The gap between a chess engine's raw calculating power and a human "
    "grandmaster's intuition has narrowed to the point where the "
    "interesting comparison is no longer who wins, but how differently "
    "the two arrive at similar moves -- one by exhaustively pruning a vast "
    "search tree, the other by pattern recognition built from a lifetime "
    "of studied positions.",
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

from runtime.laguna_config import build_laguna_config
from runtime.backends.laguna import LagunaBackend
from runtime.backends.laguna_dflash import DFlashEngine

CTX = len(prompt_ids)
MAX_TOKENS = 640
BLOCK_SIZE = 64
max_model_len = CTX + MAX_TOKENS + 2048
bps = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 64

vllm_config = build_laguna_config(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=max_model_len,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
)
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
        "prompt_len": CTX,
        "tokens": tokens,
        "acceptance_rate": stats["acceptance_rate"],
        "num_steps": stats["num_steps"],
    },
    OUT_PATH,
)
print(f"saved to {OUT_PATH}", file=sys.stderr)
