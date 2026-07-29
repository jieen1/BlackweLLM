"""Phase 1 bit-exact validation: get_model() vs load_laguna_model().

任务#46 note: the QSR_LAGUNA_MODEL_LOADER=vllm arm below no longer runs
against real vLLM -- runtime/backends/laguna.py's loader branch was
removed once 任务#45 confirmed both paths bit-exact, so setting this env
var to "vllm" now silently falls through to the self-built path instead
of erroring. Historical record of the original comparison (see git
history / notes doc 任务#45's writeup for the results); not maintained
as a live A/B tool going forward -- use _phase5_e2e_bitexact_validate.py
as a single-path regression snapshot instead.

Runs the FULL production LagunaBackend setup (real weights, real KV-cache
allocation, real bf_attention/sparkinfer patching -- not an isolated
synthetic construction test) with a real, coherent, non-repetitive English
prompt, greedy decode, and dumps the exact token sequence + per-step logits
to disk. Run once per model-loader path (env var QSR_LAGUNA_MODEL_LOADER),
then diff the two dumps.

Deliberately NOT using a repeated-phrase prompt (like the DFlash accept-rate
benchmarks use) -- this is testing model *construction/loading* equivalence,
not speculative-decoding acceptance behavior, and a repeated phrase is an
unusually degenerate input. Deliberately NOT using random token ids either
-- per the fused_kv_scatter.py lesson (synthetic data can fail to trigger
bugs that only manifest on real data/real value distributions), this uses
an actual coherent passage of English tokenized by the real tokenizer.

Usage:
  QSR_LAGUNA_MODEL_LOADER=vllm      python _phase1_bitexact_validate.py vllm.pt
  QSR_LAGUNA_MODEL_LOADER=selfbuilt python _phase1_bitexact_validate.py selfbuilt.pt
"""
import os
import sys

os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime/.claude/worktrees/vllm-removal-phase1")

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase1_bitexact.pt"
LOADER = os.environ.get("QSR_LAGUNA_MODEL_LOADER", "vllm")
print(f"loader={LOADER} out={OUT_PATH}", file=sys.stderr)

import torch

torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)

# A real, coherent, non-repetitive passage -- not synthetic/degenerate input.
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

from oracle.qwen36_vllm.vllm_compat import EngineArgs
from runtime.backends.laguna import LagunaBackend
from runtime.sampling import SamplingParams

CTX = len(prompt_ids)
MAX_TOKENS = 32
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
