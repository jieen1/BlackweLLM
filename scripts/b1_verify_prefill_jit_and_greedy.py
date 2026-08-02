"""B1: full-model evidence for the extend-mode (prefill) JIT-recompile fix,
plus the greedy bit-exactness guard that fix has to survive.

Two things in one model load (the load is the expensive part):

1. **Prefill cost per novel prompt length.** Runs a series of prompts whose
   token counts are all distinct and none of which the process has seen before,
   and reports each one's prefill wall time. The fix is working iff exactly ONE
   of them pays sparkinfer's CuTe compile and every later one is fast --
   *including* the ones that cross ``packed_qo_len == 32``
   (``seq_len == 6`` at this model's ``gqa_group_size=6``), which is the
   boundary the pre-fix planner turned into a second compile bucket.

2. **Greedy bit-exactness.** Dumps the greedy token ids and the raw last-step
   logits for every prompt to ``--out``. Run once before the change and once
   after, then ``--compare a.pt b.pt``: any difference in ids or in the logit
   bytes is a regression, because nothing about this change is supposed to
   alter arithmetic for prompts whose tile geometry did not move.

Prompts are built by slicing real tokenizer output to an exact token count, so
"length" here is an exact, reproducible number rather than whatever a sentence
happened to tokenize to.

Run with:
    ~/.venvs/vllm/bin/python scripts/b1_verify_prefill_jit_and_greedy.py --out before.pt
    ~/.venvs/vllm/bin/python scripts/b1_verify_prefill_jit_and_greedy.py \
        --compare before.pt after.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _REPO)

import runtime  # noqa: E402

assert runtime.__file__.startswith(_REPO), runtime.__file__

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

MODEL_PATH = standard_checkpoint_path()
MAX_NEW_TOKENS = 16
MAX_SEQ_LEN = 512

# All distinct, none repeated, deliberately straddling seq_len==6 (the pre-fix
# cta_tile_q bucket boundary: packed_qo_len = seq_len * gqa_group_size(6), and
# the planner switched tiles at packed_qo_len > 32).
LENGTHS = [3, 5, 7, 11, 17, 29, 47, 83, 149, 251]

_SOURCE_TEXT = (
    "The quick brown fox jumps over the lazy dog while the committee reviews "
    "the quarterly report on distributed inference runtimes, paged attention "
    "kernels, speculative decoding acceptance rates, and the memory budget "
    "that a single ninety-six gigabyte accelerator can actually sustain when "
    "sixteen full-attention layers and forty-eight gated delta-net layers are "
    "interleaved across sixty-four transformer blocks in a mixture of experts "
    "configuration that quantizes dense projections to four bits while leaving "
    "attention projections at eight bits and the key value cache in bfloat16 "
    "for numerical headroom during long context prefill and steady state "
    "decode alike, which is the regime this runtime is being rebuilt for."
)


def run_prompt(model, input_ids: torch.Tensor) -> tuple[list[int], torch.Tensor, float]:
    state = model.new_generation_state(device=input_ids.device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hidden = model(input_ids, state)
    logits = model.compute_logits(hidden[:, -1:, :])
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    generated: list[int] = []
    next_token = int(logits[0, -1].argmax().item())
    generated.append(next_token)
    last_logits = logits[0, -1].detach().clone()
    for _ in range(MAX_NEW_TOKENS - 1):
        tok = torch.tensor([[next_token]], device=input_ids.device, dtype=torch.long)
        hidden = model(tok, state)
        logits = model.compute_logits(hidden)
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        last_logits = logits[0, -1].detach().clone()
    return generated, last_logits.to(torch.float32).cpu(), prefill_s


def compare(path_a: Path, path_b: Path) -> int:
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    if sorted(a) != sorted(b):
        print(f"FAIL: different prompt sets: {sorted(a)} vs {sorted(b)}")
        return 1
    bad = 0
    print(f"{'seq_len':>8} {'ids match':>10} {'logits bit-exact':>18} {'max_abs_diff':>13}")
    for key in sorted(a):
        ids_a, log_a = a[key]["ids"], a[key]["logits"]
        ids_b, log_b = b[key]["ids"], b[key]["logits"]
        ids_ok = ids_a == ids_b
        exact = torch.equal(log_a, log_b)
        diff = float((log_a - log_b).abs().max().item())
        if not ids_ok or not exact:
            bad += 1
        print(f"{key:>8} {str(ids_ok):>10} {str(exact):>18} {diff:>13.3e}")
    print("\nRESULT:", "IDENTICAL" if bad == 0 else f"{bad} prompt(s) differ")
    return 0 if bad == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "load with warmup_attention=True. Off by default so the table below "
            "shows the raw cost of the first compile; on, every row including the "
            "first should be fast and the compile shows up in the load line."
        ),
    )
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.out is None:
        parser.error("need --out or --compare")

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Repeated so the longest requested length is available; the content is
    # irrelevant, only the exact token count is.
    source = tokenizer(_SOURCE_TEXT * 3, return_tensors="pt").input_ids[0]
    assert int(source.shape[0]) >= max(LENGTHS), int(source.shape[0])

    t0 = time.time()
    model = load_qwen36_model(
        MODEL_PATH,
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=MAX_SEQ_LEN,
        warmup_attention=args.warmup,
    )
    print(f"load_qwen36_model(warmup={args.warmup}): {time.time()-t0:.1f}s")

    results: dict[int, dict[str, object]] = {}
    print(f"\n{'seq_len':>8} {'prefill_s':>10}  first 8 generated ids")
    print("-" * 60)
    for seq_len in LENGTHS:
        input_ids = source[:seq_len].unsqueeze(0).to("cuda")
        ids, logits, prefill_s = run_prompt(model, input_ids)
        results[seq_len] = {"ids": ids, "logits": logits}
        print(f"{seq_len:>8} {prefill_s:>10.3f}  {ids[:8]}")

    torch.save(results, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
