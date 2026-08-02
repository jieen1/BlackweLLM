"""Full-model guard for the Laguna prefill ``cta_tile_q`` fix: real weights,
real FP8 KV, both layer groups (full-attention + the ``window_left=511`` SWA
group), greedy output compared bit-for-bit before vs after.

Laguna is this runtime's only production model, so "the kernel A/B says the
tiling change is arithmetic-free" is not enough on its own -- this runs the
whole 48-layer stack and dumps, per prompt:

* the raw last-position prefill logits (the tensor the change can actually
  touch), and
* the greedy token ids for a short continuation (what a user would see).

Both configurations run in ONE process against ONE loaded model: pass A with
the plan budget removed from every prefill workspace (the pre-fix planner
call, which picks ``cta_tile_q`` from the live query length), pass B with it
in place (post-fix, capacity-derived). The fixed-capacity workspace object is
deliberately NOT rebuilt between passes -- its key does not include the
budget -- so the only thing that differs between A and B is the plan, which
is exactly the variable under test. One model load, and no chance of the two
sides differing because of weights, allocation order or a stale checkout.

The compile-count / wall-time evidence lives in the model-free
``scripts/laguna_probe_extend_jit_buckets.py`` instead, which can be run in
either configuration cheaply; prefill times are still printed here, but only
pass A's are meaningful (pass B reuses pass A's compiles).

Run with:
    ~/.venvs/vllm/bin/python scripts/laguna_verify_prefill_jit_and_greedy.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _REPO)

import runtime  # noqa: E402

assert runtime.__file__.startswith(_REPO), runtime.__file__

import torch  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
MAX_NEW_TOKENS = 24
BLOCKS_PER_SLOT = 1024  # block_size=64 -> 64K tokens, far above what we use

# All distinct, straddling every ``cta_tile_q`` boundary this geometry has
# (gqa_group_size=6, head_dim=128 -> the planner flips tiles twice).
LENGTHS = [2, 5, 7, 9, 11, 17, 33, 65, 129, 257]

def make_token_ids(count: int) -> list[int]:
    """A fixed, reproducible id sequence.

    Deliberately not tokenizer output: this checks that two code paths give
    bit-identical numbers for identical inputs, which does not depend on the
    prompt meaning anything, and ``AutoTokenizer.from_pretrained`` on this
    checkpoint trips a rope-config validator in the installed transformers.
    Ids stay well inside the vocabulary and away from the special-token block.
    """
    with open(Path(MODEL_PATH) / "config.json") as fh:
        vocab_size = int(json.load(fh)["vocab_size"])
    span = vocab_size - 2000
    return [1000 + (i * 7919 + 13) % span for i in range(count)]


def build_backend():
    engine_args = EngineArgs(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=131072,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        trust_remote_code=True,
        moe_backend=os.environ.get("QSR_MOE_BACKEND", "marlin"),
    )
    from runtime.backends.laguna import LagunaBackend

    return LagunaBackend(
        engine_args.create_engine_config(),
        num_slots=1,
        block_size=64,
        blocks_per_slot=BLOCKS_PER_SLOT,
    )


def run_prompt(backend, prompt_ids: list[int]):
    """Return (last prefill logits, greedy ids, prefill wall seconds).

    Two passes on a fresh slot: one that keeps the raw prefill logits (the
    production prefill entry point only hands back an argmax), one that runs
    the ordinary greedy loop. Both start from ``reset_slot``, so neither sees
    state left by the other.
    """
    backend.reset_slot(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = backend._prefill_with_swa_scratch(0, prompt_ids)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    last_logits = logits[-1].detach().to(torch.float32).cpu().clone()

    backend.reset_slot(0)
    first = backend.prefill(0, prompt_ids)
    ids = [first]
    token = first
    for _ in range(MAX_NEW_TOKENS - 1):
        token = backend.decode(0, token)
        ids.append(token)
    return last_logits, ids, prefill_s


def iter_prefill_workspaces(backend):
    seen: set[int] = set()
    for layer_names in backend._layer_groups.values():
        impl = backend.static_forward_context[layer_names[0]].impl
        ws = getattr(impl, "_prefill_ws", None)
        if ws is None or id(ws) in seen:
            continue
        seen.add(id(ws))
        yield ws


def run_pass(backend, source, label: str) -> dict[int, dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    print(f"\n=== pass {label} ===")
    print(f"{'len':>6} {'prefill_s':>10}  first 8 greedy ids")
    print("-" * 60)
    for seq_len in LENGTHS:
        logits, ids, prefill_s = run_prompt(backend, list(source[:seq_len]))
        results[seq_len] = {"ids": ids, "logits": logits}
        print(f"{seq_len:>6} {prefill_s:>10.3f}  {ids[:8]}")
    return results


def compare(a, b) -> int:
    bad = 0
    print(f"\n{'len':>6} {'ids match':>10} {'logits bit-exact':>18} {'max_abs_diff':>13}")
    for key in sorted(a):
        ids_ok = a[key]["ids"] == b[key]["ids"]
        exact = torch.equal(a[key]["logits"], b[key]["logits"])
        diff = float((a[key]["logits"] - b[key]["logits"]).abs().max().item())
        if not ids_ok or not exact:
            bad += 1
        print(f"{key:>6} {str(ids_ok):>10} {str(exact):>18} {diff:>13.3e}")
    print("\nRESULT:", "IDENTICAL" if bad == 0 else f"{bad} prompt(s) differ")
    return 0 if bad == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="optional: also save both passes")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="call warmup_paged_attention_shapes() before the passes (production path)",
    )
    args = parser.parse_args()

    source = make_token_ids(max(LENGTHS))

    t0 = time.time()
    backend = build_backend()
    print(f"LagunaBackend loaded in {time.time()-t0:.1f}s")
    if args.warmup:
        t0 = time.time()
        backend.warmup_paged_attention_shapes(slot=0)
        print(f"warmup_paged_attention_shapes: {time.time()-t0:.1f}s")

    workspaces = list(iter_prefill_workspaces(backend))
    budgets = [ws._extend_plan_budget for ws in workspaces]
    print(f"prefill workspace groups: {len(workspaces)}")
    assert workspaces and all(b is not None for b in budgets), (
        "expected every prefill workspace to carry an extend plan budget; "
        "this script is comparing against a build that does not have the fix"
    )

    for ws in workspaces:
        ws._extend_plan_budget = None
    before = run_pass(backend, source, "A: no plan budget (pre-fix planner)")

    for ws, budget in zip(workspaces, budgets):
        ws._extend_plan_budget = budget
    after = run_pass(backend, source, "B: capacity-derived cta_tile_q (post-fix)")

    if args.out is not None:
        torch.save({"before": before, "after": after}, args.out)
        print(f"wrote {args.out}")
    return compare(before, after)


if __name__ == "__main__":
    raise SystemExit(main())
