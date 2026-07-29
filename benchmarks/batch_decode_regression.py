"""Regression harness for ``DirectModelRunner.decode_batch``/``_forward_batch``
-- the real batched-metadata, single-forward-call replacement for looping
single ``decode()`` calls per slot (2026-07-16 batch-support round,
following the 2026-07-16 slot-0-reservation fix / Stage A-D closed loop
in notes/direct-model-runner-design.md).

Validates that a batch=N ``_forward_batch()`` call produces logits
BYTEWISE IDENTICAL to running each request individually through the
already-verified single-request path (``runner._forward(slot, [tok],
start_pos=kv_len, is_decode=True)``) -- on genuinely independent physical
slots initialized identically with the SAME prompt, never on the same
slot before/after (which would be confounded by cache-state mutation
between the two calls: decoding on a slot advances its kv_len and
overwrites the cache row the second call would need to reread).

Usage:
    python -m benchmarks.batch_decode_regression --batch 1
    python -m benchmarks.batch_decode_regression --batch 2
    python -m benchmarks.batch_decode_regression --batch 4
    python -m benchmarks.batch_decode_regression --batch 4 --varlen
    python -m benchmarks.batch_decode_regression --batch 2 --repeat 20
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"

PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at a temperature of",
    "The chemical symbol for gold is",
]
VARLEN_PROMPTS = [
    "Hi",
    "The capital of France is",
    "The largest planet in the solar system in our galaxy is",
    "Water, when heated at standard atmospheric pressure, boils at a "
    "temperature of approximately",
]


def _hash(tensor) -> str:
    return hashlib.sha256(tensor.float().cpu().numpy().tobytes()).hexdigest()[:16]


def _run_once(batch: int, varlen: bool) -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    # 2*batch logical slots: the first `batch` are the single-request
    # reference group, the second `batch` are the batched-call group --
    # kept fully independent (disjoint physical slots) so neither call's
    # cache-state mutation confounds the other.
    runner = DirectModelRunner(vllm_config, num_slots=2 * batch, block_size=64, blocks_per_slot=32)
    tok = AutoTokenizer.from_pretrained(MODEL)

    prompt_pool = VARLEN_PROMPTS if varlen else PROMPTS
    prompts = [prompt_pool[i % len(prompt_pool)] for i in range(batch)]
    prompt_ids = [tok.encode(p, add_special_tokens=False) for p in prompts]

    ref_slots = list(range(batch))
    batch_slots = list(range(batch, 2 * batch))

    next_tokens_ref = [runner.prefill(slot, ids) for slot, ids in zip(ref_slots, prompt_ids)]
    next_tokens_batch = [runner.prefill(slot, ids) for slot, ids in zip(batch_slots, prompt_ids)]

    if next_tokens_ref != next_tokens_batch:
        return {
            "passed": False,
            "error": "prefill greedy tokens diverged between twin groups",
            "next_tokens_ref": next_tokens_ref,
            "next_tokens_batch": next_tokens_batch,
        }

    kv_lengths = [runner.slot_kv_len[s] for s in ref_slots]
    if kv_lengths != [runner.slot_kv_len[s] for s in batch_slots]:
        return {"passed": False, "error": "twin groups have mismatched kv_len after prefill"}

    # Reference: single-request path, one call per slot (the pre-existing,
    # already-verified-correct behavior a naive Python-loop decode_batch
    # would reduce to).
    ref_logits = [
        runner._forward(slot, [tok_id], start_pos=kv_len, is_decode=True)[-1]
        for slot, tok_id, kv_len in zip(ref_slots, next_tokens_ref, kv_lengths)
    ]

    # New: one real batched call across all `batch` slots at once.
    batch_logits = runner._forward_batch(batch_slots, next_tokens_batch, kv_lengths)

    per_request = []
    all_match = True
    for i in range(batch):
        ref_row = ref_logits[i]
        batch_row = batch_logits[i]
        match = torch.equal(ref_row.float().cpu(), batch_row.float().cpu())
        all_match = all_match and match
        ref_next = int(ref_row.argmax(dim=-1).item())
        batch_next = int(batch_row.argmax(dim=-1).item())
        per_request.append(
            {
                "prompt": prompts[i],
                "ref_hash": _hash(ref_row),
                "batch_hash": _hash(batch_row),
                "ref_next_token": tok.decode([ref_next]),
                "batch_next_token": tok.decode([batch_next]),
                "match": match,
            }
        )

    return {"passed": all_match, "batch": batch, "varlen": varlen, "per_request": per_request}


def _run_subprocess(batch: int, varlen: bool) -> dict:
    args = [sys.executable, "-m", "benchmarks.batch_decode_regression", "--single-run-json", "--batch", str(batch)]
    if varlen:
        args.append("--varlen")
    proc = subprocess.run(
        args,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_RESULT: ") :])
    return {
        "passed": False,
        "error": "no result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once(args.batch, args.varlen)
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = _run_once(args.batch, args.varlen)
        print(result)
        return 0 if result["passed"] else 1

    results = [_run_subprocess(args.batch, args.varlen) for _ in range(args.repeat)]
    for i, r in enumerate(results):
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"run {i + 1}/{args.repeat}: {status}")
        if not r.get("passed"):
            print(f"  detail: {r}")

    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed (batch={args.batch}, varlen={args.varlen}) ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
