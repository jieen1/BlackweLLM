"""Fixed-input regression probe for the direct model runner's known-wrong
single-prefill output.

This is deliberately NOT another round of bottom-up kernel bisection --
per the 2026-07-16 direction, that diagnostic line is frozen. This script
exists to give the known bug a stable, repeatable, committed signature
(fixed prompt/seed/kernel selection, first-token id + a logits hash) so its
failure rate and exact behavior can be tracked over time, independent of
whatever ad hoc debugging happens next. See
notes/direct-model-runner-design.md's "Known independent defects" section
for the two real, specific, already-diagnosed anomalies (a Triton
causal_conv1d_fn cold-start bug, a CUTLASS SM120 GEMM race) neither of which
turned out to be the root cause of this wrong output on its own.

Usage (single run):
    ./.venv/bin/python -m benchmarks.single_prefill_regression

Usage (N repeated fresh-process runs, tallying pass/fail):
    ./.venv/bin/python -m benchmarks.single_prefill_regression --repeat 20
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
EXPECTED_TEXT = " Paris"
PROMPT = "The capital of France is"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"


def _run_once() -> dict:
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
    runner = DirectModelRunner(vllm_config, block_size=64, blocks_per_slot=32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    # Reuses the runner's own _forward() (not a hand-rolled duplicate of its
    # metadata/slot_mapping construction) so this regression probe always
    # reflects the runner's real current behavior, not a stale copy of it.
    logits = runner._forward(0, prompt_ids, start_pos=0, is_decode=False)

    last_logits = logits[-1].float().cpu()
    first_token_id = int(last_logits.argmax().item())
    first_token_text = tok.decode([first_token_id])
    logits_hash = hashlib.sha256(last_logits.numpy().tobytes()).hexdigest()[:16]
    top5 = last_logits.topk(5)
    return {
        "first_token_id": first_token_id,
        "first_token_text": first_token_text,
        "logits_hash": logits_hash,
        "top5_ids": top5.indices.tolist(),
        "top5_vals": [round(v, 3) for v in top5.values.tolist()],
        "passed": first_token_text.strip() == EXPECTED_TEXT.strip(),
    }


def _run_subprocess(index: int) -> dict:
    """Each repeat runs in its own fresh process -- matching every prior
    round's observation that this bug's behavior is sensitive to process-
    level state (Triton kernel cache warmth, CUDA context history)."""
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.single_prefill_regression", "--single-run-json"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=180,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_RESULT: ") :])
    return {
        "passed": False,
        "error": "no result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--single-run-json",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: used by the subprocess launcher
    )
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once()
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = _run_once()
        print(result)
        return 0 if result["passed"] else 1

    results = []
    for i in range(args.repeat):
        result = _run_subprocess(i)
        results.append(result)
        status = "PASS" if result.get("passed") else "FAIL"
        print(
            f"run {i + 1}/{args.repeat}: {status}  "
            f"first_token={result.get('first_token_text')!r}  "
            f"hash={result.get('logits_hash')}  "
            f"top5={result.get('top5_ids')}"
        )

    n_pass = sum(1 for r in results if r.get("passed"))
    unique_hashes = {r.get("logits_hash") for r in results if r.get("logits_hash")}
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    print(f"=== {len(unique_hashes)} distinct logits hash(es) across all runs ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
