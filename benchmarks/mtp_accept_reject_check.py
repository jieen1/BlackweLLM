"""Numerical verification of MTP greedy accept/reject boundary logic
against ``DirectModelRunner.verify_batch()`` -- constructs draft
sequences with a DELIBERATE, KNOWN mismatch at a specific position and
confirms the accept/reject logic (a) accepts exactly the tokens it should,
(b) discards everything from the first mismatch onward, and (c) recovers
the TRUE next token (the target model's own real prediction) at the
rejection point -- not garbage, and not silently different from what
non-speculative decoding would have produced. Not signal-probe: every
check compares actual token ids against a trusted reference computed via
the already-verified qo_len=1 decode path, not "does the output still
look plausible."

Draft convention (matches ``verify_batch``'s qo_len=K+1 layout):
``draft_tokens = [anchor, d_0, ..., d_{K-1}]`` -- the anchor is the
already-committed last real token; the K following entries are the
speculative continuation being verified. Verify position ``p``'s logits
predict "what comes after draft_tokens[p]", compared against
``draft_tokens[p+1]`` for ``p < K``; position ``K``'s logits (nothing left
to compare against) become the bonus token if every prior comparison
passed.

Usage:
    python -m benchmarks.mtp_accept_reject_check
    python -m benchmarks.mtp_accept_reject_check --repeat 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"
K = 3  # num_speculative_tokens, matching this project's real production MTP K=3


def _run_once() -> dict:

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import (
        DirectModelRunner,
        build_vllm_config,
        determine_accept_reject,
    )

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    # One slot per scenario (all-accept, reject-at-1, reject-at-0), each
    # prefilled identically and independently -- avoids any cross-scenario
    # state interference.
    runner = DirectModelRunner(vllm_config, num_slots=3, block_size=64, blocks_per_slot=640)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    # --- Establish the TRUSTED reference continuation via the
    # already-verified qo_len=1 decode path, on a throwaway slot never
    # used for the actual verify_batch() calls below. ---
    ref_slot_holder = runner.prefill(0, prompt_ids)  # slot 0 doubles as scratch below
    anchor = ref_slot_holder
    kv_len = runner.slot_kv_len[0]
    real_tokens = [anchor]
    cur = anchor
    cur_kv = kv_len
    for _ in range(K + 1):
        logits = runner._forward_batch([0], [cur], [cur_kv])
        cur = int(logits[0].argmax(dim=-1).item())
        cur_kv += 1
        real_tokens.append(cur)
    # real_tokens = [anchor, real_d0, real_d1, real_d2, real_bonus_if_continued]
    runner.reset_slot(0)

    decoy = 100  # an arbitrary token id, asserted below to differ from the real continuation
    scenarios = {
        "all_accept": [anchor, real_tokens[1], real_tokens[2], real_tokens[3]],
        "reject_at_1": [anchor, real_tokens[1], decoy, decoy],
        "reject_at_0": [anchor, decoy, decoy, decoy],
    }
    for name, draft in scenarios.items():
        if decoy in real_tokens[1:4]:
            return {"passed": False, "error": f"decoy token {decoy} collided with a real token, pick another"}

    results = {}
    all_passed = True
    for i, (name, draft) in enumerate(scenarios.items()):
        slot = i
        next_tok = runner.prefill(slot, prompt_ids)
        if next_tok != anchor:
            return {"passed": False, "error": f"prefill greedy token diverged for scenario {name}"}
        kv = runner.slot_kv_len[slot]
        verify_logits = runner.verify_batch([slot], [draft], [kv])

        decision = determine_accept_reject(draft, verify_logits)
        result = {"scenario": name, "draft": draft, **decision}

        if name == "all_accept":
            ok = decision["num_accepted"] == K and decision["rejected_at"] is None
            # The committed real drafts must equal real_tokens[1:1+K] exactly.
            ok = ok and decision["committed"][:K] == real_tokens[1 : 1 + K]
        elif name == "reject_at_1":
            ok = decision["num_accepted"] == 1 and decision["rejected_at"] == 1
            # Position 0 (real_1, accepted) then the TRUE recovery token at
            # the rejection point must equal real_tokens[2] (what the
            # target model actually predicts after real_1) -- NOT the
            # decoy, and not anything else.
            ok = ok and decision["committed"] == [real_tokens[1], real_tokens[2]]
        elif name == "reject_at_0":
            ok = decision["num_accepted"] == 0 and decision["rejected_at"] == 0
            ok = ok and decision["committed"] == [real_tokens[1]]
        else:
            ok = False

        result["ok"] = ok
        all_passed = all_passed and ok
        results[name] = result

    return {"passed": all_passed, "real_tokens": real_tokens, "scenarios": results}


def _run_subprocess() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.mtp_accept_reject_check", "--single-run-json"],
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
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
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

    results = [_run_subprocess() for _ in range(args.repeat)]
    for i, r in enumerate(results):
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"run {i + 1}/{args.repeat}: {status}")
        if not r.get("passed"):
            print(f"  detail: {r}")

    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
