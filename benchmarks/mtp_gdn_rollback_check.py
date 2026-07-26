"""Numerical twin verification of GDN state snapshot/restore
(``DirectModelRunner.snapshot_gdn_state``/``restore_gdn_state``) -- the
building block chosen (2026-07-17, "Option A" -- see
notes/direct-model-runner-design.md's MTP-semantics design section) for
MTP verify's GDN state commit/rollback problem: unlike attention's paged
KV cache (content-addressed by position, so "reject positions after N"
just means "stop advancing kv_len past N" -- the stale KV bytes past that
point are simply never read again), GDN's recurrent/chunked state has no
position index to roll back to. It is a single accumulated value per slot
that a forward call updates in place, so undoing "a few extra steps ran"
requires an explicit snapshot taken BEFORE those steps and a restore
after determining they must be discarded.

Not signal-probe: this drives two independent physical slots, prefilled
identically, through DIFFERENT numbers of real decode steps, and checks
that "detour then restore" produces logits BYTEWISE IDENTICAL to "never
took the detour at all" -- the decisive test for whether restore()
actually undoes the detour's real state changes, not just makes the
generated text look plausible afterward.

Usage:
    python -m benchmarks.mtp_gdn_rollback_check
    python -m benchmarks.mtp_gdn_rollback_check --repeat 3
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

LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4


def _run_once() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, _physical_slot, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    runner = DirectModelRunner(vllm_config, num_slots=2, block_size=64, blocks_per_slot=640)
    tok = AutoTokenizer.from_pretrained(MODEL)

    detour_slot = 0
    reference_slot = 1

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    tok_a_detour = runner.prefill(detour_slot, prompt_ids)
    tok_a_ref = runner.prefill(reference_slot, prompt_ids)
    if tok_a_detour != tok_a_ref:
        return {"passed": False, "error": "prefill greedy tokens diverged between twin slots"}

    # --- Step A: decode ONE real token (token_a) on both slots. ---
    kv0 = runner.slot_kv_len[detour_slot]
    logits_a_detour = runner._forward(detour_slot, [tok_a_detour], start_pos=kv0, is_decode=True)
    logits_a_ref = runner._forward(reference_slot, [tok_a_ref], start_pos=kv0, is_decode=True)
    if not torch.equal(logits_a_detour.cpu(), logits_a_ref.cpu()):
        return {"passed": False, "error": "twin slots diverged after identical step A -- test setup itself is broken"}

    kv_after_a = runner.slot_kv_len[detour_slot]  # == reference_slot's too
    tok_b = int(logits_a_detour[-1].argmax(dim=-1).item())

    # --- Snapshot detour_slot's GDN state right here, before any detour. ---
    snapshot = runner.snapshot_gdn_state(detour_slot)
    snapshot_kv_len = runner.slot_kv_len[detour_slot]

    # --- Detour: run several extra REAL decode steps on detour_slot only
    # (simulating "K draft/verify steps happened"), advancing its state
    # for real. ---
    cur = tok_b
    for _ in range(4):
        logits = runner._forward(detour_slot, [cur], start_pos=runner.slot_kv_len[detour_slot], is_decode=True)
        cur = int(logits[-1].argmax(dim=-1).item())

    # --- Restore: roll back GDN state AND kv_len bookkeeping to right
    # after step A -- simulating "all of that detour got rejected". ---
    runner.restore_gdn_state(detour_slot, snapshot)
    runner.slot_kv_len[detour_slot] = snapshot_kv_len

    # --- Decode token_b on BOTH slots now: detour_slot (post-restore) and
    # reference_slot (which never took any detour at all). If restore()
    # genuinely undid the detour, these must match bytewise. ---
    logits_b_detour = runner._forward(detour_slot, [tok_b], start_pos=kv_after_a, is_decode=True)
    logits_b_ref = runner._forward(reference_slot, [tok_b], start_pos=kv_after_a, is_decode=True)

    max_abs_diff = (logits_b_detour.float() - logits_b_ref.float()).abs().max().item()
    allclose = torch.allclose(logits_b_detour.float(), logits_b_ref.float(), atol=LOGITS_ATOL, rtol=LOGITS_RTOL)
    exact_equal = torch.equal(logits_b_detour.cpu(), logits_b_ref.cpu())

    # Also directly compare the GDN state tensors themselves post-restore
    # (the most direct check, independent of what the logits comparison
    # might mask).
    gdn_state_close = True
    gdn_max_diffs = []
    for name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[name]
        d_phys = _physical_slot(detour_slot)
        r_phys = _physical_slot(reference_slot)
        conv_diff = (conv_state[d_phys].float() - conv_state[r_phys].float()).abs().max().item()
        ssm_diff = (ssm_state[d_phys].float() - ssm_state[r_phys].float()).abs().max().item()
        gdn_max_diffs.append({"layer": name, "conv_diff": conv_diff, "ssm_diff": ssm_diff})
        if conv_diff > 1e-4 or ssm_diff > 1e-4:
            gdn_state_close = False

    passed = bool(allclose) and gdn_state_close
    return {
        "passed": passed,
        "logits_max_abs_diff": max_abs_diff,
        "logits_allclose": allclose,
        "logits_exact_equal": exact_equal,
        "gdn_state_close": gdn_state_close,
        "gdn_state_sample": gdn_max_diffs[:3],
        "num_gdn_layers_checked": len(runner.gdn_layer_names),
    }


def _run_subprocess() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.mtp_gdn_rollback_check", "--single-run-json"],
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
