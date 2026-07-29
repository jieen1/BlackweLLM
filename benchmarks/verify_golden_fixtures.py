#!/usr/bin/env python3
"""Golden fixture parity verifier — 路线图 P0 #1 的裁判脚本。

重新录制 golden fixtures 并与已落盘的基准对比：
  * committed token 序列必须 bit-exact（greedy 确定性）
  * MTP 接受序列必须 bit-exact
  * GDN state L2 norm 允许 1e-3 相对误差（浮点累积）
  * Logits top-16 ids 必须 bit-exact，values 允许 1e-2 绝对误差

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.verify_golden_fixtures \
        [--fixture-dir benchmarks/fixtures/golden/] \
        [--num-prompts 4] [--decode-steps 50] [--concurrency 1]

Exit code 0 = PASS, 1 = FAIL (with detailed diff report).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

EOS_TOKEN_ID = 248046
RESERVED_PHYSICAL_SLOTS = 1


def capture_gdn_norms(runner, slot: int) -> list[float]:
    physical = slot + RESERVED_PHYSICAL_SLOTS
    norms = []
    for name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[name]
        norms.append(float(conv_state[physical].norm().item()))
        norms.append(float(ssm_state[physical].norm().item()))
    return norms


def run_recording(num_prompts: int, decode_steps: int, concurrency: int):
    """Run the same recording logic and return results dict."""

    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture_path = os.path.join(_REPO_ROOT, "benchmarks/fixtures/w1s_prompts.json")
    with open(fixture_path) as f:
        fixture = json.load(f)
    prompts = fixture["prompt_token_ids"][:num_prompts]
    prompt_len = len(prompts[0])

    MODEL = "unsloth/Qwen3.6-27B-NVFP4"
    K = 3
    SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}
    num_slots = max(concurrency, 1)
    blocks_per_slot = 4096
    max_model_len = prompt_len + decode_steps * (K + 1) + 2048

    config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        speculative_config=SPECULATIVE_CONFIG,
    )
    runner = DirectModelRunner(
        config,
        num_slots=num_slots,
        block_size=64,
        blocks_per_slot=blocks_per_slot,
        enable_block_table=True,
        enable_prefix_cache=False,
        enable_persistent_prefix_cache=False,
        enable_cudagraph=False,
    )

    # Monkey-patch to capture logits (E1 Phase 2: actual method lives on backend)
    captured_logits = []
    _target = runner.backend if hasattr(runner, "backend") else runner
    _orig = _target.verify_batch_spec

    def _hooked(*a, **kw):
        result = _orig(*a, **kw)
        logits = result[0] if isinstance(result, tuple) else result
        captured_logits.append(logits.detach().clone())
        return result

    _target.verify_batch_spec = _hooked

    slots = list(range(min(concurrency, len(prompts))))
    batch_prompts = prompts[:len(slots)]

    prefill_result = runner.mtp_prefill_batch(slots, batch_prompts)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}
    committed_tokens = {s: [anchors[s]] for s in slots}
    logits_record = {s: [] for s in slots}
    gdn_norms_record = {s: [] for s in slots}
    mtp_accept_record = {s: [] for s in slots}

    for s in slots:
        gdn_norms_record[s].append(capture_gdn_norms(runner, s))

    for step in range(decode_steps):
        captured_logits.clear()
        decisions = runner.mtp_verify_and_commit_batch(
            slots,
            {s: anchors[s] for s in slots},
            {s: drafts[s] for s in slots},
        )

        if captured_logits:
            verify_logits = captured_logits[0]
            qo_len = K + 1
            for i, s in enumerate(slots):
                last_pos = i * qo_len + K
                if last_pos < verify_logits.shape[0]:
                    topk_vals, topk_ids = verify_logits[last_pos].topk(16, dim=-1)
                    logits_record[s].append([
                        {"id": int(topk_ids[j].item()), "val": round(float(topk_vals[j].item()), 4)}
                        for j in range(16)
                    ])

        for s in slots:
            decision = decisions[s]
            new_tokens = decision["committed"]
            mtp_accept_record[s].append(decision.get("num_accepted", 0))
            for t in new_tokens:
                if t == EOS_TOKEN_ID:
                    break
                committed_tokens[s].append(t)
            anchors[s] = decision["next_anchor"]
            drafts[s] = decision["next_draft_tokens"]

        for s in slots:
            gdn_norms_record[s].append(capture_gdn_norms(runner, s))

    _target.verify_batch_spec = _orig

    return {
        "tokens": {str(s): committed_tokens[s] for s in slots},
        "logits": {str(s): logits_record[s] for s in slots},
        "gdn_norms": {str(s): gdn_norms_record[s] for s in slots},
        "mtp_accept": {str(s): mtp_accept_record[s] for s in slots},
    }


def compare_results(golden: dict, fresh: dict, gdn_rtol: float = 1e-3,
                    logits_atol: float = 1e-2) -> list[str]:
    """Compare golden vs fresh results. Returns list of failure messages."""
    failures = []

    # 1. Token sequence — must be bit-exact
    for slot_key in golden["tokens"]:
        g_tokens = golden["tokens"][slot_key]
        f_tokens = fresh["tokens"].get(slot_key, [])
        if g_tokens != f_tokens:
            # Find first divergence
            min_len = min(len(g_tokens), len(f_tokens))
            diverge_at = next(
                (i for i in range(min_len) if g_tokens[i] != f_tokens[i]),
                min_len,
            )
            failures.append(
                f"TOKEN MISMATCH slot {slot_key}: diverged at position {diverge_at}, "
                f"golden={g_tokens[diverge_at] if diverge_at < len(g_tokens) else 'EOF'} "
                f"fresh={f_tokens[diverge_at] if diverge_at < len(f_tokens) else 'EOF'} "
                f"(golden len={len(g_tokens)}, fresh len={len(f_tokens)})"
            )

    # 2. MTP acceptance — must be bit-exact
    for slot_key in golden["mtp_accept"]:
        g_acc = golden["mtp_accept"][slot_key]
        f_acc = fresh["mtp_accept"].get(slot_key, [])
        if g_acc != f_acc:
            min_len = min(len(g_acc), len(f_acc))
            diverge_at = next(
                (i for i in range(min_len) if g_acc[i] != f_acc[i]),
                min_len,
            )
            failures.append(
                f"MTP ACCEPT MISMATCH slot {slot_key}: diverged at step {diverge_at}, "
                f"golden={g_acc[diverge_at] if diverge_at < len(g_acc) else 'EOF'} "
                f"fresh={f_acc[diverge_at] if diverge_at < len(f_acc) else 'EOF'}"
            )

    # 3. GDN norms — relative tolerance
    for slot_key in golden["gdn_norms"]:
        g_norms = golden["gdn_norms"][slot_key]
        f_norms = fresh["gdn_norms"].get(slot_key, [])
        if len(g_norms) != len(f_norms):
            failures.append(
                f"GDN NORMS LENGTH MISMATCH slot {slot_key}: "
                f"golden={len(g_norms)} steps, fresh={len(f_norms)} steps"
            )
            continue
        max_rel_err = 0.0
        worst_step = -1
        worst_idx = -1
        for step_i, (g_step, f_step) in enumerate(zip(g_norms, f_norms)):
            for idx, (gv, fv) in enumerate(zip(g_step, f_step)):
                if gv == 0 and fv == 0:
                    continue
                denom = max(abs(gv), 1e-12)
                rel_err = abs(gv - fv) / denom
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                    worst_step = step_i
                    worst_idx = idx
        if max_rel_err > gdn_rtol:
            failures.append(
                f"GDN NORMS DRIFT slot {slot_key}: max rel_err={max_rel_err:.6f} "
                f"at step {worst_step}, idx {worst_idx} (tol={gdn_rtol})"
            )

    # 4. Logits top-k ids — must be bit-exact; values within tolerance
    for slot_key in golden["logits"]:
        g_logits = golden["logits"][slot_key]
        f_logits = fresh["logits"].get(slot_key, [])
        if len(g_logits) != len(f_logits):
            failures.append(
                f"LOGITS LENGTH MISMATCH slot {slot_key}: "
                f"golden={len(g_logits)} steps, fresh={len(f_logits)} steps"
            )
            continue
        for step_i, (g_step, f_step) in enumerate(zip(g_logits, f_logits)):
            g_ids = [e["id"] for e in g_step]
            f_ids = [e["id"] for e in f_step]
            if g_ids != f_ids:
                failures.append(
                    f"LOGITS TOP-K ID MISMATCH slot {slot_key} step {step_i}: "
                    f"golden top-3={g_ids[:3]} fresh top-3={f_ids[:3]}"
                )
                break
            max_val_err = max(
                abs(g_step[j]["val"] - f_step[j]["val"])
                for j in range(len(g_step))
            )
            if max_val_err > logits_atol:
                failures.append(
                    f"LOGITS VALUE DRIFT slot {slot_key} step {step_i}: "
                    f"max |Δval|={max_val_err:.6f} (tol={logits_atol})"
                )
                break

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify golden fixture parity")
    parser.add_argument("--fixture-dir", type=str,
                        default="benchmarks/fixtures/golden/")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--gdn-rtol", type=float, default=1e-3)
    parser.add_argument("--logits-atol", type=float, default=1e-2)
    args = parser.parse_args()

    fixture_dir = os.path.join(_REPO_ROOT, args.fixture_dir)

    # Load golden
    print(f"Loading golden fixtures from {fixture_dir}...")
    with open(os.path.join(fixture_dir, "golden_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(fixture_dir, "golden_tokens.json")) as f:
        g_tokens = json.load(f)
    with open(os.path.join(fixture_dir, "golden_logits.json")) as f:
        g_logits = json.load(f)
    with open(os.path.join(fixture_dir, "golden_gdn_norms.json")) as f:
        g_gdn = json.load(f)
    with open(os.path.join(fixture_dir, "golden_mtp_accept.json")) as f:
        g_mtp = json.load(f)

    golden = {
        "tokens": g_tokens,
        "logits": g_logits,
        "gdn_norms": g_gdn,
        "mtp_accept": g_mtp,
    }

    # Use params from golden meta if not overridden
    num_prompts = args.num_prompts or meta["params"]["num_prompts"]
    decode_steps = args.decode_steps or meta["params"]["decode_steps"]
    concurrency = args.concurrency or meta["params"]["concurrency"]

    print(f"Re-recording (num_prompts={num_prompts}, decode_steps={decode_steps}, "
          f"concurrency={concurrency})...")
    t0 = time.perf_counter()
    fresh = run_recording(num_prompts, decode_steps, concurrency)
    elapsed = time.perf_counter() - t0
    print(f"Re-recording done in {elapsed:.1f}s")

    # Compare
    print("Comparing...")
    failures = compare_results(golden, fresh, args.gdn_rtol, args.logits_atol)

    if failures:
        print(f"\n❌ PARITY CHECK FAILED ({len(failures)} issue(s)):")
        for f_msg in failures:
            print(f"  • {f_msg}")
        sys.exit(1)
    else:
        print("\n✅ PARITY CHECK PASSED")
        print("  Tokens: bit-exact ✓")
        print("  MTP acceptance: bit-exact ✓")
        print(f"  GDN norms: within rtol={args.gdn_rtol} ✓")
        print(f"  Logits top-k: ids bit-exact, values within atol={args.logits_atol} ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
