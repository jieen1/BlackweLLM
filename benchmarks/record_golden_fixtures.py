#!/usr/bin/env python3
"""Golden fixture recorder — V0 前置裁判（路线图 P0 #1）。

用冻结 prompt 集（w1s_prompts.json）在当前正常系统上录制 greedy decode 的：
  * committed token 序列（最强不变量：greedy 下确定性）
  * 每步 final logits top-16（token ids + values）
  * GDN state L2 norm（48 层 × conv+ssm，每步）
  * MTP 接受序列（每步 accepted count）

后续所有 kernel 替换（A2 GEMM、B7-V1 FLA 切换等）必须过此 fixture parity。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.record_golden_fixtures \
        --num-prompts 4 --decode-steps 50 --out benchmarks/fixtures/golden/

输出：
    golden_meta.json       — 录制参数、环境、checksum
    golden_tokens.json     — 每条 prompt 的 committed token 序列
    golden_logits.json     — 每步 top-16 logits (ids + values)
    golden_gdn_norms.json  — 每步 48 层 GDN state L2 norm (conv + ssm)
    golden_mtp_accept.json — 每步 MTP accepted count
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
    """Capture L2 norms of GDN conv_state and ssm_state for all 48 layers.
    Returns 96 values: [conv_norm_0, ssm_norm_0, conv_norm_1, ssm_norm_1, ...]."""
    physical = slot + RESERVED_PHYSICAL_SLOTS
    norms = []
    for name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[name]
        norms.append(round(float(conv_state[physical].norm().item()), 6))
        norms.append(round(float(ssm_state[physical].norm().item()), 6))
    return norms


def capture_logits_topk(logits_tensor, k: int = 16) -> list[dict]:
    """Capture top-k logit ids and values from a 1-D logits tensor [vocab]."""
    topk_vals, topk_ids = logits_tensor.topk(k, dim=-1)
    return [
        {"id": int(topk_ids[i].item()), "val": round(float(topk_vals[i].item()), 4)}
        for i in range(k)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record golden fixtures")
    parser.add_argument("--num-prompts", type=int, default=4,
                        help="Number of prompts from w1s fixture to use")
    parser.add_argument("--decode-steps", type=int, default=50,
                        help="Number of MTP verify rounds to record")
    parser.add_argument("--out", type=str, default="benchmarks/fixtures/golden/",
                        help="Output directory")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Batch size (1=single slot, 4=batched)")
    args = parser.parse_args()

    import torch

    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture_path = os.path.join(_REPO_ROOT, "benchmarks/fixtures/w1s_prompts.json")
    with open(fixture_path) as f:
        fixture = json.load(f)
    all_prompts = fixture["prompt_token_ids"]
    prompts = all_prompts[:args.num_prompts]
    prompt_len = len(prompts[0])
    print(f"Loaded {len(prompts)} prompts, prompt_len={prompt_len}")

    MODEL = "unsloth/Qwen3.6-27B-NVFP4"
    K = 3
    SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}
    num_slots = max(args.concurrency, 1)
    blocks_per_slot = 4096
    max_model_len = prompt_len + args.decode_steps * (K + 1) + 2048
    print(f"Building runner (num_slots={num_slots}, blocks_per_slot={blocks_per_slot})...")
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
    print(f"Runner ready. GDN layers: {len(runner.gdn_layer_names)}")

    # --- Monkey-patch verify_batch_spec to capture logits ---
    captured_logits: list[torch.Tensor] = []
    _patch_target = runner.backend if hasattr(runner, "backend") else runner
    _orig_verify_batch_spec = _patch_target.verify_batch_spec

    def _hooked_verify_batch_spec(*a, **kw):
        result = _orig_verify_batch_spec(*a, **kw)
        logits = result[0] if isinstance(result, tuple) else result
        captured_logits.append(logits.detach().clone())
        return result

    _patch_target.verify_batch_spec = _hooked_verify_batch_spec

    # --- Run recording ---
    slots = list(range(min(args.concurrency, len(prompts))))
    batch_prompts = prompts[:len(slots)]

    print(f"Prefilling {len(slots)} prompt(s)...")
    prefill_result = runner.mtp_prefill_batch(slots, batch_prompts)

    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}
    committed_tokens = {s: [anchors[s]] for s in slots}
    logits_record: dict[int, list] = {s: [] for s in slots}
    gdn_norms_record: dict[int, list] = {s: [] for s in slots}
    mtp_accept_record: dict[int, list] = {s: [] for s in slots}

    # Capture initial GDN norms after prefill
    for s in slots:
        gdn_norms_record[s].append(capture_gdn_norms(runner, s))

    print(f"Running {args.decode_steps} MTP verify rounds...")
    t0 = time.perf_counter()
    for step in range(args.decode_steps):
        captured_logits.clear()

        decisions = runner.mtp_verify_and_commit_batch(
            slots,
            {s: anchors[s] for s in slots},
            {s: drafts[s] for s in slots},
        )

        # Extract logits top-k from captured verify logits
        if captured_logits:
            verify_logits = captured_logits[0]
            qo_len = K + 1
            for i, s in enumerate(slots):
                last_pos = i * qo_len + K
                if last_pos < verify_logits.shape[0]:
                    logits_record[s].append(capture_logits_topk(verify_logits[last_pos]))
                else:
                    logits_record[s].append([])

        for s in slots:
            decision = decisions[s]
            new_tokens = decision["committed"]
            num_accepted = decision.get("num_accepted", 0)
            mtp_accept_record[s].append(num_accepted)

            for t in new_tokens:
                if t == EOS_TOKEN_ID:
                    break
                committed_tokens[s].append(t)

            anchors[s] = decision["next_anchor"]
            drafts[s] = decision["next_draft_tokens"]

        # GDN state norms after this verify step
        for s in slots:
            gdn_norms_record[s].append(capture_gdn_norms(runner, s))

        if (step + 1) % 10 == 0:
            total_committed = sum(len(committed_tokens[s]) for s in slots)
            elapsed = time.perf_counter() - t0
            print(f"  step {step+1}/{args.decode_steps}, "
                  f"total committed: {total_committed}, {elapsed:.1f}s")

    # Restore original method
    _patch_target.verify_batch_spec = _orig_verify_batch_spec

    # --- Save results ---
    os.makedirs(args.out, exist_ok=True)

    meta = {
        "date": datetime.datetime.now().isoformat(),
        "script": "benchmarks/record_golden_fixtures.py",
        "params": {
            "num_prompts": args.num_prompts,
            "decode_steps": args.decode_steps,
            "concurrency": args.concurrency,
            "prompt_len": prompt_len,
            "fixture_source": "benchmarks/fixtures/w1s_prompts.json",
            "cudagraph": False,
            "prefix_cache": False,
            "greedy": True,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
        },
        "fixture_checksum": hashlib.sha256(
            json.dumps(fixture["prompt_token_ids"][:args.num_prompts]).encode()
        ).hexdigest()[:16],
        "gdn_layers": len(runner.gdn_layer_names),
        "gdn_norms_per_step": len(gdn_norms_record[slots[0]][0]) if gdn_norms_record[slots[0]] else 0,
    }
    with open(os.path.join(args.out, "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tokens_out = {str(s): committed_tokens[s] for s in slots}
    with open(os.path.join(args.out, "golden_tokens.json"), "w") as f:
        json.dump(tokens_out, f)

    logits_out = {str(s): logits_record[s] for s in slots}
    with open(os.path.join(args.out, "golden_logits.json"), "w") as f:
        json.dump(logits_out, f)

    mtp_out = {str(s): mtp_accept_record[s] for s in slots}
    with open(os.path.join(args.out, "golden_mtp_accept.json"), "w") as f:
        json.dump(mtp_out, f)

    gdn_out = {str(s): gdn_norms_record[s] for s in slots}
    with open(os.path.join(args.out, "golden_gdn_norms.json"), "w") as f:
        json.dump(gdn_out, f)

    print("\n=== Golden Fixtures Recorded ===")
    print(f"  Output: {args.out}")
    for s in slots:
        n_tokens = len(committed_tokens[s])
        avg_accept = sum(mtp_accept_record[s]) / max(len(mtp_accept_record[s]), 1)
        print(f"  Slot {s}: {n_tokens} committed tokens, "
              f"avg MTP accept={avg_accept:.2f}")
    gdn_steps = len(gdn_norms_record[slots[0]])
    gdn_vals = len(gdn_norms_record[slots[0]][0]) if gdn_steps > 0 else 0
    print(f"  GDN norms: {gdn_steps} steps × {gdn_vals} values/step")
    print(f"  Logits top-k: {len(logits_record[slots[0]])} steps × 16 entries")
    print("  Files: golden_meta.json, golden_tokens.json, golden_logits.json, "
          "golden_mtp_accept.json, golden_gdn_norms.json")


if __name__ == "__main__":
    main()
