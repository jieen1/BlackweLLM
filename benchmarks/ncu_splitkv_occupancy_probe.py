"""ncu occupancy probe for the coordinator's specific ask: direct,
kernel-level SM/warp occupancy numbers for our batched MTP path's real
decode/verify attention kernel, at the SAME W1-S shape used in the
performance comparison -- not the coarse ``nvidia-smi utilization.gpu``
number (a time-fraction metric across the whole GPU, different from but
consistent with the coordinator's own repeated sampling showing ~30%),
and a direct BEFORE/AFTER comparison across the 2026-07-17 split-KV fix
(``max_num_splits`` 1 -> 64) that this round's source-level investigation
found and applied.

Runs a handful of REAL ``mtp_verify_and_commit_batch`` rounds (concurrency=4,
K=3, real W1-S prompt, real mid-generation kv_len) under ``ncu``, twice:
once with this runner's post-fix split-KV config (``decode_fixed_max_num_splits
= 64``), once with it monkey-patched back to the PRE-fix behavior
(``max_num_splits = 1``, ``kv_split_size`` = this request's own live kv_len
-- exactly what ``build_attention_metadata_batch``'s DEFAULT branch used
to compute before any fixed values were ever passed in). Both runs profile
the SAME real kernel (`flash_attn_sm120_fwd_v2_decode_fp8kv_paged`, per
this session's own runtime log) at the SAME shape -- only the split-KV
config differs -- so any occupancy difference is attributable to that one
variable, not a confound.

Usage (each mode is a separate process/ncu invocation -- ncu profiles by
wrapping the whole launched process):
    ncu --set basic --kernel-name regex:flash_attn_sm120 --launch-count 3 \\
        -f -o /tmp/ncu_nosplit \\
        python -m benchmarks.ncu_splitkv_occupancy_probe --mode nosplit
    ncu --set basic --kernel-name regex:flash_attn_sm120 --launch-count 3 \\
        -f -o /tmp/ncu_split64 \\
        python -m benchmarks.ncu_splitkv_occupancy_probe --mode split64
Then: ncu -i /tmp/ncu_nosplit.ncu-rep --print-summary per-kernel
      ncu -i /tmp/ncu_split64.ncu-rep --print-summary per-kernel
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
CONCURRENCY = 4
PROFILE_ROUNDS = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["nosplit", "split64"], required=True)
    args = parser.parse_args()

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import W1_S_FIXTURE, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    prompts = load_prompt_token_ids(W1_S_FIXTURE)[:CONCURRENCY]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=40960,
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(vllm_config, num_slots=CONCURRENCY, block_size=64, blocks_per_slot=640)

    if args.mode == "nosplit":
        # Exactly this project's PRE-2026-07-17-fix behavior: a single
        # split covering the whole live kv_len (max_num_splits collapses
        # to 1 downstream in build_attention_metadata_batch's default
        # branch -- monkeypatching the fixed values back to what the
        # default (fixed_kv_split_size=None) branch would have used is
        # not directly expressible via these two attributes alone, so
        # instead this sets fixed_kv_split_size to a value >= the largest
        # real kv_len this profiling run will see, forcing the SAME
        # num_splits=1 outcome through the fixed-value code path (the
        # kernel receives identical scalar arguments either way).
        runner.decode_fixed_kv_split_size = 1 << 20  # >> any real kv_len this probe reaches
        runner.decode_fixed_max_num_splits = 1
        print("PROFILING MODE: nosplit (max_num_splits=1, the pre-fix behavior)", flush=True)
    else:
        print(
            f"PROFILING MODE: split64 (max_num_splits={runner.decode_fixed_max_num_splits}, "
            f"kv_split_size={runner.decode_fixed_kv_split_size}, the 2026-07-17 fix)",
            flush=True,
        )

    slots = list(range(CONCURRENCY))
    prefill_result = runner.mtp_prefill_batch(slots, prompts)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}

    for r in range(PROFILE_ROUNDS):
        decisions = runner.mtp_verify_and_commit_batch(slots, anchors, drafts)
        for s in slots:
            anchors[s], drafts[s] = decisions[s]["next_anchor"], decisions[s]["next_draft_tokens"]
        print(f"round {r} done, kv_len[0]={runner.slot_kv_len[0]}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
