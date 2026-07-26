"""D1 third follow-up diagnostic: does the decode/verify ROUND-LOOP cost
(mtp_verify_and_commit_batch, real Phase-2 spec-decode-GDN mechanism) scale
disproportionately with kv_len? notes/2026-07-18-session-review-and-next-
steps.md section 13.8 left the residual ~2.63x 16K/c=4 gap unexplained;
this project's own measured prefill-forward-call scaling (separately
profiled: only ~17% worse than linear for a 4x token-count increase, see
this session's nsys ledger) rules out the prefill forward pass itself as
the dominant driver -- this script checks the OTHER major wall-time
component directly: real per-round decode/verify cost at kv_len~16384 vs
kv_len~4096, post-prefill, same process/same model load.

PROFILING-ONLY. Does not modify any file under ``runtime/``. Calls the
real, unmodified ``mtp_prefill_batch``/``mtp_verify_and_commit_batch``
methods directly, continuing generation organically (each round's real
anchor/draft output feeds the next round, exactly as
``mtp_w1s_our_runtime_perf.py``'s own ``_run_batch_batched`` does) so this
is not a synthetic shape. Two independent slot groups (0-3 at ctx16k,
4-7 at ctx4k) in one ``DirectModelRunner`` (num_slots=8), one model load.

Usage:
    python -m benchmarks.d1_decode_round_kvlen_diag [--num-rounds 20]
    nsys profile -c cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt \\
        -o d1_decode_rounds --force-overwrite=true \\
        python -m benchmarks.d1_decode_round_kvlen_diag --num-rounds 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _nvidia_smi_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return int(out.strip())


def _run_rounds(torch, runner, label: str, slots: list[int], anchors, drafts, num_rounds: int, log: list):
    round_times = []
    for r in range(num_rounds):
        torch.cuda.nvtx.range_push(f"{label}_round_{r}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decisions = runner.mtp_verify_and_commit_batch(
            slots, {s: anchors[s] for s in slots}, {s: drafts[s] for s in slots}
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch.cuda.nvtx.range_pop()
        round_ms = 1000 * (t1 - t0)
        round_times.append(round_ms)
        n_acc = [decisions[s]["num_accepted"] for s in slots]
        kv_len_now = runner.slot_kv_len[slots[0]]
        print(f"  [{label}] round {r}: {round_ms:.3f}ms  num_accepted={n_acc}  kv_len={kv_len_now}", flush=True)
        anchors = {s: decisions[s]["next_anchor"] for s in slots}
        drafts = {s: decisions[s]["next_draft_tokens"] for s in slots}
    log.append({"label": label, "round_times_ms": round_times})
    mean_ms = sum(round_times) / len(round_times)
    print(f"=== {label}: mean round time over {num_rounds} rounds = {mean_ms:.3f}ms "
          f"(min={min(round_times):.3f} max={max(round_times):.3f}) ===", flush=True)
    return mean_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rounds", type=int, default=20)
    args = ap.parse_args()

    print("=== pre-run GPU check ===", flush=True)
    print(f"nvidia-smi memory.used: {_nvidia_smi_mib()} MiB", flush=True)

    import torch
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import D1_CTX16K_FIXTURE, W1_S_FIXTURE, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    prompts_16k = load_prompt_token_ids(D1_CTX16K_FIXTURE)[:4]
    prompts_4k = load_prompt_token_ids(W1_S_FIXTURE)[:4]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, D1_CTX16K_FIXTURE.prompt_len + 256 + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(
        vllm_config, num_slots=8, block_size=64, blocks_per_slot=640, enable_cudagraph=False,
    )
    print(f"\n=== after model load: nvidia-smi {_nvidia_smi_mib()} MiB ===", flush=True)

    slots_16k = [0, 1, 2, 3]
    slots_4k = [4, 5, 6, 7]

    print("\n=== prefill ctx16k (slots 0-3) ===", flush=True)
    t0 = time.perf_counter()
    prefill_16k = runner.mtp_prefill_batch(slots_16k, prompts_16k)
    torch.cuda.synchronize()
    print(f"prefill_16k wall={time.perf_counter()-t0:.3f}s", flush=True)

    print("\n=== prefill ctx4k (slots 4-7) ===", flush=True)
    t0 = time.perf_counter()
    prefill_4k = runner.mtp_prefill_batch(slots_4k, prompts_4k)
    torch.cuda.synchronize()
    print(f"prefill_4k wall={time.perf_counter()-t0:.3f}s", flush=True)

    profiling = True
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as e:  # pragma: no cover
        print(f"WARN cudaProfilerStart failed: {e}", flush=True)
        profiling = False

    log: list = []
    anchors_16k = {s: prefill_16k[s]["anchor"] for s in slots_16k}
    drafts_16k = {s: prefill_16k[s]["draft_tokens"] for s in slots_16k}
    mean_16k = _run_rounds(torch, runner, "kvlen16384", slots_16k, anchors_16k, drafts_16k, args.num_rounds, log)

    anchors_4k = {s: prefill_4k[s]["anchor"] for s in slots_4k}
    drafts_4k = {s: prefill_4k[s]["draft_tokens"] for s in slots_4k}
    mean_4k = _run_rounds(torch, runner, "kvlen4096", slots_4k, anchors_4k, drafts_4k, args.num_rounds, log)

    if profiling:
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception as e:  # pragma: no cover
            print(f"WARN cudaProfilerStop failed: {e}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"mean round time @ kv_len~16384: {mean_16k:.3f}ms", flush=True)
    print(f"mean round time @ kv_len~4096:  {mean_4k:.3f}ms", flush=True)
    print(f"ratio: {mean_16k/mean_4k:.3f}x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
