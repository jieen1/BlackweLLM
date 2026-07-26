"""Round-level GPU memory growth diagnostic for D3 (2026-07-18 session
review's "GPU memory usage climbs within a single long-running process
and got dangerously close to OOM" finding).

Unlike the W1-S perf script's per-REP granularity (which already showed
memory climbing 45123 -> 72075 -> 79924 MiB across 3 reps, per
2026-07-17-post-ragged-round-next-steps.md 9.7, and 97227/97887 MiB in
the 2026-07-18 review's fresh run), this script samples
``torch.cuda.memory_allocated()``/``torch.cuda.memory_reserved()`` (the
allocator's own live-tensor-bytes and reserved-segment-bytes counters,
NOT just nvidia-smi) at EVERY individual decode/verify round boundary,
across MANY passes over the W1-S fixture in the SAME process -- fine
enough resolution to see whether growth is continuous (consistent with
the caching allocator reserving ever-larger segments to satisfy
monotonically growing per-round staging-tensor sizes as kv_len grows) or
happens in discrete jumps at specific points (prefill, batch boundary,
first-time-only, ...).

``memory_allocated()`` flat + ``memory_reserved()`` climbing = pure
allocator fragmentation, no true leak (the fix is preallocation, not
looking for a ref-holding bug). Both climbing together = live tensors
genuinely accumulating (a true leak -- need to find what's holding the
references).

Usage:
    python -m benchmarks.memory_growth_diag --passes 5 --concurrency 4 --fixture n16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _nvidia_smi_used_mib() -> int:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=5, help="full passes over the n16 fixture")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fixture", choices=["n16", "n128"], default="n16")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--cudagraph", action="store_true", default=True)
    parser.add_argument("--no-cudagraph", dest="cudagraph", action="store_false")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import W1_S_FIXTURE, W1_S_FIXTURE_N128, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture = {"n16": W1_S_FIXTURE, "n128": W1_S_FIXTURE_N128}[args.fixture]
    prompts = load_prompt_token_ids(fixture)
    if args.num_requests is not None:
        prompts = prompts[: args.num_requests]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, fixture.prompt_len + args.max_tokens + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    num_slots = 2 * args.concurrency if args.cudagraph else args.concurrency
    runner = DirectModelRunner(
        vllm_config,
        num_slots=num_slots,
        block_size=64,
        blocks_per_slot=640,
        enable_cudagraph=args.cudagraph,
    )
    torch.cuda.synchronize()

    print(
        f"after load: allocated={torch.cuda.memory_allocated()/2**20:.1f}MiB "
        f"reserved={torch.cuda.memory_reserved()/2**20:.1f}MiB "
        f"nvidia-smi={_nvidia_smi_used_mib()}MiB",
        flush=True,
    )

    rows: list[dict] = []
    round_idx = 0
    t_start = time.perf_counter()

    for pass_idx in range(args.passes):
        num_batches = (len(prompts) + args.concurrency - 1) // args.concurrency
        for batch_idx, batch_start in enumerate(range(0, len(prompts), args.concurrency)):
            batch = prompts[batch_start : batch_start + args.concurrency]
            num = len(batch)
            slots = list(range(num))
            for s in slots:
                if runner.slot_kv_len[s] != 0:
                    runner.reset_slot(s)

            committed_len = {s: 0 for s in slots}
            prefill_result = runner.mtp_prefill_batch(slots, batch)
            anchors = {s: prefill_result[s]["anchor"] for s in slots}
            drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}

            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            rows.append({
                "round_idx": round_idx, "pass": pass_idx, "batch": batch_idx,
                "event": "prefill", "allocated_mib": allocated / 2**20,
                "reserved_mib": reserved / 2**20,
                "elapsed_s": time.perf_counter() - t_start,
            })
            round_idx += 1

            active = list(slots)
            while active:
                decisions = runner.mtp_verify_and_commit_batch(
                    active, {s: anchors[s] for s in active}, {s: drafts[s] for s in active}
                )
                newly_finished = []
                for s in active:
                    decision = decisions[s]
                    n_acc = decision["num_accepted"]
                    committed_len[s] += n_acc + 1
                    anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]
                    if committed_len[s] >= args.max_tokens:
                        newly_finished.append(s)
                for s in newly_finished:
                    active.remove(s)

                if round_idx % 10 == 0:
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    rows.append({
                        "round_idx": round_idx, "pass": pass_idx, "batch": batch_idx,
                        "event": "verify", "allocated_mib": allocated / 2**20,
                        "reserved_mib": reserved / 2**20,
                        "elapsed_s": time.perf_counter() - t_start,
                    })
                round_idx += 1

            # End-of-batch sample (always, plus nvidia-smi cross-check
            # every few batches to catch anything the allocator counters
            # themselves might miss, e.g. other processes/driver overhead).
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            smi = _nvidia_smi_used_mib()
            print(
                f"pass {pass_idx} batch {batch_idx}/{num_batches - 1} round {round_idx}: "
                f"allocated={allocated/2**20:.1f}MiB reserved={reserved/2**20:.1f}MiB "
                f"nvidia-smi={smi}MiB elapsed={time.perf_counter()-t_start:.0f}s",
                flush=True,
            )
            rows.append({
                "round_idx": round_idx, "pass": pass_idx, "batch": batch_idx,
                "event": "batch_end", "allocated_mib": allocated / 2**20,
                "reserved_mib": reserved / 2**20, "nvidia_smi_mib": smi,
                "elapsed_s": time.perf_counter() - t_start,
            })

    print(torch.cuda.memory_summary(), flush=True)

    if args.out_csv:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.out_csv}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
