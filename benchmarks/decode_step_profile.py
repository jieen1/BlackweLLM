"""Deep per-step profiling of the MTP decode path.

Measures WHERE time goes in each decode step at 128K/c=4 (warm cache-hit)
and at short context (W1-S, ~4K tokens) for comparison.

Sub-components measured per step (via monkey-patched CUDA events):
  - t_verify: target model verify forward (attention + GDN + GEMM + logits)
  - t_accept_reject: accept/reject logic (GPU argmax + host round-trip)
  - t_draft_step0: MTP draft model step-0 resync
  - t_draft_continuation: K-1 draft continuation steps
  - t_metadata: attention/GDN metadata construction (CPU)
  - t_total_step: total GPU-synchronized wall time per step

Additionally, torch.profiler captures kernel-level breakdown for a subset
of steps to categorize GPU time into attention/GEMM/GDN/other.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.decode_step_profile --fixture ctx128k --concurrency 4
    /home/bot/.venvs/vllm/bin/python -m benchmarks.decode_step_profile --fixture w1s --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}
_MODEL_MAX_POSITION_EMBEDDINGS = 262144
DEFAULT_SUFFIX_LEN = 10240
NUM_PROFILE_STEPS = 50
NUM_PROFILER_STEPS = 5  # steps captured with torch.profiler for kernel breakdown


@dataclass
class StepTimings:
    verify_ms: float = 0.0
    accept_reject_ms: float = 0.0
    draft_step0_ms: float = 0.0
    draft_continuation_ms: float = 0.0
    metadata_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def python_overhead_ms(self) -> float:
        accounted = (self.verify_ms + self.accept_reject_ms +
                     self.draft_step0_ms + self.draft_continuation_ms +
                     self.metadata_ms)
        return max(0.0, self.total_ms - accounted)


def _gpu_mem_mib() -> dict:
    import torch
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
    }


def _nvidia_smi_mem_mib() -> int:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception:
        return -1


def _clear_persistent_cache(runner) -> None:
    for s in range(runner.num_slots):
        if runner.slot_kv_len[s] != 0 or runner.block_table[s]:
            runner.reset_slot(s)
    runner.block_pool.hash_to_block.clear()
    for b in runner.block_pool.blocks:
        b.block_hash = None
    runner.gdn_ckpt_meta.clear()
    runner._gdn_ckpt_by_hash.clear()
    runner._gdn_ckpt_free = list(range(runner.gdn_ckpt_max_checkpoints))
    runner._gdn_ckpt_lru.clear()
    for s in range(runner.num_slots):
        runner.slot_block_hashes[s] = []
        runner.slot_published_blocks[s] = 0
        runner.slot_committed_tokens[s] = []


def _make_suffix(base_prompt: list[int], suffix_len: int, salt: int = 0) -> list[int]:
    last_token = base_prompt[-1]
    return [(last_token + 1 + salt + i) % 151936 for i in range(suffix_len)]


def _build_runner(fixture_prompt_len: int, suffix_len: int, max_tokens: int,
                  concurrency: int, gpu_mem_util: float, blocks_margin: int):
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    blocks_per_slot = -(-(fixture_prompt_len + suffix_len + max_tokens + 8) // 16) + blocks_margin
    max_model_len = min(
        fixture_prompt_len + suffix_len + max_tokens + 2048,
        _MODEL_MAX_POSITION_EMBEDDINGS,
    )
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        speculative_config=SPECULATIVE_CONFIG,
    )
    runner = DirectModelRunner(
        vllm_config,
        num_slots=concurrency,
        block_size=64,
        blocks_per_slot=blocks_per_slot,
        enable_block_table=True,
        enable_prefix_cache=True,
        enable_persistent_prefix_cache=True,
        enable_cudagraph=True,
    )
    runner.precapture_cuda_graphs(batch_sizes=list(range(1, concurrency + 1)))
    return runner, blocks_per_slot, max_model_len


def _setup_warm_scenario(runner, prefixes: list[list[int]], suffix_len: int,
                         chunk_size: int | None) -> tuple[dict, dict]:
    """Set up the warm cache-hit scenario: cold-populate prefixes, then
    warm-prefill prefix+suffix for each slot. Returns (anchors, drafts)."""
    import torch
    slots = list(range(len(prefixes)))

    # Phase 1: cold-populate each prefix (populates persistent cache)
    _clear_persistent_cache(runner)
    for s in slots:
        runner.mtp_prefill_with_cache([s], [prefixes[s]], chunk_size)
    torch.cuda.synchronize()

    # Phase 2: reset slots, warm-prefill prefix+suffix (cache hit)
    for s in slots:
        runner.reset_slot(s)
    warm_prompts = [prefixes[i] + _make_suffix(prefixes[i], suffix_len, salt=i)
                    for i in range(len(prefixes))]
    pr = runner.mtp_prefill_with_cache(slots, warm_prompts, chunk_size)
    torch.cuda.synchronize()

    anchors = {s: pr[s]["anchor"] for s in slots}
    drafts = {s: pr[s]["draft_tokens"] for s in slots}
    return anchors, drafts


def _setup_short_scenario(runner, prompts: list[list[int]]) -> tuple[dict, dict]:
    """Set up short-context scenario: simple prefill of each prompt."""
    import torch
    slots = list(range(len(prompts)))
    _clear_persistent_cache(runner)
    pr = runner.mtp_prefill_with_cache(slots, prompts, None)
    torch.cuda.synchronize()
    anchors = {s: pr[s]["anchor"] for s in slots}
    drafts = {s: pr[s]["draft_tokens"] for s in slots}
    return anchors, drafts


def _run_profiled_decode_steps(runner, slots, anchors, drafts, num_steps):
    """Run decode steps with per-component CUDA event timing via monkey-patching."""
    import torch

    from runtime.direct_model_runner import determine_accept_reject_batch

    timings: list[StepTimings] = []
    cur_anchors = dict(anchors)
    cur_drafts = dict(drafts)
    total_committed = 0
    total_draft_tokens = 0

    # Monkey-patch sub-components for timing
    orig_verify_batch_spec = (runner.backend if hasattr(runner, "backend") else runner).verify_batch_spec
    orig_accept_reject = determine_accept_reject_batch
    orig_mtp_sync_and_propose_batch = (runner.backend if hasattr(runner, "backend") else runner)._mtp_sync_and_propose_batch
    orig_mtp_run_continuation_steps = runner._mtp_run_continuation_steps

    step_timing = StepTimings()

    def timed_verify_batch_spec(*args, **kwargs):
        nonlocal step_timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = orig_verify_batch_spec(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        step_timing.verify_ms += start.elapsed_time(end)
        return result

    def timed_accept_reject(slots_arg, drafts_arg, logits, k_arg):
        nonlocal step_timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = orig_accept_reject(slots_arg, drafts_arg, logits, k_arg)
        end.record()
        torch.cuda.synchronize()
        step_timing.accept_reject_ms += start.elapsed_time(end)
        return result

    def timed_mtp_sync_and_propose_batch(*args, **kwargs):
        nonlocal step_timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = orig_mtp_sync_and_propose_batch(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        # This includes both step0 and continuation
        step_timing.draft_step0_ms += start.elapsed_time(end)
        return result

    # Patch
    _target = runner.backend if hasattr(runner, "backend") else runner
    _target.verify_batch_spec = timed_verify_batch_spec
    import runtime.direct_model_runner as rm
    rm.determine_accept_reject_batch = timed_accept_reject
    _target._mtp_sync_and_propose_batch = timed_mtp_sync_and_propose_batch

    try:
        for step_idx in range(num_steps):
            step_timing = StepTimings()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            decs = runner.mtp_verify_and_commit_batch(slots, cur_anchors, cur_drafts)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_timing.total_ms = (t1 - t0) * 1000.0

            for s in slots:
                d = decs[s]
                total_committed += d["num_accepted"] + 1
                total_draft_tokens += len(cur_drafts[s])
                cur_anchors[s] = d["next_anchor"]
                cur_drafts[s] = d["next_draft_tokens"]

            timings.append(step_timing)
    finally:
        # Restore originals
        _target.verify_batch_spec = orig_verify_batch_spec
        rm.determine_accept_reject_batch = orig_accept_reject
        _target._mtp_sync_and_propose_batch = orig_mtp_sync_and_propose_batch

    return timings, total_committed, total_draft_tokens


def _run_profiler_steps(runner, slots, anchors, drafts, num_steps):
    """Run a few steps under torch.profiler to get kernel-level breakdown."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    cur_anchors = dict(anchors)
    cur_drafts = dict(drafts)

    # Warmup
    for _ in range(3):
        decs = runner.mtp_verify_and_commit_batch(slots, cur_anchors, cur_drafts)
        for s in slots:
            cur_anchors[s] = decs[s]["next_anchor"]
            cur_drafts[s] = decs[s]["next_draft_tokens"]
    torch.cuda.synchronize()

    # Profiled steps
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_steps):
            decs = runner.mtp_verify_and_commit_batch(slots, cur_anchors, cur_drafts)
            for s in slots:
                cur_anchors[s] = decs[s]["next_anchor"]
                cur_drafts[s] = decs[s]["next_draft_tokens"]
        torch.cuda.synchronize()

    return prof


def _categorize_kernel(name: str) -> str:
    """Categorize a CUDA kernel name into a component bucket."""
    name_lower = name.lower()
    # GDN / SSM recurrent kernels
    if any(k in name_lower for k in ["delta_rule", "gating", "ssm", "gdn",
                                      "fused_sigmoid", "recurrent"]):
        return "gdn"
    # Attention kernels (SM120 GQA, flash attention, splitkv)
    if any(k in name_lower for k in ["gqa", "flash", "fmha", "attention",
                                      "splitkv", "split_kv", "paged",
                                      "sm120", "decode_kernel"]):
        return "attention"
    # GEMM / matmul / linear layers
    if any(k in name_lower for k in ["gemm", "cutlass", "cublas", "nvfp4",
                                      "matmul", "mma", "warp", "sm90_xmma",
                                      "ampere_", "ffma", "hmma"]):
        return "gemm"
    # Embedding / layernorm / activation / elementwise
    if any(k in name_lower for k in ["embedding", "layer_norm", "layernorm",
                                      "rmsnorm", "silu", "gelu", "softmax",
                                      "elementwise", "vectorized", "copy",
                                      "fill", "cat_", "index"]):
        return "other_compute"
    return "other"


def _analyze_profiler(prof, num_steps: int) -> dict:
    """Analyze profiler key_averages into categorized breakdown."""
    events = prof.key_averages()
    categories: dict[str, float] = {
        "attention": 0.0, "gemm": 0.0, "gdn": 0.0,
        "other_compute": 0.0, "other": 0.0,
    }
    total_cuda_ms = 0.0
    top_kernels: list[tuple[str, float, int]] = []

    for evt in events:
        if evt.device_time_total > 0:
            cuda_ms = evt.device_time_total / 1000.0  # us -> ms
            cat = _categorize_kernel(evt.key)
            categories[cat] += cuda_ms
            total_cuda_ms += cuda_ms
            top_kernels.append((evt.key, cuda_ms, evt.count))

    # Sort by time descending
    top_kernels.sort(key=lambda x: -x[1])

    # Per-step averages
    per_step = {k: round(v / num_steps, 4) for k, v in categories.items()}
    per_step["total_cuda_ms"] = round(total_cuda_ms / num_steps, 4)

    pct = {}
    if total_cuda_ms > 0:
        for k, v in categories.items():
            pct[k] = round(100.0 * v / total_cuda_ms, 2)

    return {
        "per_step_ms": per_step,
        "pct_of_cuda_time": pct,
        "total_cuda_ms_all_steps": round(total_cuda_ms, 3),
        "num_profiled_steps": num_steps,
        "top_15_kernels": [
            {"name": name[:100], "total_ms": round(ms, 3), "count": cnt}
            for name, ms, cnt in top_kernels[:15]
        ],
    }


def _report(timings: list[StepTimings], total_committed: int,
            total_draft: int, num_steps: int, concurrency: int,
            profiler_analysis: dict | None, label: str) -> dict:
    """Print and return the profiling report."""
    import statistics

    n = len(timings)
    # Skip first 5 steps as warmup
    warmup = min(5, n // 4)
    stable = timings[warmup:]
    n_stable = len(stable)

    def stats(values):
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "total": 0}
        return {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "total": round(sum(values), 3),
        }

    verify_stats = stats([t.verify_ms for t in stable])
    accept_stats = stats([t.accept_reject_ms for t in stable])
    draft_stats = stats([t.draft_step0_ms for t in stable])
    total_stats = stats([t.total_ms for t in stable])
    overhead_stats = stats([t.python_overhead_ms for t in stable])

    mean_total = total_stats["mean"]
    committed_per_step = total_committed / num_steps
    steps_per_sec = 1000.0 / mean_total if mean_total > 0 else 0
    tokens_per_sec = committed_per_step * steps_per_sec

    # Percentage breakdown (of mean total)
    pct = {}
    if mean_total > 0:
        pct["verify"] = round(100.0 * verify_stats["mean"] / mean_total, 2)
        pct["accept_reject"] = round(100.0 * accept_stats["mean"] / mean_total, 2)
        pct["draft (step0+continuation)"] = round(100.0 * draft_stats["mean"] / mean_total, 2)
        pct["python_overhead"] = round(100.0 * overhead_stats["mean"] / mean_total, 2)

    print(f"\n{'='*78}")
    print(f"DECODE STEP PROFILE — {label}")
    print(f"{'='*78}")
    print(f"  Steps measured: {n} (warmup={warmup}, stable={n_stable})")
    print(f"  Concurrency: {concurrency}, K={K}")
    print(f"  Committed tokens/step: {committed_per_step:.2f}")
    print(f"  Steps/sec: {steps_per_sec:.2f}")
    print(f"  Accepted tokens/sec: {tokens_per_sec:.2f}")
    print(f"  Acceptance rate: {total_committed / (total_committed + total_draft):.4f}"
          if (total_committed + total_draft) > 0 else "")
    print(f"\n  {'Component':<30} {'Mean(ms)':>10} {'Std(ms)':>10} {'% total':>10}")
    print(f"  {'-'*60}")
    print(f"  {'verify (target fwd+logits)':<30} {verify_stats['mean']:>10.4f} {verify_stats['std']:>10.4f} {pct.get('verify', 0):>9.1f}%")
    print(f"  {'accept/reject':<30} {accept_stats['mean']:>10.4f} {accept_stats['std']:>10.4f} {pct.get('accept_reject', 0):>9.1f}%")
    print(f"  {'draft (step0+K-1)':<30} {draft_stats['mean']:>10.4f} {draft_stats['std']:>10.4f} {pct.get('draft (step0+continuation)', 0):>9.1f}%")
    print(f"  {'python overhead':<30} {overhead_stats['mean']:>10.4f} {overhead_stats['std']:>10.4f} {pct.get('python_overhead', 0):>9.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL STEP':<30} {mean_total:>10.4f} {total_stats['std']:>10.4f} {'100.0':>9}%")

    if profiler_analysis:
        print(f"\n  --- Kernel-level breakdown (torch.profiler, {profiler_analysis['num_profiled_steps']} steps) ---")
        ps = profiler_analysis["per_step_ms"]
        pp = profiler_analysis["pct_of_cuda_time"]
        print(f"  {'Kernel Category':<25} {'Per-step(ms)':>12} {'% CUDA':>10}")
        print(f"  {'-'*47}")
        for cat in ["attention", "gemm", "gdn", "other_compute", "other"]:
            print(f"  {cat:<25} {ps.get(cat, 0):>12.4f} {pp.get(cat, 0):>9.1f}%")
        print(f"  {'-'*47}")
        print(f"  {'TOTAL CUDA':<25} {ps.get('total_cuda_ms', 0):>12.4f} {'100.0':>9}%")

        print("\n  --- Top 15 kernels by total CUDA time ---")
        for i, k in enumerate(profiler_analysis["top_15_kernels"]):
            print(f"  {i+1:>2}. [{k['total_ms']:>8.3f}ms, {k['count']:>4}x] {k['name']}")

    result = {
        "label": label,
        "num_steps": n,
        "warmup_steps": warmup,
        "stable_steps": n_stable,
        "concurrency": concurrency,
        "committed_per_step": round(committed_per_step, 3),
        "steps_per_sec": round(steps_per_sec, 2),
        "accepted_tokens_per_sec": round(tokens_per_sec, 2),
        "acceptance_rate": round(total_committed / (total_committed + total_draft), 4)
            if (total_committed + total_draft) > 0 else 0,
        "components": {
            "verify": verify_stats,
            "accept_reject": accept_stats,
            "draft_step0_plus_continuation": draft_stats,
            "python_overhead": overhead_stats,
            "total_step": total_stats,
        },
        "pct_breakdown": pct,
        "profiler_kernel_breakdown": profiler_analysis,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep per-step profiling of the MTP decode path"
    )
    parser.add_argument("--fixture", choices=["ctx128k", "ctx64k", "w1s"], default="ctx128k")
    parser.add_argument("--concurrency", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--suffix-len", type=int, default=DEFAULT_SUFFIX_LEN)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--num-steps", type=int, default=NUM_PROFILE_STEPS)
    parser.add_argument("--profiler-steps", type=int, default=NUM_PROFILER_STEPS)
    parser.add_argument("--blocks-margin", type=int, default=16)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import (
        CTX128K_FIXTURE,
        D1_CTX64K_FIXTURE,
        W1_S_FIXTURE,
        load_prompt_token_ids,
    )

    fixture_map = {
        "ctx128k": (CTX128K_FIXTURE, "128K warm cache-hit"),
        "ctx64k": (D1_CTX64K_FIXTURE, "64K warm cache-hit"),
        "w1s": (W1_S_FIXTURE, "W1-S short context (~4K)"),
    }
    fixture, label = fixture_map[args.fixture]
    P = fixture.prompt_len
    c = args.concurrency

    print("=== decode_step_profile ===")
    print(f"fixture={args.fixture} (P={P}), concurrency={c}, suffix_len={args.suffix_len}, "
          f"num_steps={args.num_steps}, profiler_steps={args.profiler_steps}")

    prompts = load_prompt_token_ids(fixture)
    if len(prompts) < c:
        print(f"ERROR: fixture has {len(prompts)} prompts, need {c}")
        return 2
    prefixes = [prompts[i] for i in range(c)]

    # Build runner
    print("\nBuilding runner...")
    try:
        runner, blocks_per_slot, max_model_len = _build_runner(
            P, args.suffix_len, args.max_tokens, c, args.gpu_mem_util, args.blocks_margin
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(f"OOM building runner: {exc}")
        return 3

    print(f"Runner up: blocks_per_slot={blocks_per_slot}, max_model_len={max_model_len}, "
          f"mem={_gpu_mem_mib()}, smi={_nvidia_smi_mem_mib()}MiB")

    # Setup scenario
    print(f"\nSetting up {'warm cache-hit' if args.fixture != 'w1s' else 'short-context'} scenario...")
    try:
        if args.fixture == "w1s":
            anchors, drafts = _setup_short_scenario(runner, prefixes)
        else:
            anchors, drafts = _setup_warm_scenario(
                runner, prefixes, args.suffix_len, args.chunk_size
            )
    except torch.cuda.OutOfMemoryError as exc:
        print(f"OOM in setup: {exc}")
        return 3

    slots = list(range(c))
    kv_lens = [runner.slot_kv_len[s] for s in slots]
    print(f"Setup complete. kv_lens={kv_lens}, mem={_gpu_mem_mib()}")

    # Phase 1: CUDA-event-timed decode steps
    print(f"\n--- Phase 1: Running {args.num_steps} profiled decode steps ---")
    timings, total_committed, total_draft = _run_profiled_decode_steps(
        runner, slots, anchors, drafts, args.num_steps
    )

    # Phase 2: torch.profiler kernel breakdown (fresh state)
    print(f"\n--- Phase 2: torch.profiler kernel breakdown ({args.profiler_steps} steps) ---")
    # Re-setup for profiler (fresh decode state)
    if args.fixture == "w1s":
        anchors2, drafts2 = _setup_short_scenario(runner, prefixes)
    else:
        anchors2, drafts2 = _setup_warm_scenario(
            runner, prefixes, args.suffix_len, args.chunk_size
        )
    profiler_analysis = None
    try:
        prof = _run_profiler_steps(runner, slots, anchors2, drafts2, args.profiler_steps)
        profiler_analysis = _analyze_profiler(prof, args.profiler_steps)
    except Exception as exc:
        print(f"Profiler failed (non-fatal): {exc}")

    # Report
    result = _report(
        timings, total_committed, total_draft, args.num_steps, c,
        profiler_analysis, f"{label}, c={c}"
    )
    result["fixture"] = args.fixture
    result["P"] = P
    result["suffix_len"] = args.suffix_len
    result["kv_lens_at_decode_start"] = kv_lens
    result["mem_after_decode"] = _gpu_mem_mib()
    result["smi_mem_mib"] = _nvidia_smi_mem_mib()

    print(f"\n{'='*78}")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
