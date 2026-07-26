"""D1-followup diagnostic: profile ``mtp_prefill_batch`` at the exact
c=4/16K shape that ``notes/2026-07-18-session-review-and-next-steps.md``
section 12 found to be 4.85x slower than native and to peak at 99.2% of
GPU memory capacity. That section's own hypothesis ("one non-chunked
forward over the full concurrency x context product") was explicitly
flagged as unconfirmed. This script gets real evidence.

PROFILING-ONLY. Does not modify any file under ``runtime/``. It
monkey-patches (at the Python object level, only within this script's own
process) ``runner.model.forward`` / ``runner.model.compute_logits`` /
``runner.mtp_model.forward`` / ``runner.mtp_model.compute_logits`` with
thin wrappers that record wall time + ``torch.cuda.memory_allocated()``
before/after each real call, then delegate to the ORIGINAL unwrapped
method -- the real ``mtp_prefill_batch`` is called completely unmodified,
so this is behaviorally identical to an uninstrumented run (same
convention ``phase0_nsys_gap_ledger_diag.py`` established: instrument via
calling the real, unmodified constituent pieces, never re-derive logic).

Usage:
    python -m benchmarks.mtp_prefill_batch_memory_diag --concurrency 4 --fixture ctx16k
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _nvidia_smi_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return int(out.strip())


def _wrap(name: str, orig, log: list):
    def wrapped(*args, **kwargs):
        torch = sys.modules["torch"]
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mem_after = torch.cuda.memory_allocated()
        shape = tuple(out.shape) if hasattr(out, "shape") else None
        dtype = str(out.dtype) if hasattr(out, "dtype") else None
        numel_bytes = (out.numel() * out.element_size()) if hasattr(out, "numel") else None
        entry = {
            "call": name,
            "time_s": round(t1 - t0, 4),
            "mem_before_mib": round(mem_before / 2**20, 1),
            "mem_after_mib": round(mem_after / 2**20, 1),
            "delta_mib": round((mem_after - mem_before) / 2**20, 1),
            "output_shape": shape,
            "output_dtype": dtype,
            "output_tensor_mib": round(numel_bytes / 2**20, 1) if numel_bytes else None,
        }
        log.append(entry)
        print(f"  [{name}] {entry['time_s']}s  mem {entry['mem_before_mib']:.0f}->"
              f"{entry['mem_after_mib']:.0f} MiB (delta {entry['delta_mib']:+.0f})  "
              f"out={shape} {dtype} ({entry['output_tensor_mib']} MiB)")
        return out
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--fixture", choices=["n16", "ctx16k", "ctx32k"], default="ctx16k")
    args = ap.parse_args()

    print("=== pre-run GPU check ===")
    print(f"nvidia-smi memory.used: {_nvidia_smi_mib()} MiB")

    import torch
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import (
        D1_CTX16K_FIXTURE,
        D1_CTX32K_FIXTURE,
        W1_S_FIXTURE,
        load_prompt_token_ids,
    )
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture = {"n16": W1_S_FIXTURE, "ctx16k": D1_CTX16K_FIXTURE, "ctx32k": D1_CTX32K_FIXTURE}[args.fixture]
    prompts = load_prompt_token_ids(fixture)[: args.concurrency]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, fixture.prompt_len + 256 + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    # No CUDA-graph doubling -- this diagnostic only touches the ONE
    # prefill call, isolating the effect from any capture-warmup capacity.
    runner = DirectModelRunner(
        vllm_config,
        num_slots=args.concurrency,
        block_size=64,
        blocks_per_slot=640,
        enable_cudagraph=False,
    )

    gc.collect()
    torch.cuda.synchronize()
    print(f"\n=== after model load: nvidia-smi {_nvidia_smi_mib()} MiB, "
          f"torch allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB ===")

    log: list = []
    runner.model.forward = _wrap("target_model.forward", runner.model.forward, log)
    runner.model.compute_logits = _wrap("target_model.compute_logits", runner.model.compute_logits, log)
    runner.mtp_model.forward = _wrap("draft_model.forward", runner.mtp_model.forward, log)
    runner.mtp_model.compute_logits = _wrap("draft_model.compute_logits", runner.mtp_model.compute_logits, log)

    torch.cuda.reset_peak_memory_stats()
    slots = list(range(len(prompts)))

    print(f"\n=== mtp_prefill_batch(concurrency={len(slots)}, prompt_len={fixture.prompt_len}) ===")
    t0 = time.perf_counter()
    result = runner.mtp_prefill_batch(slots, prompts)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    nvsmi_after = _nvidia_smi_mib()

    print(f"\n=== total wall time: {t1 - t0:.3f}s ===")
    print(f"peak torch allocated: {peak_allocated/2**20:.0f} MiB")
    print(f"peak torch reserved:  {peak_reserved/2**20:.0f} MiB")
    print(f"nvidia-smi after:     {nvsmi_after} MiB")

    total_wrapped_time = sum(e["time_s"] for e in log)
    print(f"\nsum of wrapped-call times: {total_wrapped_time:.3f}s "
          f"({100*total_wrapped_time/(t1-t0):.1f}% of total wall time)")

    print("\n=== per-call breakdown ===")
    for e in log:
        print(f"  {e['call']:28s} {e['time_s']:8.4f}s  out_tensor={e['output_tensor_mib']:>10.1f} MiB"
              f"  live_delta={e['delta_mib']:>10.1f} MiB")

    print(f"\nanchors returned: {[result[s]['anchor'] for s in slots]}")
    print("\n=== post-run GPU check (process still alive) ===")
    print(f"nvidia-smi memory.used: {_nvidia_smi_mib()} MiB")


if __name__ == "__main__":
    main()
