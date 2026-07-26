"""D1 second-follow-up diagnostic: real nsys kernel-family gap ledger for
``mtp_prefill_batch`` at TWO context lengths (4K and 16K, both concurrency=4)
in ONE process/ONE profiled trace, so the two are directly comparable without
cross-process variance. Written to answer
notes/2026-07-18-session-review-and-next-steps.md section 13.8's open
question: is the residual ~2.63x 16K/c=4 gap explained by attention's
inherent O(L^2) prefill scaling (expected, shape-neutral -- native pays the
same tax), a disproportionately-slower GEMM/FFN path, or something specific
to this runtime's kernel dispatch/host overhead that native does not pay.

PROFILING-ONLY. Does not modify any file under ``runtime/``. Uses two
DISTINCT groups of fresh slots (0-3 for the 16K call, 4-7 for the 4K call)
in a single ``DirectModelRunner`` (num_slots=8, matching this project's own
established convention of doubling slots for isolated-scenario capacity,
e.g. ``mtp_batch_verify_check.py``'s ``ref_slots=[4,5,6,7]`` at
``num_slots=8``) so only ONE model load is needed. Each call gets its own
NVTX range (``prefill_ctx16k`` / ``prefill_ctx4k``) plus sub-ranges around
the real ``target_model.forward``/``compute_logits`` and
``draft_model.forward``/``compute_logits`` calls (same monkey-patch
technique ``mtp_prefill_batch_memory_diag.py`` established), so the nsys
trace can be sliced per phase without editing ``direct_model_runner.py``.

Usage (must run under nsys to get the kernel ledger; can also run bare for
just the host-timer numbers):
    nsys profile -c cudaProfilerApi --capture-range-end=stop \\
        --trace=cuda,nvtx,osrt -o d1_prefill_shapes --force-overwrite=true \\
        python -m benchmarks.d1_prefill_shape_nsys_diag
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Toggle for the D1 second-follow-up investigation (2026-07-18): does the
# already-built, already-correctness-tested v2 prefill kernel
# (flash_attn_sm120_fwd_prefill_v2_fp8kv_paged) close some of the residual
# 16K/c=4 gap left after the vocab-head fix? Controlled by the SAME env var
# the real SM120GQABackend reads (SM120_GQA_USE_V2_PREFILL_KERNEL), set
# externally by whoever runs this script -- no default forced here so this
# diagnostic can be run both ways without editing it.

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _nvidia_smi_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    return int(out.strip())


def _wrap(name: str, orig, log: list, torch):
    def wrapped(*args, **kwargs):
        torch.cuda.nvtx.range_push(name)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch.cuda.nvtx.range_pop()
        shape = tuple(out.shape) if hasattr(out, "shape") else None
        entry = {"call": name, "time_s": round(t1 - t0, 4), "output_shape": shape}
        log.append(entry)
        print(f"  [{name}] {entry['time_s']}s  out={shape}", flush=True)
        return out
    return wrapped


def _run_one_shape(torch, runner, label: str, slots: list[int], prompts: list[list[int]], log: list) -> float:
    print(f"\n=== {label}: mtp_prefill_batch(concurrency={len(slots)}, prompt_len={len(prompts[0])}) ===", flush=True)
    torch.cuda.nvtx.range_push(label)
    t0 = time.perf_counter()
    result = runner.mtp_prefill_batch(slots, prompts)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.cuda.nvtx.range_pop()
    wall_s = t1 - t0
    print(f"=== {label} total wall time: {wall_s:.3f}s ===", flush=True)
    print(f"anchors: {[result[s]['anchor'] for s in slots]}", flush=True)
    return wall_s


def main() -> int:
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
        vllm_config,
        num_slots=8,
        block_size=64,
        blocks_per_slot=640,
        enable_cudagraph=False,
    )

    gc.collect()
    torch.cuda.synchronize()
    print(f"\n=== after model load: nvidia-smi {_nvidia_smi_mib()} MiB, "
          f"torch allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB ===", flush=True)

    log: list = []
    runner.model.forward = _wrap("target_model.forward", runner.model.forward, log, torch)
    runner.model.compute_logits = _wrap("target_model.compute_logits", runner.model.compute_logits, log, torch)
    runner.mtp_model.forward = _wrap("draft_model.forward", runner.mtp_model.forward, log, torch)
    runner.mtp_model.compute_logits = _wrap("draft_model.compute_logits", runner.mtp_model.compute_logits, log, torch)

    profiling = True
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as e:  # pragma: no cover
        print(f"WARN cudaProfilerStart failed: {e}", flush=True)
        profiling = False

    torch.cuda.reset_peak_memory_stats()
    wall_16k = _run_one_shape(torch, runner, "prefill_ctx16k", [0, 1, 2, 3], prompts_16k, log)
    peak_16k = torch.cuda.max_memory_allocated()
    nvsmi_16k = _nvidia_smi_mib()

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_4k = _run_one_shape(torch, runner, "prefill_ctx4k", [4, 5, 6, 7], prompts_4k, log)
    peak_4k = torch.cuda.max_memory_allocated()
    nvsmi_4k = _nvidia_smi_mib()

    if profiling:
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception as e:  # pragma: no cover
            print(f"WARN cudaProfilerStop failed: {e}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"ctx16k: wall={wall_16k:.3f}s peak_allocated={peak_16k/2**20:.0f}MiB nvsmi={nvsmi_16k}MiB", flush=True)
    print(f"ctx4k:  wall={wall_4k:.3f}s peak_allocated={peak_4k/2**20:.0f}MiB nvsmi={nvsmi_4k}MiB", flush=True)
    print(f"ratio (16k/4k) wall time: {wall_16k/wall_4k:.3f}x (token-count ratio is exactly 4.0x)", flush=True)

    print("\n=== per-call breakdown ===", flush=True)
    for e in log:
        print(f"  {e['call']:28s} {e['time_s']:8.4f}s  out={e['output_shape']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
