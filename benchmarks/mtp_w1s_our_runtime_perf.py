"""No-vLLM W1-S performance gate for the production Qwen3.6 MTP backend.

This is the direct successor to the historical runner benchmark.  It keeps
the frozen W1-S prompt ids and its accounting boundary: every request's
prefill plus every MTP verify/commit round is included in wall time; a round
commits ``num_accepted + 1`` tokens.  Unlike the retired version, it imports
neither ``vllm`` nor ``oracle`` and drives the exact
``Qwen36Backend.prefill_chunked_begin`` / ``mtp_verify_and_commit_batch``
production APIs.

The historical headline was W1-S (4096 input / 256 output / c=4 / K=3 / n=16),
measured with the prefix cache enabled (``--enable-prefix-caching``): repeat
repeats served from cached KV/GDN state.  This runner keeps that warm
accounting boundary -- the persistent prefix arena is sized so every frozen
prompt's entry stays resident across repeats -- and its first repeat remains
the cold-prefill anchor.
Run a B=1 preflight before the explicit c=4 measurement on the shared card:

    python -m benchmarks.mtp_w1s_our_runtime_perf --num-requests 1 --concurrency 1 --max-tokens 16
    python -m benchmarks.mtp_w1s_our_runtime_perf --num-requests 16 --concurrency 4 --max-tokens 256 --repeats 3

The current backend's prefill entry point is intentionally measured as it is
served today.  It does not claim a historical batched-prefill implementation
that no longer exists; the JSON records that fact so ``bf diff`` cannot hide
it behind a superficially matching fixture name.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from benchmarks.workloads import W1_S_FIXTURE, W1_S_FIXTURE_N128, load_prompt_token_ids
from runtime.backends.qwen36 import Qwen36Backend
from runtime.checkpoints import standard_checkpoint_path
from runtime.model_loading import load_qwen36_model

K = 3
_ROOT = Path(__file__).resolve().parent.parent


def _gpu_thermal() -> dict[str, int]:
    fields = "temperature.gpu,clocks.current.sm,memory.used"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    temp, clock, memory = (int(value.strip()) for value in output.splitlines()[0].split(","))
    return {"temperature_c": temp, "clock_sm_mhz": clock, "memory_used_mib": memory}


def _reset_slots(backend: Qwen36Backend, slots: list[int]) -> None:
    for slot in slots:
        if not backend.slot_state(slot).is_fresh:
            backend.reset_slot(slot)


def _run_batch(
    backend: Qwen36Backend,
    prompts: list[list[int]],
    *,
    max_tokens: int,
) -> dict[str, float | int | list[float]]:
    """One production batch, timed at the historical request boundary."""
    slots = list(range(len(prompts)))
    _reset_slots(backend, slots)
    gpu_s = wall_s = 0.0
    prefill_gpu_s = prefill_wall_s = 0.0
    verify_gpu_s = verify_wall_s = 0.0
    ttft_s: list[float] = []
    itl_s: list[float] = []

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    # W1-S fixes every prompt at 4096 tokens.  The production helper defaults
    # to a latency-oriented 512-token chunk, so request the complete fixture
    # here to retain the historical "prefill plus speculative decode" timing
    # boundary rather than accidentally timing only its first chunk.
    prefill = backend.prefill_chunked_begin(
        slots,
        prompts,
        chunk_size=max(len(prompt) for prompt in prompts),
    )
    end.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert prefill.done, "W1-S prompt must finish in the production prefill call"
    prefill_gpu_s = begin.elapsed_time(end) / 1000.0
    prefill_wall_s = elapsed
    gpu_s += prefill_gpu_s
    wall_s += prefill_wall_s
    ttft_s.extend([elapsed] * len(slots))

    anchors = {slot: prefill.result[slot]["anchor"] for slot in slots}
    drafts = {slot: list(prefill.result[slot]["draft_tokens"]) for slot in slots}
    if any(len(drafts[slot]) != K for slot in slots):
        raise RuntimeError("MTP prefill did not produce the configured K drafts")
    committed = {slot: 0 for slot in slots}
    accepted = draft_tokens = draft_rounds = 0
    active = list(slots)
    while active:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        decisions = backend.mtp_verify_and_commit_batch(
            active,
            {slot: anchors[slot] for slot in active},
            {slot: drafts[slot] for slot in active},
        )
        end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        round_gpu_s = begin.elapsed_time(end) / 1000.0
        gpu_s += round_gpu_s
        wall_s += elapsed
        verify_gpu_s += round_gpu_s
        verify_wall_s += elapsed

        finished: list[int] = []
        for slot in active:
            decision = decisions[slot]
            accepted_this_round = int(decision["num_accepted"])
            new_tokens = list(decision["committed"])
            if len(new_tokens) != accepted_this_round + 1:
                raise RuntimeError("MTP decision violates committed = accepted + recovery/bonus")
            if len(decision["next_draft_tokens"]) != K:
                raise RuntimeError("MTP round did not replenish exactly K drafts")
            accepted += accepted_this_round
            draft_tokens += K
            draft_rounds += 1
            committed[slot] += len(new_tokens)
            itl_s.append(elapsed / len(new_tokens))
            anchors[slot] = int(decision["next_anchor"])
            drafts[slot] = list(decision["next_draft_tokens"])
            if committed[slot] >= max_tokens:
                finished.append(slot)
        for slot in finished:
            active.remove(slot)

    return {
        "accepted": accepted,
        "draft_tokens": draft_tokens,
        "draft_rounds": draft_rounds,
        "committed": sum(committed.values()),
        "gpu_busy_s": gpu_s,
        "wall_s": wall_s,
        "prefill_gpu_s": prefill_gpu_s,
        "prefill_wall_s": prefill_wall_s,
        "verify_gpu_s": verify_gpu_s,
        "verify_wall_s": verify_wall_s,
        "ttft_s": ttft_s,
        "itl_s": itl_s,
    }


def _run_rep(
    backend: Qwen36Backend,
    prompts: list[list[int]],
    *,
    concurrency: int,
    max_tokens: int,
    rep: int,
) -> dict[str, object]:
    thermal_before = _gpu_thermal()
    started = time.perf_counter()
    totals = {"accepted": 0, "draft_tokens": 0, "draft_rounds": 0, "committed": 0}
    gpu_busy_s = wall_s = 0.0
    prefill_gpu_s = prefill_wall_s = verify_gpu_s = verify_wall_s = 0.0
    ttft_s: list[float] = []
    itl_s: list[float] = []
    for offset in range(0, len(prompts), concurrency):
        result = _run_batch(backend, prompts[offset : offset + concurrency], max_tokens=max_tokens)
        for name in totals:
            totals[name] += int(result[name])
        gpu_busy_s += float(result["gpu_busy_s"])
        wall_s += float(result["wall_s"])
        prefill_gpu_s += float(result["prefill_gpu_s"])
        prefill_wall_s += float(result["prefill_wall_s"])
        verify_gpu_s += float(result["verify_gpu_s"])
        verify_wall_s += float(result["verify_wall_s"])
        ttft_s.extend(result["ttft_s"])
        itl_s.extend(result["itl_s"])
        print(f"  rep {rep}: {min(offset + concurrency, len(prompts))}/{len(prompts)} requests", flush=True)
    e2e_s = time.perf_counter() - started
    ttft_s.sort()
    itl_s.sort()
    return {
        "rep": rep,
        "total_committed_tokens": totals["committed"],
        "num_drafts": totals["draft_rounds"],
        "num_accepted_tokens": totals["accepted"],
        "num_draft_tokens": totals["draft_tokens"],
        "draft_acceptance_rate_pct": 100.0 * totals["accepted"] / totals["draft_tokens"],
        "accepted_tokens_per_sec": totals["committed"] / e2e_s,
        "ms_per_accepted_token": 1000.0 * e2e_s / totals["committed"],
        "ms_per_draft": 1000.0 * e2e_s / totals["draft_rounds"],
        "wall_s_e2e": e2e_s,
        "gpu_busy_s": gpu_busy_s,
        "wall_s_measured_calls": wall_s,
        "prefill_gpu_s": prefill_gpu_s,
        "prefill_wall_s": prefill_wall_s,
        "verify_gpu_s": verify_gpu_s,
        "verify_wall_s": verify_wall_s,
        "gpu_busy_pct": 100.0 * gpu_busy_s / wall_s if wall_s else math.nan,
        "launch_gap_pct": 100.0 * (1.0 - gpu_busy_s / wall_s) if wall_s else math.nan,
        "ttft_mean_ms": 1000.0 * sum(ttft_s) / len(ttft_s),
        "ttft_p99_ms": 1000.0 * ttft_s[min(len(ttft_s) - 1, int(0.99 * len(ttft_s)))],
        "itl_mean_ms": 1000.0 * sum(itl_s) / len(itl_s),
        "itl_p99_ms": 1000.0 * itl_s[min(len(itl_s) - 1, int(0.99 * len(itl_s)))],
        "num_itl_samples": len(itl_s),
        "thermal_before": thermal_before,
        "thermal_after": _gpu_thermal(),
    }


def _profile_steady_rounds(
    backend: Qwen36Backend,
    prompts: list[list[int]],
    *,
    rounds: int,
    trace_path: Path | None,
) -> dict[str, object]:
    """Profile production B-wide MTP rounds after an untimed W1-S prefill.

    This is intentionally part of the frozen-fixture performance gate, not a
    new ad-hoc probe: the captured calls are the same public production API as
    the throughput measurement above.  It runs only when explicitly requested
    and after normal repetitions, so profiler instrumentation cannot affect the
    headline throughput result.
    """
    if rounds <= 0:
        return {}
    from torch.profiler import ProfilerActivity, profile, record_function

    slots = list(range(len(prompts)))
    _reset_slots(backend, slots)
    prefill = backend.prefill_chunked_begin(
        slots,
        prompts,
        chunk_size=max(len(prompt) for prompt in prompts),
    )
    assert prefill.done
    anchors = {slot: int(prefill.result[slot]["anchor"]) for slot in slots}
    drafts = {slot: list(prefill.result[slot]["draft_tokens"]) for slot in slots}

    engine = backend._mtp
    assert engine is not None and engine._verify_cg is not None
    phase_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def instrument(obj: object, name: str, label: str):
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with record_function(label):
                result = original(*args, **kwargs)
            end.record()
            phase_events.setdefault(label, []).append((start, end))
            return result

        setattr(obj, name, wrapped)
        return original

    restores = [
        (engine._verify_cg, "replay", instrument(engine._verify_cg, "replay", "mtp.verify_graph")),
        (engine.model, "compute_logits", instrument(engine.model, "compute_logits", "mtp.lm_head")),
        (
            engine,
            "_sync_real_suffix_batch_ragged",
            instrument(engine, "_sync_real_suffix_batch_ragged", "mtp.sync"),
        ),
        (
            engine,
            "_continue_draft_batch",
            instrument(engine, "_continue_draft_batch", "mtp.draft"),
        ),
    ]
    torch.cuda.synchronize()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as profiler:
            for _ in range(rounds):
                decisions = backend.mtp_verify_and_commit_batch(slots, anchors, drafts)
                anchors = {slot: int(decisions[slot]["next_anchor"]) for slot in slots}
                drafts = {slot: list(decisions[slot]["next_draft_tokens"]) for slot in slots}
            torch.cuda.synchronize()
    finally:
        for obj, name, original in restores:
            setattr(obj, name, original)

    if trace_path is not None:
        profiler.export_chrome_trace(str(trace_path))
        print(f"wrote W1-S CPU/CUDA trace: {trace_path}")

    def top_self(attr: str) -> list[dict[str, float | str]]:
        rows = [
            {"name": event.key, "ms": float(getattr(event, attr, 0.0)) / 1000.0}
            for event in profiler.key_averages()
            if getattr(event, attr, 0.0) > 0
        ]
        return sorted(rows, key=lambda row: float(row["ms"]), reverse=True)[:20]

    phase_cuda_ms = {
        name: sum(start.elapsed_time(end) for start, end in pairs) / rounds
        for name, pairs in phase_events.items()
    }
    print(f"W1-S steady-round profiler: {rounds} B={len(slots)} round(s)")
    for name, elapsed_ms in phase_cuda_ms.items():
        print(f"  {elapsed_ms:8.3f} ms/round  {name}")
    return {
        "rounds": rounds,
        "slots": len(slots),
        "phase_cuda_ms": phase_cuda_ms,
        "self_cpu_ms": top_self("self_cpu_time_total"),
        "self_cuda_ms": top_self("self_device_time_total"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fixture", choices=["n16", "n128"], default="n16")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="truncate the frozen prompt for a low-memory integration smoke; default keeps W1-S",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model-path", default=str(standard_checkpoint_path()))
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument(
        "--profile-rounds",
        type=int,
        default=0,
        help="profile this many additional steady B-wide MTP rounds after throughput repetitions",
    )
    parser.add_argument(
        "--profile-trace-path",
        type=Path,
        default=None,
        help="optional Chrome trace path for --profile-rounds",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.repeats < 1 or args.max_tokens < 1 or args.profile_rounds < 0:
        parser.error("concurrency, repeats, and max-tokens must be positive; profile-rounds cannot be negative")

    fixture = W1_S_FIXTURE if args.fixture == "n16" else W1_S_FIXTURE_N128
    prompts = load_prompt_token_ids(fixture)
    if args.num_requests is not None:
        prompts = prompts[: args.num_requests]
    prompt_len = fixture.prompt_len
    if args.prompt_tokens is not None:
        if not 1 <= args.prompt_tokens <= fixture.prompt_len:
            parser.error("prompt-tokens must be in [1, fixture prompt length]")
        prompt_len = args.prompt_tokens
        prompts = [prompt[:prompt_len] for prompt in prompts]
    if not prompts:
        parser.error("num-requests must select at least one frozen prompt")
    max_seq_len = prompt_len + args.max_tokens + K + 16
    # Warm-caliber arena sizing: the persistent prefix arena holds one
    # prompt-boundary entry per distinct prompt (ceil(prompt_len/page_size)
    # scratch pages each), exactly the repeat-hit shape the historical
    # 8 GiB block cache served.  page_size is Qwen36Backend's 128-token
    # attention page.
    page_size = 128
    prompt_pages = (prompt_len + page_size - 1) // page_size
    max_seq_len = max(max_seq_len, len(prompts) * prompt_pages * page_size + page_size)
    print(f"fixture={fixture.path} prompt_len={prompt_len} requests={len(prompts)}")
    print(f"concurrency={args.concurrency} K={K} max_seq_len={max_seq_len}")
    model = load_qwen36_model(
        args.model_path,
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=max_seq_len,
        enable_mtp=True,
    )
    backend = Qwen36Backend(
        model,
        num_slots=args.concurrency,
        max_seq_len=max_seq_len,
        block_size=64,
        device="cuda",
        dtype=torch.bfloat16,
        enable_prefix_cache=True,
        # Every distinct prompt keeps a prompt-boundary checkpoint plus the
        # generation drift entry; 3 GiB covers the 16-prompt fixture twice.
        checkpoint_byte_budget=3 * 2**30,
    )
    backend.enable_mtp(num_speculative_tokens=K, enable_resync=False)
    engine = backend._mtp
    assert engine is not None
    if not engine.cuda_graphs_healthy():
        raise RuntimeError(f"MTP CUDA Graph capture is unhealthy: {engine.cg_status}")

    reps = [
        _run_rep(backend, prompts, concurrency=args.concurrency, max_tokens=args.max_tokens, rep=index + 1)
        for index in range(args.repeats)
    ]
    steady_round_profile = _profile_steady_rounds(
        backend,
        prompts[: args.concurrency],
        rounds=args.profile_rounds,
        trace_path=args.profile_trace_path,
    )
    passed = all(int(rep["total_committed_tokens"]) > 0 for rep in reps)
    result = {
        "passed": passed,
        "model_path": args.model_path,
        "fixture": fixture.path,
        "fixture_seed": fixture.seed,
        "num_requests": len(prompts),
        "prompt_len": prompt_len,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "k": K,
        "repeats": args.repeats,
        "backend": "qwen36-no-vllm",
        "prefill_batched": backend.stats["prefill_batched_forwards"] > 0,
        "prefill_batched_forwards": backend.stats["prefill_batched_forwards"],
        "mtp_cg_status": engine.cg_status,
        "backend_stats": {
            name: value
            for name, value in backend.stats.items()
            if isinstance(value, (int, float))
        },
        "steady_round_profile": steady_round_profile,
        "reps": reps,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.result_path is not None:
        args.result_path.write_text(text, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
