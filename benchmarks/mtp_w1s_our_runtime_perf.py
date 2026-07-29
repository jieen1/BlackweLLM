"""This runtime's side of the REAL end-to-end performance comparison
(accepted tokens/s, ms/accepted token, ms/draft, TTFT, ITL, and --
the metric this whole project's premise rests on -- GPU-busy%/launch-gap)
against native vLLM, at the W1-S shape (frozen, versioned prompt token
ids, same fixture `w1s_native_bench.py` uses).

Unlike `mtp_w1s_our_runtime.py` (which only needed aggregate accept/reject
counts), every real GPU-issuing call (`mtp_prefill`'s target forward,
`mtp_verify_and_commit`'s verify/recompute) is bracketed with
`torch.cuda.Event` (the SAME technique `mtp_trace_driven_probe.py`'s
synthetic-trace probe used, now applied to the REAL, already-verified
MTP state machine on REAL data instead of a synthetic trace) so GPU-busy
time can be compared directly against wall-clock time.

By default this runtime processes slots ONE AT A TIME in round-robin
(never batches multiple slots into one kernel launch) -- so "GPU busy%
here" measures how much of THIS runtime's own wall-clock time is real GPU
work vs. Python-dispatch/launch overhead for that architecture, not a
best-case synthetic scenario. Eager mode only (no CUDA graph capture) --
explicitly the point of comparison the coordinator asked for: does
removing Python/vLLM scheduling overhead show an advantage even without
the graph-capture optimization this project has not yet integrated into
the real MTP accept/reject flow.

``--batched`` (2026-07-17, cross-slot batching round): switches to the
NEW ``mtp_prefill_batch``/``mtp_verify_and_commit_batch`` coordinator
(``_run_batch_batched`` below) -- ONE shared kernel launch per round
covering every concurrent slot (draft model included, not just target),
instead of ``concurrency`` separate sequential single-slot calls. This is
the direct re-measurement of whether real batching (as opposed to merely
removing vLLM's own scheduling layer) closes the ~12.46x gap the
single-slot round-robin path showed against native. Correctness of the
batched coordinator itself is verified separately in
``benchmarks/mtp_batch_verify_check.py`` -- this script only measures
performance, it does not re-verify correctness.

Usage:
    python -m benchmarks.mtp_w1s_our_runtime_perf --max-tokens 256 --concurrency 4 --fixture n128 --num-requests 16
    python -m benchmarks.mtp_w1s_our_runtime_perf --max-tokens 256 --concurrency 4 --fixture n128 --num-requests 16 --batched
"""

from __future__ import annotations

import argparse
import math
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

# 2026-07-19, 256K-feasibility task: confirmed from the real checkpoint's
# config.json (text_config.max_position_embeddings) -- this model's own
# native RoPE ("default" rope_type, no YaRN/NTK scaling configured) supports
# AT MOST this many total positions. Capping max_model_len here (rather than
# letting the max(40960, prompt_len+max_tokens+1024) formula below exceed it
# for the ctx256k fixture) avoids passing vLLM an out-of-architectural-range
# max_model_len -- not merely a config nicety, since going beyond this figure
# would be genuinely out-of-distribution for the model's trained/supported
# context, not just an engineering inconvenience. Never binds for any
# existing (<=64K) fixture.
_MODEL_MAX_POSITION_EMBEDDINGS = 262144


def _gpu_thermal() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    temp, clock, mem = [x.strip() for x in out.split(",")]
    return {"temperature_c": int(temp), "clock_sm_mhz": int(clock), "memory_used_mib": int(mem)}


def _sanity_check_reps(reps: list[dict]) -> bool:
    """Cheap liveness/sanity signal only -- NOT a correctness check. This
    script never re-verifies generated-token correctness (see module
    docstring; that's `mtp_batch_verify_check.py` / `mtp_chunked_prefill_
    check.py`'s job). All this confirms is that every rep actually produced
    real output -- committed tokens and a valid (non-NaN) acceptance rate --
    rather than silently completing with an empty/degenerate run (e.g. a
    runner that returns immediately without ever calling the target/draft
    model). Previously this was a hardcoded ``"passed": True`` literal with
    no check behind it at all (see notes/2026-07-18-session-review-and-
    next-steps.md section 20.3) -- this replaces that with the cheapest
    real thing this script's own already-computed numbers can verify."""
    if not reps:
        return False
    for rep in reps:
        if rep["total_committed_tokens"] <= 0:
            return False
        if math.isnan(rep["draft_acceptance_rate_pct"]):
            return False
    return True


def _run_batch(torch, runner, prompts_batch: list[list[int]], target_output_len: int) -> dict:
    num = len(prompts_batch)
    slots = list(range(num))
    for s in slots:
        if runner.slot_kv_len[s] != 0:
            runner.reset_slot(s)

    anchor = {}
    draft_tokens = {}
    committed_len = {s: 0 for s in slots}
    per_slot = {
        s: {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0, "gpu_busy_s": 0.0, "wall_s": 0.0}
        for s in slots
    }
    ttft_s = {}
    itl_samples: list[float] = []

    for i, s in enumerate(slots):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start_evt.record()
        pr = runner.mtp_prefill(s, prompts_batch[i])
        end_evt.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        anchor[s] = pr["anchor"]
        draft_tokens[s] = pr["draft_tokens"]
        ttft_s[s] = t1 - t0
        per_slot[s]["wall_s"] += t1 - t0
        per_slot[s]["gpu_busy_s"] += start_evt.elapsed_time(end_evt) / 1000.0

    finished = set()
    while len(finished) < num:
        for s in slots:
            if s in finished:
                continue
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start_evt.record()
            decision = runner.mtp_verify_and_commit(s, anchor[s], draft_tokens[s])
            end_evt.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_acc = decision["num_accepted"]
            tokens_this_round = n_acc + 1
            per_slot[s]["num_drafts"] += 1
            per_slot[s]["num_draft_tokens"] += K
            per_slot[s]["num_accepted_tokens"] += n_acc
            per_slot[s]["wall_s"] += t1 - t0
            per_slot[s]["gpu_busy_s"] += start_evt.elapsed_time(end_evt) / 1000.0
            committed_len[s] += tokens_this_round
            itl_samples.append((t1 - t0) / tokens_this_round)

            anchor[s], draft_tokens[s] = decision["next_anchor"], decision["next_draft_tokens"]
            if committed_len[s] >= target_output_len:
                finished.add(s)

    return {"per_slot": per_slot, "ttft_s": ttft_s, "itl_samples": itl_samples}


def _run_batch_batched(
    torch, runner, prompts_batch: list[list[int]], target_output_len: int, chunk_size: int | None = None
) -> dict:
    """2026-07-17 cross-slot-batched analogue of ``_run_batch``: ONE
    ``mtp_prefill_batch`` call covering every slot in this request batch,
    then a loop of ONE ``mtp_verify_and_commit_batch`` call per round
    (shrinking the active-slot list as individual slots reach their own
    ``target_output_len`` -- the SAME "mixed-stage" handling
    ``mtp_verify_and_commit_batch`` itself supports internally, just at
    the coarser "finished vs. still-generating" granularity here). GPU-busy/
    wall time is bracketed at the BATCH-CALL level, not per-slot (a real
    batched call is one shared kernel launch -- there is no per-slot
    sub-interval to attribute it to individually, unlike the round-robin
    path's naturally-serial single-slot calls).

    ``chunk_size`` (2026-07-19, chunked-prefill round, default ``None``
    preserving every existing invocation byte-for-byte): forwarded
    directly to ``mtp_prefill_batch``'s identical new parameter -- see
    that method's docstring. Only affects the ONE prefill call below;
    the decode/verify round loop is unchanged either way."""
    num = len(prompts_batch)
    slots = list(range(num))
    for s in slots:
        if runner.slot_kv_len[s] != 0:
            runner.reset_slot(s)

    committed_len = {s: 0 for s in slots}
    per_slot = {s: {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0} for s in slots}
    ttft_s = {}
    itl_samples: list[float] = []
    total_gpu_busy_s = 0.0
    total_wall_s = 0.0

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start_evt.record()
    # P3.3a: production prefill is the unified entrypoint. Flag off => it
    # delegates straight to mtp_prefill_batch (byte-for-byte P2), so this swap
    # regresses nothing when the persistent cache is disabled (the default).
    prefill_result = runner.mtp_prefill_with_cache(slots, prompts_batch, chunk_size=chunk_size)
    end_evt.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_wall_s = t1 - t0
    total_gpu_busy_s += start_evt.elapsed_time(end_evt) / 1000.0
    total_wall_s += prefill_wall_s
    for s in slots:
        # Every slot in this batch gets its first token at the SAME real
        # moment (one shared kernel launch) -- this identical-TTFT-across-
        # the-batch result is an expected, correct property of real
        # batching, not a measurement artifact.
        ttft_s[s] = prefill_wall_s

    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}

    active = list(slots)
    while active:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start_evt.record()
        decisions = runner.mtp_verify_and_commit_batch(
            active, {s: anchors[s] for s in active}, {s: drafts[s] for s in active}
        )
        end_evt.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        round_wall_s = t1 - t0
        total_gpu_busy_s += start_evt.elapsed_time(end_evt) / 1000.0
        total_wall_s += round_wall_s

        newly_finished = []
        for s in active:
            decision = decisions[s]
            n_acc = decision["num_accepted"]
            tokens_this_round = n_acc + 1
            per_slot[s]["num_drafts"] += 1
            per_slot[s]["num_draft_tokens"] += K
            per_slot[s]["num_accepted_tokens"] += n_acc
            committed_len[s] += tokens_this_round
            # Per-stream ITL attribution: this round's shared wall-clock
            # divided by THIS slot's own committed-token count -- matches
            # how a real multi-tenant server's client-observed ITL would
            # be measured (from the stream's perspective), regardless of
            # the shared batched kernel launch underneath.
            itl_samples.append(round_wall_s / tokens_this_round)
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]
            if committed_len[s] >= target_output_len:
                newly_finished.append(s)
        for s in newly_finished:
            active.remove(s)

    return {
        "per_slot": per_slot,
        "ttft_s": ttft_s,
        "itl_samples": itl_samples,
        "batch_gpu_busy_s": total_gpu_busy_s,
        "batch_wall_s": total_wall_s,
    }


def _run_measurement(
    torch,
    runner,
    prompts: list[list[int]],
    max_tokens: int,
    concurrency: int,
    rep: int,
    batched: bool = False,
    chunk_size: int | None = None,
) -> dict:
    """ONE repetition of the full W1-S request set against an
    ALREADY-LOADED runner (no reload between reps -- reps are for
    repeated-measurement variance, not independent process launches;
    see this file's module docstring / the design doc for why this is a
    deliberate, documented deviation from literal interleaved A/B this
    round, given the cost of reloading a 27B model per leg)."""
    thermal_before = _gpu_thermal()
    all_ttfts: list[float] = []
    all_itls: list[float] = []
    total_drafts = total_draft_tokens = total_accepted = 0
    total_gpu_busy_s = total_wall_s = 0.0

    t_start = time.perf_counter()
    num_batches = (len(prompts) + concurrency - 1) // concurrency
    for batch_idx, batch_start in enumerate(range(0, len(prompts), concurrency)):
        batch = prompts[batch_start : batch_start + concurrency]
        out = (
            _run_batch_batched(torch, runner, batch, max_tokens, chunk_size=chunk_size)
            if batched
            else _run_batch(torch, runner, batch, max_tokens)
        )
        for s, stats in out["per_slot"].items():
            total_drafts += stats["num_drafts"]
            total_draft_tokens += stats["num_draft_tokens"]
            total_accepted += stats["num_accepted_tokens"]
            if not batched:
                # Single-slot path: each slot's own bracketed calls are
                # genuinely additive (real, disjoint, serially-issued GPU
                # work) -- summing per-slot is correct here.
                total_gpu_busy_s += stats["gpu_busy_s"]
                total_wall_s += stats["wall_s"]
        if batched:
            # Batched path: gpu_busy_s/wall_s are BATCH-LEVEL quantities
            # (one shared kernel launch per round covering every active
            # slot) -- summing them per-slot would double/quadruple-count
            # the same interval, so pull the batch-level totals directly.
            total_gpu_busy_s += out["batch_gpu_busy_s"]
            total_wall_s += out["batch_wall_s"]
        all_ttfts.extend(out["ttft_s"].values())
        all_itls.extend(out["itl_samples"])
        elapsed = time.perf_counter() - t_start
        print(f"  ... rep {rep} batch {batch_idx + 1}/{num_batches} done ({elapsed:.0f}s elapsed)", flush=True)

    wall_s_e2e = time.perf_counter() - t_start
    thermal_after = _gpu_thermal()
    total_committed = total_accepted + total_drafts

    all_ttfts.sort()
    all_itls.sort()

    return {
        "rep": rep,
        "wall_s_e2e": wall_s_e2e,
        "num_drafts": total_drafts,
        "num_draft_tokens": total_draft_tokens,
        "num_accepted_tokens": total_accepted,
        "draft_acceptance_rate_pct": total_accepted / total_draft_tokens * 100.0 if total_draft_tokens else float("nan"),
        "total_committed_tokens": total_committed,
        "accepted_tokens_per_sec": total_committed / wall_s_e2e if wall_s_e2e > 0 else float("nan"),
        "ms_per_accepted_token": wall_s_e2e * 1000.0 / total_committed if total_committed > 0 else float("nan"),
        "ms_per_draft": wall_s_e2e * 1000.0 / total_drafts if total_drafts > 0 else float("nan"),
        "ttft_mean_ms": sum(all_ttfts) / len(all_ttfts) * 1000.0 if all_ttfts else float("nan"),
        "ttft_p99_ms": all_ttfts[int(len(all_ttfts) * 0.99)] * 1000.0 if all_ttfts else float("nan"),
        "itl_mean_ms": sum(all_itls) / len(all_itls) * 1000.0 if all_itls else float("nan"),
        "itl_p99_ms": all_itls[int(len(all_itls) * 0.99)] * 1000.0 if all_itls else float("nan"),
        "num_itl_samples": len(all_itls),
        "gpu_busy_s_summed_across_slots": total_gpu_busy_s,
        "wall_s_summed_across_slots": total_wall_s,
        "gpu_busy_pct": total_gpu_busy_s / total_wall_s * 100.0 if total_wall_s > 0 else float("nan"),
        "launch_gap_pct": (1.0 - total_gpu_busy_s / total_wall_s) * 100.0 if total_wall_s > 0 else float("nan"),
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
    }


def _run_once(
    max_tokens: int,
    concurrency: int,
    fixture_key: str,
    num_requests: int | None,
    repeats: int,
    batched: bool = False,
    cudagraph: bool = False,
    blocks_per_slot: int = 2560,
    chunk_size: int | None = None,
) -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import (
        CTX128K_FIXTURE,
        CTX200K_FIXTURE,
        CTX256K_FIXTURE,
        D1_CTX16K_FIXTURE,
        D1_CTX32K_FIXTURE,
        D1_CTX64K_FIXTURE,
        W1_S_FIXTURE,
        W1_S_FIXTURE_N128,
        load_prompt_token_ids,
    )
    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture = {
        "n16": W1_S_FIXTURE,
        "n128": W1_S_FIXTURE_N128,
        # 2026-07-18, Phase D1 shape-generalization sweep: same-formula/
        # same-seed constructed fixtures at longer context, NOT the
        # official W2/W2-S line -- see workloads.py's own docstring on
        # these two.
        "ctx16k": D1_CTX16K_FIXTURE,
        "ctx32k": D1_CTX32K_FIXTURE,
        "ctx64k": D1_CTX64K_FIXTURE,
        # 2026-07-19, multi-agent-coding 256K-feasibility task: see
        # workloads.py's own docstring on these three (num_requests=4, not
        # 16, deliberately).
        "ctx128k": CTX128K_FIXTURE,
        "ctx200k": CTX200K_FIXTURE,
        "ctx256k": CTX256K_FIXTURE,
    }[fixture_key]
    # 2026-07-18/19, D1 64K-capacity-raise task (notes/2026-07-18-
    # session-review-and-next-steps.md section 16), generalized 2026-07-19
    # from a ctx64k-only special case to ANY fixture: this runtime's
    # per-slot KV-cache capacity ceiling is blocks_per_slot*block_size
    # tokens -- a prompt that exceeds it fails during prefill alone, at ANY
    # concurrency (the check is per-slot, not per-batch). Fail fast with a
    # clear, actionable message instead of the generic RuntimeError from
    # deep inside build_attention_metadata_batch mid-prefill.
    if blocks_per_slot * 16 < fixture.prompt_len + max_tokens:
        raise SystemExit(
            f"--fixture {fixture_key} needs --blocks-per-slot >= "
            f"{-(-(fixture.prompt_len + max_tokens) // 16)} (current "
            f"blocks_per_slot={blocks_per_slot} only covers "
            f"{blocks_per_slot * 16} tokens/slot); pass --blocks-per-slot "
            f"explicitly for this fixture."
        )
    prompts = load_prompt_token_ids(fixture)
    if num_requests is not None:
        prompts = prompts[:num_requests]

    thermal_before_load = _gpu_thermal()

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=min(
            max(40960, fixture.prompt_len + max_tokens + 1024), _MODEL_MAX_POSITION_EMBEDDINGS
        ),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    # 2026-07-17, Phase 3: ``enable_cudagraph`` reserves the LAST
    # ``concurrency`` logical slots of ``num_slots`` permanently for
    # ``CapturedBatchDecodeGraph``'s own disposable warmup (see
    # ``DirectModelRunner._get_verify_graph``'s docstring) -- real request
    # traffic only ever uses logical slots ``0..concurrency-1`` either way
    # (unaffected below), so doubling ``num_slots`` here is purely reserved
    # spare capacity, never exposed to real requests.
    num_slots = 2 * concurrency if cudagraph else concurrency
    runner = DirectModelRunner(
        vllm_config,
        num_slots=num_slots,
        block_size=64,
        blocks_per_slot=blocks_per_slot,
        enable_cudagraph=cudagraph,
    )
    thermal_after_load = _gpu_thermal()

    reps = [
        _run_measurement(
            torch, runner, prompts, max_tokens, concurrency, r + 1, batched=batched, chunk_size=chunk_size
        )
        for r in range(repeats)
    ]

    return {
        # Liveness/sanity signal only -- see `_sanity_check_reps`'s
        # docstring. This is NOT a correctness check (this script doesn't
        # re-verify generated tokens); it was a hardcoded `True` literal
        # before 2026-07-18's fix (notes/2026-07-18-session-review-and-
        # next-steps.md section 20.3).
        "passed": _sanity_check_reps(reps),
        "num_requests": len(prompts),
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "k": K,
        "batched": batched,
        "cudagraph": cudagraph,
        "blocks_per_slot": blocks_per_slot,
        "chunk_size": chunk_size,
        "repeats": repeats,
        "reps": reps,
        "thermal_before_load": thermal_before_load,
        "thermal_after_load": thermal_after_load,
        "fixture": fixture.path,
        "fixture_seed": fixture.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--fixture",
        choices=["n16", "n128", "ctx16k", "ctx32k", "ctx64k", "ctx128k", "ctx200k", "ctx256k"],
        default="n16",
    )
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--batched",
        action="store_true",
        help="use the 2026-07-17 cross-slot batched MTP coordinator "
        "(mtp_prefill_batch/mtp_verify_and_commit_batch, ONE shared kernel "
        "launch per round across all concurrent slots) instead of the "
        "original single-slot round-robin path.",
    )
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="2026-07-17, Phase 3: CUDA-graph-capture the verify forward "
        "inside mtp_verify_and_commit_batch (requires --batched). Doubles "
        "num_slots (the extra half is reserved, disposable capture-warmup "
        "capacity -- see DirectModelRunner._get_verify_graph).",
    )
    parser.add_argument(
        "--blocks-per-slot",
        type=int,
        default=2560,
        help="2026-07-18/19, D1 64K-capacity-raise task: DirectModelRunner's "
        "per-slot KV-cache capacity ceiling is blocks_per_slot * block_size "
        "(default 2560*16=40960 tokens/slot -- covers every existing 4K/16K/"
        "32K shape this project has measured). Left at its default for every "
        "existing invocation; only long-context (>32K) fixtures need this "
        "raised (e.g. 5120 for ctx64k, ~24%% margin over the 65536+256-token "
        "minimum). Raising it costs VRAM ONLY for THIS runner instance -- it "
        "is a per-instance constructor arg, not a global default -- but ALSO "
        "rescales this runner's own decode_fixed_kv_split_size (see "
        "DirectModelRunner.__init__), so any change should be re-validated "
        "against the regression suite, not assumed safe.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="2026-07-19, chunked-prefill round: split mtp_prefill_batch's "
        "target/draft-model forward calls into sequential chunk_size-token "
        "pieces instead of one giant forward covering the whole prompt -- "
        "bounds peak prefill activation memory to chunk_size*concurrency "
        "regardless of total prompt length (see "
        "oracle.qwen36_vllm.direct_model_runner.DirectModelRunner.mtp_prefill_batch's "
        "docstring and notes/2026-07-18-session-review-and-next-steps.md "
        "section 19). Requires --batched (the singular, non-batched path "
        "has no chunked prefill). Default None preserves every existing "
        "invocation's single-shot prefill behavior byte-for-byte.",
    )
    args = parser.parse_args()
    if args.cudagraph and not args.batched:
        parser.error("--cudagraph requires --batched")
    if args.chunk_size is not None and not args.batched:
        parser.error("--chunk-size requires --batched")

    result = _run_once(
        args.max_tokens,
        args.concurrency,
        args.fixture,
        args.num_requests,
        args.repeats,
        batched=args.batched,
        cudagraph=args.cudagraph,
        blocks_per_slot=args.blocks_per_slot,
        chunk_size=args.chunk_size,
    )

    import json

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
