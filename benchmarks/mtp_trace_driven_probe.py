"""Sol Phase 1: trace-driven MTP scheduling-overhead probe.

Strictly time-boxed (per the coordinator's explicit instruction), NOT a
real drafter. There is no `Qwen3_5MTP` model loaded here and no real
proposal step -- the accept/reject outcome for every round/slot is a
SYNTHETIC, pre-generated trace (a fixed per-token acceptance probability
fed through a seeded RNG), decoupled from what any real draft model
would actually produce. What IS real: every GPU operation this script
drives is the actual, already-verified production mechanism -- a real
batched `verify_batch()` forward (qo_len=K+1, concurrency=4), real
`snapshot_gdn_state()`/`restore_gdn_state()`, and a real committed-length
recompute forward on any round a slot's trace outcome is a partial/full
reject. This directly fixes, for this script's own bookkeeping, the
code-level gap Codex-sol flagged in `_forward_batch()`/`verify_batch()`
(they advance `slot_kv_len` by the full `qo_len` unconditionally,
conflating "positions physically written speculatively" with "positions
actually committed") -- see the manual `slot_kv_len` correction below.

Purpose: measure ONLY control-plane/scheduling overhead -- GPU-busy time
(via CUDA events) vs. wall-clock time (via `time.perf_counter()`) across
a realistic MTP verify+rollback round shape -- to answer one question:
"once you remove Python/vLLM scheduling overhead, how much headroom is
actually available on the launch-gap/GPU-busy dimension?" If GPU-busy%
is already high here, the eventual full-drafter engineering investment
(see notes/direct-model-runner-design.md's worst-case redesign scope)
has correspondingly less to gain; if it's low, that's a concrete signal
the full integration is worth its cost.

**MANDATORY LABELING for every number this script prints**: any
"acceptance rate" or "accepted tokens/s" here is a CONTROLLED-TRACE
SCHEDULING-OVERHEAD UPPER-BOUND ESTIMATE, not a real MTP acceptance-rate
or performance conclusion, and does NOT substitute for the real W1/W2
acceptance-rate-gate comparison against native vLLM MTP (still pending,
blocked on real draft-model integration).

Usage:
    python -m benchmarks.mtp_trace_driven_probe
"""

from __future__ import annotations

import os
import random
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"
K = 3  # num_speculative_tokens, matching this project's real production MTP K=3
CONCURRENCY = 4
NUM_ROUNDS = 20
WARMUP_ROUNDS = 3
SEED = 12345
DUMMY = 100  # arbitrary valid token id; content is irrelevant for a timing-only probe

# Three synthetic per-token-acceptance-probability configs:
#   1.0  -- best case, never rejects (equivalent to the "self-drafting"
#           upper-bound fallback proposed in the prior round's design doc)
#   0.65 -- this project's own earlier real MTP measurement for this
#           model/workload (see notes/PROGRESS.md's NVFP4 competitive
#           benchmark round) used ONLY as a representative target shape,
#           NOT re-derived or re-measured here
#   0.0  -- worst case, every round rejects at the first draft position
TRACE_CONFIGS = [
    ("best_case_p1.00", 1.0),
    ("realistic_p0.65", 0.65),
    ("worst_case_p0.00", 0.0),
]


def make_trace(num_rounds: int, concurrency: int, k: int, p: float, seed: int) -> list[list[int]]:
    """Synthetic per-round, per-slot accept_len in [0, k]. Per-token
    Bernoulli(p) trials, first failure truncates -- NOT derived from any
    real model prediction."""
    rng = random.Random(seed)
    trace = []
    for _ in range(num_rounds):
        row = []
        for _ in range(concurrency):
            accept_len = 0
            for _ in range(k):
                if rng.random() < p:
                    accept_len += 1
                else:
                    break
            row.append(accept_len)
        trace.append(row)
    return trace


def _run_config(runner, prompt_ids: list[int], label: str, p: float) -> dict:
    import torch

    for slot in range(CONCURRENCY):
        if runner.slot_kv_len[slot] != 0:
            runner.reset_slot(slot)
        runner.prefill(slot, prompt_ids)

    trace = make_trace(NUM_ROUNDS, CONCURRENCY, K, p, SEED)
    round_stats = []

    for r, accept_lens in enumerate(trace):
        kv_lengths = [runner.slot_kv_len[s] for s in range(CONCURRENCY)]
        draft_tokens = [[DUMMY] * (K + 1) for _ in range(CONCURRENCY)]

        # Always snapshot before verify -- a real system cannot know the
        # outcome ahead of time, so this cost is real and unconditional.
        snapshots = [runner.snapshot_gdn_state(s) for s in range(CONCURRENCY)]

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start_evt.record()
        runner.verify_batch(list(range(CONCURRENCY)), draft_tokens, kv_lengths)
        end_evt.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        verify_gpu_ms = start_evt.elapsed_time(end_evt)
        verify_wall_ms = (t1 - t0) * 1000.0

        # verify_batch()/_forward_batch() unconditionally advanced
        # slot_kv_len by the full qo_len=K+1 for every slot (the exact
        # "speculative-write vs. committed-length" conflation Codex-sol
        # flagged). Correct it per-slot down to the (synthetic) real
        # committed length, restoring GDN + recomputing where needed --
        # this IS the real production recovery mechanism, just driven by
        # a synthetic outcome instead of a real accept/reject decision.
        recompute_gpu_ms = 0.0
        recompute_wall_ms = 0.0
        num_recompute_slots = 0
        for s in range(CONCURRENCY):
            accept_len = accept_lens[s]
            if accept_len == K:
                # full accept: verify_batch() now always runs with
                # commit=False (2026-07-17 fix, separating "physically
                # written" from "committed" -- see direct_model_runner.py's
                # _forward_batch docstring), so this script must commit
                # explicitly here instead of relying on verify's own advance.
                runner.slot_kv_len[s] = kv_lengths[s] + K + 1
                continue
            num_recompute_slots += 1
            runner.restore_gdn_state(s, snapshots[s])
            runner.slot_kv_len[s] = kv_lengths[s]
            committed_len = accept_len + 1  # accepted drafts + 1 recovery/bonus token
            # _forward_batch's token_ids shape convention: flat list when
            # qo_len==1, list-of-per-slot-lists when qo_len>1 (single slot
            # here, so either a bare [tok] or [[tok, tok, ...]]).
            recompute_tokens = [DUMMY] if committed_len == 1 else [[DUMMY] * committed_len]
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            tr0 = time.perf_counter()
            se.record()
            runner._forward_batch([s], recompute_tokens, [kv_lengths[s]], qo_len=committed_len)
            ee.record()
            torch.cuda.synchronize()
            tr1 = time.perf_counter()
            recompute_gpu_ms += se.elapsed_time(ee)
            recompute_wall_ms += (tr1 - tr0) * 1000.0

        round_stats.append(
            {
                "round": r,
                "accept_lens": accept_lens,
                "num_recompute_slots": num_recompute_slots,
                "verify_gpu_ms": verify_gpu_ms,
                "verify_wall_ms": verify_wall_ms,
                "recompute_gpu_ms": recompute_gpu_ms,
                "recompute_wall_ms": recompute_wall_ms,
                "total_gpu_ms": verify_gpu_ms + recompute_gpu_ms,
                "total_wall_ms": verify_wall_ms + recompute_wall_ms,
            }
        )

    measured = round_stats[WARMUP_ROUNDS:]
    n = len(measured)
    total_gpu = sum(rs["total_gpu_ms"] for rs in measured)
    total_wall = sum(rs["total_wall_ms"] for rs in measured)
    total_committed_tokens = sum(
        sum(min(a, K) + 1 for a in rs["accept_lens"]) for rs in measured
    )
    total_recompute_slot_rounds = sum(rs["num_recompute_slots"] for rs in measured)

    return {
        "label": label,
        "target_per_token_accept_p": p,
        "num_rounds_measured": n,
        "avg_wall_ms_per_round": total_wall / n,
        "avg_gpu_busy_ms_per_round": total_gpu / n,
        "gpu_busy_pct": 100.0 * total_gpu / total_wall if total_wall else float("nan"),
        "launch_gap_pct": 100.0 * (1.0 - total_gpu / total_wall) if total_wall else float("nan"),
        "avg_recompute_slot_rounds_per_round": total_recompute_slot_rounds / n,
        "realized_committed_tokens_per_round_per_slot": total_committed_tokens / n / CONCURRENCY,
        "controlled_trace_accepted_tokens_per_sec_UPPER_BOUND": (
            total_committed_tokens / (total_wall / 1000.0) if total_wall else float("nan")
        ),
    }


def _run_once() -> dict:
    import torch  # noqa: F401  (import kept local, matches other benchmark scripts' pattern)

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    runner = DirectModelRunner(vllm_config, num_slots=CONCURRENCY, block_size=64, blocks_per_slot=32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    results = []
    for label, p in TRACE_CONFIGS:
        results.append(_run_config(runner, prompt_ids, label, p))
    return {"configs": results}


def main() -> int:
    result = _run_once()
    print(
        "=== Sol Phase 1 trace-driven MTP scheduling-overhead probe "
        "(CONTROLLED-TRACE UPPER BOUND, not real MTP acceptance/perf) ==="
    )
    print(f"K={K} concurrency={CONCURRENCY} rounds_measured/config={NUM_ROUNDS - WARMUP_ROUNDS}\n")
    for cfg in result["configs"]:
        print(f"--- {cfg['label']} (target per-token accept p={cfg['target_per_token_accept_p']}) ---")
        print(f"  avg wall/round:            {cfg['avg_wall_ms_per_round']:.2f} ms")
        print(f"  avg GPU-busy/round:        {cfg['avg_gpu_busy_ms_per_round']:.2f} ms")
        print(f"  GPU-busy %%:                {cfg['gpu_busy_pct']:.1f}%")
        print(f"  launch-gap %% (overhead):   {cfg['launch_gap_pct']:.1f}%")
        print(f"  avg recompute-slot-rounds/round: {cfg['avg_recompute_slot_rounds_per_round']:.2f} / {CONCURRENCY}")
        print(
            "  realized committed tokens/round/slot (CONTROLLED-TRACE, "
            f"not a real acceptance rate): {cfg['realized_committed_tokens_per_round_per_slot']:.2f} / {K + 1}"
        )
        print(
            "  controlled-trace accepted tokens/s UPPER BOUND: "
            f"{cfg['controlled_trace_accepted_tokens_per_sec_UPPER_BOUND']:.1f}"
        )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
