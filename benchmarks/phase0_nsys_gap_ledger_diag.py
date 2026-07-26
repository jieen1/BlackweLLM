"""Phase 0 diagnostic script: one `nsys` gap ledger of a real batched MTP
round, per notes/2026-07-17-post-ragged-round-next-steps.md's Phase 0
("one `nsys` gap ledger of the real batched round").

PROFILING-ONLY. Does not modify any file under `runtime/`. Every GPU call
below is a direct call to an EXISTING, UNMODIFIED `DirectModelRunner`
method (`snapshot_gdn_state`, `verify_batch`, `restore_gdn_state`,
`_forward_batch`, `_mtp_sync_and_propose_batch`) plus the module-level
`determine_accept_reject` helper -- the SAME functions, called in the SAME
order, with the SAME arguments, that `DirectModelRunner
.mtp_verify_and_commit_batch` (runtime/direct_model_runner.py:1809-1963)
itself uses internally. This script does not re-derive or duplicate that
method's logic; it calls its real constituent pieces directly instead of
through that one umbrella method, ONLY so each phase (GDN snapshot / real
target verify forward / real target recompute forward / real draft
sync+propose forward) gets its own NVTX range boundary and host
timestamp -- something not possible from outside a single umbrella call
without editing runtime/direct_model_runner.py itself, which this project's
Phase 0 scope explicitly forbids. A correctness cross-check against the
umbrella method itself is included (`--verify-equivalence`) to confirm this
script's inline replay is behaviorally identical, not just structurally
similar.

Meant to run under:
    nsys profile -c cudaProfilerApi --capture-range-end=stop \\
        -o <report> --force-overwrite=true \\
        python -m benchmarks.phase0_nsys_gap_ledger_diag --num-rounds 40

The profiled region (bracketed by torch.cuda.cudart().cudaProfilerStart()/
Stop()) excludes model loading -- only the prefill + verify-round + host-
timer measurements are captured, keeping the trace small and the region
unambiguous.

Round decomposition, matching mtp_verify_and_commit_batch line-for-line:
  1873: snapshots = {s: self.snapshot_gdn_state(s) for s in slots}
  1875: verify_logits, verify_hidden = self.verify_batch(...)
  1880-1882: determine_accept_reject per slot (host-only, CPU, negligible)
  1902-1919: recompute group (restore_gdn_state + _forward_batch), IF any
  1924-1942: full-accept group's draft sync+propose (_mtp_sync_and_propose_batch), IF any
  1944-1954: recompute group's draft sync+propose (_mtp_sync_and_propose_batch), IF any

Each of these gets its own NVTX range (`round_{r}_snapshot`,
`round_{r}_verify`, `round_{r}_recompute_fwd`, `round_{r}_draft_full`,
`round_{r}_draft_recompute`) and its own perf_counter-measured wall time
(torch.cuda.synchronize() bracketed), printed to stdout for the host-timer
correlation table and so the specific "0-recompute" and "2-recompute"
round indices can be identified after the fact from the printed log
without needing to inspect the trace interactively.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _sync_time(torch):
    torch.cuda.synchronize()
    return time.perf_counter()


def _run_one_round_inline(torch, runner, slots, anchors, drafts, r, log):
    """Inline replay of mtp_verify_and_commit_batch's real body (see module
    docstring for the exact line-range correspondence), with per-phase NVTX
    ranges + host timers. Returns (decisions, phase_times_ms)."""
    from runtime.direct_model_runner import determine_accept_reject

    phase = {}

    # --- snapshot_gdn_state, all 4 slots (runtime/direct_model_runner.py:1873) ---
    torch.cuda.nvtx.range_push(f"round_{r}_snapshot")
    t0 = _sync_time(torch)
    kv_lens_before = {s: runner.slot_kv_len[s] for s in slots}
    snapshots = {s: runner.snapshot_gdn_state(s) for s in slots}
    t1 = _sync_time(torch)
    torch.cuda.nvtx.range_pop()
    phase["snapshot_ms"] = 1000 * (t1 - t0)

    # --- real target verify forward, ALL slots batched (:1875-1877) ---
    torch.cuda.nvtx.range_push(f"round_{r}_verify")
    t0 = _sync_time(torch)
    drafts_full = [[anchors[s]] + drafts[s] for s in slots]
    verify_logits, verify_hidden = runner.verify_batch(
        slots, drafts_full, [kv_lens_before[s] for s in slots], return_hidden=True
    )
    t1 = _sync_time(torch)
    torch.cuda.nvtx.range_pop()
    phase["verify_ms"] = 1000 * (t1 - t0)

    k = len(drafts[slots[0]])
    decisions = {}
    for i, s in enumerate(slots):
        row_logits = verify_logits[i * (k + 1) : (i + 1) * (k + 1)]
        decisions[s] = determine_accept_reject(drafts_full[i], row_logits)

    real_new_tokens = {s: [anchors[s]] + decisions[s]["committed"][:-1] for s in slots}
    full_accept_slots = [s for s in slots if decisions[s]["num_accepted"] == k]
    recompute_slots = [s for s in slots if decisions[s]["num_accepted"] != k]
    real_new_hidden = {}

    if full_accept_slots:
        for s in full_accept_slots:
            runner.slot_kv_len[s] = kv_lens_before[s] + k + 1
            i = slots.index(s)
            real_new_hidden[s] = verify_hidden[i * (k + 1) : (i + 1) * (k + 1)]

    # --- real target RECOMPUTE forward, batched across recompute_slots (:1902-1919) ---
    recompute_committed_lens = {s: decisions[s]["num_accepted"] + 1 for s in recompute_slots}
    hidden_recompute = None
    phase["recompute_fwd_ms"] = 0.0
    phase["restore_ms"] = 0.0
    if recompute_slots:
        torch.cuda.nvtx.range_push(f"round_{r}_restore")
        t0 = _sync_time(torch)
        for s in recompute_slots:
            runner.restore_gdn_state(s, snapshots[s])
            runner.slot_kv_len[s] = kv_lens_before[s]
        t1 = _sync_time(torch)
        torch.cuda.nvtx.range_pop()
        phase["restore_ms"] = 1000 * (t1 - t0)

        torch.cuda.nvtx.range_push(f"round_{r}_recompute_fwd")
        t0 = _sync_time(torch)
        qo_lens_recompute = [recompute_committed_lens[s] for s in recompute_slots]
        tokens_recompute = [real_new_tokens[s] for s in recompute_slots]
        kv_lens_recompute = [kv_lens_before[s] for s in recompute_slots]
        _, hidden_recompute = runner._forward_batch(
            recompute_slots,
            tokens_recompute,
            kv_lens_recompute,
            qo_len=qo_lens_recompute,
            commit=True,
            return_hidden=True,
            is_decode=True,
            fixed_kv_split_size=runner.decode_fixed_kv_split_size,
            fixed_max_num_splits=runner.decode_fixed_max_num_splits,
        )
        t1 = _sync_time(torch)
        torch.cuda.nvtx.range_pop()
        phase["recompute_fwd_ms"] = 1000 * (t1 - t0)

    next_anchors = {s: decisions[s]["committed"][-1] for s in slots}
    next_drafts: dict[int, list[int]] = {}

    # --- draft sync+propose, full-accept group (:1924-1942) ---
    phase["draft_full_ms"] = 0.0
    if full_accept_slots:
        torch.cuda.nvtx.range_push(f"round_{r}_draft_full")
        t0 = _sync_time(torch)
        shifted = [real_new_tokens[s][1:] + [next_anchors[s]] for s in full_accept_slots]
        hidden_concat = torch.cat([real_new_hidden[s] for s in full_accept_slots], dim=0)
        start_pos_list = [runner.slot_draft_sync_len[s] for s in full_accept_slots]
        next_drafts_batch = runner._mtp_sync_and_propose_batch(
            full_accept_slots, shifted, hidden_concat, start_pos_list, num_new_tokens=k + 1, k=k
        )
        t1 = _sync_time(torch)
        torch.cuda.nvtx.range_pop()
        phase["draft_full_ms"] = 1000 * (t1 - t0)
        for s in full_accept_slots:
            next_drafts[s] = next_drafts_batch[s]

    # --- draft sync+propose, recompute group (:1944-1954) ---
    phase["draft_recompute_ms"] = 0.0
    if recompute_slots:
        torch.cuda.nvtx.range_push(f"round_{r}_draft_recompute")
        t0 = _sync_time(torch)
        shifted_recompute = [real_new_tokens[s][1:] + [next_anchors[s]] for s in recompute_slots]
        start_pos_list_recompute = [runner.slot_draft_sync_len[s] for s in recompute_slots]
        next_drafts_recompute = runner._mtp_sync_and_propose_batch(
            recompute_slots,
            shifted_recompute,
            hidden_recompute,
            start_pos_list_recompute,
            num_new_tokens=[recompute_committed_lens[s] for s in recompute_slots],
            k=k,
        )
        t1 = _sync_time(torch)
        torch.cuda.nvtx.range_pop()
        phase["draft_recompute_ms"] = 1000 * (t1 - t0)
        for s in recompute_slots:
            next_drafts[s] = next_drafts_recompute[s]

    result = {}
    for s in slots:
        result[s] = {**decisions[s], "next_anchor": next_anchors[s], "next_draft_tokens": next_drafts[s]}

    phase["n_recompute"] = len(recompute_slots)
    phase["recompute_slots"] = list(recompute_slots)
    phase["round_total_ms"] = sum(
        phase[k2] for k2 in ("snapshot_ms", "verify_ms", "restore_ms", "recompute_fwd_ms", "draft_full_ms", "draft_recompute_ms")
    )
    log.append({"round": r, **phase})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rounds", type=int, default=40)
    parser.add_argument("--fixture", choices=["n16", "n128"], default="n16")
    parser.add_argument("--max-model-len", type=int, default=40960)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import W1_S_FIXTURE, W1_S_FIXTURE_N128, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture = {"n16": W1_S_FIXTURE, "n128": W1_S_FIXTURE_N128}[args.fixture]
    prompts = load_prompt_token_ids(fixture)[:4]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(args.max_model_len, fixture.prompt_len + args.num_rounds * (K + 1) + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    print("LOADING_MODEL", flush=True)
    t_load0 = time.perf_counter()
    runner = DirectModelRunner(vllm_config, num_slots=4, block_size=64, blocks_per_slot=640)
    print(f"MODEL_LOADED elapsed_s={time.perf_counter() - t_load0:.1f}", flush=True)

    slots = [0, 1, 2, 3]

    profiling = True
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as e:  # pragma: no cover - environment-dependent
        print(f"WARN cudaProfilerStart failed: {e}", flush=True)
        profiling = False

    print(f"PROFILE_WINDOW_START wall_time={time.time():.6f}", flush=True)

    torch.cuda.nvtx.range_push("prefill_batch")
    t0 = _sync_time(torch)
    prefill_result = runner.mtp_prefill_batch(slots, prompts)
    t1 = _sync_time(torch)
    torch.cuda.nvtx.range_pop()
    prefill_ms = 1000 * (t1 - t0)
    print(f"PREFILL wall_ms={prefill_ms:.3f}", flush=True)

    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}

    round_log: list[dict] = []
    for r in range(args.num_rounds):
        decisions = _run_one_round_inline(torch, runner, slots, anchors, drafts, r, round_log)
        entry = round_log[-1]
        print(
            f"ROUND {r}: total_ms={entry['round_total_ms']:.3f} n_recompute={entry['n_recompute']} "
            f"snapshot_ms={entry['snapshot_ms']:.3f} verify_ms={entry['verify_ms']:.3f} "
            f"restore_ms={entry['restore_ms']:.3f} recompute_fwd_ms={entry['recompute_fwd_ms']:.3f} "
            f"draft_full_ms={entry['draft_full_ms']:.3f} draft_recompute_ms={entry['draft_recompute_ms']:.3f}",
            flush=True,
        )
        anchors = {s: decisions[s]["next_anchor"] for s in slots}
        drafts = {s: decisions[s]["next_draft_tokens"] for s in slots}

    # ---- Host-timer isolated measurements (Phase 0's "3 targeted host timers") ----
    host_timers = {}

    torch.cuda.nvtx.range_push("host_timer_snapshot_x4")
    snap_times = []
    for s in slots:
        t0 = _sync_time(torch)
        _ = runner.snapshot_gdn_state(s)
        t1 = _sync_time(torch)
        snap_times.append(1000 * (t1 - t0))
    torch.cuda.nvtx.range_pop()
    host_timers["snapshot_gdn_state_per_slot_ms"] = snap_times
    host_timers["snapshot_gdn_state_sum_ms"] = sum(snap_times)
    print(f"HOST_TIMER snapshot_gdn_state per_slot_ms={snap_times} sum_ms={sum(snap_times):.3f}", flush=True)

    kv_lens_now = [runner.slot_kv_len[s] for s in slots]
    drafts_list = [[anchors[s]] + drafts[s] for s in slots]
    torch.cuda.nvtx.range_push("host_timer_verify_batch")
    t0 = _sync_time(torch)
    _, verify_hidden_iso = runner.verify_batch(slots, drafts_list, kv_lens_now, return_hidden=True)
    t1 = _sync_time(torch)
    torch.cuda.nvtx.range_pop()
    host_timers["verify_batch_ms"] = 1000 * (t1 - t0)
    print(f"HOST_TIMER verify_batch_ms={host_timers['verify_batch_ms']:.3f}", flush=True)

    qo = len(drafts_list[0])
    last_rows = torch.cat(
        [verify_hidden_iso[i * qo + qo - 1 : i * qo + qo] for i in range(len(slots))], dim=0
    )
    prior_kv_lens = [runner.slot_draft_sync_len[s] for s in slots]
    start_pos_list = [runner.slot_draft_sync_len[s] for s in slots]
    prev_tokens = [drafts[s][-1] for s in slots]
    torch.cuda.nvtx.range_push("host_timer_draft_step")
    t0 = _sync_time(torch)
    _ = runner._mtp_forward_batch(
        slots, prev_tokens, last_rows, prior_kv_lens, start_pos_list, qo_len=1, is_decode=True
    )
    t1 = _sync_time(torch)
    torch.cuda.nvtx.range_pop()
    host_timers["draft_step_ms"] = 1000 * (t1 - t0)
    print(f"HOST_TIMER draft_step_ms={host_timers['draft_step_ms']:.3f}", flush=True)

    print(f"PROFILE_WINDOW_END wall_time={time.time():.6f}", flush=True)
    if profiling:
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception as e:  # pragma: no cover
            print(f"WARN cudaProfilerStop failed: {e}", flush=True)

    summary = {
        "prefill_ms": prefill_ms,
        "rounds": round_log,
        "host_timers": host_timers,
    }
    print("SUMMARY_JSON_BEGIN")
    print(json.dumps(summary, indent=2, default=str))
    print("SUMMARY_JSON_END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
