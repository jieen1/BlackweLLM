"""Correctness gate for prefix-cache P2 -- fan-out fork (Pattern A,
same-round sharing) (``notes/prefix-cache-design.md`` sec 5, "P2 -- Fan-out
fork (Pattern A, same-round sharing; self-contained)").

P2 makes >=2 same-round requests that share a token prefix compute that
prefix ONCE and reference it from every sibling, instead of recomputing it
N times:

* the group LEADER (``slots[0]``) prefills ``[0, Lc)`` -- ``Lc`` = the
  block-aligned common-prefix length, capped at ``min(prompt_len) - 1`` so
  every request keeps >=1 suffix token to recompute for its own logits (sec
  3.8) -- forcing a GDN checkpoint boundary there (``snapshot_gdn_state``);
* each SIBLING references the leader's ``[0, Lc)`` attention blocks
  (``BlockPool.reference``, ``ref_cnt += 1``, all 17 attention layers share
  one block-id namespace), ``restore_gdn_state`` the leader's snapshot
  (``allow_cross_slot=True``), sets ``slot_draft_sync_len = Lc``, and
  continue-prefills ONLY its own suffix ``[Lc, prompt_len)`` through the
  already-validated chunked-prefill continuation machinery.

No persistent hash index, no eviction -- the shared entry lives only for
this one admission round (P3 builds the persistent cache).

This script has a pure-Python part (no GPU/model) and a real-GPU part,
mirroring ``prefix_cache_block_table_check.py`` / ``prefix_cache_allocator_
check.py``. Checks, each mapped to the design doc's correctness invariants
(sec 4) and P2 test list:

Pure Python:
* ``reference_refcount`` -- the focused unit assertion P2 requires for the
  new ``BlockPool.reference`` primitive and the ``ref_cnt > 1`` free path:
  reference increments ``ref_cnt``; ``free`` decrements and re-queues a
  block ONLY when ``ref_cnt`` hits 0 (so a shared block is never handed back
  out while any slot still references it -- INV9); INV7/range guards hold on
  ``reference``; a shared block survives one referencer's free and is only
  re-queued after the LAST referencer frees it (no leak, R10).
* ``common_prefix`` -- ``DirectModelRunner._common_prefix_len`` returns the
  true longest-common token prefix (the fork's direct-comparison detector).

Real GPU (one ``DirectModelRunner``, ``enable_block_table=True`` +
``enable_prefix_cache=True``, MTP K=3), for N=2..4 siblings sharing a large
prefix with distinct suffixes:
* ``inv1_prefill_anchor`` (INV1) -- each sibling's fork prefill anchor
  matches an INDEPENDENT cold single-slot prefill of the same prompt.
* ``inv1_decode_equiv`` (INV1) -- over multiple MTP decode rounds, each
  sibling's committed tokens stay consistent with an independent cold
  reference that replays the SAME committed tokens (near-tie logit
  comparison, ``NEAR_TIE_LOGIT_MARGIN = 2.0`` per R6 -- NOT bytewise).
* ``inv2_signal_probe`` (INV2) -- per-slot marker tokens in the SUFFIX;
  zero cross-sibling crosstalk (no sibling's continuation echoes another
  sibling's distinct marker).
* ``inv4_multiround_mtp`` (INV4) -- multi-round MTP decode after the fork
  stays oracle-aligned per step (``draft_sync_len == kv_len`` invariant +
  near-tie next-token agreement), plus a ``draft_acceptance_rate`` sanity
  check vs a cold baseline.
* ``no_block_leak`` (R10/INV9) -- after fork + decode + ``reset_slot`` on
  every used slot, the pool's free count returns to baseline and every
  ``ref_cnt`` is 0 (shared blocks released exactly once per referencer).

Usage:
    python -m benchmarks.prefix_cache_fanout_check
"""

from __future__ import annotations

import os
import sys

# Repo-root bootstrap so this runs both as ``python -m benchmarks.<name>``
# (cwd = repo root) and as a direct ``python benchmarks/<name>.py`` path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
NUM_ROUNDS = 6
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

NEAR_TIE_LOGIT_MARGIN = 2.0  # established methodology (R6): a real, evidenced
# near-exact tie in this model's own distribution is kernel-path-sensitive
# fp8/batch non-associativity noise, NOT state corruption; distinct real
# candidates are typically separated by 8-13+ logit units. A mismatch is only
# a real failure if the reference's own logit for the fork's committed token
# is NOT within this margin of the reference's top candidate.

# Shared "system-prompt-like" prefix every sibling inherits (Pattern A: N
# sub-agents launched together share an identical context bundle). Long
# enough to span many 16-token blocks so the fork shares a multi-block
# prefix, not a trivial one.
SHARED_PREFIX = (
    "You are a careful coding assistant working inside a large repository. "
    "The weather today is mild and pleasant. The repository contains many "
    "modules, tests, and documentation files that must be kept consistent. "
    "Always read the existing code before changing it, and always run the "
    "relevant tests after a change. Keep diffs small and reviewable. "
) * 6

# Distinct 5-digit markers, no shared full-string overlap (signal-probe
# crosstalk targets, after batch_decode_signal_probe.py's methodology).
MARKERS = [84317, 52968, 71053, 39642]


def _make_prompt(slot_index: int) -> str:
    """Shared prefix + a sibling-distinct suffix that embeds this slot's own
    marker with a strong in-context copy cue. The suffix starts with a
    per-slot token (``Agent-<i>``) so the longest common prefix across
    siblings is exactly the shared prefix (the suffixes diverge immediately)."""
    marker = MARKERS[slot_index]
    suffix = (
        f"Agent-{slot_index} private task. The value of X is {marker}. "
        f"Repeat agent {slot_index}'s value of X exactly. The value of X is"
    )
    return SHARED_PREFIX + suffix


# ---------------------------------------------------------------------------
# Pure-Python part (no GPU, no model load).
# ---------------------------------------------------------------------------


def _check_reference_refcount() -> dict:
    """Focused unit assertion for ``BlockPool.reference`` + the ``ref_cnt > 1``
    free path (P2's new primitive; design doc sec 3.2 refcount rules, INV7/
    INV9, risk R10)."""
    from runtime.direct_model_runner import BlockPool

    errors = []
    pool = BlockPool(num_blocks=20, reserved=1)
    initial_free = pool.num_free_blocks()
    if initial_free != 19:
        errors.append(f"expected 19 free blocks initially, got {initial_free}")

    # Allocate 3 blocks for the "leader"; ref_cnt 0 -> 1 each.
    leader_blocks = pool.allocate(3)
    if any(pool.blocks[b].ref_cnt != 1 for b in leader_blocks):
        errors.append(f"leader blocks not ref_cnt==1 after allocate: {leader_blocks}")

    # Two siblings reference the same 3 blocks -> ref_cnt 1 -> 3.
    pool.reference(leader_blocks)
    pool.reference(leader_blocks)
    for b in leader_blocks:
        if pool.blocks[b].ref_cnt != 3:
            errors.append(f"block {b} ref_cnt={pool.blocks[b].ref_cnt} != 3 after 2 references")
    # Referenced blocks must NOT be in the free queue (INV9).
    if pool.num_free_blocks() != initial_free - 3:
        errors.append(
            f"referencing changed the free count: {initial_free} -> {pool.num_free_blocks()} "
            "(referenced blocks must stay out of the free queue)"
        )

    # Free once (leader releases): ref_cnt 3 -> 2, still NOT re-queued.
    pool.free(leader_blocks)
    for b in leader_blocks:
        if pool.blocks[b].ref_cnt != 2:
            errors.append(f"block {b} ref_cnt={pool.blocks[b].ref_cnt} != 2 after 1st free")
    if pool.num_free_blocks() != initial_free - 3:
        errors.append("a still-shared block (ref_cnt=2) was re-queued into the free pool")

    # Free the second referencer: ref_cnt 2 -> 1, still not re-queued.
    pool.free(leader_blocks)
    if pool.num_free_blocks() != initial_free - 3:
        errors.append("a still-shared block (ref_cnt=1) was re-queued into the free pool")

    # Free the LAST referencer: ref_cnt 1 -> 0, NOW re-queued (exactly once).
    pool.free(leader_blocks)
    for b in leader_blocks:
        if pool.blocks[b].ref_cnt != 0:
            errors.append(f"block {b} ref_cnt={pool.blocks[b].ref_cnt} != 0 after last free")
    if pool.num_free_blocks() != initial_free:
        errors.append(
            f"free count {pool.num_free_blocks()} != baseline {initial_free} after all "
            "referencers freed (shared blocks must return exactly once, no leak)"
        )

    # INV7 / range guards on reference.
    for bad in [0, -1]:
        raised = False
        try:
            pool.reference([bad])
        except RuntimeError:
            raised = True
        if not raised:
            errors.append(f"reference did not reject reserved/invalid block {bad}")
    raised = False
    try:
        pool.reference([pool.num_blocks + 5])
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("reference did not reject an out-of-range block id")

    # Cannot reference a block that is not currently allocated (ref_cnt == 0):
    # it may already be back in the free queue and handed to another slot.
    fresh = pool.allocate(1)
    pool.free(fresh)  # back to ref_cnt 0, in the free queue
    raised = False
    try:
        pool.reference(fresh)
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("reference did not reject a ref_cnt==0 (free-queued) block")

    return {"passed": not errors, "errors": errors}


def _check_common_prefix() -> dict:
    from runtime.direct_model_runner import DirectModelRunner

    errors = []
    cases = [
        ([[1, 2, 3, 4], [1, 2, 3, 4]], 4),
        ([[1, 2, 3, 4], [1, 2, 9, 9]], 2),
        ([[1, 2, 3], [1, 2, 3, 4, 5]], 3),
        ([[5, 6], [7, 8]], 0),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 3),
        ([[1, 2, 3], [1, 2, 3], [1, 9, 3]], 1),
        ([], 0),
    ]
    for prompts, want in cases:
        got = DirectModelRunner._common_prefix_len(prompts)
        if got != want:
            errors.append(f"_common_prefix_len({prompts}) = {got}, want {want}")
    return {"passed": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Real-GPU part.
# ---------------------------------------------------------------------------


def _ref_check(runner, ref_slot, real_new_tokens, fork_next_anchor) -> dict:
    """Independent cold single-slot reference replay of one round's real
    committed tokens (mtp_batch_verify_check.py's established pattern): the
    reference slot consumes the REAL tokens the fork path committed and its
    own next-token prediction is compared against the fork's next anchor
    within the near-tie margin. Decouples each round from any prior round's
    possible near-tie divergence."""
    ref_logits = runner._forward(
        ref_slot,
        real_new_tokens,
        start_pos=runner.slot_kv_len[ref_slot],
        is_decode=(len(real_new_tokens) == 1),
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    ref_top1_logit = float(ref_last.max().item())
    ref_logit_for_fork_choice = float(ref_last[fork_next_anchor].item())
    near_tie_margin = ref_top1_logit - ref_logit_for_fork_choice
    exact_match = ref_predicted_next == fork_next_anchor
    return {
        "exact_match": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _reset_all(runner, slots) -> None:
    for s in slots:
        if runner.slot_kv_len[s] != 0 or runner.block_table[s]:
            runner.reset_slot(s)


def _run_fanout_case(runner, tok, n: int) -> dict:
    """One N-sibling fan-out case: fork-prefill N siblings sharing the prefix,
    then verify INV1 (prefill anchor + decode equivalence vs independent cold
    references), INV2 (signal-probe crosstalk), INV4 (multi-round MTP +
    acceptance sanity)."""
    fork_slots = list(range(n))
    ref_slots = list(range(4, 4 + n))  # fixed offset; never collides for n<=4
    prompts_text = [_make_prompt(i) for i in range(n)]
    prompts_ids = [tok.encode(p, add_special_tokens=False) for p in prompts_text]
    _reset_all(runner, fork_slots + ref_slots)

    result: dict = {"n": n, "checks": {}}

    # --- Fork prefill (the P2 path under test). ---
    fork_pr = runner.mtp_prefill_fanout_batch(fork_slots, prompts_ids)
    fork_anchors = {s: fork_pr[s]["anchor"] for s in fork_slots}
    fork_drafts = {s: fork_pr[s]["draft_tokens"] for s in fork_slots}

    # Confirm the fork actually engaged (shared a multi-block prefix), not a
    # silent fallback -- the leader's [0, Lc) blocks must be referenced by
    # every sibling (ref_cnt > 1 on the shared head blocks).
    shared_head = runner.block_table[fork_slots[0]][:1]
    fork_engaged = bool(shared_head) and all(
        runner.block_table[s][:1] == shared_head for s in fork_slots[1:]
    ) and all(runner.block_pool.blocks[b].ref_cnt >= 2 for b in shared_head)
    result["checks"]["fork_actually_engaged"] = fork_engaged
    result["shared_block_ref_cnt"] = (
        runner.block_pool.blocks[shared_head[0]].ref_cnt if shared_head else None
    )

    # --- INV1 (prefill): independent cold single-slot prefill anchors match. ---
    prefill_anchor_ok = True
    for i, s in enumerate(fork_slots):
        ref_first = runner.prefill(ref_slots[i], prompts_ids[i])
        if ref_first != fork_anchors[s]:
            prefill_anchor_ok = False
    result["checks"]["inv1_prefill_anchor"] = prefill_anchor_ok

    # --- INV1 (decode) + INV4: multi-round MTP after the fork. ---
    per_slot_rounds = {s: [] for s in fork_slots}
    total_draft_tokens = 0
    total_accepted = 0
    for _r in range(NUM_ROUNDS):
        decisions = runner.mtp_verify_and_commit_batch(fork_slots, fork_anchors, fork_drafts)
        for i, s in enumerate(fork_slots):
            decision = decisions[s]
            real_new_tokens = [fork_anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            total_draft_tokens += len(fork_drafts[s])
            total_accepted += decision["num_accepted"] + 1
            per_slot_rounds[s].append(
                {
                    "num_accepted": decision["num_accepted"],
                    "draft_sync_len_matches_kv_len": (
                        runner.slot_draft_sync_len[s] == runner.slot_kv_len[s]
                    ),
                    **ref_report,
                }
            )
            fork_anchors[s], fork_drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    inv1_decode_ok = all(
        all(r["content_ok"] for r in per_slot_rounds[s]) for s in fork_slots
    )
    inv4_sync_ok = all(
        all(r["draft_sync_len_matches_kv_len"] for r in per_slot_rounds[s]) for s in fork_slots
    )
    result["checks"]["inv1_decode_equiv"] = inv1_decode_ok
    result["checks"]["inv4_multiround_mtp"] = inv4_sync_ok and inv1_decode_ok
    fork_acceptance_rate = total_accepted / (total_draft_tokens + total_accepted) \
        if (total_draft_tokens + total_accepted) else 0.0
    result["fork_acceptance_rate"] = fork_acceptance_rate

    # --- INV2 (signal-probe): per-slot marker tokens live in the SUFFIX.
    # Fresh fork prefill (the MTP rounds above already advanced the fork
    # slots past the cue), then greedy-decode [prefill_anchor] + a short
    # burst: the prefill anchor IS the marker's first token (the cue ends
    # "The value of X is"), so the marker is spelled by [anchor] + burst.
    # Assert each sibling reproduces its OWN marker and NONE of the others'
    # (a distinct 5-digit number from another slot is unambiguous leakage). ---
    _reset_all(runner, fork_slots)
    inv2_pr = runner.mtp_prefill_fanout_batch(fork_slots, prompts_ids)
    gen_text: dict[int, str] = {}
    for i, s in enumerate(fork_slots):
        out_tokens: list[int] = [inv2_pr[s]["anchor"]]
        cur = inv2_pr[s]["anchor"]
        for _ in range(12):
            logits = runner._forward(s, [cur], start_pos=runner.slot_kv_len[s], is_decode=True)
            cur = int(logits[-1].argmax(dim=-1).item())
            out_tokens.append(cur)
        gen_text[s] = tok.decode(out_tokens, skip_special_tokens=True)

    inv2_ok = True
    inv2_detail: dict[str, object] = {}
    for i, s in enumerate(fork_slots):
        own = str(MARKERS[i])
        own_present = own in gen_text[s].replace(" ", "")
        cross_present = [
            str(MARKERS[j])
            for j in range(n)
            if j != i and str(MARKERS[j]) in gen_text[s].replace(" ", "")
        ]
        inv2_detail[f"slot{s}"] = {
            "own_marker": own,
            "own_present": own_present,
            "cross_leak": cross_present,
            "gen": gen_text[s][:60],
        }
        if cross_present:
            inv2_ok = False
        if not own_present:
            inv2_ok = False
    result["checks"]["inv2_signal_probe"] = inv2_ok
    result["inv2_detail"] = inv2_detail

    # --- INV4 acceptance sanity vs a cold baseline (single slot, prompt 0). ---
    baseline_slot = ref_slots[0]
    runner.reset_slot(baseline_slot)
    cpr = runner.mtp_prefill(baseline_slot, prompts_ids[0])
    c_anchor, c_drafts = cpr["anchor"], cpr["draft_tokens"]
    c_draft_tokens = 0
    c_accepted = 0
    for _r in range(NUM_ROUNDS):
        dec = runner.mtp_verify_and_commit(baseline_slot, c_anchor, c_drafts)
        c_draft_tokens += len(c_drafts)
        c_accepted += dec["num_accepted"] + 1
        c_anchor, c_drafts = dec["next_anchor"], dec["next_draft_tokens"]
    cold_acceptance_rate = c_accepted / (c_draft_tokens + c_accepted) \
        if (c_draft_tokens + c_accepted) else 0.0
    result["cold_acceptance_rate"] = cold_acceptance_rate
    # Sanity: the fork's acceptance rate must track the cold baseline (the
    # fork produces the same committed tokens, so draft quality is the same).
    result["checks"]["inv4_acceptance_sanity"] = (
        abs(fork_acceptance_rate - cold_acceptance_rate) < 0.20
    )

    # Leave every slot this case touched clean for the next case / the
    # block-leak check (also exercises reset_slot's shared-block release).
    _reset_all(runner, fork_slots + ref_slots)
    return result


def _check_no_block_leak(runner, tok) -> dict:
    """Fork + decode + reset every used slot; the pool's free count must
    return to baseline and every ref_cnt must be 0 (shared blocks released
    exactly once per referencer -- R10/INV9 with sharing)."""
    errors = []
    n = 3
    fork_slots = list(range(n))
    prompts_ids = [tok.encode(_make_prompt(i), add_special_tokens=False) for i in range(n)]
    # Establish a clean baseline: release every slot (prior cases may have
    # left some holding blocks), so free_before is the full empty-pool count.
    _reset_all(runner, list(range(runner.num_slots)))
    free_before = runner.block_pool.num_free_blocks()

    fork_pr = runner.mtp_prefill_fanout_batch(fork_slots, prompts_ids)
    fa = {s: fork_pr[s]["anchor"] for s in fork_slots}
    fd = {s: fork_pr[s]["draft_tokens"] for s in fork_slots}
    # A couple of decode rounds to grow each slot's private suffix blocks.
    for _ in range(2):
        decs = runner.mtp_verify_and_commit_batch(fork_slots, fa, fd)
        for s in fork_slots:
            fa[s], fd[s] = decs[s]["next_anchor"], decs[s]["next_draft_tokens"]

    # Shared head blocks must currently be referenced by all n slots.
    head = runner.block_table[fork_slots[0]][:1]
    if head and runner.block_pool.blocks[head[0]].ref_cnt != n:
        errors.append(
            f"shared head block ref_cnt={runner.block_pool.blocks[head[0]].ref_cnt} != {n} mid-case"
        )

    _reset_all(runner, fork_slots)
    free_after = runner.block_pool.num_free_blocks()
    if free_after != free_before:
        errors.append(
            f"block leak: free count {free_before} -> {free_after} after fork+decode+reset "
            "(shared blocks not released exactly once per referencer)"
        )
    leaked = [b.block_id for b in runner.block_pool.blocks if b.ref_cnt != 0]
    if leaked:
        errors.append(f"non-zero ref_cnt blocks remain after reset: {leaked[:10]}")
    for s in fork_slots:
        if runner.block_table[s] != []:
            errors.append(f"block_table[{s}] not cleared after reset: {runner.block_table[s]}")

    return {"passed": not errors, "errors": errors}


def _run_gpu_checks() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # num_slots=8: fork slots [0..3] + independent cold reference slots [4..7].
    # enable_cudagraph stays False -- this gate targets fork CORRECTNESS
    # (INV1/INV2/INV4); CUDA-graph replay parity over cache-hit block tables
    # is INV5, scoped to P3.
    runner = DirectModelRunner(
        vllm_config,
        num_slots=8,
        block_size=64,
        blocks_per_slot=32,
        enable_block_table=True,
        enable_prefix_cache=True,
        enable_cudagraph=False,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    case_results = {}
    overall = True
    for n in (2, 3, 4):
        case = _run_fanout_case(runner, tok, n)
        case_results[f"n{n}"] = case
        for name, ok in case["checks"].items():
            if not ok:
                overall = False

    leak = _check_no_block_leak(runner, tok)
    if not leak["passed"]:
        overall = False

    return {"passed": overall, "cases": case_results, "no_block_leak": leak}


def main() -> int:
    print("=== prefix_cache_fanout_check (P2 fan-out fork) ===")

    ref_result = _check_reference_refcount()
    print(f"reference_refcount: {'PASS' if ref_result['passed'] else 'FAIL'}")
    for e in ref_result["errors"]:
        print(f"  - {e}")

    cp_result = _check_common_prefix()
    print(f"common_prefix: {'PASS' if cp_result['passed'] else 'FAIL'}")
    for e in cp_result["errors"]:
        print(f"  - {e}")

    gpu_result = _run_gpu_checks()
    for case_name, case in gpu_result["cases"].items():
        print(f"{case_name}:")
        print(f"  fork_actually_engaged: {case['checks'].get('fork_actually_engaged')} "
              f"(shared_block_ref_cnt={case.get('shared_block_ref_cnt')})")
        for name, ok in case["checks"].items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        print(f"  fork_acceptance_rate={case.get('fork_acceptance_rate'):.4f} "
              f"cold_acceptance_rate={case.get('cold_acceptance_rate'):.4f}")
        for slot_key, detail in case.get("inv2_detail", {}).items():
            print(f"    {slot_key}: own={detail['own_marker']} "
                  f"own_present={detail['own_present']} cross_leak={detail['cross_leak']} "
                  f"gen={detail['gen']!r}")
    leak = gpu_result["no_block_leak"]
    print(f"no_block_leak: {'PASS' if leak['passed'] else 'FAIL'}")
    for e in leak["errors"]:
        print(f"  - {e}")

    overall = ref_result["passed"] and cp_result["passed"] and gpu_result["passed"]
    print(f"\npassed: {str(overall).lower()}")
    print(f"=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
