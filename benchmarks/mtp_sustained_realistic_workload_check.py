"""Realistic coding-agent-shaped sustained workload, WITH CUDA graphs
enabled (2026-07-18, closing the audit's top P1 -- see
``notes/2026-07-19-comprehensive-audit-and-forward-plan.md`` section 4.2
and ``notes/2026-07-18-session-review-and-next-steps.md`` section 23).

Every prior benchmark in this project is either (a) a synthetic uniform-
shape sweep (``mtp_w1s_our_runtime_perf.py``'s W1-S/W2-S fixtures: one
fixed prompt length, one fixed max_tokens, sequential-token-id content
that this project's own ``workloads.py`` documents as artificially
inflating draft-acceptance rate relative to real text), or (b) a short
correctness probe in EAGER mode (``mtp_async_arrival_check.py``:
``enable_cudagraph=False``, 6 requests, ~13-21s wall time). Nothing in
this project's history has combined: real, varied, distinguishable text
content shaped like actual coding-agent traffic (short chat questions,
medium code-explanation requests, long code-with-context requests);
continuous, staggered arrival with requests finishing and being replaced
over a sustained period (not one fixed batch that finishes together);
AND ``enable_cudagraph=True`` -- the exact combination the audit flagged
as never jointly exercised (ragged-length batched prefill admitting
multiple fresh requests at once, and a GROWING/shrinking active-batch
composition, interacting with the CUDA-graph verify/decode replay path).

**Design, following ``mtp_async_arrival_check.py``'s established pattern,
generalized from "6 hand-picked requests" to "a continuous stream over a
real wall-clock duration":**

- Three prompt classes (short chat / medium code-explanation / long
  code-with-context), each with its own (prompt_len, max_tokens) range,
  weighted 40/35/25 -- a real mix, not one uniform shape.
- Real, varied, distinguishable text: a single real-code filler corpus
  (several distinct Python functions/classes concatenated and repeated),
  tokenized ONCE; each request takes a random offset+length slice of it
  (so different requests show different code content, not the same
  repeated block) plus a unique per-request marker comment and a
  class-appropriate question. Deliberately NOT the "capital of France"
  factual-completion template ``mtp_async_arrival_check.py`` used -- that
  template's own investigation (see that module's docstring, section 22
  of the session-review doc) found it prone to a "forced-past-natural-
  stop degenerate repetition" artifact; open-ended code-explanation
  completions are far less likely to degenerate the same way at these
  generation lengths.
- Arrival is WALL-CLOCK-gated (``arrival_time_s``, a cumulative random
  gap per request), not round-index-gated like the original script's
  hand-picked waves -- a fixed request pool can't know its own round
  cadence in advance for a real-duration test the way 6 hand-picked
  waves could. The "queued if no slot is free yet" real admission-control
  wait is otherwise identical in spirit.
- Continuous batching: whenever >=1 production slot is free AND >=1
  request is eligible, admit ``min(free, eligible)`` of them together in
  ONE ragged ``mtp_prefill_batch`` call -- this naturally reproduces the
  audit's flagged scenario (a ragged multi-request admission joining an
  in-flight batch) many times over a long run, not as a single
  hand-crafted one-off.
- ``enable_cudagraph=True``, ``num_slots=16`` (production slots
  ``0..capacity-1``; a dedicated, REUSED independent-reference slot per
  production slot at ``capacity+p``; a dedicated, REUSED margin-diagnostic
  slot per production slot at ``2*capacity+p``, only touched on an actual
  divergence) -- satisfies ``num_slots >= 2*capacity`` for BOTH
  ``CapturedBatchDecodeGraph``/``CapturedMTPDraftStepGraph``'s own
  precapture range (``num_slots // 2 == 8 >= capacity``), reusing the
  exact reservation contract those classes' docstrings establish.

Correctness methodology: reuses this project's own established
independent-single-slot-reference-replay mechanism VERBATIM (imported
directly from ``mtp_async_arrival_check``, not re-implemented) --
``_ref_check``/``_near_tie_margin_diag``/the streak-based benign-near-tie
reclassification (``_looks_like_repetition_artifact``/
``_token_text_is_coherent``/``MAX_BENIGN_STREAK_ROUNDS``/
``NEAR_TIE_LOGIT_MARGIN``) -- applied every round, for every active slot,
for the ENTIRE run, not a start-of-run spot check.

Usage:
    python -m benchmarks.mtp_sustained_realistic_workload_check --duration-s 3600
    python -m benchmarks.mtp_sustained_realistic_workload_check --duration-s 90   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3

# -- Realistic coding-agent request-class shapes (prompt tokens, max new
# tokens), weighted to look like a real mixed workload, not one uniform
# fixture. --
REQUEST_CLASSES = {
    "short_chat": {
        "weight": 0.40,
        "prompt_len_range": (80, 220),
        "max_tokens_range": (24, 64),
        "questions": [
            "What does this function return?",
            "Is this code thread-safe? Answer in one sentence.",
            "Name one bug in this snippet.",
            "What is the time complexity of this loop?",
            "Does this function handle an empty input correctly?",
            "What would this print for an empty list?",
        ],
    },
    "medium_explain": {
        "weight": 0.35,
        "prompt_len_range": (500, 1100),
        "max_tokens_range": (80, 200),
        "questions": [
            "Explain what the following function does and suggest one improvement.",
            "Review this code for potential edge-case bugs.",
            "Rewrite the last function to be more efficient, and explain why.",
            "Add type hints and a docstring to the last function defined above.",
            "Explain the control flow of this code as if to a junior engineer.",
            "What test cases would you write to cover this code?",
        ],
    },
    "long_context": {
        "weight": 0.25,
        "prompt_len_range": (1800, 3800),
        "max_tokens_range": (150, 350),
        "questions": [
            "Given the module above, explain how the functions interact and "
            "identify any race conditions or shared mutable state.",
            "Given this file, write a short docstring summarizing every "
            "function defined in it.",
            "Trace through this code and explain the output for a small "
            "example input.",
            "Refactor this module to remove duplicated logic across its "
            "functions, and explain your changes.",
            "Identify the three most error-prone functions in this file and "
            "explain why.",
        ],
    },
}

# -- Real, distinguishable filler "code" content (NOT sequential-token-id
# synthetic input, NOT the single-sentence "capital of France" template)
# -- several distinct, real Python functions/classes concatenated. Each
# request takes a random offset+length slice of this corpus (see
# ``_build_coding_prompt``), so different concurrent requests genuinely
# show different code content, not the same repeated block. --
_CODE_UNITS = [
    '''def bubble_sort(items):
    """Sort a list in place using bubble sort; returns the same list."""
    n = len(items)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
                swapped = True
        if not swapped:
            break
    return items
''',
    '''def is_palindrome(text):
    """Return True iff `text`, ignoring case and non-alphanumerics, reads
    the same forwards and backwards."""
    cleaned = [c.lower() for c in text if c.isalnum()]
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
''',
    '''def binary_search(sorted_items, target):
    """Return the index of `target` in `sorted_items`, or -1 if absent."""
    lo, hi = 0, len(sorted_items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_items[mid] == target:
            return mid
        elif sorted_items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    '''class LRUCache:
    """A minimal fixed-capacity least-recently-used cache."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._data = {}
        self._order = []

    def get(self, key):
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._order.remove(key)
        elif len(self._data) >= self.capacity:
            oldest = self._order.pop(0)
            del self._data[oldest]
        self._data[key] = value
        self._order.append(key)
''',
    '''def merge_sorted_lists(a, b):
    """Merge two already-sorted lists into one sorted list."""
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    merged.extend(a[i:])
    merged.extend(b[j:])
    return merged
''',
    '''def fibonacci_memo(n, cache=None):
    """Return the n-th Fibonacci number using memoized recursion."""
    if cache is None:
        cache = {}
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    result = fibonacci_memo(n - 1, cache) + fibonacci_memo(n - 2, cache)
    cache[n] = result
    return result
''',
    '''def graph_bfs(adjacency, start):
    """Breadth-first traversal order of a graph given as an adjacency dict."""
    visited = {start}
    order = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
''',
    '''class RateLimiter:
    """A simple token-bucket rate limiter."""

    def __init__(self, rate_per_sec, burst):
        self.rate = rate_per_sec
        self.burst = burst
        self.tokens = burst
        self.last_refill = 0.0

    def allow(self, now):
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
''',
    '''def matrix_multiply(a, b):
    """Multiply two 2D matrices represented as lists of lists."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("incompatible shapes")
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            if a[i][k] == 0:
                continue
            for j in range(cols_b):
                result[i][j] += a[i][k] * b[k][j]
    return result
''',
    '''def quicksort(items):
    """Return a new sorted list using the Lomuto quicksort scheme."""
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    middle = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]
    return quicksort(left) + middle + quicksort(right)
''',
]
_FILLER_CORPUS_TEXT = ("\n".join(_CODE_UNITS) + "\n") * 8


def _build_filler_ids(tok) -> list[int]:
    return tok.encode(_FILLER_CORPUS_TEXT, add_special_tokens=False)


def _pick_class(rng: random.Random) -> str:
    names = list(REQUEST_CLASSES.keys())
    weights = [REQUEST_CLASSES[n]["weight"] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _build_coding_prompt(tok, filler_ids: list[int], rng: random.Random, req_id: int, cls_name: str):
    """Real, distinguishable coding-agent-shaped prompt: a unique marker
    comment + a random slice of real code + a class-appropriate question.
    Returns (prompt_token_ids, max_tokens, cls_name)."""
    cls = REQUEST_CLASSES[cls_name]
    target_len = rng.randint(*cls["prompt_len_range"])
    max_tokens = rng.randint(*cls["max_tokens_range"])
    question = rng.choice(cls["questions"])
    marker_text = f"# request-marker: RQ{req_id:06d}\n"
    marker_ids = tok.encode(marker_text, add_special_tokens=False)
    question_ids = tok.encode("\n\n" + question, add_special_tokens=False)
    filler_needed = max(1, target_len - len(marker_ids) - len(question_ids))
    max_start = max(0, len(filler_ids) - filler_needed)
    start = rng.randint(0, max_start)
    filler_slice = filler_ids[start : start + filler_needed]
    if len(filler_slice) < filler_needed:
        filler_slice = filler_slice + filler_ids[: filler_needed - len(filler_slice)]
    prompt_ids = marker_ids + filler_slice + question_ids
    return prompt_ids, max_tokens, cls_name


def _build_request_pool(tok, pool_size: int, seed: int, mean_arrival_gap_s: float):
    """Pre-build ``pool_size`` real, varied coding-agent-shaped requests
    plus a wall-clock arrival schedule (cumulative random gaps -- a real
    Poisson-ish staggered arrival stream, decoupled from round cadence so
    a real-duration test doesn't need to guess how many rounds fit in an
    hour up front). CPU-only, cheap (one tokenization of the filler corpus,
    then pure slicing per request)."""
    rng = random.Random(seed)
    filler_ids = _build_filler_ids(tok)
    pool = []
    arrival_t = 0.0
    for req_id in range(pool_size):
        arrival_t += rng.uniform(0.5, mean_arrival_gap_s * 2 - 0.5)
        cls_name = _pick_class(rng)
        prompt_ids, max_tokens, cls_name = _build_coding_prompt(tok, filler_ids, rng, req_id, cls_name)
        pool.append(
            {
                "req_id": req_id,
                "arrival_t": arrival_t,
                "cls": cls_name,
                "prompt_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
                "max_tokens": max_tokens,
            }
        )
    return pool


def _run_sustained(runner, tok, pool: list[dict], capacity: int, duration_s: float,
                    progress_interval_s: float, nvidia_smi_interval_s: float):
    import torch

    from benchmarks.mtp_async_arrival_check import (
        MAX_BENIGN_STREAK_ROUNDS,
        _looks_like_repetition_artifact,
        _near_tie_margin_diag,
        _ref_check,
        _token_text_is_coherent,
    )
    from benchmarks.mtp_w1s_our_runtime_perf import _gpu_thermal

    ref_slot_for = {p: capacity + p for p in range(capacity)}
    diag_slot_for = {p: 2 * capacity + p for p in range(capacity)}

    free_slots = list(range(capacity))
    active: dict[int, dict] = {}
    waiting: list[dict] = []  # eligible (arrival_t passed) but no free slot yet
    pool_iter = iter(pool)
    next_req = next(pool_iter, None)

    finished: list[dict] = []
    events: list[dict] = []
    round_log: list[dict] = []
    memory_trace: list[dict] = []
    correctness_ok = True
    correctness_failures: list[dict] = []

    t_start = time.perf_counter()
    last_progress_t = t_start
    last_nvidia_smi_t = t_start
    round_idx = 0
    total_gpu_wall_s = 0.0
    admission_closed = False
    admitted_count = 0
    max_rounds = 20_000_000  # generous structural safety cap only

    def _now() -> float:
        return time.perf_counter() - t_start

    def _sample_memory(tag: str) -> None:
        torch.cuda.synchronize()
        entry = {
            "tag": tag,
            "elapsed_s": _now(),
            "round": round_idx,
            "num_active": len(active),
            "num_finished": len(finished),
            "cuda_allocated_mib": torch.cuda.memory_allocated() / (1024 * 1024),
            "cuda_reserved_mib": torch.cuda.memory_reserved() / (1024 * 1024),
        }
        memory_trace.append(entry)

    _sample_memory("start")
    print(json.dumps({"heartbeat": "run_start", "num_pool_requests": len(pool), "capacity": capacity}), flush=True)

    while True:
        elapsed = _now()
        if elapsed >= duration_s:
            admission_closed = True

        # -- Arrival: pull every pool request whose arrival_t has passed
        # into the waiting queue (real admission-control queue if no slot
        # is free yet -- identical in spirit to mtp_async_arrival_check.py). --
        if not admission_closed:
            while next_req is not None and next_req["arrival_t"] <= elapsed:
                waiting.append(next_req)
                next_req = next(pool_iter, None)

        # -- Admission: admit min(free, waiting) together in ONE ragged
        # mtp_prefill_batch call -- naturally exercises multi-request
        # ragged admission joining an in-flight batch, many times over,
        # WITH cudagraph enabled on the production verify/decode path
        # (this driver's core new coverage vs. every prior async-arrival
        # test, which ran enable_cudagraph=False). --
        if free_slots and waiting:
            n = min(len(free_slots), len(waiting))
            admit_now = [(free_slots.pop(0), waiting.pop(0)) for _ in range(n)]
            new_slots = [s for s, _ in admit_now]
            new_prompts = [r["prompt_ids"] for _, r in admit_now]
            for slot, _ in admit_now:
                if runner.slot_kv_len[slot] != 0:
                    runner.reset_slot(slot)
            # P3.3a: unified production prefill entrypoint (delegates to
            # mtp_prefill_batch when the persistent cache flag is off => P2).
            prefill_result = runner.mtp_prefill_with_cache(new_slots, new_prompts)
            for slot, req in admit_now:
                anchor = prefill_result[slot]["anchor"]
                drafts = prefill_result[slot]["draft_tokens"]
                ref_slot = ref_slot_for[slot]
                if runner.slot_kv_len[ref_slot] != 0:
                    runner.reset_slot(ref_slot)
                ref_first = runner.prefill(ref_slot, req["prompt_ids"])
                if ref_first != anchor:
                    diag_slot = diag_slot_for[slot]
                    runner.reset_slot(diag_slot)
                    diag = _near_tie_margin_diag(runner, diag_slot, req["prompt_ids"], anchor)
                    if not diag["within_tolerance"]:
                        correctness_ok = False
                        correctness_failures.append(
                            {"req_id": req["req_id"], "stage": "admission_bootstrap", "diag": diag}
                        )
                active[slot] = {
                    "req_id": req["req_id"],
                    "cls": req["cls"],
                    "prompt_len": req["prompt_len"],
                    "max_tokens": req["max_tokens"],
                    "anchor": anchor,
                    "drafts": drafts,
                    "committed_len": 0,
                    "committed_tokens": [],
                    "admitted_round": round_idx,
                    "admitted_t": elapsed,
                }
                admitted_count += 1
                events.append(
                    {
                        "event": "admit", "req_id": req["req_id"], "slot": slot, "round": round_idx,
                        "cls": req["cls"], "prompt_len": req["prompt_len"],
                        "batch_size_this_admission": len(admit_now),
                    }
                )

        if not active:
            if admission_closed and not waiting and next_req is None:
                break
            # Idle wait for the next arrival/admission -- deliberately does
            # NOT increment round_idx (round_idx counts real verify/commit
            # rounds only) and sleeps briefly instead of busy-spinning: an
            # earlier version of this loop incremented round_idx here with
            # no sleep, burning through the 20,000,000-iteration safety cap
            # in under 2 real seconds (a tight Python loop does ~10M
            # iterations/sec) BEFORE the first pool arrival's arrival_t
            # (~4s) was ever reached -- found by a short smoke test
            # (--duration-s 60) that returned 0 admissions / FAIL almost
            # instantly. A true stall (should not happen given the arrival
            # schedule) is still caught by the wall-clock-based safety
            # valve below, not by a round counter an idle loop can trivially
            # exhaust.
            if elapsed > duration_s + 1800:
                print(
                    json.dumps({"heartbeat": "stuck_safety_exit", "elapsed_s": elapsed, "round": round_idx}),
                    flush=True,
                )
                break
            time.sleep(0.02)
            continue

        active_slots = list(active.keys())
        t0 = time.perf_counter()
        decisions = runner.mtp_verify_and_commit_batch(
            active_slots,
            {s: active[s]["anchor"] for s in active_slots},
            {s: active[s]["drafts"] for s in active_slots},
        )
        torch.cuda.synchronize()
        total_gpu_wall_s += time.perf_counter() - t0

        newly_finished = []
        for s in active_slots:
            st = active[s]
            decision = decisions[s]
            real_new_tokens = [st["anchor"]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slot_for[s], real_new_tokens, decision["next_anchor"], tok=tok)
            extended_context = None
            if not ref_report["content_ok_within_near_tie_tolerance"]:
                extended_context = tok.decode((st["committed_tokens"] + real_new_tokens)[-40:])
            round_log.append(
                {
                    "round": round_idx,
                    "req_id": st["req_id"],
                    "batch_composition": list(active_slots),
                    "ref_report": ref_report,
                    "extended_context": extended_context,
                }
            )
            st["committed_tokens"].extend(decision["committed"])
            st["committed_len"] += decision["num_accepted"] + 1
            st["anchor"], st["drafts"] = decision["next_anchor"], decision["next_draft_tokens"]

            if st["committed_len"] >= st["max_tokens"]:
                st["finished_round"] = round_idx
                st["finished_t"] = elapsed
                finished.append(
                    {
                        "req_id": st["req_id"], "slot": s, "cls": st["cls"],
                        "prompt_len": st["prompt_len"], "committed_len": st["committed_len"],
                        "admitted_round": st["admitted_round"], "finished_round": round_idx,
                        "admitted_t": st["admitted_t"], "finished_t": st["finished_t"],
                        "decoded_completion": tok.decode(st["committed_tokens"]),
                    }
                )
                events.append({"event": "finish", "req_id": st["req_id"], "slot": s, "round": round_idx})
                newly_finished.append(s)

        for s in newly_finished:
            del active[s]
            runner.reset_slot(s)
            free_slots.append(s)

        round_idx += 1

        now_t = time.perf_counter()
        if now_t - last_progress_t >= progress_interval_s:
            last_progress_t = now_t
            committed_so_far = sum(f["committed_len"] for f in finished)
            wall_so_far = _now()
            print(
                json.dumps(
                    {
                        "heartbeat": "progress",
                        "elapsed_s": round(wall_so_far, 1),
                        "round": round_idx,
                        "admitted": admitted_count,
                        "finished": len(finished),
                        "active": len(active),
                        "waiting": len(waiting),
                        "total_committed_tokens": committed_so_far,
                        "accepted_tok_s_so_far": round(committed_so_far / wall_so_far, 2) if wall_so_far > 0 else None,
                        "correctness_ok_so_far": correctness_ok,
                        "cuda_allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
                        "cuda_reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 1),
                    }
                ),
                flush=True,
            )
            _sample_memory("progress")
        if now_t - last_nvidia_smi_t >= nvidia_smi_interval_s:
            last_nvidia_smi_t = now_t
            try:
                thermal = _gpu_thermal()
            except Exception as exc:  # pragma: no cover -- diagnostic only
                thermal = {"error": str(exc)}
            print(json.dumps({"heartbeat": "nvidia_smi", "elapsed_s": round(_now(), 1), **thermal}), flush=True)

        if round_idx >= max_rounds:
            break

    _sample_memory("end")
    wall_s_e2e = _now()
    total_committed_tokens = sum(f["committed_len"] for f in finished)

    # -- Post-hoc streak-based benign-near-tie reclassification -- SAME
    # mechanical criteria this project already established and validated
    # (imported, not re-implemented) in mtp_async_arrival_check.py section
    # 22: resolves within MAX_BENIGN_STREAK_ROUNDS, every round in the
    # streak matches the documented repetition-artifact pattern, every
    # diverging token pair is coherent text. Any streak failing even one
    # criterion stays a hard failure -- NEAR_TIE_LOGIT_MARGIN itself is
    # never loosened. --
    benign_near_tie_events: list[dict] = []
    by_req: dict[int, list[dict]] = {}
    for row in round_log:
        by_req.setdefault(row["req_id"], []).append(row)
    for req_id, rows in by_req.items():
        rows.sort(key=lambda r: r["round"])
        i = 0
        while i < len(rows):
            if rows[i]["ref_report"]["content_ok_within_near_tie_tolerance"]:
                i += 1
                continue
            streak = [rows[i]]
            j = i + 1
            while j < len(rows) and not rows[j]["ref_report"]["content_ok_within_near_tie_tolerance"]:
                streak.append(rows[j])
                j += 1
            resolves = j < len(rows)
            streak_len_ok = len(streak) <= MAX_BENIGN_STREAK_ROUNDS
            pattern_ok = all(
                r["extended_context"] is not None and _looks_like_repetition_artifact(r["extended_context"])
                for r in streak
            )
            tokens_ok = all(
                _token_text_is_coherent(r["ref_report"].get("ref_predicted_token_text", ""))
                and _token_text_is_coherent(r["ref_report"].get("mtp_next_anchor_token_text", ""))
                for r in streak
            )
            if resolves and streak_len_ok and pattern_ok and tokens_ok:
                benign_near_tie_events.append(
                    {
                        "req_id": req_id,
                        "rounds": [r["round"] for r in streak],
                        "margins": [r["ref_report"]["near_tie_margin"] for r in streak],
                        "heals_at_round": rows[j]["round"],
                    }
                )
            else:
                for r in streak:
                    correctness_ok = False
                    correctness_failures.append(
                        {
                            "req_id": req_id,
                            "stage": f"round {r['round']}",
                            "batch_composition": r["batch_composition"],
                            "ref_report": r["ref_report"],
                            "reclassification_checks": {
                                "resolves_within_next_round": resolves,
                                "streak_len_ok": streak_len_ok,
                                "matches_documented_repetition_artifact": pattern_ok,
                                "tokens_coherent": tokens_ok,
                            },
                        }
                    )
            i = j

    # -- Secondary, informational-only identity/leak signal: no finished
    # request's decoded completion should contain another request's unique
    # marker string. This is a soft signal (a real code-explanation
    # completion has no particular reason to echo ANY marker at all) --
    # the hard correctness gate is the independent-reference check above,
    # exactly as this project's established convention treats similar
    # secondary signals (e.g. the "capital of France" city check in
    # mtp_async_arrival_check.py). --
    marker_leak_events = []
    all_markers = {f["req_id"]: f"RQ{f['req_id']:06d}" for f in finished}
    for f in finished:
        for other_id, other_marker in all_markers.items():
            if other_id != f["req_id"] and other_marker in f["decoded_completion"]:
                marker_leak_events.append({"req_id": f["req_id"], "leaked_marker_of_req": other_id})

    return {
        "passed": bool(correctness_ok and round_idx < max_rounds and not active),
        "correctness_ok": correctness_ok,
        "correctness_failures": correctness_failures,
        "benign_near_tie_events_informational": benign_near_tie_events,
        "marker_leak_events_informational": marker_leak_events,
        "capacity": capacity,
        "num_pool_requests_available": len(pool),
        "num_admitted": admitted_count,
        "num_finished": len(finished),
        "rounds_used": round_idx,
        "wall_s_e2e": wall_s_e2e,
        "total_committed_tokens": total_committed_tokens,
        "accepted_tokens_per_sec": total_committed_tokens / wall_s_e2e if wall_s_e2e > 0 else float("nan"),
        "production_path_gpu_wall_s_summed": total_gpu_wall_s,
        "production_path_accepted_tokens_per_sec": (
            total_committed_tokens / total_gpu_wall_s if total_gpu_wall_s > 0 else float("nan")
        ),
        "memory_trace": memory_trace,
        "events": events,
        "finished": finished,
        "class_mix_admitted": {
            cls: sum(1 for e in events if e["event"] == "admit" and e["cls"] == cls)
            for cls in REQUEST_CLASSES
        },
    }


def _run_once(duration_s: float, capacity: int, num_slots: int, pool_size: int, seed: int,
              progress_interval_s: float, nvidia_smi_interval_s: float, blocks_per_slot: int) -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    tok = AutoTokenizer.from_pretrained(MODEL)

    mean_gap = 3600.0 / max(1, pool_size) * capacity * 0.5  # generous coverage margin, see module docstring
    mean_gap = max(1.0, min(4.0, mean_gap))
    pool = _build_request_pool(tok, pool_size=pool_size, seed=seed, mean_arrival_gap_s=mean_gap)
    max_prompt_len = max(r["prompt_len"] for r in pool)
    max_needed = max_prompt_len + max(r["max_tokens"] for r in pool) + 64

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(8192, max_needed),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(
        vllm_config,
        num_slots=num_slots,
        block_size=64,
        blocks_per_slot=blocks_per_slot,
        enable_cudagraph=True,
    )
    result = _run_sustained(
        runner, tok, pool, capacity=capacity, duration_s=duration_s,
        progress_interval_s=progress_interval_s, nvidia_smi_interval_s=nvidia_smi_interval_s,
    )
    result["config"] = {
        "duration_s_target": duration_s,
        "capacity": capacity,
        "num_slots": num_slots,
        "blocks_per_slot": blocks_per_slot,
        "pool_size": pool_size,
        "seed": seed,
        "mean_arrival_gap_s": mean_gap,
        "enable_cudagraph": True,
        "kv_cache_dtype": "fp8_e4m3",
        "K": K,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--blocks-per-slot", type=int, default=512)
    parser.add_argument("--pool-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    parser.add_argument("--nvidia-smi-interval-s", type=float, default=300.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result = _run_once(
        duration_s=args.duration_s,
        capacity=args.capacity,
        num_slots=args.num_slots,
        pool_size=args.pool_size,
        seed=args.seed,
        progress_interval_s=args.progress_interval_s,
        nvidia_smi_interval_s=args.nvidia_smi_interval_s,
        blocks_per_slot=args.blocks_per_slot,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
    summary = {k: v for k, v in result.items() if k not in ("events", "finished", "memory_trace")}
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n=== {'PASS' if result['passed'] else 'FAIL'} ===")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
