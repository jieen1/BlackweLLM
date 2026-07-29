"""CUDA Graph capture/replay correctness test for
``oracle.qwen36_vllm.direct_model_runner.CapturedBatchDecodeGraph`` (qo_len=1, fixed
batch_size=4 -- step 1 of the 2026-07-16 CUDA Graph round, MTP capture is
a deliberately separate follow-on step per the coordinator's explicit
staging).

This project's own read of the sibling ``sm120-flash-attention`` project's
documented CUDA Graph history flagged two concrete failure modes to guard
against, not just "does it look right once":
  1. Metadata tensors without fixed addresses across capture/replay ->
     illegal memory access (a REAL crash that project hit).
  2. kv_split_size/max_num_splits frozen at capture time from a value that
     doesn't bound ALL real kv_len values the graph will ever replay at ->
     silently wrong results at a kv_len larger than capture-time data.

Per the coordinator's explicit instruction, this test deliberately
captures at a SMALL kv_len (a realistic "just prefilled" shape) and then
replays at kv_len distributions FAR more extreme than that -- including a
slot pushed to within a few pages of this test's OWN configured per-slot
page-table limit (``blocks_per_slot * block_size`` = 2048 tokens here --
a small value chosen for this correctness test's speed, NOT a GPU
hardware limit, and far below the 4K/32K a real W1/W2 workload would
need; a performance-benchmark round would configure this much larger) --
not just the happy path of replaying near the capture-time shape.

Correctness is checked via this project's established signal-probe
methodology (a unique numeric marker per slot, verified recoverable with
zero cross-slot leakage), not bytewise comparison (see
notes/direct-model-runner-design.md's "batch>=2 numerical mismatch"
section for why bytewise isn't the right tool here either).

Usage:
    python -m benchmarks.cudagraph_decode_regression
    python -m benchmarks.cudagraph_decode_regression --repeat 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"

NUMBERS = [84317, 52968, 71053, 39642]
FILLER_SENTENCE = "The weather today is mild and pleasant. "


def _make_prompt(number: int, filler_tokens_target: int = 0, tok=None) -> str:
    if filler_tokens_target <= 0:
        return f"The value of X is {number}. The value of X is"
    # Repeat the filler sentence enough times to exceed the target token
    # count, then let the tokenizer re-tokenize the whole prompt (we only
    # need to be IN THE NEIGHBORHOOD of the target -- exact length doesn't
    # matter for this test, just "near this test's configured per-slot
    # page-table limit").
    repeats = max(1, filler_tokens_target // 8)
    filler = FILLER_SENTENCE * repeats
    return f"{filler}The value of X is {number}. The value of X is"


def _crosstalk_check(text: str, own_number: int, other_numbers: list[int]) -> dict:
    own = str(own_number)
    contains_own = own in text
    leaked_other = any(str(n) in text for n in other_numbers if n != own_number)
    return {"contains_own": contains_own, "leaked_other": leaked_other, "text": text}


def _fragment_pool_via_churn(
    runner,
    target_slot: int,
    scratch_slots: list[int],
    tok,
    future_need: int,
    margin: int = 8,
    cycles: int = 3,
) -> list[int]:
    """P1 (notes/prefix-cache-design.md sec 5) fragmentation recipe:
    deliberately force ``target_slot`` through several allocate/reset/
    reallocate cycles (interleaved with unrelated scratch-slot churn), so
    the NEXT time ``target_slot`` grows its block_table (a single big
    allocation of ``future_need`` blocks, e.g. this test's own upcoming
    near-capacity reprefill), the ids it receives are provably
    non-contiguous -- the real scenario this phase exists to prove the
    block-table + CUDA-graph path tolerates, not just P0's trivial
    always-contiguous case.

    This runner's pool is sized generously (way more headroom than the
    handful of slots this test actually uses at once -- P1's own free
    pool is deliberately larger than P0's static per-slot partition ever
    needed, see ``BlockPool``'s docstring), so a naive small churn alone
    would never be OBSERVED by a single big allocation: it would just
    keep drawing from the still-enormous untouched ascending tail of
    virgin ids, oblivious to whatever small fragmentation happened
    elsewhere in the pool. This function first SPONGES UP (allocates and
    holds -- returned to the caller to free back afterward) everything
    beyond a small ``future_need + margin``-sized working set, so the
    subsequent real allocate/reset/reallocate churn (via actual
    ``DirectModelRunner.prefill``/``reset_slot`` calls, not raw pool
    manipulation) genuinely ROTATES the small set the big allocation will
    imminently draw from -- one pop-then-immediate-reappend-at-tail per
    churn step advances the queue's front by exactly one position, so
    after enough churn steps the big allocation's own pull straddles the
    wrap point and is provably non-contiguous (the low, early-rotated ids
    end up appended after the high, not-yet-rotated ones).

    ``target_slot`` is left FRESH (``reset_slot``ed) when this returns --
    the caller does its own real prefill next, at whatever shape it
    needs. Returns the sponge (a list of held block ids) -- the caller
    must ``runner.block_pool.free(sponge)`` once its own big allocation is
    done, to leave the pool healthy for the rest of the test."""
    pool = runner.block_pool
    working_set_target = future_need + margin
    free_now = pool.num_free_blocks()
    sponge_n = max(0, free_now - working_set_target)
    sponge = pool.allocate(sponge_n) if sponge_n > 0 else []

    tiny_ids = tok.encode("Q", add_special_tokens=False)
    for _ in range(cycles):
        runner.prefill(target_slot, tiny_ids)
        runner.reset_slot(target_slot)
        for s in scratch_slots:
            runner.prefill(s, tiny_ids)
            runner.reset_slot(s)
    return sponge


def _run_once(enable_block_table: bool = False) -> dict:

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import (
        CapturedBatchDecodeGraph,
        DirectModelRunner,
        build_vllm_config,
    )

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    block_size, blocks_per_slot = 16, 128
    capacity = block_size * blocks_per_slot  # this test's configured per-slot page-table limit (2048 tokens)
    batch = 4
    # 2*batch: batch real slots under test + batch permanently reserved for
    # CapturedBatchDecodeGraph's own disposable capture() warmup (see that
    # class's docstring's "state-neutral capture" section -- 2026-07-17
    # correctness fix). P1 (2026-07-19): when enable_block_table, TWO extra
    # scratch slots (num_slots + 0/+1) are reserved purely for this test's
    # own deliberate fragmentation churn -- distinct from both the real
    # `slots` under test and CapturedBatchDecodeGraph's own warmup range
    # (which shifts to stay the LAST `batch` slots either way).
    num_slots = 2 * batch + (2 if enable_block_table else 0)
    runner = DirectModelRunner(
        vllm_config,
        num_slots=num_slots,
        block_size=block_size,
        blocks_per_slot=blocks_per_slot,
        enable_block_table=enable_block_table,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    slots = list(range(batch))
    # Scratch slots sit BETWEEN the real slots [0, batch) and
    # CapturedBatchDecodeGraph's own permanently-reserved warmup range
    # (the LAST `batch` slots of `num_slots` -- see that class's
    # docstring), so they never collide with either.
    scratch_slots = [batch, batch + 1] if enable_block_table else []
    steps_log: list[dict] = []

    def check_all(label: str, extra_generated: dict[int, list[int]] | None = None) -> dict:
        """Signal-probe check against each slot's identity marker, using
        whatever has been generated so far (prefill's first token plus
        anything accumulated in extra_generated)."""
        result = {"label": label, "per_slot": []}
        ok = True
        for i in slots:
            gen = extra_generated.get(i, []) if extra_generated else []
            text = tok.decode(gen) if gen else ""
            c = _crosstalk_check(text, NUMBERS[i], NUMBERS)
            result["per_slot"].append({"slot": i, "number": NUMBERS[i], **c})
            if gen and (c["leaked_other"] or not c["contains_own"]):
                ok = False
        result["ok"] = ok
        steps_log.append(result)
        return result

    # --- Step 1: prefill all 4 slots with SHORT prompts -- deliberately
    # small, this is the capture-time shape. ---
    prompt_ids = [tok.encode(_make_prompt(NUMBERS[i]), add_special_tokens=False) for i in slots]
    next_tokens = [runner.prefill(slot, ids) for slot, ids in zip(slots, prompt_ids)]
    kv_lengths = [runner.slot_kv_len[s] for s in slots]
    generated = {i: [next_tokens[i]] for i in slots}

    # --- Step 2: capture the graph at this small shape. ---
    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=1)
    # capture() is now self-contained (2026-07-17 state-neutral-capture
    # fix): it uses its own permanently reserved warmup slots internally,
    # never touching `slots` (the ones actually checked below).
    graph.capture()

    # --- Step 3: replay AT THE CAPTURE-TIME SHAPE first (sanity). ---
    cur_tokens = list(next_tokens)
    logits = graph.replay(slots, cur_tokens, kv_lengths)
    cur_tokens = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
    kv_lengths = [k + 1 for k in kv_lengths]
    for i, t in enumerate(cur_tokens):
        generated[i].append(t)
    check_all("replay@capture-time-shape", generated)

    # --- Step 4: many sequential replays, normal growth (8 more steps). ---
    for _ in range(8):
        logits = graph.replay(slots, cur_tokens, kv_lengths)
        cur_tokens = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
        kv_lengths = [k + 1 for k in kv_lengths]
        for i, t in enumerate(cur_tokens):
            generated[i].append(t)
    check_all("replay@sequential-growth", generated)

    # --- Step 5: EXTREME case -- push slot 3 to near this slot's hard
    # capacity (2048 tokens), far beyond anything capture-time (~15
    # tokens) or the sequential-growth step (~25 tokens) ever saw, then
    # replay the SAME captured graph across the resulting MIXED, highly
    # heterogeneous kv_len distribution (slot 3 near-max, slots 0-2 still
    # small). ---
    runner.reset_slot(3)
    long_prompt = _make_prompt(NUMBERS[3], filler_tokens_target=capacity - 100)
    long_ids = tok.encode(long_prompt, add_special_tokens=False)
    actual_len = len(long_ids)
    if actual_len >= capacity:
        long_ids = long_ids[: capacity - 20]

    # P1 (notes/prefix-cache-design.md sec 5): when enable_block_table,
    # force slot 3 through several allocate/reset/reallocate cycles
    # (interleaved with scratch-slot churn) BEFORE its real long-prompt
    # prefill below, so the resulting block_table is provably
    # non-contiguous -- exercising exactly the scenario this phase must
    # prove the CUDA-graph replay path tolerates (INV5), not just the
    # trivial always-contiguous case.
    sponge: list[int] = []
    if enable_block_table:
        future_need = (len(long_ids) + block_size - 1) // block_size
        sponge = _fragment_pool_via_churn(
            runner, target_slot=3, scratch_slots=scratch_slots, tok=tok, future_need=future_need
        )
    next_tok_3 = runner.prefill(3, long_ids)
    if enable_block_table:
        runner.block_pool.free(sponge)
    kv_lengths[3] = runner.slot_kv_len[3]
    cur_tokens[3] = next_tok_3
    generated[3] = [next_tok_3]
    steps_log.append({"label": "slot3_reprefilled_long", "kv_len": kv_lengths[3], "capacity": capacity})

    if enable_block_table:
        table3 = runner.block_table[3]
        is_contiguous = all(table3[i + 1] - table3[i] == 1 for i in range(len(table3) - 1))
        steps_log.append(
            {
                "label": "slot3_block_table_fragmentation_proof",
                "block_table": table3,
                "num_blocks": len(table3),
                "is_contiguous": is_contiguous,
            }
        )
        if is_contiguous:
            return {
                "passed": False,
                "capacity": capacity,
                "steps": steps_log,
                "error": (
                    f"slot 3's block_table {table3} is a contiguous run -- the "
                    "deliberate churn recipe failed to produce the non-contiguous "
                    "placement this phase's fragmented-CUDA-graph test must exercise"
                ),
            }

    for _ in range(8):
        logits = graph.replay(slots, cur_tokens, kv_lengths)
        cur_tokens = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
        kv_lengths = [k + 1 for k in kv_lengths]
        for i, t in enumerate(cur_tokens):
            generated[i].append(t)
    extreme_result = check_all("replay@extreme-mixed-kv_len(slot3-near-capacity)", generated)

    # --- Step 6: the OTHER extreme -- a slot at the smallest possible
    # kv_len (freshly re-prefilled with a 1-token prompt) replayed
    # alongside the others' now-much-larger kv_len. ---
    runner.reset_slot(1)
    tiny_ids = tok.encode("X", add_special_tokens=False)
    next_tok_1 = runner.prefill(1, tiny_ids)
    kv_lengths[1] = runner.slot_kv_len[1]
    cur_tokens[1] = next_tok_1
    generated[1] = [next_tok_1]

    logits = graph.replay(slots, cur_tokens, kv_lengths)
    cur_tokens = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
    kv_lengths = [k + 1 for k in kv_lengths]
    for i, t in enumerate(cur_tokens):
        generated[i].append(t)
    # slot 1's prompt ("X") carries no identity marker -- only check the
    # OTHER slots for crosstalk/self-consistency here; slot 1 itself is
    # exercised for "does an extremely small kv_len replay without
    # crashing/corrupting neighbors", not identity recovery.
    tiny_result = check_all("replay@tiny-kv_len-neighbor", {i: generated[i] for i in slots if i != 1})

    passed = extreme_result["ok"] and tiny_result["ok"]
    return {"passed": passed, "capacity": capacity, "steps": steps_log}


def _run_subprocess(enable_block_table: bool = False) -> dict:
    cmd = [sys.executable, "-m", "benchmarks.cudagraph_decode_regression", "--single-run-json"]
    if enable_block_table:
        cmd.append("--enable-block-table")
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_RESULT: ") :])
    return {
        "passed": False,
        "error": "no result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
    # P1 (notes/prefix-cache-design.md sec 5): default False preserves this
    # script's original behavior byte-for-byte. When passed, constructs the
    # runner with enable_block_table=True and forces slot 3 through a
    # deliberate allocate/reset/reallocate churn (see
    # _fragment_pool_via_churn) before its long-prompt reprefill, so the
    # CUDA-graph capture/replay path is exercised against a PROVABLY
    # non-contiguous block_table -- INV5, this phase's own real purpose.
    parser.add_argument("--enable-block-table", action="store_true")
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once(enable_block_table=args.enable_block_table)
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = _run_once(enable_block_table=args.enable_block_table)
        print(result)
        return 0 if result["passed"] else 1

    results = [_run_subprocess(enable_block_table=args.enable_block_table) for _ in range(args.repeat)]
    for i, r in enumerate(results):
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"run {i + 1}/{args.repeat}: {status}")
        if not r.get("passed"):
            print(f"  detail: {r}")

    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n=== {n_pass}/{args.repeat} passed ===")
    return 0 if n_pass == args.repeat else 1


if __name__ == "__main__":
    sys.exit(main())
