"""CUDA Graph capture/replay correctness test for MTP verify (qo_len>1),
reusing ``CapturedBatchDecodeGraph`` generalized from the qo_len=1 batch
decode round (see notes/direct-model-runner-design.md's "CUDA Graph
capture/replay, step 1" and its MTP-extension follow-up).

Same discipline as the qo_len=1 test
(``cudagraph_decode_regression.py``): capture at a small (~15-token)
shape, then replay at kv_len distributions FAR more extreme than that --
including a slot pushed to within a few pages of this test's OWN
configured per-slot page-table limit (2048 tokens here, a small value
chosen for this correctness test's speed, NOT a GPU hardware limit) --
checked via signal-probe (unique numeric marker per slot, verified
recoverable with zero cross-slot leakage), not bytewise comparison.

State-neutral capture (2026-07-17, a correctness fix now built into
``CapturedBatchDecodeGraph`` itself, found while designing this test --
see notes/direct-model-runner-design.md for the full writeup):
``capture()``'s warmup (3 REAL executions on a side stream before the
graph trace -- the trace itself, inside ``with torch.cuda.graph(g):``,
does NOT execute anything, only records; confirmed against the sibling
project's own kernel-level CUDA-graph test) is not idempotent for GDN's
recurrent/chunked state update under repeated identical input (unlike
attention's KV cache, where writing the same value at the same position
repeatedly is harmless). ``capture()`` now reserves its OWN dedicated,
disposable warmup slots internally (never any slot a caller passes to
``replay()``) -- it takes no external slot/token/kv_length arguments at
all. This test still uses a SEPARATE ``ref_slots`` group of its own, for
a different, legitimate purpose: establishing trusted draft tokens fed
into the real ``replay()`` checks below (not capture warmup).

Usage:
    python -m benchmarks.cudagraph_mtp_regression
    python -m benchmarks.cudagraph_mtp_regression --repeat 3
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


def _make_prompt(number: int, filler_tokens_target: int = 0) -> str:
    if filler_tokens_target <= 0:
        return f"The value of X is {number}. The value of X is"
    filler = "The weather today is mild and pleasant. " * max(1, filler_tokens_target // 8)
    return f"{filler}The value of X is {number}. The value of X is"


def _establish_draft(runner, slot: int, first_token: int, kv_len: int, qo_len: int) -> list[int]:
    """Extend a REAL greedy continuation on ``slot`` via the
    already-verified qo_len=1 path, returning a coherent ``qo_len``-token
    draft. This genuinely advances ``slot``'s own bookkeeping/GDN state --
    callers must treat ``slot`` as spent for this purpose only, never
    reuse it as one of the slots checked via the captured graph."""
    tokens = [first_token]
    cur = first_token
    cur_kv = kv_len
    for _ in range(qo_len - 1):
        logits = runner._forward_batch([slot], [cur], [cur_kv])
        cur = int(logits[0].argmax(dim=-1).item())
        cur_kv += 1
        tokens.append(cur)
    return tokens[:qo_len]


def _signal_check(text: str, own_number: int, other_numbers: list[int], qo_len: int) -> dict:
    own = str(own_number)[:qo_len]
    contains_own = own in text
    leaked_other = any(str(n)[:qo_len] in text for n in other_numbers if n != own_number)
    return {"contains_own": contains_own, "leaked_other": leaked_other, "text": text}


def _run_once() -> dict:

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import (
        CapturedBatchDecodeGraph,
        DirectModelRunner,
        build_vllm_config,
    )

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    block_size, blocks_per_slot = 16, 128
    capacity = block_size * blocks_per_slot  # this test's configured per-slot page-table limit (2048 tokens)
    batch = 4
    qo_len = 4  # K=3 draft + 1 bonus token, the real production MTP shape

    # Dedicated slot groups, kept strictly separate by purpose:
    #   ref_slots    -- establish trusted draft tokens for the real replay()
    #                   checks below (spent afterward, and reset+reused for
    #                   the second, independent extreme-shape check).
    #   graph_slots  -- the ACTUAL slots checked via the captured graph;
    #                   touched by nothing except their own prefill until
    #                   the real replay() calls below.
    # A THIRD, separate batch of slots is reserved internally by
    # CapturedBatchDecodeGraph itself for capture()'s own disposable
    # warmup (2026-07-17 state-neutral-capture fix -- capture() no longer
    # takes external slot/token/kv_length args at all), hence 3*batch.
    ref_slots = list(range(batch))
    graph_slots = list(range(batch, 2 * batch))
    runner = DirectModelRunner(
        vllm_config, num_slots=3 * batch, block_size=block_size, blocks_per_slot=blocks_per_slot
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    numbers = list(NUMBERS[:batch])
    numbers[-1] = numbers[0]  # duplicate pair for a same-batch self-consistency check
    prompt_ids = [tok.encode(_make_prompt(n), add_special_tokens=False) for n in numbers]

    # --- ref_slots: prefill, then establish a real qo_len-token draft via
    # the already-verified qo_len=1 path. ---
    ref_next = [runner.prefill(s, ids) for s, ids in zip(ref_slots, prompt_ids)]
    ref_kv_before = [runner.slot_kv_len[s] for s in ref_slots]
    ref_draft = [
        _establish_draft(runner, s, t, kv, qo_len) for s, t, kv in zip(ref_slots, ref_next, ref_kv_before)
    ]

    # --- graph_slots: prefill with the IDENTICAL prompts -- pristine
    # state, matching ref_slots' kv_len from right after their own
    # prefill (before the establishing loop above touched them). ---
    graph_next = [runner.prefill(s, ids) for s, ids in zip(graph_slots, prompt_ids)]
    if graph_next != ref_next:
        return {"passed": False, "error": "prefill greedy tokens diverged between ref_slots and graph_slots"}
    graph_kv = [runner.slot_kv_len[s] for s in graph_slots]

    # --- Capture: self-contained, uses CapturedBatchDecodeGraph's own
    # internally reserved warmup slots -- never touches ref_slots or
    # graph_slots (2026-07-17 state-neutral-capture fix). ---
    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=qo_len)
    graph.capture()
    print("CAPTURE_OK")

    steps_log: list[dict] = []

    def _do_replay(label: str, first_tokens, draft, kv_lengths, check_self_consistency: bool) -> bool:
        logits = graph.replay(graph_slots, draft, kv_lengths)
        predicted = [
            [int(logits[i * qo_len + p].argmax(dim=-1).item()) for p in range(qo_len)] for i in range(batch)
        ]
        texts = [tok.decode([first_tokens[i]] + predicted[i]) for i in range(batch)]
        crosstalk = [
            {"slot": i, "number": numbers[i], **_signal_check(texts[i], numbers[i], numbers, qo_len)}
            for i in range(batch)
        ]
        signal_ok = all(c["contains_own"] and not c["leaked_other"] for c in crosstalk)
        entry = {"label": label, "crosstalk": crosstalk}
        ok = signal_ok
        if check_self_consistency:
            self_consistent = texts[0] == texts[-1]
            entry["self_consistent"] = self_consistent
            ok = ok and self_consistent
        steps_log.append(entry)
        return ok

    # --- Replay 1: graph_slots, at their pristine (capture-time-matching)
    # shape, using the SAME trusted draft established via ref_slots. This
    # is the FIRST time graph_slots' state is touched by anything beyond
    # their own prefill. ---
    small_shape_ok = _do_replay("replay@capture-time-shape", graph_next, ref_draft, graph_kv, True)

    # --- Replay 2: EXTREME -- an INDEPENDENT single-shot verify (not a
    # continuation of replay 1's own output -- reusing predicted tokens
    # as a new draft would feed a token sequence that doesn't actually
    # continue the identity number, confounding the crosstalk signal with
    # an unrelated content mismatch). Reset ALL graph_slots and re-prefill
    # them fresh: 3 short (slots 0-2) + 1 near this runtime's hard
    # per-slot capacity (slot 3). Establish each slot's OWN fresh,
    # correctly-positioned trusted draft via ref_slots (reset + re-used
    # for this purpose too), matching its NEW kv_len exactly, before the
    # ONE decisive replay call. ---
    extreme_idx = batch - 1
    filler_targets = [0] * batch
    filler_targets[extreme_idx] = capacity - 100
    extreme_prompt_ids = []
    for i in range(batch):
        p = _make_prompt(numbers[i], filler_tokens_target=filler_targets[i])
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) >= capacity:
            ids = ids[: capacity - 20]
        extreme_prompt_ids.append(ids)

    for s in ref_slots:
        runner.reset_slot(s)
    for s in graph_slots:
        runner.reset_slot(s)

    ref_next2 = [runner.prefill(s, ids) for s, ids in zip(ref_slots, extreme_prompt_ids)]
    ref_kv2 = [runner.slot_kv_len[s] for s in ref_slots]
    ref_draft2 = [
        _establish_draft(runner, s, t, kv, qo_len) for s, t, kv in zip(ref_slots, ref_next2, ref_kv2)
    ]

    graph_next2 = [runner.prefill(s, ids) for s, ids in zip(graph_slots, extreme_prompt_ids)]
    if graph_next2 != ref_next2:
        return {"passed": False, "error": "prefill greedy tokens diverged between ref_slots and graph_slots (extreme case)"}
    graph_kv2 = [runner.slot_kv_len[s] for s in graph_slots]
    steps_log.append({"label": "extreme_reprefill", "kv_lengths": graph_kv2, "capacity": capacity})

    extreme_ok = _do_replay("replay@extreme-mixed-kv_len(MTP)", graph_next2, ref_draft2, graph_kv2, False)

    return {"passed": small_shape_ok and extreme_ok, "capacity": capacity, "qo_len": qo_len, "steps": steps_log}


def _run_subprocess() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.cudagraph_mtp_regression", "--single-run-json"],
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
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once()
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat == 1:
        result = _run_once()
        print(result)
        return 0 if result["passed"] else 1

    results = [_run_subprocess() for _ in range(args.repeat)]
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
