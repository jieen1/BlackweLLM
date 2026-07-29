"""Smallest practical repro of CapturedBatchDecodeGraph capture + replay,
for running under compute-sanitizer specifically. Even the "minimal"
10-call repro (cudagraph_decode_sanitizer_repro.py, batch_size=4) proved
impractically slow under full memcheck (multi-hour stalls); this cuts
further:

  - batch_size=2 instead of 4 (the persistent-buffer/kv_split_size
    mechanism being checked doesn't depend on batch size -- 4-slot
    correctness is already independently verified via extensive
    uninstrumented testing in cudagraph_decode_regression.py, 3/3 PASS
    including a case near this test's own configured per-slot page-table
    limit, not a GPU hardware limit).
  - Exactly ONE replay, directly at an extreme (near-capacity) kv_len --
    the single most decisive check (capture-time shape vs a drastically
    different replay-time shape) -- instead of a small-shape sanity
    replay followed by an extreme one.

Total forward-pass-equivalent calls: 2 prefills + 3 warmup + 1
capture-trace + 1 replay = 7 (vs the fuller repro's 10, vs the full
regression test's 21).

Run with --tool initcheck (lighter than memcheck, targets uninitialized-
memory/invalid-pointer issues -- the exact class of the sibling project's
documented illegal-memory-access history) and
--kernel-name-exclude kernel_substring=causal_conv1d (this project's own
pre-existing, already-investigated Triton cold-start defect, unrelated to
CUDA graphs, otherwise floods the report budget before ever reaching the
code under test -- see notes/direct-model-runner-design.md's "Known
independent defects").
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
NUMBERS = [84317, 52968]


def _make_prompt(number: int, filler_tokens_target: int = 0) -> str:
    if filler_tokens_target <= 0:
        return f"The value of X is {number}. The value of X is"
    filler = "The weather today is mild and pleasant. " * max(1, filler_tokens_target // 8)
    return f"{filler}The value of X is {number}. The value of X is"


def main() -> int:
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
    capacity = block_size * blocks_per_slot  # this test's configured per-slot page-table limit, not a GPU hardware limit
    batch = 2
    # 2*batch: batch real slots under test + batch permanently reserved for
    # CapturedBatchDecodeGraph's own disposable capture() warmup (2026-07-17
    # state-neutral-capture fix).
    runner = DirectModelRunner(
        vllm_config, num_slots=2 * batch, block_size=block_size, blocks_per_slot=blocks_per_slot
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    slots = list(range(batch))

    # Slot 0: short prompt (small kv_len, this becomes the capture shape).
    # Slot 1: prefilled LONG up front (near capacity) so capture ALSO sees
    # a small kv_len for it, and the one replay jumps it to near-capacity.
    prompt_ids = [tok.encode(_make_prompt(NUMBERS[i]), add_special_tokens=False) for i in slots]
    next_tokens = [runner.prefill(slot, ids) for slot, ids in zip(slots, prompt_ids)]
    kv_lengths = [runner.slot_kv_len[s] for s in slots]

    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=1)
    graph.capture()  # self-contained, uses its own reserved warmup slots
    print("CAPTURE_OK")

    # Push slot 1 to near this test's configured per-slot limit in ONE prefill, then
    # replay the captured graph directly at this extreme, mixed kv_len
    # distribution (slot 0 still small, slot 1 near-max) -- the single
    # most decisive test of address/split-size staleness under replay.
    runner.reset_slot(1)
    long_ids = tok.encode(_make_prompt(NUMBERS[1], filler_tokens_target=capacity - 100), add_special_tokens=False)
    if len(long_ids) >= capacity:
        long_ids = long_ids[: capacity - 20]
    next_tok_1 = runner.prefill(1, long_ids)
    kv_lengths = [runner.slot_kv_len[0], runner.slot_kv_len[1]]
    cur_tokens = [next_tokens[0], next_tok_1]
    print(f"SLOT1_KV_LEN={kv_lengths[1]} CAPACITY={capacity}")

    logits = graph.replay(slots, cur_tokens, kv_lengths)
    pred = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
    print("REPLAY_EXTREME_OK", pred)
    print("SANITIZER_MICRO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
