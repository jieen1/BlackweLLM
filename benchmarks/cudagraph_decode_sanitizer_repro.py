"""Minimal repro of ``CapturedBatchDecodeGraph`` capture + replay, for
running under ``compute-sanitizer`` specifically. The full ladder in
``benchmarks/cudagraph_decode_regression.py`` (21 total forward-pass-
equivalent calls: 4 prefills + 3 warmup + 1 capture-trace + 13 replays)
was found impractically slow under compute-sanitizer instrumentation
(>60 minutes without completing, vs ~20s uninstrumented) -- this script
cuts that down to the essential minimum needed to exercise both of the
documented CUDA-Graph failure modes this round is guarding against:
  1. Metadata tensors without fixed addresses -> illegal memory access.
  2. kv_split_size/max_num_splits stale at a kv_len larger than capture
     time -> silently wrong results / potential out-of-bounds access.

Sequence: prefill 4 slots (short prompts) -> capture at that small shape
-> ONE replay at the same small shape (sanity) -> re-prefill one slot to
near this runtime's hard per-slot capacity -> ONE replay at that mixed,
extreme kv_len distribution. 6 total forward-pass-equivalent calls
(4 prefills + 3 warmup + 1 capture-trace + 2 replays = 10, still far
fewer than the full regression script's 21).
"""

from __future__ import annotations

import os
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
    batch = 4
    # 2*batch: batch real slots under test + batch permanently reserved for
    # CapturedBatchDecodeGraph's own disposable capture() warmup (2026-07-17
    # state-neutral-capture fix).
    runner = DirectModelRunner(
        vllm_config, num_slots=2 * batch, block_size=block_size, blocks_per_slot=blocks_per_slot
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    slots = list(range(batch))

    prompt_ids = [tok.encode(_make_prompt(NUMBERS[i]), add_special_tokens=False) for i in slots]
    next_tokens = [runner.prefill(slot, ids) for slot, ids in zip(slots, prompt_ids)]
    kv_lengths = [runner.slot_kv_len[s] for s in slots]

    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=1)
    graph.capture()  # self-contained, uses its own reserved warmup slots
    print("CAPTURE_OK")

    logits = graph.replay(slots, next_tokens, kv_lengths)
    pred_small = [int(logits[i].argmax(dim=-1).item()) for i in range(batch)]
    print("REPLAY_SMALL_OK", pred_small)
    kv_lengths = [k + 1 for k in kv_lengths]  # replay() advanced runner.slot_kv_len for every slot

    # Push slot 3 to near this slot's hard capacity in one shot.
    runner.reset_slot(3)
    long_ids = tok.encode(_make_prompt(NUMBERS[3], filler_tokens_target=capacity - 100), add_special_tokens=False)
    if len(long_ids) >= capacity:
        long_ids = long_ids[: capacity - 20]
    next_tok_3 = runner.prefill(3, long_ids)
    kv_lengths = list(kv_lengths)
    kv_lengths[3] = runner.slot_kv_len[3]
    cur_tokens = list(pred_small)
    cur_tokens[3] = next_tok_3
    print(f"SLOT3_KV_LEN={kv_lengths[3]} CAPACITY={capacity}")

    logits2 = graph.replay(slots, cur_tokens, kv_lengths)
    pred_extreme = [int(logits2[i].argmax(dim=-1).item()) for i in range(batch)]
    print("REPLAY_EXTREME_OK", pred_extreme)

    print("SANITIZER_REPRO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
