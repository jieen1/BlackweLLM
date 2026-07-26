"""Empirical capacity/memory check before expanding slot capacity to real
W1(4096in/1024out)/W2(32768in/1024out) scale. Rather than trust a hand
derivation, measures real GPU memory via nvidia-smi at each stage:
(a) after loading target+draft model weights, (b) after allocating the
expanded per-slot KV/GDN caches, (c) after a real, single-shot (this
runtime does not chunk prefill) W2-sized prefill on one slot -- the
biggest open question is peak ACTIVATION memory during a monolithic
32768-token forward pass through 64 layers, not the persistent KV cache
(attention KV is ~34KB/token/layer-set here, i.e. ~1.2GB/slot even at
W2 size -- GDN state is fixed-size regardless of context length, per
qwen_gdn_linear_attn.py's get_state_shape(), which depends only on
model config, not sequence length).

Usage:
    python -m benchmarks.capacity_w1w2_check
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"

# W1 = 4096in/1024out = 5120 total; W2 = 32768in/1024out = 33792 total.
# block_size=64 kept unchanged (matches every existing test); blocks_per_slot
# bumped so 16 * blocks_per_slot covers W2 with margin.
BLOCK_SIZE = 16
BLOCKS_PER_SLOT = 2560  # 16 * 2560 = 40960 tokens/slot capacity
NUM_SLOTS = 8  # matches the 4-slot-isolation test's need (4 mtp + 4 ref)

W2_INPUT_LEN = 32768


def _gpu_mem_mib() -> float:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out.splitlines()[0])


def _run_once() -> dict:
    mem_before_load = _gpu_mem_mib()

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=40960,
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(vllm_config, num_slots=NUM_SLOTS, block_size=BLOCK_SIZE, blocks_per_slot=BLOCKS_PER_SLOT)
    mem_after_alloc = _gpu_mem_mib()

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Build a real ~32768-token prompt by repeating real text (tokenizer
    # applied to real English sentences, not random ids) up to the target
    # length -- content doesn't matter for a memory/capacity check, only
    # real shape/length does.
    base_sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank while "
        "the autumn leaves fall gently onto the old stone bridge. "
    )
    ids = tok.encode(base_sentence, add_special_tokens=False)
    prompt_ids = (ids * (W2_INPUT_LEN // len(ids) + 1))[:W2_INPUT_LEN]

    mem_before_prefill = _gpu_mem_mib()
    anchor = runner.prefill(0, prompt_ids)
    mem_after_prefill = _gpu_mem_mib()

    # One real decode step too, to confirm the slot is usable afterward
    # (not just "didn't crash on prefill").
    next_tok = runner.decode(0, anchor)
    mem_after_decode = _gpu_mem_mib()

    # Sequential (NOT batched) W2-sized prefills on 3 MORE slots -- the
    # real open question for "does concurrency=4 fit": does memory
    # PLATEAU (PyTorch's caching allocator reuses freed activation memory
    # across successive prefill calls) or keep GROWING per additional
    # slot (which would indicate a real leak and mean concurrency=4 does
    # NOT actually fit despite one slot fitting)? mtp_prefill/prefill
    # process ONE slot per call (never batched across slots in this
    # runtime's design), so each prior slot's transient activation memory
    # should be freed before the next slot's prefill call begins.
    sequential_mem_mib = [mem_after_prefill]
    for slot in (1, 2, 3):
        runner.prefill(slot, prompt_ids)
        sequential_mem_mib.append(_gpu_mem_mib())

    return {
        "block_size": BLOCK_SIZE,
        "blocks_per_slot": BLOCKS_PER_SLOT,
        "capacity_per_slot_tokens": BLOCK_SIZE * BLOCKS_PER_SLOT,
        "num_slots": NUM_SLOTS,
        "w2_input_len_tested": W2_INPUT_LEN,
        "mem_before_load_mib": mem_before_load,
        "mem_after_model_and_cache_alloc_mib": mem_after_alloc,
        "mem_before_prefill_mib": mem_before_prefill,
        "mem_after_prefill_mib": mem_after_prefill,
        "mem_after_decode_mib": mem_after_decode,
        "weights_and_cache_delta_mib": mem_after_alloc - mem_before_load,
        "prefill_activation_delta_mib": mem_after_prefill - mem_before_prefill,
        "sequential_prefill_mem_mib_by_slot_0_1_2_3": sequential_mem_mib,
        "gpu_total_mib": 97887.0,
        "anchor_token": anchor,
        "next_token": next_tok,
        "passed": True,
    }


def main() -> int:
    import json

    try:
        result = _run_once()
    except Exception as e:  # noqa: BLE001 -- this check's whole point is to surface OOM/capacity errors
        import traceback

        result = {"passed": False, "error": str(e), "traceback": traceback.format_exc()}
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
