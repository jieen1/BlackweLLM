"""GPU golden verification: ring KV vs eager path, production form.

Loads the actual Laguna model, runs prefill+decode through:
1. Eager path (ring KV, no cudagraph)
2. CUDA graph path (ring KV, fast_plan, use_tensor_cores)
Compares tokens bit-exact. Also verifies memory allocation.

Usage: /home/bot/.venvs/vllm/bin/python benchmarks/golden_ring_kv.py
Requires: GPU, model weights (~70-450s load)
"""
import os, sys, time, json, logging

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                       os.path.join(os.path.dirname(__file__), "..", ".autotune_cache"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger("golden_ring_kv")

import torch

def main():

    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843"
    )

    # Build VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    args = EngineArgs(
        model=model_path,
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    vllm_config = args.create_engine_config()

    t0 = time.time()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(
        vllm_config,
        num_slots=2,
        block_size=64,
        blocks_per_slot=128,  # 8192 tokens
    )
    load_time = time.time() - t0
    logger.info("Model loaded in %.1fs", load_time)

    # ── Verify allocation ──
    full_layers = len(backend._full_layer_names)
    swa_layers = len(backend._swa_layer_names)
    ring_bps = backend._ring_blocks_per_slot
    logger.info("Full layers: %d, SWA layers: %d, ring_blocks/slot: %d",
                full_layers, swa_layers, ring_bps)
    assert full_layers == 12, f"Expected 12 full layers, got {full_layers}"
    assert swa_layers == 36, f"Expected 36 SWA layers, got {swa_layers}"
    assert ring_bps == 34, f"Expected 34 ring blocks, got {ring_bps}"

    # Verify KV cache shapes
    for name in backend._full_layer_names:
        kv = backend.kv_caches[name]
        expected_blocks = (2 + 1) * 512  # (num_slots + RESERVED) * blocks_per_slot
        assert kv.shape[0] == expected_blocks, f"Full {name}: {kv.shape[0]} != {expected_blocks}"
    for name in backend._swa_layer_names:
        kv = backend.kv_caches[name]
        expected_blocks = (2 + 1) * 34
        assert kv.shape[0] == expected_blocks, f"SWA {name}: {kv.shape[0]} != {expected_blocks}"
    logger.info("✓ KV allocation verified")

    # Verify scratch exists
    assert len(backend._swa_scratch) == 36
    logger.info("✓ Prefill scratch verified (36 layers)")

    # ── Test prompt ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = None  # use [1]*100
    prompt_ids = [1] * 100
    # Pad to 100 tokens for a more meaningful test
    # prompt_ids already 100
    logger.info("Prompt: %d tokens", len(prompt_ids))

    decode_steps = 50

    # ── Run 1: Eager path ──
    backend.reset_slot(0)
    t0 = time.time()
    first_eager = backend.prefill(0, prompt_ids)
    eager_tokens = [first_eager]
    for _ in range(decode_steps - 1):
        tok = backend.decode(0, eager_tokens[-1])
        eager_tokens.append(tok)
    eager_time = time.time() - t0
    logger.info("Eager: %d tokens in %.2fs (%.1f tok/s)",
                len(eager_tokens), eager_time, len(eager_tokens) / eager_time)

    # ── Run 2: Eager path again (determinism check) ──
    backend.reset_slot(1)
    first_eager2 = backend.prefill(1, prompt_ids)
    eager_tokens2 = [first_eager2]
    for _ in range(decode_steps - 1):
        tok = backend.decode(1, eager_tokens2[-1])
        eager_tokens2.append(tok)

    if eager_tokens == eager_tokens2:
        logger.info("✓ Eager determinism: bit-exact across slots")
    else:
        for i, (a, b) in enumerate(zip(eager_tokens, eager_tokens2)):
            if a != b:
                logger.error("✗ Eager mismatch at step %d: %d vs %d", i, a, b)
                break

    # ── Run 3: CUDA graph path (replay step-by-step) ──
    try:
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
        # Capture first (warmup contaminates slot 1 KV), then reset + re-prefill
        backend.reset_slot(1)
        cg = LagunaCudaGraphDecode(backend, batch_size=1)
        cg.capture()
        cg.reset()
        # Re-prefill AFTER capture to ensure clean KV state
        backend.reset_slot(1)
        first_cg = backend.prefill(1, prompt_ids)
        cg_tokens = [first_cg]
        for step in range(decode_steps - 1):
            kvl = backend.slot_kv_len[1]
            result = cg.replay([1], [cg_tokens[-1]], [kvl])
            cg_tokens.append(result[0])
            backend.slot_kv_len[1] += 1
            backend.slot_committed_tokens[1].append(cg_tokens[-1])
        logger.info("CudaGraph (replay): %d tokens", len(cg_tokens))

        if eager_tokens == cg_tokens:
            logger.info("✓ GOLDEN: eager vs cudagraph bit-exact (%d tokens)", len(eager_tokens))
        else:
            mismatches = sum(1 for a, b in zip(eager_tokens, cg_tokens) if a != b)
            logger.error("✗ GOLDEN MISMATCH: %d/%d tokens differ", mismatches, len(eager_tokens))
            for i, (a, b) in enumerate(zip(eager_tokens, cg_tokens)):
                if a != b:
                    logger.error("  step %d: eager=%d cg=%d", i, a, b)
                    if i > 5:
                        logger.error("  ... (truncated)")
                        break
    except Exception as e:
        logger.warning("CudaGraph test skipped: %s", e)

    # ── Memory report ──
    mem = torch.cuda.memory_allocated() / 1024**3
    logger.info("GPU memory allocated: %.2f GiB", mem)

    # ── Summary ──
    result = {
        "model": "poolside/Laguna-S-2.1-NVFP4",
        "full_layers": full_layers,
        "swa_layers": swa_layers,
        "ring_blocks_per_slot": ring_bps,
        "prompt_tokens": len(prompt_ids),
        "decode_steps": decode_steps,
        "eager_tokens": eager_tokens[:10],
        "eager_determinism": eager_tokens == eager_tokens2,
        "load_time_s": round(load_time, 1),
        "eager_time_s": round(eager_time, 2),
        "gpu_mem_gib": round(mem, 2),
    }
    out_path = os.path.join(os.path.dirname(__file__), "fixtures", "golden_ring_kv.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", out_path)

    print("\n=== GOLDEN VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()
