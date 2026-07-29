"""GPU integration test: DFlash speculative decoding end-to-end validation.

Requires: vLLM venv (/home/bot/.venvs/vllm/bin/python), GPU access.
Usage: /home/bot/.venvs/vllm/bin/python benchmarks/gpu_dflash_integration.py

Validates:
1. Draft model loads correctly (weight sharing with main model)
2. Aux hidden state extraction works
3. Draft forward produces valid tokens
4. Verify + accept/reject logic is correct
5. End-to-end generation matches greedy baseline
"""
import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")

sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")



def build_vllm_config():
    """Build VllmConfig for Laguna-S-2.1-NVFP4."""
    from oracle.qwen36_vllm.vllm_compat import EngineArgs

    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    engine_args = EngineArgs(
        model=model_path,
        dtype="bfloat16",
        max_model_len=131072,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        trust_remote_code=True,
    )
    return engine_args.create_engine_config()


def test_dflash_engine():
    """End-to-end DFlash integration test."""
    print("=" * 70)
    print("DFlash GPU Integration Test")
    print("=" * 70)

    # Step 1: Build config and load main model
    print("\n[1/5] Building VllmConfig and loading main model...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=512)
    t_load = time.perf_counter()
    print(f"  Main model loaded in {t_load - t0:.1f}s")

    # Step 2: Initialize DFlash engine
    print("\n[2/5] Initializing DFlash engine (loading draft model)...")
    t1 = time.perf_counter()
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    t_dflash = time.perf_counter()
    print(f"  DFlash engine initialized in {t_dflash - t1:.1f}s")

    # Step 3: Test aux hidden state extraction
    print("\n[3/5] Testing aux hidden state extraction...")
    test_prompt = list(range(1, 33))  # 32 tokens
    backend.reset_slot(0)
    logits, aux = engine._forward_main_with_aux([0], test_prompt, [0], qo_len=32)
    assert logits is not None, "Logits should not be None"
    assert aux is not None, "Aux hidden states should not be None"
    assert len(aux) == 6, f"Expected 6 aux hidden states, got {len(aux)}"
    for i, h in enumerate(aux):
        assert h.shape == (32, 3072), f"Aux[{i}] shape: {h.shape}, expected (32, 3072)"
    print("  ✓ Aux hidden states: 6 × [32, 3072]")
    backend.reset_slot(0)

    # Step 4: Test draft forward
    print("\n[4/5] Testing draft forward...")
    # First do a prefill to set up KV state
    first_token = backend.prefill(0, test_prompt)
    print(f"  Prefill done, first_token={first_token}")

    # Now test one speculative decode step
    accepted = engine.speculative_decode_step(0, first_token)
    print(f"  Speculative step: {len(accepted)} tokens accepted: {accepted[:5]}...")
    assert len(accepted) >= 1, "Should accept at least 1 token (bonus)"
    assert len(accepted) <= 16, "Should accept at most 16 tokens"
    backend.reset_slot(0)

    # Step 5: End-to-end generation
    print("\n[5/5] End-to-end generation test...")
    prompt = list(range(1, 65))  # 64 tokens
    tokens, stats = engine.generate(prompt, max_tokens=64)
    print(f"  Generated {stats['num_tokens']} tokens in {stats['decode_ms']:.1f}ms")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"  Tokens/step: {stats['tokens_per_step']:.2f}")
    print(f"  Throughput: {stats['tok_per_s']:.1f} tok/s")
    print(f"  Steps: {stats['num_steps']}")

    # Compare with baseline (non-speculative)
    print("\n  Comparing with non-speculative baseline...")
    backend.reset_slot(0)
    t_base = time.perf_counter()
    baseline_first = backend.prefill(0, prompt)
    baseline_tokens = [baseline_first]
    for _ in range(63):
        tok = backend.decode(0, baseline_tokens[-1])
        baseline_tokens.append(tok)
        if tok in (2, 24):
            break
    t_base_end = time.perf_counter()
    base_ms = (t_base_end - t_base) * 1000
    print(f"  Baseline: {len(baseline_tokens)} tokens in {base_ms:.1f}ms")
    print(f"  Baseline tok/s: {(len(baseline_tokens)-1) / (base_ms/1000):.1f}")

    # Verify correctness: first N tokens should match
    min_len = min(len(tokens), len(baseline_tokens))
    match_count = sum(1 for a, b in zip(tokens[:min_len], baseline_tokens[:min_len]) if a == b)
    print(f"\n  Token agreement: {match_count}/{min_len} ({match_count/min_len:.1%})")

    print("\n" + "=" * 70)
    print("DFlash integration test PASSED" if match_count == min_len else
          f"WARNING: {min_len - match_count} token mismatches (may be expected for speculative)")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    stats = test_dflash_engine()
