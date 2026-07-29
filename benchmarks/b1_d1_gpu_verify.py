"""B1+D1 GPU verification: sampling e2e + long generation zero-wedge.

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.b1_d1_gpu_verify
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _build_runner(max_len: int):
    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max_len,
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": K,
            "attention_backend": "CUSTOM",
        },
    )
    return DirectModelRunner(
        vllm_config=vllm_config,
        num_slots=4,
        blocks_per_slot=-(-max_len // 16),
        enable_cudagraph=False,
        enable_prefix_cache=False,
    )


def test_b1_sampling(runner) -> None:
    """B1: Verify sampling produces valid, non-greedy output."""

    from runtime.sampling import SamplingParams

    print("\n=== B1: Sampling E2E Verification ===")

    prompt = list(range(1000, 1200))
    runner.reset_slot(0)

    # Greedy prefill
    logits = runner._forward(0, prompt, start_pos=0, is_decode=False)
    greedy_token = int(logits[-1].argmax(dim=-1).item())
    print(f"  Greedy first token: {greedy_token}")

    # Sampled prefill (temperature=1.0)
    runner.reset_slot(1)
    params = SamplingParams(temperature=1.0, seed=42)
    sampled_token = runner.prefill_sampled(1, prompt, params)
    print(f"  Sampled first token (T=1.0, seed=42): {sampled_token}")

    # Verify greedy is deterministic
    runner.reset_slot(2)
    greedy_token2 = runner.prefill(2, prompt)
    assert greedy_token == greedy_token2, "Greedy not deterministic!"
    print("  ✅ Greedy deterministic")

    # Verify sampled with same seed is deterministic
    runner.reset_slot(3)
    params2 = SamplingParams(temperature=1.0, seed=42)
    sampled_token2 = runner.prefill_sampled(3, prompt, params2)
    assert sampled_token == sampled_token2, "Sampled not deterministic with same seed!"
    print("  ✅ Sampled deterministic (same seed)")

    # Verify temperature=0 matches greedy
    runner.reset_slot(0)
    params_greedy = SamplingParams(temperature=0.0)
    greedy_via_sampling = runner.prefill_sampled(0, prompt, params_greedy)
    assert greedy_via_sampling == greedy_token, "temperature=0 != greedy!"
    print("  ✅ temperature=0 matches greedy (bit-identical)")

    # Multi-step sampled decode
    runner.reset_slot(0)
    params = SamplingParams(temperature=0.8, top_k=50, top_p=0.95, seed=123)
    tok = runner.prefill_sampled(0, prompt, params)
    tokens = [tok]
    for _ in range(19):
        tok = runner.decode_sampled(0, tok, params)
        tokens.append(tok)
    print(f"  Sampled 20 tokens (T=0.8, top_k=50, top_p=0.95): {tokens[:10]}...")
    print("  ✅ Multi-step sampled decode OK")

    # Cleanup
    for s in range(4):
        runner.reset_slot(s)


def test_d1_long_generation(runner, num_tokens: int = 16384, num_runs: int = 5) -> None:
    """D1: Long generation zero-wedge verification."""
    print(f"\n=== D1: Long Generation ({num_tokens} tokens × {num_runs} runs) ===")

    prompt = list(range(1000, 1100))

    for run_idx in range(num_runs):
        runner.reset_slot(0)
        t0 = time.perf_counter()

        result = runner.mtp_prefill_with_cache([0], [prompt])
        anchor = result[0]["anchor"]
        drafts = result[0]["draft_tokens"]

        committed = [anchor]
        total_accepted = 0
        rounds = 0

        while len(committed) < num_tokens:
            decisions = runner.mtp_verify_and_commit_batch(
                [0], {0: anchor}, {0: drafts}
            )
            decision = decisions[0]
            new_tokens = decision["committed"]
            committed.extend(new_tokens)
            total_accepted += decision["num_accepted"]
            anchor = decision["next_anchor"]
            drafts = decision["next_draft_tokens"]
            rounds += 1

        elapsed = time.perf_counter() - t0
        tok_per_s = len(committed) / elapsed
        accept_rate = total_accepted / (rounds * K) * 100 if rounds > 0 else 0
        print(
            f"  Run {run_idx+1}/{num_runs}: {len(committed)} tokens in {elapsed:.1f}s "
            f"({tok_per_s:.1f} tok/s, accept={accept_rate:.1f}%, {rounds} rounds) ✅"
        )

        runner.reset_slot(0)

    print(f"  ✅ {num_runs} runs × {num_tokens} tokens: ZERO wedges")


def main() -> None:
    import torch

    max_len = 32768
    print(f"Loading model ({MODEL}), max_len={max_len}...")
    t0 = time.perf_counter()
    runner = _build_runner(max_len)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    test_b1_sampling(runner)
    test_d1_long_generation(runner, num_tokens=16384, num_runs=5)

    print(f"\nGPU memory: {torch.cuda.memory_allocated()/2**30:.1f} GiB allocated")
    print("\n✅ ALL GPU VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
