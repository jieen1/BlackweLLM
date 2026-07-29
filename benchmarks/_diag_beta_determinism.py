#!/usr/bin/env python3
"""β-v2: always-plan replay, same input 3x, bit-compare logits."""
from __future__ import annotations
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QSR_A2_CUSTOM_GEMM", "0")

import torch
MODEL = "poolside/Laguna-S-2.1-NVFP4"

def main():
    from oracle.qwen36_vllm.vllm_compat import EngineArgs, CUDAGraphMode
    args = EngineArgs(
        model=MODEL, max_model_len=4096, gpu_memory_utilization=0.85,
        dtype="bfloat16", disable_log_stats=True, async_scheduling=False,
    )
    config = args.create_engine_config()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=128)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = tokenizer.encode("The capital of France is")

    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    cg = LagunaCudaGraphDecode(backend, batch_size=1)
    cg.capture()

    backend.reset_slot(0)
    first = backend.prefill(0, prompt_ids)
    print(f"First token: {first} = {tokenizer.decode([first])!r}")

    kv_len = backend.slot_kv_len[0]
    print("\n=== β-v2: always-plan, same input 3x replay ===")

    logits_list = []
    tokens_list = []
    for run in range(3):
        cg.reset()
        result = cg.replay([0], [first], [kv_len])
        logits_list.append(cg._logits[0].clone())
        tokens_list.append(result[0])
        print(f"  Run {run}: token={result[0]} ({tokenizer.decode([result[0]])!r})")

    # Bit-compare all pairs
    for i in range(3):
        for j in range(i+1, 3):
            bit_exact = torch.equal(logits_list[i], logits_list[j])
            max_diff = (logits_list[i].float() - logits_list[j].float()).abs().max().item()
            print(f"  Run {i} vs {j}: bit-exact={'✅' if bit_exact else '❌'}  max_diff={max_diff}")

    # Compare with eager
    print("\n=== Eager reference ===")
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    eager_logits = backend._forward([0], [first], [kv_len], qo_len=1, is_decode=True)[0].clone()
    eager_tok = int(eager_logits.argmax(dim=-1).item())
    print(f"  Eager token: {eager_tok} ({tokenizer.decode([eager_tok])!r})")

    for i in range(3):
        bit_exact = torch.equal(logits_list[i], eager_logits)
        max_diff = (logits_list[i].float() - eager_logits.float()).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            logits_list[i].float().unsqueeze(0), eager_logits.float().unsqueeze(0)
        ).item()
        print(f"  Graph run {i} vs eager: bit-exact={'✅' if bit_exact else '❌'}  max_diff={max_diff:.4f}  cosine={cos:.6f}")

    # Multi-step determinism: 20 tokens, twice
    print("\n=== Multi-step determinism (20 tokens × 2 runs) ===")
    def run_decode(n_steps):
        cg.reset()
        backend.reset_slot(0)
        ft = backend.prefill(0, prompt_ids)
        tokens = [ft]
        for _ in range(n_steps - 1):
            kvl = backend.slot_kv_len[0]
            r = cg.replay([0], [tokens[-1]], [kvl])
            tokens.append(r[0])
            backend.slot_kv_len[0] += 1
            backend.slot_committed_tokens[0].append(tokens[-2])
            if r[0] in (2, 24):
                break
        return tokens

    run1 = run_decode(20)
    run2 = run_decode(20)
    match = run1 == run2
    t1 = tokenizer.decode(run1, skip_special_tokens=True)
    t2 = tokenizer.decode(run2, skip_special_tokens=True)
    print(f"  Run 1: {t1[:80]!r}")
    print(f"  Run 2: {t2[:80]!r}")
    print(f"  Match: {'✅ PASS' if match else '❌ FAIL'}")

if __name__ == "__main__":
    main()
