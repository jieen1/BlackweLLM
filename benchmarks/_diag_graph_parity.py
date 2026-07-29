#!/usr/bin/env python3
"""Minimal diagnostic: graph vs eager parity, step by step."""
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
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
    args = EngineArgs(
        model=MODEL, max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=True, dtype="bfloat16", disable_log_stats=True,
        async_scheduling=False,
    )
    config = args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=128)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt!r} -> {len(prompt_ids)} tokens: {prompt_ids}")

    # === Eager path: 5 decode steps ===
    backend.reset_slot(0)
    eager_first = backend.prefill(0, prompt_ids)
    print(f"\nEager first token: {eager_first} = {tokenizer.decode([eager_first])!r}")
    eager_tokens = [eager_first]
    eager_logits = []
    for step in range(5):
        kv_len = backend.slot_kv_len[0]
        logits = backend._forward([0], [eager_tokens[-1]], [kv_len], qo_len=1, is_decode=True)
        l = logits[0].float().cpu()
        eager_logits.append(l)
        tok = int(l.argmax(dim=-1).item())
        eager_tokens.append(tok)
        backend.slot_kv_len[0] += 1
        backend.slot_committed_tokens[0].append(eager_tokens[-2])
        top5 = torch.topk(l, 5)
        top5_toks = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        top5_vals = top5.values.tolist()
        print(f"  Eager step {step}: kv_len={kv_len} -> tok={tok}({tokenizer.decode([tok])!r})  top5={list(zip(top5_toks, [f'{v:.2f}' for v in top5_vals]))}")

    # === Graph path: same prompt, same steps ===
    backend.reset_slot(0)
    graph_first = backend.prefill(0, prompt_ids)
    print(f"\nGraph first token: {graph_first} = {tokenizer.decode([graph_first])!r}")
    assert graph_first == eager_first, f"Prefill mismatch: {graph_first} vs {eager_first}"

    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    cg = LagunaCudaGraphDecode(backend, batch_size=1)
    cg.capture()

    graph_tokens = [graph_first]
    graph_logits = []
    for step in range(5):
        kv_len = backend.slot_kv_len[0]
        result = cg.replay([0], [graph_tokens[-1]], [kv_len])
        tok = result[0]
        l = cg._logits[0].float().cpu()
        graph_logits.append(l)
        graph_tokens.append(tok)
        backend.slot_kv_len[0] += 1
        backend.slot_committed_tokens[0].append(graph_tokens[-2])
        top5 = torch.topk(l, 5)
        top5_toks = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        top5_vals = top5.values.tolist()
        print(f"  Graph step {step}: kv_len={kv_len} -> tok={tok}({tokenizer.decode([tok])!r})  top5={list(zip(top5_toks, [f'{v:.2f}' for v in top5_vals]))}")

    # === Compare ===
    print("\n=== Comparison ===")
    for step in range(5):
        el = eager_logits[step]
        gl = graph_logits[step]
        diff = (el - gl).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(el.unsqueeze(0), gl.unsqueeze(0)).item()
        match = eager_tokens[step+1] == graph_tokens[step+1]
        print(f"  Step {step}: max_diff={diff:.4f}  cosine={cos:.6f}  token_match={'✅' if match else '❌'}")

    # === Test: replay with SAME slot as capture (slot 3) ===
    print("\n=== Same-slot test (slot 3) ===")
    backend.reset_slot(3)
    s3_first = backend.prefill(3, prompt_ids)
    print(f"Slot 3 first token: {s3_first} = {tokenizer.decode([s3_first])!r}")
    s3_tokens = [s3_first]
    for step in range(3):
        kv_len = backend.slot_kv_len[3]
        result = cg.replay([3], [s3_tokens[-1]], [kv_len])
        tok = result[0]
        s3_tokens.append(tok)
        backend.slot_kv_len[3] += 1
        backend.slot_committed_tokens[3].append(s3_tokens[-2])
        l = cg._logits[0].float().cpu()
        top5 = torch.topk(l, 5)
        top5_toks = [tokenizer.decode([t]) for t in top5.indices.tolist()]
        top5_vals = top5.values.tolist()
        print(f"  Slot3 step {step}: kv_len={kv_len} -> tok={tok}({tokenizer.decode([tok])!r})  top5={list(zip(top5_toks, [f'{v:.2f}' for v in top5_vals]))}")

    # Compare slot 3 graph vs eager
    backend.reset_slot(3)
    s3e_first = backend.prefill(3, prompt_ids)
    s3e_tokens = [s3e_first]
    for step in range(3):
        kv_len = backend.slot_kv_len[3]
        logits = backend._forward([3], [s3e_tokens[-1]], [kv_len], qo_len=1, is_decode=True)
        tok = int(logits[0].argmax(dim=-1).item())
        s3e_tokens.append(tok)
        backend.slot_kv_len[3] += 1
        backend.slot_committed_tokens[3].append(s3e_tokens[-2])
        print(f"  Slot3 eager step {step}: tok={tok}({tokenizer.decode([tok])!r})")

    print(f"\n  Slot3 graph tokens: {s3_tokens}")
    print(f"  Slot3 eager tokens: {s3e_tokens}")
    print(f"  Match: {'✅' if s3_tokens == s3e_tokens else '❌'}")

if __name__ == "__main__":
    main()
