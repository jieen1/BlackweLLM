#!/usr/bin/env python3
"""Wrapper-level parity: compare metadata + test disable_split_kv control."""
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
    from runtime.compat_vllm import EngineArgs
    args = EngineArgs(
        model=MODEL, max_model_len=4096, gpu_memory_utilization=0.80,
        enforce_eager=True, dtype="bfloat16", disable_log_stats=True,
        async_scheduling=False,
    )
    config = args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(config, num_slots=4, block_size=16, blocks_per_slot=256)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = tokenizer.encode("The capital of France is")

    # Prefill
    backend.reset_slot(0)
    first = backend.prefill(0, prompt_ids)
    kv_len = backend.slot_kv_len[0]
    print(f"Prefill: first={first} ({tokenizer.decode([first])!r}), kv_len={kv_len}")

    # --- Eager decode (reference) ---
    eager_logits = backend._forward([0], [first], [kv_len], qo_len=1, is_decode=True)[0].clone()
    print(f"Eager top5: {torch.topk(eager_logits.float(), 5).values.tolist()}")

    # --- Metadata from eager path ---
    backend._fill_decode_buffers([0], [first], [kv_len])
    common_meta = backend._build_common_attn_metadata([0], [kv_len], [1], True)
    print("\n=== Eager metadata ===")
    print(f"  seq_lens: {common_meta.seq_lens.tolist()}")
    print(f"  num_actual_tokens: {common_meta.num_actual_tokens}")
    print(f"  block_table_tensor[0,:5]: {common_meta.block_table_tensor[0,:5].tolist()}")
    print(f"  slot_mapping[:5]: {common_meta.slot_mapping[:5].tolist()}")
    print(f"  query_start_loc: {common_meta.query_start_loc.tolist()}")

    # --- Metadata from graph path ---
    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    cg = LagunaCudaGraphDecode(backend, batch_size=1)
    cg.capture()
    cg.reset()
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    graph_result = cg.replay([0], [first], [kv_len])
    graph_logits = cg._logits[0].clone()
    print("\n=== Graph metadata (after replay) ===")
    print(f"  input_ids: {cg._input_ids[:1].tolist()}")
    print(f"  positions: {cg._positions[:1].tolist()}")
    print(f"  slot_mapping[:5]: {cg._slot_mapping[:5].tolist()}")
    print(f"  block_table[0,:5]: {cg._block_table[0,:5].tolist()}")
    print(f"  indptr: {cg._fi_indptr_cpu[:2].tolist()}")
    print(f"  last_page_len: {cg._fi_last_page_len_cpu[:1].tolist()}")
    print(f"  indices[:5]: {cg._fi_indices_gpu[:5].tolist()}")

    cos = torch.nn.functional.cosine_similarity(
        graph_logits.float().unsqueeze(0), eager_logits.float().unsqueeze(0)
    ).item()
    max_diff = (graph_logits.float() - eager_logits.float()).abs().max().item()
    print(f"\nGraph vs eager: cos={cos:.6f}  max_diff={max_diff:.4f}")

    # --- sm_scale / window_left comparison ---
    print("\n=== Builder params ===")
    for gk, builder in backend._metadata_builders.items():
        print(f"  builder[{gk}]: sm_scale={builder.sm_scale}, window_left={builder.window_left}, "
              f"num_qo_heads={builder.num_qo_heads}, num_kv_heads={builder.num_kv_heads}, "
              f"head_dim={builder.head_dim}, page_size={builder.page_size}")
        print(f"    decode_fixed_split_size={builder.decode_fixed_split_size}, "
              f"disable_split_kv={builder.disable_split_kv}")
        print(f"    logits_soft_cap={builder.logits_soft_cap}")
        print(f"    kv_cache_dtype={builder.kv_cache_dtype}")

    # --- Eager builder's internal decode wrapper params ---
    print("\n=== Eager decode wrapper ===")
    for gk, builder in backend._metadata_builders.items():
        w = builder._decode_wrapper
        if w is not None:
            print(f"  wrapper[{gk}]: _sm_scale={w._sm_scale}, is_cuda_graph={w.is_cuda_graph_enabled}")

    # --- Graph wrapper params ---
    print("\n=== Graph decode wrapper ===")
    for gk, w in cg._decode_wrappers.items():
        print(f"  wrapper[{gk}]: _sm_scale={w._sm_scale}, is_cuda_graph={w.is_cuda_graph_enabled}")

if __name__ == "__main__":
    main()
