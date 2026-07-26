#!/usr/bin/env python3
"""Isolate: is cosine 0.93 from cudagraph kernel path or our metadata construction?
Test A: non-cudagraph wrapper + our graph metadata → compare with eager
Test B: cudagraph wrapper + disable_split_kv=True → compare with eager
"""
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

def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()

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

    # Prefill + eager reference
    backend.reset_slot(0)
    first = backend.prefill(0, prompt_ids)
    kv_len = backend.slot_kv_len[0]
    eager_logits = backend._forward([0], [first], [kv_len], qo_len=1, is_decode=True)[0].clone()
    print(f"Eager: top1={tokenizer.decode([eager_logits.argmax().item()])!r}")

    # === Test A: Use eager builder's own wrapper but with graph-style manual plan ===
    # This tests if our _run_plan parameters differ from builder.build()
    from runtime.compat_vllm import set_forward_context
    from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata, fast_plan_decode

    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)

    # Manually plan the eager builder's non-cudagraph wrapper with our parameters
    for group_key, builder in backend._metadata_builders.items():
        wl, nqh, nkvh = group_key
        wrapper = builder._decode_wrapper  # non-cudagraph wrapper
        fast_plan_decode(
            wrapper,
            indptr_cpu=torch.tensor([0, 1], dtype=torch.int32),
            indices=torch.tensor([256], dtype=torch.int32, device="cuda"),
            last_page_len_cpu=torch.tensor([7], dtype=torch.int32),
            num_qo_heads=nqh,
            num_kv_heads=nkvh,
            head_dim=128,
            page_size=16,
            pos_encoding_mode="NONE",
            window_left=wl,
            logits_soft_cap=None,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            sm_scale=builder.sm_scale,
            non_blocking=True,
            fixed_split_size=-1,  # match eager default
            disable_split_kv=False,
        )

    # Build metadata with the manually-planned wrappers
    slot_mapping = torch.tensor([4102], dtype=torch.long, device="cuda")
    input_ids = torch.tensor([first], dtype=torch.long, device="cuda")
    positions = torch.tensor([kv_len], dtype=torch.long, device="cuda")

    attn_metadata_dict = {}
    slot_mapping_dict = {}
    for group_key, builder in backend._metadata_builders.items():
        metadata = FlashInferMetadata(
            num_actual_tokens=1, slot_mapping=slot_mapping,
            q_data_type_prefill=torch.bfloat16, q_data_type_decode=torch.bfloat16,
            num_decodes=1, num_decode_tokens=1, num_prefills=0, num_prefill_tokens=0,
            causal=True, use_cascade=False, prefill=None,
            decode=FIDecode(wrapper=builder._decode_wrapper),
            cascade_wrapper=None,
        )
        for name in backend._layer_groups[group_key]:
            attn_metadata_dict[name] = metadata
            slot_mapping_dict[name] = slot_mapping

    with set_forward_context(attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict):
        hidden = backend.model.forward(input_ids, positions)
    manual_logits = backend.model.compute_logits(hidden)[0].clone()

    cos_a = cos_sim(manual_logits, eager_logits)
    print(f"\nTest A (manual plan, non-cudagraph wrapper): cos={cos_a:.6f}")
    print(f"  top1={tokenizer.decode([manual_logits.argmax().item()])!r}")

    # === Test B: Use eager builder's wrapper with fixed_split_size=2048 ===
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)

    for group_key, builder in backend._metadata_builders.items():
        wl, nqh, nkvh = group_key
        wrapper = builder._decode_wrapper
        fast_plan_decode(
            wrapper,
            indptr_cpu=torch.tensor([0, 1], dtype=torch.int32),
            indices=torch.tensor([256], dtype=torch.int32, device="cuda"),
            last_page_len_cpu=torch.tensor([7], dtype=torch.int32),
            num_qo_heads=nqh, num_kv_heads=nkvh, head_dim=128, page_size=16,
            pos_encoding_mode="NONE", window_left=wl, logits_soft_cap=None,
            q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
            sm_scale=builder.sm_scale, non_blocking=True,
            fixed_split_size=2048,  # match graph path
            disable_split_kv=False,
        )

    attn_metadata_dict2 = {}
    slot_mapping_dict2 = {}
    for group_key, builder in backend._metadata_builders.items():
        metadata = FlashInferMetadata(
            num_actual_tokens=1, slot_mapping=slot_mapping,
            q_data_type_prefill=torch.bfloat16, q_data_type_decode=torch.bfloat16,
            num_decodes=1, num_decode_tokens=1, num_prefills=0, num_prefill_tokens=0,
            causal=True, use_cascade=False, prefill=None,
            decode=FIDecode(wrapper=builder._decode_wrapper),
            cascade_wrapper=None,
        )
        for name in backend._layer_groups[group_key]:
            attn_metadata_dict2[name] = metadata
            slot_mapping_dict2[name] = slot_mapping

    with set_forward_context(attn_metadata_dict2, backend.vllm_config, slot_mapping=slot_mapping_dict2):
        hidden2 = backend.model.forward(input_ids, positions)
    manual_logits2 = backend.model.compute_logits(hidden2)[0].clone()

    cos_b = cos_sim(manual_logits2, eager_logits)
    print(f"\nTest B (manual plan, non-cudagraph, fixed_split=2048): cos={cos_b:.6f}")
    print(f"  top1={tokenizer.decode([manual_logits2.argmax().item()])!r}")

    # === Test C: builder.build() path (pure eager reference) ===
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    eager_logits2 = backend._forward([0], [first], [kv_len], qo_len=1, is_decode=True)[0].clone()
    cos_c = cos_sim(eager_logits2, eager_logits)
    print(f"\nTest C (eager again, sanity): cos={cos_c:.6f}")

if __name__ == "__main__":
    main()
