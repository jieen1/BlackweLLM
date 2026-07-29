"""Check if draft KV cache is populated and being read correctly."""
import os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "0"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

def make_prompt(tokenizer, n):
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < n:
        tokens.extend(chunk)
    return tokens[:n]

def main():
    from runtime.legacy_qwen36_vllm import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    engine_args = EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
        moe_backend="marlin",
    )
    vllm_config = engine_args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)

    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Short prompt for fast debugging
    ctx_len = 512
    prompt = make_prompt(tokenizer, ctx_len)
    print(f"Prompt: {len(prompt)} tokens")

    # Prefill
    slot = 0
    backend.reset_slot(slot)
    for kv_tensor in engine._draft_kv_caches.values():
        kv_tensor.zero_()

    first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt)
    print(f"First token: {first_token} = '{tokenizer.decode([first_token])}'")

    # Check aux hidden states
    if aux_hidden_states is not None:
        print(f"\nAux hidden states: {len(aux_hidden_states)} slices")
        for i, hs in enumerate(aux_hidden_states):
            print(f"  Slice {i}: shape={hs.shape}, norm={hs.norm().item():.2f}, "
                  f"mean={hs.mean().item():.4f}, std={hs.std().item():.4f}")
        
        # Precompute draft context KV
        aux_len = aux_hidden_states[0].shape[0]
        aux_offset = len(prompt) - aux_len
        engine._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)
    del aux_hidden_states
    torch.cuda.empty_cache()

    # Check draft KV cache after precompute
    print("\n--- Draft KV cache after precompute ---")
    for name, kv in list(engine._draft_kv_caches.items())[:2]:
        nonzero = (kv.view(torch.uint8) != 0).sum().item()
        total = kv.numel()
        print(f"  {name}: shape={kv.shape}, nonzero={nonzero}/{total} "
              f"({100*nonzero/total:.1f}%), dtype={kv.dtype}")

    # Now do one decode step and check the draft output
    kv_len = backend.slot_kv_len[slot]
    print(f"\n--- Decode step (kv_len={kv_len}) ---")
    
    # Main decode
    logits, aux = engine._forward_main_with_aux([slot], [first_token], [kv_len], qo_len=1)
    bonus_token = int(logits[0].argmax(dim=-1).item())
    backend.slot_kv_len[slot] += 1
    print(f"Bonus token: {bonus_token} = '{tokenizer.decode([bonus_token])}'")
    
    # Precompute context KV for this position
    if aux is not None:
        combined_input = torch.cat(aux, dim=-1)
        combined = engine.draft_model.combine_hidden_states(combined_input)
        print(f"Combined hidden: shape={combined.shape}, norm={combined.norm().item():.2f}")
        engine._precompute_context_kv(slot, combined, kv_len)
    
    # Draft forward WITH context
    draft_tokens_with_ctx = engine._draft_forward(slot, bonus_token, kv_len + 1)
    print(f"\nDraft tokens (with context): {draft_tokens_with_ctx[:8]}")
    print(f"  Decoded: '{tokenizer.decode(draft_tokens_with_ctx[:8])}'")
    
    # Draft forward WITHOUT context (zero KV)
    # Save and zero the draft KV
    saved_kvs = {}
    for name, kv in engine._draft_kv_caches.items():
        saved_kvs[name] = kv.clone()
        kv.zero_()
    
    draft_tokens_no_ctx = engine._draft_forward(slot, bonus_token, kv_len + 1)
    print(f"\nDraft tokens (NO context): {draft_tokens_no_ctx[:8]}")
    print(f"  Decoded: '{tokenizer.decode(draft_tokens_no_ctx[:8])}'")
    
    # Restore
    for name, kv in saved_kvs.items():
        engine._draft_kv_caches[name].copy_(kv)
    
    # Compare
    same = sum(1 for a, b in zip(draft_tokens_with_ctx, draft_tokens_no_ctx) if a == b)
    print(f"\nSame tokens with/without context: {same}/15")
    if same == 15:
        print("!!! DRAFT MODEL IS NOT READING KV CACHE !!!")
    else:
        print("Draft model IS reading KV cache (outputs differ)")
    
    # Also check: what does the MAIN model predict for these positions?
    # Run main model verify to get ground truth
    verify_tokens = [bonus_token] + draft_tokens_with_ctx
    verify_logits, _ = engine._forward_main_with_aux(
        [slot], verify_tokens, [kv_len + 1], qo_len=16
    )
    verify_argmax = verify_logits[:15].argmax(dim=-1).tolist()
    print(f"\nMain model verify (ground truth): {verify_argmax[:8]}")
    print(f"  Decoded: '{tokenizer.decode(verify_argmax[:8])}'")
    
    # Check if draft matches main
    match = sum(1 for d, v in zip(draft_tokens_with_ctx, verify_argmax) if d == v)
    print(f"Draft matches main: {match}/15")

if __name__ == "__main__":
    main()
