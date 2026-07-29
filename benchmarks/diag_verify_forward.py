"""Compare main model: sequential decode (qo=1) vs parallel verify (qo=16)."""
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
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ctx_len = 512
    prompt = make_prompt(tokenizer, ctx_len)
    print(f"Prompt: {len(prompt)} tokens")

    # Prefill
    slot = 0
    backend.reset_slot(slot)
    first_token = backend.prefill(slot, prompt)
    print(f"First token: {first_token} = '{tokenizer.decode([first_token])}'")
    
    # Sequential decode: get 16 tokens one by one (ground truth)
    print("\n--- Sequential decode (qo=1, ground truth) ---")
    seq_tokens = [first_token]
    for i in range(15):
        tok = backend.decode(slot, seq_tokens[-1])
        seq_tokens.append(tok)
    print(f"Sequential: {seq_tokens}")
    print(f"  Decoded: '{tokenizer.decode(seq_tokens)}'")
    
    # Reset and prefill again
    backend.reset_slot(slot)
    first_token2 = backend.prefill(slot, prompt)
    assert first_token2 == first_token, f"Prefill mismatch: {first_token2} vs {first_token}"
    
    # Now try parallel forward with qo=16
    # Use the same tokens as input (teacher forcing)
    print("\n--- Parallel forward (qo=16, teacher forcing) ---")
    kv_len = backend.slot_kv_len[slot]
    print(f"kv_len before verify: {kv_len}")
    
    # Build input: [first_token] + seq_tokens[1:]  (the 16 tokens we want to verify)
    verify_input = seq_tokens[:16]
    print(f"Verify input: {verify_input}")
    
    # Call _forward_main_with_aux with qo_len=16
    # We need the engine for _forward_main_with_aux, but let's use backend directly
    # Actually let's just call the backend's forward path
    
    # Build attention metadata for qo=16
    from oracle.qwen36_vllm.vllm_compat import set_forward_context, set_current_vllm_config
    
    num_reqs = 1
    qo_lens = [16]
    kv_lengths = [kv_len]
    is_decode = False
    
    common_meta = backend._build_common_attn_metadata(
        [slot], kv_lengths, qo_lens, is_decode
    )
    
    attn_metadata_dict = {}
    slot_mapping_dict = {}
    
    swa_meta = None
    if backend._ring_blocks_per_slot > 0 and backend._swa_layer_names:
        swa_meta = backend._build_swa_attn_metadata(
            [slot], kv_lengths, qo_lens, is_decode, swa_mode="verify_ring"
        )
    
    for group_key, builder in backend._metadata_builders.items():
        wl = group_key[0]
        is_swa_group = wl >= 0
        meta = swa_meta if (is_swa_group and swa_meta is not None) else common_meta
        with set_current_vllm_config(backend.vllm_config):
            metadata = builder.build(common_prefix_len=0, common_attn_metadata=meta)
        for name in backend._layer_groups[group_key]:
            attn_metadata_dict[name] = metadata
            slot_mapping_dict[name] = meta.slot_mapping
    
    # Build input tensors
    input_ids = torch.tensor(verify_input, dtype=torch.long, device=backend.device)
    positions = torch.tensor(
        list(range(kv_len, kv_len + 16)), dtype=torch.long, device=backend.device
    )
    
    print(f"Positions: {positions.tolist()}")
    print(f"Slot mapping (first 16): {common_meta.slot_mapping[:16].tolist()}")
    
    with set_forward_context(attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict):
        result = backend.model.forward(input_ids, positions)
    
    if isinstance(result, tuple):
        hidden_states = result[0]
    else:
        hidden_states = result
    
    logits = backend.model.compute_logits(hidden_states)
    parallel_argmax = logits[:15].argmax(dim=-1).tolist()
    
    print(f"\nParallel verify logits argmax: {parallel_argmax}")
    print(f"  Decoded: '{tokenizer.decode(parallel_argmax)}'")
    
    # Compare: parallel_argmax[i] should predict seq_tokens[i+1]
    # logits[0] predicts what comes after verify_input[0] = first_token
    # So logits[0] should == seq_tokens[1]
    print("\n--- Comparison ---")
    print(f"{'Pos':<4} {'Expected':<10} {'Got':<10} {'Match':<6}")
    matches = 0
    for i in range(15):
        expected = seq_tokens[i + 1]
        got = parallel_argmax[i]
        match = "✓" if expected == got else "✗"
        if expected == got:
            matches += 1
        exp_text = tokenizer.decode([expected])
        got_text = tokenizer.decode([got])
        print(f"{i:<4} {expected} '{exp_text}'{'':<4} {got} '{got_text}'{'':<4} {match}")
    
    print(f"\nMatch rate: {matches}/15 = {matches/15:.1%}")
    if matches < 10:
        print("!!! PARALLEL VERIFY IS BROKEN !!!")
        print("The main model's qo=16 forward produces different results than sequential decode.")
        print("This is the root cause of low acceptance rate.")

if __name__ == "__main__":
    main()
