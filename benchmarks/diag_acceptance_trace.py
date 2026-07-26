"""Trace acceptance: show draft vs verify tokens for first N steps."""
import os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "0"  # eager for debugging
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
    from runtime.compat_vllm import EngineArgs
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
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=8448)

    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use short context first (4K) to isolate the issue
    ctx_len = 4096
    prompt = make_prompt(tokenizer, ctx_len)
    print(f"Prompt: {len(prompt)} tokens")
    print(f"Prompt text (last 100 tokens): {tokenizer.decode(prompt[-100:])}")
    
    # Prefill
    slot = 0
    backend.reset_slot(slot)
    for kv_tensor in engine._draft_kv_caches.values():
        kv_tensor.zero_()
    
    first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt)
    print(f"\nFirst token: {first_token} = '{tokenizer.decode([first_token])}'")
    
    # Precompute draft context KV
    if aux_hidden_states is not None:
        aux_len = aux_hidden_states[0].shape[0]
        aux_offset = len(prompt) - aux_len
        print(f"Aux hidden states: {aux_len} positions, offset={aux_offset}")
        engine._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)
    del aux_hidden_states
    torch.cuda.empty_cache()
    
    # Trace first 10 speculative steps
    print(f"\n{'='*80}")
    print("Tracing speculative decode steps (eager mode, no CG)")
    print(f"{'='*80}")
    
    last_token = first_token
    total_accepted = 0
    total_draft = 0
    
    for step in range(10):
        kv_len = backend.slot_kv_len[slot]
        
        # Step 1: Main decode
        logits, aux = engine._forward_main_with_aux([slot], [last_token], [kv_len], qo_len=1)
        bonus_token = int(logits[0].argmax(dim=-1).item())
        backend.slot_kv_len[slot] += 1
        
        # Step 2: Combine + precompute
        if aux is not None:
            combined_input = torch.cat(aux, dim=-1)
            combined = engine.draft_model.combine_hidden_states(combined_input)
            engine._precompute_context_kv(slot, combined, kv_len)
        
        # Step 3: Draft forward
        draft_tokens = engine._draft_forward(slot, bonus_token, kv_len + 1)
        
        # Step 4: Verify (eager)
        verify_tokens = [bonus_token] + draft_tokens
        verify_logits, _ = engine._forward_main_with_aux(
            [slot], verify_tokens, [kv_len + 1], qo_len=16
        )
        verify_argmax = verify_logits[:15].argmax(dim=-1).tolist()
        
        # Accept/reject
        accepted = [bonus_token]
        num_accepted = 0
        for i, (vtok, dtok) in enumerate(zip(verify_argmax, draft_tokens)):
            if vtok == dtok:
                accepted.append(dtok)
                num_accepted += 1
            else:
                accepted.append(vtok)
                num_accepted += 1
                break
        
        # Update state
        backend.slot_kv_len[slot] += num_accepted
        for tok in accepted:
            backend.slot_committed_tokens[slot].append(tok)
        
        total_draft += 15
        total_accepted += num_accepted
        
        # Print trace
        bonus_text = tokenizer.decode([bonus_token])
        draft_text = tokenizer.decode(draft_tokens[:8])
        verify_text = tokenizer.decode(verify_argmax[:8])
        accepted_text = tokenizer.decode(accepted)
        
        print(f"\nStep {step+1} (kv_len={kv_len}):")
        print(f"  Bonus: {bonus_token} '{bonus_text}'")
        print(f"  Draft[0:8]:  {draft_tokens[:8]}")
        print(f"  Verify[0:8]: {verify_argmax[:8]}")
        print(f"  Match: {''.join('✓' if d==v else '✗' for d,v in zip(draft_tokens[:15], verify_argmax[:15]))}")
        print(f"  Accepted: {num_accepted}/15 → '{accepted_text}'")
        
        last_token = accepted[-1]
    
    print(f"\n{'='*80}")
    print(f"Overall: {total_accepted}/{total_draft} = {total_accepted/max(total_draft,1):.1%} acceptance")
    print(f"Avg tokens/step: {(total_accepted + 10) / 10:.2f}")  # +10 for bonus tokens

if __name__ == "__main__":
    main()
