"""Compare CG vs eager at each stage of speculative decode step.
Isolates: decode aux → combine → draft → verify."""
import os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
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
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)

    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use 4K context (single chunk, known good)
    ctx_len = 4096
    prompt = make_prompt(tokenizer, ctx_len)
    
    # First generate to trigger CG capture
    tokens, stats = engine.generate(prompt, max_tokens=64)
    print(f"Initial generate: {stats['tok_per_s']:.1f} tok/s, accept={stats['acceptance_rate']:.1%}")
    print(f"CG: decode={engine._cuda_graph is not None}, verify={engine._verify_cg is not None}, draft={engine._draft_cg is not None}")
    
    if engine._cuda_graph is None or engine._verify_cg is None or engine._draft_cg is None:
        print("ERROR: CGs not captured")
        return
    
    # Now do a controlled comparison: reset and prefill
    slot = 0
    backend.reset_slot(slot)
    for kv_tensor in engine._draft_kv_caches.values():
        kv_tensor.zero_()
    
    first_token, aux = backend.prefill_with_aux(slot, prompt)
    if aux is not None:
        aux_len = aux[0].shape[0]
        aux_offset = len(prompt) - aux_len
        engine._bulk_precompute_context_kv(slot, aux, aux_len, aux_offset)
    del aux
    torch.cuda.empty_cache()
    
    # Run 5 speculative steps, comparing CG vs eager at each stage
    print(f"\n{'='*70}")
    print("Per-stage CG vs Eager comparison (5 steps)")
    print(f"{'='*70}")
    
    last_token = first_token
    
    for step in range(5):
        kv_len = backend.slot_kv_len[slot]
        
        # === STAGE 1: Decode with aux ===
        # CG path
        cg_next, cg_aux = engine._cuda_graph.replay_with_aux([slot], [last_token], [kv_len])
        cg_bonus = cg_next[0]
        
        # Eager path (need to reset KV state since CG already wrote)
        # Actually CG and eager share the same KV cache, so we can't easily compare
        # Instead: compare CG aux vs what eager would produce
        # Save CG aux, then do eager decode on a FRESH state
        cg_aux_norms = [a.norm().item() for a in cg_aux] if cg_aux else []
        
        # For eager comparison, we need to NOT advance kv_len
        # The CG already wrote to KV cache at position kv_len
        # Let's just check if the CG aux looks reasonable
        backend.slot_kv_len[slot] += 1
        
        # === STAGE 2: Combine + precompute ===
        if cg_aux is not None:
            combined_input = torch.cat(cg_aux, dim=-1)
            combined = engine.draft_model.combine_hidden_states(combined_input)
            engine._precompute_context_kv(slot, combined, kv_len)
            combined_norm = combined.norm().item()
        else:
            combined_norm = 0.0
        
        # === STAGE 3: Draft ===
        # CG draft
        cg_draft = engine._draft_cg.replay(slot, cg_bonus, kv_len + 1)
        
        # Eager draft
        eager_draft = engine._draft_forward(slot, cg_bonus, kv_len + 1)
        
        draft_match = sum(1 for a, b in zip(cg_draft, eager_draft) if a == b)
        
        # === STAGE 4: Verify ===
        verify_tokens = [cg_bonus] + cg_draft
        cg_verify_logits = engine._verify_cg.replay(slot, verify_tokens, kv_len + 1)
        cg_verify_argmax = cg_verify_logits[:15].argmax(dim=-1).tolist()
        
        # Eager verify
        eager_verify_logits, _ = engine._forward_main_with_aux(
            [slot], verify_tokens, [kv_len + 1], qo_len=16
        )
        eager_verify_argmax = eager_verify_logits[:15].argmax(dim=-1).tolist()
        
        verify_match = sum(1 for a, b in zip(cg_verify_argmax, eager_verify_argmax) if a == b)
        
        # Acceptance
        accepted = [cg_bonus]
        num_accepted = 0
        for vtok, dtok in zip(cg_verify_argmax, cg_draft):
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
        
        print(f"\nStep {step+1} (kv_len={kv_len}):")
        print(f"  Bonus: {cg_bonus} '{tokenizer.decode([cg_bonus])}'")
        print(f"  Aux norms: {[f'{n:.1f}' for n in cg_aux_norms[:3]]}...")
        print(f"  Combined norm: {combined_norm:.2f}")
        print(f"  Draft CG vs eager: {draft_match}/15 match")
        if draft_match < 15:
            print(f"    CG:    {cg_draft[:8]}")
            print(f"    Eager: {eager_draft[:8]}")
        print(f"  Verify CG vs eager: {verify_match}/15 match")
        if verify_match < 15:
            print(f"    CG:    {cg_verify_argmax[:8]}")
            print(f"    Eager: {eager_verify_argmax[:8]}")
        print(f"  Accepted: {num_accepted}/15")
        
        last_token = accepted[-1]
    
    # Now run full generate with CG to get overall acceptance
    print(f"\n{'='*70}")
    print("Full generate comparison (CG vs eager)")
    print(f"{'='*70}")
    
    # CG generate
    tokens_cg, stats_cg = engine.generate(prompt, max_tokens=128)
    print(f"CG:    {stats_cg['tok_per_s']:.1f} tok/s, accept={stats_cg['acceptance_rate']:.1%}, tok/step={stats_cg['tokens_per_step']:.2f}")
    
    # Eager generate
    os.environ["QSR_DFLASH_CUDA_GRAPH"] = "0"
    engine._cuda_graph = None
    engine._verify_cg = None
    engine._draft_cg = None
    engine._cg_captured = False
    engine._use_cuda_graph = False
    
    tokens_eager, stats_eager = engine.generate(prompt, max_tokens=128)
    print(f"Eager: {stats_eager['tok_per_s']:.1f} tok/s, accept={stats_eager['acceptance_rate']:.1%}, tok/step={stats_eager['tokens_per_step']:.2f}")

if __name__ == "__main__":
    main()
