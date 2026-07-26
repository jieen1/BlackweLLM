"""Compare CG verify vs eager verify logits position by position."""
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

    ctx_len = 4096
    prompt = make_prompt(tokenizer, ctx_len)
    print(f"Prompt: {len(prompt)} tokens")

    # Prefill + generate a few tokens to get CG captured
    slot = 0
    tokens, stats = engine.generate(prompt, max_tokens=64)
    print(f"Generate: {stats['tok_per_s']:.1f} tok/s, accept={stats['acceptance_rate']:.1%}")
    print(f"CG: verify={engine._verify_cg is not None}, draft={engine._draft_cg is not None}")
    
    if engine._verify_cg is None:
        print("ERROR: verify CG not captured")
        return

    # Now do a controlled comparison:
    # 1. Reset and prefill
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
    
    # Do one decode step to get bonus token + aux
    kv_len = backend.slot_kv_len[slot]
    logits, aux = engine._forward_main_with_aux([slot], [first_token], [kv_len], qo_len=1)
    bonus_token = int(logits[0].argmax(dim=-1).item())
    backend.slot_kv_len[slot] += 1
    
    if aux is not None:
        combined_input = torch.cat(aux, dim=-1)
        combined = engine.draft_model.combine_hidden_states(combined_input)
        engine._precompute_context_kv(slot, combined, kv_len)
    
    # Get draft tokens
    draft_tokens = engine._draft_forward(slot, bonus_token, kv_len + 1)
    verify_tokens = [bonus_token] + draft_tokens
    print(f"\nVerify tokens: {verify_tokens[:8]}...")
    print(f"kv_len for verify: {kv_len + 1}")
    
    # EAGER verify
    eager_logits, _ = engine._forward_main_with_aux(
        [slot], verify_tokens, [kv_len + 1], qo_len=16
    )
    eager_argmax = eager_logits[:15].argmax(dim=-1).tolist()
    
    # CG verify
    cg_logits = engine._verify_cg.replay(slot, verify_tokens, kv_len + 1)
    cg_argmax = cg_logits[:15].argmax(dim=-1).tolist()
    
    # Compare
    print(f"\n{'Pos':<4} {'Eager':<12} {'CG':<12} {'Match':<6} {'CosSim':<8}")
    print("-" * 50)
    matches = 0
    for i in range(15):
        e_tok = eager_argmax[i]
        c_tok = cg_argmax[i]
        match = "✓" if e_tok == c_tok else "✗"
        if e_tok == c_tok:
            matches += 1
        # Cosine similarity of logits
        e_logits = eager_logits[i].float()
        c_logits = cg_logits[i].float()
        cos = torch.nn.functional.cosine_similarity(e_logits.unsqueeze(0), c_logits.unsqueeze(0)).item()
        e_text = tokenizer.decode([e_tok])
        c_text = tokenizer.decode([c_tok])
        print(f"{i:<4} {e_tok} '{e_text}'{'':<4} {c_tok} '{c_text}'{'':<4} {match:<6} {cos:.4f}")
    
    print(f"\nArgmax match: {matches}/15 = {matches/15:.1%}")
    
    # Overall cosine similarity
    cos_all = torch.nn.functional.cosine_similarity(
        eager_logits[:15].float().view(1, -1),
        cg_logits[:15].float().view(1, -1)
    ).item()
    print(f"Overall cosine similarity: {cos_all:.6f}")
    
    # Check top-5 overlap
    for i in range(3):
        e_top5 = set(eager_logits[i].topk(5).indices.tolist())
        c_top5 = set(cg_logits[i].topk(5).indices.tolist())
        overlap = len(e_top5 & c_top5)
        print(f"  Pos {i} top-5 overlap: {overlap}/5")

if __name__ == "__main__":
    main()
