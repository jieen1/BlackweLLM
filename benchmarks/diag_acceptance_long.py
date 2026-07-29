"""Diagnostic: acceptance rate at different context lengths, eager vs CG decode.

Tests:
1. Short prompt (512 tokens), eager decode
2. 4K prompt, eager decode  
3. 4K prompt, CG decode
4. 8K prompt, eager decode

Usage: /home/bot/.venvs/vllm/bin/python benchmarks/diag_acceptance_long.py
"""
import os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "0"  # eager
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch

def build_vllm_config():
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    engine_args = EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
    )
    return engine_args.create_engine_config()

def make_prompt(tokenizer, target_len):
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of artificial intelligence and machine learning, "
        "the importance of efficient inference cannot be overstated. "
    )
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < target_len:
        tokens.extend(chunk)
    return tokens[:target_len]

def run_eager_steps(engine, backend, slot, first_token, aux_hs, prompt_len, n_steps=10):
    """Run n_steps eager speculative decode steps, return acceptance stats."""
    from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS
    
    if aux_hs is not None:
        aux_len = aux_hs[0].shape[0]
        aux_offset = prompt_len - aux_len
        engine._bulk_precompute_context_kv(slot, aux_hs, aux_len, aux_offset)
    del aux_hs
    torch.cuda.empty_cache()

    last_token = first_token
    total_accepted = 0
    total_draft = 0
    
    for step in range(n_steps):
        kv_len = backend.slot_kv_len[slot]
        logits, aux = engine._forward_main_with_aux([slot], [last_token], [kv_len], qo_len=1)
        bonus = int(logits[0].argmax(dim=-1).item())
        backend.slot_kv_len[slot] += 1
        
        if aux is not None:
            combined_input = torch.cat(aux, dim=-1)
            combined = engine.draft_model.combine_hidden_states(combined_input)
            engine._precompute_context_kv(slot, combined, kv_len)
        
        draft_tokens = engine._draft_forward(slot, bonus, kv_len + 1)
        accepted, num_accepted = engine._verify(slot, bonus, draft_tokens, kv_len + 1)
        
        backend.slot_kv_len[slot] += num_accepted
        for tok in accepted:
            backend.slot_committed_tokens[slot].append(tok)
        
        total_draft += NUM_SPECULATIVE_TOKENS
        total_accepted += len(accepted) - 1
        last_token = accepted[-1]
    
    return total_accepted, total_draft

def main():
    print("=" * 60)
    print("Acceptance Rate vs Context Length (Eager)")
    print("=" * 60)

    print("\nLoading model...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=4096)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
            "snapshots/07614121b31898586430f189d27a25a0be310843/"
        ), trust_remote_code=True,
    )

    n_steps = 10
    for ctx_len in [512, 2048, 4096, 8192, 16384]:
        prompt_ids = make_prompt(tokenizer, ctx_len)
        actual_len = len(prompt_ids)
        
        slot = 0
        backend.reset_slot(slot)
        torch.cuda.empty_cache()
        
        t0 = time.perf_counter()
        first_token, aux_hs = backend.prefill_with_aux(slot, prompt_ids)
        t_prefill = time.perf_counter() - t0
        
        accepted, drafted = run_eager_steps(
            engine, backend, slot, first_token, aux_hs, actual_len, n_steps
        )
        rate = accepted / max(drafted, 1)
        tok_per_step = 1 + accepted / max(n_steps, 1)
        
        print(f"  {actual_len:>6} tokens: accept={rate:.1%} ({accepted}/{drafted}), "
              f"tok/step={tok_per_step:.2f}, prefill={t_prefill:.1f}s")
        
        backend.reset_slot(slot)

    # Now test with CG decode
    print("\n--- With Decode CUDA Graph ---")
    os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
    engine2 = DFlashEngine(backend)
    
    for ctx_len in [512, 4096]:
        prompt_ids = make_prompt(tokenizer, ctx_len)
        actual_len = len(prompt_ids)
        
        t0 = time.perf_counter()
        _, stats = engine2.generate(prompt_ids, max_tokens=160)
        t_total = time.perf_counter() - t0
        
        print(f"  {actual_len:>6} tokens (CG): accept={stats['acceptance_rate']:.1%}, "
              f"tok/step={stats['tokens_per_step']:.2f}, "
              f"tok/s={stats['tok_per_s']:.1f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
