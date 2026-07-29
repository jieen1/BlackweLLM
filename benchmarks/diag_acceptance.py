"""Diagnostic: why is DFlash acceptance rate low?

Checks:
1. Aux hidden states are non-None and non-zero
2. combine_hidden_states produces reasonable output
3. Draft model produces reasonable logits
4. Draft tokens vs verify tokens comparison
5. Draft KV cache state

Usage: /home/bot/.venvs/vllm/bin/python benchmarks/diag_acceptance.py
"""
import os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "0"  # eager for debugging
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

def main():
    print("=" * 60)
    print("DFlash Acceptance Diagnostic")
    print("=" * 60)

    print("\n[1] Loading model...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=4096)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    print("\n[2] Initializing DFlash...")
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    print("  DFlash ready")

    # Check aux hidden state layers
    model = backend.model
    print("\n[3] Model aux hidden state config:")
    if hasattr(model, '_aux_hidden_state_layers'):
        print(f"  _aux_hidden_state_layers = {model._aux_hidden_state_layers}")
    elif hasattr(model, 'model') and hasattr(model.model, '_aux_hidden_state_layers'):
        print(f"  model.model._aux_hidden_state_layers = {model.model._aux_hidden_state_layers}")
    else:
        print("  WARNING: no _aux_hidden_state_layers found!")
        # Check all attributes
        for attr in dir(model):
            if 'aux' in attr.lower():
                print(f"  model.{attr} = {getattr(model, attr, 'N/A')}")
        if hasattr(model, 'model'):
            for attr in dir(model.model):
                if 'aux' in attr.lower():
                    print(f"  model.model.{attr} = {getattr(model.model, attr, 'N/A')}")

    # Short prefill
    print("\n[4] Short prefill + decode diagnostic...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
            "snapshots/07614121b31898586430f189d27a25a0be310843/"
        ), trust_remote_code=True,
    )
    
    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:512]
    print(f"  Prompt: {len(prompt_ids)} tokens")

    slot = 0
    backend.reset_slot(slot)
    torch.cuda.empty_cache()

    # Prefill with aux
    first_token, aux_hs = backend.prefill_with_aux(slot, prompt_ids)
    print(f"  First token: {first_token} ({tokenizer.decode([first_token])})")
    print(f"  Aux hidden states: {type(aux_hs)}")
    if aux_hs is not None:
        print(f"  Num aux layers: {len(aux_hs)}")
        for i, h in enumerate(aux_hs):
            print(f"    Layer {i}: shape={h.shape}, norm={h.float().norm():.4f}, "
                  f"mean={h.float().mean():.6f}, std={h.float().std():.6f}")
    else:
        print("  WARNING: aux_hs is None! Draft model has no context.")

    # Bulk precompute draft KV
    if aux_hs is not None:
        aux_len = aux_hs[0].shape[0]
        aux_offset = len(prompt_ids) - aux_len
        engine._bulk_precompute_context_kv(slot, aux_hs, aux_len, aux_offset)
        print(f"  Precomputed draft KV: {aux_len} positions (offset={aux_offset})")

    del aux_hs
    torch.cuda.empty_cache()

    # Run 5 speculative steps with detailed logging
    print("\n[5] Speculative decode steps (detailed):")
    last_token = first_token
    total_accepted = 0
    total_draft = 0

    for step in range(5):
        kv_len = backend.slot_kv_len[slot]
        
        # Step 1: Main decode with aux
        logits, aux_hs = engine._forward_main_with_aux(
            [slot], [last_token], [kv_len], qo_len=1
        )
        bonus_token = int(logits[0].argmax(dim=-1).item())
        backend.slot_kv_len[slot] += 1

        # Step 2: Combine + precompute
        if aux_hs is not None:
            combined_input = torch.cat(aux_hs, dim=-1)
            combined = engine.draft_model.combine_hidden_states(combined_input)
            engine._precompute_context_kv(slot, combined, kv_len)
            if step == 0:
                print(f"  Step {step}: combined norm={combined.float().norm():.4f}")

        # Step 3: Draft forward
        draft_tokens = engine._draft_forward(slot, bonus_token, kv_len + 1)
        
        # Step 4: Verify
        accepted, num_accepted = engine._verify(
            slot, bonus_token, draft_tokens, kv_len + 1
        )

        # Update state
        backend.slot_kv_len[slot] += num_accepted
        for tok in accepted:
            backend.slot_committed_tokens[slot].append(tok)

        total_draft += 15
        total_accepted += len(accepted) - 1

        bonus_text = tokenizer.decode([bonus_token])
        draft_text = tokenizer.decode(draft_tokens[:5])
        accepted_text = tokenizer.decode(accepted)
        
        print(f"  Step {step}: kv_len={kv_len}, bonus='{bonus_text}'({bonus_token})")
        print(f"    Draft[0:5]: {draft_tokens[:5]} = '{draft_text}'")
        print(f"    Accepted({len(accepted)}): {accepted} = '{accepted_text}'")
        print(f"    num_accepted={num_accepted}")

        last_token = accepted[-1]

    print(f"\n  Overall: {total_accepted}/{total_draft} = {total_accepted/max(total_draft,1):.1%}")

    # Check draft KV cache state
    print("\n[6] Draft KV cache state:")
    for name, kv in engine._draft_kv_caches.items():
        nonzero = (kv != 0).sum().item()
        total = kv.numel()
        print(f"  {name}: shape={kv.shape}, nonzero={nonzero}/{total} ({nonzero/total:.1%})")
        break  # just check first layer

    backend.reset_slot(slot)
    print("\nDone.")

if __name__ == "__main__":
    main()
