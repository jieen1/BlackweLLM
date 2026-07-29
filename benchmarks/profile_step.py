"""Profile each component of speculative_decode_step."""
import os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

def build_vllm_config():
from runtime.legacy_qwen36_vllm import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    return EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
    ).create_engine_config()

def make_prompt(tokenizer, n):
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < n:
        tokens.extend(chunk)
    return tokens[:n]

def main():
    print("Loading model...")
    vllm_config = build_vllm_config()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=4352)
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser("~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/07614121b31898586430f189d27a25a0be310843/"),
        trust_remote_code=True,
    )

    # Warmup + CG capture
    prompt = make_prompt(tokenizer, 4096)
    print("Warmup generate (CG capture)...")
    engine.generate(prompt, max_tokens=64)
    print(f"CG captured: verify={engine._verify_cg is not None}, draft={engine._draft_cg is not None}")

    # Profile individual steps
    print("\nProfiling speculative_decode_step components (4K context, 20 steps)...")
    prompt = make_prompt(tokenizer, 4096)
    slot = 0
    backend.reset_slot(slot)
    for kv in engine._draft_kv_caches.values():
        kv.zero_()
    torch.cuda.empty_cache()

    first_token, aux_hs = backend.prefill_with_aux(slot, prompt)
    if aux_hs is not None:
        aux_len = aux_hs[0].shape[0]
        engine._bulk_precompute_context_kv(slot, aux_hs, aux_len, len(prompt) - aux_len)
    del aux_hs
    torch.cuda.empty_cache()

    # Warmup a few steps
    last_token = first_token
    for _ in range(3):
        accepted = engine.speculative_decode_step(slot, last_token)
        last_token = accepted[-1]

    # Timed steps
    n_steps = 20
    torch.cuda.synchronize()

    # Profile each component
    times = {"decode_cg": [], "combine": [], "draft": [], "verify": [], "accept": [], "total": []}

    for _ in range(n_steps):
        kv_len = backend.slot_kv_len[slot]
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: Decode CG
        if engine._cuda_graph is not None:
            next_tokens, aux = engine._cuda_graph.replay_with_aux([slot], [last_token], [kv_len])
            bonus_token = next_tokens[0]
        else:
            logits, aux = engine._forward_main_with_aux([slot], [last_token], [kv_len], qo_len=1)
            bonus_token = int(logits[0].argmax(dim=-1).item())
        backend.slot_kv_len[slot] += 1
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Step 2: Combine + precompute
        if aux is not None:
            combined_input = torch.cat(aux, dim=-1)
            combined = engine.draft_model.combine_hidden_states(combined_input)
            engine._precompute_context_kv(slot, combined, kv_len)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Step 3: Draft
        if engine._draft_cg is not None:
            draft_tokens = engine._draft_cg.replay(slot, bonus_token, kv_len + 1)
        else:
            draft_tokens = engine._draft_forward(slot, bonus_token, kv_len + 1)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Step 4: Verify
        if engine._verify_cg is not None:
            verify_tokens = [bonus_token] + draft_tokens
            verify_logits = engine._verify_cg.replay(slot, verify_tokens, kv_len + 1)
            accepted, num_accepted = engine._accept_reject(verify_logits, draft_tokens, bonus_token)
        else:
            accepted, num_accepted = engine._verify(slot, bonus_token, draft_tokens, kv_len + 1)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        # Update state
        backend.slot_kv_len[slot] += num_accepted
        for tok in accepted:
            backend.slot_committed_tokens[slot].append(tok)
        last_token = accepted[-1]
        t5 = time.perf_counter()

        times["decode_cg"].append((t1-t0)*1000)
        times["combine"].append((t2-t1)*1000)
        times["draft"].append((t3-t2)*1000)
        times["verify"].append((t4-t3)*1000)
        times["accept"].append((t5-t4)*1000)
        times["total"].append((t5-t0)*1000)

    print(f"\n{'Component':<15} {'Mean(ms)':<10} {'Min':<10} {'Max':<10} {'%Total':<8}")
    print("-" * 55)
    total_mean = sum(sum(v)/len(v) for v in times.values() if v)
    for key in ["decode_cg", "combine", "draft", "verify", "accept", "total"]:
        vals = times[key]
        mean = sum(vals)/len(vals)
        pct = mean/total_mean*100 if key != "total" else 100
        print(f"{key:<15} {mean:<10.2f} {min(vals):<10.2f} {max(vals):<10.2f} {pct:<8.1f}")

    backend.reset_slot(slot)

if __name__ == "__main__":
    main()
