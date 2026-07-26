"""Diagnostic: Compare CG vs eager aux hidden states at increasing kv_len.

Tests whether the decode CUDA graph produces correct aux hidden states
at large context lengths. If aux diverges, that explains the CG
acceptance degradation at long context.

Usage:
    USE_LIBUV=0 /home/bot/.venvs/vllm/bin/python -m benchmarks.diag_cg_aux_drift
"""
import os, sys, time, torch, json
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch.nn.functional as F

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float().unsqueeze(0),
                                b.flatten().float().unsqueeze(0)).item()

def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

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

    print("Loading backend...", flush=True)
    t0 = time.time()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)
    print(f"  loaded in {time.time()-t0:.1f}s, mem={gpu_mem_mb():.0f}MB", flush=True)

    print("Initializing DFlash engine...", flush=True)
    from runtime.backends.laguna_dflash import DFlashEngine
    from runtime.backends.dflash_constants import AUX_LAYER_IDS
    engine = DFlashEngine(backend)
    has_cg = engine._cuda_graph is not None
    print(f"  CG: {has_cg}, Draft CG: {engine._draft_cg is not None}, "
          f"Verify CG: {engine._verify_cg is not None}", flush=True)
    print(f"  mem={gpu_mem_mb():.0f}MB", flush=True)

    if not has_cg:
        print("ERROR: No CG available", flush=True)
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def make_prompt(n):
        base = "The quick brown fox jumps over the lazy dog. "
        tokens = []
        chunk = tokenizer.encode(base, add_special_tokens=False)
        while len(tokens) < n:
            tokens.extend(chunk)
        return tokens[:n]

    # Test at different context lengths
    test_lengths = [4096, 16384, 65536]
    results = []

    for target_len in test_lengths:
        print(f"\n{'='*60}", flush=True)
        print(f"Testing kv_len={target_len}...", flush=True)

        slot = 0
        backend.reset_slot(slot)

        prompt_ids = make_prompt(target_len)

        # Prefill (handles chunking internally)
        t_p = time.time()
        first_token = backend.prefill(slot, prompt_ids)
        prefill_s = time.time() - t_p
        kv_len = backend.slot_kv_len[slot]
        print(f"  Prefill: kv_len={kv_len}, {prefill_s:.1f}s, first_token={first_token}", flush=True)

        last_token = backend.slot_committed_tokens[slot][-1]

        # --- CG decode step ---
        torch.cuda.synchronize()
        t_cg = time.time()
        cg_tokens, cg_aux = engine._cuda_graph.replay_with_aux(
            [slot], [last_token], [kv_len]
        )
        torch.cuda.synchronize()
        cg_ms = (time.time() - t_cg) * 1000
        cg_token = cg_tokens[0]
        cg_aux_copy = [a.clone() for a in cg_aux] if cg_aux else None

        # --- Eager decode step (same input) ---
        torch.cuda.synchronize()
        t_e = time.time()
        eager_logits, eager_aux = engine._forward_main_with_aux(
            [slot], [last_token], [kv_len], qo_len=1
        )
        torch.cuda.synchronize()
        eager_ms = (time.time() - t_e) * 1000
        eager_token = int(eager_logits[0].argmax(dim=-1).item())
        eager_aux_copy = [a.clone() for a in eager_aux] if eager_aux else None

        # Compare
        token_match = (cg_token == eager_token)
        print(f"  Token: CG={cg_token} eager={eager_token} match={token_match}", flush=True)
        print(f"  Time: CG={cg_ms:.1f}ms eager={eager_ms:.1f}ms", flush=True)

        if cg_aux_copy and eager_aux_copy:
            aux_sims = []
            for i, (ca, ea) in enumerate(zip(cg_aux_copy, eager_aux_copy)):
                sim = cosine_sim(ca, ea)
                max_diff = float((ca.float() - ea.float()).abs().max())
                aux_sims.append(sim)
                layer_id = AUX_LAYER_IDS[i] if i < len(AUX_LAYER_IDS) else i
                print(f"  Aux[{i}] layer={layer_id}: cos={sim:.6f} max_diff={max_diff:.4e} "
                      f"norm={ca.float().norm():.1f}", flush=True)
            avg_sim = sum(aux_sims) / len(aux_sims)
            min_sim = min(aux_sims)
            print(f"  => avg_cos={avg_sim:.6f} min_cos={min_sim:.6f}", flush=True)
        else:
            aux_sims = []
            avg_sim = min_sim = 0.0
            print("  WARNING: aux is None!", flush=True)

        results.append({
            "kv_len": target_len,
            "actual_kv_len": kv_len,
            "token_match": token_match,
            "cg_token": cg_token,
            "eager_token": eager_token,
            "cg_ms": round(cg_ms, 1),
            "eager_ms": round(eager_ms, 1),
            "aux_cosines": [round(s, 6) for s in aux_sims],
            "avg_aux_cosine": round(avg_sim, 6),
            "min_aux_cosine": round(min_sim, 6) if aux_sims else 0,
        })

        del cg_aux_copy, eager_aux_copy
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: CG vs Eager decode aux hidden states")
    print(f"{'kv_len':>8} {'match':>6} {'avg_cos':>10} {'min_cos':>10} {'cg_ms':>7} {'eager_ms':>9}")
    for r in results:
        print(f"{r['kv_len']:>8} {str(r['token_match']):>6} {r['avg_aux_cosine']:>10.6f} "
              f"{r['min_aux_cosine']:>10.6f} {r['cg_ms']:>7.1f} {r['eager_ms']:>9.1f}")

    out_path = "benchmarks/fixtures/diag_cg_aux_drift.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
