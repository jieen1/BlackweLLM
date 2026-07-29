"""Per-stage step latency profiler for DFlash speculative decoding.

Measures exact timing of each stage in the speculative decode step:
1. Main decode CG (replay_with_aux)
2. Combine hidden states + precompute context KV
3. Draft CG (replay)
4. Verify CG (replay)
5. Accept/reject + bookkeeping

Usage:
    USE_LIBUV=0 /home/bot/.venvs/vllm/bin/python -m benchmarks.diag_step_latency
"""
import os, sys, time, torch, json
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

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
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < 65536:
        tokens.extend(chunk)
    prompt = tokens[:65536]

    # Warmup (captures CGs)
    print("Warmup...", flush=True)
    out, stats = engine.generate(prompt, max_tokens=64)
    print(f"  accept={stats['acceptance_rate']:.1%} tok/s={stats['tok_per_s']:.1f}", flush=True)
    print(f"  Draft CG: {engine._draft_cg is not None}, Verify CG: {engine._verify_cg is not None}", flush=True)

    # Profile individual steps
    print("\nProfiling 50 speculative decode steps at 64K...", flush=True)
    slot = 0
    backend.reset_slot(slot)
    first_token = backend.prefill(slot, prompt)
    kv_len = backend.slot_kv_len[slot]
    last_token = backend.slot_committed_tokens[slot][-1]

    # Timing accumulators
    timings = {"decode_cg": [], "combine_precompute": [], "draft_cg": [],
               "verify_cg": [], "accept_reject": [], "total_step": []}

    num_steps = 50
    for step in range(num_steps):
        kv_len = backend.slot_kv_len[slot]
        torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        # Stage 1: Main decode CG
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if engine._cuda_graph is not None:
            next_tokens, aux_hidden_states = engine._cuda_graph.replay_with_aux(
                [slot], [last_token], [kv_len]
            )
            bonus_token = next_tokens[0]
        else:
            logits, aux_hidden_states = engine._forward_main_with_aux(
                [slot], [last_token], [kv_len], qo_len=1
            )
            bonus_token = int(logits[0].argmax(dim=-1).item())
        torch.cuda.synchronize()
        timings["decode_cg"].append(time.perf_counter() - t0)
        backend.slot_kv_len[slot] += 1

        # Stage 2: Combine + precompute context KV
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if aux_hidden_states is not None:
            combined_input = torch.cat(aux_hidden_states, dim=-1)
            combined = engine.draft_model.combine_hidden_states(combined_input)
            engine._precompute_context_kv(slot, combined, kv_len)
        torch.cuda.synchronize()
        timings["combine_precompute"].append(time.perf_counter() - t0)

        # Stage 3: Draft CG
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if engine._draft_cg is not None:
            draft_tokens = engine._draft_cg.replay(slot, bonus_token, kv_len + 1)
        else:
            draft_tokens = engine._draft_forward(slot, bonus_token, kv_len + 1)
        torch.cuda.synchronize()
        timings["draft_cg"].append(time.perf_counter() - t0)

        # Stage 4: Verify CG
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if engine._verify_cg is not None:
            verify_tokens = [bonus_token] + draft_tokens
            verify_logits = engine._verify_cg.replay(slot, verify_tokens, kv_len + 1)
        else:
            verify_logits = None
        torch.cuda.synchronize()
        timings["verify_cg"].append(time.perf_counter() - t0)

        # Stage 5: Accept/reject
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if verify_logits is not None:
            accepted, num_accepted = engine._accept_reject(
                verify_logits, draft_tokens, bonus_token
            )
        else:
            accepted = [bonus_token]
            num_accepted = 0
        torch.cuda.synchronize()
        timings["accept_reject"].append(time.perf_counter() - t0)

        # Update state
        backend.slot_kv_len[slot] += num_accepted
        for tok in accepted:
            backend.slot_committed_tokens[slot].append(tok)
        last_token = accepted[-1] if accepted else bonus_token

        torch.cuda.synchronize()
        timings["total_step"].append(time.perf_counter() - t_total_start)

    # Report
    print(f"\n{'='*60}")
    print(f"STEP LATENCY BREAKDOWN (64K context, {num_steps} steps)")
    print(f"{'='*60}")
    print(f"{'Stage':<25} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'% Total':>8}")
    print(f"{'-'*63}")
    total_mean = sum(timings['total_step']) / num_steps * 1000
    for stage in ["decode_cg", "combine_precompute", "draft_cg", "verify_cg", "accept_reject"]:
        vals = sorted(timings[stage])
        mean_ms = sum(vals) / len(vals) * 1000
        p50_ms = vals[len(vals)//2] * 1000
        p95_ms = vals[int(len(vals)*0.95)] * 1000
        pct = mean_ms / total_mean * 100
        print(f"  {stage:<23} {mean_ms:>10.2f} {p50_ms:>10.2f} {p95_ms:>10.2f} {pct:>7.1f}%")
    print(f"{'-'*63}")
    print(f"  {'TOTAL':<23} {total_mean:>10.2f} {sorted(timings['total_step'])[num_steps//2]*1000:>10.2f} "
          f"{sorted(timings['total_step'])[int(num_steps*0.95)]*1000:>10.2f} {'100.0':>7}%")

    # Save
    results = {stage: {"mean_ms": sum(v)/len(v)*1000, "p50_ms": sorted(v)[len(v)//2]*1000,
                       "p95_ms": sorted(v)[int(len(v)*0.95)]*1000}
               for stage, v in timings.items()}
    out_path = "benchmarks/fixtures/diag_step_latency.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "context": 65536,
                   "num_steps": num_steps, "stages": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
