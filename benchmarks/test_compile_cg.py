"""Test: torch.compile (vLLM pipeline) + our custom CG capture.

Expected: ~4-8ms savings from elementwise fusion (1.8ms → ~0.2ms)
"""
import os, sys, time, json
os.environ['USE_LIBUV'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE', '1')
sys.path.insert(0, '/home/bot/project/qwen-sm120-runtime')

import torch
torch.set_grad_enabled(False)

from oracle.qwen36_vllm.vllm_compat import EngineArgs

MODEL = os.path.expanduser(
    '~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/'
    'snapshots/07614121b31898586430f189d27a25a0be310843/')

def main():
    max_len = 70000
    gpu_util = 0.90
    
    print("=" * 60)
    print("TEST: torch.compile + custom CG capture")
    print("=" * 60)
    
    # Configure: compile ON, vLLM CG OFF (we do our own)
    from vllm.config import CompilationConfig
    comp_config = CompilationConfig(
        mode=3,  # VLLM_COMPILE
        cudagraph_mode=0,  # NONE - we capture our own CG
    )
    
    engine_args = EngineArgs(
        model=MODEL,
        dtype='bfloat16',
        max_model_len=max_len,
        gpu_memory_utilization=gpu_util,
        enforce_eager=False,  # Allow compilation
        trust_remote_code=True,
        compilation_config=comp_config,
    )
    
    print("\n[1] Creating engine config...")
    t0 = time.time()
    vllm_config = engine_args.create_engine_config()
    print(f"    Config created in {time.time()-t0:.1f}s")
    print(f"    compilation mode: {vllm_config.compilation_config.mode}")
    print(f"    cudagraph_mode: {vllm_config.compilation_config.cudagraph_mode}")
    
    # Load model via our backend
    from runtime.backends.laguna import LagunaBackend
    
    print("\n[2] Loading model via LagunaBackend...")
    t0 = time.time()
    backend = LagunaBackend(
        vllm_config, num_slots=1, block_size=64, blocks_per_slot=1088)
    load_time = time.time() - t0
    print(f"    Model loaded in {load_time:.1f}s")
    
    # Prefill a 64K prompt
    print("\n[3] Prefilling 64K tokens...")
    prompt_ids = list(range(1000, 1000 + 65536))
    t0 = time.time()
    first_token = backend.prefill(0, prompt_ids)
    prefill_time = time.time() - t0
    kv_len = backend.slot_kv_len[0]
    print(f"    Prefill done: {prefill_time:.2f}s, first_token={first_token}, kv_len={kv_len}")
    
    # Capture CG
    print("\n[4] Capturing CUDA Graph...")
    t0 = time.time()
    backend._ensure_decode_cg()
    cg_time = time.time() - t0
    
    if backend._decode_cg is None:
        print("    CG capture FAILED! Falling back to eager decode.")
        # Benchmark eager decode
        print("\n[5] Benchmarking EAGER decode (100 steps)...")
        tok = first_token
        cur_kv = kv_len
        for _ in range(10):
            tok = backend.decode(0, tok)
            cur_kv += 1
        torch.cuda.synchronize()
        t0 = time.time()
        n_steps = 100
        for _ in range(n_steps):
            tok = backend.decode(0, tok)
            cur_kv += 1
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        step_ms = elapsed / n_steps * 1000
        tok_per_s = n_steps / elapsed
        print(f"    EAGER: {step_ms:.2f} ms/step, {tok_per_s:.1f} tok/s")
        return
    
    print(f"    CG captured in {cg_time:.1f}s")
    
    # Benchmark decode
    print("\n[5] Benchmarking CG decode (100 steps at 64K)...")
    cg = backend._decode_cg
    tok = first_token
    cur_kv = kv_len
    
    # Warmup
    for _ in range(10):
        results = cg.replay([0], [tok], [cur_kv])
        tok = results[0]
        cur_kv += 1
    
    torch.cuda.synchronize()
    t0 = time.time()
    n_steps = 100
    for _ in range(n_steps):
        results = cg.replay([0], [tok], [cur_kv])
        tok = results[0]
        cur_kv += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    step_ms = elapsed / n_steps * 1000
    tok_per_s = n_steps / elapsed
    
    print(f"\n{'=' * 60}")
    print("RESULTS (torch.compile + CG, 64K context):")
    print(f"  Step latency: {step_ms:.2f} ms")
    print(f"  Throughput:   {tok_per_s:.1f} tok/s")
    print(f"  Prefill:      {prefill_time:.2f}s")
    print(f"  Load time:    {load_time:.1f}s")
    print(f"  CG capture:   {cg_time:.1f}s")
    print(f"{'=' * 60}")
    
    # Compare with baseline
    baseline_step = 14.65  # From previous session (eager + CG, 64K)
    baseline_tps = 68.3
    print(f"\n  vs baseline (eager+CG): {baseline_step:.2f}ms → {step_ms:.2f}ms "
          f"({(baseline_step-step_ms)/baseline_step*100:.1f}% faster)")
    print(f"  vs baseline tok/s: {baseline_tps:.1f} → {tok_per_s:.1f}")
    
    # Save results
    results = {
        "config": "torch.compile(VLLM_COMPILE) + custom CG",
        "context_len": 65536,
        "step_ms": round(step_ms, 2),
        "tok_per_s": round(tok_per_s, 1),
        "prefill_s": round(prefill_time, 2),
        "load_s": round(load_time, 1),
        "cg_capture_s": round(cg_time, 1),
        "baseline_step_ms": baseline_step,
        "baseline_tok_per_s": baseline_tps,
    }
    out_path = "benchmarks/fixtures/compile_cg_bench.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
