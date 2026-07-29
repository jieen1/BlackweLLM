"""Profile a single decode step at 64K to identify all GPU kernels and their timing.

Uses CUDA events to measure total step time, then torch profiler for kernel breakdown.
"""
import os, sys, time, json
os.environ['USE_LIBUV'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE', '1')
sys.path.insert(0, '/home/bot/project/qwen-sm120-runtime')

import torch
torch.set_grad_enabled(False)

from runtime.legacy_qwen36_vllm import EngineArgs

MODEL = os.path.expanduser(
    '~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/'
    'snapshots/07614121b31898586430f189d27a25a0be310843/')

def main():
    engine_args = EngineArgs(
        model=MODEL, dtype='bfloat16', max_model_len=70000,
        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True)
    vllm_config = engine_args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend
    print("[1] Loading model...")
    t0 = time.time()
    backend = LagunaBackend(vllm_config, num_slots=1, block_size=64, blocks_per_slot=1088)
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # Prefill 64K
    print("[2] Prefill 64K...")
    prompt = list(range(1000, 66536))
    t0 = time.time()
    first = backend.prefill(0, prompt)
    print(f"    Prefill: {time.time()-t0:.2f}s, first={first}")
    kv_len = backend.slot_kv_len[0]

    # Capture CG
    print("[3] Capture CG...")
    t0 = time.time()
    backend._ensure_decode_cg()
    print(f"    CG captured in {time.time()-t0:.1f}s")
    cg = backend._decode_cg

    # Warmup
    tok = first
    cur_kv = kv_len
    for _ in range(20):
        results = cg.replay([0], [tok], [cur_kv])
        tok = results[0]
        cur_kv += 1

    # Profile with torch profiler
    print("[4] Profiling 10 decode steps...")
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            results = cg.replay([0], [tok], [cur_kv])
            tok = results[0]
            cur_kv += 1
        torch.cuda.synchronize()

    # Print kernel breakdown
    print("\n" + "=" * 80)
    print("KERNEL BREAKDOWN (10 steps, sorted by total CUDA time)")
    print("=" * 80)
    
    events = prof.key_averages()
    events_sorted = sorted(events, key=lambda e: e.device_time_total, reverse=True)
    
    total_cuda_us = sum(e.device_time_total for e in events_sorted)
    print(f"\nTotal CUDA time: {total_cuda_us/10:.0f} us/step ({total_cuda_us/10/1000:.2f} ms/step)")
    print(f"\n{'Kernel':<60} {'Count':>6} {'Total(us)':>10} {'Per-step(us)':>12} {'%':>6}")
    print("-" * 100)
    
    cumulative = 0
    for e in events_sorted[:40]:
        if e.device_time_total == 0:
            continue
        pct = e.device_time_total / total_cuda_us * 100
        cumulative += pct
        per_step = e.device_time_total / 10
        name = e.key[:58]
        print(f"{name:<60} {e.count:>6} {e.device_time_total:>10.0f} {per_step:>12.0f} {pct:>5.1f}%")
        if cumulative > 95:
            break
    
    # Also measure wall-clock step time
    print("\n[5] Wall-clock measurement (100 steps)...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        results = cg.replay([0], [tok], [cur_kv])
        tok = results[0]
        cur_kv += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"    {elapsed/100*1000:.2f} ms/step, {100/elapsed:.1f} tok/s")
    
    # Save
    results_data = {
        "context_len": cur_kv,
        "step_ms": round(elapsed/100*1000, 2),
        "tok_per_s": round(100/elapsed, 1),
        "total_cuda_us_per_step": round(total_cuda_us/10, 0),
        "top_kernels": [
            {"name": e.key[:80], "per_step_us": round(e.device_time_total/10, 1), 
             "pct": round(e.device_time_total/total_cuda_us*100, 1)}
            for e in events_sorted[:20] if e.device_time_total > 0
        ]
    }
    with open("benchmarks/fixtures/decode_profile_64k.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print("\nSaved to benchmarks/fixtures/decode_profile_64k.json")

if __name__ == "__main__":
    main()
