"""Profile plan vs replay by monkey-patching CG methods."""
import os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

def build_vllm_config():
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
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

timings = {}

def timed(name, fn):
    """Wrap fn to record fill/plan/replay breakdown."""
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.setdefault(name, []).append((t1-t0)*1000)
        return result
    return wrapper

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
    prompt = make_prompt(tokenizer, 4096)
    print("Warmup + CG capture...")
    engine.generate(prompt, max_tokens=64)
    print(f"CG: verify={engine._verify_cg is not None}, draft={engine._draft_cg is not None}")

    # Monkey-patch to time individual methods
    vcg = engine._verify_cg
    dcg = engine._draft_cg
    dcg_decode = engine._cuda_graph

    orig_v_fill = vcg._fill_buffers
    orig_v_plan = vcg._run_plan
    orig_v_replay = vcg._graph.replay
    
    orig_d_fill = dcg._fill_buffers
    orig_d_plan = dcg._run_plan
    orig_d_replay = dcg._graph.replay

    orig_dec_replay_fn = dcg_decode.replay

    vcg._fill_buffers = timed("v_fill", orig_v_fill)
    vcg._run_plan = timed("v_plan", orig_v_plan)
    vcg._graph.replay = timed("v_replay", orig_v_replay)

    dcg._fill_buffers = timed("d_fill", orig_d_fill)
    dcg._run_plan = timed("d_plan", orig_d_plan)
    dcg._graph.replay = timed("d_replay", orig_d_replay)

    dcg_decode.replay = timed("dec_total", orig_dec_replay_fn)

    # Run measured steps
    print("\nRunning 20 measured steps (4K context)...")
    prompt = make_prompt(tokenizer, 4096)
    _, stats = engine.generate(prompt, max_tokens=200)
    print(f"  accept={stats['acceptance_rate']:.1%}, tok/s={stats['tok_per_s']:.1f}")

    # Print breakdown
    print(f"\n{'Component':<15} {'Mean(ms)':<10} {'Min':<10} {'Max':<10} {'Count':<6}")
    print("-" * 55)
    for key in sorted(timings.keys()):
        vals = timings[key]
        mean = sum(vals)/len(vals)
        print(f"{key:<15} {mean:<10.2f} {min(vals):<10.2f} {max(vals):<10.2f} {len(vals):<6}")

    # Summary
    v_total = sum(sum(timings.get(k, [0]))/len(timings.get(k, [1])) for k in ["v_fill", "v_plan", "v_replay"])
    d_total = sum(sum(timings.get(k, [0]))/len(timings.get(k, [1])) for k in ["d_fill", "d_plan", "d_replay"])
    dec_total = sum(timings.get("dec_total", [0]))/len(timings.get("dec_total", [1]))
    print(f"\nVerify total: {v_total:.2f}ms (fill={sum(timings['v_fill'])/len(timings['v_fill']):.2f} + plan={sum(timings['v_plan'])/len(timings['v_plan']):.2f} + replay={sum(timings['v_replay'])/len(timings['v_replay']):.2f})")
    print(f"Draft total:  {d_total:.2f}ms (fill={sum(timings['d_fill'])/len(timings['d_fill']):.2f} + plan={sum(timings['d_plan'])/len(timings['d_plan']):.2f} + replay={sum(timings['d_replay'])/len(timings['d_replay']):.2f})")
    print(f"Decode total: {dec_total:.2f}ms")

if __name__ == "__main__":
    main()
