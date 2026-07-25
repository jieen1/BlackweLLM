"""vLLM benchmark — identical parameters to BlackForge bench."""
import json, os, sys, time, subprocess

def main():
    os.environ["USE_LIBUV"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch; torch.set_grad_enabled(False)

    MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/07614121b31898586430f189d27a25a0be310843/")
    DRAFT = os.path.expanduser("~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/snapshots/723794750422b3efbf3a7b3af76dffb4ba035943/")
    base_len = int(sys.argv[1]); suffix_len = int(sys.argv[2])
    max_new = int(sys.argv[3]); gpu_util = float(sys.argv[4])

    def mem():
        try:
            o = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
            return round(float(o.strip().split("\n")[0])/1024,1)
        except: return -1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base_text = ("The quick brown fox jumps over the lazy dog. "
        "In a world of artificial intelligence and machine learning, "
        "the importance of efficient inference cannot be overstated. "
        "Modern language models require careful optimization of memory "
        "bandwidth and compute utilization to achieve real-time performance. "
        "Speculative decoding offers a promising approach by using a smaller "
        "draft model to propose multiple tokens in parallel, which are then "
        "verified by the larger target model in a single forward pass. ")
    chunk = tok.encode(base_text, add_special_tokens=False)
    base_ids = []
    while len(base_ids) < base_len: base_ids.extend(chunk)
    base_ids = base_ids[:base_len]
    suffix_ids = [(t+50000)%100352 for t in chunk*(suffix_len//len(chunk)+1)]
    suffix_ids = suffix_ids[:suffix_len]
    full_ids = base_ids + suffix_ids
    base_prompt = tok.decode(base_ids)
    full_prompt = tok.decode(full_ids)

    from vllm import LLM, SamplingParams
    max_len = len(full_ids) + 2048  # tight fit to avoid OOM
    print(f"  Initializing vLLM (max_len={max_len}, gpu_util={gpu_util})...", file=sys.stderr)
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=gpu_util, trust_remote_code=True,
              enforce_eager=False, enable_prefix_caching=True,
              speculative_config=dict(method="dflash", num_speculative_tokens=15, model=DRAFT))
    init_s = time.perf_counter() - t0
    print(f"  vLLM init: {init_s:.0f}s, mem={mem()}GiB", file=sys.stderr)

    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    llm.generate(["Hello"], sp)

    results = dict(runtime="vllm", base_ctx=base_len, suffix=suffix_len, init_s=round(init_s,1))

    # Phase 1: warmup (full prefill base, populates prefix cache)
    print(f"  [warmup] {base_len} tokens...", file=sys.stderr)
    t0 = time.perf_counter()
    o1 = llm.generate([base_prompt], sp); torch.cuda.synchronize()
    t1 = time.perf_counter() - t0
    n1 = len(o1[0].outputs[0].token_ids)
    results["warmup"] = dict(phase="warmup", prompt_len=base_len, n_tokens=n1,
        total_s=round(t1,2), tok_s=round(n1/max(t1,0.01),1), mem_gib=mem())
    print(f"    {t1:.1f}s {n1}tok {n1/max(t1,0.01):.0f}t/s mem={mem()}GiB", file=sys.stderr)

    # Phase 2: prefix cached (base + suffix, prefix cache HIT)
    print(f"  [cached] {len(full_ids)} tokens (prefix={base_len})...", file=sys.stderr)
    t0 = time.perf_counter()
    o2 = llm.generate([full_prompt], sp); torch.cuda.synchronize()
    t2 = time.perf_counter() - t0
    n2 = len(o2[0].outputs[0].token_ids)
    results["prefix_cached"] = dict(phase="prefix_cached", prompt_len=len(full_ids),
        cached_len=base_len, suffix_len=suffix_len, n_tokens=n2,
        total_s=round(t2,2), tok_s=round(n2/max(t2,0.01),1), mem_gib=mem(),
        preview=o2[0].outputs[0].text[:80])
    print(f"    {t2:.1f}s {n2}tok {n2/max(t2,0.01):.0f}t/s mem={mem()}GiB", file=sys.stderr)

    # Phase 3: cold (evict cache, full prefill)
    print(f"  [cold] {len(full_ids)} tokens...", file=sys.stderr)
    llm.generate(["Completely different prompt to evict cache. " * 200], sp)
    t0 = time.perf_counter()
    o3 = llm.generate([full_prompt], sp); torch.cuda.synchronize()
    t3 = time.perf_counter() - t0
    n3 = len(o3[0].outputs[0].token_ids)
    results["cold"] = dict(phase="cold", prompt_len=len(full_ids), n_tokens=n3,
        total_s=round(t3,2), tok_s=round(n3/max(t3,0.01),1), mem_gib=mem())
    print(f"    {t3:.1f}s {n3}tok {n3/max(t3,0.01):.0f}t/s mem={mem()}GiB", file=sys.stderr)

    print(json.dumps(results))

if __name__ == "__main__":
    main()
