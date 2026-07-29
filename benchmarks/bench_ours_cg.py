"""BlackweLLM benchmark with CUDA Graph enabled."""
import json, os, sys, time, subprocess
os.environ["USE_LIBUV"]="0"; os.environ["HF_HUB_OFFLINE"]="1"
os.environ["QSR_DFLASH_CUDA_GRAPH"]="1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch; torch.set_grad_enabled(False)

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/07614121b31898586430f189d27a25a0be310843/")
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

from runtime.legacy_qwen36_vllm import EngineArgs
total_len = len(full_ids)
max_len = max(total_len + 1024, 262144)
bps = (total_len + 15) // 16 + 512
print(f"Config: base={base_len}, suffix={suffix_len}, bps={bps}, max_len={max_len}, gpu_util={gpu_util}", file=sys.stderr)
vc = EngineArgs(model=MODEL, dtype="bfloat16", max_model_len=max_len,
    gpu_memory_utilization=gpu_util, enforce_eager=True, trust_remote_code=True).create_engine_config()
from runtime.backends.laguna import LagunaBackend
backend = LagunaBackend(vc, num_slots=1, block_size=64, blocks_per_slot=bps)
from runtime.backends.laguna_dflash import DFlashEngine
engine = DFlashEngine(backend)

# JIT warmup + CG capture
backend.reset_slot(0)
engine.generate(base_ids[:256], max_tokens=5, temperature=0.0, slot=0, enable_prefix_cache=False)
torch.cuda.synchronize()
print(f"Init done, mem={mem()}GiB, cg={engine._cg_captured}", file=sys.stderr)

results = dict(runtime="blackwellm", base_ctx=base_len, suffix=suffix_len, cg=True)

# Phase 1: warmup (full prefill base, populates prefix cache)
print(f"  [warmup] {base_len} tokens...", file=sys.stderr)
backend.reset_slot(0)
for kv_tensor in engine._draft_kv_caches.values(): kv_tensor.zero_()
t0 = time.perf_counter()
tw, sw = engine.generate(base_ids, max_tokens=max_new, temperature=0.0, slot=0, enable_prefix_cache=False)
torch.cuda.synchronize()
t1 = time.perf_counter() - t0
results["warmup"] = dict(phase="warmup", prompt_len=base_len, n_tokens=len(tw),
    total_s=round(t1,2), prefill_ms=round(sw["prefill_ms"],1), decode_ms=round(sw["decode_ms"],1),
    accept=round(sw["acceptance_rate"],3), tok_step=round(sw["tokens_per_step"],2),
    tok_s=round(sw["tok_per_s"],1), mem_gib=mem())
print(f"    {t1:.1f}s pf={sw['prefill_ms']/1000:.1f}s dec={sw['tok_per_s']:.0f}t/s acc={sw['acceptance_rate']:.0%} mem={mem()}GiB", file=sys.stderr)

# Phase 2: prefix cached (base + suffix, prefix cache HIT)
print(f"  [cached] {base_len}+{suffix_len} tokens...", file=sys.stderr)
t0 = time.perf_counter()
tc, sc = engine.generate(full_ids, max_tokens=max_new, temperature=0.0, slot=0, enable_prefix_cache=True)
torch.cuda.synchronize()
t2 = time.perf_counter() - t0
results["prefix_cached"] = dict(phase="prefix_cached", prompt_len=len(full_ids),
    cached_len=base_len, suffix_len=suffix_len, n_tokens=len(tc),
    total_s=round(t2,2), prefill_ms=round(sc["prefill_ms"],1), decode_ms=round(sc["decode_ms"],1),
    accept=round(sc["acceptance_rate"],3), tok_step=round(sc["tokens_per_step"],2),
    tok_s=round(sc["tok_per_s"],1), mem_gib=mem(), preview=tok.decode(tc[:50])[:80])
print(f"    {t2:.1f}s pf={sc['prefill_ms']/1000:.1f}s dec={sc['tok_per_s']:.0f}t/s acc={sc['acceptance_rate']:.0%} mem={mem()}GiB", file=sys.stderr)

# Phase 3: cold (full prefill, no cache)
print(f"  [cold] {len(full_ids)} tokens...", file=sys.stderr)
backend.reset_slot(0)
for kv_tensor in engine._draft_kv_caches.values(): kv_tensor.zero_()
t0 = time.perf_counter()
tk, sk = engine.generate(full_ids, max_tokens=max_new, temperature=0.0, slot=0, enable_prefix_cache=False)
torch.cuda.synchronize()
t3 = time.perf_counter() - t0
results["cold"] = dict(phase="cold", prompt_len=len(full_ids), n_tokens=len(tk),
    total_s=round(t3,2), prefill_ms=round(sk["prefill_ms"],1), decode_ms=round(sk["decode_ms"],1),
    accept=round(sk["acceptance_rate"],3), tok_step=round(sk["tokens_per_step"],2),
    tok_s=round(sk["tok_per_s"],1), mem_gib=mem())
print(f"    {t3:.1f}s pf={sk['prefill_ms']/1000:.1f}s dec={sk['tok_per_s']:.0f}t/s acc={sk['acceptance_rate']:.0%} mem={mem()}GiB", file=sys.stderr)

print(json.dumps(results))
