"""Comprehensive acceptance rate sweep across diverse prompts."""
import time, torch, json, hashlib

SLOT = 0
MAX_TOKENS = 128

def run(prompt_ids, label):
    backend.reset_slot(SLOT)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens, stats = engine.generate_verify_only(
        prompt_ids=prompt_ids, max_tokens=MAX_TOKENS,
        enable_prefix_cache=False, slot=SLOT,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    ar = stats['acceptance_rate']
    tps = stats['tokens_per_step']
    out = tokenizer.decode(tokens[:30])[:60]
    print(f"  {label:30s}  accept={ar*100:5.1f}%  tps={tps:5.2f}  "
          f"tok/s={stats['tok_per_s']:6.1f}  out={out!r}")
    return {"label": label, "accept": ar, "tps": tps, "tok_s": stats['tok_per_s'],
            "prompt_len": len(prompt_ids), "steps": stats['num_steps']}

def encode(text, max_len=None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:max_len] if max_len and len(ids) > max_len else ids

results = []
print("=" * 90)
print("ACCEPTANCE RATE SWEEP")
print("=" * 90)

# --- Category 1: Natural repeating text (different phrases) ---
print("\n[Natural repeating text, ~4K tokens]")
for phrase in [
    "The quick brown fox jumps over the lazy dog. ",
    "In a galaxy far far away, there lived a brave explorer. ",
    "Machine learning models require large datasets for training. ",
]:
    ids = encode(phrase * 2000)[:4096]
    r = run(ids, f"repeat: {phrase[:25]}...")
    results.append(r)

# --- Category 2: Real instructions / QA ---
print("\n[Real instructions / QA, ~1K tokens]")
for text in [
    "Explain the theory of relativity in simple terms. " * 50,
    "Write a Python function to sort a list using quicksort algorithm. " * 50,
    "What are the main differences between TCP and UDP protocols? " * 50,
    "Describe the process of photosynthesis step by step in detail. " * 50,
]:
    ids = encode(text)[:2048]
    r = run(ids, f"qa: {text[:30]}...")
    results.append(r)

# --- Category 3: Code ---
print("\n[Code, ~2K tokens]")
code = """
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x
""" * 30
ids = encode(code)[:4096]
r = run(ids, "code: transformer")
results.append(r)

# --- Category 4: Chinese text ---
print("\n[Chinese text, ~2K tokens]")
cn = "人工智能正在改变世界的方方面面，从医疗到教育，从交通到金融。" * 200
ids = encode(cn)[:4096]
r = run(ids, "chinese repeat")
results.append(r)

cn2 = "请详细解释量子计算的基本原理，包括量子比特、叠加态和纠缠态的概念。" * 100
ids = encode(cn2)[:2048]
r = run(ids, "chinese qa")
results.append(r)

# --- Category 5: Synthetic token IDs (benchmark style) ---
print("\n[Synthetic token IDs]")
import itertools
for n in [512, 4096]:
    ids = list(itertools.islice(itertools.cycle(range(1000, 1100)), n))
    r = run(ids, f"ids-1000-1100 @{n}")
    results.append(r)

ids = list(itertools.islice(itertools.cycle(range(5000, 5100)), 4096))
r = run(ids, "ids-5000-5100 @4K")
results.append(r)

ids = list(range(1, 4097))
r = run(ids, "ids-sequential @4K")
results.append(r)

# --- Category 6: Long natural text (64K) ---
print("\n[Long context 64K]")
ids = encode("The quick brown fox jumps over the lazy dog. " * 50000)[:65536]
backend.reset_slot(SLOT)
torch.cuda.synchronize()
t0 = time.perf_counter()
tokens, stats = engine.generate_verify_only(
    prompt_ids=ids, max_tokens=256, enable_prefix_cache=False, slot=SLOT)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
ar = stats['acceptance_rate']
print(f"  {'fox-64K':30s}  accept={ar*100:5.1f}%  tps={stats['tokens_per_step']:5.2f}  "
      f"tok/s={stats['tok_per_s']:6.1f}  wall={wall:.1f}s")
results.append({"label": "fox-64K", "accept": ar, "tps": stats['tokens_per_step'],
                "tok_s": stats['tok_per_s'], "prompt_len": len(ids)})

# --- Summary ---
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
ars = [r['accept'] for r in results]
natural = [r['accept'] for r in results if not r['label'].startswith('ids')]
synthetic = [r['accept'] for r in results if r['label'].startswith('ids')]
print(f"Total prompts: {len(results)}")
print(f"Overall avg:   {sum(ars)/len(ars)*100:.1f}%")
if natural:
    print(f"Natural avg:   {sum(natural)/len(natural)*100:.1f}%  (n={len(natural)})")
if synthetic:
    print(f"Synthetic avg: {sum(synthetic)/len(synthetic)*100:.1f}%  (n={len(synthetic)})")
print(f"Min:           {min(ars)*100:.1f}%  ({[r['label'] for r in results if r['accept']==min(ars)][0]})")
print(f"Max:           {max(ars)*100:.1f}%  ({[r['label'] for r in results if r['accept']==max(ars)][0]})")
