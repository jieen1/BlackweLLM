# Speed Analysis: Why Current DFlash is Slower Than Historical M=1 Decode

**Date**: 2026-07-26
**Status**: Root cause identified, NOT a script bug or config issue

## The Numbers

| Benchmark | Date | Config | tok/s | ms/step | Notes |
|-----------|------|--------|-------|---------|-------|
| `full_benchmark.json` (66d5913) | 07-25 | M=1 decode CG, **NO DFlash** | **80.4** | 12.45 | sparkinfer attn+MoE |
| `laguna_dflash_off.json` | 07-23 | M=1 decode CG, vLLM engine | 54.6 | 18.32 | flashinfer_cutlass MoE |
| `laguna_dflash_on.json` | 07-23 | DFlash, vLLM engine | **367** | 2.72/tok | torch.compile + piecewise CG |
| `vllm_comparison_20260725.json` | 07-25 | DFlash, our engine | 43.9 | — | 33.9% acceptance (bad) |
| `full_comparison_ours.json` | 07-26 | DFlash, our engine | **45-49** | ~304 | 89% acceptance (good) |

## Root Cause: Apples-to-Oranges Comparison

**The 80.4 tok/s and the 45-49 tok/s measure fundamentally different workloads:**

### 80.4 tok/s (commit 66d5913)
- `dflash: false` — NO speculative decoding
- M=1 decode with CUDA Graph replay
- Tight loop: `cg.replay([0], [tok_id], [kv])` × 127 iterations
- ONE graph replay per token, zero Python overhead
- 12.45ms per token

### 45-49 tok/s (current)
- `dflash: true` — speculative decoding with M=16 verify
- Each step: draft 15 tokens (CG) → verify M=16 (EAGER) → accept/reject
- Verify runs `_forward_verify_with_aux()` eagerly — 48 layers × multiple kernels × Python dispatch
- ~304ms per step, producing ~14.2 tokens (89% acceptance)
- 304ms / 14.2 tokens = 21.4ms per token

### Per-token cost comparison
- M=1 CG: 12.45ms/token
- DFlash verify: 21.4ms/token (71% MORE expensive per token!)

## Why DFlash is Negative Value for Us

The math is simple:

```
M=1 CG:     12.45ms → 1 token  → 80.4 tok/s
DFlash:     304ms   → 14.2 tok → 46.7 tok/s

For DFlash to match M=1: 14.2 tokens × 12.45ms = 177ms budget
Actual step time: 304ms → 72% over budget
```

Even with PERFECT verify CG (eliminating all Python overhead):
```
Best case M=16 CG: ~16 × 12.45ms = 199ms (compute scales with M)
14.2 / 0.199 = 71 tok/s → still less than M=1's 80.4
```

DFlash only helps if verify M=16 costs LESS than 16× M=1. This requires kernel fusion.

## Why vLLM Gets 367 tok/s with DFlash

vLLM's execution model is fundamentally different:

| Component | Our Runtime | vLLM |
|-----------|-------------|------|
| Kernel fusion | ❌ None (pure eager) | ✅ torch.compile fuses ~150→~50 kernels |
| CUDA Graph | ✅ Single full graph (M=1 only) | ✅ Piecewise (multiple segments, all M values) |
| Verify M=16 | Eager (Python dispatch per kernel) | Compiled + CG (zero dispatch overhead) |
| Memory bandwidth | 3× wasted (unfused norm/residual) | Optimal (fused reads/writes) |

vLLM's M=16 verify is ~30ms (not 300ms) because:
1. Fused kernels: norm+residual+activation in ONE kernel (saves 2× memory bandwidth)
2. Piecewise CG: zero Python dispatch overhead
3. Attention is memory-bound on KV read (same for M=1 and M=16)
4. MoE with fused quantization: less kernel overhead

## What's NOT the Problem

- ❌ sparkinfer attention kernel: proven 6-11% FASTER than FlashInfer
- ❌ sparkinfer MoE kernel: works correctly, deterministic mode functional
- ❌ Benchmark script methodology: measurements are correct
- ❌ Configuration: parameters match historical runs
- ❌ Acceptance rate: 89% is excellent (was 33% in May, fixed)

## What IS the Problem

1. **No kernel fusion**: Each RMSNorm+residual is 3 separate kernels (load→norm→store, load→mul→store, load→add→store). vLLM fuses these into 1. This wastes ~2× memory bandwidth per layer.

2. **Verify runs eagerly**: M=16 verify dispatches ~480 kernel launches through Python. Each launch has ~20-50μs overhead. Total: ~14ms pure Python overhead per step.

3. **Verify CG disabled**: Was measured 7% slower than eager (42 vs 45 tok/s). This is suspicious — CG should never be slower. Likely a bug in the CG implementation (stale worklist, unnecessary metadata rebuild, etc.)

4. **MoE at M=16 may be suboptimal**: With 256 experts and top_k=10, M=16 produces 160 expert activations. The kernel launch pattern might not be optimal for this batch size.

## Fix Priority

### P0: Run M=1 decode CG without DFlash (immediate, proven)
- Already achieves 80.4 tok/s at 64K
- Just use `full_benchmark.py` path (no DFlash)
- This is our best current performance

### P1: Fix verify CG (should be faster than eager, not slower)
- Investigate why CG was 7% slower than eager
- Likely causes: stale worklist rebuild, unnecessary `_fill_buffers` overhead, metadata copy
- Target: verify CG ≤ eager time

### P2: Kernel fusion for verify path
- Triton fused RMSNorm+residual already exists (commit 073cac8)
- Need to verify it's active in verify path
- Additional fusion: MoE quantization, attention pre/post processing

### P3: torch.compile or equivalent
- Long-term: compile the model forward for M=16
- Eliminates all Python dispatch overhead
- Enables kernel fusion beyond what manual Triton can do

## Verification Plan

When GPU is available:
1. Run `full_benchmark.py` (M=1, no DFlash) → confirm still ~80 tok/s
2. Run DFlash with `QSR_VERIFY_CUDA_GRAPH=1` → profile verify step
3. Run DFlash with `QSR_VERIFY_CUDA_GRAPH=0` → compare
4. Profile M=16 verify with `torch.cuda.Event` timing per component
5. Compare per-layer timing: attention vs MoE vs norms
