# Speed Optimization Status — 2026-07-26

## Current Performance (vLLM 0.26.0, commit 4e99b7c)

| Config | tok/s | ms/step | Notes |
|--------|-------|---------|-------|
| M=1 decode CG, 64K | **67.5** | 14.8 | After fixes |
| M=1 decode CG, 64K (before) | 64.8 | 15.4 | Before fixes |
| Historical best (vLLM 0.25.1) | 80.4 | 12.45 | commit 66d5913 |

## Fixes Applied This Session

### 1. RMSNorm Triton Patch Fixed (laguna.py:1423)
- **Bug**: vLLM 0.26.0 `CustomOp.__init__` binds `_forward_method` at construction time.
  Class-level `RMSNorm.forward_cuda = _triton_forward` had NO effect on existing instances.
- **Fix**: After class patch, iterate all 193 RMSNorm instances and rebind `_forward_method`.
- **Impact**: Triton fused norm+residual now active. Neutral for M=1 (0.36ms vs 0.20ms C++),
  but essential for M>1 (DFlash verify) where fusion saves memory bandwidth.

### 2. KV Scatter → C++ reshape_and_cache_flash (bf_attention.py, laguna_cuda_graph.py, laguna_sparkinfer_attn.py)
- **Bug**: Manual Python indexing (6 ops × 48 layers = 288 kernels/step, ~1.1ms).
- **Fix**: Single C++ kernel per layer (48 kernels/step, ~0.13ms).
- **Impact**: **-0.97ms/step** (biggest single win).
- **Prerequisite**: KV layout changed to `[2, num_blocks, block_size, heads, dim]`
  so `kv_cache[0]`/`[1]` are contiguous (required by C++ op).

## Profiler Breakdown (M=1 decode CG, 64K, per step)

```
GPU compute:           13.03 ms  (84%)
CPU fill_buffers:       1.36 ms  (9%)
.item() D2H sync:       0.18 ms  (1%)
Other overhead:         1.01 ms  (6%)
─────────────────────────────────────
Total wall time:       15.58 ms  → 64.2 tok/s
```

### GPU Kernel Breakdown (per step)
| Kernel | Time | Calls | % |
|--------|------|-------|---|
| sparkinfer MoE | 2.29ms | 47 | 18% |
| cutlass_80_wmma_bf16 (dense GEMM) | 2.28ms | 48 | 18% |
| cutlass_80_wmma_s1616 (dense GEMM) | 2.17ms | 49 | 17% |
| gemvx (GEMV) | 1.99ms | 239 | 15% |
| sparkinfer attention | 1.59ms | 48+48 | 12% |
| Triton fused norm | 0.36ms | 193 | 3% |
| RoPE | 0.18ms | 48 | 1% |
| reshape_and_cache_flash | 0.13ms | 48 | 1% |
| MoE topkGating | 0.23ms | 47 | 2% |
| Other elementwise | ~1.8ms | ~500 | 14% |

## Root Cause of Remaining Gap (13ms GPU vs 11ms target)

**cuBLAS selects SM80 CUTLASS WMMA kernels on SM120 hardware.**

Evidence:
- Profiler shows `cutlass_80_wmma_tensorop_bf16_32x32` (SM80 Ampere kernel)
- Isolated F.linear test: 3072→8192 takes 19.9μs
- In-model same operation: 47.5μs (2.4× slower)
- `torch.mv` (GEMV) is 1.65× faster than `torch.mm` for M=1, K=N=3072

The model uses vLLM's `QKVParallelLinear` and `RowParallelLinear` which go through
a different code path than `F.linear`, resulting in suboptimal kernel selection.

**Potential saving if fixed: ~2ms/step → 75+ tok/s**

## Next Steps (Priority Order)

1. **P0: Fix dense GEMM kernel selection** (~2ms potential saving)
   - Investigate why QKVParallelLinear/RowParallelLinear use SM80 kernels
   - Try: patch these layers to use F.linear for M=1
   - Try: force cuBLAS algorithm selection via CUBLAS_GEMM_ALGO
   - Escalate to sparkinfer/vLLM team if it's a cuBLAS SM120 heuristic bug

2. **P1: Reduce CPU overhead** (~1ms potential saving)
   - Optimize `_fill_buffers_b1` Python code (SWA ring calculation)
   - Pre-compute page table updates (only changes on block boundary)
   - Consider: batch scalar writes into single H2D copy

3. **P2: Fuse elementwise operations** (~0.5-1ms potential saving)
   - Many small elementwise kernels (residual add, SiLU, etc.)
   - Triton fusion for MoE pre/post processing
   - Requires model code changes

4. **P3: DFlash verify CG** (speculative decoding speedup)
   - Currently runs eager (304ms/step for M=16)
   - With CG + kernel fusion: target ~30ms/step
   - Blocked on P0-P2 (need fast M=16 GEMM)

## Environment
- GPU: RTX PRO 6000 Blackwell (SM120, 188 SMs, 96GB)
- CUDA: 13.3, PyTorch: 2.13.0, vLLM: 0.26.0
- sparkinfer: blackforge-main @ 3fa9b54
