# FA4 SM120 Optimization Roadmap

## Current Best Performance
- fox-64K: 365 tok/s (+19.3% vs baseline 306)
- galaxy-4K: 403 tok/s (+18.5%)
- code-4K: 97.8% accept (quality preserved)
- Memory: 73.2 GB (-3 GB)

## Applied Optimizations
1. SparkInfer analytic decode path TP=1 unlock (gating-only)
2. FP8 PV verify-only (decoupled from FP8 QK)
3. chunk_pages=32 (+2-5% at 4K)
4. blocks_per_slot=2064 (-3GB memory)

## FA4 Techniques Already in SparkInfer
- TMA loads (producer warp) ✅
- Warp specialization (setmaxregister) ✅
- Device-side analytic schedule ✅
- Compact GQA sync ✅
- FP8 PV (verify path) ✅

## Attempted & Failed
| Optimization | Result | Root Cause |
|---|---|---|
| FP8 PV decode path | CG hang | num_warps_kv=4 incompatible with FP8 PV MMA |
| MoE tile 32x128 | CG hang | New kernel variant compilation timeout |
| Verify gate K=15 | -1 to -5% | total_q=16 verify kernel slower than generic extend |
| TURBO + analytic | code-4K 58.6% | FP8 QK precision loss (3-bit mantissa) |
| Context-length guard | catastrophic | Prefill/decode precision mismatch |

## Root Cause: CG Hang = Kernel JIT Compilation
- SparkInfer uses CuTe DSL kernels with JIT compilation
- New kernel variants trigger 5-15 min compilation on first use
- Disk cache at ~/.cache/sparkinfer/compile/ (255 kernels, 201MB)
- After first compilation, subsequent uses are fast (cache hit)
- NOT a bug - just need to wait for compilation

## Next Steps (Priority Order)
### P0: FP8 PV for decode path (kernel modification)
- Requires modifying forward_paged.py to support FP8 PV with num_warps_kv=4
- The MMA instruction needs to handle 4 KV warps reading from FP8 V
- Estimated: +2-5% at 64K
- Risk: Medium (kernel-level change)

### P1: Prefill/extend FA4 techniques
- Apply TMA + warp specialization to the extend kernel
- Persistent tile scheduler for variable-length prefill
- Estimated: +10-20% prefill speed
- Risk: High (major kernel restructuring)

### P2: TURBO quality fix (kernel-level)
- Per-head Q scale for FP8 QK MMA
- Hadamard rotation (incoherent processing)
- Estimated: +6% at 64K with quality preserved
- Risk: High (kernel-level change, quality-sensitive)

### P3: Dense GEMM FP8 with fused quantization
- Fuse FP8 quantization into RMSNorm kernel
- Only beneficial for large-N shapes (o_proj, lm_head)
- Estimated: +3-5% total
- Risk: Medium (Triton kernel modification)

## SM120 Hardware Facts
- NO BF16/FP16 tensor core (SM80 mma.sync fallback)
- NO wgmma, NO tcgen05, NO TMEM
- Native MMA: FP8 m16n8k32, NVFP4 m16n8k64
- TMA: YES (SM90-style)
- SMEM: 99KB opt-in
- 188 SMs, 1.79 TB/s bandwidth
