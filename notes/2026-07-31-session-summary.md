# Session Summary: 2026-07-31 Performance Optimization

## Current Best Configuration
```bash
export QSR_VERIFY_CG_MAX_PAGES=1040
export QSR_VERIFY_CG_CTAS_PER_SM=4
export SPARKINFER_PAGED_DECODE_GRAPH_CHUNK_PAGES=32
# NO SPARKINFER_TURBO_ATTN (quality regression)
```

## Performance Results

### Analytic Decode Path (no TURBO) — RECOMMENDED
| Workload | tok/s | Accept | vs Baseline |
|----------|-------|--------|-------------|
| fox-64K | 353-368 | 96.9% | +15-20% |
| fox-4K | 353-357 | 96.3-97.0% | +2-3% |
| galaxy-4K | 395-401 | 100% | +16-18% |
| code-4K | 341-359 | 97.8% | ✅ quality preserved |

### TURBO=1 (FP8 QK MMA) — quality regression
| Workload | tok/s | Accept | Issue |
|----------|-------|--------|-------|
| fox-64K | 375-383 | 100% | Best 64K perf |
| code-4K | 225-232 | 58.6% | ❌ quality destroyed |

## Key Changes Made
1. **SparkInfer analytic decode path unlocked for TP=1** (gating-only, zero kernel changes)
   - `_forward.py`: Relaxed `num_q_heads==24, num_kv_heads==4, page_size==128` to `gqa_group_size==6, page_size in (64,128)`
   - `planner.py`: Same relaxation for `_is_laguna_fp8_gqa6_analytic_decode_graph`
   - Enables: warp-specialized producer/consumer, device-side analytic schedule, compact GQA sync

2. **chunk_pages=32**: +2-5% at 4K, ~0% at 64K

## Research Completed (8 agents total)
1. **FA4 SM120 portability**: SM120 has NO wgmma, uses mma.sync. FA4 official SM120 = SM80 stub. Portable: TMA, warp-spec, FP8 softmax, persistent scheduler. SparkInfer analytic path already uses TMA+warp-spec.
2. **CUTLASS SM120 GEMM**: SM120 has NO BF16 tensor core. FP8/NVFP4 only. torch._scaled_mm works for M≥4.
3. **MoE bandwidth**: Already 100% saturated. No tuning helps. Only system-level opts (batch amortization, L2 locality).
4. **FlashInfer SM120**: Three-tier MoE dispatch (micro/static/dynamic). MAC tuning ladders. XQA NVFP4 KV.
5. **Cursor Warp Decode**: Output-centric MoE, 1.84x on Blackwell. Long-term moonshot.
6. **sm120-flash-attention**: Custom kernel 2.4-3x slower than SparkInfer. Bottleneck: cp.async wait chain.

## Root Cause Analysis
- **TURBO quality regression**: FP8 e4m3 has 3 mantissa bits (vs BF16's 7). Checkpoint k_scale NOT used (default 1.0). Precision loss is inherent to FP8, not a scaling bug.
- **Specialized decode path**: Was gated on TP=2 config (24/4 heads, page_size=128). Kernel body is shape-generic; only gating predicates needed relaxation.

## Next Steps (Priority Order)
1. **TURBO quality fix** (kernel-level): Per-head descale, Hadamard rotation, or adaptive FP8/BF16 switching
2. **FA4 techniques for prefill/extend path**: TMA loads, persistent scheduler, FP8 softmax
3. **num_stages≥2 for FP8 attention**: Policy change in forward_paged.py (SMEM feasible: 36KB << 99KB)
4. **Warp Decode MoE**: Output-centric parallelism (2-4 week effort)
