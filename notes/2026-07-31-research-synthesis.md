# Research Synthesis: SM120 Optimization Directions (2026-07-31)

## Current Performance
- fox-64K: 383 tok/s, accept=100% (TURBO=1, CG=1040, CTAS=4, chunk_pages=32)
- fox-4K: 356 tok/s, accept=96.3%
- code-4K: 230 tok/s, accept=58.6% (TURBO quality regression)

## Kernel Profile (64K, TURBO=1)
- MoE: 43.7% (5.22s, 752 calls, 6.94ms/call)
- Attention: 33.1% (3.95s, 384 calls, 10.27ms/call)
- Dense GEMM: 12.9% (1.54s, 794 calls) — SM80 kernels (SM120 has NO BF16 tensor core!)
- Other: 10.3%

## Key Findings

### 1. SM120 Hardware (from CUTLASS research)
- NO BF16/FP16 tensor core — SM80 mma.sync fallback is correct behavior
- Native MMA: FP8 (2x throughput) and NVFP4 (4x) only
- 99KB SMEM, no TMEM, no tcgen05, no cluster multicast
- FA4 NOT available on SM120

### 2. FP8 Dense GEMM (measured)
- torch._scaled_mm works on SM120 for M≥4 (TN layout)
- M=1 FAILS (cuBLAS limitation)
- scaled_mm alone: 1.62x total across all model shapes
- Best shapes: o_proj 2.2-2.4x, lm_head 1.97x, g_proj 1.59x
- Worst shapes: shared_gate 0.55x, qkv_swa 0.79x
- Dynamic quantization overhead (~42us/call) kills benefit at M=16
- Fixed scale reduces overhead to ~4us/call but quality risk

### 3. SparkInfer Specialized Decode Path (from sm120-fa research)
- Current: generic path, attention at 37% bandwidth ceiling (480/1300 GB/s)
- Specialized path gated on: page_size=128, num_q_heads=24, num_kv_heads=4 (TP=2!)
- Our TP=1: num_q_heads=48/72, num_kv_heads=8, page_size=64
- Unlocking could give up to 2.7x attention improvement
- Requires: generalize traits.py matching + page_size migration

### 4. Adaptive Split-K (measured)
- chunk_pages=32 vs default 52: +2-5% at 4K, ~0% at 64K
- Already applied (SPARKINFER_PAGED_DECODE_GRAPH_CHUNK_PAGES=32)

### 5. FlashInfer SM120 Features (from FlashInfer research)
- XQA NVFP4 KV cache: SM120-exclusive, halves KV memory traffic
- SM120 MoE Micro/Static dispatch: purpose-built for tiny decode
- CuTe-DSL GQA decode: native SWA support, 8x KV read reduction
- SM120 MoE NVFP4 W4A4: full support with MAC tuning for 188 SMs

### 6. Cursor Warp Decode (from web research)
- Output-centric MoE parallelism: 1.84x MoE speedup on Blackwell
- Eliminates scatter/gather bookkeeping
- Architecture-agnostic technique

## Priority Order (impact × feasibility)
1. Fix TURBO code-4K quality → enable +6.2% by default
2. Unlock specialized decode path → up to 2.7x attention
3. FP8 dense GEMM (selective, fixed scale) → ~3-5% total
4. SWA native sliding window → 8x KV read reduction for 36 SWA layers
5. MoE output-centric parallelism → 1.84x MoE (long-term)
