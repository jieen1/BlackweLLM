# Research: Warp Decode, FlashInfer SM120 MoE, DeepGEMM MoE

Date: 2026-07-31
Scope: Techniques applicable to our SparkInfer MoE kernel on SM120

---

## Part 1: Cursor Warp Decode

**Source:** https://cursor.com/blog/warp-decode (2026-04-06)
**Authors:** Less Wright, Federico Cassano, Zhiyuan Zhang
**Claimed:** 1.84x MoE decode speedup on Blackwell B200, 1.4x accuracy improvement

### 1.1 Output-Centric vs Expert-Centric Parallelism

**Traditional (expert-centric):** Organize all computation around experts.
Collect tokens each expert needs → run GEMM → reassemble results.
8 stages total, 5 of which are pure bookkeeping (no computation).

**Warp decode (output-centric):** Each warp is assigned exactly one output
value (neuron). The warp streams weight data directly from memory, aggregates
across all routed experts in a single running total, writes one result.

The key insight: for decode (B=1..32), there isn't enough shared work per
expert to justify the expert-centric overhead. The parallelism axis is flipped
from "which expert am I computing?" to "which output neuron am I producing?"

### 1.2 The 5 Bookkeeping Steps Eliminated

The traditional 8-stage MoE decode pipeline:
1. **Route** (top-k computation) — KEEP
2. **Gather/pad** tokens per expert — ELIMINATED
3. **Scatter** to expert-major layout — ELIMINATED
4. **Quantize** activations (BF16→MXFP8) — ELIMINATED
5. **FC1** (gate/up GEMM) — KEEP (restructured)
6. **Activation** (SiLU) — KEEP (fused)
7. **FC2** (down GEMM) — KEEP (restructured)
8. **Combine/scatter** back to token layout — ELIMINATED

Plus the implicit 5th bookkeeping: **intermediate buffer management**
(activation gather buffer + per-expert output buffer).

Warp decode compresses this to **2 kernels**: gate/up kernel + down kernel.

### 1.3 Top-K Routing Handling

In the **down kernel**, each warp owns one output dimension for one token.
It loops over all top-k routed experts, loading the relevant down-projection
weight row and streaming over intermediate activations, while folding each
expert's routing weight into a single running FP32 accumulator.

The routing weight is folded into the accumulator *within the warp* — the
top-k intermediate results never materialize in memory.

### 1.4 Memory Access Pattern Difference

**Traditional:**
- Activation gather buffer: full copy of input into expert-major layout
  (at B=1, a complete redundant copy)
- Per-expert output buffer: 8 × 2048 × 2B = 32KB per token in BF16,
  allocated → written → read once → discarded
- Multiple global memory round-trips for staging

**Warp decode:**
- Input activations: read-only, streamed once
- Weights: streamed per-warp from global memory (random access by expert ID)
- Accumulator: private registers, never touches global memory
- Output: single scalar write per warp
- Eliminates 32+ KB intermediate buffer traffic per token → frees L2 for weights

### 1.5 NVFP4 Applicability

The blog describes **MXFP8 weights** with BF16 activations and FP32
accumulators. The accuracy gain comes from *removing* the BF16→MXFP8
activation quantization step.

For NVFP4: the technique is architecturally applicable but the weight format
differs. NVFP4 uses block-scaled FP4 (block_size=16) with per-block scale
factors. A warp-decode NVFP4 kernel would need to:
- Unpack FP4 weights on-the-fly within each warp
- Apply block scale factors during the dot product
- Handle the global scale (alpha) per expert

The fundamental output-centric parallelism is format-agnostic. The weight
streaming pattern (one row per warp) works with any quantization format.
The main question is whether FP4 unpack + block-scale application fits
efficiently within the warp's register budget.

### 1.6 Minimum Batch Size

Warp decode works at **B=1** (single token). The throughput gain is flat
across all context-length buckets. Tested up to B=32 where it sustains
3.95 TB/s (58% of B200's 6.8 TB/s peak).

**Not suitable for prefill or large batches** — expert-centric packing wins
there because many tokens share the same expert.

### 1.7 Hardware Efficiency

- B200 peak contiguous read: 6.8 TB/s
- Warp decode at B=32: 3.95 TB/s (58% of peak)
- Gap attributed to random access patterns from expert routing
  (non-adjacent experts like 5, 8, 14, 19)
- Cosine similarity vs FP32 reference: > 0.999996
- Max absolute difference: 0.001953

### 1.8 Kernel Architecture

**Gate/up kernel:**
- CTA = 8 warps
- Each warp owns one intermediate neuron for one (token, expert) pair
- Loads expert ID → reads gate + up weight rows → streams activation vector
- MXFP8→FP32 conversion on the fly
- Both dot products accumulate in private registers
- Activation vector read once, reused for both projections (no smem staging)
- Warp-level reduction → SiLU(gate) × up → write one intermediate value

**Down kernel:**
- Each warp owns one output dimension for one token
- Loops over all top-k experts
- Loads down-projection weight row per expert
- Folds routing weight into running FP32 accumulator
- Warp-level butterfly reduction via `__shfl_xor_sync` (PTX `shfl.sync.bfly`)
- No shared memory, no L1 round-trips, no bank conflicts, no barriers
- Final weighted top-k combination is part of the projection itself

---

## Part 2: FlashInfer SM120 MoE Architecture

**Source:** `/home/bot/project/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/`

### 2.1 Three-Tier Dispatch

| Tier | Backend | Routed Pairs | Use Case |
|------|---------|-------------|----------|
| **Micro** | `MoEMicroKernel` / `MoEDirectMicroKernel` | ≤20 (top_k=1) or ≤40 (multi-topk) | Tiny decode (1-4 tokens) |
| **Static** | `MoEStaticKernel` | ≤640 | Decode |
| **Dynamic** | `MoEDynamicKernel` | >640 | Prefill / large batch |

Selection logic (`select_sm120_moe_backend`):
```python
routed_rows = num_tokens * num_topk
if routed_rows <= static_compact_cutover_pairs:  # default 640
    return "static"
return "dynamic"
```
Micro is selected within the static launcher when `routed_rows <= micro_cutover`.

### 2.2 MAC (Max Active Clusters) Tuning Ladders

Profiling-derived lookup tables mapping routed_rows → optimal cluster count:

**Micro MAC ladder:**
| Max routed rows | Optimal MAC |
|----------------|-------------|
| 2 | 84 |
| 4 | 127 |
| 8 | 107 |
| 10 | 84 |
| 16 | 63 |
| 20 | 84 |

**Static MAC ladder:**
| Max routed rows | Optimal MAC |
|----------------|-------------|
| 24 | 148 |
| 32 | 169 |
| 40 | 132 |
| 48 | 149 |
| 64 | 134 |
| 80 | 175 |
| 96 | 171 |
| 120 | 125 |
| 128 | 130 |
| 160 | 171 |
| 192 | 166 |
| 256 | 141 |
| 320 | 158 |
| 512 | 175 |
| 640 | 188 |

MAC is further clamped by: `min(tuned_mac, work_tiles, hardware_limit)`.

### 2.3 Micro Kernel Design (Tiny Decode)

**MoEMicroKernel** (compact, MMA-based):
- Precompacted routing via Triton pre-pass (`triton_compact.py`)
  - Remaps global expert IDs → dense local indices (0,1,2,...)
  - Single CTA, single-block Triton kernel
- Two-phase algorithm in one resident launch:
  - Phase 0+1: cooperative init + route/pack (quantize tokens into expert-major)
  - Resident-grid barrier
  - Phase 2: FC1 → activation → quant → FC2 → scatter
- `single_token` fast path: when num_tokens==1 and every expert has exactly
  one row, skip atomic routing → O(1) work assignment
- FC1 amortized across all FC2 output tiles (intermediate slice caching)
- Scatter via `bf16x2` atomic add into token-major output

**MoEDirectMicroKernel** (warp-level direct dot products):
- For very small routed decode batches
- Warp-level direct dot products instead of full MMA tiles
- `_BLOCK_SIZE = 16`, `_NUM_WARPS = 16`, `_BLOCK_DIM = 512`
- `_K_PER_CTA = 16`, `_MAX_DIRECT_K_SEGMENTS = 12`
- FC1 chunks decomposed for warp-level parallelism
- Shared memory for intermediate storage (`smem_xh`)
- FP4 dot products via `fp4_dot4_sum`, `fp4_dot8_sum` intrinsics

### 2.4 Static Kernel Design

- Same two-phase algorithm as micro but without precompaction
- Phase 0: cooperative init / clear row counts
- Phase 1: walk routed pairs, atomic append to row_counts, quantize into
  expert-major packed A + scale storage
- Resident-grid barrier
- Phase 2: compact static work loop assigns (m_tile, intermediate_slice, expert)
- Tile shapes: 128×128 default, 64×128 for small routed_rows
- Per-expert FC1 activation scales (checkpoint-correct)

### 2.5 Dynamic Kernel Design

- Queue-driven: replaces resident-grid barrier with global ready-task queue
- All CTAs start as producers (route/pack), then become consumers (compute)
- Tasks published per ready (expert, m_tile, slice_group) as tiles complete
- No global route/pack → compute barrier
- Route/pack is warp-private instead of CTA-broadcast
- Compute driven by global append-only ready-task queue

### 2.6 NVFP4 Details

- `_NVFP4_BLOCK_SIZE = 16` (block scale factor granularity)
- `SF_VEC_SIZE = 16` (scale factor vector size)
- Scale factors in CUTLASS/CuTe block-scaled MMA layout
- `swizzle_block_scale()` for physical layout transform
- Supports both NVFP4 (W4A4) and W4A16 (BF16 activations) modes
- W4A16 path via `moe_w4a16_kernel.py` (separate dispatch)
- FC2 intermediate quantization: cooperative FP4 quantization into shared A

### 2.7 Key Techniques

1. **Precompacted routing** (Triton): dense local expert IDs eliminate
   global-to-local remapping overhead in the kernel
2. **Resident-grid barrier**: single-launch route+compute without host handoff
3. **FC1 slice amortization**: FC1 computed once per intermediate slice,
   reused across all FC2 output tiles
4. **Adaptive tile sizing**: 64×128 vs 128×128 based on SM utilization
5. **MAC tuning**: profiling-derived cluster counts per batch size
6. **Single-token fast path**: O(1) work assignment when B=1
7. **Direct micro kernel**: warp-level dot products bypass MMA entirely
   for the smallest workloads

---

## Part 3: DeepGEMM MoE Capabilities

**Source:** `/home/bot/project/DeepGEMM/` (v2.6.1)

### 3.1 Architecture Support

- **SM90** (Hopper): Full support
- **SM100** (Blackwell datacenter): Full support
- **SM120/SM121** (Blackwell consumer): **NO SUPPORT**
  - Zero files reference sm120/sm121/sm12x
  - All MMA code targets sm90 or sm100 (`mma/sm90.cuh`, `mma/sm100.cuh`)
  - Uses `tcgen05` PTX (SM100-only tensor core generation)

### 3.2 Mega MoE (Fused MoE + Communication)

The flagship MoE feature: `fp8_fp4_mega_moe` and `bf16_mega_moe`.
- Fuses routing, FC1, activation, FC2, and **inter-rank communication**
- Uses symmetric memory buffers (`torch.distributed._symmetric_memory`)
- 2-CTA cluster design (SM100)
- L1/L2 interleaved scheduling with warmup waves to prevent L1→L2 deadlock
- Ring buffer for intermediate activation storage
- Supports shared experts alongside routed experts
- FP8×FP4 GEMM: FP8 activations × FP4 weights with UE8M0 scale factors
- SwiGLU activation with optional clamping
- Multi-rank expert parallelism (num_experts / num_ranks)

### 3.3 Grouped GEMM Support

| Type | API | Layout |
|------|-----|--------|
| m-grouped contiguous | `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous` | Contiguous M groups |
| m-grouped masked | `m_grouped_fp8_fp4_gemm_nt_masked` | Per-group M mask |
| k-grouped contiguous | `k_grouped_fp8_gemm_{nt,tn}_contiguous` | Contiguous K groups |

All support FP8×FP4 with block-scaled quantization.

### 3.4 FP8×FP4 GEMM

- `fp8_fp4_gemm_{nt,nn,tn,tt}`: dense GEMM with FP8 A × FP4 B
- Block scale factors with UE8M0 packing (4 × UE8M0 → 1 int32)
- Granular K scaling (gran_k=32 typical)
- TMA-aligned scale factor layout
- SM100 uses `tcgen05` MMA instructions

### 3.5 Applicability to SM120

**Not directly applicable.** DeepGEMM's SM100 kernels use:
- `tcgen05` PTX instructions (SM100-only)
- 2-CTA clusters (SM100 cluster semantics)
- TMEM allocator (`cute::TMEM::Allocator2Sm`)

SM120 lacks tcgen05 and uses different MMA instructions. The scheduling
concepts (L1/L2 interleaving, ring buffers, warmup waves) are architecturally
interesting but would require complete reimplementation for SM120's MMA path.

---

## Part 4: Ranked Techniques for Our SparkInfer MoE Kernel

Our current kernel: sparkinfer NVFP4 MoE, E=256, K=3072, I=1024, top_k=10.
CUDA graph: ~38μs/layer. SM120 single GPU, batch 1-4 decode.

### Rank 1: Output-Centric Warp Decode for FC2 (Down Projection)

**Source:** Cursor Warp Decode
**Estimated impact:** HIGH (1.3-1.8x decode MoE speedup)
**Implementation complexity:** HIGH
**Risk:** MEDIUM

The down kernel pattern (each warp owns one output neuron, loops over top-k
experts) is directly applicable to our decode path. Our model has:
- hidden_size=3072 → 3072 independent output warps per token
- top_k=10 → each warp loops over 10 experts
- intermediate_size=1024 → each warp streams 1024 FP4 values per expert

This eliminates: expert-major packing, intermediate buffer, combine/scatter.
The routing weight folding into the accumulator is especially valuable for
our top_k=10 (vs Cursor's top_k=8).

**Key adaptation needed:** NVFP4 weight unpacking within the warp. FP4 block
size 16 means each warp processes 1024/16 = 64 scale factors per expert.
Register pressure: 10 experts × (weight row + scale) must stream, not reside.

**Why rank 1:** Our decode is exactly the regime where warp decode wins
(B=1-4, not enough shared work per expert). The 38μs/layer × 47 layers =
1.8ms total MoE time is dominated by the expert-centric overhead that warp
decode eliminates.

### Rank 2: Direct Micro Kernel (Warp-Level Dot Products for B=1)

**Source:** FlashInfer `MoEDirectMicroKernel`
**Estimated impact:** MEDIUM (1.2-1.5x for B=1 decode)
**Implementation complexity:** MEDIUM
**Risk:** LOW

FlashInfer's direct micro kernel bypasses MMA tiles entirely for tiny batches,
using warp-level FP4 dot products (`fp4_dot4_sum`, `fp4_dot8_sum`). With
16 warps × 32 threads = 512 threads per CTA, and K_PER_CTA=16, this is
purpose-built for the B=1 case.

Our sparkinfer kernel already handles B=1 but through the standard MMA path.
A direct micro kernel would eliminate MMA tile setup overhead for the common
single-token decode case.

**Key adaptation:** The `fp4_dot4_sum` / `fp4_dot8_sum` intrinsics are
SM120-compatible (they're register-level FP4 dot products, not tcgen05).
The shared memory intermediate storage pattern (`smem_xh`) is portable.

**Why rank 2:** Lower risk than full warp decode, proven architecture,
directly targets our dominant B=1 decode workload.

### Rank 3: Precompacted Routing (Triton Pre-Pass)

**Source:** FlashInfer `triton_compact.py`
**Estimated impact:** LOW-MEDIUM (5-15% decode latency reduction)
**Implementation complexity:** LOW
**Risk:** LOW

Remap global expert IDs (0-255) to dense local indices (0..active_count-1)
before the main kernel. This eliminates global-to-local expert remapping
inside the kernel and enables the single_token fast path.

For our E=256, top_k=10: only ~10 unique experts per token at B=1.
Compacting 256→10 expert indices simplifies all downstream indexing.

**Why rank 3:** Easy to implement, low risk, composable with ranks 1-2.
The Triton kernel is a single CTA, single-block launch (~1μs).

### Rank 4: MAC Tuning Ladder

**Source:** FlashInfer `_MICRO_MAC_LADDER` / `_STATIC_MAC_LADDER`
**Estimated impact:** LOW-MEDIUM (5-20% depending on current MAC)
**Implementation complexity:** LOW
**Risk:** LOW

Profile-driven cluster count selection per batch size. FlashInfer's ladders
show significant variation (84→127→107→84→63→84 for micro) — the optimal
MAC is non-monotonic and hardware-specific.

Our SM120 has fewer SMs than B200 (148 SMs), so the ladder values would
differ, but the *technique* of profiling-derived MAC selection is directly
applicable.

**Why rank 4:** Requires profiling infrastructure (we have `bf exec`),
easy to implement, but impact depends on how far current MAC is from optimal.

### Rank 5: FC1 Slice Amortization

**Source:** FlashInfer static/micro kernel compute design
**Estimated impact:** MEDIUM (10-25% for multi-token decode)
**Implementation complexity:** MEDIUM
**Risk:** LOW

FC1 computed once per intermediate slice, then reused across all FC2 output
tiles. This amortizes the gate+up GEMM cost across the down projection's
output dimension.

Our sparkinfer kernel may already do this internally, but verifying and
exposing this optimization could help for B=2-4 decode where multiple tokens
share intermediate slices.

### Rank 6: Dynamic Queue-Driven Dispatch (for Prefill)

**Source:** FlashInfer `MoEDynamicKernel`
**Estimated impact:** MEDIUM for prefill, NONE for decode
**Implementation complexity:** HIGH
**Risk:** MEDIUM

Replace resident-grid barrier with global ready-task queue. CTAs transition
from producers (route/pack) to consumers (compute) without global sync.
Only relevant for our prefill path, not the decode-dominant workload.

### Rank 7: DeepGEMM Mega MoE Scheduling Concepts

**Source:** DeepGEMM `mega_moe.cuh`
**Estimated impact:** LOW (conceptual, not directly portable)
**Implementation complexity:** VERY HIGH
**Risk:** HIGH

L1/L2 interleaved scheduling with warmup waves, ring buffer intermediate
storage, and fused communication. Architecturally interesting but targets
SM100 with tcgen05 — would require complete reimplementation for SM120.
The multi-rank communication fusion is irrelevant for our single-GPU setup.

**Only applicable concept:** The L1 warmup wave calculation to prevent
L1→L2 deadlock in interleaved schedules, if we ever implement a multi-phase
fused kernel.

---

## Summary Matrix

| Rank | Technique | Source | Impact | Complexity | Risk | Decode? | NVFP4? |
|------|-----------|--------|--------|------------|------|---------|--------|
| 1 | Output-centric warp decode (FC2) | Cursor | HIGH | HIGH | MED | ✓ | Needs adaptation |
| 2 | Direct micro kernel (warp dots) | FlashInfer | MED | MED | LOW | ✓ | Native |
| 3 | Precompacted routing | FlashInfer | LOW-MED | LOW | LOW | ✓ | N/A |
| 4 | MAC tuning ladder | FlashInfer | LOW-MED | LOW | LOW | ✓ | N/A |
| 5 | FC1 slice amortization | FlashInfer | MED | MED | LOW | B>1 | Native |
| 6 | Dynamic queue dispatch | FlashInfer | MED | HIGH | MED | Prefill | Native |
| 7 | Mega MoE scheduling | DeepGEMM | LOW | V.HIGH | HIGH | N/A | SM100 only |

### Recommended Implementation Order

1. **Quick wins (days):** Ranks 3+4 — precompacted routing + MAC tuning.
   Low risk, composable, immediate measurable improvement.

2. **Medium effort (1-2 weeks):** Rank 2 — direct micro kernel for B=1.
   Port FlashInfer's warp-level dot product approach to our NVFP4 weights.

3. **Strategic investment (2-4 weeks):** Rank 1 — output-centric warp decode
   for the down projection. Highest potential impact but requires NVFP4
   unpacking within warps and new kernel architecture.

### Key Risk: NVFP4 + Warp Decode

Cursor's warp decode uses MXFP8 weights (1 byte/weight, simple conversion).
NVFP4 uses 4-bit weights with block-16 scale factors. The warp must:
- Unpack 2 FP4 values per byte
- Look up block scale factor every 16 elements
- Apply global scale (alpha) per expert

This adds register pressure and instruction overhead per dot product step.
The feasibility depends on whether the unpack+scale overhead is hidden by
memory latency (likely yes, since the kernel is memory-bound at decode).
