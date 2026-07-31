# FlashAttention 4 → SM120 Portability Research

> Date: 2026-07-31
> Sources: FA4 repo (Dao-AILab/flash-attention @ HEAD), sm120-flash-attention project,
> kekzl/imp source, CUTLASS docs, PTX ISA 9.3, project research notes (04-最新技术动向调研.md)

---

## 1. FA4 Algorithm Breakdown: SM100-Specific vs Portable

### 1.1 SM100 Forward Kernel Architecture (flash_fwd_sm100.py, 3202 lines)

The SM100 kernel uses **16 warps per CTA** with extreme role specialization:

| Warps | Role | Function |
|-------|------|----------|
| 0-3 | softmax0 | Row-max, exp2, row-sum for S tiles (q_stage=0) |
| 4-7 | softmax1 | Row-max, exp2, row-sum for S tiles (q_stage=1) |
| 8-11 | correction | O accumulator rescaling (multiply by row_scale) |
| 12 | MMA | Single warp drives tcgen05 tensor core |
| 13 | epilogue | O write-back (TMA store or register→gmem) |
| 14 | load | TMA producer (K, V, Q loads) |
| 15 | empty | Spare / CLC scheduler warp |

**Key SM100-only hardware dependencies:**

| Feature | Hardware | SM120 Status |
|---------|----------|--------------|
| tcgen05 MMA | Tensor Memory Generator v5 | ❌ ABSENT |
| TMEM (Tensor Memory) | 256-column accumulator store | ❌ ABSENT |
| 2-CTA cluster MMA | Cluster shape (2,1), shared MMA | ❌ ABSENT (cluster fixed 1×1×1) |
| CLC (Command Launch Control) | Hardware tile scheduler | ❌ ABSENT |
| 228KB SMEM | Shared memory capacity | ❌ Only 99KB |
| TMA (cp.async.bulk) | Bulk async copy | ⚠️ UNVERIFIED on SM120 |
| Multicast | Cluster-wide SMEM broadcast | ❌ ABSENT |

**Portable algorithmic innovations (hardware-independent):**

| Technique | Description | Portability |
|-----------|-------------|-------------|
| Warp specialization | Separate softmax/correction/load/MMA warps | ✅ Fully portable via setmaxregister |
| FP8 softmax with online rescaling | P quantized to FP8 with rescale_threshold | ✅ Fully portable |
| Persistent tile scheduling | Software static/dynamic scheduler | ✅ Portable (no CLC needed) |
| ex2 emulation tuning | Trade MUFU exp2 for ALU emulation | ✅ Fully portable |
| Low-precision P scaling | rescale_threshold for FP8/FP4 P | ✅ Fully portable |
| Online softmax (log-sum-exp) | Standard FA2/3 online algorithm | ✅ Already implemented |
| Pack GQA | Pack multiple Q heads into M dimension | ✅ Portable |
| Split-KV + merge | Flash-decoding parallelization | ✅ Already implemented |

### 1.2 SM90 Forward Kernel (flash_fwd_sm90.py) — Closest SM120 Analog

The SM90 kernel uses a simpler but effective warp specialization:

- **1 producer warp** (32 threads): TMA loads for K, V, Q
  - `setmaxregister_decrease(24-56)` — minimal registers
- **4 consumer warps** (128 threads): wgmma + softmax + epilogue
  - `setmaxregister_increase(160-256)` — maximum registers
- **Pipeline**: Async K/V pipeline with producer/consumer mbarrier sync
- **intra_wg_overlap**: Overlap softmax computation with next wgmma issue
- **Tile scheduler**: Static persistent (non-causal), single-tile (causal)

**SM120 applicability**: SM90 uses wgmma (SM90-only) and TMA. SM120 has NEITHER
(see §1.3). However, the warp specialization PATTERN (producer/consumer split,
setmaxregister, pipeline synchronization) is fully portable to SM120 using
mma.sync + cp.async.

### 1.3 CRITICAL CORRECTION: SM120 Hardware Capabilities

**The task description's claim that "SM120 HAS wgmma (SM90-style)" is INCORRECT.**

Evidence (multiple independent sources):
1. FA4 `flash_fwd_sm120.py`: Forces `self.arch = Arch.sm_80`, uses SM80 MMA paths
2. FA4 `interface.py:658`: `self.use_tma_O = Arch.sm_90 <= self.arch < Arch.sm_120` — explicitly excludes SM120 from TMA
3. sm120-flash-attention kernel comment: "sm120 has no wgmma/tcgen05"
4. CUTLASS blackwell_functionality.md: SM120 supports only "pingpong/cooperative" schedules with mma.sync
5. PTX ISA 9.3: wgmma.mma_async is sm_90a only; sm_120 has mma.sync.aligned variants

**Actual SM120 MMA capabilities:**
- `mma.sync.aligned.m16n8k16` (BF16/FP16) — SM80-era
- `mma.sync.aligned.m16n8k32` (FP8 e4m3/e5m2) — SM89-era
- `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (NVFP4) — SM120-native
- `mma.sync.aligned.kind::mxf8f6f4.block_scale` (MX formats) — SM120-native

**SM120 memory capabilities:**
- `cp.async.cg.shared.global` (16-byte) — SM80-era async copy
- `cp.async.ca.shared.global` (4-byte) — for non-aligned strides
- TMA (cp.async.bulk): **UNVERIFIED** — FA4 excludes SM120, but hardware may support it
- 99KB SMEM (vs 228KB SM100, 163KB SM80, 228KB SM90)
- setmaxregister: ✅ Available

---

## 2. SM120 Adaptation Strategy Per Technique

### 2.1 2-CTA Cluster Pingpong → Warp-Level Pingpong

**SM100 mechanism**: Two CTAs in a cluster share a single tcgen05 MMA operation.
While CTA0's MMA executes, CTA1 runs softmax, and vice versa. This hides
softmax latency behind MMA throughput.

**SM120 adaptation**: No clusters, no tcgen05. Alternative approaches:

**Option A: Intra-CTA warp pingpong (RECOMMENDED)**
- Split warps into two groups: MMA warps and softmax warps
- MMA warps issue mma.sync for QK^T, then signal softmax warps via named barrier
- Softmax warps compute exp2/rescale while MMA warps issue PV mma.sync
- Use `bar.sync` / `bar.arrive` for fine-grained synchronization
- This is essentially what FA4 SM100 does but within a single CTA

**Option B: Double-buffered Q tiles (q_stage=2)**
- Process two Q subtiles alternately: while softmax runs on Q[0], MMA runs QK^T for Q[1]
- Requires 2× Q register storage (expensive at D=256)
- FA4 SM100 uses q_stage=2; SM120 register pressure may prevent this

**Option C: Software pipelining (cp.async overlap)**
- Overlap next KV tile's cp.async load with current tile's MMA+softmax
- Already partially implemented in existing kernel (double-buffered cp.async)
- Limited by cp.async wait chain (identified as bottleneck in closure notes)

**Risk assessment**: Option A is most promising. The existing V3 warp-specialized
decode kernel already demonstrates QK/PV warp splitting on SM120. Extending this
to a full pingpong pattern for prefill is the natural next step.

### 2.2 FP8 Softmax with Online Rescaling

**SM100 mechanism**: After computing S = QK^T in FP32, the softmax warps:
1. Compute row_max, apply exp2
2. Cast P to FP8 (e4m3) with `rescale_threshold` to use FP8's dynamic range
3. Feed FP8 P into PV MMA (tcgen05 with FP8 operands)
4. Correction warps rescale O accumulator by row_scale

**SM120 adaptation**: ✅ FULLY PORTABLE
- Online softmax algorithm is identical (FP32 row_max, exp2, row_sum)
- P quantization to FP8: Use `cvt.rn.satfinite.e4m3x2.f32` (available on SM120)
- FP8 PV MMA: Use `mma.sync.aligned.m16n8k32` with FP8 operands (SM89-era, available)
- rescale_threshold: Pure software, directly portable
- ex2 emulation: Trade MUFU exp2 for ALU (portable, tuning needed)

**Implementation priority**: HIGH — this is the lowest-hanging fruit.
The existing kernel already has FP8 QK and FP8 KV paths. Adding FP8 P
(softmax output quantization) with proper rescaling is incremental.

### 2.3 Persistent Kernels with Dynamic Tile Scheduler

**SM100 mechanism**: CLC (Command Launch Control) hardware scheduler assigns
work tiles to CTAs with zero CPU overhead. CTAs loop until all tiles processed.

**SM120 adaptation**: ✅ PORTABLE (software scheduler)
- Use `atomicAdd` on a global work counter for dynamic scheduling
- Or use static persistent scheduling (pre-assign tiles to SMs)
- FA4's `StaticPersistentTileScheduler` is pure software — directly portable
- For causal attention, use LPT (Longest Processing Time) ordering

**Implementation**: 
```
// Software persistent tile scheduler
__device__ int g_tile_counter;
while (true) {
    int tile_idx = atomicAdd(&g_tile_counter, 1);
    if (tile_idx >= total_tiles) break;
    // decode (batch, head, m_block) from tile_idx
    // process tile...
}
```

**Risk**: LOW. Software persistent scheduling adds ~1-2% overhead vs CLC but
is well-understood. FlashInfer already uses this pattern on SM120.

### 2.4 TMEM-Based Dataflow → Register File Dataflow

**SM100 mechanism**: S (QK^T result) and O (output accumulator) live in TMEM
(256 columns × 128 rows). This frees the register file entirely for softmax
computation and control flow. The MMA warp reads/writes TMEM directly.

**SM120 impact**: This is the BIGGEST performance gap.

On SM120, S and O must live in registers:
- O accumulator for D=256: 128 floats/lane (32 N-tiles × 4 elements) = **128 registers**
- S tile for 128×128: 64 floats/lane = **64 registers**
- Softmax state (row_max, row_sum, scale): ~8 registers
- Control flow, addresses, loop vars: ~30 registers
- **Total: ~230 registers/thread** → hits 255 limit, 1 CTA/SM

**Mitigation strategies:**

1. **D-dimension PV tiling** (split D=256 into 2×128 passes):
   - O_acc peak: 64 floats/lane instead of 128
   - Requires storing P in SMEM between passes (99KB budget allows this)
   - Identified in closure notes as "P0" — the real register lever
   - Cost: 2× QK^T computation (or P caching in SMEM)

2. **Smaller M tiles** (64 instead of 128):
   - Halves S register pressure
   - Reduces MMA utilization (more iterations)
   - FA4 SM120 already uses 128×64 for D>64

3. **Warp specialization to separate S and O**:
   - QK warps compute S, write to SMEM
   - PV warps read S from SMEM, accumulate O in registers
   - Neither warp needs both S and O simultaneously
   - This is the V3 warpspec approach, extended

**Performance impact estimate**: Register pressure is the #1 limiter.
SM100's TMEM eliminates this entirely. SM120 must trade either tile size
(lower MMA efficiency) or occupancy (1 CTA/SM). Estimated 20-35% throughput
loss from register pressure alone.

### 2.5 Warp Specialization (Producer/Consumer)

**SM100 mechanism**: 16 warps with 7 distinct roles (see §1.1 table).
setmaxregister partitions the register file: softmax warps get 176-256 regs,
correction warps get 64-160 regs, MMA warp gets the rest.

**SM120 adaptation**: ✅ PORTABLE (already partially demonstrated)

The existing V3 decode kernel (`decode_v3_warpspec.cuh`) implements:
- QK warps: Compute QK^T + softmax, write P to SMEM
- PV warps: Read P from SMEM, compute PV, accumulate O
- Communication via SMEM (alpha, p_scale per M-tile)
- `__launch_bounds__(256, V3_MINBLOCKS=2)` for 2 CTAs/SM target

**Extension to prefill**: The V3 pattern can be extended to prefill with:
- Dedicated load warps (cp.async producer)
- Dedicated softmax warps (exp2, rescale)
- Dedicated MMA warps (QK^T and PV)
- Named barriers for phase synchronization

**Key constraint**: Without wgmma, each mma.sync is a warp-level operation
(32 threads). SM90's wgmma operates at warpgroup level (128 threads).
This means SM120 needs 4× more warps to achieve the same MMA throughput,
increasing synchronization overhead.

---

## 3. Community Backport Status

### 3.1 FA4 Official SM120 Support (Dao-AILab/flash-attention)

**Status**: MINIMAL COMPATIBILITY SHIM — NOT OPTIMIZED

- `flash_fwd_sm120.py`: 60 lines, subclasses `FlashAttentionForwardSm80`
- Forces `self.arch = Arch.sm_80` — uses SM80 MMA instructions
- Tile sizes: 128×128 (D≤64), 128×64 (D>64)
- No SplitKV, no paged KV, no block sparsity, no FP8
- 128 threads (4 warps), single-stage pipeline
- `flash_bwd_sm120.py`: Similarly minimal (SM80 backward + 99KB SMEM check)

**Assessment**: This is a "it compiles and produces correct results" shim.
Performance is equivalent to FA2-era kernels. No FA4 innovations are ported.

### 3.2 kekzl/imp (MIT, v0.18.1, active)

**Status**: MOST ADVANCED COMMUNITY IMPLEMENTATION

- Full MXFP4 FMHA on SM120 (`attention_fmha_mxfp4_sm120.cu`)
- Two-level P quantization (Level 1: row renorm to 448×6; Level 2: per-16 UE4M3)
- NVFP4 KV cache with hardware `mma.sync.kind::mxf4nvf4.block_scale`
- head_dim=256 support (FA2 hd=256 prefill: +26% over WMMA path)
- Qwen3.5/3.6 GDN+attention hybrid architecture support
- Reverse-engineered mxf4nvf4 fragment/scale register layout

**Limitations**:
- Decode only ~18 tok/s for dense 27B (not competitive)
- NVFP4 decode numbers: "—" (not measured)
- Signature 266-338 tok/s numbers are from 30-35B MoE models, not dense 27B
- Independent engine, not vLLM-integrated

**Key insight from their MISSION_JOURNAL**: "Tensor-core decode attention (WMMA)
vs scalar: 118.4µs vs 119.0µs — identical. Bottleneck is softmax/dequant/latency,
not MMA." This independently validates that MMA throughput is NOT the limiter
on SM120 — memory latency and softmax overhead are.

### 3.3 SageAttention3

**Status**: BROKEN ON SM120
- Issue #311: Compilation failure on RTX PRO 6000 (same SM120 arch)
- Template syntax errors, community gave up, uses sm_89 fallback
- Numerical methodology (two-level P quantization) is referenceable
- Code is NOT usable as-is

### 3.4 FlashInfer / vLLM Ecosystem

**Status**: SM100-ONLY FOR QUANTIZED FMHA
- FlashInfer PR #3857: Quantized FMHA CuTe DSL kernel — SM100 only
- CUTLASS example 77 (Blackwell FMHA): SM100 only
- CUTLASS example 93 (low-latency GQA decode): SM100/103 only
- RFC #3628 (SM120 support): 0 comments, empty issue
- vLLM `--kv-cache-dtype nvfp4`: Crashes on first request on SM120 (issue #43562)

### 3.5 Other References

- **Florian Mattana FP4 kernel**: No P requantization (only QK is FP4, PV is scalar FP32). Reports "quantization = 66% of instructions" but this is WITHOUT solving the hard problem.
- **gau-nernst fa-5090**: Referenced in project notes, details unclear.
- **BlackFlash**: Present in project references directory.

---

## 4. SM120 Performance Ceiling vs SM100

### 4.1 Why SM100 Achieves 3.1 PFLOPs/s

| Factor | Contribution | SM120 Equivalent |
|--------|-------------|-----------------|
| tcgen05 MMA throughput | ~2× wgmma throughput | mma.sync: ~0.5× wgmma |
| TMEM (zero register pressure) | Enables max tile sizes | Register-limited tiles |
| 228KB SMEM | Deep pipelines (3-4 stage) | 99KB → 1-2 stage |
| 2-CTA cluster | 2× effective MMA width | Single CTA only |
| CLC hardware scheduler | Zero-overhead persistence | Software atomicAdd |
| TMA bulk copy | High-bandwidth async load | cp.async (lower BW) |

### 4.2 SM120 Ceiling Estimation

**Theoretical MMA throughput ratio:**
- SM100 tcgen05 (BF16): ~2× SM90 wgmma
- SM120 mma.sync (BF16): ~0.25× SM90 wgmma (m16n8k16 vs wgmma m64n256k16)
- **SM120/SM100 MMA ratio: ~12-15%** (raw instruction throughput)

**But attention is NOT purely MMA-bound.** For decode (memory-bound):
- SM120 HBM bandwidth: ~1.8 TB/s (RTX 5090)
- SM100 HBM bandwidth: ~8 TB/s (B200)
- **Bandwidth ratio: ~22%**

**For prefill (compute-bound at long sequences):**
- Effective FLOPS ratio dominated by MMA throughput: ~12-15%
- But register pressure reduces utilization further: ~60-70% of theoretical
- **Effective prefill ratio: ~8-12%**

**Realistic performance ceiling (SM120 vs SM100, same workload):**
- Decode (memory-bound): **20-25%** of SM100 throughput
- Prefill (compute-bound): **10-15%** of SM100 throughput
- Mixed inference: **15-20%** of SM100 throughput

**However**, the relevant comparison for this project is SM120 custom kernel
vs SM120 native FlashInfer. The question is whether FA4 techniques can close
the current 14.7% gap.

### 4.3 Gap Analysis: Custom Kernel vs Native FlashInfer on SM120

Current state (from closure notes):
- ms/draft: custom 14.7% SLOWER (10-20% range)
- ms/accepted-token: custom 8.6% SLOWER (5-13% range)
- Root cause: cp.async KV-tile sequential wait chain + V transpose SMEM round-trip

What FA4 techniques could address:
1. **Warp specialization** → hide softmax latency behind MMA (addresses barrier stalls)
2. **FP8 P with rescaling** → reduce PV MMA data movement (addresses bandwidth)
3. **Persistent scheduling** → reduce launch overhead for small tiles
4. **D-dimension tiling** → reduce register pressure → 2 CTAs/SM → better latency hiding

What FA4 techniques CANNOT address on SM120:
- Fundamental MMA throughput gap (no wgmma/tcgen05)
- SMEM capacity limit (99KB is fixed)
- cp.async bandwidth (no TMA confirmed)

---

## 5. What Our sm120-flash-attention Kernel Already Has vs FA4

### 5.1 Already Implemented

| FA4 Technique | Our Implementation | Status |
|---------------|-------------------|--------|
| Online softmax | ✅ Full (FA2-style, 4-lane butterfly) | Production |
| FP8 QK^T | ✅ Phase 2 (per-tensor scale) | Production |
| FP8 KV storage | ✅ Phase 2 round 2 (per-row-max P requant) | Production |
| NVFP4 QK^T | ✅ Phase 3 (two-level, mxf4nvf4.block_scale) | Verified |
| NVFP4 KV storage | ✅ Phase 3 round 2 | Verified |
| Paged KV cache | ✅ Phase 4 (CSR page table) | Production |
| Split-KV decode | ✅ Phase 4 round 2 (flash-decoding) | Production |
| Warp specialization | ✅ V3 decode (QK/PV warp split) | Experimental |
| cp.async double-buffer | ✅ 2-stage pipeline | Production |
| Varlen batch | ✅ Phase 4 | Production |
| vLLM integration | ✅ Phase 5 (backend registration) | Production |
| CUDA Graph compatibility | ✅ max_num_splits_override convention | Production |

### 5.2 Missing from FA4

| FA4 Technique | Gap | Priority | Effort |
|---------------|-----|----------|--------|
| Warp-specialized prefill | V3 is decode-only; prefill is monolithic | HIGH | 2-3 weeks |
| FP8 P with rescale_threshold | P requant exists but no threshold tuning | HIGH | 1 week |
| Persistent tile scheduler | Fixed grid launch, no persistence | MEDIUM | 1 week |
| D-dimension PV tiling | O_acc dominates registers (255 regs) | HIGH | 2 weeks |
| ex2 emulation tuning | Hardware exp2 only, no ALU tradeoff | LOW | 3 days |
| Pack GQA (M-dim packing) | GQA via grid.y, not M-packing | MEDIUM | 1 week |
| TMA loads (if available) | cp.async only | MEDIUM | 1 week (if HW supports) |
| q_stage=2 (Q double-buffer) | Single Q stage | LOW | 1 week (register-limited) |
| Named barriers (fine-grained sync) | __syncthreads only | MEDIUM | 3 days |
| LPT causal scheduling | Naive tile ordering | LOW | 2 days |

---

## 6. Implementation Plan with Priorities

### Phase A: Register Pressure Reduction (HIGHEST PRIORITY)
**Goal**: Enable 2 CTAs/SM occupancy for decode kernel
**Technique**: D-dimension PV tiling (split D=256 into 2×128 passes)

1. Compute QK^T for full D=256 → S in registers (64 regs)
2. Run softmax → P in registers
3. Store P to SMEM (99KB allows 128×128×2B = 32KB per tile)
4. PV pass 1: O_acc[D=0:128] using P from SMEM (64 regs)
5. PV pass 2: O_acc[D=128:256] using P from SMEM (64 regs)
6. Peak O_acc: 64 floats/lane instead of 128 → ~191 regs → 2 CTAs/SM

**Risk**: P caching in SMEM adds latency. Must verify net positive.
**Validation**: ncu occupancy check + A/B benchmark vs current V3.

### Phase B: Warp-Specialized Prefill (HIGH PRIORITY)
**Goal**: Extend V3 warpspec pattern to prefill kernel
**Technique**: 3-role warp specialization

- Load warps (1-2 warps): cp.async K/V tiles, signal via named barrier
- QK+softmax warps (2-4 warps): mma.sync QK^T, online softmax, P→SMEM
- PV warps (2-4 warps): Read P from SMEM, mma.sync PV, accumulate O

**Key design**: Use `bar.arrive`/`bar.sync` named barriers for phase sync.
setmaxregister to partition registers per role.

**Risk**: Synchronization overhead may negate benefits for small tiles.
**Validation**: Compare vs monolithic prefill on W2 workload (32K/1K, c=4).

### Phase C: FP8 P with Rescale Threshold (HIGH PRIORITY, LOW EFFORT)
**Goal**: Improve PV MMA efficiency via FP8 P with proper dynamic range
**Technique**: Port FA4's `rescale_threshold` concept

- After softmax, P ∈ (0, 1]. FP8 e4m3 max = 448.
- rescale_threshold = 8.0 (BF16) or 0.0 (FP8 input)
- Scale P by 2^(max_offset + threshold) before FP8 cast
- Top probabilities saturate but FP32 denominator counts them fully

**Risk**: LOW — pure software change, numerically validated in FA4.
**Validation**: Cosine similarity vs BF16 reference ≥ 0.9999.

### Phase D: Persistent Tile Scheduler (MEDIUM PRIORITY)
**Goal**: Eliminate per-tile launch overhead for long sequences
**Technique**: Software atomicAdd work counter

- Launch 188 CTAs (one per SM)
- Each CTA loops: `tile_idx = atomicAdd(&counter, 1)`
- Decode (batch, head, m_block) from linear tile_idx
- For causal: LPT ordering (longest rows first)

**Risk**: LOW — well-understood pattern, FlashInfer uses similar.
**Validation**: Compare launch overhead on 128K+ sequences.

### Phase E: TMA Investigation (MEDIUM PRIORITY, UNCERTAIN)
**Goal**: Determine if SM120 supports cp.async.bulk (TMA)
**Technique**: Write minimal test kernel using cp.async.bulk on sm_120a

- If TMA works: Replace cp.async with TMA for K/V loads (higher BW)
- If TMA fails: Confirm cp.async is the ceiling, document

**Risk**: HIGH uncertainty — may not compile or may have errata.
**Validation**: Compile + run minimal TMA test on target hardware.

### Phase F: ex2 Emulation Tuning (LOW PRIORITY)
**Goal**: Offload exp2 from MUFU to ALU when MUFU is bottleneck
**Technique**: Port FA4's ex2_emu_freq/ex2_emu_res tuning knobs

- Replace some `ex2.approx.f32` with polynomial approximation
- Tune frequency per (causal, hdim) configuration
- FA4 reports +3.4-5.5% on B200 for causal hd128 FP8

**Risk**: LOW — pure software, accuracy-neutral per FA4 validation.
**Validation**: ncu MUFU utilization + end-to-end benchmark.

---

## 7. Risk Assessment

### High Risks
1. **Register pressure is fundamental**: D=256 with mma.sync requires ~230 regs.
   D-dim tiling helps but adds SMEM traffic. May not reach 2 CTAs/SM.
2. **No wgmma**: 4× more warp-level MMA instructions needed vs SM90.
   Synchronization overhead may dominate any warp-specialization gains.
3. **cp.async bottleneck**: Closure notes identify "cp.async KV-tile sequential
   wait chain" as THE bottleneck. Without TMA, this may be unfixable.
4. **Previous failures**: Three optimization experiments (GQA=3, D-parallel,
   triple-buffer) all proved WORSE. The kernel may be near its architectural
   ceiling with mma.sync + cp.async.

### Medium Risks
5. **Closure protocol**: The sm120-flash-attention project was formally closed
   with a "stop-loss" protocol. Reopening requires evidence of ≥5% end-to-end
   improvement potential. FA4 techniques may not clear this bar.
6. **kekzl/imp precedent**: Even with full MXFP4 FMHA, they achieve only
   ~18 tok/s on dense 27B. The problem may be inherently hard on SM120.
7. **TMA uncertainty**: If TMA IS available on SM120, it changes the calculus
   significantly. If not, cp.async is the ceiling.

### Low Risks
8. **FP8 P numerical stability**: Well-validated in FA4, low risk.
9. **Persistent scheduler overhead**: ~1-2%, well-understood.
10. **ex2 emulation accuracy**: FA4 validates accuracy-neutrality.

---

## 8. Strategic Recommendation

### The Core Question
The sm120-flash-attention project was closed because custom kernels are 14.7%
SLOWER than native FlashInfer. Can FA4 techniques reverse this?

### Assessment
FA4's innovations fall into two categories:

**Category 1: Hardware-dependent (NOT portable)**
- tcgen05 MMA, TMEM, 2-CTA cluster, CLC, 228KB SMEM
- These account for the MAJORITY of FA4's performance advantage
- SM120 cannot replicate these

**Category 2: Algorithmic (PORTABLE)**
- Warp specialization, FP8 P rescaling, persistent scheduling, ex2 tuning
- These provide INCREMENTAL gains (5-15% each, optimistic)
- The existing kernel already has partial warp specialization (V3 decode)

### Honest Estimate
Best-case improvement from porting ALL portable FA4 techniques:
- Decode: 10-20% improvement (warp spec + D-tiling + FP8 P)
- Prefill: 5-15% improvement (warp spec + persistent + FP8 P)
- End-to-end: 5-12% improvement

This MIGHT close the 14.7% gap, but the probability is ~30-40% given:
- Previous optimization attempts all failed
- cp.async wait chain is the identified bottleneck (FA4 techniques don't fix this)
- Register pressure is fundamental without TMEM

### Recommended Path
1. **First**: TMA investigation (1 day). If TMA works on SM120, everything changes.
2. **If no TMA**: D-dimension PV tiling (Phase A) is the highest-leverage single change.
3. **If Phase A shows ≥5% kernel-level gain**: Proceed with Phase B+C.
4. **Stop-loss**: If Phase A shows <3% gain, the architectural ceiling is confirmed.
   Accept native FlashInfer as the production path and redirect effort elsewhere.

### Alternative: Adopt kekzl/imp Approach
Rather than porting FA4 techniques to the existing kernel, consider adopting
kekzl/imp's MXFP4-first architecture (packed-M, hardware NVFP4 MMA throughout).
Their fragment layout reverse-engineering solves the hardest undocumented problem.
But their dense-27B performance (~18 tok/s) suggests this path also has limits.

---

## Appendix A: FA4 Repo File Map (SM100/SM120 relevant)

```
flash_attn/cute/
├── flash_fwd_sm100.py      # 3202 lines — THE FA4 forward kernel (tcgen05, TMEM, 16 warps)
├── flash_fwd_sm120.py      # 60 lines — SM80 shim with 99KB SMEM check
├── flash_fwd_sm90.py       # SM90 wgmma kernel (producer/consumer warp spec)
├── flash_fwd.py            # Base class + SM80 implementation
├── flash_bwd_sm120.py      # SM80 backward shim
├── sm100_hd256_2cta_fmha_forward.py  # 2-CTA dedicated hd256 kernel
├── tile_scheduler.py       # CLC + static + dynamic schedulers
├── softmax.py              # Softmax + SoftmaxSm100 (rescale_threshold, ex2_emu)
├── blackwell_helpers.py    # SM100 utilities
├── mma_sm100_desc.py       # tcgen05 MMA descriptors
├── named_barrier.py        # Named barrier infrastructure
├── paged_kv.py             # Paged KV manager
├── pack_gqa.py             # GQA M-dimension packing
└── interface.py            # Dispatch: arch//10 → kernel class
```

## Appendix B: Key Register Budget Numbers (SM100 FA4)

```python
# From _TUNING_CONFIG in flash_fwd_sm100.py:
# (use_2cta, is_causal, hdim, is_sm103) → {num_regs_softmax, num_regs_correction}
(True, False, 128, False): {"num_regs_softmax": 176, "num_regs_correction": 88}
(False, True, 128, False): {"num_regs_softmax": 192, "num_regs_correction": 72}
(True, False, 256, False): {"num_regs_softmax": 256, "num_regs_correction": 160}
# num_regs_other = 512 - num_regs_softmax*2 - num_regs_correction
# For hd256: num_regs_other = 32 (fixed)
```

## Appendix C: Existing Kernel Performance Data

From closure notes (2026-07-16):
- Custom vs native: ms/draft +14.7% (10-20%), ms/acc_tok +8.6% (5-13%) — custom SLOWER
- Prefill v2: 18.7-20.75% slower (dense), 27-47% slower (chunked)
- Decode nativefp8: 2.4-3.0× slower at kernel level
- V3 warpspec decode: Experimental, not benchmarked against native
- ncu decode: occupancy 21.4%, Compute(SM) 60.79%, Memory 49.51%
- Power: 299.71W (ceiling), SM clock 1695-1710MHz (vs GEMM 1500-1560MHz)
