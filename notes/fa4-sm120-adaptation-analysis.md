# FA4 SM100→SM120 Adaptation Analysis

> Date: 2026-07-31
> Source: Dao-AILab/flash-attention @ HEAD (cloned 2026-07-31)
> Target: SM120 (RTX 5090 / Blackwell GeForce / DGX Spark)

---

## 1. FA4 Source Structure

### Key Files (flash_attn/cute/)

| File | Lines | Role |
|------|-------|------|
| `flash_fwd_sm100.py` | 3202 | **Primary SM100 forward kernel** — warp-specialized, TMEM, 2-CTA, persistent |
| `flash_fwd_sm120.py` | 61 | **SM120 "kernel"** — thin subclass of SM80, adjusts SMEM cap only |
| `flash_bwd_sm100.py` | 4172 | SM100 backward |
| `flash_bwd_sm120.py` | 55 | SM120 backward — same SM80 subclass pattern |
| `sm100_hd256_2cta_fmha_forward.py` | 1918 | Dedicated hd256 2-CTA forward (fixed tile 128×128) |
| `tile_scheduler.py` | 1650 | Persistent/CLC/Static tile schedulers |
| `softmax.py` | 710 | Online softmax (SM80 register-based + SM100 TMEM-based) |
| `blackwell_helpers.py` | 1115 | tcgen05 MMA PTX wrappers, SMEM descriptor construction |
| `pipeline.py` | 402 | Pipeline abstractions (TMA↔UMMA, async↔UMMA bridges) |
| `interface.py` | 3146 | Dispatch logic — routes arch to kernel class |
| `flash_fwd_sm90.py` | 1545 | SM90 (Hopper) forward — TMA + warpgroup MMA |
| `flash_fwd.py` | 1243 | SM80 forward — mma.sync baseline |
| **Total cute/ directory** | **48,666** | |

### Critical Finding: Official SM120 Support is Minimal

The FA4 `flash_fwd_sm120.py` is **61 lines** — a trivial subclass of `FlashAttentionForwardSm80`:
- Forces `self.arch = Arch.sm_80` (SM80 MMA instructions)
- Overrides `can_implement()` to check 99 KB SMEM (vs SM80's 163 KB)
- No TMA, no warp specialization, no persistent scheduling, no 2-CTA, no TMEM

The interface dispatches SM120 with:
- 128 threads (4 warps), tile 128×64 (D>64) or 128×128 (D≤64)
- No split-KV, no paged KV, no block sparsity
- `num_splits == 1` asserted

**This means the official FA4 SM120 path leaves ~70% of SM120's hardware on the table.**

---

## 2. SM100 Kernel Architecture Deep Dive

### 2.1 Warp Specialization (12 warps / 384 threads per CTA)

| Warp IDs | Role | Register Budget | Function |
|----------|------|-----------------|----------|
| 0-3 | Softmax-0 | 176-256 | Online softmax for Q-stage 0, reads S from TMEM, writes P to TMEM |
| 4-7 | Softmax-1 / Load | 176-256 | Q-stage 1 softmax OR KV loading (non-TMA fallback) |
| 8 | MMA | 32-48 | Issues tcgen05 UMMA instructions (QK^T and PV) |
| 9 | Load / TMA | 32-48 | Issues TMA loads for Q/K/V |
| 10-11 | Epilogue / Empty | 32-48 | O store (TMA S2G) or idle; warp 10 = CLC scheduler if enabled |

Key insight: **The MMA warp is a single warp** issuing asynchronous UMMA instructions.
All tensor-core compute is serialized through one warp's instruction stream, but UMMA
itself is async — the warp fires-and-forgets, then waits on pipeline barriers.

### 2.2 TMEM (Tensor Memory) Layout

SM100 has 512 columns of TMEM (per CTA pair in 2-CTA mode):

```
Offset:  0        128       256       384       512
         |---S0---|---S1---|---O_stage0---|---O_stage1---|
         (128 col) (128 col) (128 col)     (128 col)
```

- **S (scores)**: QK^T accumulator, 2 stages for pingpong (S0, S1)
- **O (output)**: PV accumulator, 2 stages for Q-stage overlap
- **P (probabilities)**: Reuses S's TMEM columns (offset + n_block_size/2), after softmax converts S→P in-place
- **row_max/row_sum**: Stored in TMEM at `tmem_vec_offset` (overlaps S start)

TMEM is the **MMA accumulator** — UMMA writes directly to TMEM, softmax warps read from
TMEM via `ld.tmem`, and the P@V MMA reads P from TMEM (operand source = TMEM).

### 2.3 2-CTA Pingpong (Cluster Shape 2×1)

Two CTAs form a cluster and share TMEM via `tcgen05` 2-CTA instructions:
- `cta_group_size = 2`, `cluster_shape_mn = (2, 1)`
- Each CTA owns `m_block_size` rows of Q; the 2-CTA MMA instruction spans both
- MMA tiler M = `2 × m_block_size` (e.g., 256 rows total)
- **Coordination**: Pipeline barriers (`PipelineUmmaAsync`, `PipelineTmaUmma`) with
  `cta_layout_vmnk` encoding the cluster topology
- **Data exchange**: TMEM is shared within the cluster; SMEM is per-CTA but K/V tiles
  are multicast via TMA (`tma_copy_bytes *= cta_group_size`)

The "pingpong" refers to alternating S stages: while MMA computes QK^T into S1,
softmax warps process S0 and write P0 back to TMEM, then MMA does P0@V.

### 2.4 Persistent Tile Scheduler

Three modes in `tile_scheduler.py`:
1. **SingleTileScheduler**: 1 tile per CTA (non-persistent, grid = total_tiles)
2. **StaticPersistentTileScheduler**: Grid-stride loop, CTA count = SM count
3. **CLC (Command Launch Control)**: Hardware work-stealing via `ClcDynamicPersistentTileScheduler`
   - Uses `PipelineClcFetchAsync` for async work acquisition
   - Producer warp issues CLC query → mbarrier signals → consumer gets coordinates
   - Only available with TMA KV (`use_tma_KV=True`)

For SM100 forward: `is_persistent=True` by default, using static persistent or CLC
depending on `use_clc_scheduler` flag.

### 2.5 FP8/Low-Precision Softmax

- `SoftmaxSm100` class: reads S from TMEM, applies online softmax, writes P back to TMEM
- FP8 path: `descale_tensors` (q_descale, k_descale, v_descale) applied during softmax
- P is cast to input dtype (FP8/BF16) before P@V MMA
- `rescale_threshold` tuning: allows P to exceed dtype max (saturates top probabilities)
- exp2 emulation: `_TUNING_CONFIG` controls `ex2_emu_freq` to offload MUFU pressure

---

## 3. SM100-Specific Instruction Inventory → SM120 Substitutions

| SM100 Feature | PTX/Hardware | SM120 Status | Substitution |
|---------------|-------------|--------------|--------------|
| **tcgen05 UMMA** | `tcgen05.mma` (async, TMEM accumulator) | ❌ **Not available** | `mma.sync.aligned.m16n8k16` (SM80-era, register accumulator) |
| **TMEM** (512 col) | `ld.tmem`, `st.tmem`, TmemAllocator | ❌ **Not available** | Registers + SMEM staging |
| **TMA** (Tensor Memory Accelerator) | `cp.async.bulk.tensor` | ✅ **Available** (SM120 has TMA) | Direct reuse |
| **2-CTA cluster MMA** | `cluster_shape=(2,1)`, multicast TMA | ⚠️ **Partial** — SM120 supports clusters but not 2-CTA UMMA | Single-CTA; TMA multicast still works for K/V sharing |
| **CLC scheduler** | `ClcDynamicPersistentTileScheduler` | ❌ **Not available** (requires SM100 CLC hardware) | Static persistent (grid-stride) or software work-stealing |
| **SMEM capacity** | 227 KB (SM100) | ⚠️ **99 KB** (SM120) | Smaller tiles, fewer stages |
| **Warp specialization** | 12 warps, setmaxregister | ✅ **Available** | Reuse pattern with fewer warps (SMEM constraint) |
| **Pipeline barriers** | mbarrier, arrive/wait | ✅ **Available** | Direct reuse |
| **exp2 emulation** | Software (no HW dependency) | ✅ **Available** | Direct reuse |
| **NVFP4/MXF4 MMA** | `tcgen05.mma...kind::mxf4nvf4` | ❌ **Not available** (requires tcgen05) | `mma.sync...kind::mxf4nvf4` if available on SM120, else FP8 e4m3 |
| **SMEM swizzle** | 128B swizzle | ✅ **Available** | Direct reuse |
| **setmaxregister** | Per-warp register budget | ✅ **Available** | Direct reuse |

### Critical Gaps (showstoppers for direct port):
1. **No TMEM** → accumulator must live in registers (255 reg/thread limit) or SMEM
2. **No tcgen05 UMMA** → must use `mma.sync` (SM80-era), which is synchronous and register-accumulating
3. **99 KB SMEM** (vs 227 KB) → fewer pipeline stages, smaller tiles
4. **No CLC** → must use software persistent scheduling

### What SM120 DOES have (advantages over SM80):
- TMA (huge bandwidth improvement over cp.async)
- 128B SMEM swizzle (bank-conflict-free)
- Warp specialization + setmaxregister
- mbarrier-based pipelines
- Higher clock / newer tensor cores (same ISA as SM80 mma.sync but faster execution)

---

## 4. Our Existing sm120-flash-attention Kernel: Gap Analysis

### What's Implemented (14,778 lines CUDA + 870 lines Python wrapper)

| Phase | Feature | Status |
|-------|---------|--------|
| Phase 1 | BF16 baseline, hd256, GQA, causal | ✅ Working |
| Phase 2 | FP8 e4m3 QK^T, FP8 KV | ✅ Working |
| Phase 3 | NVFP4 (e2m1 + ue4m3) QK^T and KV | ✅ Working |
| Phase 4 | Paged KV cache + varlen prefill | ✅ Working |
| Phase 4 R2 | Decode split-KV (flash-decoding) | ✅ Working |
| V2 | Tensor-core mxf4nvf4 decode | ✅ Working |
| V3 | Warp-specialized FP8-KV decode | ✅ Working |

### Performance vs Native FlashInfer (from closure doc)

- **Custom kernel is 14.7% SLOWER** (ms/draft), 8.6% slower (ms/accepted-token)
- Prefill: 18.7-20.75% slower
- Chunked-prefill: 27-47% slower
- Decode nativefp8 kernel-level: 2.4-3.0× slower (but end-to-end nativefp8 ON is ~3% faster)

### Root Causes Identified (from optimization findings)

1. **Register pressure**: 255 regs/thread maxed by O_acc (D=256 output accumulator)
2. **cp.async sequential wait chain**: KV tile loading is the bottleneck, not compute
3. **V transpose SMEM roundtrip**: Extra SMEM traffic for V layout conversion
4. **Single CTA/SM occupancy**: 255 regs × 256 threads = 65,280 > 65,536 reg file → 1 CTA/SM
5. **No TMA**: Uses cp.async (SM80-era) instead of TMA (available on SM120!)

### Why Research Was Closed (2026-07-16)

- Two "surpass native" claims (prefill v2, decode v2 nativefp8) both disproven on re-test
- Final A/B test: 0.0-0.5% improvement, far below 10% kernel / 2% e2e threshold
- All optimization levers tried: occupancy, register reduction, warp count, triple-buffering, GQA grouping — all net negative
- **Key insight from 2026-07-20 experiments**: bottleneck is memory latency (cp.async wait chain), not compute or occupancy

---

## 5. Community Backport Assessment

### BlackFlash (fms-zth/BlackFlash)

- **Target**: RTX 5060 Laptop (SM120), head_dim=64 only
- **Approach**: Handwritten FA2 with TMA + mma.sync + swizzle + double-buffer + warp specialization
- **Performance**: 31.45 TFLOPS (vs cuDNN 51.14) = **0.62× cuDNN** at scale
- **Architecture**: 9 warps (1 producer + 8 consumer), 2-stage double buffer, 64×64 tiles
- **Assessment**: Proves TMA + warp-spec works on SM120, but:
  - Only hd64 (our model needs hd256)
  - No GQA, no paged KV, no FP8/NVFP4
  - 0.62× cuDNN = not competitive
  - 518 lines CUDA (toy scale)

### PyPI / Other

- No `flash-attn-4-sm120` package exists
- No other significant community SM120 attention kernels found
- Rogala/AI_Attention: pre-compiled packages, not a kernel implementation

### Verdict: No usable community backport exists. BlackFlash validates the approach but is far from production.

---

## 6. Priority-Ordered SM120 Adaptation Plan

### Strategy: TMA-First Rewrite (not a port of SM100 kernel)

The SM100 kernel cannot be directly ported (no TMEM, no tcgen05, no CLC).
Instead: take FA4's **architectural ideas** (warp specialization, persistent scheduling,
pipeline overlap) and implement them with SM120-available primitives (TMA, mma.sync,
mbarrier, setmaxregister).

### Priority 1: TMA Integration (replaces cp.async) — **Highest ROI**
**Effort: 3-5 days**

Our kernel's proven bottleneck is the cp.async sequential wait chain. SM120 has TMA.
- Replace `cp.async.cg.shared.global` with `cp.async.bulk.tensor` (TMA)
- TMA provides: higher bandwidth, hardware address generation, multicast capability
- Requires: TMA descriptor setup (host-side), mbarrier-based completion signaling
- Expected impact: **20-40% decode speedup** (eliminates the identified bottleneck)
- Reference: BlackFlash proves TMA works on SM120; FA4's `PipelineTmaAsync` shows the pattern

### Priority 2: Warp Specialization (producer/consumer split) — **High ROI**
**Effort: 2-3 days**

Current kernel: all warps do load+compute (homogeneous).
Target: dedicated load warp(s) + dedicated compute warps.
- 1-2 producer warps: issue TMA loads, manage pipeline barriers
- 6-10 consumer warps: mma.sync compute (QK^T, softmax, PV)
- setmaxregister: producers get 48 regs, consumers get 200+ regs
- Pipeline: mbarrier full/empty signaling between producer and consumer
- Expected impact: **10-20% throughput** (overlap load and compute)
- Reference: FA4 SM100 warp layout, BlackFlash 1+8 pattern

### Priority 3: Persistent Kernel (grid-stride loop) — **Medium ROI**
**Effort: 1-2 days**

Current: launch one CTA per tile (high launch overhead for decode with many small tiles).
Target: launch `num_SMs` CTAs, each loops over multiple tiles.
- Reduces kernel launch overhead
- Enables work-stealing heuristics (LPT for causal)
- Simple grid-stride is sufficient (no CLC needed)
- Expected impact: **5-15% for decode** (many small tiles), minimal for prefill
- Reference: FA4 `StaticPersistentTileScheduler`

### Priority 4: SMEM Layout Optimization (swizzle + staging) — **Medium ROI**
**Effort: 2-3 days**

Current: likely has bank conflicts (no swizzle mentioned in our kernel).
Target: 128B swizzle for all SMEM tiles, optimized staging.
- SM120 has 99 KB SMEM → budget: Q(128×256×2=64KB) + K/V(2 stages × 128×256×2 = too much)
- Must use smaller tiles: 64×256 Q + 2×(64×256 K/V) = 32+32+32 = 96 KB (tight!)
- Or: 128×128 tiles with D-split (2 passes over D dimension)
- Expected impact: **10-20%** (bank conflicts can cost 2-4× on affected accesses)

### Priority 5: V-Transpose Elimination — **Medium ROI**
**Effort: 1-2 days**

Current: V requires SMEM roundtrip for layout conversion.
Target: Use `ldmatrix.trans` or TMA with transpose to load V in correct layout.
- Eliminates one full SMEM write+read cycle per KV tile
- Expected impact: **5-10%** (removes identified overhead)

### Priority 6: 2-CTA Cluster (TMA multicast for K/V) — **Low-Medium ROI**
**Effort: 3-5 days**

SM120 supports thread block clusters. While 2-CTA UMMA isn't available,
TMA multicast can share K/V loads across 2 CTAs:
- 2 CTAs process different Q heads (GQA: same KV head)
- TMA multicast loads K/V once, delivers to both CTAs' SMEM
- Halves KV bandwidth requirement
- Expected impact: **15-25% for GQA-heavy workloads** (our 24:4 ratio = 6× reuse)
- Risk: cluster setup overhead may negate gains for small batches

### Priority 7: FP8 Softmax with Rescaling — **Low ROI (already have FP8)**
**Effort: 1 day**

Our kernel already has FP8 QK and FP8 KV. FA4's contribution:
- Per-row-max P requantization (we have this)
- `rescale_threshold` for better FP8 P precision
- exp2 emulation tuning
- Expected impact: **1-3% accuracy improvement** at same speed, or **2-5% speed** at same accuracy

---

## 7. Effort Summary & Risk Assessment

| Priority | Task | Effort | Expected Speedup | Risk |
|----------|------|--------|-----------------|------|
| P1 | TMA integration | 3-5d | 20-40% | Medium (new API, descriptor setup) |
| P2 | Warp specialization | 2-3d | 10-20% | Low (well-understood pattern) |
| P3 | Persistent kernel | 1-2d | 5-15% | Low |
| P4 | SMEM swizzle + staging | 2-3d | 10-20% | Low |
| P5 | V-transpose elimination | 1-2d | 5-10% | Low |
| P6 | 2-CTA TMA multicast | 3-5d | 15-25% | High (cluster API maturity) |
| P7 | FP8 softmax tuning | 1d | 1-5% | Low |

**Total estimated effort: 13-21 days for full implementation**
**Critical path: P1 + P2 + P4 (7-11 days) for the majority of gains**

### Combined Theoretical Speedup (P1-P5): 40-80% over current custom kernel
### Gap to Native FlashInfer: Currently -14.7% → projected +10-30% with P1-P5

---

## 8. Key Architectural Decisions

1. **Do NOT port the SM100 CuTe DSL kernel** — it's fundamentally TMEM/tcgen05-dependent.
   Write CUDA C++ with TMA intrinsics instead (our existing codebase is CUDA C++).

2. **Do NOT use the official FA4 SM120 path** — it's the SM80 kernel with a SMEM check.
   Our existing kernel already surpasses it (we have FP8, NVFP4, paged KV, warp-spec decode).

3. **DO steal FA4's architectural patterns**:
   - Warp specialization topology (dedicated load/MMA/softmax/epilogue warps)
   - Pipeline barrier protocol (mbarrier full/empty, producer/consumer)
   - Persistent scheduling (grid-stride with LPT heuristic for causal)
   - TMA descriptor setup and multicast patterns

4. **Target tile size for SM120 hd256**: 64×128 (M×N) with 2 KV stages
   - Q: 64×256×2B = 32 KB
   - K: 2×(128×256×2B) = 128 KB → too much! Must split D or reduce N
   - **Better**: 64×64 tiles with D-split (2 passes), or 128×64 with 1 KV stage
   - **Best for decode**: 16×128 (small M for decode) with 2-3 KV stages

5. **Register budget**: With warp specialization, compute warps can use 200+ regs
   (producers use 48), solving the 255-reg-wall problem that blocked 2-CTA/SM occupancy.
