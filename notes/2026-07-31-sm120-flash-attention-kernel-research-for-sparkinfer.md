# sm120-flash-attention kernel research → SparkInfer paged decode applicability (2026-07-31)

Deep-dive of `/home/bot/project/sm120-flash-attention` (custom CUDA flash-attention
for Qwen3.6-27B on SM120, research line formally closed 2026-07-16, final KWIDE win
landed 2026-07-21) to extract techniques applicable to our SparkInfer paged decode
kernel (33.1% of GPU time at 64K, 384 calls, 10.27ms/call).

Sources: `kernel/csrc/flash_attn_sm120.cu` (14K lines), commits `7d1937d` (KWIDE+V272),
`bea946d` (K-only 16B A/B), `0f56006` (decisive split-KV 3-way A/B),
`notes/attention-kernel-research-closure.md`, `notes/2026-07-20-kernel-optimization-findings.md`,
`notes/perf-cleanup-bank-conflicts.md`, `experiments/`, plus SparkInfer
`forward_paged.py`/`traits.py`/`planner.py` and our own `notes/2026-07-22-a6-attention-split-k-investigation.md`,
`notes/2026-07-27-dflash-bandwidth-roofline-moe-gemm-attention.md`,
`notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md`.

Context delta: sm120-fa targeted D=256, GQA 24:4 (TP=2), custom ldmatrix-based MMA kernel.
Our production: D=128, GQA 48:8 full / 72:8 SWA (TP=1), FP8 KV, page_size=64, SparkInfer CuTe-DSL kernel.

## 1. cp.async patterns (16-byte vs 4-byte) and measured impact

**The single biggest measured win in the whole research line.** The nativefp8 decode
kernel loaded K via 16B `cp.async.cg` (row stride 272, 16B-aligned) but V via 2×4B
`cp.async.ca` (row stride 260, not 16B-aligned). Commit `7d1937d` padded V_SMEM_STRIDE
260→272, enabling 16B cp.async for V too:

```
Micro-benchmark (qo=4, 128K, c=4, splits=32):
  No-KWIDE:   1.540ms, 174.3 GB/s
  KWIDE+V272: 0.988ms, 271.7 GB/s   (1.56x / +55.9% kernel speedup)
E2e 128K/c=4 MTP K=3 warm: 183.43 → 222.44 tok/s (+21.3%)
Cumulative from project baseline: 104.7 → 222.44 tok/s (+112.5%)
```

Key details:
- `cp.async.cg.shared.global [dst],[src],16` (cache-global) vs `cp.async.ca...4`
  (cache-all). 16B halves instruction count per chunk AND raises memory throughput.
- **Hard constraint**: cp.async.cg 16B requires 16-byte alignment at EVERY row start.
  NAT_STRIDE=260 hit a real "misaligned address" launch failure (row×260 only 4B-aligned)
  — caught by actual compile-and-run, not assumed.
- **K-only widening was NOT enough**: earlier A/B (`bea946d`) widened K only (V stayed
  4B): +2.76%/+3.12%/+3.16%/+4.19% on four prefill shapes — below the 5-10% gate, not
  merged. The 1.56x required BOTH K and V wide.
- **Warp-uniform bug hazard**: widening to 16B changes lane→row mapping (32 lanes × 16B
  = 512B = 2 rows at D=256), breaking the "one warp per row" invariant used to broadcast
  physical page numbers via `shfl_sync` — half a warp got wrong page ids → OOB →
  `cudaDeviceSynchronize()` hang (kv_len=577/page_size=16). Fix: per-thread independent
  `page_ids` reads in the wide branch. Anyone widening a paged-KV load path must re-audit
  page-id broadcasting.
- SparkInfer already uses 16B cp.async (`cp.async.cg.shared.global.L2::128B`, note the
  **L2 prefetch hint** sm120-fa never used) in `_issue_paged_kv_cp_async_64x128` and TMA
  (`cp.async.bulk`) plane loads — this lesson is already baked in there.
- TMA (`cp.async.bulk`) was **rejected** by sm120-fa for paged KV: "strided layout
  (KVH×D=1024 between tokens) makes TMA inefficient"; a `decode_tma_load.cuh` helper
  (mbarrier-based, one bulk load per page vs ~64 cp.async4) was built but never reached
  production. SparkInfer makes TMA work by constructing per-page staged plane source
  tensors — viable only because its cache layout is TMA-friendly.

## 2. SMEM layout techniques and bank conflict fixes

**Stride padding for 16B alignment + bank-shift analysis** (V272):
- 272/4 = 68 words/row, 68 % 32 = 4 → 4-bank shift per row (good spread).
- Stride 256 → 64 % 32 = 0 → **8-way bank conflict** (rejected).
- Stride 260 → 1-bank shift, but not 16B-aligned (blocks cp.async16).
- Cost: +768 bytes SMEM. Directive recorded: "Do not change V_SMEM_STRIDE without
  verifying 16-byte alignment and bank conflict pattern."

**Dedicated bank-conflict cleanup pass** (`perf-cleanup-bank-conflicts.md`) — the most
important lesson is the *honest negative*:
- fp8kv kernel: dominant conflict = V transpose reads, row_stride 32B, 2-way, 83.7% of
  total. Fix: pad 32→48B. Result: excess wavefronts −83.7%, **wall-clock FLAT (+0.2%)**.
  Occupancy pinned at 16.66% (8 warps/SM) — bank conflicts were NOT the binding constraint.
- nvfp4kv kernel: dominant conflict = two-level K quantize reads, 4-way, 95.1% of total.
  Fix: transpose into `K_interleaved` + register-cached single read (v2, after v1
  regressed and was re-diagnosed). Result: excess wavefronts −91.9%, wall-clock −0.2%
  (small real win). Same occupancy ceiling.
- Methodology worth stealing: per-source-line conflict attribution via
  `ncu --page source --print-source cuda,sass --csv --metrics
  derived__memory_l1_wavefronts_shared_excessive,memory_l1_wavefronts_shared,memory_l1_wavefronts_shared_excessive_ideal`
  (generic bank counters are kernel-wide only). And: `--set full` replay perturbs
  Duration — use `--metrics gpu__time_duration.sum` alone for before/after timing.
- Python bank-conflict modeling matched the read side but NOT always the write side;
  real hardware was more forgiving. Treat models as hypothesis generators, ncu as truth.

**SparkInfer's approach**: XOR-swizzled "permuted-128b" SMEM layout
(`dst = row*8 + (vec ^ (row%8))`) — bank-conflict-free by construction and shaped to
feed the cooperative FP8→BF16 widening stage directly. No padding needed.

## 3. Tile size / warp count choices for SM120

sm120-fa production kernel: BLOCK_Q=64, BLOCK_KV=32, 8 warps/CTA (4 Q slices × warp
pairs splitting the KV range, merged once after the loop), 96KB SMEM vs 99KB opt-in,
Ampere-style `mma.sync m16n8k16` + `ldmatrix` (**sm120 has no wgmma/tcgen05**), Q staged
to registers once and kept resident for the whole KV loop, P repacked directly from
QK^T accumulator registers into PV A-fragments (no SMEM round trip — the C-fragment of
m16n8k16 has the same per-lane layout as half an A-fragment).

**Three warp/pipeline experiments falsified at the real MTP shape (qo=4/128K/c=4)**:

| Experiment | Hypothesis | Result | Why it failed |
|---|---|---|---|
| GQA_GROUP 6→3 | halve Q regs → 2 CTAs/SM | 3.89ms vs 3.23ms (1.9× slower) | Regs still 255 — pressure is O_acc's D-dim (~128 floats/lane), GQA-independent; grid.y doubled → KV reads 2× |
| D-parallel (8 warps split D) | more warps hide latency | 3.32ms (1.6× slower) | barrier stalls 3.22→0.62 BUT DRAM 57→26%, SM 35→49%: QK^T recomputed 8× → compute-bound |
| Triple-buffered cp.async | deeper pipeline | 2.51ms (1.2× slower) | more SMEM → occupancy down; double-buffering already sufficient (small per-tile compute) |

Core conclusion: the bottleneck was **neither warp count nor occupancy** — it was the
sequential cp.async KV-tile wait chain + V-transpose SMEM round trip (memory latency not
hidden). Double-buffered M-tile parallelism was already near-optimal for that structure.
Supporting: decode warp-count probes (2/4/8 warps) all net-negative (occupancy 1→2 but
slower); GQA-in-M MMA packing 1.6×/3.6× slower (regs 245 → occupancy 3→1 blocks/SM);
ldmatrix replacement of scalar fragment loads cut instructions 13.6% but was time-flat
(per-instruction latency higher, no spare warps to hide it).

**SparkInfer decode choices** (generic FP8 path, our shape): cta_tile_q=16,
num_warps_q=1, num_warps_kv=4, num_stages=**1** (forced for FP8 or multi-KV-warp),
ctas_per_sm=**1** (heuristic for FP8/page64/gqa6/batch≤4; BF16 gets 4-6). The Laguna
specializations instead use: N32 KV stage (exact_num_mma_kv=2 → one physical 128-token
page per iteration), cooperative FP8→BF16 widening done once per CTA and reused across
all 4 query warps ("removes per-fragment FP8 conversion state"), compact_sync_rows =
gqa_group_size (sync only GQA rows, not full tile), role-specialized decode (+1 producer
warp: `launch_warps_kv = num_warps_kv + 1`), 2 CTAs/SM when SMEM admits, verify mode
N64 stage (exact_num_mma_kv=4: "halves live probability fragment... trades one extra
loop iteration per page for substantially lower register and SMEM pressure").

## 4. Split-KV: why ~0% benefit (and why that does NOT transfer to decode)

The decisive 3-way A/B (`0f56006`) was about **prefill/chunked-prefill shapes**:
native auto-split vs native no-split vs our no-split v2, weighted by real production
call frequency (early 15.58% / mid 7.79% / late 76.62%):

```
split-KV benefit: early 0.0%, mid +0.4%, late +0.5%, full +0.0%, weighted +0.48%
(gates: ≥10% kernel AND ≥2% e2e — failed by 1-2 orders of magnitude)
```

Root cause verified via ncu: at those shapes the grid already vastly exceeds 188 SMs
(early 592 blocks/~3.15 waves, late 3000 blocks/~16 waves), so native's own heuristic
already chose no split — auto and disabled produced IDENTICAL grid=(750,1,4) and
IDENTICAL duration (29.32-29.37ms). "FlashInfer splits, we don't, that's the gap" was
**refuted**. This formally closed the research line (custom kernel ended 14.7% slower
ms/draft, 8.6% slower ms/accepted-token than native FlashInfer).

**Why this does not transfer to our decode**: decode at batch 1-4 × 8 KV heads = 8-32
CTAs without splitting — far below 188 SMs, so split-KV is essential for SM fill. Our
own A6 investigation measured split-size sensitivity directly (batch=4, qo=4, FP8):

| KV len | split=2048 | split=4096 | verdict |
|---|---|---|---|
| 32K | 0.329ms | 0.321ms | noise |
| **64K** | **0.503ms (1068 GB/s)** | 0.623ms (861 GB/s) | **2048 wins by 19.3%** |
| 128K | 1.084ms | 1.064ms | noise (4096 marginally) |

sm120-fa found the same sensitivity for their shape (4096 optimal at 128K; 8192/2048/
1024/512 all slower). Global split=2048/max_splits=128 is NOT the answer: +70%
regression at 32K from workspace/reduction overhead. The win is **adaptive, context-aware
split selection** (≤32K: 4096; 64K: 2048; ≥128K: 4096).

## 5. Top 3 techniques most applicable to SparkInfer paged decode

### #1 — Unlock the specialized decode path for our real shapes (est. attention up to ~2.7×, ~5% e2e; HIGH confidence in headroom, MEDIUM in realization cost)

Our own roofline work (2026-07-27) already quantified it: at 64K verify, attention runs
at **~37% of measured bandwidth ceiling** (480 of ~1300 GB/s) while MoE is at ~100% and
dense GEMM 86-95%. Root cause: all 5 Laguna kernel specializations in SparkInfer
`traits.py` require `num_kv_heads==4` AND `page_size==128`; production TP=1 runs
`num_kv_heads==8`, `page_size==64` → every path (decode/extend/verify) falls back to the
generic kernel. The sm120-fa research independently validates the *mechanism* of the win:
their cumulative +112.5% came from exactly the structural features the specialized
SparkInfer path already embodies (role-specialized load/compute split, V-transpose
elimination via layout, wide vectorized loads, compact GQA-scoped sync, residency-aware
SMEM budgeting), and SparkInfer's own published numbers for the kv_heads=4 shape were
2.1×/1.4×.

Two-part action (both already scoped in our notes):
- **Downstream (this repo)**: page_size 64→128 migration — relax `laguna.py` block_size
  check, recompute `blocks_per_slot` 1088→~544 to hold per-slot capacity, re-verify SWA
  ring math, full acceptance-rate + CUDA-graph regression (history: a KV-layout change
  once crashed acceptance to 0.13% via stale CG addresses). Correctness at page_size=128
  already verified for our real shapes (cos ≥ 0.999991 all four scenarios).
- **Upstream (sparkinfer traits generalization)**: match on `gqa_group_size` (6/9 —
  invariant under TP) instead of absolute head counts. CTA-internal tuning is portable
  (`cta_tile_q` selection uses `packed_qo_len = qo_len × gqa_group_size`); only
  whole-grid occupancy tuning (`resolve_decode_graph_ctas_per_sm`) needs re-validation
  since kv_heads 4→8 doubles independent CTA groups.
- Expectation management: page_size=128 alone does NOT grant the specializations (they
  still require kv_heads=4); the traits generalization is the actual key. Re-measure with
  the bandwidth-roofline method after; do not extrapolate the 2.1×/1.4×.

### #2 — Adaptive split-K for decode (est. ~19% attention at 64K, ~5% e2e; HIGH confidence, measured)

Already measured in A6: at our single most expensive attention shape (64K), split=2048
beats the production split=4096 by 19.3% (1068 vs 861 GB/s — 32 splits fill 188 SMs
better than 16). Implementation options: context-aware split selection in the runner, or
multiple CUDA-graph buckets keyed by kv_len. Keep the global default (split=4096,
max_splits=64) for ≤32K where the 128-split workspace costs +70%. sm120-fa's split-KV
~0% result is explicitly a prefill phenomenon and does not argue against this.

### #3 — Re-examine FP8-decode residency: num_stages=1 + ctas_per_sm=1 (est. 5-15% kernel-level IF residency lifts; LOW-MEDIUM confidence, needs ncu)

The generic FP8 decode path for our shape runs num_stages=1 (forced whenever
`num_warps_kv > 1 or kv_is_fp8`) and the planner heuristic pins ctas_per_sm=1 for
FP8/page64/gqa6/batch≤4 — while the BF16 path gets 3 stages and 4-6 CTAs/SM. At D=128
(versus sm120-fa's D=256, which was architecturally pinned to 1 CTA/SM at 96KB SMEM),
there should be SMEM room for deeper pipelining or 2 CTAs/SM; sm120-fa's whole body of
negative results (triple-buffering, warp-count, D-parallel) says do NOT guess here —
the win exists only if ncu shows the KV wait chain (long-scoreboard) as the top stall
AND the SMEM budget admits the change without dropping residency. Concrete first step:
ncu the current generic FP8 decode at 64K for stall reasons + achieved occupancy, then
A/B `num_stages` and `ctas_per_sm` overrides. Also in this bucket, cheap and already
identified: wire the per-head K/V descale (kernel supports 2D `[batch,heads]` descale at
`forward_paged.py:5429`) to unblock `SPARKINFER_TURBO_ATTN` (native FP8 QK/PV MMA,
measured +6.2% at 64K, currently default-OFF due to the code-4K acceptance regression
from per-tensor scale precision loss).

**Explicitly NOT recommended** (sm120-fa falsified, transferable as anti-patterns):
triple+ buffering at small per-tile compute; more warps as a latency-hiding strategy
when the bottleneck is the load wait chain; GQA-group reduction as a register lever
(O_acc's D-dimension dominates, not GQA); bank-conflict fixes as a wall-clock strategy
while occupancy-bound (−91.9% conflicts → flat time); TMA for paged KV unless the cache
layout is TMA-staged (SparkInfer's is; sm120-fa's wasn't).

## 6. SM120-specific hardware constraints discovered

- **No wgmma/tcgen05 on sm_120**: Ampere-style `mma.sync.aligned.m16n8k16` + `ldmatrix`
  is the MMA surface (this kernel family); ldmatrix has higher per-instruction latency
  than scalar loads — only pays off with spare warp-level parallelism.
- **SMEM**: 99KB (101,376B) opt-in per CTA; 2 CTAs/SM requires 2×launch_smem ≤
  `shared_memory_per_multiprocessor`. sm120-fa's D=256 decode used 67,840B (paged nativefp8)
  to 96KB (prefill) and was pinned at 1 CTA/SM = 16.66% occupancy — an architectural
  ceiling, not a bug. D=128 halves the K/V stage requirement — our headroom.
- **Registers**: hard cap 255/thread; output accumulator (O_acc, D-dim) dominates —
  ~128 floats/lane at D=256, GQA-independent. `setmaxnreg` ceiling 224 used in warpspec
  experiments; O_acc D-halving got 254→189 regs at +23% instructions (unmeasured trade).
- **cp.async.cg 16B requires 16B alignment at every row start** — misalignment is a hard
  launch failure, not UB. Use `cp.async.ca` 4B for non-aligned strides.
- **L2::128B prefetch hint** available on cp.async (`cp.async.cg.shared.global.L2::128B`)
  — SparkInfer uses it, sm120-fa didn't.
- **TMA works on sm_120** (mbarrier init/arrive_expect_tx/try_wait parity pattern in
  `decode_tma_load.cuh`) but is only efficient for contiguous/staged layouts; per-page
  plane staging (SparkInfer) makes it viable for paged KV.
- **NVFP4 hardware cvt instructions unavailable**: `cvt.rn.satfinite.e2m1x2.f32` /
  `cvt.rn.f16x2.e2m1x2` fail to compile on sm_120 AND sm_100 with CUDA 13.3
  ("Feature not supported on .target") — software E2M1 encode/decode costs ~46.6% scalar
  instructions in NVFP4 kernels; NVFP4-KV measured 2.46×/1.92× slower than FP8-KV.
- **Measured (not spec) bandwidth**: RTX PRO 6000 Blackwell Max-Q ≈1300 GB/s HBM
  (cold-cache, L2 flushed); L2 = 128MiB — tight-loop benchmarks without L2 flush produce
  false numbers (this bit the project twice).
- **Power wall**: attention kernel pins 299.71W but runs SM clock 1695-1710MHz, 9-13%
  higher than pure GEMM (1500-1560MHz) — the 277.94 TFLOPS SOL denominator used in
  reports is conservative for attention workloads.
- **188 SMs**: split-K sizing should target enough CTAs to fill waves; decode at batch≤4
  needs split-KV to reach meaningful SM occupancy at all.
