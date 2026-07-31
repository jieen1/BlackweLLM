# FlashAttention-4 → SM120 Portability Research

> Date: 2026-07-31 · Method: source-mining `fa4-latest` (Dao-AILab `flash-attn-4`,
> commit `c75d019`), CUTLASS 4.6.1, local `sm120-flash-attention`, `BlackFlash`,
> `flashinfer`. Web search was unavailable this session; all claims below are
> grounded in local source/docs with file references. Confidence tagged per section.

---

## 0. Executive summary

1. **The task's hardware premise needs one correction.** SM120 (RTX PRO 6000
   Blackwell, cc 12.0) does **not** use `wgmma` and does **not** use `tcgen05`.
   Its fast tensor-core path is **warp-level `mma.sync`** — including the new
   block-scaled `mma.sync.kind::f8f6f4` (FP8, atom **m16n8k32**) and
   `kind::mxf4nvf4.block_scale` (NVFP4, atom **m16n8k64**) — combined with
   **SM90-style TMA** and **warp specialization**. The local kernel's comment
   "sm120 has no wgmma/tcgen05" is **correct**; CUTLASS confirms it
   (`sm120_mma_tma.hpp` static-asserts the MMA is *not* a GMMA/wgmma op).
   *Confidence: high.*

2. **FA4's headline 3.1 PFLOPs/s on B200 comes from SM100-only hardware**
   (tcgen05 5th-gen tensor cores 2–4× Hopper throughput, TMEM, 2-CTA cluster
   MMA, CLC dynamic scheduling, 228 KB SMEM). **None of that exists on SM120.**
   The parts that *are* portable are the **algorithmic/software** ideas:
   warp specialization, MMA↔softmax overlap (pingpong), FP8 softmax with
   bounded online rescaling, persistent + dynamic tile scheduling, deep
   TMA pipelining. *Confidence: high.*

3. **There is already a working SM120 template for the portable subset.**
   CUTLASS ships `blackwell_geforce` CuTeDSL GEMMs with
   **persistent + pingpong + warp-specialized + TMA + block-scaled MMA**
   (`dense_blockscaled_gemm_persistent_pingpong.py`), and the local
   **BlackFlash** kernel proves TMA + warp-spec attention runs on this exact
   GPU class (~31 TFLOPS BF16). The port is *engineering*, not *research*.
   *Confidence: high.*

4. **The local kernel's measured bottleneck is exactly what these techniques
   fix.** Profiling (2026-07-20) shows decode is **memory-latency bound**:
   "cp.async KV-tile sequential wait chain + V-transpose SMEM roundtrip",
   ~544 GB/s ≈ 30% of peak. TMA + a dedicated DMA warp + deeper pipelining +
   V-transpose elimination is the direct remedy. *Confidence: high.*

5. **But the bar is high and the line was closed for a reason.** The custom
   kernel is currently **2.4–3.0× slower than native FlashInfer** on decode and
   **18.7–20.75% slower** on dense prefill; attention is the dominant gap
   (+65.5% vs native). The Amdahl target for the primary W2 workload is
   **S ≥ 1.374×** on the attention kernel for a 5% e2e win. Two prior
   "beat native" claims were overturned on strict re-test. A FA4-style rewrite
   is the *right* architecture but is multi-week and must be validator-gated.
   *Confidence: high on the numbers; medium on achieving the target.*

---

## 1. FA4 algorithm breakdown: SM100-specific vs portable

Source: `fa4-latest/flash_attn/cute/` (FA4 = `flash-attn-4` package, CuTeDSL).
`CLAUDE.md`: "FlashAttention-4 (FA4) … Targets Hopper (SM90) and Blackwell
(SM100/SM110)." FA3 lives in `hopper/`.

### FA4 forward (`flash_fwd_sm100.py`, `sm100_hd256_2cta_fmha_forward.py`)

| Component | What FA4 does | HW dependency | SM120? |
|---|---|---|---|
| **MMA engine** | `tcgen05.mma` (UMMA), async, accumulator in TMEM | SM100 5th-gen TC | ❌ use `mma.sync` (warp-level, acc in registers) |
| **2-CTA cluster MMA** | `cluster_shape=(2,1)`, `CtaGroup.TWO`, both CTAs feed one MMA | SM100 cluster + 2-CTA instr | ❌ SM120 cluster must be (1,1) |
| **TMEM dataflow** | S/P/O live in tensor memory; P read from TMEM into PV MMA (`OperandSource.TMEM`) | SM100 TMEM | ❌ keep S/P/O in register file |
| **Warp specialization** | `softmax_warp_ids=(0,1,2,3)`, `mma_warp_id=8`; MMA warp issues async UMMA, softmax warps overlap | software | ✅ **portable** (via `setmaxnreg`) |
| **MMA↔softmax overlap** | softmax on tile *i* overlaps MMA on tile *i+1*; `mbar_S_full_P_full_O_rescaled` barriers | software + TMEM | ✅ **portable** (overlap MMA↔softmax in regs) |
| **FP8 softmax + bounded rescale** | `max_offset` (P·2^max_offset fills e4m3 range), `rescale_threshold` (let row-max lag → skip rescale when ≈1), ex2 emulation | software | ✅ **fully portable** |
| **Persistent kernel** | grid = #SMs, grid-stride over tiles | software | ✅ **portable** (`StaticPersistentTileScheduler`) |
| **Dynamic tile scheduler** | **CLC** (`ClcDynamicPersistentTileScheduler`) HW work-stealing | SM100 CLC | ⚠️ partial → software atomic-counter work queue |
| **TMA loads** | `cp.async.bulk.tensor` (SM90-style TMA), multicast across cluster | TMA (portable) + multicast (SM100) | ✅ TMA portable; ❌ drop multicast |
| **HW row-max** | `tcgen05.ld.red` computes row-max during S load (SM103) | SM100/103 TMEM | ❌ software warp-reduce max |
| **Split-KV + combine** | partial kernels + `flash_fwd_combine.py` | software | ✅ **portable** (local kernel already has it) |
| **Paged KV** | `paged_kv.py` | software | ✅ **portable** (local kernel already has it) |

### Why 2-CTA pingpong is SM100-only (and what "pingpong" means on SM120)

FA4's *2-CTA* pingpong (`sm100_hd256_2cta_fmha_forward.py`) pairs two CTAs in a
cluster so one does QK while the other does PV, hiding softmax latency behind
the partner's MMA. It needs cluster MMA + TMEM. **SM120 cannot do this.**

But "pingpong" has a *second*, fully-portable meaning, demonstrated by the
CUTLASS SM120 GEMM: **two MMA warp-groups in one CTA alternate tiles** —
*"while the first warp group executes the epilogue for the current tile, the
second warp group executes the MMA mainloop"* (`dense_blockscaled_gemm_persistent_pingpong.py`
docstring). This is the SM90/FA3 pingpong idea (`sm90_gemm_warpspecialized_pingpong.hpp`,
2 MMA warp-groups, `MmaRegisterRequirement=240`) and **works on SM120**.

---

## 2. SM120 adaptation strategy, technique by technique

### (1) 2-CTA cluster pingpong → **single-CTA, 2-warp-group pingpong**  *(portable, high value)*
- Drop the cluster entirely (`cluster_shape=(1,1)` is mandatory on Blackwell
  GeForce — CUTLASS example asserts this).
- Use **2 MMA warp-groups (8 warps) + 1 TMA DMA warp** per CTA. Warp-group A
  does softmax/epilogue on tile *i* while warp-group B runs QK/PV MMA on tile
  *i+1*, coordinated by named/order mbarriers (`math_wg_order_barrier`) and
  `setmaxnreg` (load=40, mma=232 regs — straight from the CUTLASS SM120 example).
- This recovers the *latency-hiding* goal of FA4 pingpong without clusters.

### (2) FP8 softmax with online rescaling → **fully portable**  *(high value, cheap)*
Port from `softmax.py` / `flash_fwd_sm100.py`:
- `max_offset = log2(448)` for e4m3: scale P by 2^max_offset so probabilities
  fill the FP8 dynamic range before the PV MMA (the local kernel already does a
  per-row-max `p_scale`; FA4's contribution is the *bounded-lag* refinement).
- `rescale_threshold`: let the running row-max lag by N log2 units so the
  O-rescale factor stays ≈1 and the expensive per-tile O rescale (and FP8
  re-quant error) is skipped when the max barely moved.
- `ex2_emu` (`tune_ex2_emu.py`, `ex2_emulation_2`): FP8 fwd is **MUFU/exp2-bound**;
  emulate some `exp2` with FMA to relieve the SFU. Directly applicable since the
  local kernel is FP8-PV.
- Descale tensors (`q_descale/k_descale/v_descale`, FA3 semantics) for
  per-head/per-tensor FP8 scales — cleaner than ad-hoc in-kernel amax.

### (3) Persistent kernel + dynamic tile scheduler → **portable (static); dynamic via SW**  *(medium value)*
- `StaticPersistentTileScheduler` (grid-stride): `grid_x = min(#SMs, total_tiles)`;
  `advance_to_next_work: tile_idx += grid_dim`. **Runs unchanged on SM120.**
- FA4's *dynamic* scheduler is **CLC hardware** (SM100-only). SM120 substitute:
  a global **atomic work counter** (software work-stealing) for load balance
  across variable-length requests — only worth it for varlen/prefill imbalance;
  for fixed decode shapes static grid-stride is enough.
- Persistent kernels also amortize launch/prologue cost — relevant because the
  local prefill profiling showed prologue/epilogue overhead amortizes poorly on
  small chunks (0.5%→6.3%).

### (4) TMEM-based dataflow → **register file; quantify the cost**  *(forced regression)*
- SM120 keeps S/P/O accumulators in **registers**. Cost: register pressure.
  The local kernel already hits **255 regs/thread**, dominated by the D=256
  output accumulator `O_acc` (~128 floats/lane) — this caps occupancy at 1
  CTA/SM and is *the* reason TMEM would have helped.
- Mitigations (all local-team-identified): **D-dimension PV tiling** (2 passes of
  D/2=128 → O_acc peak 64 floats/lane → ≤128 regs → 2 CTAs/SM), or fewer threads.
  Note the local team found D-parallel *worse* (recomputes QK^T); D-*tiling*
  (recompute P, reuse from SMEM) is the untried variant.
- Net: expect SM120 attention to be **register/occupancy-constrained** in a way
  SM100 is not. This is the main structural ceiling vs FA4.

### (5) Warp specialization (producer/consumer) → **portable via setmaxnreg**  *(high value)*
- SM120 supports `setmaxnreg` (CUTLASS SM120 example uses
  `setmaxnreg_increase(232)` for MMA warps, 40 regs for the DMA warp).
- Pattern: **1 dedicated TMA producer warp** + **8 MMA consumer warps**. This is
  proven on SM120 by both the CUTLASS GEMM and **BlackFlash** (1 producer + 8
  consumer warps, 288 threads). The local kernel's `decode_v3_warpspec.cuh`
  already has a *phase-based* QK/PV warp split but **not** an async
  producer/consumer pipeline with a dedicated DMA warp — that is the gap.

### (6) TMA + deep pipelining → **portable, the single biggest lever**  *(highest value)*
- Replace `cp.async` (16B/4B per thread, software-managed) with **TMA bulk
  tensor copies** (`cp.async.bulk.tensor`, SM90-style, which SM120 supports —
  proven by BlackFlash and CUTLASS `sm120_mma_tma.hpp` using `SM90_TMA_LOAD`).
- TMA + mbarrier `arrive_and_expect_tx` lets a **single DMA warp** stage many
  KV tiles ahead, breaking the "cp.async sequential wait chain" that the local
  team identified as *the* decode bottleneck.
- Add **V-transpose elimination**: load V already-transposed via TMA/ldmatrix
  so PV avoids the SMEM roundtrip (local team's named #2 bottleneck).

---

## 3. Community / official backport status

| Artifact | Status | Evidence |
|---|---|---|
| **Official FA4 SM120** (`flash_attn/cute/flash_fwd_sm120.py`) | **SM80 fallback only** — subclasses `FlashAttentionForwardSm80`, forces `self.arch = Arch.sm_80`, uses legacy `mma.sync.m16n8k16`, only overrides the 99 KB SMEM check. **No TMA, no warp-spec, no persistent, no FP8.** | file header + `can_implement()` |
| **Official FA4 SM120 backward** (`flash_bwd_sm120.py`) | Same — SM80 fallback. | file |
| **CUTLASS SM120 GEMM** (`blackwell_geforce/`) | **Real, portable building blocks**: persistent + pingpong + warp-spec + TMA + block-scaled MMA (FP8 m16n8k32, NVFP4 m16n8k64). GEMM only, not attention. | `dense_blockscaled_gemm_persistent_{pingpong,cooperative}.py` |
| **BlackFlash** (local) | From-scratch **FA2** for SM120 (RTX 5060 Laptop) with TMA + warp-spec + double-buffer + mbarriers. ~31 TFLOPS BF16 vs cuDNN 51 (0.62×) on large shapes; 1.05× cuDNN on tiny shapes. Proves the HW path; not FA4-class. | `BlackFlash/README.md` |
| **FlashInfer SM120** | Native oracle has SM120 `xqa_*` decode kernels (head_dim=256, GQA 1/4/8, e4m3 KV, spec-decode) + FA2 prefill. This is what the local kernel must beat. | `flashinfer/build/aot/cached_ops/xqa_*head_dim_256*` |
| **SecondNatureComputing / HF "flash-attn-4-sm120"** | **Could not verify** — web search unavailable this session. No local clone found. Treat as unconfirmed. | — |

**Bottom line:** nobody has shipped an FA4-class (persistent + pingpong +
warp-spec + TMA + FP8-softmax) *attention* kernel for SM120. The official repo
punts to SM80. The ingredients exist (CUTLASS GEMM + BlackFlash attention); the
integration is open.

---

## 4. Local kernel: already-has vs missing-from-FA4

`sm120-flash-attention/kernel/` (`flash_attn_sm120.cu` 14 009 lines +
`decode_v3_warpspec.cuh`, `decode_v2_nvfp4kv.cuh`, `decode_tma_load.cuh`).

| FA4 technique | Local kernel status |
|---|---|
| Online softmax (row_max/row_sum rescale) | ✅ has |
| FP8 (e4m3) QK + KV + **P quantization** (per-row-max p_scale) | ✅ has (Phase 2) |
| NVFP4 (e2m1 + ue4m3 microscale) QK + KV, two-level P requant | ✅ has (Phase 3) |
| Paged KV cache (CSR page table) | ✅ has (Phase 4) |
| Split-KV flash-decoding + merge kernel | ✅ has (decode) |
| Warp specialization | ⚠️ partial — `decode_v3_warpspec` has a *phase* QK/PV warp split, **not** async producer/consumer |
| Double buffering | ✅ has — but **cp.async**, 2-stage |
| **TMA loads** | ❌ missing (uses `cp.async16`/`cp.async4`) — `decode_tma_load.cuh` is a 114-line stub |
| **Dedicated DMA warp + deep pipeline** | ❌ missing |
| **Persistent kernel / tile scheduler** | ❌ missing (fixed grid) |
| **Pingpong (2 warp-groups alternating)** | ❌ missing |
| **FP8 bounded-rescale (`rescale_threshold`) + ex2 emulation** | ❌ missing |
| **V-transpose elimination** | ❌ missing (named bottleneck) |
| MMA instruction | `mma.sync.m16n8k16` (bf16) / `m16n8k32` (fp8) — **correct for SM120** |

**Gap summary:** the local kernel has FA4's *quantization & paging & split-KV*
story but lacks FA4's *execution* story (TMA, warp-spec pipeline, persistent
scheduling, pingpong, FP8-softmax micro-opts). The execution story is precisely
what the profiling says is needed.

---

## 5. Estimated SM120 performance ceiling

Grounded in local measurements + CUTLASS/BlackFlash evidence (web unavailable
for vendor specs; using the project's own numbers as ground truth):

- **Decode is bandwidth-bound.** Current 544 GB/s ≈ 30% of peak ⇒ effective
  peak ≈ 1.8 TB/s (consistent with RTX PRO 6000 Blackwell GDDR7). A TMA +
  warp-spec + deeper-pipeline rewrite should be able to reach **~60–75% of peak
  bandwidth** (FA3-class kernels on Hopper hit ~this), i.e. roughly
  **2–2.5× over the current decode kernel** on memory-bound shapes. Whether that
  beats FlashInfer's `xqa` (already SM120-tuned) is **uncertain** — FlashInfer is
  the moving target and currently leads by 2.4–3.0×.
- **Prefill is compute-bound at large shapes.** BF16 is already near-SOL on this
  Max-Q card (no GeForce FP32-accumulate throttle), so **BF16 prefill has little
  headroom**; the win is FP8/NVFP4 prefill via block-scaled MMA throughput +
  better pipelining. BlackFlash shows a handwritten SM120 FA2 reaches ~31 TFLOPS
  vs cuDNN ~51 — i.e. **~60% of a tuned vendor kernel** is realistic for a
  from-scratch effort; closing to ≥90% is the hard part.
- **Structural ceiling vs FA4/SM100:** no TMEM ⇒ register/occupancy-limited
  (255 regs, 1 CTA/SM today); no 2-CTA MMA ⇒ half the MMA parallelism per Q tile;
  no CLC ⇒ weaker varlen load-balance. Expect SM120 attention to top out well
  below SM100's utilization even with a perfect port.
- **Amdahl reality (from local roadmap):** W1 short-context has f≈3.8–4.2% →
  **physically cannot** reach 5% e2e. Only **W2 (32K/1K, c=4, MTP K=3,
  f_native=18.38%)** is viable, needing **S ≥ 1.374×** on attention. The
  decode-heavy MTP verify path is where the 2–2.5× bandwidth headroom matters.

**Honest estimate:** a disciplined FA4-style port can plausibly reach
**~1.5–2× over the current custom kernel** on W2 decode, which *might* clear the
1.374× Amdahl bar and approach FlashInfer — but **beating** FlashInfer end-to-end
is not guaranteed and has failed twice before under strict re-test.

---

## 6. Concrete implementation plan (prioritized)

> Posture: this is a **high-risk, multi-week** rewrite of a line that was formally
> closed (2026-07-16) because the custom kernel was slower than native. Run it
> **validator-gated** (`$ultraqa` / `bf diff` discipline) with native FlashInfer
> as the standing oracle. Do **not** regress the production default (native) while
> building. Re-use `bf`/`bfdiag` for every comparison; never compare runs without
> `bf diff` (the closure doc lost a day to an incomparable A/B).

**P0 — TMA load path + dedicated DMA warp (highest leverage, attacks the named bottleneck)**
- Port `decode_tma_load.cuh` stub → real `cp.async.bulk.tensor` KV loads with
  mbarrier `arrive_and_expect_tx`; 1 producer warp, N-stage ring buffer.
- Reference: BlackFlash (proven on this GPU class) + CUTLASS `sm120_mma_tma.hpp`.
- Gate: parity vs current kernel (`bf divergence`), then bandwidth ≥ 1.6× current.

**P1 — Async warp-specialized pipeline (producer/consumer, not phase split)**
- Convert `decode_v3_warpspec` phase-split into a true producer/consumer pipeline
  with `setmaxnreg` (DMA=40, MMA≈232) + named barriers.
- Gate: occupancy/latency improvement under `ncu`; no correctness regression.

**P2 — V-transpose elimination**
- Load V pre-transposed via TMA/ldmatrix layout so PV skips the SMEM roundtrip
  (named bottleneck #2). Gate: parity + measured SMEM-traffic drop.

**P3 — FP8 softmax micro-opts (cheap, additive)**
- Add `rescale_threshold` bounded-lag rescale + `max_offset` range-filling +
  `ex2` emulation (port from `softmax.py`, `tune_ex2_emu.py`). Gate: cos≥0.99999
  vs SDPA + MUFU-stall reduction in `ncu`.

**P4 — Persistent kernel + static tile scheduler (prefill / varlen)**
- Grid = #SMs, grid-stride (`StaticPersistentTileScheduler` pattern); add SW
  atomic work-stealing only if varlen imbalance shows up. Gate: small-chunk
  prefill prologue-overhead reduction.

**P5 — 2-warp-group pingpong (advanced, only if P0–P4 leave headroom)**
- Two MMA warp-groups alternating tiles via order-mbarriers (CUTLASS SM120
  pingpong pattern). Gate: MMA-utilization uplift; this is the closest portable
  analog of FA4's overlap goal.

**P6 — Register/occupancy: D-dimension PV tiling**
- 2-pass D/2=128 to cut O_acc to 64 floats/lane → ≤128 regs → 2 CTAs/SM.
  (Distinct from the failed D-*parallel*; reuses P from SMEM.) Gate: 2 CTAs/SM
  confirmed + net speedup.

**Explicitly out of scope (SM100-only, do not attempt on SM120):** tcgen05/UMMA,
TMEM dataflow, 2-CTA cluster MMA, cluster multicast, CLC dynamic scheduler,
hardware row-max (`tcgen05.ld.red`).

---

## 7. Key file references

- FA4 fwd (SM100): `fa4-latest/flash_attn/cute/flash_fwd_sm100.py`,
  `sm100_hd256_2cta_fmha_forward.py` (2-CTA pingpong, TMEM, warp-spec)
- FA4 softmax: `fa4-latest/flash_attn/cute/softmax.py` (`SoftmaxSm100`,
  `rescale_threshold`, `max_offset`, `ex2_emu`)
- FA4 scheduler: `fa4-latest/flash_attn/cute/tile_scheduler.py`
  (`StaticPersistentTileScheduler` portable; `ClcDynamicPersistentTileScheduler` SM100)
- FA4 SM80-fallback SM120: `fa4-latest/flash_attn/cute/flash_fwd_sm120.py`
- FA3 Hopper warp-spec/pingpong: `fa4-latest/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`,
  `flash_fwd_kernel_sm90.h` (LoadWG=1, MmaWG=2, reg budgets)
- CUTLASS SM120 GEMM: `cutlass-4.6.1/examples/python/CuTeDSL/cute/blackwell_geforce/
  kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py`
- CUTLASS SM120 MMA collective: `cutlass-4.6.1/include/cutlass/gemm/collective/sm120_mma_tma.hpp`,
  `sm120_blockscaled_mma_tma.hpp`; MMA atoms in `blockscaled_gemm_dispatch.py`
- Local kernel: `sm120-flash-attention/kernel/flash_attn_sm120.py`,
  `csrc/flash_attn_sm120.cu`, `csrc/decode_v3_warpspec.cuh`, `csrc/decode_tma_load.cuh`
- Local evidence: `sm120-flash-attention/notes/attention-kernel-research-closure.md`,
  `notes/2026-07-20-kernel-optimization-findings.md`,
  `notes/qwen36-sm120-custom-kernel-roadmap.md`, `notes/03-本地环境说明.md`
- BlackFlash: `sm120-flash-attention/BlackFlash/README.md`
