# 2026-08-04 W1-S same-caliber gap decomposition (today vs historical)

## Setup

Both arms measured on the SAME frozen W1-S workload (n16 fixture,
4096in/256out, c=4, K=3, greedy), same GPU (RTX PRO 6000 Blackwell Max-Q,
188 SM). Historical numbers from
`notes/2026-08-03-historical-implementation-survey.md` §4.1 (verified: the
historical `w1s_native_bench.py` computes accepted/(t_end−t_start) over the
WHOLE workload, i.e. INCLUDING TTFT/prefill — same convention as today's
`accepted_tokens_per_sec` field is NOT: today's field is actually
committed/wall_e2e, the naming is a bug in `mtp_w1s_our_runtime_perf.py`'s
rep dict).

Reproduction (today):

```bash
PYTHONPATH=/home/bot/project/qwen-sm120-runtime \
~/.venvs/vllm/bin/python -m benchmarks.mtp_w1s_our_runtime_perf \
  --concurrency 4 --num-requests 16 --max-tokens 256 --repeats 1 --fixture n16 \
  --result-path /tmp/qwen_w1s_c4_today_result.json
```

## Anchors (quality is intact)

| anchor | historical | today | verdict |
|---|---:|---:|---|
| total_committed_tokens | 4116 | 4119 | ✅ same workload |
| draft_acceptance_rate | 70.29% | 70.92% | ✅ parity |

## Headline

| | historical final (07-18) | today (current tree) |
|---|---:|---:|
| wall e2e | ≈20.4 s (derived: 136.75 accepted tok/s) | **69.56 s** |
| accepted tok/s e2e | 136.75 | 40.3 |
| committed tok/s e2e | ≈201.5 | 59.2 |
| prefill wall | ≤5–7 s (derived) | **41.16 s** |
| decode wall | ≈13–15 s (derived) | 28.40 s |
| decode round | ≤12 ms | **21.6 ms** |

Gap 3.4× e2e; components: prefill ~35 s of it, decode ~13 s.

## Prefill attribution (nsys, 4-req c=4 batch, /tmp/qwen_prefill_attr.nsys-rep)

prefill_wall 10.1 s, prefill_gpu 9.55 s → **94.5% GPU-bound, NOT
launch-gap**. Kernel breakdown:

| kernel | share | note |
|---|---:|---|
| W4A16FusedMoeKernel (64 layers, M=16384) | **44.8%** (3.80 s) | ≈56 ms/layer ≈ **156 TFLOPS** |
| torch scaled_mm CUTLASS (FP8 dense) | 13.5% + variants | |
| torch elementwise/copy/reduce/quant zoo | ~15% | per-Linear dynamic quant etc. |
| attention extend | 1.8% | not the issue |
| GDN chunk kernels | ~2% | not the issue |

`lm_head` full-position bug (historical §5.1 ⚡) does NOT exist today:
`qwen36.py:760` computes logits only at `hidden_batch[:, -1, :]`.

## The honest ceiling statement

156 TFLOPS on the W4A16 path is the dequant-inside-kernel BF16-MMA cost of
the QUALITY contract. Historical prefill speed sat on vLLM's native NVFP4
**W4A4** block-scaled MMA (4× BF16 MMA rate, half the weight bytes) — the
SAME scheme this repo measured and REJECTED on this checkpoint
(`notes/2026-08-03-w4a4-blockscaled-negative-result.md`: single-layer
cosine 0.988 vs 0.99999, B1-R fails across the board). Therefore:

**W1-S prefill parity with history is not reachable under the W4A16
quality contract via "port the historical kernels" — the historical prefill
MMA rate came from a numerically rejected scheme.** The quality-preserving
prefill levers that remain: (1) the ~15% torch elementwise/quant zoo,
(2) 156→~220 TFLOPS kernel headroom (~1.4×), (3) M-4 interleaving (hides
prefill behind decode, does not reduce its GPU cost).

Decode rounds have NO such ceiling: historical decode also ran W4A4, but
decode is bandwidth/latency-bound at M≤16, where W4A16's weight bytes are
identical to W4A4's; the 1.85× decode gap is wiring/structure, not scheme.

**Correction (same day, roofline check):** one verify round streams ALL
weights once — ~9.99 GiB raw FP8 dense + ~8.65 GiB packed NVFP4 MLP ≈
18.6 GiB. At 16.1 ms verify-graph CUDA that is **1.15 TB/s = 64% of the
1.79 TB/s GDDR7 peak**; historical ≤12 ms rounds imply ~87%. So the
decode-side quality-preserving ceiling is verify 16.1 → ~13 ms (round
21.6 → ~18.5), not 12 ms; the remaining decode headroom is bandwidth
efficiency, worth ~15% of decode wall, not 1.85×.

## Decode round attribution (3 steady profile rounds, c=4)

round = verify_graph 16.1 ms + draft 4.2 + sync 1.3 + lm_head 0.8 ≈ 22 ms
(measured 21.6). Inside/around verify:

- torch `_scaled_mm` CUTLASS GEMMs dominate the dense FP8 projections;
- ~418/round small reduce/abs/div elementwise launches = per-Linear dynamic
  per-token activation quantization (historical fused this into the CUTLASS
  prologue / `scaled_fp8_quant` single launch);
- W4A16 fused MoE ≈ 10 ms/round (its own 2026-08-04 thread: occupancy
  candidate rejected, launch structure is the remaining target);
- native fp8_w8a8 .so A/B (`QSR_NATIVE_W8A8_FP8_CHANNEL=all`): verify-graph
  CUDA 48.4→41.5 ms/3 rounds BUT e2e WORSE (49.26 vs 59.21 committed tok/s;
  decode-phase wall 28.4→41.4 s while captured-phase CUDA totals are equal)
  — the loss sits outside the captured graphs at small-M geometries;
  blanket switch rejected, per-shape routing remains open.

## Next actions in expected-gain order

1. Decode bandwidth efficiency (64%→~87% of peak): find the kernel mix
   stealing bandwidth in the verify graph (target round 21.6→~18.5 ms).
2. Prefill: remove the torch quant/elementwise zoo from the prefill path
   (~15% of prefill GPU).
3. Decode: per-shape FP8 GEMM routing (native .so only where it provably
   wins; torch elsewhere) — small-M blanket switch measured WORSE.
4. M-4 interleaved chunked prefill — note: GPU is 94% busy, so this buys
   TTFT/latency, NOT total-wall throughput; do it for the latency metric.
5. W4A16 prefill kernel headroom (156→~220 TFLOPS) — sparkinfer-side, last.

## Attainable ceiling under the quality contract (measured bound)

Best case with items 1–3+5: prefill GPU 39.2→~26 s, decode wall 28.4→~24 s
→ e2e ≈ 50 s ≈ committed ~82 / accepted ~55 tok/s. Historical 20.4 s wall
is **not reachable while preserving today's output quality**: its prefill
MMA rate came from W4A4, which this repo's own gates reject on this
checkpoint. Closing the remaining 2.5× requires a quality-contract
decision, not more engineering.

## Decode-side kernel attribution (nsys, 4-req 256-token decode-heavy run)

Decode round today: 20.5 ms wall (1297 rounds / 26.6 s); historical ≈11.6 ms.
Per-kernel from `/tmp/qwen_decode_attr.nsys-rep`:

- **lm_head is the single biggest decode kernel**: the 248,320×5,120 FP8
  lm_head (`torch._scaled_mm`) ran 356× at avg **6.03 ms** (~1× per round),
  streaming its 2.54 GiB weights at only ~450 GB/s — roofline is ~1.6 ms.
  The self-owned `fp8_w8a8_sm120.so` (historical CUTLASS port) measured
  **3.1 ms at M=4/16** (1.2–1.5× faster than torch) on the same shape,
  previously verified max_abs=0 vs the historical `cutlass_scaled_mm` at
  M=1. Wired per-shape behind `QSR_NATIVE_W8A8_LM_HEAD=1`
  (`compressed_tensors_linear.py`, output_size==248320 only; the blanket
  all-shapes native switch measured slightly worse e2e).
- W4A16 fused MoE: 45.2% of run GPU (prefill-dominated; decode portion
  streams at ~830 GB/s — near roofline, no headroom).
- FP8 dense GEMMs (all non-lm_head): ~16% — native vs torch measured at
  parity for these shapes (c=4 A/B: 76.6 vs 78.8 committed/s).
- torch quant/elementwise/copy zoo: ~11% of run GPU (prefill-weighted).
- sparkinfer `blockscaled` FP4 small-M bandwidth ceiling: tile sweep at
  real layer-5 MLP shapes (2026-08-04) shows the DEFAULT tile is already
  the best supported option — (64,64) is 30-50% SLOWER, (16,128)/(64,32)
  are not implementable in the current FP4 dense kernel. M≤16 blockscaled
  tops out at ~330-440 GB/s; fixing it is a kernel-structure job (M-tile
  64 is the smallest supported), not a config job.

## lm_head per-shape native routing: NEGATIVE at e2e (2026-08-04)

`QSR_NATIVE_W8A8_LM_HEAD=1` (output_size==248320 only), same-caliber W1-S:
wall 64.3 s vs baseline 52.2 s; decode 31.9 vs 26.5 s; mem end 74.7 vs
61.4 GiB; acceptance 70.4 (fine). The 3.1-vs-3.8 ms microbench win did not
survive the full-model graphed run — same failure pattern as the all-shapes
native switch and the W4A4 prefill path: isolated-kernel gains on this box
are being erased by whole-process memory/state effects (this machine runs a
~33 GiB co-tenant; baseline wall itself has ranged 52.2–69.6 s across
identical-config runs today). The gate stays in-tree, default OFF. Before
any further kernel-swap claim, establish the same-arm repeat noise envelope
on this machine; deltas inside it are not evidence.
