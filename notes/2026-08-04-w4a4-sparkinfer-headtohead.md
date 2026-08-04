# 2026-08-04 W4A4 sparkinfer-vs-flashinfer head-to-head and prefill wiring

Directive (user, 2026-08-04): the historical vLLM-era scheme is the quality
bar; if a re-implementation fails our gates, treat it as an implementation
problem to root-cause, not a scheme to reject. Prefer sparkinfer kernels over
flashinfer; if sparkinfer lags, find the cause.

## Numerics: sparkinfer's W4A4 is correct; the 08-03 rejection stands as
scheme-level, not an implementation bug

Single-layer (real layer-5 weights, unsloth checkpoint), BF16-dequant
reference from the W4A16 arithmetic:

| path | cos vs W4A16/BF16 reference |
|---|---:|
| sparkinfer `blockscaled.mm` (gate_proj, M=64) | **0.99546** |
| sparkinfer `blockscaled.mm` (down_proj) | 0.99545 |

The quantizer recipe was audited line-by-line against the NVFP4 spec
(`quantize_grouped_nvfp4_torch`: `sf = e4m3(gs * bmax/6)`,
`code = e2m1(x / (sf/gs))`) — it matches. A flashinfer-b12x reference arm
was built for cross-check but diverged on the wide-N shape in MY harness
(`cos(fi_gemm, ref) = 0.67` on gate_proj while sparkinfer scored 0.995);
the divergence sits on the flashinfer calling side (weight swizzle/pad or
global-scale direction in my harness), NOT in sparkinfer — so sparkinfer is
kept as the production kernel without further b12x chasing. The earlier
harness also had a tvm_ffi/cutlass-dsl kwarg incompatibility
(`make_kwargs_wrapper ... map_dataclass_to_tuple`) that needed a shim.

Conclusion: sparkinfer's blockscaled W4A4 arithmetic is faithful to the
historical NVFP4 recipe. The remaining 0.9955-vs-0.99999 delta vs the W4A16
path is the genuine scheme-level activation-quantization error — which the
historical implementation also carried (it ran this same scheme), and which
the user's quality bar accepts.

## Speed: sparkinfer matches flashinfer b12x within ±20%

Clean GEMM-kernel-only bench (operands prepared ONCE, layer-5 shapes):

| proj | M | sparkinfer | b12x | ratio | sparkinfer BW/TFLOPS |
|---|---:|---:|---:|---:|---|
| gate_proj | 16 | 0.093 ms | 0.079 ms | 0.85 | 478 GB/s |
| gate_proj | 1024 | 0.281 ms | 0.580 ms | 2.06 | — |
| gate_proj | 16384 | 4.764 ms | 4.475 ms | 0.94 | ~622 TFLOPS |
| down_proj | 16 | 0.071 ms | 0.069 ms | 0.98 | 630 GB/s |
| down_proj | 1024 | 0.321 ms | 0.299 ms | 0.93 | — |
| down_proj | 16384 | 5.984 ms | 6.082 ms | 1.02 | — |

Two trap-avoidance notes:
1. `dense_gemm`'s `expected_m` regime hint MUST be passed: without it the
   M=16384 gate GEMM measured 87 ms instead of 37–47 ms (decode-tuned tile
   reused for a prefill shape).
2. My first microbench timed per-call operand preparation (44 MB weight
   copies + pure-Torch quantization) and read "30 GB/s" — pure bench
   contamination. Prepare operands once; then decode shapes stream weights
   at 478–662 GB/s.

Vs the production W4A16 fused kernel (nsys 2026-08-04): prefill is 156
TFLOPS there vs ~622 TFLOPS here — **the ~4x prefill MLP win is real**.
Decode M≤16: W4A16 fused streams at ~830 GB/s vs 478–662 GB/s here, and
keeps the fused FC1→activation→FC2 structure — decode/verify stay W4A16.

## Wiring landed (opt-in, default OFF)

`runtime/model/qwen36_model.py`: `QSR_QWEN36_MLP_W4A4=1` routes MLP forwards
with ≥64 rows (prefill / chunked prefill, all eager) through
`_forward_w4a4_blockscaled` (3× `blockscaled.mm` + torch NVFP4 activation
quantization); decode/verify (M≤16) stay on the W4A16 fused kernel and its
captured graphs. Operands are built once per layer inside
`_ensure_w4a16_fused_ready` before the raw-Parameter free (+one packed-
weight copy resident while enabled). Checkpoints without activation global
scales (nvidia modelopt) mark the path unavailable and fall back silently.

Known costs: pure-Torch activation quantization (fine at prefill M, would
need a CUDA/Triton quantizer before any decode use); +8.65 GiB operand copy.

## Open items

- ~~Full-model acceptance anchor (W1-S c=4 committed/acceptance vs the
  70.3%-class historical anchor) with the gate ON.~~ DONE — see negative
  result below.
- If decode-side W4A4 is ever wanted: a fused CUDA NVFP4 activation
  quantizer in sparkinfer (only pure-Torch oracle quantizers exist today),
  and an M≤16 blockscaled bandwidth root-cause first.

## Full-model verdict: W4A4 prefill path does NOT win e2e (2026-08-04)

Same-caliber W1-S (n16, c=4, K=3) with `QSR_QWEN36_MLP_W4A4=1` (M≥64
routed to W4A4, decode stays W4A16), two variants, vs baseline:

| arm | wall | committed/s | prefill | decode | accept% | mem end |
|---|---:|---:|---:|---:|---:|---:|
| baseline (W4A16 only) | 52.2 s | 78.8 | 25.6 s | 26.5 s | 72.4 | 61.4 GiB |
| W4A4, cloned operands | 91.9 s | 44.8 | 46.4 s | 45.4 s | 71.4 | 89.4 GiB |
| W4A4, aliased raw weights | 82.0 s | 50.2 | 40.4 s | 41.5 s | 71.4 | 88.7 GiB |

Quality anchor held (acceptance 71.4 vs 72.4, committed 4115-4116 vs
historical 4116) — but BOTH phases got dramatically slower despite the
single-layer M=4096 measurement showing W4A4 1.84× faster than W4A16.
Root cause of the e2e loss:

1. Weight residency: the W4A4 operands alias the RAW packed weights, which
   therefore cannot be freed, while decode still needs the W4A16 repack —
   two MLP weight residencies (~17.3 GiB) plus this box's unrelated 33 GiB
   co-tenant pushed the card to 88-89 GiB/97.9 GiB. Allocator pressure
   slowed BOTH phases (decode, which never executes W4A4, slowed +15 s
   too — pure machine-state effect).
2. The historical implementation avoided this by running ONE kernel family
   (vLLM NVFP4 W4A4) for ALL M with ONE weight residency.

**Conclusion: do not pursue W4A4-for-prefill with W4A16-for-decode; the
two-representation split is structurally memory-hostile on this box.** If
W4A4 is revisited, it must be all-M (single residency, historical layout),
which first requires the decode-side blockscaled bandwidth fix. The wiring
(`QSR_QWEN36_MLP_W4A4`, default OFF) and the bit-exact Triton quantizer
stay in-tree as components for that future path.

## All-M W4A4 (historical layout): CONFIRMED e2e win (2026-08-04)

`QSR_QWEN36_MLP_W4A4=1 QSR_QWEN36_MLP_W4A4_ALL=1` routes EVERY MLP forward
(prefill, decode, verify, draft) through blockscaled W4A4 -- the single
kernel family / single weight residency layout of the historical
implementation. Required two fixes to get there: (1) the Triton quantizer's
``.item()`` global-scale read was CUDA-graph-capture-unsafe (verify capture
failed with ``cudaErrorStreamCaptureUnsupported``) -- now passed by device
pointer; bit-parity re-verified 5/5 after the change. (2) weight operands
alias the raw packed weights (no second residency).

Same-caliber W1-S (n16, c=4, K=3), two back-to-back repeats vs the same-day
baseline arms:

| arm | wall | committed/s | prefill | decode | accept% | committed |
|---|---:|---:|---:|---:|---:|---:|
| **W4A4 all-M, run 1** | **50.2 s** | **82.0** | **22.0 s** | 28.1 s | 71.2 | 4115 |
| **W4A4 all-M, run 2** | **51.8 s** | **79.4** | **21.4 s** | 30.4 s | 71.2 | 4115 |
| baseline (W4A16), lucky | 52.2 s | 78.8 | 25.6 s | 26.5 s | 72.4 | 4115 |
| baseline (W4A16), repeat | 60.2 / 58.7 s | 68.2 / 69.9 | 33.1 / 31.8 s | 27.0 / 26.9 s | 70.8 / 74.7 | 4117 / 4113 |

- Prefill wall: **21.4-22.0 s vs typical baseline 31.8-33.1 s = -32%**;
  better than the lucky baseline's 25.6 s too.
- Decode wall: +1.5 to +3.5 s vs baseline 26.5-27.0 -- the known
  blockscaled small-M bandwidth deficit (~330-440 GB/s vs the W4A16 fused
  kernel's ~830 GB/s), now quantified in e2e terms.
- Net: wall 50.2-51.8 s vs typical baseline 58.7-60.2 s = **~14% e2e
  faster**; committed/s 79-82 vs 68-70. Committed-token anchor 4115 both
  runs (historical anchor 4116); acceptance 71.2% both runs (historical
  70.29%). All three MTP graphs captured with the W4A4 path inside.
- Memory end state 80.0 GiB (vs 61.4 baseline): the raw packed weights
  stay resident (aliased), the W4A16 repack is never built. Higher than
  baseline but below the allocator-pressure zone that sank the earlier
  two-residency variants (88-89 GiB).

Remaining gap to historical (~20.4 s wall): decode rounds are still 28-30 s
(21.6->~23 ms/round; historical ~11.6 ms). Measured 2026-08-04: the FP4
dense kernel's small-M ceiling is NOT fixable by config -- swap_ab variants
((64,16)/(64,32), tma and cpasync, all bit-exact) measured 152-354 GB/s,
SLOWER than the default tile's 435-532 GB/s; M-tile 64 is the smallest
supported and (16,*) is structurally rejected by can_implement. Final config-level attempts on the FP4 side (sparkinfer branch
``fp4-splitk-20260804``, commit ``f4213aa``, 2026-08-04): (a) split-K wired
for FP4 -- M=4 gate 383 GB/s (slower) AND numerically wrong (max_rel
2.4e4; the FP4 partial-accumulation path is not wired correctly), reverted;
(b) M-tile 16 allowed by ``can_implement`` -- (16,128)/(16,64) compile and
are BIT-EXACT (maxdiff=0) but 508-648 GB/s, no better than the default
586-667 GB/s. **The small-M ceiling is kernel pipeline structure, not tile
configuration; the remaining decode-side work is days-scale kernel
development (FP4 load pipeline; FP8 per-token-x-per-channel epilogue for
the torch._scaled_mm shapes).**

## FlashInfer B12x reference: sparkinfer is AT PARITY, not behind (2026-08-04)

Direct same-shape speed comparison on real layer-5 operands (activation
quantized by flashinfer's own quantizer, weights from the checkpoint via
exact dequant->requant; needed a tvm_ffi make_kwargs_wrapper shim for the
cutlass-dsl 4.6.0 drift):

| shape | M | sparkinfer blockscaled | flashinfer b12x |
|---|---:|---:|---:|
| gate_proj (N=17408, K=5120) | 4 | 491-667 GB/s | 411 GB/s |
| gate_proj | 16 | 435-586 GB/s | 352 GB/s |
| down_proj (N=5120, K=17408) | 4 | (same class) | 491 GB/s |
| down_proj | 16 | (same class) | 675 GB/s |

The historical implementation's NVFP4 kernel WAS flashinfer B12x (vLLM
priority-list patch in ``oracle/qwen36_vllm/nvfp4_b12x_patch.py``). It
streams small-M decode shapes at the same 350-675 GB/s as sparkinfer --
the standalone small-M FP4 GEMM ceiling is shared, sparkinfer is NOT the
decode gap. The decode-side deficit lives in the FP8 dense layers routed
through ``torch._scaled_mm`` (~400 GB/s class) and in per-round
orchestration; the sparkinfer-native fix is extending
``gemm.tensor_fp8_channel_linear`` (in progress upstream, M=1-only today)
to M<=16 with a per-token scale epilogue.

## Combined historical-kernel mode: 1.77x e2e, confirmed twice (2026-08-04)

`QSR_QWEN36_HIST_KERNELS=1` (single switch) = all-M W4A4 NVFP4 MLP via
sparkinfer blockscaled + all FP8 dense layers via the self-owned
``fp8_w8a8_sm120.so`` (historical CUTLASS port). Interleaved same-caliber
W1-S (n16, c=4, K=3), two repeats per arm:

| arm | wall | committed/s | prefill | decode | accept% | committed |
|---|---:|---:|---:|---:|---:|---:|
| combo run 1 | **33.3 s** | **123.7** | 15.8 s | 17.4 s | 72.3 | 4120 |
| combo run 2 | **33.2 s** | **124.1** | 16.1 s | 17.1 s | 72.3 | 4120 |
| W4A4 + torch FP8 | 42.0 / 39.0 s | 98.0 / 105.6 | 19.7 / 17.8 s | 22.3 / 21.1 s | 71.2 | 4115 |
| W4A16 baseline (typical) | 58.7 / 60.2 s | 68.2 / 69.9 | 31.8 / 33.1 s | 26.9 / 27.0 s | 70.8 / 74.7 | ~4116 |

- 1.77x faster than baseline; accepted ~85 tok/s (up from 40.3 at session
  start = 2.1x); decode round ~13.3 ms vs historical ~11.6 ms.
- Quality anchors: acceptance 72.3% (historical 70.29%), committed 4120
  (historical 4116) -- AT the historical quality level.
- The FP8-dense half of the combo is the self-owned .so (a port of the
  historical CUTLASS kernel, M=1 bit-exact vs it), NOT sparkinfer yet; the
  sparkinfer-native replacement is extending ``tensor_fp8_channel_linear``
  (M=1-only today) to M<=16 with a per-token row-scale epilogue.

## Per-shape large-M FP8 routing: NEGATIVE at e2e (2026-08-04)

Microbench said: M=4096 torch._scaled_mm 40-60% faster on narrow-output
shapes, native faster on N>=16384. Implemented rows>1024 && N<16384 ->
torch routing on top of the combo mode. E2E W1-S: wall 35.5 s vs 33.3 s
combo (prefill 17.9 vs 16.4, decode 17.6 vs 16.9) and acceptance shifted
72.3->67.8 (prefill numerics change flips the greedy trajectory).
REVERTED. Microbench-to-e2e extrapolation keeps failing on this box
(co-tenant + allocator state); only e2e-proven changes are kept.

## Fused silu*mul*quant attempts: CLOSED as dead end (2026-08-04)

Two attempts to fuse ``F.silu(gate)*up + NVFP4 quantize`` for the W4A4 MLP
path: (1) fp32 silu*mul variant -- 2.2% of e4m3 scale bytes flip (fp32 vs
bf16 silu semantics), cascading to 3.6 value-grid diffs, not acceptable;
(2) "bit-safe" variant replicating torch's op chain (libdevice exact exp +
bf16 rounding at each step) -- still NOT bit-identical (torch's exact silu
op chain not reproduced; chasing further ulp-level op-order parity has
negative expected value vs the ~0.4-0.6 s potential e2e gain). Both
reverted; the torch chain stays. The prefill elementwise zoo (~33% of
prefill GPU) remains open but requires op-fusion work that preserves torch
semantics bit-exactly -- scoped as days-scale, not incremental.

## Triton zero-centered RMSNorm for prefill: prefill -1.7s but anchor break, REVERTED (2026-08-04)

Qwen36RMSNorm rows>=128 (prefill-only; decode/verify rows<=16 keep the
torch chain so captured graphs are untouched) routed to the existing Triton
rms_norm with (1+w) precomputed. Numerics vs torch chain: cos 0.99999988,
1-2 bf16 ulp. E2E W1-S: prefill 14.7 s vs 15.8-16.4 ref (-1.7 s real),
decode per-round unchanged (13.4 vs 13.0 ms), BUT acceptance collapsed to
57.1% vs the 72.3% arms (historical anchor 70.29%): the ulp-level norm
delta cascades through 64 layers into a systematically different greedy
trajectory. Quality-anchor regression -> REVERTED per the anchors-first
discipline. Confirms the general wall: every op-level change that is not
bit-exact moves the acceptance anchor unpredictably; prefill zoo fusion
needs bit-exact semantics (accumulation-order-exact variance), which the
current Triton kernel does not provide.

## Bit-exact fused FP8 per-token quantizer: anchor held, e2e WORSE, wiring reverted (2026-08-04)

runtime/kernels/fp8_per_token_quant.py: fused Triton replacement of the
6-op torch quant chain, BIT-EXACT (needed div.rn.f32 inline asm -- triton
div.full and fdiv(ieee_rounding=True) both measure +-1ulp off and flip
e4m3 subnormal-midpoint ties; tests/test_fp8_per_token_quant.py 6/6).
E2E W1-S with the fusion wired: anchor held exactly (72.3%, committed
4120 -- bit-exact as designed) BUT wall 37.9 s vs 33.3 s ref: prefill flat
(16.2 vs 16.4 -- torch's vectorized chain is not dispatch-bound) and
decode +4.7 s (one-program-per-row kernel with BLOCK_K=8192 massively
under-utilizes at rows<=16). Wiring REVERTED; kernel + bit-parity tests
kept as tooling (a 2-D-grid variant would be needed before any rewire).
