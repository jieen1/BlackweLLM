# Qwen3.8 DFlash2 end-to-end hotspot audit

## Scope

This note records the first whole-chain optimization pass for the Gittensor
`Qwen3.8-27B-NVFP4-RTX5090` checkpoint. Measurements use the isolated runtime
on 2026-08-25 with FP8 KV, prefix cache, CUDA Graphs, 128K context, four
concurrent requests, DFlash2 `K=7`, and the `torch-nightly` environment. The
existing service was not changed.

## Profile evidence

The c4 Nsight Systems run was
`/tmp/qwen38_dflash2_fullchain_nsys_c4_20260825.nsys-rep`. GPU time was
dominated by the target model rather than the DFlash2 accept algorithm:

| Hotspot | Share of profiled GPU time |
| --- | ---: |
| FlashInfer FP8 paged full-attention verify | 30.3% |
| ModelOpt W4A4 dense GEMMs | 26.1% |
| Two full-vocabulary BF16 `lm_head` GEMMs | 10.4% |
| GDN recurrent indexed multistep | 3.7% |

The DFlash2 graph epilogue's small device-to-host decision publication is a
real synchronization point, but it is not the largest GPU consumer. Disabling
FlashInfer split-KV is not an optimization candidate: the real Qwen geometry
microbenchmark measured approximately 0.231 ms (B1) and 0.936 ms (B4) with
split-KV versus 17.79 ms and 18.25 ms without it.

## Implemented candidate

The Gittensor export keeps `lm_head.weight` as BF16 with shape
`[248320, 5120]`. `runtime/model/fp8_lm_head.py` adds an opt-in serving-time
conversion to per-output-channel E4M3 weights plus the existing shared
per-token activation quantizer, then invokes b12x's native SM120 FP8 GEMM.
The conversion happens after checkpoint loading and before any target/draft
CUDA Graph capture, so DFlash2 and the target share one prepared head.

The switch is deliberately off by default:

```text
QSR_NATIVE_QWEN38_LM_HEAD_FP8=1
```

GGUF, unsloth checkpoint-native compressed heads, and all other shapes keep
their existing path.

## Validation

The actual Gittensor head microbenchmark measured roughly 1.64 ms BF16 versus
0.90 ms native FP8 at M=7 and 2.33 ms versus 0.98 ms at M=32. An isolated
end-to-end c4 run improved warm per-request decode from the prior 155.51
tok/s to 170.43 tok/s (first warm wave) and 167.80 tok/s (second warm wave).
The candidate produced the same output SHA as the baseline. Acceptance was
896/1020, but it took 129 rounds instead of the baseline's 128, so the
candidate is not promoted to the default until broader natural-language
quality and acceptance coverage is complete.

The next pass kept the native FP8 head and changed only the exact Gittensor
W4A4 MLP-down shape `[M, 17408] @ [17408, 5120]`: DFlash2 verify rows M=8..32
use b12x's measured TMA `(64, 64)` MMA tile; M=1, large prefill, and all other
shapes retain the default selector. The isolated c4 result was 171.52 and
170.28 tok/s for the two warm waves, versus 170.43 and 167.80 tok/s for the
same service before the tile change (mean +1.06%). Both runs had 896 accepted
tokens, 1020 committed tokens, and the same output SHA. The c1 cold result was
247.52 tok/s; it is recorded but not used as the primary comparison because
the optimization target is steady-state batched decode.

Targeted regression tests pass (`39 passed`), touched-file ruff and full
repository ruff pass, and the CUDA-Graph replay smoke test showed stable
eager/replay outputs after capture.

## Next whole-chain candidates

The next profiling pass should target the 30.3% FlashInfer verify-attention
share. The W4A4 dense-GEMM path now has a shape-scoped decode tile for the
measured verify regime, but broader shapes still need separate evidence before
any selector changes. Do not spend the next pass changing DFlash2 acceptance
semantics or disabling split-KV unless a new profile proves that premise has
changed.
