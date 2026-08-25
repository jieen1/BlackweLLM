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
`[248320, 5120]`. `runtime/model/fp8_lm_head.py` converts it after loading to
per-output-channel E4M3 weights, uses the existing shared per-token
activation quantizer, and invokes b12x's native SM120 FP8 GEMM. The
conversion happens before any target/draft CUDA Graph capture, so DFlash2 and
the target share one prepared head.

The ModelOpt checkpoint family enables this path automatically:

```text
QSR_NATIVE_QWEN38_LM_HEAD_FP8=0  # immediate BF16 rollback
```

`QSR_NATIVE_QWEN38_LM_HEAD_FP8=1` is an explicit opt-in for another eligible
plain head. GGUF, Unsloth checkpoint-native compressed heads, and all other
shapes keep their existing path.

## Validation

The actual Gittensor head microbenchmark measured roughly 1.64 ms BF16 versus
0.90 ms native FP8 at M=7 and 2.33 ms versus 0.98 ms at M=32. A fair fresh
process 4-wave comparison at 128K/C=4, with FP8 KV, prefix cache, CUDA Graphs,
DFlash2 K=7, and all other settings identical, measured:

| Head path | warm decode/request | aggregate E2E | acceptance | output SHA |
| --- | ---: | ---: | ---: | --- |
| BF16 baseline | 155.47 tok/s | 534.44 tok/s | 896/896 | `75b43a8a…` |
| native FP8 | 163.10 tok/s | 562.51 tok/s | 896/896 | `75b43a8a…` |

That is +4.91% decode and +5.25% aggregate E2E. The native path is therefore
promoted for ModelOpt by default, with the environment rollback above. A
separate default-path smoke (no environment setting) prepared the head,
captured all DFlash2 graphs, and produced the same output SHA.

The next pass kept the native FP8 head and changed only the exact Gittensor
W4A4 MLP-down shape `[M, 17408] @ [17408, 5120]`: DFlash2 verify rows M=8..32
use b12x's measured TMA `(64, 64)` MMA tile; M=1, large prefill, and all other
shapes retain the default selector. The isolated c4 result was 171.52 and
170.28 tok/s for the two warm waves, versus 170.43 and 167.80 tok/s for the
same service before the tile change (mean +1.06%). Both runs had 896 accepted
tokens, 1020 committed tokens, and the same output SHA. The c1 cold result was
247.52 tok/s; it is recorded but not used as the primary comparison because
the optimization target is steady-state batched decode.

The FP8 draft KV A/B was also quality-neutral on the same fixture (7/7
acceptance and identical output); its measured end-to-end difference was
below noise, so it remains a memory/traffic consistency change rather than a
claimed speed win. Layered K RMSNorm, target-tap compaction, and the fused
dynamic-convolution kernel each pass numerical tests and improve their local
microbenchmarks, but did not produce a stable end-to-end delta on this
prefix-hit decode fixture.

Full torch-nightly regression passes (`2661 passed, 15 skipped`), full
repository Ruff passes, and the CUDA-Graph replay smoke test showed stable
eager/replay outputs after capture. The documented `/tmp/ci-sim` interpreter
was absent on this machine, so the CPU-only gate was not run.

## Next whole-chain candidates

The next profiling pass should target the 30.3% FlashInfer verify-attention
share. The W4A4 dense-GEMM path now has a shape-scoped decode tile for the
measured verify regime, but broader shapes still need separate evidence before
any selector changes. Do not spend the next pass changing DFlash2 acceptance
semantics or disabling split-KV unless a new profile proves that premise has
changed.
