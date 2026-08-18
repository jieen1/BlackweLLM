# Qwen3.8 DSpark: live duplicate-prefix parity run (2026-08-19)

状态：🟢 **同口径 cold 与 steady decode 均超过 SGLang 参考**

## 固定口径

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q, SM120, driver 610.88
- local Python: `/home/bot/.venvs/torch-nightly/bin/python`
- model: `unsloth/Qwen3.8-27B-NVFP4`
- draft: `RadixArk/Qwen3.8-27B-DSpark`, DSpark `K=7`
- four concurrent requests; each prompt is exactly 131072 tokens; `max_tokens=256`
- identical tokenizer, filler prompt, request parameters, completion length, block/page size 128
- CUDA Graph enabled for decode, DSpark draft and ragged verify; FP8 KV; prefix cache enabled
- local prefill chunk `8192`, admission coalesce `10 ms`, exact duplicate admission dedup enabled
- every completion SHA must equal:
  `75b43a8a0ae256dca5668dd5e73028d24f8700d46b7d5623526bc29711dff306`

SGLang reference is `/tmp/sglang_dspark_same_128k_c4_20260818.json`, commit
`b296e1a503`, with the same prompt fixture and tokenizer. Its steady value is
the second warm wave: `1024 / 2.609 = 392.49 tok/s`; the first warm wave
(`5.937s`) includes its one-time warmup/JIT overhead.

## Result after live-prefix publish

Artifact set:

- `benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c4_qwen38_live_publish_c4_20260819.json`
- matching `server_stats_*` and `server_trace_*` files
- `benchmarks/fixtures/server_profile_qwen38_dynamic_128k_c4_qwen38_live_profile_c4_20260819.log`
  and the matching client log for the phase breakdown
- `benchmarks/fixtures/sglang_dspark_same_128k_c4_20260818.json` (copied from
  the original `/tmp` reference so the comparison input is retained)
- `benchmarks/fixtures/qwen38_live_publish_c4_20260819.env` (exact run env)

| runtime | cold wall | warm 0 | warm 1 | completion SHA |
|---|---:|---:|---:|---|
| local QSR | **40.7986s** | **443.74 tok/s** | **450.88 tok/s** | all 12 exact |
| SGLang | 50.2784s | 172.47 tok/s* | **392.49 tok/s** | all exact |

`*` The SGLang first warm wave is `1024 / 5.9373 = 172.47 tok/s` and is
reported only to make the warmup visible; the steady comparison is warm 1.
Local is `+14.88%` over SGLang warm 1. Local cold is `18.85%` shorter.

The local cold stats prove that the four identical prompts did not each pay
the 128K prefill:

- `prefix_cache_hits=3`, `prefix_persistent_restores=3`, `prefix_cache_misses=1`
- `prefill_batched_forwards=17` (one B1 prefill plus the normal graph/commit
  sequence), versus 33 before live KV publication
- DSpark `136` rounds, `904` accepted and `1020` committed tokens
- all CUDA graph debug states are `captured`, including `dspark_verify_ragged`
- `prefix_publish_failures=0`; all completion SHA values match the SGLang run

## Root cause and change

The scheduler-side duplicate grouping was already able to select one request
first and defer the other three. The remaining 170s cold path was caused by
the dynamic arena only publishing KV block hashes on `reset_slot`. At the
moment the deferred requests were admitted, the first request still owned its
live bundles, so the persistent metadata hit had no arena bundle to restore
and the three requests recomputed the prompt.

`Qwen36Backend._store_persistent_prefix(..., prompt_hidden=...)` now publishes
the exact prompt-boundary chained block hashes while the source bundles are
still live. The next request restores them by `incref`; existing write-time
COW keeps the source's later decode writes from modifying the shared prefix.
The DSpark draft state remains in its independent scratch snapshot and is
restored together with the shared backbone prefix.

Regression coverage includes a live-source fan-out allocator test and the
engine/backend admission tests. The focused suite passed `102` tests; Ruff
passed for the changed backend and dynamic-arena test.

## Remaining optimization target

The target is now cleared on the exact 4x128K workload, but the next pass
should continue from the retained trace rather than changing the benchmark
口径. The local warm c4 trace reports about `53.7 ms` per B4 DSpark round and
about `7.5` committed tokens per round. Kernel attribution from the prior
same-toolchain profile still points to small-M W8A8 and W4A4 GEMMs as the
largest decode body; any further optimization must preserve the current
completion SHA, captured ragged verify path, and the live-prefix COW invariant.

The existing opt-in graph-folded context-KV epilogue was also measured with
the same workload. Its artifact is
`server_perf_grid_qwen38_dynamic_128k_c4_qwen38_fusedctx_c4_20260819.json`:
warm `448.55/420.95 tok/s`, versus the default `450.75/448.27 tok/s` in the
paired profile run. It is rejected for the default path; the extra graph
work did not reliably amortize the separate `3.24 ms` hidden-sync phase.
