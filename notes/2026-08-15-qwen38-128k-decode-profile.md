# Qwen3.8 NVFP4: 128K-context decode profile

Date: 2026-08-15  
Hardware: one NVIDIA Blackwell SM120 GPU  
Model: `unsloth/Qwen3.8-27B-NVFP4`, snapshot
`9c73e2daee1d0fd494ffbd1d8753f2174a953796`

## Exact server configuration

The reusable command is
[`scripts/run_qwen38_128k_decode_bench.sh`](../scripts/run_qwen38_128k_decode_bench.sh).
The measured server used:

- `qwen36` backend, MTP enabled with `K=3`, CUDA Graphs enabled.
- FP8 E4M3 KV, block size 16.
- `capacity=4`, `num_slots=4`, logical ceiling 256K tokens per slot.
- Elastic dynamic KV pool: 19,629,342,720 bytes, 4,160 page bundles,
  watermark 8 bundles. Each bundle contains backbone and MTP KV.
- Persistent prefix cache and flight recorder enabled.
- Request timeout 900 seconds; offline checkpoint loading.
- Completion endpoint, exact 131,072-token digit-filler prompt, 256 generated
  tokens. The recorded harness used the local Qwen3.6 tokenizer to construct
  the filler; server-side Qwen3.8 tokenization independently reported exactly
  131,072 prompt tokens.

## Results

These values deliberately separate prefill-inclusive request throughput from
steady-state decode. The latter comes from `/debug/traces`, where the retained
128K prefix made `prefill_ms=0` for the measured decode phase.

| Workload | Decode evidence | Decode throughput | Prefill-inclusive result |
|---|---:|---:|---:|
| 1 request, 128K context | 252 tokens / 2.3311 s, 64 rounds | **108.1 tok/s** | 256 tokens / 61.38 s = 4.17 tok/s; TTFT 58.87 s |
| 4 requests, wave 1 | 4 × 254 tokens / 4.9691 s, 92 rounds | **204.4 tok/s aggregate** | not used for decode claim |
| 4 requests, wave 2 | 4 × 254 tokens / 5.1447 s, 92 rounds | **197.6 tok/s aggregate** | not used for decode claim |
| 4 requests, wave 3 | 4 × 254 tokens / 4.9886 s, 92 rounds | **203.6 tok/s aggregate** | not used for decode claim |

The three four-request waves average approximately **201.9 aggregate tok/s**,
or about 50.5 tok/s per request. This is a descriptive summary of one fixed
server run, not a before/after optimization claim.

GPU memory was 56,650 MiB after startup and reached 60,072 MiB during/after the
four-request 128K run. The configured dynamic KV budget was 18.28 GiB; pages are
committed and retained on demand rather than preassigning four complete 256K
rows.

## What the trace established

The MTP acceptance histogram after the three four-request waves was
`[24, 532, 256, 304]` across 1,116 slot-rounds. It corresponds to 1.753 accepted
draft tokens out of three on average (58.4% draft acceptance), or 2.753 committed
tokens per slot-round including the target token. At the observed round time,
perfect acceptance would raise the mathematical ceiling to about 292.5 aggregate
tok/s. This identifies acceptance as a high-value measurement target; it does
not by itself prove an implementation bug.

The repeated 128K waves also exposed a separate correctness/performance issue:
dynamic arena KV pages hit (`prefix_kv_hit_tokens=1,048,448`), but recurrent GDN
checkpoint restores remained zero (`prefix_state_hit_tokens=0`). Therefore the
old server recomputed the prompt despite retaining its KV. The accompanying code
change stores the recurrent checkpoint and anchor for dynamic entries, revives
the retained backbone/MTP bundle mapping without a copy, and preserves the live
GDN speculative-state column during restore. New regression tests cover full
repeat restoration, partial-page reservation accounting, and the GDN column.

## Raw evidence

- [`server_perf_grid_qwen38_dynamic_128k_c1_20260815.json`](../benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c1_20260815.json)
- [`server_trace_qwen38_dynamic_128k_c1_20260815.json`](../benchmarks/fixtures/server_trace_qwen38_dynamic_128k_c1_20260815.json)
- [`server_perf_grid_qwen38_dynamic_128k_c4_20260815.json`](../benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c4_20260815.json)
- [`server_trace_qwen38_dynamic_128k_c4_20260815.json`](../benchmarks/fixtures/server_trace_qwen38_dynamic_128k_c4_20260815.json)
- [`server_stats_qwen38_dynamic_128k_c4_20260815.json`](../benchmarks/fixtures/server_stats_qwen38_dynamic_128k_c4_20260815.json)

## Next performance work, in evidence order

1. Profile two representative decode rounds (`QSR_PROFILE_ROUNDS=2`) and retain
   the profiler artifact before changing kernels. The 54--56 ms four-slot round
   time is real, but this run does not attribute it to a particular kernel.
2. Run a content-matched MTP `K=1/2/3` sweep and inspect reject-position shape.
   The measured 58.4% acceptance leaves more token-yield headroom than a small
   launch-overhead reduction, but quality and accepted tokens/s are the gate.
3. Re-run the exact-repeat 128K warm cell after the dynamic GDN checkpoint fix.
   This validates TTFT/prefill elimination; it must not be presented as a
   steady-state decode optimization.
4. Treat KV dtype as a cold-start A/B. FP8 KV was used here; changing it requires
   a fresh process plus correctness/top-k agreement checks.
