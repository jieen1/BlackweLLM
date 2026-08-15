# Qwen3.8 dynamic MTP 首请求接受率回退：CUDA Graph page table 映射代际

Date: 2026-08-15  
Status: CLOSED — root cause fixed and fresh-process GPU verification passed.

## 结论

CUDA 13.3 / Triton 3.7.1 并没有直接改变 Qwen3.8-27B-NVFP4 的 MTP
数值。真正的问题是 dynamic KV arena 与 MTP CUDA Graph 的 page-table
缓存契约不完整：draft 和 teacher-forced sync 图只用 `tuple(slots)` 判断是否
需要重拷 page table，但同一个 slot 在 capture 后、reset 后和新请求分配时会
指向不同的物理 bundle。

因此 fresh process 的第一条 slot-0 请求沿用了 capture 阶段的旧物理映射；
换到另一个 slot 时，`tuple(slots)` 改变，page table 被偶然刷新，于是表现成
“cold 低接受、warm 正常”。这不是 GPU 低频、prefix cache、首次 JIT 或输入
问题，而是 wrong-address/right-shape 的静默正确性 bug。

修复将 target verify、MTP draft、sync decode 和 sync verify 四处缓存键统一为：

```text
legacy:  (slot, ...)
dynamic: ((slot, page_table_version), ...)
```

dynamic row 发生 allocate/release/COW remap 后，下一次 replay 必然重新复制
当前 `_global_page_table`。没有保留任何“启动时多跑几轮”的掩盖式 warmup。

## 环境与工作树基线

- repo: `main@7eced58`（调查开始时与 `origin/main` 一致）
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q, SM120, driver 610.88
- CUDA runtime 13.3
- torch `2.13.0a0+gitcf30153`
- Triton 3.7.1
- CUTLASS DSL 4.7
- model: `unsloth/Qwen3.8-27B-NVFP4@9c73e2d...`
- serving: MTP K=3, CUDA Graph, FP8 KV, elastic 19,629,342,720-byte pool,
  capacity/slots=4, logical max 256K/slot
- workload: exact 131,072-token raw completion prompt, 256 greedy output tokens

GPU idle clock was 427–667 MHz, but the real 128K run held roughly 85–97% GPU
utilization and ~1.7–2.2 GHz SM clock. Idle clock was not used as performance
evidence.

## 可判别实验

| Experiment | Prefix | First work in process | MTP histogram (accepted 0..3) | committed/round | decode tok/s |
|---|---:|---|---:|---:|---:|
| original fresh | on | 128K request | `[7,31,21,31]` | 2.844 | 74.95 |
| original fresh | off | 128K request | `[7,31,21,31]` | 2.844 | 75.27 |
| prefill-only warmup | off | 2048-row MTP prefill, then 128K | `[7,31,21,31]` | 2.844 | 74.86 |
| one complete MTP-round warmup | off | 5-token prefill + one round, then 128K | `[5,10,15,47]` | 3.351 | 87.40 |
| short real request first | off | complete 4K request, then 128K | `[0,0,0,64]` | 4.000 | 102.23 |
| **mapping-version fix** | off | **128K request** | **`[0,0,0,64]`** | **4.000** | **101.58** |
| **mapping-version fix** | on | **128K cold** | **`[0,0,0,64]`** | **4.000** | **99.35** |
| **mapping-version fix** | on | **same 128K persistent hit** | **`[0,0,0,64]`** | **4.000** | **103.55** |

Every listed 128K path produced the same completion SHA-256:
`75b43a8a0ae256dca5668dd5e73028d24f8700d46b7d5623526bc29711dff306`.
The wrong MTP mapping degraded speculation efficiency without changing the final
greedy target output in this fixture; that does not make the addressing bug safe.

Key artifacts:

- failing fresh process:
  `benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c1_original_fresh_r1_20260815.json`
- fixed fresh process, prefix disabled:
  `benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c1_mapping_version_fix_noprefix_20260815.json`
- fixed production cold + warm:
  `benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c1_mapping_version_fix_coldwarm_20260815.json`
- fixed true-cold c4 submission:
  `benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c4_mapping_version_fix_noprefix_cold_20260815.json`

The true-cold c4 submission was not four simultaneous active decodes: full-sequence
reservation admitted three requests first and left one waiting, so wall time was
251.4s and the server-level mean decode metric (5.05 tok/s) includes queueing and is
not a kernel throughput number. Its correctness signal remains useful: all 256 MTP
rounds were full accepts, every request returned the same expected SHA, and the
three-request active wave traced 87.5 tok/s/request while the final B1 wave traced
106.4 tok/s.

## 独立发现：128K prefix publication 的 O(n²) 主机开销

`_chained_block_keys` previously passed `token_ids[:end]` at every block boundary,
even though `parent_hash` already authenticated all prior blocks. At 128K and
block size 16 this processed about 537 million token values.

The fixed implementation hashes only the new block slice. CPU microbenchmark on
the exact 128K shape changed 5.352s to 10.8ms (494.4x). Bulk arena publication now
checks global invariants once per transaction instead of once per key. Under
`QSR_ASSERT_LEVEL=1`, post-request bookkeeping fell from about 8.34s to 53–55ms;
production persistent-hit TTFT fell from 13.97s to 51–52ms.

This optimization is independent of the MTP acceptance fix: disabling prefix cache
did not change the original acceptance failure, while fixing the page-table mapping
restored full acceptance with prefix cache still disabled.

## 被排除的解释

- **GPU 667MHz**: it was the idle clock; active clocks/utilization were normal.
- **CUDA 13.3 / Triton compiler numerics**: same target output and deterministic
  slot/history dependence; mapping-version fix alone restores first-request parity.
- **persistent prefix cache**: failure reproduces with prefix cache disabled.
- **long-prefill JIT/planner warmup**: a real 2048-row MTP prefill does not help.
- **generic first MTP replay**: one complete round changes but does not close the
  gap, which is consistent with repeatedly mutating stale graph-addressed storage,
  not a legitimate one-time compiler transition.
- **measurement aggregation**: the prior 57.6ms “warm B1” trace mixed cumulative
  B4 rows. Request traces now expose bounded per-round rows directly.

## Verification contract retained in code

- Unit test proves a dynamic page-table cache key changes when the same slot's
  mapping version changes; legacy keys retain the old zero-overhead shape.
- Benchmark artifacts now retain exact completion text/SHA and acceptance
  histogram deltas/derived committed-per-round values.
- `/debug/traces` exposes recorded per-round committed-token/time rows so B1/B4
  and one-round/sustained effects cannot be conflated again.

