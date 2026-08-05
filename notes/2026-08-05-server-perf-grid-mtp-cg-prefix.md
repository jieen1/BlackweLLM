# Qwen3.6 服务端性能网格：严格上下文 × 并发（MTP K=3 + CUDA Graph + persistent prefix cache）

日期：2026-08-05 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（snapshot
`ccdaab7e68af2409599b8949a8f2685703c9bae5`）· 单卡 RTX PRO 6000 Blackwell
Max-Q（SM120，96 GB）· 服务：本仓库自有 runtime（`server.app` /
`qwen36` 后端），**无任何外部推理引擎**。

## 一句话

best 服务配置（MTP K=3、decode+MTP 四张 CUDA Graph、persistent prefix
cache、FP8 e4m3 KV、3×256K）下，5 档严格上下文 × 3 档并发共 15 个 cell 全部
完成；**每个上下文的 WARM 波都 100% 命中 persistent cache**（`restores ==
并发数`），250K 的 COLD→WARM 首 token 从 **316.0 s 降到 0.48 s（655.6×）**。
过程中发现并修复了两个相互叠加的持久化缓存 bug（见 §6）：单请求 per-slot
chunked prefill 缺 COW 拆页会**直接覆写共享 scratch KV 字节**，以及 live slot
的已提交别名页无法被逐出导致大条目 store 静默失败。

## 1. 目的与可比性

在“性能无回退”的核查要求下，为 qwen36 best 服务安排一轮**完整的服务端性能
网格**：上下文长度 × 并发矩阵，全部开启 MTP（K=3）、CUDA Graph（decode +
MTP anchor/draft/sync/verify）、prefix cache（含 persistent 跨槽复用）。

对比锚点（文档引用，数字不变）：

* CG 解码吞吐（MTP off，服务路径，短 prompt，decode-only）：
  c=1/2/4 = **28.56 / 47.71 / 68.59 tok/s**
  （`notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md`）。
* MTP 服务路径修复前：MTP-on **7.80 tok/s vs MTP-off 28.0 tok/s**
  （`notes/2026-08-03-mtp-serving-gpu-verification.md`）——本轮数据是
  **MTP 四图全 captured 之后**的端到端数字。
* 质量重跑中的速度证据：2+2 对话 21.8 s（eager）→ 4.1 s（MTP+CG）；
  4096-token HumanEval 726.9 s → 46.5 s
  （`notes/2026-08-05-qwen36-quality-rerun.md` 时间线）。
* prefix cache 理论 ceiling：native vLLM 精确重复 **15.4×**（
  `benchmarks/prefix_cache_warm_throughput_check.py`、
  `notes/prefix-cache-design.md` §25.3）。本运行时是
  **completion-boundary-only 持久复用**（只存 block 对齐边界、只按完整
  prompt 前缀命中），因此达不到原生逐 token 的 15.4× ceiling；本网格的
  冷→热 TTFT 比值就是本实现的实际可达倍数。

## 2. 服务配置（best，`scripts/run_qwen36_quality.sh server start best`）

| 参数 | 值 |
|---|---|
| 后端 | `qwen36`（自建 runtime，无 vllm） |
| capacity / num_slots | 3 / 4（+1 为 decode CG 捕获 warmup 槽） |
| 每槽上下文 | 262144 tokens（block_size=16 × blocks_per_slot=16384） |
| MTP | on，K=3（与历史 speculative config 一致），resync off |
| CUDA Graph | decode（batch≤4）+ MTP anchor/draft/sync/verify 全部 captured |
| prefix cache | on（persistent 跨槽 scratch arena + slot-local 前缀） |
| KV cache dtype | fp8_e4m3 |
| gpu_memory_utilization | 0.92 |
| request timeout | 0（禁用，与历史一致） |
| tool call parser | `qwen3_coder` |
| 模型 | `unsloth/Qwen3.6-27B-NVFP4` snapshot `ccdaab7e...` |

## 3. 网格定义（`benchmarks/server_perf_grid.py`）

每个 cell = 一个上下文长度 × 一个并发数，两波：

1. **COLD 波**：`c` 个并发相同请求打空缓存 → 填充 persistent cache。
2. **WARM 波**：同样的 `c` 个并发请求再来一次 → 应全部持久命中。

headline 数字全部取自服务端自己的 Prometheus metrics（`/metrics`，
`endpoint="chat"`）与引擎计数器（`/debug/stats`）的增量，客户端只负责
驱动并发和记录事件时间戳——MTP 下一个 SSE 事件可携带多个 token，且 Qwen3.6
把大量预算花在不可见文本的 reasoning 上，客户端解析会引入系统性偏差。

上下文长度**严格**且 block 对齐：chat 模板开销按上下文逐点实测（不是常数——
2026-08-05 实测 raw 32758→served 32767、raw 32759→served 32769，**32768
本身不可达**），served prompt_tokens 必须是 block_size=16 的倍数，persistent
缓存只在 block 对齐边界存 checkpoint。每波校验
`prompt_tokens_total 增量 == served_context × c`。可达值：

| 目标 | served（chat）tokens | raw prompt tokens |
|---:|---:|---:|
| 4K | 4096 | 4086 |
| 32K | 32752 | 32742 |
| 64K | 65536 | 65526 |
| 128K | 131072 | 131062 |
| 250K | 256000 | 255990 |

filler 为 `"0123456789 "`（decode→re-encode 1:1 精确往返）。⚠️ 不同长度的
prompt **不是前缀关系**（模板边界合并使长 prompt 的头部在几百 token 内即
分叉），所以长上下文 COLD 波不会从短上下文条目上吃到部分命中——每档 COLD
都是真冷，WARM 只吃同长度条目。这是设计使然，也让 COLD 数字互不污染。

命令：

```bash
/home/bot/.venvs/vllm/bin/python -u benchmarks/server_perf_grid.py \
  --base-url http://127.0.0.1:8300 --model qwen3.6 \
  --contexts 4k,32k,64k,128k,250k --concurrency 1,2,3 \
  --max-tokens 256 --warm-rounds 1 \
  --out benchmarks/fixtures/server_perf_grid_20260805_v2.json
```

输出逐 cell 增量落盘，`--resume` 可从上次中断处续跑；原始 JSON 在
`benchmarks/fixtures/server_perf_grid_20260805_v2.json`（每个 wave 含
per-request TTFT、metrics/stats 增量、MTP/CG 计数器）。

## 4. 结果（正式网格，2026-08-05 20:56:33–21:07:36 本地 = 12:56:33–13:07:36 UTC，修复后）

> 指标：TTFT = 服务端 time-to-first-token 均值（s）；decode = 服务端
> `request_time_per_output_token` 折算的 decode tok/s（MTP 后每事件多 token，
> 以 tokenizer 重编码的文本增量校准）；e2e = aggregate 端到端 tok/s；
> restore = 该波 `prefix_persistent_restores` 增量（应等于并发数）。
> COLD c≥2 波因同长度条目已存在，实际是持久命中（表中注明），只有各档
> c=1 COLD 是真冷。

| 上下文 | c | COLD TTFT s | COLD decode tok/s | COLD e2e tok/s | COLD hits/restores | WARM TTFT s | WARM decode tok/s | WARM e2e tok/s | WARM hits/restores | 冷→热 TTFT 比值 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| 4K | 1 | 2.064 | 67.9 | 44.0 | 0/0 | 0.080 | 69.8 | 68.5 | 1/1 | 25.7× |
| 4K | 2 | 0.101 | 64.7 | 125.8 | 2/2 | 0.054 | 68.7 | 135.1 | 2/2 | 1.9× |
| 4K | 3 | 0.120 | 65.8 | 191.2 | 3/3 | 0.092 | 54.9 | 161.5 | 3/3 | 1.3× |
| 32K | 1 | 21.921 | 64.0 | 9.9 | 0/0 | 0.065 | 67.1 | 66.2 | 1/1 | 335.7× |
| 32K | 2 | 0.133 | 61.7 | 118.5 | 2/2 | 0.122 | 57.3 | 110.7 | 2/2 | 1.1× |
| 32K | 3 | 0.209 | 35.8 | 104.5 | 3/3 | 0.180 | 39.4 | 115.2 | 3/3 | 1.2× |
| 64K | 1 | 47.167 | 63.5 | 5.0 | 0/0 | 0.115 | 58.2 | 56.9 | 1/1 | 411.6× |
| 64K | 2 | 0.203 | 46.8 | 90.0 | 2/2 | 0.224 | 51.5 | 98.8 | 2/2 | 0.9× |
| 64K | 3 | 0.299 | 34.1 | 97.0 | 3/3 | 0.287 | 34.6 | 98.2 | 3/3 | 1.0× |
| 128K | 1 | 123.065 | 48.0 | 2.0 | 0/0 | 0.258 | 48.4 | 46.2 | 1/1 | 477.0× |
| 128K | 2 | 0.345 | 37.8 | 71.9 | 2/2 | 0.373 | 36.5 | 69.1 | 2/2 | 0.9× |
| 128K | 3 | 0.532 | 31.6 | 88.2 | 3/3 | 0.566 | 31.5 | 88.1 | 3/3 | 0.9× |
| 250K | 1 | 315.999 | 42.8 | 0.8 | 0/0 | 0.482 | 41.4 | 38.5 | 1/1 | 655.6× |
| 250K | 2 | 0.681 | 27.3 | 50.7 | 2/2 | 0.665 | 29.2 | 54.0 | 2/2 | 1.0× |
| 250K | 3 | 0.919 | 24.4 | 66.3 | 3/3 | 0.982 | 24.5 | 67.2 | 3/3 | 0.9× |

### 与历史锚点的对比

**MTP 不再有吞吐回退**（对比修复前 0.28× 的结论）：MTP-off + CG 的
c=1/2/4 解码锚点是 28.56 / 47.71 / 68.59 tok/s（短 prompt、decode-only
时间窗，`notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md`）。本轮
MTP on 四图全开后，4K 上下文的服务端 decode tok/s 为 **67.9 / 64.7 / 65.8
（c=1/2/3）**——c=1/2 反超 MTP-off 锚点，c=3 与锚点 c=4 同量级；最重的
250K 上下文 c=1 也有 42.8 tok/s，远离修复前 MTP-on 7.80 tok/s 的谷底。
口径提醒：锚点是短 prompt + decode-only 时间窗，本轮是长 prompt prefill
后 256-token 生成内按服务端 per-output-token 折算，两者都排除 prefill，
可比但不是同一个 harness。

**prefix cache 命中收益**（对比 native 15.4× 天花板）：native vLLM 的
`--enable-prefix-caching` 在 256K/c=4 字节精确重复上测得整波墙钟
775.9 s → 49.6 s（≈15.4×，`notes/prefix-cache-design.md` §25.4）。本网格
逐档 c=1 冷→热：

| 上下文 | COLD TTFT | WARM TTFT | TTFT 比值 | 整波墙钟比值 |
|---|---:|---:|---:|---:|
| 4K | 2.064 s | 0.080 s | 25.7× | 1.6× |
| 32K | 21.921 s | 0.065 s | 335.7× | 6.7× |
| 64K | 47.167 s | 0.115 s | 411.6× | 11.4× |
| 128K | 123.065 s | 0.258 s | 477.0× | 23.2× |
| 250K | 315.999 s | 0.482 s | 655.6× | 48.4× |

TTFT 比值随上下文拉大（COLD 越长越吃亏）；250K 整波墙钟仍快 **48.4×**。
本运行时是 completion-boundary-only 持久复用（只存 block 对齐边界、只按
完整 prompt 前缀命中），达不到 native 逐 token 前缀缓存的机制上限；上面
这组比值就是本实现的实际可达倍数。c≥2 波因同长度条目已由 c=1 波存入而
全部持久命中（比值 ~1×，见 §4 注）。

**与质量重跑的速度证据同源**：2+2 对话 21.8 s（eager）→ 4.1 s（MTP+CG）；
4096-token HumanEval 726.9 s → 46.5 s
（`notes/2026-08-05-qwen36-quality-rerun.md` 时间线）——同一 best 服务配置
下的端到端数字，质量侧无回退（MMLU-Pro 84.54% 精确复现等）。

## 5. 数据完整性检查

* 每个 WARM 波 `prompt_tokens_total` 增量 == served × c，且
  `prefix_persistent_restores == c`（15/15 cell，见 §4）。
* 全网格累计（服务端 `/debug/stats`，与各 wave `stats_delta` 之和一致）：
  `requests_completed=60`、`prefix_cache_hits=55`、`prefix_cache_misses=5`、
  `prefix_persistent_stores=5`、`prefix_persistent_restores=55`、
  `prefix_persistent_evictions=4`、`checkpoints_taken=407`。
* MTP/CG 计数：`mtp_draft_graph_replays=2551`、
  `mtp_verify_graph_replays=2491`、`mtp_batched_sync_replays=2491`
  （JSON `stats_delta` 可逐 cell 核对）。`decode_graph_replays=0` 是
  **预期**的：MTP on 时每轮 decode 走 MTP draft/sync/verify 四图
  （`runtime/backends/qwen36_mtp.py`），plain decode CG 计数只在 MTP-off
  路径累加。
* 服务日志 `engine ready: ... cudagraph=True prefix_cache=True mtp=True(K=3)`，
  `_cuda_graph_dbg` 显示 decode + mtp draft/sync/verify 均 `captured`。
* 单测：`tests/test_qwen36_backend.py`、`tests/test_qwen36_slot_pool.py`
  等全套 **1874 passed / 3 skipped**；torch-free CI 镜像 **1150 passed /
  192 skipped**；`ruff check` 通过。
* 注：JSON 顶层 `server_stats_after` 的原始快照因 harness 最后一次
  `/debug/stats` 读取瞬时报零，已用同一服务进程（PID 3140874，网格后无新
  请求）的计数补齐并写入 `server_stats_after_note`；各 wave `stats_delta`
  是权威记录。

## 6. 过程中发现并修复的两个 bug（2026-08-05）

第一版网格跑到 250K 时，WARM 波 `hits=0 restores=0`、TTFT 与 COLD 相同
（~309 s）——persistent store 静默失败。计数证据：
`prefix_persistent_stores=4`（只有 4K/32K/64K/128K），
`prefix_persistent_evictions=3`（250K 尝试逐出 4 条只成功了 3 条）。CPU
stub 池完整复现（见 `tests/` 新增回归测试），根因是两个相互叠加的问题：

### 6.1 单请求 chunked prefill 直接写穿共享 scratch 页（正确性 bug）

`prefill_chunked_step` 的 per-slot 路径直接 `self.model(input_ids,
state)`，而 state 的 attention cache 持有 `_global_page_table[slot]`
的 view（`runtime/model/qwen36_slots.py` 构造时
`page_table=self._global_page_table[row:row+1]`）；`append` 按 page table
写物理页。batched 路径在 `build_prefill_batch` 里先
`prepare_kv_writes`（COW），**per-slot 路径却没有**。于是单请求（c=1）的
长 prefill 落在别名页上时直接覆写 persistent scratch 的 KV 字节，且别名
refcount 不释放。250K COLD 实际把 128K 条目的 KV 内容覆盖成了自己的头部。

修复：per-slot forward 前按
`prepare_kv_writes(slot, hit + start, end - start)` 做同样的 COW 拆页
（池不存在时 no-op，与 `_prefill_forward` 的既有防御一致）。
回归：`test_per_slot_chunked_prefill_cow_detaches_aliased_page`（断言
scratch KV 字节逐位不变 + 别名被拆）。

### 6.2 live slot 的已提交别名页无法逐出（store 死锁）

即使 6.1 修好，page 对齐的部分命中（如 128K 条目 + 更长后缀）不会写
已别名页；prefill-commit store 时 live slot 仍 pin 着旧条目，而
`_evict_persistent_until` 的 detach 循环只处理 `slot_kv_len == 0` 的 idle
slot → 250K 条目（需 2000/2048 页）逐出 3 条后卡住 → `_store_persistent_prefix`
静默 return → WARM 全冷。

修复：`detach_scratch_aliases` 允许 live slot，但只拆 **`slot_kv_len`
已提交范围内**的别名页——已提交页对剩余请求只读，私有化不改变任何可观察
行为；边界外的页留给 `prepare_kv_writes` 首次写入时 COW。
回归：`test_detach_scratch_aliases_remaps_a_live_slots_committed_range_only`
（pool 层）与 `test_prefill_commit_store_evicts_entry_aliased_by_live_slot`
（backend 层，断言 evictions=2、store 成功）。

两个修复合并后的 CPU 五上下文复现：250K store 成功（5 stores / 4 evictions）、
WARM restore 完整、128K scratch KV 未被污染。

修复后的正式网格（即 §4）：`prefix_persistent_stores=5`、
`prefix_persistent_evictions=4`、`prefix_persistent_restores=55`——5 档
上下文各成功 store 1 条，250K 入池时逐出 4 条旧条目，WARM 波 15/15 全命中。

## 7. 相关

* [`2026-08-05-qwen36-quality-rerun.md`](2026-08-05-qwen36-quality-rerun.md)
  —— 质量无回退的完整证据（MMLU-Pro 84.54% 精确复现等）。
* [`2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md`](2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md)
  —— 完整命中路径的 GDN 状态修复（本网格的前置）。
* [`2026-08-03-cudagraph-vs-eager-decode-throughput.md`](2026-08-03-cudagraph-vs-eager-decode-throughput.md)
  / [`2026-08-03-mtp-serving-gpu-verification.md`](2026-08-03-mtp-serving-gpu-verification.md)
  —— 历史吞吐锚点。
* `benchmarks/server_perf_grid.py` —— 本网格 harness（逐 cell 续跑）。
* `benchmarks/fixtures/server_perf_grid_20260805.json` —— 修复前的失败版本
  （250K WARM 假冷），保留作对照。
