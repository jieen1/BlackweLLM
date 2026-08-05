# 128K 解码性能回退的主机侧剖析（MTP K=3 + CG + prefix cache）

日期：2026-08-05 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（snapshot
`ccdaab7e68af2409599b8949a8f2685703c9bae5`）· 单卡 RTX PRO 6000 Blackwell
Max-Q（SM120，96 GB）· 服务：本仓库自有 runtime（`server.app` /
`qwen36` 后端），无任何外部推理引擎。

## 一句话

历史 README 的 **128K/c4 222 tok/s、64K/c4 267 tok/s（warm 236.69）** 出自
`docs/archive/` 里的 DirectModelRunner 直连协议（无 HTTP、精确重复前缀 +
10240 新后缀、warm=精确重复）。当前生产服务同协议下 128K 端到端只有
**~105–112 tok/s**（约 0.5×），64K 只有 **146–148 tok/s**（约 0.62×）。
nsys + 阶段计时把差距定位在**主机侧**，不是 GPU 内核：每轮 ~127 ms 里 GPU
内核只占 ~52.6 ms，其余是 accept 等待、checkpoint 克隆、ragged 同步与
draft 续跑的多次 D2H 回传。优化路径已列出（§6），每步都有确定性收益。

## 1. 历史数字的真实出处（先对齐口径，不再怀疑数字）

| 出处 | 数字 | 协议 |
|---|---|---|
| `docs/archive/2026-07-30-architecture-two-tenant.md:439-444` | 128K/c4 **222 tok/s**；64K/c4 **267 tok/s**；decode attn 0.988 ms（1.56× FlashInfer） | KWIDE + V272 内核 |
| `docs/archive/2026-07-20-PROGRESS.md:4185-4244` | 128K/c4 warm **222.44**；64K/c4 warm **236.69** | warm=精确重复前缀；267 是早期回声，置信度低 |
| `docs/roadmap.md:446` | README 222/267 已标注为旧数字 | 文档整理时留下的注记 |

历史 128K 协议 = **4 请求 × 131072 token、max_tokens=256、无 HTTP 的
DirectModelRunner、warm=精确重复前缀命中**。昨天提交 `46d5652` 只对齐了
**4K W1-S（227.5 committed tok/s）**，128K/64K **从未对齐过**——README 的
222/267 不是当前 build 跑出来的。

## 2. 当前生产服务数字（同口径端到端）

服务配置与历史 best profile 完全一致（3×256K、MTP K=3、decode+MTP 四张
CUDA Graph、persistent prefix cache、FP8 e4m3 KV、gpu_mem_util 0.92、
request timeout 0）。启动环境：`/tmp/qwen36_server_env.sh`
（`QSR_PROFILE_ROUNDS=1` 只开阶段计时，不改任何数值路径）。

历史协议 harness（raw tokenization、无 chat template、warm 波 = 已缓存前缀
+ 10240 fresh 后缀）：

```bash
/home/bot/.venvs/vllm/bin/python -u benchmarks/server_perf_grid.py \
  --base-url http://127.0.0.1:8300 --model qwen3.6 \
  --endpoint completions --contexts 64k,128k,200k \
  --concurrency 1 --max-tokens 256 --warm-rounds 3 \
  --warm-suffix-tokens 10240
```

已落盘的 fixture（raw 数据逐 wave 保留）：

| fixture | 内容 | 关键数字 |
|---|---|---|
| `server_perf_grid_histproto_c1_20260805.json` | 64K/128K/200K × c1 | 128K warm ~58 tok/s（server e2e），64K warm ~77 tok/s |
| `server_perf_grid_histproto_c4_20260805.json` | 64K/128K × c4（3×256K 容量只装得下 3 个实际并发） | 128K c4 v1 失败波，后由 v2 重测 |
| `server_perf_grid_histproto_c4_20260805_v2.json` | 64K/128K × c4 | 128K warm 30.05 tok/s/request（聚合 ~120）；64K warm 37.12 tok/s/request（聚合 ~146） |

注意：`aggregate_e2e_tok_per_s` 是**每请求**端到端速率（总生成 token /
并发数 / wall），聚合吞吐要 × 并发数。后续对比统一用**聚合端到端 tok/s**。

profile 服务（PID 3436258，日志
`logs/quality/server_profile_qwen36_20260805.log`）在缓存热透后跑了
128K×3 warm：前端 wall **~3.44 s/轮**、`mtp_round` 中位 **83.5 ms**、
引擎 step 中位 **~84 ms**，单轮平均提交 ~2.5 token → 聚合端到端
**~105–112 tok/s**，为历史 222 的 **~0.5×**。

## 3. nsys 归因（128K c3 warm，`/tmp/qwen128k_c3_warm.sqlite`）

每轮 ~127 ms 的分解：

* **GPU kernel 合计 ~52.6 ms**：verify graph 42.1 ms + sync graph 2.9 ms +
  draft graph 5.4 ms + eager 1.7 ms——GPU 侧不是主瓶颈。
* **`cudaMemcpyAsync` 8877 次**，单次最久 27 ms，且全是 ≤256 B 的 D2H
  （`first_by_slot = .tolist()`、`_step_tokens` 回传等）。
* **`cudaStreamSynchronize` 320 次共 3.5 s**——主机侧等待是主要差距。

## 4. 生产阶段计时（`QSR_PROFILE_ROUNDS=1`，63 轮 mtp_round_b3）

| 阶段 | 中位 | 均值 | p90 | 最大 | 说明 |
|---|---:|---:|---:|---:|---|
| setup | 0.0 ms | 0.0 ms | 0.0 ms | 0.1 ms | 每轮固定簿记 |
| verify_replay | 2.9 ms | 3.6 ms | 4.8 ms | 25.3 ms | verify CG replay（捕获开销，GPU 快） |
| compute_logits | 0.2 ms | 0.3 ms | 0.5 ms | 0.6 ms | logits 计算 |
| **accept_decision** | **43.1 ms** | 52.8 ms | 81.2 ms | 105.1 ms | 等 verify 图 GPU 完成 + 决策主机逻辑 |
| **commit_loop** | 0.2 ms | 7.4 ms | 28.2 ms | 32.8 ms | 中位极低，但 23/63 轮 >5 ms = checkpoint 克隆尖峰（约每 4 轮一次） |
| **sync_ragged** | 7.7 ms | 13.3 ms | 26.3 ms | 47.7 ms | ragged 同步含 D2H `.tolist()` |
| **draft_batch** | 6.7 ms | 12.6 ms | 27.3 ms | 45.3 ms | draft 续跑含 seed token 主机中转 |
| total | 83.5 ms | 90.1 ms | 123.9 ms | 151.6 ms | 单轮合计（前端 wall ~114 ms） |
| engine_step | 83.7 ms | 90.1 ms | 124.3 ms | 151.8 ms | 引擎侧同一轮 |
| bookkeep | ~0.0 ms | 6.9 ms | 0.1 ms | 250.4 ms | 引擎簿记（尖峰与 GC/调度相关） |

结论：**128K 的差距主要是主机侧串行等待**，不是 attention/MoE 内核。
`accept_decision` 的 42.9 ms 是 verify 图 GPU 等待（固有成本），而
checkpoint 克隆尖峰 + 两次 D2H 回传 + draft seed 中转是可优化的确定性开销。

## 5. 质量侧影响

本轮只做剖析，未改任何数值路径。`QSR_PROFILE_ROUNDS=1` 只插入
`time.perf_counter()` 计时，不改变采样、不接受、不改变图形捕获参数；
profile 服务与生产服务加载的是同一份 checkpoint、同一组环境变量（除
profile 开关外逐字节一致）。因此剖析结论可外推回无 profile 的生产 build。

## 6. 优化计划（按确定性收益排序）

1. **checkpoint 克隆改单次持久 buffer 拷贝**（杀 23/63 轮的 30 ms 尖峰）：
   `runtime/model/qwen36_slots.py:858` `capture_recurrent_state` 现在逐
   tensor `.clone()`（16 层 GDN × 2 tensor）。改为 `torch.cat` 全部 GDN
   状态到一块持久复用的大 buffer + 每槽固定偏移，一次性 D2D copy；
   `restore_recurrent_state` 用 `torch._foreach_copy_` 语义保留。需要保持
   返回 list 的调用方兼容（`tests/test_qwen36_slot_pool.py:329/339/356`）。
   预期：每 4 轮省 ~30 ms → 128K c3 每轮 ~8 ms 收益。
2. **sync→draft seed token 走 device 直连**（杀一次 D2H ~7 ms）：
   `runtime/backends/qwen36_mtp.py` 现在 `first_by_slot = .tolist()` 后
   由 `_continue_draft_batch` 写 host 再上传。让 replay 路径直接返回
   device 的 `_step_tokens`，`_continue_draft_batch` 直接 D2D 拷贝，
   保留 dict/list 兼容层给 tests/benchmarks。
3. **verify fill 的 COW 快速路径**：`Qwen36MTPVerifyCudaGraph._fill` 每轮
   给每槽调 `prepare_kv_writes(slot, past_len, qo_len)`；若 passthrough
   （无共享前缀）直接跳过，避免重复 COW 检测。
4. **谨慎、需质量锚点**：`accept_decision` 的 43 ms 是 verify 图 GPU 等待；
   历史 128K 用 split-KV=32 + KWIDE 达 0.988 ms/层，当前 decode attention
   是 `split_kv=False`。改 split 会改变数值路径 → 必须单独过质量锚点，
   不在本轮顺手改。

每完成一步：重启服务（`setsid` + 环境变量文件）→ 跑 128K c3 warm 一轮
验证 → 记录阶段分布与端到端数字 → 更新本 note 与 fixture → 提交。

## 7. 已落地优化（2026-08-06 凌晨，提交 `94e4f59` 之后）

### 7.1 `_prefix_hash` 用 `array('I')` 打包（已实现，待实测）

原实现逐 token `buf += int(tok).to_bytes(4, ...)`，128K 时实测 **8.3 ms**；
改 `array("I", token_ids[:length])` + 一次 `tobytes()` 后 **1.1 ms**（7.5×）。
这个调用在 decode 热路径上每 block 边界 checkpoint 一次（128K 下约每 2 轮），
是 commit_loop 20-30 ms 尖峰的主要可确认成分（克隆本身实测只有 0.42 ms，
不是尖峰来源——与交接时的假设不同，已用微基准排除）。

### 7.2 sync→draft seed token 改 device 直连（已实现，待实测）

`_sync_real_suffix_batch_ragged` 不再对 `first_drafts` 调 `.tolist()`：
返回 device 行，`_continue_draft_batch` 把 device seed 直接 D2D 拷进 draft
图自己的 `input_ids`（`Qwen36MTPDraftCudaGraph._fill` 新增 tensor 分支），
host int 转换推迟到 draft replay 之后（此时流已同步，`.tolist()` 不再引入
新的阻塞点）。每轮省掉一次 mid-round D2H + H2D 往返（实测 5-15 ms 级主机
空档，nsys 里表现为 20-80 ms 的 GPU 空闲尖峰）。

兼容性：`replay_batch`/`_continue_draft_batch` 同时接受 `list[int]` 与
device tensor；eager 与旧 stub 路径不受影响。新增
`test_ragged_sync_hands_device_seeds_to_the_draft_graph_without_tolist`
（引擎层）与 `test_draft_fill_stages_device_seeds_with_a_direct_copy`
（源码纪律层）两个回归测试；相关 125 个单测通过，ruff 通过。

### 7.3 明确的非目标（本轮不做的）

* checkpoint 克隆 arena：实测 32 次 `.clone()` 只要 0.42 ms，不是 30 ms
  尖峰来源，不做（避免改变 `capture_recurrent_state` 返回语义的风险）。
* `accept_decision` 的 43 ms 是 verify 图 GPU 执行（42 ms），要动只能改
  split-KV/KWIDE 数值路径，质量锚点单独验证，不顺手改。
* verify fill 的 `prepare_kv_writes` 每轮每槽只覆盖 1 个逻辑页，
  passthrough 时约 0.1-0.3 ms，不是瓶颈。

## 8. 相关

* `runtime/round_profile.py` —— env 门控的阶段计时器（本轮新增）。
* `runtime/backends/qwen36_mtp.py` / `server/engine.py` —— 阶段计时埋点。
* `benchmarks/server_perf_grid.py` —— 历史协议 harness（`--endpoint
  completions --warm-suffix-tokens`）。
* `logs/quality/server_profile_qwen36_20260805.log` —— 63 轮原始阶段 JSON。
* `/tmp/qwen128k_c3_warm.sqlite` —— nsys 原始 trace（30 MB）。
* [`2026-08-05-server-perf-grid-mtp-cg-prefix.md`](2026-08-05-server-perf-grid-mtp-cg-prefix.md)
  —— 同一服务的完整网格（含 prefix cache 命中修复）。
