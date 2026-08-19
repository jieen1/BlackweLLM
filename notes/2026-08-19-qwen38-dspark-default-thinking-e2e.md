# Qwen3.8 DSpark 默认路径与 thinking budget 端到端验收

日期：2026-08-19
状态：🟢 有效
范围：`qwen36` runtime backend，Qwen3.8-27B-NVFP4，单张 RTX PRO 6000
Blackwell Max-Q（SM120）

## 结论

Qwen3.6/Qwen3.8 家族现在默认走 DSpark，不再默认走 MTP。清理所有相关
环境变量后的真实进程启动日志确认：

```text
capacity=4 num_slots=4 block_size=128 blocks_per_slot=2048
KV=fp8_e4m3, mode=elastic, pool=19629342720 bytes
dspark=True(K=7, verify=compact, require_cg=True)
mtp=False
```

native MTP 仍保留为显式回退路径：`--mtp` 或
`QSR_SERVER_ENABLE_MTP=1`（同时关闭 DSpark）；显式 `--dspark` 反向关闭
MTP。Qwen 默认额外采用 `prefill_chunk=8192`、admission coalesce 10 ms、
FlashInfer prefill/GDN、MLP FlashInfer + W4A4、batch 内 prefix dedup，且
保留 persistent prefix cache。

## 低层 thinking 限制修复

之前 compact ragged DSpark 在 budget boundary 把请求降级成逐 slot
`round()`，fresh slot 会触发 GDN `spec_forward` 前置条件，并且失去批处理。
修复在 `runtime/backends/qwen36_dspark.py` 的 batched verify 层完成：

1. boundary round 临时使用完整 K+1 verify 宽度，让受限位置仍在 graph-owned
   logits 中可见；
2. 只对目标 slot/位置的 logit 加约束 bias，然后用同一批次的 full draft
   重算 accept/reject；
3. 约束后的 host decision 作为下一轮 anchor/lens 的权威来源，必要时显式
   同步 target hidden；普通 DSpark round 仍走设备 accept epilogue；
4. 通过 `dspark_thinking_force_batched_replays` 计数器确认约束回放没有走
   per-slot fallback。

这是 runtime 层的边界修复，不改变 DSpark K=7 steady-state planner。

## 真实 API 两组测试

进程：真实 Qwen3.8-27B-NVFP4 权重 +
`RadixArk/Qwen3.8-27B-DSpark` draft，端口 `8301`，HF 离线；默认配置由
`server.app` 自动选择，没有传 `--dspark`，所有 `QSR_SERVER_*`/DSpark
默认覆盖变量先 `unset`。所有请求 HTTP 200，答案内容保持 `blue/Blue`。

| 模式 | trial 1 | trial 2 | trial 3 | completion tokens | thinking |
|---|---:|---:|---:|---:|---:|
| thinking 关闭 | 0.458725 s | 0.135051 s | 0.125625 s | 1 / 1 / 1 | 0 |
| `thinking_token_budget=32`, xhigh | 0.366897 s | 0.382824 s | 0.387041 s | 34 / 34 / 34 | 32 |

关闭 thinking 的首个请求包含冷/槽初始化影响；warm 两次均值为
`0.130338 s`。限制模式均值为 `0.378921 s`，比 warm 关闭模式多
`0.248583 s`，这正是刻意生成 32 个 thinking-boundary token 的成本，不能
归因成 DSpark 性能回退。限制模式的 34 个 completion token 中包含 31 个
普通 reasoning token、`</think>` 边界 token，以及答案/结束部分 2 个 token。

预算测试后统计确认 `dspark_thinking_force_batched_replays=11`，DSpark
verify/draft graph 持续 replay，`mtp_verify_graph_replays=0`；因此限制模式
实际经过了批量 DSpark 路径。

另外用缓存的真实 `unsloth/Qwen3.6-27B-NVFP4` 权重启动了第二个清理进程，
没有传 `--dspark` 或相关环境变量，同样得到 `dspark=True(K=7)`，四个 draft
graph、ragged verify graph 和 target graph 全部 captured。关闭 thinking 的
请求返回 `BLUE`、completion 1 token；`thinking_token_budget=32` 请求返回
`BLUE`、completion 34 token。该进程的 stats 为
`dspark_thinking_force_batched_replays=3`、`dspark_verify_graph_replays=16`、
`dspark_draft_graph_replays=18`、`mtp_verify_graph_replays=0`。

## 4×131K 性能复测

使用现有 `scripts/run_qwen38_128k_decode_bench.sh c4`，每个请求
`prompt=131072`、`max_tokens=256`，四并发，block size 128，冷一轮、warm
两轮。清理后的全新 server 进程结果：

| | cold | warm 1 | warm 2 |
|---|---:|---:|---:|
| 本次默认 DSpark | 40.6841 s / 25.17 agg tok/s | 443.12 agg tok/s | 444.12 agg tok/s |
| 历史 DSpark 基线 | 40.7986 s | 443.74 agg tok/s | 450.88 agg tok/s |

本次冷启动快 `0.1145 s`（约 `0.28%`）；两轮 warm 均在既有主机抖动范围
内，均值 `443.62` 对历史均值 `447.31`，差 `0.83%`，没有可归因于本次
默认切换的性能回退信号。四路请求均成功，prefix hit/restore 为 cold
`3/3`、warm `4/4`，四路 completion SHA 全部为：

```text
75b43a8a0ae256dca5668dd5e73028d24f8700d46b7d5623526bc29711dff306
```

原始产物保存在：
`/tmp/qwen38_dspark_default_clean_20260819/`。

## 验证清单

- 目标回归：`66 passed`（DSpark、MTP engine、thinking budget、server defaults）。
- venv 全量：`2525 passed, 16 skipped, 21 warnings`。
- CPU-only 系统解释器：`1423 passed, 206 skipped`；唯一未收集的
  `tests/test_server_perf_grid_observability.py` 依赖本机没有安装的
  `aiohttp`，完整 venv 套件已覆盖该测试。
- 目标文件 ruff：通过；`compileall runtime server loader model bfdiag bfprobe`：通过。
- 全仓 ruff 仍被工作区已有的 `.tmp_qwen38_vllm_tap_server.py`、
  `sitecustomize.py`、`runtime/kernels/nvfp4_decode_attn.py` 与
  `runtime/preflight.py` 的 12 个既有错误阻断；本次改动未触碰这些文件。

性能 JSON、server trace/stats 与本记录均使用同一轮真实进程，后续做默认
参数 A/B 时应继续沿用该 workload，并先确认 `mtp_verify_graph_replays=0`
和 completion SHA。
