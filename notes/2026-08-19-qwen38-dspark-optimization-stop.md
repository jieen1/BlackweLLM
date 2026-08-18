# Qwen3.8 DSpark：同口径 profiling 收敛与优化暂停（2026-08-19）

状态：🟢 **本轮排查完成；未找到足以进入默认路径的明确净收益改动，暂停继续试错。**

## 固定比较口径

所有端到端结果使用同一份 4 请求 fixture：

- `unsloth/Qwen3.8-27B-NVFP4`，DSpark `K=7`；四并发；每个 prompt 恰好
  `131072` tokens；`max_tokens=256`。
- 同 tokenizer、同 filler prompt、同请求参数、同 completion 长度、同
  `block_size=128`。
- CUDA Graph、prefix cache、FP8 KV、ragged verify、`prefill_chunk=8192`、
  admission coalesce `10 ms` 均开启；local 使用 4 slots / 256K logical
  max / dynamic pool。
- 12 个 completion 的 SHA 均为
  `75b43a8a0ae256dca5668dd5e73028d24f8700d46b7d5623526bc29711dff306`。
- SGLang 原始结果已保存在
  `benchmarks/fixtures/sglang_dspark_same_128k_c4_20260818.json`，精确环境在
  `benchmarks/fixtures/qwen38_live_publish_c4_20260819.env`。

## 已确认的同口径结果

SGLang steady warm 为 `1024 / 2.609 = 392.49 tok/s`。local 当前已提交基线
`live_publish` 的两次 warm wave 为 `443.74 / 450.88 tok/s`，即分别高出约
`13.1% / 14.9%`；acceptance 为 `904 / 1020`，136 个 DSpark rounds。

本轮所有候选都保持 completion SHA 不变，结果如下：

| 候选 | warm 0 / warm 1 tok/s | acceptance | 判定 |
|---|---:|---:|---|
| local 基线 `live_publish` | 443.74 / 450.88 | 904 / 1020 | 当前基线 |
| graph-fused context KV | 433.46 / 410.67 | 902 / 1020、868 / 1020 | 回退，稳定变慢 |
| ragged graph tier probe | 442.28 / 441.27 | 904 / 1020 | 无收益，回退 |
| ragged tier + confidence 0.90 | 415.27 / 421.77 | 888 / 1020 | 接受率与吞吐均变差，回退 |
| RMS/FP8 fusion | 441.61 / 441.29 | 904 / 1020 | 相对 control 无净收益，回退 |

显式设置 `QSR_QWEN36_VERIFY_ATTN_BACKEND=flashinfer` 的 A/B 得到
`452.64 / 454.78 tok/s`，但 `auto` 默认已经选择同一 FlashInfer kernel
family；该差异没有对应的源码变化，视为运行噪声/调度差异，不能计作优化收益。

## Profiling 结论

local steady profile 共解析 102 个 `dspark_ragged_round_b4` 记录，中位数为：

| phase | 中位耗时 |
|---|---:|
| round total | 53.58 ms |
| verify fill | 0.55 ms（少量 12–14 ms outlier） |
| accept decision / GPU wait | 49.04 ms |
| `target_hidden_sync` | 3.24 ms |
| draft batch | 0.44 ms |
| graph launch | 0.04 ms |

SGLang 源码侧的 static target verify 同样是 FlashInfer paged causal
prefill + CUDA Graph；local 并不是因为用了完全不同的 attention 入口而慢。
local Nsight 的 target verify 小 kernel 约 `0.16–0.21 ms`，SGLang static
trace 的对应 MaskMode1 kernel 约 `0.223 ms`。这两份 trace 不是完全相同的
4-request shape，因此只用于排除“local FlashInfer kernel 本身明显更慢”，
不能单独作为端到端收益证据。

已验证的 graph-fused context-KV 候选没有消掉约 `3.24 ms` 的 sync 成本，反而
在端到端上下降；tier 方案也没有改善实际 K=7 workload。故当前没有一个可以
安全进入默认路径、同时具备明确净收益和完整同口径验证的改动。

## 代码与证据处理

- 本轮未把未经证实的 ragged tier 实验代码留在工作树；默认 runtime 行为与
  已验证基线一致。
- 新增的端到端、stats、trace fixture 均保留在
  `benchmarks/fixtures/`，包括 FlashInfer A/B、tier、RMS/FP8、fused-context
  和 profiling companion 数据。
- 原始 Nsight SQLite 与 SGLang torch profiler 压缩 trace 仍保留在本机 `/tmp`
  路径；它们体积较大，不纳入 Git。路径见本轮提交说明和既有 live-prefix note。
- 后续若继续优化，应从小 M W8A8/W4A4 GEMM 的 graph-node kernel-family
  归因开始，并重新使用这份 4×131072 prompt fixture；在此之前不再凭 phase
  名称或单次 warm 数字改代码。
