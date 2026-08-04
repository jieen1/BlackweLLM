# Qwen3.6 W4A16 当前状态（2026-08-04）

这份记录是当前性能调查的结论摘要；原始 profiling、命令和历史对照见
[`notes/2026-08-04-qwen36-w4a16-profile-and-split-plan.md`](../notes/2026-08-04-qwen36-w4a16-profile-and-split-plan.md)。

## 已确认事实

- Unsloth Qwen3.6-27B-NVFP4 使用 `weight_layout="packed"`，实际热点是
  SparkInfer 的 `W4A16FusedMoeKernel` packed TC-decode 路径（M=4），不是
  modelopt micro 路径。
- 当前生产路径保持原始量化权重和 BF16 激活；没有加入 BF16 权重反量化缓存，
  也没有把 W4A4 作为“优化”方案。W4A4 的单层误差和完整质量门禁均不接受。
- 2026-08-04 的 Nsight Compute 实测：动态 shared memory 54.27 KiB/CTA，
  每 SM 只能驻留 1 个 block，理论/实测 occupancy 16.67%，active
  warps/scheduler 1.97，no eligible warp 54.30%，DRAM 41.33%，SM
  compute 38.39%。因此当前主要问题是 packed W4A16 的 shared-memory/驻留度
  瓶颈，而不是已经证实的 Python 或 CUDA Graph 调度开销。
- 之前的 modelopt split 实验没有命中这个 checkpoint 的 packed 路径；其跨进程
  时间差不可比，不能作为性能结论。该修正已写入原始调查记录。

## 下一道性能门禁

~~只评估保持 packed W4A16 数值契约的低 shared-memory / 更高驻留度候选~~
**已评估，2026-08-04 结案：occupancy 不是主导杠杆（负结果，有完整实测）。**
放开 1-block/SM pin 的 opt-in 候选做到了 3 blocks/SM、单层 kernel 快 18%，
但全模型 W1-S 门禁下 verify 图相位反而慢 6%——fused 常驻轮的 grid-barrier
参与者翻倍与 tile_k 减半把隔离环境的收益全部吃掉。原始记录（含 ncu 表、
oracle、A/B 数字与复现命令）见
[`notes/2026-08-04-qwen36-w4a16-profile-and-split-plan.md`](../notes/2026-08-04-qwen36-w4a16-profile-and-split-plan.md)
末节；候选分支 `qwen-w4a16-occupancy-20260804` 保留不合并。

**同日后续进展（2026-08-04 晚）**：按用户裁定"质量基准=历史方案"，全 M W4A4
（单一 kernel 家族、单一权重驻留，历史布局）已接线并两次重复验证：**e2e 墙钟
50.2/51.8 s vs 基线典型 58.7/60.2 s（约 +14%），prefill -32%**；接受率 71.2%
（历史锚 70.29%）、committed 4115（历史锚 4116）。随后接上自有 W8A8 FP8 dense
kernel（历史 CUTLASS 移植）组成完整历史 kernel 组合：**33.3/33.2 s（1.77×）、
接受率 72.3%、committed 4120，双双达到历史质量锚**。该组合模式现已**改为生产
默认**（`QSR_QWEN36_MLP_W4A4` / `QSR_QWEN36_MLP_W4A4_ALL` /
`QSR_NATIVE_W8A8_FP8_CHANNEL` 默认开，`=0` 为诊断回退；自有 `.so` 缺失时自动
降级回 torch._scaled_mm）。证据与负结果链：
[`notes/2026-08-04-w4a4-sparkinfer-headtohead.md`](../notes/2026-08-04-w4a4-sparkinfer-headtohead.md)、
[`notes/2026-08-04-w1s-same-caliber-gap-decomposition.md`](../notes/2026-08-04-w1s-same-caliber-gap-decomposition.md)。
剩余差距（33.3 s vs 历史 ~20.4 s）几乎全在 prefill（16.2 vs ~5 s），decode 引擎
已在历史 10% 以内；prefill 侧的配置级杠杆已全部测完关闭，剩余是 kernel 开发
（per-row-alpha FP8 epilogue、累加序精确的 norm 融合）。Laguna-X-2.1 路径不变。
