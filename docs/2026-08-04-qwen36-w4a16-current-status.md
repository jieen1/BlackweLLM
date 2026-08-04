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

只评估保持 packed W4A16 数值契约的低 shared-memory / 更高驻留度候选，先做单
GPU、单并发、同一 warm-engine 配置的 correctness 和 Nsight 对照，再决定是否
进入生产。Laguna-X-2.1 路径不变。
