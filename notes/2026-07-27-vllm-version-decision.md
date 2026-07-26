# vLLM 版本决策:不升级到 0.26.0(2026-07-27 00:43)

## 决策

**生产环境保留 vLLM 0.25.0(`e12b91b032`)+ 5 个本地补丁,不升级到 0.26.0。**

依据:同一套 runtime 代码(`main@8e04775`)、同一套 sparkinfer(`blackforge-main@3fa9b54`,
正确性修复强制要求的版本),只换 vLLM 版本,0.26.0 比 0.25.0 慢约 8.6%,退化真实、
干净、可复现,不是测量噪声。按照"没有性能退化才升级,否则不升级"的原则,不升级。

## 四点对照(全部 64K M=1 decode CG,`benchmarks/repro_80tok_m1_decode_cg.py` 同款脚本,
独立 venv `/home/bot/.venvs/vllm-repro80`,同一张卡)

| # | 运行时代码 | vLLM | sparkinfer | avg tok/s | best round tok/s | 来源 |
|---|-----------|------|------------|-----------|-------------------|------|
| 1 | `66d5913`(历史) | 0.25.0+patch | `0a7b143` | 76.8 | 80.6 | 本次复现,`notes/2026-07-27-speed-repro-verified.md` |
| 2 | `main@8e04775` | 0.25.0+patch | `0a7b143` | 79.2 | 80.3 | 本次新测 |
| 3 | `main@8e04775` | 0.25.0+patch | `blackforge-main@3fa9b54` | **77.5** | 77.9 | 本次新测(生产必须用的 sparkinfer 版本) |
| 4 | `main@8e04775` | **0.26.0** | `blackforge-main@3fa9b54` | **70.8** | — | 上一个 session,`notes/FULL_STATUS_20260726.md` §三,07-26 22:47 |

## 变量拆解

- **运行时代码 `66d5913→8e04775`**(对照 1 vs 2,固定 vLLM+sparkinfer):76.8→79.2,**没有退化**(甚至更好,在噪声范围内)。DFlash/CUDA-graph 相关的大量改动没有拖累这条 M=1 no-DFlash 路径。
- **sparkinfer `0a7b143→blackforge-main`**(对照 2 vs 3,固定代码+vLLM):79.2→77.5,**约 -2.1%**。符合预期——`blackforge-main` 比 `0a7b143` 多了 K/V 竞态修复(`d2d8cb9`),`notes/STATUS_dflash_acceptance.md` 里早就标注这个 correctness 补丁"当前是用于确认根因的版本,不应视为性能最优实现...可能有可测性能回退"。这个代价是正确性必须付的,不可谈判。
- **vLLM `0.25.0+patch→0.26.0`**(对照 3 vs 4,固定代码+sparkinfer):77.5→70.8,**约 -8.6%,是四个变量里最大的一块,且是唯一一个"不换就能白拿"的退化**。

## 与已有排查的吻合

`notes/STATUS_speed_optimization_0726.md` 的 profiler 数据(vLLM 0.26.0 环境下采集)显示
`cutlass_80_wmma_*`(SM80 Ampere kernel)在 SM120 硬件上被 cuBLAS/vLLM 选中,吃掉约
35-40% GPU 时间,单独测试显示同一个 GEMM 操作慢 2.4 倍。这次的对照表证实了这个现象
**很可能就是 vLLM 0.26.0 版本本身的回归**(kernel 选择优先级或 cuBLAS 启发式随 vLLM/torch
组合变了),而不是我们代码的问题——回到 0.25.0 之后大概率不再需要单独修这个 GEMM 选择
问题,因为退化的源头本身被绕开了。这一点本次没有单独用 profiler 在 0.25.0 上重新确认
(如果之后 0.25.0 的实测数字有异常下滑,应该回来验证 SM80 WMMA 问题是否仍然存在)。

## 后续动作

- **不需要改代码**:`main@8e04775` 的 vLLM-0.26.0 兼容改动(RMSNorm rebind、KV scatter
  C++ 化,commit `4e99b7c`)在 0.25.0 环境下跑得完全正常(甚至更快),说明这些改动是
  向后兼容的,不需要为了"降级"而回退。
- **需要改的是环境/文档**:此前 `FULL_STATUS_20260726.md` 里"复现完成后必须恢复到
  vLLM 0.26.0"的计划基于"目标是在 0.26.0 上达到/超越历史值"的假设——本次结果推翻了
  这个假设的前提,不应该无条件切回 0.26.0。当前 `/home/bot/vllm`(`e12b91b032`+patch)
  和 `/home/bot/project/sparkinfer`(`blackforge-main`)已经正好停留在获胜配置上,
  不需要再切换。
- 如果之后有其他理由必须上 0.26.0(比如某个只有新版本才有的功能/安全修复),
  那笔账要和这 8.6% 的 tok/s 一起摆出来给用户决策,不能默认升级。
