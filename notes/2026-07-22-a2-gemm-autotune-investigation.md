# A2: NVFP4 GEMM Autotune 调查报告（2026-07-22）

> 路线图 A2 前置调查：用真实数据评估 NVFP4 GEMM 的优化空间。

## 测试条件

- GPU: RTX PRO 6000 Blackwell (SM120, 96 GB, 188 SMs)
- 模型: unsloth/Qwen3.6-27B-NVFP4
- Decode: M=4 (c=1, K=3 MTP verify), eager mode
- 基准: CUTLASS sm120 NVFP4 GEMM (vLLM 0.25.1.dev0 内置)
- 峰值 memcpy BW: 789 GB/s

## Decode GEMM Shape 实测（a2_gemm_shape_profile.py）

| Shape (M×N×K) | 层 | 次/round | GFLOP/round | 占比 |
|---|---|---:|---:|---:|
| 4×34816×5120 | gate_up_proj | 56 | 79.9 | 46% |
| 4×5120×17408 | down_proj | 56 | 39.9 | 23% |
| 4×17408×17408 | down_proj_attn | 8 | 19.4 | 11% |
| 4×6144×6144 | out_proj | 64 | 19.3 | 11% |
| 4×5120×5120 | in_proj_qkvz | 72 | 15.1 | 9% |
| 4×96×5120 | in_proj_ba | 48 | 0.2 | 0% |

## 微基准实测（a2_gemm_microbench.py）

| Shape | CUTLASS ms | TFLOPS | BW GB/s | bf16 ms | NVFP4 vs bf16 |
|---|---:|---:|---:|---:|---:|
| 4×34816×5120 gate_up | 0.1291 | 9.56 | 778.9 | 0.2508 | **1.68×** |
| 4×5120×17408 down | 0.0797 | 8.40 | 630.0 | 0.1186 | **1.40×** |
| 4×17408×17408 down_attn | 0.2541 | 10.95 | 671.5 | 0.4222 | **1.91×** |
| 4×6144×6144 out_proj | 0.0298 | 10.12 | 634.8 | 0.0241 | **0.81×** |
| 4×5120×5120 in_proj | 0.0385 | 5.45 | 341.8 | 0.0174 | **0.45×** |
| 4×96×5120 in_proj_ba | 0.0329 | 0.12 | 7.8 | 0.0236 | **0.72×** |

**关键发现：CUTLASS NVFP4 在 3 个小 shape 上比 bf16 更慢**（out_proj 0.81×, in_proj 0.45×, in_proj_ba 0.72×）。

## 带宽利用率分析

| Shape | 权重 MB | 总读取 MB | 实测 BW | % peak | 优化空间 |
|---|---:|---:|---:|---:|---:|
| gate_up_proj | 89.1 | 100.6 | 778.9 GB/s | **98.7%** | 1.3% |
| down_proj | 44.6 | 50.2 | 630.0 GB/s | **79.8%** | 20.2% |
| down_proj_attn | 151.5 | 170.6 | 671.5 GB/s | **85.1%** | 14.9% |

## Split-K 实验（Python 层）

| Shape | 1-split | 2-split | 4-split | 8-split | 最优 |
|---|---:|---:|---:|---:|---|
| gate_up | 0.1291 | 0.1495 | 0.1730 | 0.2406 | 1-split |
| down_proj | 0.0797 | 0.0897 | 0.1232 | 0.1821 | 1-split |
| down_attn | 0.2541 | 0.3062 | 0.2965 | 0.4027 | 1-split |

**结论：Python 层 Split-K 完全无效**——多次 kernel launch + reduction 开销远超 SM 利用率收益。

## 理论极限计算

- 加权基准: 13.726 ms/round
- 理论极限（100% peak BW）: 12.431 ms/round
- **最大可能节省: 1.294 ms (9.4%)**
- 每轮 decode 时间 (4K c=1): ~27 ms
- **最大 e2e 提速: 4.8%**

## 裁决

1. **gate_up_proj（46% GEMM 占比）已达 98.7% 峰值带宽——无优化空间。**
2. **down_proj 有 20% 空间**（SM 利用率低：N=5120 仅产生 40 个 128×128 tile，188 SM 中 92 个空闲），但需要修改 CUTLASS tile 配置（128×64×128），需重建 vLLM C 扩展。
3. **小 shape（out_proj, in_proj）CUTLASS NVFP4 比 bf16 更慢**——128×128 tile 对 M=4, N≤6144 严重浪费。但这些 shape 的绝对时间很小（合计 ~5 ms/round），且它们走的是 FP8 GEMM 路径而非 NVFP4。
4. **A2 autotune 的实际杠杆远小于路线图预期**（路线图基于"GEMM 占 71%"推断，但未考虑 CUTLASS 已接近带宽极限）。
5. **建议：A2 降级为"down_proj tile 优化"单项**（预期 ~1 ms/round = 3.7% e2e），不再作为"贯穿全月主线"。主线应转向 B7-V1（去 vLLM 化）和 L0（Laguna 冒烟），这些是架构层面的真正杠杆。

## 工具链

- `benchmarks/a2_gemm_shape_profile.py` — hook 进 linear 层采集实际 M/N/K
- `benchmarks/a2_gemm_microbench.py` — 精确 shape 的 CUTLASS vs bf16 计时
- `benchmarks/a2_gemm_shape_survey.py` — torch.profiler 按 shape 聚合
