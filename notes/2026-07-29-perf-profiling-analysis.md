# 性能 Profiling 分析 (2026-07-29)

## DFlash 一轮时间分布 (64K context, M=16, 15/15 accepted)

| 阶段 | 时间 | 占比 |
|------|------|------|
| Verify M=16 (主模型 48 层) | 38.5ms | 87% |
| Draft CG | 3.6ms | 8% |
| Context KV update | 1.4ms | 3% |
| Accept/reject | 0.8ms | 2% |
| **Round total** | **44.2ms** | **361.8 tok/s** |

## Verify 内部 kernel 分布

| Kernel | 每轮 | 占比 | 来源 |
|--------|------|------|------|
| sparkinfer MoE (shared+route) | 23.4ms | 59% | 47 MoE layers |
| sparkinfer paged attention | 6.5ms | 16% | SWA=512, 48 layers |
| Dense GEMM (cutlass) | ~6ms | 15% | QKV/O proj, 已用 custom GEMM |
| Norm/RoPE/gating/scatter | ~4ms | 10% | fused kernels |

## 模型架构

- 48 layers: 1 dense (layer 0) + 47 MoE (layers 1-47)
- MoE: 256 experts, top_k=10, intermediate=1024/expert, shared=1024
- Dense GEMM shapes: qkv=[11264,3072], o=[3072,9216]
- Custom NVFP4 GEMM: patched=True, autotune for M=4 (M=16 未调优)

## MoE 分析

- M=16 verify: routed_rows=160, tile_m=16, tile_n=128
- max_active_clusters=188 (= SM count)
- 160 expert-tiles / 188 SMs → GPU 利用率不足
- deterministic_output=True (ROUTE_BUFFER_TOPK_SUM)
- tiny_decode kernel 只支持 M<=4，不覆盖 M=16

## 优化方向 (按 ROI 排序)

1. **MoE deterministic 开销测试** — 重启 daemon 用 ATOMIC_SCATTER 对比
2. **Dense GEMM M=16 autotune** — 当前 tile 128×128 对 M=16 有 87.5% padding
3. **Context KV update 向量化** — Python loop → torch 向量化
4. **MoE tile 调优** — SPARKINFER_DYNAMIC_TILE_MN 环境变量

## 对比基线

| 指标 | 我们 | vLLM |
|------|------|------|
| tok/s @64K | 361.8 | 383.3 |
| 差距 | -6% | — |
| 接受率 | 88.1% | 100% |

## 已保存的 profiling 数据

- `benchmarks/fixtures/profile_verify_kernels_20260729.json` — kernel 分布
- `benchmarks/fixtures/acceptance_regression_20260729.json` — 接受率基准
- `benchmarks/profile_dflash_round_v2.py` — round profile 脚本
- `benchmarks/profile_verify_kernels.py` — kernel profile 脚本
- `benchmarks/acceptance_regression.py` — 接受率回归脚本

## 优化实施记录 (2026-07-29)

### 已完成

1. **向量化 `_bulk_precompute_context_kv` slot_mapping 计算**
   - 文件: `runtime/backends/laguna_dflash.py`
   - 改动: Python for 循环 (64K 次迭代) → torch 向量化
   - 效果: Draft KV precompute 173.6ms → 6ms (**29x 加速**)
   - 影响: 64K prefill 省 ~168ms，decode 不受影响

2. **GEMM autotune 表更新**
   - 文件: `runtime/nvfp4_custom_gemm.py`, `benchmarks/a2_autotune.py`
   - 改动: 加入 Laguna 真实 shapes + M=16 autotune 数据
   - 新增 N=1024 (shared expert) 条目
   - 数据: `benchmarks/fixtures/a2_autotune_table.json`

3. **接受率回归基准建立**
   - 脚本: `benchmarks/acceptance_regression.py`
   - 数据: `benchmarks/fixtures/acceptance_regression_20260729.json`
   - 结论: 真实负载 P50=98.5%, 平均=84.5%, 不存在 15% bug

### 未完成 / 需要深入 kernel 工作

- **MoE kernel 优化 (59% of verify)**: sparkinfer 内部 CUTLASS kernel，
  tile_m=16 已是最小，max_active_clusters=188 已是最大。
  非确定性模式 (ATOMIC_SCATTER) 未能测试（daemon OOM）。
- **Decode tok/s 差距 vs vLLM**: 282 vs 383 (-26%)，主要来自
  sparkinfer MoE/attention vs Marlin/FlashInfer 的 kernel 效率差异。
