# A6 · Attention Split-K 参数调查（2026-07-22）

## 背景

A1a profiling 显示 128K 下 attention 占 28.2%（decode 15.8% + prefill 11.2%）。
当前生产配置：fixed split=4096, max_splits=64（为 262K 最大上下文设计）。

## 实验：Split-K 对 decode kernel 延迟的影响

测试条件：batch=4, qo_len=4, FP8 KV, 10 iters, 3 warmup

### 多上下文长度对比（split=2048 vs 4096）

| KV len | split=2048 | split=4096 | 差异 | 备注 |
|--------|-----------|-----------|------|------|
| 32K    | 0.329ms (816 GB/s) | 0.321ms (837 GB/s) | 4096 快 2.5% | 噪声内 |
| 64K    | **0.503ms (1068 GB/s)** | 0.623ms (861 GB/s) | **2048 快 19.3%** | 显著 |
| 128K   | 1.084ms (990 GB/s) | 1.064ms (1009 GB/s) | 4096 快 1.9% | 噪声内 |

### 128K 下多 split 对比

| Split | Max Splits | Time | BW |
|-------|-----------|------|-----|
| 2048  | 65        | 1.100ms | 976 GB/s |
| 4096  | 33        | 1.166ms | 921 GB/s |
| 8192  | 17        | 1.306ms | 822 GB/s |
| 16384 | 9         | 1.126ms | 953 GB/s |

### SM120 vs FlashInfer 全曲线

| KV len | SM120 (ms) | FlashInfer (ms) | Speedup | SM120 BW |
|--------|-----------|----------------|---------|----------|
| 4K     | 0.326     | 1.224          | 3.76×   | 103 GB/s |
| 16K    | 0.348     | 1.232          | 3.54×   | 385 GB/s |
| 32K    | 0.352     | 1.484          | 4.22×   | 763 GB/s |
| 64K    | 0.640     | 1.611          | 2.52×   | 839 GB/s |
| 128K   | 1.080     | 1.879          | 1.74×   | 994 GB/s |

## 分析

1. **64K 是最优点**：split=2048 给出 32 splits（188 SMs 的 24%），比 split=4096 的 16 splits（12%）更好地利用 SM 并行度
2. **128K 已饱和**：32+ splits 已足够，更多 splits 增加 reduction 开销
3. **短上下文（≤32K）延迟受限**：0.32-0.35ms 几乎不随 context 变化，说明 kernel launch + reduction 是瓶颈
4. **带宽利用率**：峰值 ~1068 GB/s（64K, split=2048），约为理论峰值 1.8 TB/s 的 59%

## 优化方向

1. **Adaptive split**：根据实际 KV length 动态选择 split size
   - ≤32K: split=4096（减少 reduction 开销）
   - 64K: split=2048（最大化 SM 利用率）
   - ≥128K: split=4096（平衡 split 数与 reduction）
2. **预期收益**：64K 下 attention 快 19%，e2e 约 5%（attention 占 28% × 19% ≈ 5.3%）
3. **风险**：需要修改 kernel launch 逻辑，过 golden fixtures 门禁

## 裁决

- 64K 的 19% 提升显著，值得做 adaptive split
- 但 e2e 收益约 5%，需要权衡工程复杂度
- 建议：先在 runner 中实现 context-aware split 选择，过门禁后合入
- 优先级：P2（A2 降级后，A6 升为性能主线第一项）

## 补充实验：全局改 _DECODE_TARGET_SPLITS_PER_REQ=128 的效果

将 sm120_gqa.py 中 `_DECODE_TARGET_SPLITS_PER_REQ` 从 64 改为 128（kv_split_size=2048, max_splits=128）：

| KV len | 改前 (split=4096, max=64) | 改后 (split=2048, max=128) | 变化 |
|--------|--------------------------|--------------------------|------|
| 32K    | 0.352ms                  | **0.597ms**              | **+70% 退化** |
| 64K    | 0.640ms                  | 0.557ms                  | -13% 改善 |
| 128K   | 1.080ms                  | 1.009ms                  | -6.6% 改善 |

**结论：全局改不可行。** max_splits=128 的 workspace 开销在短上下文（32K）严重退化。
原始 64 确实是最佳全局折中（confirmed）。

**真正的解法：adaptive split（根据实际 kv_len 动态选择 split size）。**
需要 kernel 级改动或多 CUDA Graph bucket，工程量大，归入 M2→M3。

已回滚，保持 _DECODE_TARGET_SPLITS_PER_REQ=64。
