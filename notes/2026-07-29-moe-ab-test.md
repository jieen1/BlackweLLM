# MoE A/B Test: Deterministic vs Non-deterministic (2026-07-29)

## 结论

**Deterministic 模式 (ROUTE_BUFFER_TOPK_SUM) 更快，不要改。**

## 数据 (4K English prompt, 256 max_tokens, daemon fresh start)

| Config | tok/s | ITL(ms) | Accept% | vs baseline |
|--------|-------|---------|---------|-------------|
| deterministic (baseline) | 355.8 | 2.81 | 96.1% | -- |
| non-deterministic (ATOMIC_SCATTER) | 328.5 | 3.04 | 95.3% | **-7.7%** |
| ready_queue work source | CRASH | -- | -- | DSLRuntimeError |

## 分析

1. **Non-deterministic 更慢**: ATOMIC_SCATTER 在 M=16 (routed_rows=160) 时
   有 atomic 竞争开销。ROUTE_BUFFER_TOPK_SUM 用 buffer 做 reduction，
   对 M=16 更高效。

2. **ready_queue 崩溃**: sparkinfer CUTLASS DSL 的 `_publish_ready_tasks`
   在 compile time 遇到 dynamic Boolean → bool 转换错误。这是 sparkinfer
   的 bug，不是我们的问题。

3. **env var 是 runtime 读取**: `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT`
   和 `SPARKINFER_DYNAMIC_WORK_SOURCE` 在每次 MoE 调用时读取，
   可以通过 bf exec monkey-patch 测试，不需要重启 daemon。

## 已验证，不需要再测

- [x] SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=0 → 更慢，不开
- [x] SPARKINFER_DYNAMIC_WORK_SOURCE=ready_queue → 崩溃，不可用
- [x] _physical_slot bug lead → RESERVED_PHYSICAL_SLOTS=0，两边一致，不是 bug

## MoE Tile/Cluster Tuning (2026-07-29, 补充)

| Config | tok/s | vs baseline |
|--------|-------|-------------|
| default (auto) | 368.9 | -- |
| tile=16x128 | 351.9 | -4.6% |
| tile=32x128 | 340.0 | -7.8% |
| tile=16x256 | ERROR | unsupported |
| tile=32x256 | ERROR | unsupported |
| tile=64x128 | CRASH | CUDA illegal memory access |

**结论: sparkinfer 自动调优已是最优，不要手动设置 tile。**

MaxActiveClusters 和 down_scale 测试因 tile=64x128 导致的 CUDA 崩溃而无法完成。
