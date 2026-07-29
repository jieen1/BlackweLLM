# CG Slot Doubling Bug Fix (2026-07-29)

## BUG: ServerEngine 要求 2× capacity 的 slot

`server/engine.py:226`:
```python
min_slots = capacity + (capacity if enable_cudagraph else 0)
```

capacity=4 + cudagraph=True → min_slots=8。Laguna 按 `num_slots × blocks_per_slot`
全量分配 KV (`laguna.py:291`)，所以 8 slot × 256K × 24KiB/token = **48 GiB KV**。

## 为什么不需要

1. **Decode CG 只用 1 个 slot 捕获** (`laguna_cuda_graph.py:310`):
   `warmup_slots = range(num_slots - 1, num_slots)` — batch_size=1
2. **DFlash CG 用共享 scratch** (`engine.py:207-220` 注释已记录):
   draft/verify CG 逐 slot 顺序 replay，不需要额外物理 slot
3. **DFlash 模式不用 decode CG**: DFlash 请求走 `dflash_round`，不走 M=1 decode

## 修复

```python
cg_extra = 0
if enable_cudagraph and not enable_dflash:
    cg_extra = 1  # 仅 M=1 decode CG 捕获需要 1 个 warmup slot
min_slots = capacity + cg_extra
```

| 配置 | 修复前 | 修复后 | 节省 |
|------|--------|--------|------|
| cap=4, DFlash, CG | 8 slots | **4 slots** | 24 GiB @256K |
| cap=4, 无DFlash, CG | 8 slots | **5 slots** | 18 GiB @256K |
| cap=1, DFlash, CG | 2 slots | **1 slot** | 6 GiB @256K |

## 修复后 4×256K 显存预算

| 组件 | 大小 |
|------|------|
| 主模型权重 (sparkinfer MoE + non-MoE) | 66.96 GiB |
| Full-attention KV (4 × 256K) | 24.00 GiB |
| SWA ring (4 slots) | 0.35 GiB |
| SWA scratch | 0.60 GiB |
| DFlash 权重 + KV | 2.11 GiB |
| Workspace/allocator 开销 | ~3.3 GiB |
| **合计** | **~97.2 GiB** |
| GPU 总量 | 95.6 GiB |

**仍然超出！** 4×256K 需要 ~97 GiB > 95.6 GiB GPU。

## 安全配置建议

| 配置 | blocks_per_slot | 预计显存 | 安全? |
|------|----------------|----------|-------|
| 4 × 192K | 3072 | ~91.2 GiB | ✓ |
| 4 × 200K | 3200 | ~91.9 GiB | ✓ |
| 4 × 256K | 4096 | ~97.2 GiB | ✗ 超出 |
| 2 × 256K | 4096 | ~85.2 GiB | ✓ |

## 另一个问题: app.py 注释不一致

app.py 注释声称 "KV pool 由 GPU memory profiling 决定"，但 Laguna 实际按
`num_slots × blocks_per_slot` 固定分配。`gpu_memory_utilization` 不约束 Laguna KV。
已修正注释。

## 长期优化方向 (调研结论，未实施)

1. **全局 BlockPool 按需分配**: 不启动时全量分配，按请求实际长度分配 page
2. **Prefix 共享**: `reconcile_prefix_hit()` 当前永远返回 0，4 agent 共享
   system prompt 可节省 3 份 KV (16K prefix → 省 1.1 GiB, 128K → 省 9 GiB)
3. **Laguna block-table 间接层**: 当前用 `slot × blocks_per_slot` 直接寻址，
   改为 page table 间接寻址后可支持动态分配和共享
4. **Session affinity**: agent 工具调用后保留 KV，不重复 prefill
