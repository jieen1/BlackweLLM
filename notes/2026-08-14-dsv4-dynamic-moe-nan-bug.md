# DSV4 动态 MoE NaN Bug（2026-08-14）

## 问题描述

`iq2_mma16_tc_dynamic_launch` kernel 在处理 routes 数 > 64 的 expert 时产生 NaN。

## 复现

```python
# expert 245 有 81 routes（lo=2063, hi=2144）
# 第一个 tile（routes 0-63）就产生 NaN
tile_eb[245] = 2063  # 正确
tile_eb[246] = 2127  # 正确
# 但 out_gate[2063:2144] 全是 NaN
```

## 已排查

| 检查项 | 结果 |
|--------|------|
| compact_xq 输入 | 无 NaN，norm=7882 |
| compact_xs 输入 | 无 NaN，norm=0.11 |
| gate_packed | 无 NaN，norm=3491431 |
| tile_eb | 正确（tile_eb[245]=2063, tile_eb[246]=2127）|
| 第一个 tile 输出 | NaN |

## 根因分析

`iq2_mma16_tc_dynamic_launch` kernel 使用 `M_PAD_C=64`，只能处理 64 routes。
对于 routes > 64 的 expert，需要分多次调用（每次 64 routes）。

但即使第一个 tile（routes 0-63）也产生 NaN，说明问题不在分 tile，
而在 kernel 本身对 `expert_bounds` 的处理。

## 关键发现

`tile_eb` 的结构：
- `tile_eb[0..244] = 0`（其他 expert 无 routes）
- `tile_eb[245] = 2063`（expert 245 的起始 offset）
- `tile_eb[246] = 2127`（expert 245 的结束 offset）
- `tile_eb[247..256] = 2127`（后续 expert 无 routes）

kernel 用 `expert_bounds[e]` 计算 `xbase`：
```cuda
const int64_t xbase = (int64_t)expert_bounds[e] * COLS;
```

对于 expert 245，`xbase = 2063 * COLS`。但 `compact_xq` 的总大小是 `R * COLS = 2184 * COLS`。
所以 `xbase = 2063 * COLS` 是有效的（< 2184 * COLS）。

但 kernel 的 grid 是 `(E, ROWS/32) = (256, inter/32)`，每个 CTA 处理一个 expert。
对于 expert 245，kernel 读取 `compact_xq[2063:2127]`（64 routes）。

但问题是：kernel 的 `route_hi = expert_bounds[e+1] = 2127`，
而 `v0 = (expert_bounds[e] + mt * 16 + lg) < route_hi`。

对于 mt=0, lg=0..7：`v0 = (2063 + 0 + lg) < 2127 = true`。
对于 mt=1, lg=0..7：`v0 = (2063 + 16 + lg) < 2127 = true`。
...
对于 mt=3, lg=0..7：`v0 = (2063 + 48 + lg) < 2127 = true`（lg=0..7 → 2111..2118 < 2127）。

所以所有 64 routes 都是有效的。但输出是 NaN。

## 可能的根因

1. **kernel 的 `xbase` 计算错误**：`xbase = expert_bounds[e] * COLS` 可能溢出或计算错误。
2. **kernel 的 `route_hi` 计算错误**：`route_hi = expert_bounds[e+1]` 可能读取错误的值。
3. **kernel 的 `v0` 掩码计算错误**：`v0 = (expert_bounds[e] + mt * 16 + lg) < route_hi` 可能计算错误。
4. **kernel 的 `facc` 累加错误**：`facc[mt][0] += (float)cg[0] * sB_g[0] * xs0` 可能溢出。

## 下一步

1. 在 kernel 中添加 debug 输出，打印 `xbase`, `route_hi`, `v0` 的值。
2. 检查 `xbase` 是否溢出（`2063 * 4096 = 8450048`，int64 不会溢出）。
3. 检查 `route_hi` 是否正确（应该是 2127）。
4. 检查 `v0` 是否正确（应该是 true）。
5. 检查 `facc` 是否溢出（`cg[0] * sB_g[0] * xs0` 可能溢出）。

## 临时解决方案

回退到 eager MoE（`grouped_moe_prefill_k32`），性能 161 tok/s。

## 长期方案

1. 修复 `iq2_mma16_tc_dynamic_launch` kernel 的 NaN bug。
2. 实现 tile-loop 结构，处理 routes > 64 的 expert。
3. 预期性能：MoE 2.19s → ~1.0s，整体 161 → ~400 tok/s。
