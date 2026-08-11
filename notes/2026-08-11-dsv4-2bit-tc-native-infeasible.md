# 新 2-bit Tensor-Core-native checkpoint format：可行性研究（§9 分支 (a)）

状态：**§9 分支 (a) 已调研并实测证伪**。在"单卡 2-bit 驻留 + 质量 gate >=0.9999 +
SM120 硬件"三重约束下，不存在 mma-ready 的 2-bit 权重格式。本文记录证据和
不可行性的精确边界，作为 §9(c) "如实记录未证可达" 的依据。

日期：2026-08-11

## 1. 动机

Phase 2B-0 实测：IQ2_XS 的 decode（grid 查表 + sign 合并）占 gate+up 的 45%
（约 2.9ms@E=256），facc 占 34%。two-plane 上限 nodecode 3.21ms 仍超 kill gate
2.4ms。§9 指定分支 (a)：若能有"mma-ready 的 2-bit 格式"，decode 可消除或极简。

## 2. 约束（三重）

1. **驻留**：routed weights 74.58 GiB 以 2-bit 级常驻（2.3125 bit/value =
   74B/256 值）。禁止 resident W8A8/NVFP4/BF16 副本（96GB 卡无法与
   2×128K KV + decode graph 共存，计划 §4.3 line 259/324）。
2. **质量**：gate/up/down 相对 exact oracle cosine >=0.9999（kill gate）。
3. **硬件**：SM120 mma 只支持 s8/f8/f16/bf16/tf32。无原生 2-bit mma。
   mma 输入必须是 8-bit 倍数，2-bit 数据必须 decode 或展宽。

## 3. 实测证据

### 3.1 线性码本质量不足

把每 32 值段的量化电平改成"对称线性码本"（2..6 bit），无表、纯算术 decode：

| bit | 平均 cos（24 行真实 blk.4 权重） | 达标? |
|---:|---:|---|
| 2 | 0.922 | 否 |
| 3 | 0.965 | 否 |
| 4 | 0.983 | 否 |
| 5 | 0.992 | 否 |
| 6 | 0.996 | 否 |

即使 6-bit 也达不到 0.9999。线性/均匀码本整体被证伪（`feas_kbit.py`）。

### 3.2 IQ2 grid 不可缩小

512-entry grid 中 **512/512 全部被真实权重使用**，每个 grid 至少命中 128 次。
无冗余可剪枝（`feas_gridsize.py`）。grid 是精度的必要部分，缩小即损质量。

### 3.3 唯一保质量路径 = 逐值真实值（需 >2-bit）

IQ2_XS 用 9-bit grid 索引 + 3-bit sign 编码 8 值（实际 12-bit/8 值），这是
质量 gate 的来源。任何 <=2-bit 的表示都无法编码同等信息。

## 4. 结论

在三条约束同时成立时，"mma-ready 2-bit 格式"不存在：

- 2-bit 驻留 → 每值 <=2.25 bit → 无法直接进 s8 mma（硬件要求 8-bit）；
- 展宽到 int8（预解码 codebook）→ 驻留 ×3.46（74B→256B），违反约束 1，
  且 DRAM 饱和实测 10.21ms（`iq2_tp_floor`）；
- 压缩 code + decode → 回到 IQ2_XS 现状（decode 成本不可避免）。

§9 预测的 "若模型/权重格式也不允许改变，则如实记录未证可达" **命中**。

## 5. 何时可行（若未来资源到位）

| 放宽哪个约束 | 结果 |
|---|---|
| 允许 ≥4-bit 驻留（NVFP4/MXFP4） | 有 mma-ready 路径，但 96GB 放不下 routed+KV |
| 有原始 BF16 权重 + imatrix 量化器 | 可设计新码本，但仍是 2-bit → decode 死结不变 |
| 放宽质量 gate 到 0.96（如 2-bit 线性） | 达标但质量不可接受 |
| SM120 未来支持 sub-8-bit mma（无此硬件） | 根本解，不在本项目控制内 |

## 6. 相关文件

- `/tmp/opencode/feas_lin2bit.py`、`feas_kbit.py`、`feas_gridsize.py`
- `notes/2026-08-11-dsv4-phase2b-scale-amortized-falsified.md`（Phase 2B-0 全链证据）
