# 1K 方案确认实验：single-output down 与真实 K32 单层 MoE（2026-08-12）

> 指令来源：评审指出 1K 方案的 13.85ms core 未含 glue/shared，且 down 是双输出
> 重复计算；要求重建源码、实现 single-output down、测完整单层。

## 1. stale artifact 确认并修复

测试加载的 .so manifest source SHA `2e875f` 与当前源码 `860d1a` 不一致。
`make build-iq2-mma16-tc` 重建后匹配。**测试本身不校验 source SHA**，这是
stale-artifact 假绿漏洞（后续需在测试里加 manifest 一致性断言）。

## 2. single-output down（独立 kernel）

新建 `/tmp/opencode/iq2_mma16_tc_single.cu`，单矩阵 kernel（无 gate/up dual）。
真实 256 experts 实测：

| down | eff_pad=32 | eff_pad=48 |
|---|---:|---:|
| double（旧） | 11.45 ms | ~16.5 ms |
| **single（新）** | **5.73 ms** | **8.25 ms** |
| 回收 | **2.0x** | ~2.0x |

cos 0.99997（正确）。**评审判断正确：down 双输出是真实的 ~2x 重复工作**。

## 3. 完整单层 MoE（K32 + single-down + real 256 experts + 1024 tok + top-6）

无 router（用给定 top-6 indices/weights），含 group/gather/gate/up/SwiGLU/
single-down/stable combine：

- gate+up @eff_pad=48: **8.81 ms**
- down single @eff_pad=48: **8.25 ms**
- **完整 MoE layer: 21.97 ms**

预算对比：MoE all-in <=17ms（**超 4.97ms**）、layer <=22.5ms（勉强通过，
但该 layer 不含 attention/mHC，完整 layer 必超）。

## 4. 关键修正

1K 计划引用 core 13.85ms（gate+up 6.93 + down 6.92）基于**复制权重**
（32 unique × 8 重复）测量，**低估真实 DRAM 权重读取**。真实 256 unique
experts：gate+up 8.81（@48）+ single-down 8.25 = **17.06ms core**，已占满
MoE 17ms 预算，glue/combine 另加 4.9ms。

## 5. 判定

**K32 complete MoE <=17ms 当前失败（实测 21.97ms）**。single-output down
已落地（~2x 于 down 段），但 gate+up 与 glue 仍超。剩余优化空间：
- gate+up 8.81ms：decode 权重读是瓶颈（L1TEX 64%、DRAM 100%），需增大
  每 block ROWS 摊薄（N128 曾失败，需重新评估）；
- glue 4.9ms：Python 侧 group/gather/combine 可 fused；
- attention/mHC 未计入。

**1K 主线（Phase B all-in gate）尚未达标**；single-down 是唯一已确认的
正向回收。见 `docs/dsv4-prefill-1k-implementation-plan.md` §5.3。
