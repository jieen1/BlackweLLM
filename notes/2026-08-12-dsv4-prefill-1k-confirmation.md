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

## 6. 最终判定：K32 all-in MoE 预算失败

真实 256 experts + eff_pad=48 + 1024 token + top-6：

| 项 | 实测 | 预算 |
|---|---:|---:|
| gate+up kernel | 8.81 ms | — |
| down (single) kernel | 8.25 ms | — |
| **K32 kernel core** | **17.06 ms** | MoE all-in <=17 ms |
| glue (sort/group/quant/scatter/combine) | ~4.9 ms | 含在 MoE 内 |
| **完整 MoE layer (无 shared/attention)** | **21.97 ms** | layer <=22.5 ms |

**判定**：K32 kernel core（17.06ms）已占满 MoE 17ms 预算，glue 使完整 MoE
21.97ms 超预算。按评审指令"不能进入 17-18ms 应立即判失败"，**1K 主线
Phase B all-in gate 触发失败**。single-output down（2x）是唯一已确认的正向
回收，但不足以弥合缺口。

**根因**：文档 13.85ms core 基于 eff_pad=32 + 复制权重（32 unique × 8），
低估真实 eff_pad=48 + 256 unique 权重的 decode/DRAM 成本。gate+up 在
eff_pad=48 是 8.81ms（vs 文档 6.93），down single 8.25ms。

**剩余选项**（§11/§9）：
1. eff_pad 从 48 压到 32（max_routes 上限收紧，需 route 拆分，代价是多次
   launch）——可测，但 max_routes=42 是真实分布；
2. gate+up decode 优化（增大每 block ROWS 摊薄 decode 权重读，N128 曾失败）；
3. 接受 1K 目标需要进一步预算放宽（如 20ms/layer 级 MoE）。

## 7. CTA-local row-W8 探索（§9 reopen 候选）

K32 失败后按 §9 探索 CTA-local row-W8（decode-to-smem + 整行 scale，消除
per-K32 facc 链）：

| 变体 | E=256 eff_pad=48 | 说明 |
|---|---:|---|
| K32 gate+up（facc 链） | 8.81 ms | 基线 |
| K32 down single | 8.25 ms | 已落地 |
| row-W8 no-facc down | 6.48 ms | facc 消除 |
| row-W8 no-facc gate+up（合并） | 8.50 ms | 共享 decode |
| **CTA-local core（no-facc）** | **15.21 ms** | 目标 <=14 |

- facc 消除带来 17.06 → 15.21 ms（1.12x）；
- **仍超 14 ms 目标 8%**，且用 placeholder row scale（真实 Phase 1 扫全行
  scale 有额外成本）；
- **数值**：row-W8 down cos 0.99965（prescreen 实测），<0.9999 kernel gate，
  仅满足 >=0.9990 routed gate；
- Phase 1（整行 scale）的 atomicMax 正确性调试未完成（smem atomicMax 行为
  异常，改用归约需额外工作）。

**结论**：CTA-local row-W8 接近但不达标——core 15.21 vs 14，数值 down
0.99965。**两个候选（K32 17.06、CTA-local 15.21）均未满足 §5.3/§9 门禁**。
当前 1K 目标的 K32 complete MoE 与 CTA-local row-W8 都无已证路径。

剩余判断：1K 目标在"gate+up+down core <=14ms"（含质量 0.9999）约束下，
现有 IQ2 decode-to-smem + mma 组织在真实 256 experts + eff_pad=48 下
不可达。需用户重新定夺预算或探索其他组织。

## 8. CTA-local 最终判定：性能可行但数值不达标

CTA-local row-W8 的完整评估：

| 指标 | CTA-local row-W8 | K32 | 门禁 |
|---|---|---|---|
| core @eff_pad=48（真实） | 15.47 ms | 17.06 ms | <=14 ms |
| core @eff_pad=32（拆分） | 11.56 ms | — | <=14 ms |
| **routed MoE out cos** | **0.99965** | **0.99990** | >=0.9990 |
| down cos | 0.99965 | 0.99987 | >=0.9990 |

**关键**：row-W8 的整行 scale（K=4096/2048）使 down 数值 0.99965，直接导致
routed MoE 输出 cos 0.99965 < 0.9990 门禁。K32 数值达标（0.99990）但
eff_pad=48 性能 17.06ms 超 14ms。

**两难**：
- CTA-local：性能可行（eff_pad=32 拆分 ~13.5ms）但数值 down 0.99965 不达标；
- K32：数值达标（0.99990）但性能 17.06ms 不达标。

eff_pad 拆分（32）可救性能，但 row-scale 的数值损失是表示的固有属性
（prescreen K4096 down 0.99965 已证实）。

**最终判定**：1K 目标的 K32 complete MoE 与 CTA-local row-W8 **均无完整
（性能+数值）已证路径**。§9 的 CTA-local reopen 条件（routed cos>=0.9990）
不满足。需用户重新定夺：放宽数值门禁、或接受当前性能、或探索新组织。

## 9. K32 split complete MoE 落地（质量优先，2026-08-12）

实现 `grouped_moe_prefill_k32`（iq2_mma16_tc.py）：
- K32 scale-amortized kernel（routed cos 0.999896，≥0.9990 达标）；
- single-output down（单输出 kernel，正式 build target + wrapper + 测试）；
- eff_pad=32 固定 + 超 32 routes 的 expert 拆第二批（11 experts, 13 over-32）。

真实 256 experts / 1024 tok / top-6 实测：

| 阶段 | CUDA 时间 |
|---|---:|
| gate+up kernel<32> | 7.0 ms |
| single kernel<32> | 5.47 ms |
| batch2 kernels | 0.56 ms |
| glue CUDA（index/scatter/sort 等） | ~2 ms |
| **纯 GPU** | ~14.9 ms |
| **wall（含 host sync）** | **25.86 ms** |
| exact 基线 | 32.65 ms |

**关键**：kernel 13.03ms 已接近 14ms（§9 CTA-local core 线）；完整 MoE 25.86ms
超 17ms all-in 主要因 **~11ms host sync**（glue 的多 torch kernel launch 间隙）。
CUDA graph 捕获失败（grouped_moe_prefill_k32 有动态 allocation / 不定形状，
即文档 §5.1 禁止项）。

**下一步**：消除 glue 动态 allocation（固定 bucket + 预分配），使 CUDA graph
可捕获 → host sync 归零。kernel 13ms 本身接近达标，glue 是主缺口。

## 10. K32 MoE 优化与 graph 捕获障碍（2026-08-12）

- `_into` 变体（caller-owned buffer）：完整 MoE 25.86 → **22.26ms**（质量不变
  0.999896）；
- **CUDA graph 捕获失败**：glue 的 `torch.nonzero`/`argsort` 动态形状
  （文档 §5.1 禁止的"动态 sort"）；torch.compile 更慢（102ms，ctypes 不可优化）；
- **结论**：进 17ms 必须 device-side grouping（固定 bucket 的 device
  route/counts/offsets/gather），即 SparkInfer recipe 工程（C1/C2 范围）；
  当前 ds4_router.py 有 device top-k 可作基础。

**当前状态**：K32 完整 MoE 质量达标、性能 22.26ms（vs exact 32.65，1.47x）。
device-side grouping 是唯一明确的剩余路径。

## 11. device-side grouping 探索（2026-08-12）

实现 Triton device counts/within/fill kernel（`dsv4_grouping.py`）：
- counts/within 正确（eager 全 256 expert 校验通过），CUDA graph 可捕获
  （5 replay 一致）；
- **device-group eager 25.56ms 不优于 python glue 22.26ms**：每调用新建
  buffers + atomic kernel 开销；
- **batch2 动态 n_over 阻止完整 graph 捕获**（固定 [256,16] 全跑浪费 6.59ms）。

**结论**：device grouping 的 graph 潜力（~15ms）需完整重构（固定 batch2
上限 + 消除所有 alloc），工程量大且 batch2 动态性是障碍。**当前最优仍是
python glue 22.26ms**（质量 0.999896 达标）。

**最终状态**：K32 完整 MoE（质量优先）22.26ms vs exact 32.65ms（1.47x）。
进 17ms 需消除 ~5.3ms host sync，路径是完整 device pipeline（SparkInfer
recipe 工程），当前未达成。

## 12. 关键瓶颈：CPU-bound glue（2026-08-12）

torch profiler：K32 MoE 3 calls → **CPU self 101ms/call，CUDA 16ms/call**。
GPU 只 16ms，但 CPU 101ms 驱动它，wall 22.26ms 是 CPU 追赶 GPU。

最大 CPU 单项：`aten::index`（高级索引 scatter）24ms/call。

**结论**：即使 kernel 全优化，Python glue 的 CPU 开销（每 torch op 的
launch/Python 开销）是墙。**唯一解法是 device-side 全 pipeline（CUDA
graph 消除 CPU 101ms）**，但 batch2 动态 n_over 阻止 graph。

**质量优先路径的最终判断**：kernel 13ms 达标（质量 0.999896），但完整
MoE 22.26ms 被 CPU-bound glue（~6ms host sync + Python 开销）拖住。
device-side pipeline 是唯一剩余路径，需解决 batch2 固定形状。

## 13. K32 kernel race 修复 + CUDA graph 状态（2026-08-12）

修复 `iq2_mma16_tc.cu` / `iq2_mma16_tc_single.cu` 缺 `__syncthreads` 的
smem race（racecheck 2.2M hazards → 0）。修复后 eager K32 MoE **cos 1.0**
（bit-exact vs python grouped）。加确定性 within（argsort-based，graph-safe）。

**CUDA graph 捕获**：完整 MoE（batch1+batch2）捕获成功，replay **15.7-16.5ms
< 17ms 达标**。但 **down 输出 graph != eager**（cos 0.565），即使 hq1/hs1/
gate 全 graph==eager。单 kernel 单独 graph==eager；组合中 down 错。

**分析**：down 是 pipeline 第二个 kernel，graph 捕获后执行与 eager 不同，
疑似 CUDA graph 对自定义 kernel 的指令序列化差异。**标记未完成**。

**当前交付**：eager K32 完整 MoE cos 1.0，22.26ms（质量优先、正确）。
CUDA graph 16ms 需进一步调试（可能需 kernel 级改造）。
