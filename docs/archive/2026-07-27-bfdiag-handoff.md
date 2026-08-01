# bfdiag 诊断平台 —— 交接说明

日期:2026-07-27
面向:BlackForge 开发团队
配套:`docs/diagnostics-guide.md`(完整使用指南)、`AGENTS.md`(agent 自动读取的精简版)

---

## 一句话

这台机器只有一块 GPU、无法并行,一次测试几分钟。所以效率的唯一杠杆是
**每次 GPU 运行能榨出多少信息**。`bfdiag` 就是为此建的工具平台,CLI 入口 `bf`。

---

## 装上就能用的东西

```bash
bf --help
# {daemon,exec,repl,submit,run,divergence,ls,show,diff,trace,probe,scan,shapes,determinism}
```

| 命令 | 干什么 | 状态 |
|---|---|---|
| `bf diff A B` | 两次运行的配置差异 + 指标差异 + **可比性判定** | ✅ CPU 实测 |
| `bf trace show` | 飞行记录仪:逐轮轨迹 + **reject_position 直方图** + CG 命中率 + 耗时 outlier | ✅ CPU 实测 |
| `bf shapes --diff 64 128` | 从真实 config 推导 kernel 形状,**列出哪些随 page_size 变了** | ✅ CPU 实测 |
| `bf divergence` | oracle 逐层对拍,定位第一个数值发散层 | ⚠️ 真实采集未经 GPU 验证 |
| `bf daemon` / `bf exec` | 常驻热引擎,免去每次 30s+ 冷启动 | ⚠️ **见下方警告** |
| `QSR_ASSERT_LEVEL=1` | 不变量断言(7 条,全部从真实代码推导) | ⚠️ 集成未经 GPU 验证 |
| `QSR_FORCE_SYNC` / `QSR_DETERMINISTIC` | 强制同步 / 确定性模式标准开关 | ⚠️ 未经 GPU 验证 |

**⚠️ 的含义**:逻辑有 CPU 单测覆盖,但**真实引擎路径一次都没在 GPU 上跑过**(整个开发过程禁用 GPU)。
第一次用请当作"首次运行"对待。每个工具的 `notes/2026-07-27-bfdiag-*.md` 都有 GPU 验证待办清单。

---

## ⚠️ 当前最重要的警告:热引擎的数字暂时不能信

用 daemon 测 DFlash 接受率(`block_size=128, CTX=10240`)得到 **0.6754**;
同配置冷启动脚本稳定复现 **0.452525**。**差 0.22,和正在追查的 bs=64→128 gap 同一量级。**

根因已定位:**两边 load-time 配置不同**。

| 参数 | 冷启动脚本(从 CTX 推导) | daemon 默认值 |
|---|---|---|
| `max_model_len` | `CTX + MAX_TOKENS + 2048` ≈ 12,352 | **131,072** |
| `blocks_per_slot` | `cdiv(max_model_len,bs) + cdiv(4096,bs)` ≈ 129 | **4096** |
| `block_size` | argv 传入(128) | **默认 64** |

`blocks_per_slot` 不是无害的容量参数:`runtime/backends/laguna_cuda_graph.py:94-96` 里
full-attention 组的 `max_pages = self.blocks_per_slot`,直接决定 sparkinfer decode workspace
规模,可能改变 kernel 分块与归约顺序 → 浮点求和顺序 → 临界 argmax 翻转 → 接受率。

**结论:在这个修复落地前,daemon 测出的接受率一律不要采信,继续用冷启动的 0.452525 做基线。**
修复方向已确定(去掉影响数值的静默默认值、改必填或统一公式推导、把完整 load_config
落进 RunRecord 让 `bf diff` 能自动拦截),正在进行。

---

## 对当前 block_size 64→128 排查最有用的三件事

### 1. `bf shapes --diff 64 128` —— SWA 对齐余量,不用跑 GPU 就能算

```bash
bf shapes --block-size 64 --block-size 128 --diff --kv-len 65600
```

已算出并经手工复核的结果:

```
kv_len=65536   bs=64: aligned_len=513   bs=128: aligned_len=513
kv_len=65600   bs=64: aligned_len=513   bs=128: aligned_len=577   ← 分叉,多 64 个 token
kv_len=10240   bs=64: aligned_len=513   bs=128: aligned_len=513
```

关键在于**它随 kv_len 变化**。从 kv_len=10240 起连续 256 步:
**128 步(50%)两边 `aligned_len` 不同,bs=128 最多多吃 64 个 token。**

机制来自 `laguna_cuda_graph.py::_fill_buffers_b1` 的 SWA 分支:
```python
window_start  = max(0, kv_len - window + 1)
aligned_start = (window_start // ps) * ps      # ← 对齐粒度 = block_size
aligned_len   = new_kv - aligned_start         # ← 作为 cache_seqlens 喂给 kernel
```
`ps` 从 64 变 128,向下取整的粒度翻倍,`aligned_len` 最多大 64。

**假设(未经实验证实)**:bs=128 在约一半的 decode 步上让 SWA 层多看到最多 64 个
本该在滑窗外的旧 token → V 加权归约覆盖范围不同 → 浮点求和顺序不同 →
近似平局处 argmax 翻转 → 接受率下降。

这与 `notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md` 记录的
"临界 argmax 翻转"现象自洽。`QSR_DEBUG_SWA_ALIGN_GRANULARITY` 是隔离这个变量的正确实验。

`--diff` 同时把 **18 个 GEMM、MoE 专家权重、q 张量** 明确列为 `unchanged` ——
这是有价值的排除项,说明这些不是嫌疑对象。

### 2. `bf trace show` —— reject_position 直方图

聚合的 `acceptance_rate=0.4525` 是一个数字;**分布的形状是一整类诊断**:

| 形态 | 指向 |
|---|---|
| 集中在 0-2(很早就拒) | draft 一开始就跟不上 → draft KV / context 状态坏了 |
| 大致均匀 | draft 逐步偏离 → 数值漂移类 |
| 集中在 12-15 | draft 大体对、末尾发散 → 窗口边界 / 对齐问题 |
| 双峰(要么全接受要么很早拒) | 状态污染,某些轮次进了坏状态 |

用法:
```bash
QSR_TRACE=1 QSR_ASSERT_LEVEL=1 QSR_BFDIAG_RUN_ID=bs64  <bs=64 运行>
QSR_TRACE=1 QSR_ASSERT_LEVEL=1 QSR_BFDIAG_RUN_ID=bs128 <bs=128 运行>
bf trace show bs128
bf trace diff bs64 bs128     # 逐轮对齐,报告第一个分叉轮次
```

**首次使用建议先跑一个 10 轮的短跑验证仪表本身**,别拿正式实验去赌。
`QSR_TRACE=0`(默认)时完全惰性,不影响现有流程。

### 3. `bf diff` —— 防止拿不可比的两个数下结论

```bash
bf diff <run_A> <run_B>
```
配置里任何"会改变结论"的字段不同,顶部会打醒目告警并以非零码退出:
```
⚠ NOT COMPARABLE: workload.block_size differs (64 → 128)
== config diff ==
  workload.block_size: 64 → 128
== metrics diff ==
  acceptance_rate: 0.985 → 0.478  (-51.5%)
```

> 做 A/B 时那个 ⚠ 是**预期的**。真正的价值是 `config diff` 只列出 `block_size` 这一行,
> **证明了除了你以为的那个变量,没有别的东西变了**。

---

## 一个已修的真实 bug(待接线)

`_fill_buffers` 里 `cache_seqlens` 写的是**未裁剪**长度,但 `page_table` 只填**裁剪后**的条目数。
生产配置下两者刚好卡在边界所以不触发;**一旦对齐粒度调大,注意力 kernel 会读到 page_table 里
未刷新的陈旧页号** —— 因为是 CUDA Graph replay,那些槽位残留的是上次的页号,读出来是别的 slot 的 KV。
症状会是"输出看起来正常但预测变差",极难查。

已做成永久性不变量 `check_page_table_covers_seqlen`(`cdiv(cache_seqlens, page_size) <= 有效条目数`),
带反例测试:证明 bs=64 生产配置不触发、对齐粒度调大后触发。

**接线补丁没有自动应用**(该文件当时有人在实时编辑),完整可照抄的补丁在
`notes/2026-07-27-bfdiag-flight-recorder.md` 第 10 节。必需的那一处:

`runtime/backends/laguna_cuda_graph.py`,文件顶部:
```python
from bfdiag.invariants import checks as bfdiag_checks
```
`LagunaCudaGraphVerify._fill_buffers` 里 `self._cache_seqlens[group_key][0] = aligned_len` 之后:
```python
                bfdiag_checks.check_page_table_covers_seqlen(group_key, aligned_len, n_ring, bs)
```

---

## 环境注意

- **测试请用 `/home/bot/.venvs/vllm/bin/python`** —— 仓库 `.venv` 有 torch 但**没装 vllm**,
  用它跑会产生一堆假失败。
- 跑测试时建议带 `CUDA_VISIBLE_DEVICES=""`(单 GPU,避免和别人抢)。
- 基线:3 个失败(`test_bf_attention` ×2、`test_vllm_dependency_boundary` ×1)是**既有的**,
  与本平台无关。

---

## 反模式(和正面指引一样重要)

| ❌ 不要 | ✅ 改成 |
|---|---|
| 在 `benchmarks/` 下新写一次性 diag 脚本(已有 144 个、32710 行,零复利) | `bf exec` 投给热引擎 |
| 直接对比两个数字下结论 | 先 `bf diff` 确认除目标变量外没有别的变化 |
| 为了看一个数重跑一遍加 `print` | 数据已在 trace 里,`bf trace show` |
| 手写 `QSR_DEBUG_*` 临时 dump | 走 `QSR_TRACE=1`;字段不够就往事件 schema 里加(永久受益) |
| 在热引擎里扫 `block_size` / `capacity` 之类 load-time 配置 | 冷启动路径,否则几组数字其实是同一个配置 |
| 手填 kernel 隔离测试的 shape | `bf shapes` / `bfdiag.shapes.harness` 从真实 config 推导 |

---

## 路线图(未做)

| 阶段 | 内容 |
|---|---|
| P1 | 探针总线(统一 T0/T1/T2 写入 API) |
| P3 | T2 全量张量 + **预触发冻结**(异常时冻结环,拿到症状之前 N 轮完整数据) |
| P4 | 进程外消费者(引擎崩溃后数据仍在) |
| P5 | **单轮确定性回放**(把"重跑 3 分钟"变成"200 毫秒") |
| P6 | **主动式 checkpoint/restore**(存档 prefill 后状态,跳过重复 prefill) |

设计与论证:`notes/2026-07-27-probe-system-design-and-plan.md`
