# 诊断工具使用指南(bfdiag)

**给谁看:任何在这个仓库里排查问题的人或 agent。**

这个项目只有一块 GPU,无法并行,跑一次测试几分钟。所以效率的唯一杠杆是
**每次 GPU 运行的信息产出量**。这套工具就是为此而建。

设计背景见 `notes/2026-07-27-probe-system-design-and-plan.md`。

---

## 三条黄金法则

> **1. 不要再写一次性诊断脚本。** `benchmarks/` 下已经有 144 个、32710 行,零复利。
>    要跑实验就投给热引擎(`bf exec`),要留证据就走 run record。
>
> **2. 比较两个数字之前,先 `bf diff`。** 2026-07-27 有人比较了两次接受率
>    (1.000 vs 0.687)得出"我们已经打平"的结论,实际上两次用了不同的 prompt,
>    推翻它花了一整天。`bf diff` 的存在就是为了让这件事不可能再发生。
>
> **3. 出问题时先读已有的 trace,不要急着重跑。** 飞行记录仪常态开启,
>    失败那次运行的轨迹已经在盘上了。重跑要几分钟,而且偶发问题未必复现。

---

## 现在能用什么

| 工具 | 命令 | 状态 |
|---|---|---|
| 运行档案 + 配置指纹 | `bf ls` / `bf show` / `bf diff` | ✅ CPU 实测通过 |
| 飞行记录仪(生命体征面板) | `bf trace show` / `bf trace diff` | ✅ CPU 实测通过 |
| 不变量断言 | `QSR_ASSERT_LEVEL=1` | ⚠️ 集成代码未经 GPU 验证 |
| 常驻热引擎 | `bf daemon` / `bf exec` / `bf repl` / `bf submit` | ⚠️ 真实 provider 从未运行过 |
| oracle 逐层对拍 | `bf divergence` | ⚠️ 真实采集路径未经 GPU 验证 |

**⚠️ 标记的含义**:逻辑本身有 CPU 单测覆盖,但**真实引擎路径一次都没跑过**
(开发全程禁用 GPU)。第一次在 GPU 上用时请当作"首次运行"对待,各工具的
`notes/2026-07-27-bfdiag-*.md` 里都有 GPU 验证待办清单。

---

## 症状 → 工具

### 接受率下降

```bash
# 1. 先排除"根本不可比"                                      (0.1 秒)
bf diff <本次run> <上次good>

# 2. 看 reject_position 分布 —— 这一步信息量最大               (0 秒,数据已在盘上)
bf trace show <run>

# 3. 专家路由是否与 vLLM 分叉                                  (开发中)
bf probe routing --compare

# 4. 逐层找第一个数值发散点                                     (~60 秒)
bf divergence --prompt <fixture>
```

**为什么第 2 步信息量最大**:聚合的 `acceptance_rate=0.687` 是一个数字,
而 `reject_position` 的**形状**直接区分几类完全不同的 bug:

| reject_position 形态 | 指向 |
|---|---|
| 集中在 0-2(很早就被拒) | draft 模型一开始就跟不上 → **draft KV / context 状态坏了**,查 draft ring 的相关算术 |
| 大致均匀分布 | draft 逐步偏离 → 数值漂移类问题 |
| 集中在高位(12-15) | draft 大体正确、末尾发散 → **窗口边界 / 对齐**问题 |
| 双峰(要么 -1 全接受,要么很早被拒) | **状态污染** —— 某些轮次进入了坏状态 |

### 输出垃圾 / NaN

```bash
QSR_ASSERT_LEVEL=1 <你的实验>    # 把"30 秒后变垃圾"变成"第 12 轮违反不变量 X"
bf trace show <run>              # 看在第几轮开始不对
bf divergence --prompt <fixture> # 逐层定位
```

### 速度突然变慢

```bash
bf diff <本次> <上次>    # 配置变了吗
bf trace show <run>      # CG 命中率 / eager fallback 次数及原因 / 轮次耗时 outlier
```

`bf trace show` 的 `-- outliers --` 段会自动标出偏离本次运行自身中位数很远的轮次
——历史上那个"270 秒的诡异延迟"就该在这里跳出来。

### 偶发 / 无法复现

**不要重跑。** 直接 `bf trace show <失败那次的 run>`。这就是飞行记录仪存在的理由。

---

## 命令速查

### 运行档案

```bash
bf ls                      # 最近的运行,一行一个
bf ls -n 20
bf show <run_id>           # 一次运行的完整档案(run_id 支持唯一前缀)
bf diff <A> <B>            # 配置差异 + 指标差异 + 可比性判定
bf diff                    # 不带参数 = 比较最近两次
bf diff <A> <B> --json
```

`bf diff` 在检测到"会改变结论"的字段不同时,顶部打印醒目告警并**以非零码退出**
(可用于脚本门禁):

```
⚠ NOT COMPARABLE: workload.block_size differs (64 → 128)

== config diff ==
  workload.block_size: 64 → 128
== metrics diff ==
  acceptance_rate: 0.985 → 0.478  (-51.5%)
```

> **重要理解**:`bf diff` 的价值**不是阻止你比较**。做 A/B 实验时那个 ⚠ 是预期的
> —— 真正的价值是 `== config diff ==` 只列出了 `block_size` 这一行,**证明了除了
> 你以为的那个变量,没有别的东西变了**。这正是 7/27 那次缺的东西。

给脚本接入 run record,两种写法:

```python
# 零侵入,一行
from bfdiag.record import auto_record; auto_record()

# 或者显式,能记指标和产物
from bfdiag.record import run_record
with run_record(script=__file__, workload={"prompt_hash": h, "k": 15, "block_size": 128}) as rec:
    rec.metric("acceptance_rate", 0.478)
    rec.artifact("profile", path)
```

脚本中途抛异常时 record **仍然落盘**,`status=failed` 且带 traceback ——
崩掉的实验也是数据。

### 飞行记录仪

```bash
QSR_TRACE=1 <你的实验>              # 开启记录(默认关,关闭时开销 ~30ns/轮)
QSR_TRACE_RING_SIZE=8192            # 环容量,默认 8192 轮

bf trace show <run_id>              # 生命体征面板(默认只显示最后 50 轮)
bf trace show <run_id> --limit 0    # 全部轮次
bf trace show <run_id> --json
bf trace diff <A> <B>               # 两条 trace 逐轮对齐,报告第一个分叉轮次
```

面板尾部的聚合段是真正要看的东西:

```
=== bfdiag trace summary ===
total rounds: 60
acceptance rate: 47.778%

-- reject_position histogram (DFlash rounds; -1 = full accept) --
   -1: 20
    2: 8
    3: 19
    4: 8
    5: 5

-- CUDA Graph path --
  cg_hit_rate: 100.000%
  cg_replay: 60

-- phase latency (ms) --
  t_verify  p50=...  p99=...
  t_round   p50=...  p99=...

-- outliers (t_round far from the run's own median) --
  none
```

### 不变量断言

```bash
QSR_ASSERT_LEVEL=0   # 关(默认)
QSR_ASSERT_LEVEL=1   # 便宜的宿主端检查 —— 排查期建议一直开
QSR_ASSERT_LEVEL=2   # 含较贵的
```

当前 6 条(全部从真实代码推导,每条都有反例测试守着):
`slot_committed_tokens` 恰好领先 `slot_kv_len` 一个、`accepted_n ≤ K+1`、
`kv_len` 单调不减、`BlockPool.allocate` 单次调用内 block id 不重复、
aux hidden 的 offset 非负、CG replay 前 slot 一致性(暂未接线)。

违反时抛 `InvariantViolation`,消息里带轮次 + 上下文 + 最近若干条 trace 事件。

### 常驻热引擎

```bash
bf daemon start          # 加载一次模型(权重 26.76s + draft 3.97s),之后常驻
bf daemon status
bf daemon stop
bf exec experiments/foo.py
bf exec -c "print(engine.describe())"
bf repl                  # 交互式
bf submit --sweep 'QSR_DFLASH_CUDA_GRAPH=0,1' script.py   # 笛卡尔积扫描
```

**冷启动 vs 热引擎 —— 这条分界必须记住,用错了不会报错但数字是假的:**

| 测什么 | 热引擎 |
|---|---|
| 稳态 decode 性能(tok/s、ITL、每轮耗时) | ✅ 可靠 |
| 接受率、逐层对拍、路由对比 | ✅ 可靠 |
| 换 prompt / 换 K / 换运行时开关 | ✅ 可靠 |
| **冷启动 prefill 性能** | ❌ 必须冷启动(见 `notes/2026-07-20-cold-prefill-allocation-sensitivity-investigation.md`) |
| **load-time 配置**:`block_size` / `capacity` / `gpu_memory_utilization` / `max_model_len` / 量化后端 | ❌ 加载时定死,换一个必须重启进程 |
| 显存压力 / OOM 边界 | ❌ 热引擎碎片状态与冷态不同 |

每次 `exec` 前会跑金丝雀自检(固定 prompt、greedy、固定步数,输出必须逐位匹配基线),
不匹配就拒绝执行并标记 daemon 为 `TAINTED`。**不要关掉它**(`QSR_BFD_CANARY=0`)
——它是防止实验之间状态污染的唯一屏障。

### oracle 逐层对拍

```bash
bf divergence --prompt <fixture>
bf divergence --prompt <fixture> --json
bf divergence --prompt <fixture> --refresh-cache
```

自动定位第一个越过阈值的层,并在该层内下钻到子模块。
oracle 侧激活值按 `(model_revision, prompt_hash)` 缓存到
`.bfdiag/oracle_cache/`,之后只跑我们这侧。

---

## 反模式(这部分和正面指引一样重要)

| ❌ 不要 | ✅ 改成 |
|---|---|
| 在 `benchmarks/` 下新写一次性 diag 脚本 | `bf exec` 投给热引擎 |
| 每个实验冷启动一次(每次付 30s+ 加载) | `bf daemon start` 一次,之后秒级 |
| 直接对比两个数字下结论 | 先 `bf diff` 确认除目标变量外没有别的变化 |
| 手搓 decode 循环调 `engine._forward_main_with_aux` 等私有方法 | 走公开路径,否则测的不是生产路径 |
| 为了看一个数重跑一遍加 `print` | 数据已经在 trace 里,`bf trace show` |
| 手写 `QSR_DEBUG_*` 临时 dump | 走 `QSR_TRACE=1`,字段不够就往事件 schema 里加(永久受益) |
| 在热引擎里扫 `block_size` / `capacity` 之类 load-time 配置 | 冷启动路径,否则四组数字其实是同一个配置 |
| 关掉金丝雀图快 | 别关,状态污染的代价远大于那几秒 |

---

## 环境变量总表

| 变量 | 默认 | 作用 |
|---|---|---|
| `QSR_BFDIAG_DIR` | `<repo>/.bfdiag` | 所有产物的根目录 |
| `QSR_BFDIAG_RUN_ID` | 自动生成 | 把 trace 关联到某次 run record |
| `QSR_TRACE` | `0` | 飞行记录仪开关 |
| `QSR_TRACE_RING_SIZE` | `8192` | 环容量(轮) |
| `QSR_ASSERT_LEVEL` | `0` | 不变量断言级别 0/1/2 |
| `QSR_BFD_SOCKET` | `.bfdiag/bfd.sock` | daemon socket |
| `QSR_BFD_CANARY` | `1` | daemon 金丝雀自检 |
| `QSR_BFD_TIMEOUT_S` | `30` | exec 默认超时 |

---

## 还没做的(路线见设计文档)

| 阶段 | 内容 |
|---|---|
| P1 | 探针总线(统一 T0/T1/T2 写入 API) |
| P2 | **T1 归约签名**(每层 32 字节指纹)+ **MoE 专家路由探针** |
| P3 | T2 全量张量 + **预触发冻结**(异常时冻结环,拿到症状之前 N 轮的完整数据) |
| P4 | 进程外消费者(引擎崩溃后数据仍在) |
| P5 | **单轮确定性回放**(把"重跑 3 分钟"变成"200 毫秒") |
