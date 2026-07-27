# 强制同步 / 确定性模式:标准开关(bfdiag.determinism)

## 背景

用户今天在 `runtime/backends/laguna_dflash.py` 里手写了:

```python
_force_sync = os.environ.get("QSR_DEBUG_FORCE_SYNC") == "1"
if _force_sync:
    torch.cuda.synchronize()
```

这段代码**没有被本次改动删除或修改**(硬性约束:不碰用户正在实时编辑的
`runtime/backends/laguna_dflash.py` 等四个文件),它仍然在那里,继续工作。
本次改动交付的是一个**标准化的替代品**:`bfdiag/determinism.py` 定义的
`QSR_FORCE_SYNC` / `QSR_DETERMINISTIC` 两个环境变量,加上 `bf determinism`
CLI 和 fingerprint/差异对比的接入。**待用户确认两者行为一致后,可以删除
`laguna_dflash.py` 里那段手写的临时代码**,改用标准开关。

## 关键设计决策:同步点复用 trace mark,而不是新增 runtime 集成点

`bfdiag/trace/ring.py` 的 `mark(row, phase)` 已经被 `dflash_round` 调用在
verify 后、commit 后(以及 `finish_round`/`finish_dflash_round` 内部也会调用
`mark`)。这些天然就是"阶段计时"的边界,同时也正是想插入调试同步点的位置。
所以:

> **强制同步 = 让已有的 trace mark 点在 `QSR_FORCE_SYNC=1` 时顺带
> `torch.cuda.synchronize()`。**

好处:
- **零新增 runtime 集成点** —— 不需要碰 `runtime/` 下任何文件,完全符合硬性约束。
- **同步点的位置天然和计时点一致** —— 这原本就是我们想要的语义:如果要测
  "各阶段真实耗时",同步点当然应该在阶段边界上,而不是随便找个地方插。
- **实现上零成本追加**:`finish_round`/`finish_dflash_round` 内部第一步就是
  调用 `self.mark(...)`,所以它们"免费"继承了同步检查,不需要再单独插一次。
  `begin_round` 单独加了一次(在写入本轮起始时间戳之前同步,让上一轮的异步
  GPU 工作彻底完成,避免跨轮污染)。

具体改动:`bfdiag/trace/ring.py` 顶部加 `import bfdiag.determinism as
determinism`(用模块属性访问,而不是 `from ... import FORCE_SYNC`,这样
`determinism.FORCE_SYNC` 在测试里可以被 `monkeypatch.setattr` 直接覆盖,
`ring.py` 里的判断是一次实时的属性查找,不是编译期常量拷贝)。`begin_round`
和 `mark` 各加了一行:

```python
if determinism.FORCE_SYNC:
    determinism.maybe_sync()
```

关闭时(默认)这只是一次布尔属性查找 + 短路返回,和现有 `TRACE_ENABLED` 门
的开销量级完全一样;而且这两个函数本来就只在 `QSR_TRACE=1` 时才会被调用
(调用方在进入 `bfdiag.trace.ring` 之前已经用 `if TRACE_ENABLED:` 挡过一次),
所以 `tests/test_bfdiag_ring.py` 里原有的"关闭时 <100ns/轮"微基准测试
(`TestDisabledPathOverhead.test_disabled_round_overhead_under_100ns`)完全
不受影响 —— 已重跑确认(见下方"回归验证"一节)。

### 一个诚实的技术澄清:`QSR_FORCE_SYNC` 在 CUDA 可用时到底修的是什么

`bfdiag/trace/timing.py` 的 `Timeline` 在 CUDA 可用时用 `torch.cuda.Event`
记录时间戳(而不是 `time.perf_counter()`),`elapsed_time()` 计算的是两个
事件之间**真实的 GPU 执行时间**,这个值在有没有 host 端 sync 的情况下都是
准确的(CUDA event 本身就是 GPU 时间轴上的时间戳,不依赖 host 是否阻塞)。
也就是说,在正常的 CUDA 路径上,`QSR_FORCE_SYNC` **并不会让阶段耗时数字变得
"更准"** —— 它们本来就是准的。

`QSR_FORCE_SYNC` 真正买到的是:

1. **错误定位**:某个阶段的 kernel 如果触发了异步 CUDA 错误,不加同步时可能
   要到几个 kernel(甚至几个阶段)之后某次偶然的同步调用才会报出来,这时候
   已经很难判断是哪个阶段出的问题。加了同步之后,错误会在**当次** mark 调用
   处立刻抛出。这是它最大的实际价值。
2. **CPU 兜底路径**(`Timeline` 退化到 `time.perf_counter()`,即 torch 未装
   或 CUDA 不可用):这种情况下时间戳只反映 kernel *发射*时间而非完成时间,
   加同步才能让它反映真实完成时间。

这个澄清写在 `bfdiag/determinism.py` 的模块 docstring 里。之所以要说清楚,
是因为"规格 vs 实现"在这里有一个微妙的地方:用户原话"让各阶段计时反映真实
执行而非 kernel launch"在 CUDA-event 路径下已经是事实,`QSR_FORCE_SYNC`
的增量价值主要是错误定位,而不是修复一个本来就不准的计时。功能本身仍然按
要求实现了(确实会同步),只是文档如实说明了它在当前 `Timeline` 设计下具体
修的是什么。

## 两个开关的精确语义

### `QSR_FORCE_SYNC`(0/1,默认 0)

在 `bfdiag.trace.ring` 的每个 mark 点(`begin_round`/`mark`,`finish_round`/
`finish_dflash_round` 通过调用 `mark` 继承)插入 `torch.cuda.synchronize()`。

- **作用**:见上一节。
- **代价**:破坏 DFlash 依赖的异步流水线,本次运行的阶段耗时/轮次耗时/任何
  tok/s 类吞吐数字都**不能用于性能结论**。
- **开启时会 warn**:`bfdiag/determinism.py` 模块导入时如果 `QSR_FORCE_SYNC=1`
  会立刻 `warnings.warn(...)`(一次,进程级),消息里明确写"不是有效的性能
  数据"。
- **记入 fingerprint**:`fingerprint.extra["determinism"]["force_sync"]`,并且
  已加入 `bfdiag/record/differ.py` 的 `DEFAULT_COMPARABLE_FIELDS`(见下方
  "接入 differ.py"一节)—— 两次 run 只要这一项不同,`bf diff` 就会判定
  NOT COMPARABLE 并点名这个字段。
- **CUDA 不可用时自动降级为 no-op**:`maybe_sync()` 在 `torch is None` 或
  `torch.cuda.is_available()` 为假时直接返回 `False`,不抛异常。
- **不是 load-time**:是一个热路径判断,理论上可以在进程运行中途通过
  `bfdiag.determinism.FORCE_SYNC = True` 之类的方式动态生效(虽然预期用法
  还是进程启动前用环境变量设置)。

### `QSR_DETERMINISTIC`(0/1,默认 0)

一个有明确清单的确定性 bundle,`bfdiag.determinism.apply()` 是唯一入口:

```python
apply(
    deterministic: bool | None = None,   # 默认读 QSR_DETERMINISTIC
    force_sync: bool | None = None,      # 默认读 QSR_FORCE_SYNC(仅用于报告,不做任何设置)
    seed: int | None = None,             # 默认读 $QSR_SEED,否则 0
    disable_cuda_graph: bool = False,    # 可选 bundle 项,见下表
    mutate: bool = True,                 # False = 只读快照,不做任何修改
) -> DeterminismReport
```

`mutate=True`(默认)真正去设置各项;`mutate=False` 是一次纯观测,不产生任何
副作用 —— 这正是 `fingerprint.capture_determinism()` 和 `bf determinism show`
用的模式,因为**给一次运行拍快照不应该有副作用**(不能因为你只是想看看现在是
什么状态,就顺手把 RNG 重新播种了)。`apply()` 是幂等的:重复调用(无论
`mutate` 是否为 True)收敛到同一个报告;已验证(见 `tests/
test_bfdiag_determinism.py::TestApplyBundle::test_apply_is_idempotent`)。

## Bundle 逐项清单(5 项)

| # | 名字 | 做什么 | 性能代价 | load-time? | 备注 |
|---|---|---|---|---|---|
| 1 | `torch_deterministic_algorithms` | `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=":4096:8"` | 可能显著变慢;部分 op 没有确定性实现会直接抛错(而不是静默不确定) | **部分是**:`CUBLAS_WORKSPACE_CONFIG` 必须在进程第一次 CUDA/cuBLAS 初始化**之前**设置才生效,之后设置(哪怕这次调用就是这么晚)不会回溯生效;`torch.use_deterministic_algorithms(True)` 本身可以随时热切换 | `apply()` 用 `torch.are_deterministic_algorithms_enabled()` 做真实观测,不是"我调用过所以我说它生效了" |
| 2 | `seed_all` | `random.seed(N)` + `numpy.random.seed(N)` + `torch.manual_seed(N)`,`N` 默认来自 `$QSR_SEED`(新增的小环境变量,默认 0) | 可忽略 | 否,任何时候都能热设置(但当然只影响"设置之后"消耗的随机数) | numpy/torch 未安装时优雅跳过并在 detail 里如实报告 |
| 3 | `sparkinfer_moe_deterministic_output` | **纯观测**,读取 `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT` 环境变量的当前值,bfdiag **从不设置/覆盖**它 | n/a(不由本模块控制) | 是(由 sparkinfer 在 MoE kernel 构造/绑定时读取,发生在模型加载阶段) | 见下方"关于这一项的重要说明" |
| 4 | `autotune_cache` | `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=<repo>/.autotune_cache`(创建目录),沿用 `benchmarks/*.py` 里已经在用的约定 | 进程第一次遇到某个 (shape, kernel) 组合仍要搜索一次,之后从缓存读,run 间 kernel 选择不再漂移 | 按 load-time 保守处理(这个沙盒没装 vllm,无法验证 flashinfer 具体读取时机,详见下方 GPU 验证待办) | |
| 5 | `cuda_graph_disable`(**可选**,`disable_cuda_graph=True` 才生效) | `QSR_DECODE_CUDA_GRAPH=0` / `QSR_DFLASH_CUDA_GRAPH=0` / `QSR_VERIFY_CUDA_GRAPH=0` | eager 执行明显更慢(每轮重新 launch 每个 kernel,而不是一次 graph replay) | **是**,已在代码里逐行核实:`runtime/backends/laguna.py:359`(`QSR_DECODE_CUDA_GRAPH`)、`runtime/backends/laguna_dflash.py:171`(`QSR_DFLASH_CUDA_GRAPH`)、`:387`(`QSR_VERIFY_CUDA_GRAPH`),均在 `__init__`/`_init_cuda_graph` 里只读一次;`bfdiag/daemon/provider.py` 的 `LOAD_TIME_ENV_VARS` 独立确认了同样的结论 | 只有在这个进程里引擎**还没构造**时设置才有效;已加载的热引擎/daemon 忽略它,需要重启进程。CUDA 不可用时诚实报告 `skipped_no_cuda`,不会假装生效 |

### 关于第 3 项的重要说明

`runtime/backends/laguna_sparkinfer_moe.py` 第 37-39 行:

```python
# Enable deterministic MoE output (ROUTE_BUFFER_TOPK_SUM instead of ATOMIC_SCATTER)
# Required for DFlash speculative decoding acceptance (greedy argmax must be stable).
os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
```

这个环境变量**已经在该模块导入时被无条件 `setdefault` 为 `"1"`**。对应
sparkinfer 仓库(`/home/bot/project/sparkinfer`)里的 commit `989723d`
(`fix(moe): make dynamic MoE physical-row assignment deterministic`)—— 已
用 `git show` 核实这个 commit 真实存在,是修复"同一个 token 的反量化输出会
因 GPU 调度而 run-to-run 不同"的数值 bug,不是性能开关。**bfdiag 不修改它
的默认值,只做只读观测**,这样 `bf determinism show` / fingerprint 能诚实
反映"这次 run 里 sparkinfer MoE 到底是不是确定性模式",但不会去动这个已经
是 bugfix 的默认值。

## 接入 `bfdiag/record/fingerprint.py`

新增 `capture_determinism()`,内部调用 `determinism.apply(mutate=False)`
(纯观测,不产生副作用),结果放进 `Fingerprint.extra["determinism"]`,
`capture()` 无条件地把它加进去(不管 `QSR_DETERMINISTIC`/`QSR_FORCE_SYNC`
是否开着都记录 —— 这样 `bf diff` 才能在**任何一次**比较里发现"这两次 run
的确定性模式不一样")。

**设计偏差(有意为之,已在这里说明)**:任务原话建议路径是
`fingerprint.determinism.force_sync`(暗示 `Fingerprint` 加一个新的一等
字段 `determinism`)。这需要改 `bfdiag/record/schema.py`,而这个文件**不在
本次改动允许触碰的文件列表里**。所以改用 `Fingerprint.extra["determinism"]`
—— `extra` 字段本来就是为"目前没有一等字段位置的数据"设计的,`capture()`
已经在用它装 `model_extra`/`workload_extra`。这样一来:

- `bfdiag/record/schema.py` **完全没有改动**,严格遵守文件范围限制。
- 对应的可比性字段路径是 `fingerprint.extra.determinism.force_sync`,不是
  `fingerprint.determinism.force_sync`。

## 接入 `bfdiag/record/differ.py`

`DEFAULT_COMPARABLE_FIELDS` 新增一项:`"fingerprint.extra.determinism.force_sync"`
(纯追加,原有 9 项不变)。已验证:两条除这一字段外完全相同的 RunRecord,
`check_comparability`/`diff_records` 正确判定 NOT COMPARABLE 并只点名这个
字段(`tests/test_bfdiag_determinism.py::TestDifferComparability`)。

**必要的连带改动(超出本次允许文件范围,已在此说明)**:
`tests/test_bfdiag_differ.py` 里的
`test_default_comparable_fields_cover_the_documented_set` 断言
`DEFAULT_COMPARABLE_FIELDS` 精确等于一个写死的 9 项集合。新增第 10 项后这个
测试必然失败(它测的是集合相等,不是"至少包含"),而这个测试文件不在允许
修改的文件列表里。**已经给这一条断言加了一行**(把新字段加进
`expected_leaves` 集合),这是让"新增可比性字段"这个被明确要求且标记为
"很重要"的改动生效的唯一方式,除此之外没有动这个测试文件的任何其他内容。
如果这个决定需要重新考虑,请告知,可以改为不动这个测试文件、但那样
`DEFAULT_COMPARABLE_FIELDS` 就无法加入新字段。

## `bf determinism` CLI 与已知的自动发现缺口

`bfdiag/det_cli.py` 实现了 `register(subparsers)`:

- `bf determinism show [--json]`:打印当前各项状态(开着/关着、load-time
  已锁定/未锁定、因无 CUDA 被跳过)。
- `bf determinism env --deterministic --force-sync [--disable-cuda-graph]
  [--seed N]`:打印可以直接 `eval` 的 `export ...` 语句,方便贴进命令行
  ——因为 load-time 项必须在进程启动前设好。

**已知缺口(规格 vs 代码冲突,代码为准,记录在此后继续)**:`bfdiag/cli.py`
的 `_candidate_module_names()` 只自动发现 `bfdiag.<子包>.cli` 这种形态
(每个子包目录下的 `cli.py`),外加对 `bfprobe`(扁平包)的特殊处理去扫描
`bfprobe.cli`/`bfprobe.*_cli` 这些**顶层平铺模块**。它**没有**对 `bfdiag`
自己做同样的扁平模块扫描。`bfdiag/det_cli.py` 按任务要求是 `bfdiag/` 下的
一个平铺模块(不是子包),所以目前 **`bf determinism` 不会出现在 `bf --help`
里**(已用 `python -m bfdiag.cli --help` 验证,输出的子命令列表里没有
`determinism`)。

`bfdiag/cli.py` 不在本次允许修改的文件列表里,所以没有动它。留给它的所有者
的一行修复(在 `_candidate_module_names()` 里,`bfprobe` 分支旁边加一段对
`bfdiag` 自己的扁平 `*_cli` 模块的同样扫描):

```python
names += sorted(
    f"bfdiag.{name}"
    for _, name, is_pkg in pkgutil.iter_modules(bfdiag.__path__)
    if not is_pkg and name.endswith("_cli")
)
```

在此之前,`bfdiag/det_cli.py` 可以独立使用:`python -m bfdiag.det_cli
determinism show` / `... determinism env --deterministic --force-sync`
都能直接跑(标准 `_build_standalone_parser()` 模式,和
`bfdiag/trace/cli.py` 一致),`register()` 本身也完全可测(见
`tests/test_bfdiag_determinism.py::TestDetCli`,挂到一个 scratch parser
上跑)。

## 一个小的、有意添加的东西:`QSR_SEED`

任务清单里"固定所有 RNG seed"这一项本身没有指定 seed 从哪来。为了让
`seed_all` 这一项也能通过环境变量标准化设置(呼应整个任务"标准开关"的
主题),加了 `QSR_SEED`(默认 `"0"`),`apply(seed=None)` 时会读它。这不是
任务原文列出的环境变量,是为了让 bundle 内部自洽而加的最小扩展,如果不需要
可以去掉(`apply()`/`bf determinism env` 里都只有一处引用)。

## 关于本次 worktree 状态的说明

任务开始时被分配的 worktree(`.claude/worktrees/agent-a4f5db1e97a097228`)
落后本地 `main` 24 个提交、完全没有 `bfdiag/`、`docs/diagnostics-guide.md`
等文件(bfdiag 整个平台是后来通过一个 `bfdiag-integration` 分支合并进
`main` 的)。发现问题后停下来问了协调者,确认"不要合并 main"specifically
指"不要把我的成果合回 main",不是"不能把 main 同步进我的 worktree"。之后
确认本次改动实际落地的 worktree 是 `.claude/worktrees/bfdiag-integration`
(与 `main` 同点,`bfdiag/` 完整),本文档及全部代码改动都在这个 worktree
的这个分支上完成。

## 需要 GPU 才能验证的待办清单

以下每一项在本沙盒里**只做了逻辑验证(CPU + monkeypatch 打桩)**,从未在
真实 GPU 上跑过,按"首次运行"对待:

1. **`QSR_FORCE_SYNC=1` 的端到端效果**:在真实 DFlash 解码循环上打开它,
   确认 (a) 阶段耗时确实变化(尤其是 CPU 兜底路径,如果曾经走到过);
   (b) 如果人为制造一个异步 CUDA 错误(比如故意让某个 kernel 越界),确认
   错误确实在对应 mark() 调用处报出,而不是几个 kernel 之后。
2. **`torch.use_deterministic_algorithms(True)` 在真实 sm120 kernel 集合上
   的行为**:确认哪些算子会因为"没有确定性实现"而抛错(需要提前知道,不然
   会在生产路径上被 `QSR_DETERMINISTIC=1` 意外炸掉)。
3. **`CUBLAS_WORKSPACE_CONFIG` 的实际生效时机**:验证"必须在第一次
   CUDA/cuBLAS 初始化前设置"这个判断在这个进程的实际启动顺序里成立
   ——比如 `runtime/backends/laguna_sparkinfer_moe.py` 或其他模块是否在
   `apply()` 有机会跑之前就已经触发了 cuBLAS 初始化。
4. **`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` 的具体读取时机**:这个沙盒没装
   vllm,只能通过 `benchmarks/*.py` 里"在 import vllm 之前设置"的既有约定
   反推它是 load-time 的,没有在真实 vllm/flashinfer 源码里确认具体在哪一行
   被读取、是否可以运行中热切换。
5. **`QSR_DECODE_CUDA_GRAPH=0` 等三个变量配合 `apply()` 使用的实际时序**:
   确认"只要在引擎构造前调用 `apply(disable_cuda_graph=True)` 就有效"这个
   假设成立 —— 比如写一个真实脚本,在 import `runtime.backends.laguna` 之前
   调用 `bfdiag.determinism.apply(deterministic=True, disable_cuda_graph=True)`,
   确认解码路径确实变成 eager。
6. **和用户手写的 `QSR_DEBUG_FORCE_SYNC` 做一次对照**:在同一个真实工作负载
   上分别开 `QSR_DEBUG_FORCE_SYNC=1`(用户手写版)和 `QSR_FORCE_SYNC=1`
   (本次交付的标准版),确认两者观测到的现象一致(尤其是错误定位能力),
   确认无误后可以把 `laguna_dflash.py` 里那段手写代码删掉。
7. **`sparkinfer_moe_deterministic_output` 观测项与真实 DFlash 接受率的
   交叉验证**:确认关掉它(`SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=0`,
   仅用于验证目的,不作为默认值改动)确实会像 989723d 的 commit message
   描述的那样带来 run-to-run 的数值差异,而开着确实稳定。

## 文件清单

| 文件 | 改动类型 |
|---|---|
| `bfdiag/determinism.py` | 新增 |
| `bfdiag/det_cli.py` | 新增 |
| `bfdiag/trace/ring.py` | 追加(import + `begin_round`/`mark` 各加一次同步检查 + 文档更新) |
| `bfdiag/record/fingerprint.py` | 追加(`capture_determinism()` + `capture()` 里一行) |
| `bfdiag/record/differ.py` | 追加(`DEFAULT_COMPARABLE_FIELDS` 加一项) |
| `tests/test_bfdiag_determinism.py` | 新增(29 个测试) |
| `tests/test_bfdiag_differ.py` | 追加一行(见上方"接入 differ.py"里的说明,超出原定范围但必要) |
| `notes/2026-07-27-bfdiag-determinism-and-sync.md` | 本文件 |

## 回归验证(已跑,全绿)

```
.venv/bin/python -m pytest -q tests/test_bfdiag_determinism.py
# 29 passed

.venv/bin/python -m pytest -q tests/test_bfdiag_fingerprint.py tests/test_bfdiag_differ.py \
    tests/test_bfdiag_ring.py tests/test_bfdiag_trace.py
# 57 passed

.venv/bin/python -m pytest -q tests/test_bfdiag_*.py
# 227 passed(全部 bfdiag 测试,含 daemon/divergence/canary 等未直接改动的模块)

.venv/bin/python -m ruff check bfdiag/ tests/test_bfdiag_determinism.py
# All checks passed!

.venv/bin/python -m ruff format --check bfdiag/determinism.py bfdiag/det_cli.py \
    tests/test_bfdiag_determinism.py
# 3 files already formatted
```

（`bfdiag/trace/ring.py`、`bfdiag/record/{fingerprint,differ}.py` 三个文件
本身在改动前就有几处与当前 ruff format 版本不完全一致的既有格式,和本次
改动无关的行不予重新格式化,保持"仅加法式"改动的最小 diff。）
