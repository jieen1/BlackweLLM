# bfdiag 飞行记录仪 + 不变量断言(方案 2+3)—— 实现记录

负责人:4 个并行 bfdiag agent 之一,只碰 `bfdiag/trace/*`、`bfdiag/invariants/*`、
`bfdiag/__init__.py`、三个集成点(`runtime/backends/laguna_dflash.py`、
`runtime/backends/laguna.py`、`runtime/block_pool.py`)、`tests/test_bfdiag_*.py`
/ `tests/test_invariants.py`,以及本文件。全程未使用 GPU(见下文"约束遵守情况"),
未加载模型,未跑 `benchmarks/`。

## 0. 环境现实先说明(与规格假设冲突之处)

- **numpy 不在本仓库的开发依赖里**(`pyproject.toml`:`numpy` 只在 `[cuda]`
  extra 里,`[dev]` 只有 `pytest`/`ruff`)。这个 sandbox 里 `python -c "import
  numpy"` 直接 `ModuleNotFoundError`。规格里写的"用 numpy structured array
  **或**多列 int64/float64 数组"——我选了后者,用标准库 `array.array`
  (`runtime/block_pool.py` 已经这么做了,是仓库既有的先例,不是我发明的)。
  这样 `bfdiag/trace/ring.py` 在没有 numpy 的环境里也能 import/单测。
- **torch 在本 sandbox 里也没装**。`runtime/backends/laguna_dflash.py`/
  `laguna.py` 顶部都是硬 `import torch`,所以这两个文件本身在本环境**无法
  被真正 import 执行**——这是仓库既有状态(`tests/test_dflash_engine.py`
  等文件早就用 `pytest.importorskip("torch")` 应对,不是我引入的问题)。
  结果:我对这两个文件的集成 hook **无法在本 sandbox 里跑起来验证**,只能
  通过仔细读代码 + `ruff check`(纯静态分析,不需要真的能 import torch)+
  人工核对行号来保证正确性。这是"需要 GPU/完整依赖才能验证"清单里最大的
  一项,见第 6 节。
- `runtime/backends/laguna.py` 在我碰它之前就已经有 **8 处与本任务无关的
  既存 ruff 违规**(I001 导入未排序 ×3、E501 行过长 ×1、F401 未用导入 ×2、
  F841 未用局部变量 ×2;全仓库基线共 45 处)。用 `git stash` 验证过:这些
  违规在我的改动之前就存在,我的一行 import 插入没有增加新违规、也没有
  改变违规数量,只是让其中一处的行号 +1(逐行 diff 验证过,附在下面)。
  按照"集成点改动要小、避免和其他 agent 冲突"的要求,我没有顺手把它们修掉
  ——修复这些和 bfdiag 无关,应该留给 `laguna.py` 的实际 owner。
  `python -m ruff check .` 因此**不是全绿**,但这是既存状态,不是本次改动
  引入的(全仓库改动前后错误数都是 45,用 diff 逐行核对过一致)。
  `python -m pytest -q` 同理:改动前后都是同样 2 个既存失败(`test_
  regression_unit.py::test_default_max_tokens_16384` 和 `test_vllm_
  dependency_boundary.py::test_vllm_direct_imports_are_an_explicit_migration_
  ledger`,根因都是"torch 未安装"/既有 vLLM 迁移台账缺项,与 bfdiag 无关)。
  bfdiag 自己的 47 个新测试(`test_bfdiag_ring.py`/`test_bfdiag_trace.py`/
  `test_invariants.py`)全部通过,`bfdiag/` 目录 `ruff check` 全绿。

## 1. 事件 schema(`bfdiag/trace/events.py`)

一轮记录的字段(逐字对应任务要求),每个字段的代码依据:

| 字段 | 类型 | 代码依据 |
|---|---|---|
| `round_idx` | int | **不是**引擎自己维护的——`DFlashEngine`/`LagunaBackend` 都没有轮次计数器,由 `RoundRing` 自己在 `begin_round` 里单调递增赋值 |
| `slot` | int | `dflash_round(self, slot, ...)` 的参数本身 |
| `kv_len_before` | int | `backend.slot_kv_len[slot]`,`laguna_dflash.py:1306` |
| `path` | str(`cg_replay`/`eager`/`cg_miss`) | 见下方"path/cg_miss_reason 的真实语义" |
| `cg_miss_reason` | str | 同上 |
| `draft_tokens_n` | int | `len(draft_tokens)`,即 K(=15,`NUM_SPECULATIVE_TOKENS`) |
| `accepted_n` | int | `_verify_only_accept_reject` 返回的 `decision["num_accepted"]`(`runtime/mtp_accept.py::determine_accept_reject_from_predictions`,0..K,只数匹配上的 draft token,不含 recovery/bonus) |
| `reject_position` | int | `decision["rejected_at"]`,`None`(全部接受)映射为 `-1` —— **本任务最重要的字段**,把 `STATUS_dflash_acceptance.md`/`2026-07-27-acceptance-rate-gap-vllm-vs-ours-same-prompt.md` 里那个单一聚合数字 0.687 变成"第几个 token 被拒"的分布 |
| `bonus_token` | int | `decision["next_anchor"]`,下一轮的 anchor |
| `mem_allocated` | int | `torch.cuda.memory_allocated()`(仅在 torch+CUDA 都可用时;否则恒为 0,CPU 单测/torch 未装两种情况都覆盖) |
| `t_main_forward`/`t_draft`/`t_verify`/`t_commit`/`t_round` | float(ms) | 见下方"计时" |

### `path`/`cg_miss_reason` 的真实语义(读代码得到,不是编的)

DFlash 的 `_verify_cg`/`_draft_cg` 是**引擎构造时一次性**捕获的
(`_init_cuda_graph`,`laguna_dflash.py`),之后**每一轮**要么一直走
`cg_replay`,要么一直走 `eager`(捕获失败,或 `QSR_VERIFY_CUDA_GRAPH=0`
关掉了 verify CG)——**DFlash 的轮次本身没有"这一轮动态判定不走 CG"的分支**。
真正动态的"这一轮判定不走 CG"分支在 `LagunaBackend._decode_cg_batch_eligible`
(`laguna.py:1382`,给 `decode_batch_sampled` 用):batch size 和捕获时的
`decode_cg.batch_size` 不严格相等、或请求要 logprobs、或有非贪心采样,就
判定不可用,这一次调用**动态地**掉回 eager——这正是
`notes/2026-07-27-dflash-concurrency-handoff.md` 里"capacity>1 时 batch
size 对不上 CUDA Graph,静默掉回一条会崩的 eager 路径"那个 bug 类。所以:

- DFlash 轮(`dflash_round`)记录的 `path` 只会是 `cg_replay`/`eager`(静态,
  取决于 `self._verify_cg is not None`),`cg_miss_reason` 在 eager 时是
  `cuda_graph_disabled`(`self._use_cuda_graph` 为假)或 `cg_unavailable`
  (CG 本该启用但捕获失败/env 关闭——当前代码在这两种情况下都只是把
  `self._verify_cg`/`self._draft_cg` 设为 `None`,**没有留痕**具体是哪一种,
  所以这里只能报一个粗粒度原因;区分"捕获异常"和"env 关闭"需要在
  `_capture_verify_cg`/`_init_cuda_graph` 里加一个新属性才能做到,我没有加
  ——不在"极小加法式改动"的预算内,留作 TODO)。
- `decode_batch_sampled` 的每个 slot 记录的 `path` 会真的出现 `cg_miss`
  (动态),`cg_miss_reason` 是 `batch_size_mismatch`/`non_greedy`/
  `logprobs_requested` 三者之一,直接对应 `_decode_cg_batch_eligible` 的
  三个条件。

### T0/T1/T2 分层探测路线图(留好扩展位,本次只实现 T0)

协调者中途补充了一个需求:这套事件环是更大的分层探测系统的第一层,后续
会加 T1(每张量归约签名:absmax/L2/mean/NaN+Inf 计数,~32B/张量,48 层
≈1.5KB/轮)和 T2(全量张量,GPU 侧环,device→device memcpy;按模型实测
参数 48 层/hidden 3072/bf16/256 专家 top-10,一轮全层 hidden ≈4.6MB,D2D
拷贝约 3µs,占 44.16ms 一轮的 0.007%)。为了不让以后加 T1/T2 时被迫重写
schema,这次在 `events.py` 里留了三个字段(`RoundEvent` 上,默认值
`site_id=0`/`tier=0`/`payload_ref=""`,T0 事件全部留默认值不变):

- `site_id`:探针点 id(索引到一份版本化的探针表,语义/名称/单位由**离线**
  的表还原,热路径不格式化字符串——NanoLog 的思路)。
- `tier`:0/1/2,标记这条记录是哪一层探测产生的。
- `payload_ref`:不定长/非标量负载的"指针"(长度+偏移,可能指向另一个
  tier 后端自己的存储),T0 用不到,留空字符串。
- `SCHEMA_VERSION = 1` 常量 + `RoundEvent.from_dict` 改成"缺字段就用
  dataclass 默认值补,而不是 `KeyError`"——这样以后加字段不会让旧
  `trace.jsonl` 读不出来。

`ring.py` 的存储本来就是 `self._cols: dict[str, array]`(名字 -> 类数组的
映射,不是写死的定长列),文档里补充说明了这个形状已经足以让未来的 T2
换一种"类数组"实现(比如 GPU 常驻缓冲区)而不用改 `begin_round`/`mark`/
`finish_round` 这套热路径调用面。这次**没有**新增任何列或改变热路径行为
——纯文档 + `RoundEvent` 的三个默认字段,零风险。

"环要按预触发缓冲设计,异常发生时冻结环而不是打开抓取"、"写满覆盖最旧、
绝不阻塞、丢弃必须靠单调 seq 号可检测并显式报告 dropped=N,绝不静默丢数据"
——这两条约束**当前 T0 设计已经满足**,不需要额外改动:

- 环从来不需要"触发后开始抓"——它一直在记录,任何时刻 `snapshot()`/违反
  不变量时读到的最近 N 条,天然就是异常发生前的窗口(预触发语义免费获得)。
- `round_idx` 是 `RoundRing.begin_round` 里的单调计数器,**跨物理行复用
  依然递增**,所以哪怕环绕写覆盖了旧数据,最早幸存行的 `round_idx` 就是
  被覆盖掉的轮数——这次新增了 `panel.RunStats.dropped` 字段(算法就是
  "最早幸存行的 round_idx",环没绕过就是 0),`bf trace show` 的文本面板
  和 `--json` 都会报告,不会静默丢数据(见 `bfdiag/trace/panel.py`,测试见
  `tests/test_bfdiag_trace.py::TestPanelStats::test_dropped_count_reflects_
  ring_wraparound`)。

## 2. 环形缓冲区设计(`bfdiag/trace/ring.py`)

`RoundRing(capacity)`:

- 存储是一个 `dict[str, array.array]`(`_cols`),数值列全部预分配、定长:
  int64 列(`round_idx`/`slot`/`kv_len_before`/`draft_tokens_n`/
  `accepted_n`/`reject_position`/`bonus_token`/`mem_allocated`)用
  `array('q', ...)`,int8 列(`path`/`cg_miss_reason`/`valid`)用
  `array('b', ...)`。另有 `_phase_codes`(`array('b', capacity*5)`,记录每
  轮实际发生的 phase 顺序)和 `_mark_count`(`array('b', capacity)`)。
- `begin_round(slot, kv_len_before) -> row`:写 4 个标量到预分配数组、调
  `Timeline.begin`(见第 3 节)、返回物理行号。**没有** heap 分配,没有
  dict/f-string,没有 GPU 同步。
- `mark(row, phase)`:往 `_phase_codes` 追加一个 phase 码 + 调
  `Timeline.mark`。同样零分配零同步。
- `finish_round(row, phase, *, path, cg_miss_reason, draft_tokens_n,
  accepted_n, reject_position, bonus_token, mem_allocated=0)`:再 mark 一次
  + 把这轮的最终字段写进数值列、`valid[row]=1`。
- **环绕语义**:`_cursor` 到 `capacity` 就回卷到 0,直接覆盖最旧的物理行
  ——这是 ring buffer 的本意,不是 bug。`round_idx` 独立于物理行、永远
  递增,`snapshot()` 只返回 `valid=1` 的行、按 `round_idx` 排序(dump 时
  才做,不是热路径)。
- **计时和数值分开**:`RoundEvent` 的 5 个耗时字段不是环里的普通列,是
  dump 时才从 `Timeline`(第 3 节)按"同一行里实际发生的 mark 顺序"算出的
  相邻差值——`dflash_round` 是 verify→commit→draft 顺序,老的
  `speculative_decode_step` 是 main_forward→draft→verify→commit 顺序,两种
  调用顺序都能正确处理(按 `_phase_codes` 里记录的**实际**先后顺序算相邻差,
  不是假设一个固定的 canonical 顺序)。

## 3. 计时(`bfdiag/trace/timing.py`)

`Timeline(capacity, marks_per_round, use_cuda=None)`:

- `use_cuda=None`(默认)时自动探测 `torch is not None and torch.cuda.
  is_available()`;单测/CPU 环境**必须显式传 `use_cuda=False`**,绝不依赖
  自动探测真的跑到 CUDA 检测那一行(本任务的所有单测都这样做,零 CUDA
  探测)。
- CUDA 后端:预分配 `capacity * marks_per_round` 个 `torch.cuda.Event
  (enable_timing=True)`,`record()`(热路径,不 sync)。
- CPU fallback 后端:预分配一个 `array('d', ...)` 扁平数组,`time.
  perf_counter()` 写入(热路径,不阻塞)。
- `resolve_deltas_ms(row, count)`:**只在 dump 时调用**,CUDA 后端这里才
  `event.synchronize()`(只同步"这一行最后一个 mark"对应的 event,不是
  全局 `torch.cuda.synchronize()`),然后 `elapsed_time` 算相邻差值。
  CPU 后端直接算 `(t[i+1]-t[i])*1000`。

## 4. 零开销怎么保证(`QSR_TRACE=0` 默认值)

- `TRACE_ENABLED`/`RING_SIZE`/`RUN_ID`/`BFDIAG_DIR`/`RUN_DIR`/`TRACE_PATH`
  都是模块级常量,**进程启动时读一次环境变量**,之后就是普通 Python bool/
  int/Path,没有运行时重复读 env 的开销。
- `_ring: RoundRing | None = RoundRing(RING_SIZE) if TRACE_ENABLED else
  None`——关闭时**根本不构造**环形缓冲区(也就不会触发 `Timeline` 的自动
  CUDA 探测),不只是"构造了但不用"。
- 每个集成点的写法统一是 `if bfdiag_trace.TRACE_ENABLED: <调用>`——关闭时
  这是一次全局 bool 的真值判断(`LOAD_GLOBAL` + `POP_JUMP_IF_FALSE`),**不
  会**先调用一个函数、函数内部再判断(那样即使关闭也要付函数调用开销)。
- `tests/test_bfdiag_ring.py::TestDisabledPathOverhead::test_disabled_
  round_overhead_under_100ns` 用 `timeit.repeat`(取 5 次里的最小值,标准
  的去噪手法)量了**一整轮**(begin + 2 次 mark + finish 共 4 个 hook 调用
  点,和 `dflash_round` 真实集成点数量一致)在关闭状态下的开销,断言
  `< 100ns`。本地实测约 **30ns**(3 倍余量),详见该测试。

## 5. `bf trace show` / `bf trace diff`

- `bfdiag/trace/panel.py::compute_stats` 算:acceptance rate(只统计
  `draft_tokens_n>0` 的 DFlash 轮,分子分母都是跨轮求和,不是逐轮平均)、
  `reject_position` 直方图(`-1` 桶 = 全部接受)、`path_counts`/`cg_hit_
  rate`、eager/cg_miss 的 `cg_miss_reason` 分布、每个 phase 的 p50/p99
  (只统计该 phase 真的跑过的轮,`>0.0` 过滤掉恒为 0 的 N/A 阶段)、
  outlier 检测(基于 `t_round` 的 MAD 稳健 z-score,同时要求绝对值
  `>=50ms` 才算 outlier——避免把"本来就很快的跑里 2ms 波动"误报,同时
  能命中"270 秒诡异延迟"这种量级的真实异常)、`dropped`(环绕丢弃计数,
  见第 1 节)。
- `render_round_table`/`render_summary`/`render_json` 是纯字符串/dict
  渲染函数,`cli.py::_cmd_show` 把它们接到 argparse。
- `bf trace diff A B`:`panel.diff_traces` 按物理位置对齐两条 trace(不是
  按 `round_idx`,因为 `round_idx` 只在单次 run 内唯一),比较**结构性
  字段**(`slot`/`kv_len_before`/`path`/`cg_miss_reason`/`draft_tokens_n`/
  `accepted_n`/`reject_position`/`bonus_token`,故意排除计时字段——计时
  字段本来就每次跑都不同,拿来 diff 永远不会收敛),报告第一个分叉的轮次
  和分叉字段。
- `bfdiag/trace/cli.py::register(subparsers)` 挂载 `trace show`/`trace
  diff`,用标准的 `set_defaults(func=...)` 模式(不知道真正的 `bfdiag/
  cli.py` dispatcher 具体怎么调用 `args.func`,但这是 argparse 多级子命令
  最通行的写法)。`if __name__ == "__main__":` 自测入口:
  `python -m bfdiag.trace.cli trace show <run_id> [--json] [--bfdiag-dir DIR]`
  和 `python -m bfdiag.trace.cli trace diff A B`。

## 6. 集成点清单(文件:行号,均为本次改动后的最终行号)

### `runtime/backends/laguna_dflash.py`

- `32-34`:`import bfdiag.invariants.checks as bfdiag_checks` /
  `bfdiag.trace.events as bfdiag_events` / `bfdiag.trace.ring as
  bfdiag_trace`。
- `1262`:`dflash_prefill_bootstrap` 里,`aux_offset = prompt_len -
  aux_len` 算完之后,`check_aux_hidden_alignment` 不变量(见第 7 节)。
- `1307`:`dflash_round` 入口,`kv_len = backend.slot_kv_len[slot]` 之后,
  `begin_round`(三元表达式,关闭时零调用)。
- `1316-1317`:verify(CG 或 eager)算完之后,mark `PHASE_VERIFY`。
- `1324`:`context_count`/`committed` 算出之后,`check_accepted_bound`
  不变量。
- `1358-1361`:`slot_kv_len`/`slot_committed_tokens` 更新之后,
  `check_kv_len_monotonic` + `check_committed_ahead_of_kv_by_one` 两个
  不变量(后者 2026-07-27 经协调者复审改名+改公式,见第 7 节开头的勘误)+
  mark `PHASE_COMMIT`。
- `1370-1378`:draft-for-next-round 算完之后(这是这一轮的最后一步),
  `finish_dflash_round`(把 `_verify_only_accept_reject` 的 decision dict
  翻译成环的数值字段,一次调用)。

**行为保证**:`QSR_TRACE=0` 且 `QSR_ASSERT_LEVEL=0` 时,`check_*` 调用是
`registry.check` 第一行 `if level > ASSERT_LEVEL: return`(级别不够直接
返回,不构造消息/不读环),trace 调用整体在 `if bfdiag_trace.TRACE_ENABLED:`
里——两者都不改变 `dflash_round` 原有的控制流/返回值,逐位相同。

### `runtime/backends/laguna.py`

- `22`:`from bfdiag.trace import ring as bfdiag_trace`。
- `1437-1442`:`decode_batch_sampled` 里,原来的
  `if self._decode_cg_batch_eligible(...):` 改成先存进 `_bf_cg_ok`
  变量(**计算过程与改动前完全一致**,只是把表达式的值存下来而不是直接
  用在 `if` 里),再 `if bfdiag_trace.TRACE_ENABLED: record_decode_batch_
  path(...)`,最后 `if _bf_cg_ok:`(和原来的 `if self._decode_cg_batch_
  eligible(...):` 分支走向完全一样)。这是"capacity>1 静默掉回 eager"
  那个 bug 类的直接观测点。

### `runtime/block_pool.py`

- `15`:`from bfdiag.invariants import checks as bfdiag_checks`。
- `403`:`BlockPool.allocate` 返回 `ids` 之前,`check_no_duplicate_ids`
  不变量。

## 7. 不变量清单(`bfdiag/invariants/checks.py`)及代码依据

API:`bfdiag.invariants.registry.check(level, name, cond, **ctx)`,
`QSR_ASSERT_LEVEL`(0 默认关 / 1 便宜 / 2 也含较贵)。违反抛
`InvariantViolation`,消息里带 `name`/`level`/`ctx` + 若 `QSR_TRACE=1`
则附最近 10 条 trace 事件(`registry._recent_trace_context`,读
`bfdiag.trace.ring.get_ring().snapshot()` 的尾部)——让报错本身是一份
小型现场报告。

### 勘误(2026-07-27,协调者复审发现,已修复)

第一版把 `slot_kv_len[slot]` 和 `len(slot_committed_tokens[slot])` 的关系
写成了**相等**(`check_kv_len_matches_committed`,断言 `kv_len ==
committed_len`)。这是**假的**——真实关系是 `committed_len == kv_len + 1`,
错误版本会在 `QSR_ASSERT_LEVEL>=1` 时**每一轮都抛** `InvariantViolation`,
比完全没有这个断言还糟(会让人误以为引擎有簿记 bug)。反证(均为真实代码,
`runtime/backends/laguna.py`):`prefill`(1123-1124 行)、`prefill_sampled`
(1137-1138)、`prefill_with_aux`(1269-1270,DFlash 实际走的 prefill 路径)、
增量续算路径(1709)、chunked 变体(1814-1815)—— 5 处结构完全一致:

```python
self.slot_kv_len[slot] = len(prompt_ids)  # 或 prompt_len,同一个数
self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
```

`first_token` 是刚从 prompt 的 forward 里采样出来的**输出**,还没有作为
输入跑过 forward,KV 里没有它的位置——所以 prefill 一结束就是 `committed_
len == kv_len + 1`,并且这个 +1 之后每一步都同步保持(`dflash_round`:
`slot_kv_len[slot] += context_count`,`context_count = 1 + num_accepted`,
`len(committed) == context_count`,两边增量相同;`decode_batch_sampled`
同理,两边都 +1)。已改名为 `check_committed_ahead_of_kv_by_one`,断言
改成 `committed_len == kv_len + 1`,docstring 里写清楚了这 5 处代码引用。
`tests/test_invariants.py::TestRealCodeRegression` 新增了专门的回归测试
(用真实公式复现 prefill/round 后的状态,并直接证明"旧断言在这个真实状态
上就是 False"),防止以后又被错改回相等。

同一次复审顺带重新推导了其余 5 个不变量(判据:能不能找到真实代码路径
构成反例),额外发现并修了一处**假阳性防护但零判别力**的问题:

- `check_aux_hidden_alignment` 原来断言 `aux_offset >= 0 and aux_offset +
  aux_len == prompt_len` 两个子句。第二个子句是 `aux_offset = prompt_len -
  aux_len`(调用点已经这么算出 `aux_offset`)这个定义式的**同义反复**——
  代入定义回去,`(prompt_len - aux_len) + aux_len == prompt_len` 对任意
  整数恒成立,合法/非法输入都一样,**不提供任何判别力**,只是看起来像是
  在多验证一件事。已删掉这个子句,只保留真正有意义的 `aux_offset >= 0`
  (等价于 `aux_len <= prompt_len`)。`tests/test_invariants.py::
  TestRealCodeRegression::test_aux_offset_plus_aux_len_equals_prompt_len_
  is_tautological` 直接演算证明了这一点。
- `check_no_duplicate_ids` 的 docstring 补充澄清了"这次调用内唯一"和
  "全局/跨调用唯一"是两个不同命题(协调者点名要求确认的一点)——这个
  函数**只**验证前者(`BlockPool.allocate(n)` 一次调用返回的 `ids` 内部
  不重复);跨调用/跨 slot 的别名问题由 `allocate`/`free` 自己的硬编码
  `RuntimeError`(`ref_cnt` 检查)负责,不是这个不变量的职责,新测试
  `test_no_duplicate_ids_does_not_check_across_separate_calls` 把这个
  范围边界显式测出来了。
- `check_accepted_bound`(协调者点 4 要求确认的参数命名)——`k` 传的是
  `len(draft_tokens)`,和 docstring 里的 `K` 一致,**确认无误,未改**。
- `check_kv_len_monotonic`——`context_count = 1 + num_accepted >= 1`
  在 `dflash_round` 里恒成立,`new_kv_len >= prev_kv_len` 这个断言本身
  没有反例(它比真实情况"严格递增"更宽松,但宽松的方向是安全的,不会
  漏判——只是没有用严格 `>` 那么"贴合"),**确认无误,未改**。
- `check_cg_replay_slot_consistency`——**未接入任何集成点**,原因不变
  (见下方说明),纯函数行为本身没有反例,**确认无误,未改**。

| 函数 | level | 依据 |
|---|---|---|
| `check_committed_ahead_of_kv_by_one` | 1 | 见上方勘误:`committed_len == kv_len + 1`(不是相等!),`laguna.py` 5 处 prefill 变体 + `dflash_round`/`decode_batch_sampled` 每轮同步推进保持这个 +1 offset |
| `check_accepted_bound` | 1 | `determine_accept_reject_from_predictions` 的循环只跑 `range(k)`,`num_accepted` 天然 0..k,`len(committed) == num_accepted+1 <= k+1`(实际上恒等于,`<=` 是保守写法,按原任务要求的形式) |
| `check_no_duplicate_ids` | 1 | 仅"这次调用内唯一"(见上方勘误),`BlockPool.allocate` 每次从空闲队列 `popleft` 一个块再从队列摘掉,同一次调用内不该有重复 id |
| `check_kv_len_monotonic` | 1 | `dflash_round` 的 `context_count = 1 + num_accepted >= 1`,一轮内 `kv_len` 只增不减(重置是走独立的 `reset_slot`,不在轮内发生) |
| `check_aux_hidden_alignment` | 2 | 见上方勘误:只断言 `aux_offset >= 0`(`dflash_prefill_bootstrap` 里 `aux_offset = prompt_len - aux_len`,若 `aux_len > prompt_len` 会算出负 offset,写 draft KV 时环形位置全错) |
| `check_page_table_covers_seqlen` | 1 | **写了但没有接入任何集成点(不许碰该文件)**——见第 10 节,含真实的、当前生产配置下潜伏(未触发)的 bug |
| `check_cg_replay_slot_consistency` | 2 | **写了但没有接入任何集成点**——见下方说明 |

`check_cg_replay_slot_consistency(slot, replay_slot)` 没有被接入的原因:
DFlash 的 CG replay(`_fill_buffers`,`laguna_dflash_cudagraph.py`,**不在
本任务文件清单里**)现在每次 replay 都从传入的 `slot` 参数重新算物理地址
(commit `30675d2` "Fix CG binding address caching: 64K DFlash acceptance
19%→86%" 修的就是"不重新算、用了缓存的旧物理地址"这个真实 bug)。在
`dflash_round` 里,传给 `.replay(slot, ...)` 的 `slot` 就是函数自己的
`slot` 参数,本地对比"两个同一个变量是否相等"没有任何保护意义,加了也是
摆设。真正端到端验证"CG 内部 buffer 指向的物理地址 == 当前 slot 的真实
物理地址"需要读 GPU 上 CG 自己的 buffer 状态,这属于第 8 节的 GPU-only
待办,也需要改 `laguna_dflash_cudagraph.py`(不在本任务文件清单)才能真正
挂上这个检查点。函数本身连同这段说明留在 `checks.py` 里,`tests/test_
invariants.py` 单独测试它作为纯函数的行为正确。

## 8. 需要 GPU / 完整依赖才能验证的待办清单

1. **集成 hook 的真实行为从未在真实引擎里跑过**(本 sandbox 没装 torch,
   `laguna_dflash.py`/`laguna.py` 根本 import 不起来)。需要在有
   torch+CUDA+模型权重的环境里:
   - `QSR_TRACE=0 QSR_ASSERT_LEVEL=0` 跑一段现有的 DFlash/Laguna 测试或
     benchmark,确认行为/输出与改动前**逐位相同**(这是硬性要求,我只能
     通过读代码保证,没法真正跑一遍对比)。
   - `QSR_TRACE=1` 跑一轮真实 DFlash 解码,`bf trace show <run_id>` 看
     `trace.jsonl` 是否产出合理事件(`path`/`reject_position`/耗时是否
     符合预期量级)。
   - `QSR_TRACE=1` 时确认 CUDA event 池(`Timeline` 的 `use_cuda=True`
     分支)真的能跑:`torch.cuda.Event(enable_timing=True)` 的批量预分配
     (默认 ring size 8192 × 5 marks = 40960 个 event)在实际显存/驱动上
     会不会有可观测的开销或失败,本 sandbox 完全没法测。
2. **`decode_batch_sampled` 的 `record_decode_batch_path` hook** 需要真的
   构造出 `capacity>1` 且 batch size 与 decode CG 不匹配的场景,确认
   `cg_miss`/`batch_size_mismatch` 真的被触发并正确记录——目前只在
   `tests/test_bfdiag_ring.py` 里用假的 `_Params`/`object()` 单测了
   `record_decode_batch_path` 函数本身的分支逻辑,没有验证真实调用点。
3. **`270 秒诡异延迟`之谜** —— 有了 `bf trace show` 的 outlier 检测
   之后,下次真复现时应该能在 `t_round` 时间线上直接跳出来,但目前没有
   真实 trace 数据验证这一点(`notes/2026-07-27-dflash-concurrency-
   handoff.md` 提到的那次复现没有留下 trace.jsonl)。
4. **`check_cg_replay_slot_consistency` 的真正意义**——如第 7 节所说,
   需要能读 GPU 上 CG buffer 的物理地址状态,以及可能需要修改
   `laguna_dflash_cudagraph.py`(需要用户批准,不在本任务文件清单内)。
5. **`mem_allocated` 字段**——`torch.cuda.memory_allocated()` 在本 sandbox
   完全没测过是否会在高频调用下引入可观测开销(理论上很便宜,只读一个
   计数器,不需要同步,但没有实测数据)。
6. **CPU 单测证明的 <100ns 只是"关闭状态"下的开销**——`QSR_TRACE=1` 时
   `RoundRing`/`Timeline` 的真实每轮开销(GPU event record 的实际耗时、
   `array.array` 写入在真实高频解码循环里的耗时)需要真实 profiling 数据。
7. **`check_page_table_covers_seqlen`(第 10 节)从未在真实 CUDA Graph replay
   上跑过**——`bfdiag/invariants/checks.py` 里的检查逻辑本身已经用真实
   公式做过 CPU 反例测试(`tests/test_invariants.py`),但补丁**还没有
   被贴进 `runtime/backends/laguna_cuda_graph.py`**(按约束不许碰这个
   文件)。需要:(a) 用户/协调者把第 10 节的补丁贴上去;(b) 在真实 GPU
   上,把 `QSR_ASSERT_LEVEL=1` 打开、故意把 `window`/`block_size` 调到
   让 `min()` 真正裁剪的组合,确认 `InvariantViolation` 真的会在
   `LagunaCudaGraphVerify._fill_buffers` 触发,而不是被某个我没读到的
   上下文吞掉;(c) 确认生产配置(`QSR_ASSERT_LEVEL=1` 但用当前的
   `block_size=64`/`128`)下补丁**不会**误报(CPU 侧数值已经验证过边界
   是"恰好不裁剪",但真实 GPU 张量的 `.item()`/dtype 转换路径没有实测)。

## 9. 架构耦合:runtime 现在硬依赖 bfdiag(刻意决定,不是副作用)

`runtime/backends/laguna.py:22`(`from bfdiag.trace import ring as
bfdiag_trace`)和 `runtime/block_pool.py:15`(`from bfdiag.invariants import
checks as bfdiag_checks`)都是**模块级** import。这意味着从现在起,**生产
runtime 没有 bfdiag 包就 import 不起来**——诊断包从"旁路挂件"变成了
runtime 的硬依赖。协调者复审时点名要求把这个决定写清楚,不能是改代码时的
无意识副作用。

**这是刻意的决定,理由**:

1. 同一个仓库、同一次发布,不存在"运行时里有 runtime 没有 bfdiag"的部署
   场景——两者一起 `git clone`/一起打包,没有独立分发 bfdiag 的需求。
2. `bfdiag` 本身**零第三方依赖**(只用标准库 `array`/`json`/`argparse`/
   `dataclasses`/`enum`,可选地探测 `torch`——没装也不报错),import
   `bfdiag.trace.ring`/`bfdiag.invariants.checks` 本身不会引入任何新的
   第三方依赖链、不会拖慢 runtime 的 import 时间(纯 Python 模块级代码,
   没有重量级初始化)。
3. `QSR_TRACE=0`/`QSR_ASSERT_LEVEL=0`(默认)时,这两个模块 import 期间
   **没有任何副作用**:不构造 `RoundRing`(见 `ring.py` 的 `_ring =
   RoundRing(...) if TRACE_ENABLED else None`),不探测 CUDA,不建目录、
   不开文件。这是这个决定成立的**前提条件**,不是可有可无的优点——如果
   bfdiag 的 import 本身有副作用(哪怕只是"建个目录"),把它做成硬依赖
   就是一个更危险的决定。

**如果将来要解耦(比如 bfdiag 要支持被完全移除、或者 runtime 要单独分发给
不需要诊断功能的场景)**,不需要现在做,但路径应该是:

- **方案 A(推荐)**:runtime 侧维护一个极薄的 no-op 默认("空对象"模式)
  ——`runtime/backends/laguna.py` 顶部不再 `import bfdiag`,而是 `try:
  from bfdiag.trace import ring as bfdiag_trace; except ImportError:
  bfdiag_trace = _NoOpTrace()`(`_NoOpTrace` 只需要 `TRACE_ENABLED =
  False` 一个属性,因为所有集成点都先判断这个 flag 才调用其他方法)。
  这样 bfdiag 缺失时行为退化成"诊断功能不可用",而不是 import 失败。
- **方案 B**:把耦合方向反过来,用依赖注入——runtime 在 `LagunaBackend.
  __init__` 时接受一个可选的 `trace_hooks` 参数(默认 `None`,内部用同样
  的 `_NoOpTrace`),由**调用方**(`server/engine.py` 或 CLI 入口)决定
  要不要 `import bfdiag` 再传进去。这样 `runtime/` 包本身完全不 import
  `bfdiag`,耦合被推到组装层。比方案 A 干净,但改动面更大(所有集成点
  从"读全局单例"变成"读实例属性"),这次没有做,留作以后需要真正解耦时
  再权衡。
- **不推荐**:把 `bfdiag` 作为 `pyproject.toml` 的 optional-dependency
  ——没有意义,因为 `bfdiag` 本身就在同一个仓库里,不是外部包,"可选安装"
  这个概念不适用。

## 10. `check_page_table_covers_seqlen`:一个真实 bug 和它的接入补丁(未接线)

2026-07-27,一个排查 `block_size` 64→128 迁移的子 agent 在
`runtime/backends/laguna_cuda_graph.py` 里发现一个真实的、独立的 bug(不是
当时那次接受率下降调查的成因,但值得永久断言守住)。**用户正在实时编辑
`runtime/backends/laguna_cuda_graph.py`,本节只读该文件,不修改它** ——
这里只提供可直接照抄的补丁文本。

### bug 是什么

`LagunaCudaGraphVerify._fill_buffers`(main 模型 verify CUDA Graph,
M=16)的 SWA 分支里,`page_table` 实际填充的条目数被 `min()` 裁剪到环的
物理容量,但 `cache_seqlens` 写入的是**裁剪前**的长度:

```python
# runtime/backends/laguna_cuda_graph.py:648, 654-655(原文,未改动)
n_ring = min(-(-aligned_len // bs), self._ring_blocks_per_slot)  # 648:裁剪
pt[0, :n_ring] = (ring_base + (block_starts % ring_slots) // bs).to(pt.dtype)  # 654:只填 n_ring 条
self._cache_seqlens[group_key][0] = aligned_len  # 655:却声明未裁剪的长度
```

一旦 `cdiv(aligned_len, bs) > ring_blocks_per_slot`(窗口实际需要的页数
超过环的物理容量),`min()` 会静默截断填充数,但 attention kernel 仍然
按 `aligned_len` 声明的长度去读 `page_table`——读到的多余条目是上一次
CUDA Graph capture/replay 残留的页号,可能指向别的 slot 的 KV。**症状是
"输出看起来正常,但预测质量悄悄变差",没有崩溃可供定位。**

当前生产配置(`block_size=64` 和 `block_size=128`,`window=512`,
`qo_max=16`)刚好卡在 `cdiv(aligned_len,bs) == ring_blocks_per_slot` 的
边界上(`_ring_blocks_for_window` 的 `+1` 冗余项刚好让 `min()` 恒为
no-op),所以还没触发——见 `tests/test_invariants.py::
TestRealCodeRegression::test_page_table_covers_seqlen_at_real_production_
block_sizes` 用真实公式复现了这个边界。**只要对齐粒度/窗口大小再变一档,
就会真的触发**——见同一个测试类里的
`test_page_table_covers_seqlen_fires_when_alignment_outgrows_the_ring`。

`LagunaCudaGraphDecode._fill_buffers`/`_fill_buffers_b1`(M=1 decode CG)
的 SWA 分支,以及所有 full-attention 分支(两个类都有),**没有**这个
`min()` 裁剪,`n_blocks`/`n_ring` 就是 `cdiv(...)` 本身,天然和
`cache_seqlens` 一致——这三处不受影响,`checks.py` 的 docstring 和
`test_invariants.py` 里各有一条测试直接演算证明。

### 新增不变量:`check_page_table_covers_seqlen`

```python
def check_page_table_covers_seqlen(
    group_key: object, cache_seqlens: int, n_filled_pages: int, page_size: int,
) -> None:
    """cdiv(cache_seqlens, page_size) <= n_filled_pages"""
```

`level=1`(纯宿主端整数运算)。语义:**page_table 里当前有效的条目数**
(不是"这次调用物理写了几条"——`LagunaCudaGraphDecode` 有"`n_ring` 没变
就跳过重写"的优化,跳过时有效条目数仍然是这次调用算出的 `n_ring`,因为
跳过重写本身就是建立在"数量没变"这个前提上的)必须能覆盖
`cache_seqlens` 声明的长度对应的页数。违反时的错误消息给出
`cache_seqlens`(声明长度)、`page_size`、`pages_needed`(需要几页,
`cdiv` 算出)、`n_filled_pages`(实际填了几页)、`deficit`(差多少)——
一眼看出差在哪。完整 docstring(含每个分支的行号引用)见
`bfdiag/invariants/checks.py`。

### 反例测试

`tests/test_invariants.py::TestRealCodeRegression` 新增 4 条(风格与之前
那批一致,全部用真实公式,不 import `runtime.*` 以保持无 torch 也能跑):

1. `test_page_table_covers_seqlen_at_real_production_block_sizes`
   (parametrize `block_size` in `[64, 128]`)——用真实
   `_ring_blocks_for_window` 公式算出生产配置下的边界值,断言**不触发**
   (数值复现"刚好卡在边界"这句观察,不只是断言这句话)。
2. `test_page_table_covers_seqlen_fires_when_alignment_outgrows_the_ring`
   ——构造 `block_size=128`、需要 7 页但环只有 6 页容量的场景,断言
   **触发**,并检查错误消息里 800/7/6/1(声明长度/需要页数/实际填充/
   差额)这几个数字都在。
3. `test_page_table_covers_seqlen_full_attention_never_clips` —— 遍历
   多个 `new_kv` 值,断言 full-attention 路径永不触发。
4. `test_page_table_covers_seqlen_swa_decode_class_never_clips_either`
   —— 遍历多个 `kv_len` 值,断言 `LagunaCudaGraphDecode`(区别于
   `LagunaCudaGraphVerify`)的 SWA 分支永不触发。

`python -m pytest -q tests/test_invariants.py`:27 个用例全过(用
`.venv/bin/python`,torch 2.11 + numpy,无 vllm)。

### 接入补丁(可直接照抄,按下面顺序应用到 `runtime/backends/laguna_cuda_graph.py`)

**必需(修真实 bug)—— `LagunaCudaGraphVerify._fill_buffers`,SWA 分支**

文件顶部,`from vllm._custom_ops import reshape_and_cache_flash` 那一行
(第 14 行)之后加一个空行 + 一行 import:

```python
from vllm._custom_ops import reshape_and_cache_flash

from bfdiag.invariants import checks as bfdiag_checks
```

`_fill_buffers`(定义在第 611 行)里,第 655 行
(`self._cache_seqlens[group_key][0] = aligned_len`)之后,同缩进(16 个
空格,和第 655 行对齐)插入一行:

```python
                self._cache_seqlens[group_key][0] = aligned_len
                bfdiag_checks.check_page_table_covers_seqlen(group_key, aligned_len, n_ring, bs)
```

**可选(纵深防御,给目前"天然正确"的三个分支也上保险,防止以后被改坏)**

同一个 `_fill_buffers` 的 full-attention 分支(第 666 行之后,16 空格缩进):

```python
                self._cache_seqlens[group_key][0] = new_kv_len
                bfdiag_checks.check_page_table_covers_seqlen(group_key, new_kv_len, n_blocks_full, bs)
```

`LagunaCudaGraphDecode._fill_buffers`(定义在第 137 行,batch>1 通用路径)
—— full-attention 分支,第 172 行之后(20 空格缩进):

```python
                    self._cache_seqlens[group_key][i] = new_kv
                    bfdiag_checks.check_page_table_covers_seqlen(group_key, new_kv, n_blocks, ps)
```

同一个函数的 SWA 分支,第 195 行之后(20 空格缩进):

```python
                    self._cache_seqlens[group_key][i] = aligned_len
                    bfdiag_checks.check_page_table_covers_seqlen(group_key, aligned_len, n_ring, ps)
```

`LagunaCudaGraphDecode._fill_buffers_b1`(定义在第 200 行,batch=1 优化
路径)—— full-attention 分支,第 223 行之后(16 空格缩进):

```python
                self._cache_seqlens[group_key][0] = new_kv
                bfdiag_checks.check_page_table_covers_seqlen(group_key, new_kv, n_blocks, ps)
```

同一个函数的 SWA 分支,第 245 行之后(16 空格缩进):

```python
                self._cache_seqlens[group_key][0] = aligned_len
                bfdiag_checks.check_page_table_covers_seqlen(group_key, aligned_len, n_ring, ps)
```

以上所有插入行在 `QSR_ASSERT_LEVEL=0`(默认)时都是
`registry.check` 第一行 `if level > ASSERT_LEVEL: return` 直接返回,
不做任何计算/不读 GPU 状态,和这份 notes 其余部分的"零开销"论证是同一套
机制,不需要额外验证。行号均为本 worktree 2026-07-27 与 `main`
fast-forward 合并后的行号(`git merge --ff-only main`,合并前 HEAD
`49ec92b`,合并后 `0504a96`);如果用户那边的实时编辑已经挪动了这些行号,
以「紧跟在对应的 `self._cache_seqlens[...] = ...` 赋值语句之后、同一
缩进层级」为准,而不是死记行号。

## 11. 交付清单(文件路径)

- `bfdiag/__init__.py`、`bfdiag/trace/{__init__,events,ring,timing,dump,
  panel,cli}.py`、`bfdiag/invariants/{__init__,registry,checks}.py`
- 集成 hook:`runtime/backends/laguna_dflash.py`(+23/-0)、
  `runtime/backends/laguna.py`(+8/-1)、`runtime/block_pool.py`(+3/-0)
- 测试:`tests/test_bfdiag_ring.py`、`tests/test_bfdiag_trace.py`、
  `tests/test_invariants.py`(共 7 条不变量、三个文件合计 59 个测试,
  全通过——`check_page_table_covers_seqlen` 是第 7 条,只有测试,未接入
  任何 runtime 文件,见第 10 节的补丁)
- 本文件

验证命令(两套环境都验证过,结论一致):

```bash
# 环境 A:本 worktree 默认 python(无 torch/numpy/vllm)
python -m pytest -q tests/test_bfdiag_ring.py tests/test_bfdiag_trace.py tests/test_invariants.py
python -m pytest -q   # 279 passed, 49 skipped, 2 failed(既存、与 bfdiag 无关)
/home/bot/.venvs/vllm/bin/ruff check bfdiag/ tests/test_bfdiag_ring.py tests/test_bfdiag_trace.py tests/test_invariants.py  # All checks passed!
python -m ruff check .  # 45 errors(与改动前逐行 diff 一致,既存,非本任务引入)

# 环境 B:主仓库 .venv(torch 2.11 + numpy,无 vllm)—— check_page_table_
# covers_seqlen 那一轮改动用这套环境验证
/home/bot/project/qwen-sm120-runtime/.venv/bin/python -m pytest -q tests/test_invariants.py  # 27 passed
/home/bot/project/qwen-sm120-runtime/.venv/bin/python -m pytest -q --continue-on-collection-errors  # 602 passed(既存基线 596 passed + 本次新增 6 个测试;35 failed/2 errors 与改动前逐项比对一致,均因本环境未装 vllm,既存)
/home/bot/project/qwen-sm120-runtime/.venv/bin/ruff check bfdiag/invariants/checks.py tests/test_invariants.py  # All checks passed!

# 各模块自测(CPU-only,不碰 GPU)
python -m bfdiag.trace.events
python -m bfdiag.trace.timing
python -m bfdiag.trace.ring
python -m bfdiag.trace.dump
python -m bfdiag.trace.panel
python -m bfdiag.invariants.registry
python -m bfdiag.invariants.checks
python -m bfdiag.trace.cli trace show <run_id> --bfdiag-dir <dir>
python -m bfdiag.trace.cli trace diff <run_a> <run_b> --bfdiag-dir <dir>
```

优先级完成情况(按任务给的顺序):(a) ring+events+timing+CPU 单测——完成;
(b) dump+`bf trace show` 面板——完成;(c) 引擎集成 hook——完成(但如第 8
节所说,无法在本 sandbox 里真实验证);(d) 不变量断言——7 条里 5 条已接入
生产代码、2 条(`check_cg_replay_slot_consistency`、
`check_page_table_covers_seqlen`)写好+测好但未接入(前者原因见第 7 节,
后者的接入补丁见第 10 节,均因约束不许碰对应的 runtime 文件);(e) `bf
trace diff`——完成。
