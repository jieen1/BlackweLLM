# bfdiag 断言可信度审计（Track C0）

> 编制：2026-08-01/08-02，分支 `fix/t0b-diag`。
> 触发：`docs/roadmap.md` §1.3 N4 —— `bfdiag/checkpoint` 依赖一个已经不成立
> 的前提（`reset_slot` 会清零 KV），守护它的回归测试从不调用真实函数。

## 0. 结论先说

- **N4（reset_slot 隔离保证）：真的失效了，已修。** `bfdiag/checkpoint/restore.py`
  和 `bfdiag/daemon/session.py::reset_laguna_engine` 都依赖
  `backend.reset_slot(slot)` 清零该 slot 的 KV——但真实的 `reset_slot`
  已经被重写为**只做簿记、完全不清零任何张量**（为了给 Laguna 自己的
  per-slot 前缀缓存保留数据）。两处都已改为自己显式清零，并补了会真的调用
  真实函数、而且在修复前会真的失败的回归测试（§1、§2）。
- **比 N4 描述的更大的一个发现：两份 bfdiag 手册都声称
  `LagunaBackend.reconcile_prefix_hit` 是一个 stub("每次 admission 都是
  cold miss"),这是假的——它是真实的、被 `server/engine.py` 生产代码
  直接调用的前缀缓存实现,正是它导致 `reset_slot` 被重写。已在
  `bfdiag/checkpoint/state.py` 和 `bfdiag/daemon/session.py` 两处更正（§3）。
- **旧的 `bug_found_not_fixed` 手册条目已删除**——它描述的 bug 存在于一个
  已经被重写过*两次*的旧版本 `reset_slot` 里，现在的代码里那两个行号指向
  完全不相关的函数,而且现在的 `reset_slot` 根本没有任何张量切片代码,
  所谓的"missing leading colon"模式已经不存在（§4）。
- **系统性发现:`bfdiag/` 里大量 `code_ref` 行号引用已经漂移**（不止
  N4 涉及的那几个)——`laguna.py` 一直在长,没有任何机制去重新核实这些引用。
  这是本次审计里**除 N4 本身之外最大的一类"平台说谎"来源**,建议见 §6。
- **除了 checkpoint/session 这一簇,没有找到第二个同等级别的"断言与真实代码
  矛盾"的例子。** 其余模块的"假数据测试"基本都诚实标注了自己在测什么
  （见 §5 的调查清单)。
- **N6(全套件 flaky)**:**根因未能确定性复现**——真实 GPU+CPU 负载下反复
  跑全套件约 28 次,一次没有复现;排除了线程泄漏(用探针实测确认关键节点
  只有 MainThread 存活)、`auto_record()` 的全局 hook 污染、`QSR_TRACE`
  模块级缓存、`test_real_world.py` 等脏脚本(已被 `collect_ignore` 排除)。
  诚实结论 + 让断言在下次失败时自解释(精确到行的检查 + 失败时dump 原始
  输出和线程列表),而不是假装找到了根因,见 §7。
- **N7(`FakeEngineProvider.load` 与 Protocol 不符)**:已修,见
  `bfdiag/daemon/provider.py::FakeEngineProvider.load` 和
  `tests/test_bfdiag_daemon.py::TestEngineProviderProtocolConformance`。

---

## 1. N4:reset_slot 隔离保证失效的完整链条

### 1.1 真实代码现状（核实,非推测)

`runtime/backends/laguna.py`(本次核实时约 1975-1995 行,行号本身会漂移,
以 `def reset_slot` 为准)现在的实现:

```python
def reset_slot(self, slot: int) -> None:
    """Release slot for reuse. Preserves KV data for prefix cache. ..."""
    if self.slot_committed_tokens[slot] and self.slot_kv_len[slot] > 0:
        self._prefix_cache_tokens[slot] = list(self.slot_committed_tokens[slot])
        self._prefix_cache_kv_len[slot] = self.slot_kv_len[slot]
    self.slot_kv_len[slot] = 0
    self.slot_committed_tokens[slot] = []
    self._prefix_chunk_snapshots[slot] = None
```

不清零任何 `kv_caches` 张量。它反而把这个 slot 刚跑完的 token 历史**保存**
进 `_prefix_cache_tokens[slot]`/`_prefix_cache_kv_len[slot]`,供
`LagunaBackend.reconcile_prefix_hit`(真实的、`server/engine.py:926` 直接
调用的生产代码,不是 stub——见 §3)之后做同 slot 前缀复用。

### 1.2 两处依赖它的旧假设,以及修法

**`bfdiag/checkpoint/restore.py`**(`restore_checkpoint` 的 Step 2)原来是:

```python
backend.reset_slot(slot)                       # 假设:这会清零该 slot 的 KV
draft_start, draft_end = draft_ring_block_range(geom)
for layer_name in geom.draft_layer_names:
    engine._draft_kv_caches[layer_name][:, draft_start:draft_end].zero_()
```

`reset_slot` 不再清零任何东西之后,checkpoint 恢复的隔离保证——"这个 slot
之前的残留数据不会泄漏进恢复结果"——对 full-attention KV 这一半失效了:
`restore_checkpoint` 后面只把 checkpoint 自己 `ceil(kv_len/block_size)`
那么多 block 写回去(这是有意的体积优化,见 `state.py` 的
`full_block_range` 注释),该 slot 静态容量里**超出这个范围的旧 block**
不会被任何东西清零,继续保留着**这个 slot 上一个占用者**的真实 KV 数据。

修法(新增 `state.full_slot_block_range`,返回该 slot 静态容量的整段范围,
不随 kv_len 缩放):

```python
backend.reset_slot(slot)
full_slot_start, full_slot_end = full_slot_block_range(geom)
for layer_name in geom.full_layer_names:
    backend.kv_caches[layer_name][:, full_slot_start:full_slot_end].zero_()
draft_start, draft_end = draft_ring_block_range(geom)
for layer_name in geom.draft_layer_names:
    engine._draft_kv_caches[layer_name][:, draft_start:draft_end].zero_()
prefix_cache_tokens = getattr(backend, "_prefix_cache_tokens", None)
if prefix_cache_tokens is not None:
    prefix_cache_tokens[slot] = None
    backend._prefix_cache_kv_len[slot] = 0
```

(SWA ring 不需要额外清零——`restore_checkpoint` 本来就无条件覆盖整段 ring
容量,不管 kv_len,所以这一半从来没有这个洞。)

最后三行清的是**另一个**、更隐蔽的洞:`reset_slot` 本身会把"这个 slot
刚才在跑什么"存进 `_prefix_cache_tokens[slot]`——这是它的设计契约,不是
bug。但恢复的是**checkpoint 里的另一段对话**,如果不清掉这个刚被
`reset_slot` 填进去的、属于**上一个占用者**的条目,以后任何一次
`reconcile_prefix_hit` 都会拿这个 slot 恢复后的 KV 去匹配错误的 token
历史。

**`bfdiag/daemon/session.py::reset_laguna_engine`**(daemon 级别的"回到
刚 load() 完的干净状态",canary 自检依赖它)有完全同构的问题,同样的手法
修的:对每个 slot 显式清零 full-attention/SWA-ring KV,并清空
`_prefix_cache_tokens`/`_prefix_cache_kv_len`。

### 1.3 fake 也在说谎,而且是同一套谎

`bfdiag/checkpoint/testing.py::FakeBackend.reset_slot` 原来是这样写的:

```python
def reset_slot(self, slot: int) -> None:
    """Mirrors LagunaBackend.reset_slot (laguna.py:1639-1653) exactly:
    zero this slot's full-attention blocks and SWA ring blocks, ..."""
    ...
    for name in self._full_layer_names:
        self.kv_caches[name][:, full_start:full_end].zero_()
    if self._ring_blocks_per_slot > 0:
        ...
        self.kv_caches[name][:, ring_start:ring_end].zero_()
```

**这个 fake 精确复刻的是旧版 reset_slot 的行为,不是当前生产代码的行为。**
`bfdiag/checkpoint/`、`bfdiag/daemon/session.py` 的每一个测试,只要用到
`FakeBackend`/`reset_all`,验证的都是"如果 reset_slot 还清零 KV 会怎样",
而不是"真实 reset_slot 不清零 KV 时会怎样"——这正是 N4 描述的"回归测试从
不调用真实函数"模式的一个更大、影响面更广的实例:不是一个测试在骗人,是
**这整个测试双身(test double)本身在骗人**。

已修:`FakeBackend.reset_slot` 现在精确复刻当前真实行为(纯簿记,同时按
真实契约条件写入 `_prefix_cache_tokens`/`_prefix_cache_kv_len`,但不清
任何张量)。修完之后,`restore.py`/`reset_laguna_engine` 的显式清零就成了
**真正在被验证的行为**,不再是"反正 fake 会帮你清,你测不出少了它"。

**证明这不是纸面推理**:用 `git stash` 只回退 `restore.py`(保留已修正的
`FakeBackend`),新增的两个隔离测试立刻真实失败:

```
tests/test_bfdiag_checkpoint_restore.py::test_restore_zeros_target_slots_leftover_full_attention_blocks_beyond_checkpoint FAILED
tests/test_bfdiag_checkpoint_restore.py::test_restore_clears_stale_persistent_prefix_cache_entry_for_target_slot FAILED
```

同样手法验证了 `bfdiag/daemon/session.py`(`tests/test_bfdiag_session.py`,
新文件,之前完全没有任何测试直接调用过 `reset_laguna_engine`——所有现有
调用点测试都是 `monkeypatch.setattr` 整个函数换掉,从未真的跑过它的函数体)。

---

## 2. 新增/改写的测试(全部真机验证过"改之前会红,改之后绿")

| 文件 | 改动 |
|---|---|
| `tests/test_bfdiag_checkpoint_state.py` | 移除合成张量demo(`test_reset_slot_axis_bug_is_real_...`);新增 `full_slot_block_range` 真实单测;`test_bug_found_not_fixed_category_is_currently_empty` 替换旧的"必须恰好一条"断言 |
| `tests/test_bfdiag_checkpoint_restore.py` | 新增两条真实调用 `restore_checkpoint` 的隔离回归测试(§1.3) |
| `tests/test_bfdiag_session.py` | **新文件**——第一次真的调用 `reset_laguna_engine`(此前只有 monkeypatch 替换测试) |
| `tests/test_laguna_reset_slot_axis.py` | **删除**——纯合成张量demo,描述的 bug 已经是历史(轴错误早就修过,而且现在的 reset_slot 根本不切片) |

---

## 3. 更大的发现:两份手册都声称 reconcile_prefix_hit 是 stub

`bfdiag/checkpoint/state.py`(`SLOT_STATE_ITEMS`)和
`bfdiag/daemon/session.py`(`RESET_CHECKLIST`)**各自独立地**写着:

> "LagunaBackend.reconcile_prefix_hit 是一个显式 stub('每次 admission 都是
> cold miss'),block_pool.py/prefix_cache.py 属于 DirectModelRunner,不适用
> 于 Laguna。"

实测:

```
$ grep -rn "reconcile_prefix_hit" server/engine.py
server/engine.py:926:    hit_depths = [self.runner.reconcile_prefix_hit(p) for p in new_prompts]
```

`reconcile_prefix_hit` 是**真实的、被生产请求 admission 路径直接调用**的
函数,不是 stub。它正是 `reset_slot` 被重写(§1)的原因:`reset_slot`
有意保留 KV,专门配合这个前缀缓存做同 slot 复用。

两处手册条目都已更正(保留 `not_applicable` 分类,但理由从"这套机制不存在"
改成"这套机制是真的,只是不作为 checkpoint SAVE 内容——恢复时必须清空而
不是保存/还原,理由见 §1.2"),并加了 `state.py`/`session.py` 里对应
"CORRECTED 2026-08-02" 的说明段落。

bfdiag 自己的 daemon/canary 路径(`LagunaEngineProvider`)不构造
`ServerEngine`,所以今天并不会真的走到 `reconcile_prefix_hit`——但
§1.2 的清空修复是防御性的,不依赖这个"今天不会命中"的事实,一旦有人把
provider 接到 `ServerEngine`(或者别处复用 restore.py/session.py),这个
坑本来就在那里等着。

---

## 4. 已删除的 `bug_found_not_fixed` 条目

`state.py` 原来有一条"BUG FOUND (not fixed): reset_slot's block-range
slice hits the wrong tensor axis",引用 `laguna.py:1647,1653`。实测:

- 这两个行号现在指向 `_prefill_with_prefix_hit`(另一个函数),不是
  `reset_slot`。
- 当前的 `reset_slot`(§1.1)**根本没有任何张量切片/`.zero_()` 调用**——
  所谓"缺一个前导冒号导致切到错误维度"的模式,在当前代码里已经没有
  载体了(这个 bug 大概是在 reset_slot 被重写成"纯簿记"之前就已经修过
  的,然后 reset_slot 又被整个换了实现——手册没跟上任何一次改动)。
- `grep` 确认 `laguna.py` 里现存的每一处 `kv_caches[name][...]` 切片都用
  的是正确的 `[:, start:end]` 形式,没有第二个实例。

保留一条描述不存在的 bug 的手册,是这个平台说谎的直接来源——已删除该
条目,并把配套的合成张量回归测试(`test_reset_slot_axis_bug_is_real_...`,
`tests/test_laguna_reset_slot_axis.py`)一并移除/替换为真实调用
`restore_checkpoint` 的测试(§1.3、§2)。`bug_found_not_fixed` 这个分类
本身保留在 `_KNOWN_CATEGORIES` 里,供以后真的发现新 bug 时用;当前为空,
用 `test_bug_found_not_fixed_category_is_currently_empty` 显式记录"空
是对的,不是被遗忘"。

---

## 5. C0 全面审计清单(`bfdiag/` 其余模块)

方法:除 `checkpoint/`、`daemon/session.py`、`daemon/provider.py` 的
load/reset(已在 §1-4 深挖)外,逐模块检查"断言的是否真实
`runtime/backends/laguna*.py` 行为,还是只在合成数据上复现抽象模式"。

| 模块 | 判定 | 说明 |
|---|---|---|
| `invariants/checks.py` + `registry.py` | **real** | 6 条里 5 条(`committed_ahead_of_kv_by_one`、`accepted_bound`、`no_duplicate_ids`、`kv_len_monotonic`、`page_table_covers_seqlen`)确认接线到生产代码(`laguna_dflash.py`、`block_pool.py`、`laguna_cuda_graph.py`)。2 条(`aux_hidden_alignment`、`cg_replay_slot_consistency`)定义了但 `runtime/` 里没有调用点——这和 `docs/diagnostics-guide.md` 自己写的"暂未接线"一致,不是新发现的谎言 |
| `invariants` 的 `check_page_table_covers_seqlen` 文档字符串 | **行号漂移** | 引用的 `laguna_cuda_graph.py` 行号(641-666/648/654/655/161-247/188/237)全部对不上当前源码;bug 模式本身仍然真实存在,只是引用烂了 |
| `tests/test_invariants.py::TestRealCodeRegression` | **fake-but-legitimate** | 用真实生产常数(window=512, qo_max=16)重新推导公式,不是任意合成数字,且明确标注了自己是独立验证,不是伪装成真实调用 |
| `divergence/*` | **honest split / real** | `capture.py` 明确写"真实 GPU 采集路径写了但从未跑过";`FakeCaptureSource` 是实际被测的东西,文档没有掩盖这个事实;`scan.py`/`report.py`/`thresholds.py` 是纯逻辑,没有真假之分 |
| `sensitivity/measure.py` | **honest split** | GPU-only 函数延迟导入且明确标注未测;`derive_shape` 声称"匹配 benchmark 公式"但没有从单一源头导入——有重复漂移风险,不是造假 |
| `sensitivity/{perturbations,verdict,cycles}.py` | **real** | 纯函数,直接测自己,没有可造假的空间 |
| `shapes/attention.py`、`shapes/model.py` | **real,故意的独立复现** | 明确写"故意不 import runtime.backends.laguna*,否则交叉验证会变成 tautology"——这是刻意设计的独立公式对照,不是意外脱钩,别误标成 pattern-demo |
| `shapes/attention.py` 的 `code_ref` | **行号漂移** | 引用 `laguna.py:48-49`,实际现在在 123 行左右 |
| `shapes/{gemm,moe,harness}.py`、`trace/*`、`record/*`、`daemon/{canary,client,protocol,queue,server}.py` | **real** | 零 `runtime/laguna*` 引用,纯自包含基础设施,没有"假装测生产代码"的风险 |
| `determinism.py` | **行号漂移** | 声称"验证于 laguna.py:359",该行现在在 `SparkinferAttentionImpl` 补丁逻辑中间,不相关 |
| `checkpoint/verify.py` | **real,诚实** | 文档字符串准确:"完全不需要 runtime.*/torch,只操作 manifest/digest" |
| `workloads.py` | **real,未在 GPU 跑过(与平台已知caveat一致)** | 真实调用 `runtime.backends.laguna`/`laguna_dflash` 内部函数,不是 fake;"未上 GPU 验证"与 diagnostics-guide.md 已写的 ⚠️ 一致,不是新发现 |
| `cold_capacity.py`、`det_cli.py` | **未深挖** | grep 未发现 `runtime`/`laguna` 引用,推测是自包含(同 trace/record 一类),但没有逐行读——标注为"未核实"而非"已排除" |

**没有找到第二个 `bug_found_not_fixed` 级别的"断言与代码矛盾"**——
checkpoint/session 这一簇看起来是最严重的一处。"独立复现公式做交叉验证"
(`shapes/attention.py`、`sensitivity/measure.py` 的 `derive_shape`、
`test_invariants.py` 的 `TestRealCodeRegression`)是一种**刻意的、良好
标注**的设计模式,不应该被当成"脱钩的假测试"扣分。

---

## 6. 系统性发现:code_ref 行号大量漂移

除 §1、§4、§5 已经列出的具体实例外,这是一个仓库级别的模式:
`runtime/backends/laguna.py` 一直在快速增长(本次审计过程中亲眼验证过
好几处曾经正确、现在完全指向不相关函数的行号引用),而 `bfdiag/` 里没有
任何机制去重新核实这些引用是否还对。**没有系统性地逐条重新核实整个
`bfdiag/` 的 code_ref**(工作量太大,超出这次的时间预算),但建议:

- 优先级:能引用**符号名**(函数名/类名)的地方别引用行号——符号名可以
  `grep` 验证,行号只能靠人。这个仓库已经有这个意识的地方(比如本次新加
  的几条改成了"exact line numbers drift, re-grep before citing further"),
  可以推广成约定。
- 如果要做一次系统性重新核实,`bfdiag/checkpoint/state.py`、
  `bfdiag/daemon/session.py`、`bfdiag/invariants/checks.py`、
  `bfdiag/shapes/attention.py`、`bfdiag/determinism.py` 是已知命中率最高
  的几个文件,可以从这几个开始。

---

## 7. N6:全套件 flaky(`test_cli_ls_labels_an_unfinished_record_running`)

已排除(任务简报给出,复核无异议):
- 标签逻辑 `finished_at is None -> "running"` 与时间无关。
- `default_store()`/`bfdiag_dir()` 每次调用都重读环境变量,不是缓存 store。

排查过程中确认排除的新猜测:
- `tests/conftest.py` 已经把 `test_real_world.py`/`test_api_compat.py`/
  `test_e2e_256k_longctx.py`(module-level 执行真实网络 I/O 和后台线程的
  脏脚本,`test_real_world.py` 甚至在 module scope 直接 `sys.exit(...)`)
  通过 `collect_ignore` 排除在正常收集之外——不是 N6 的来源(用显式路径
  `pytest tests/test_real_world.py` 才会触发它们的 module-level bug,正常
  `pytest -q` 全套件根本不会导入它们,已用 `--collect-only` 实测确认)。
  `debug/*.py` 同理被 `collect_ignore_glob` 排除。
- `sqlite3` 连接:`RunStore` 每次 `save()`/`list_runs()` 都开新连接,没有
  模块级连接池/缓存,不是候选。
- `bfdiag.record.adopt.auto_record()` 会安装全局 `sys.excepthook` +
  `atexit` hook,是仓库里唯一一处会修改进程级全局状态的 bfdiag 代码——但
  确认它只在两条测试里被调用,且都通过 `subprocess.run(...)` 隔离
  (`test_auto_record_persists_ok_on_clean_exit`/`..._failed_on_uncaught_exception`),
  从未在 pytest 主进程内直接调用过,不是候选。
- `bfdiag/trace/ring.py::TRACE_ENABLED` 是模块级缓存的 `QSR_TRACE`(在
  import 时读一次,不是每次调用重读)——这是任务简报没提到的另一种缓存,
  但和 `test_cli_ls_labels_an_unfinished_record_running` 完全无关的代码路径
  (它不碰 trace),排除。
- **线程泄漏(最初的头号嫌疑)**:在这条测试运行的确切位置临时插入
  `threading.enumerate()` 探针,跑一次正常全套件(无人工负载)——**只有
  `MainThread` 活着**,没有任何来自 `tests/test_bfdiag_daemon.py`(会启
  真实后台线程)或别处的残留线程。这基本排除了"某个测试的后台线程还没
  退出、把字节写进了这个测试的 capsys 缓冲区"这个最初最有说服力的假设。

### 复现尝试(真机,不是猜测)

用 `torch` 在真实 GPU 上跑持续矩阵乘(`nvidia-smi` 确认 GPU 利用率
40–60%)+ 4 个 CPU 满载 busy-loop 进程(24 核机器),反复跑全套件
**约 28 次**(含验证修复前后),**一次没有复现**。这和任务简报给出的
"观测到 3/5" 有明显差距——可能是:(a) 原始观测的"GPU 负载"具体条件比
一次矩阵乘 busy-loop 更特殊(比如恰好有另一个进程在做真实的磁盘 I/O 或
真实的模型加载,不只是计算);(b) 3/5 本身样本量很小,置信区间很宽;
(c) 触发条件依赖某个我没能复现的、更精确的时序窗口。**没有找到确定性的
根因**,这是诚实的结论,不是回避。

### 修法:让它"下次失败时自己讲清楚",并让断言本身更精确

既然无法确定性复现,就不该假装"改一下这里应该就好了"——那样下次再炸,
排查者面对的还是同一个"神秘失败",比现在更糟(因为会以为这个问题
"已经修过了")。改成两件确定有价值、零风险的事(见
`tests/test_bfdiag_record.py::test_cli_ls_labels_an_unfinished_record_running`):

1. **断言从"整段 capsys 输出里随便哪都行的子串匹配"改成"精确定位到这条
   run_id 自己的输出行,再检查该行"**——原来的写法连"结果是真的没打印
   出来"和"输出里混进了别的东西导致误判"都分不清;新写法先按行过滤出
   `unfinished` 开头的那一行,再单独检查它。
2. **失败时的断言消息里带上完整原始输出 + 当前存活线程列表**——下次
   (如果还会发生)第一次失败就有诊断数据,不需要再重跑、再猜。这正是
   `docs/diagnostics-guide.md` "出问题时先读已有的 trace,不要重跑"这条
   金科玉律,应用到测试失败本身。

**验证**:`pytest -q` 全套件在真实 GPU+CPU 负载下跑了 8 次(修复后,独立
于上面的 ~28 次排查性重跑),0 次失败——见提交信息的 `Tested:`。这**不是
"证明 bug 已修"**(样本量对 3/5 这种概率的 bug 来说不够),而是"没有引入
新问题,而且下次万一还炸,有数据可看"。

### 留给人拍板的事

如果这个 flaky 之后还复现,请直接把失败输出(现在会带原始输出+线程列表)
贴到 issue/notes 里,不要重跑——这次的诊断数据应该足够直接定位。如果
一段时间后仍然零复现,可以考虑这条 flaky 记录本身是否该从 roadmap 移除。
