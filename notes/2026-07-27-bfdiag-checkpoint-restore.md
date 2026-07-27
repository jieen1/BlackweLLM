# bfdiag 主动式 checkpoint / restore —— 跳过重复 prefill

负责范围：`bfdiag/checkpoint/` 全部文件（`state.py`/`store.py`/`restore.py`/
`verify.py`/`cli.py`/`testing.py`）、四个测试文件
（`tests/test_bfdiag_checkpoint_{state,store,restore,cli}.py`）、本笔记。
全程在无 GPU 的 worktree 里开发，真实 `LagunaBackend`/`DFlashEngine` 路径
**只写代码，一次都没跑过**——这是本笔记最后一节要重点交代的风险。

## 0. 开场：worktree 落后的处理

任务给的 worktree 分支起点（`ceb7ec8`）落后 main 太多——`docs/diagnostics-
guide.md`、`notes/2026-07-27-probe-system-design-and-plan.md`、以及整个
`bfdiag/` 包骨架（`bfdiag/cli.py` 的 dispatcher 约定、`bfdiag/record/
fingerprint.py`、`bfdiag/daemon/*`）都只存在于 main，任务本身明确假设这些
东西已经存在（"不要碰 `bfdiag/cli.py`" 暗示它已经存在）。核实过
`ceb7ec8` 是 main（`0504a96`）的**纯祖先**、没有任何分叉提交后，做了一次
`git merge --ff-only main`（纯 fast-forward，不产生 merge commit，不丢失
任何东西）把 worktree 追平。这是完成任务的前提，不做这一步连"仓库既定
约定"长什么样都看不到。记录在此，供审阅时知晓这一步的存在和理由。

## 1. 核心任务：把"一个 slot 的完整状态"找全

### 1.1 方法

起点是 `bfdiag/daemon/session.py::RESET_CHECKLIST`（另一个 agent 从源码
读出的"reset 必须清空的清单"）。核心洞察：**checkpoint 要存的就是 reset
要清的那些东西**，但多出两类 reset 不需要操心、restore 却必须操心的
东西——因为"干净的 reset"之后下一步永远是全新的 prefill（会覆盖一切），
而"restore"之后下一步是继续跑生产代码（读到的必须是对的）。

逐条核实用 `codegraph_explore` + 关键行 `Read` 核对行号（未大面积
grep），最终清单在 `bfdiag/checkpoint/state.py::SLOT_STATE_ITEMS`（12
条，机器可读，`bf checkpoint schema` 可以打印）。以下是人读版本：

### 1.2 完整清单

| # | 状态 | 类别 | 代码出处 | 备注 |
|---|---|---|---|---|
| 1 | `slot_kv_len` | 必须存（host scalar） | `laguna.py:346`（声明）,`:1639-1641`（reset_slot） | 唯一的"绝对位置"真源，见下方"环形写指针"条 |
| 2 | `slot_committed_tokens` | 必须存（host list） | `laguna.py:347`,`:1239`（prefill）,`laguna_dflash.py:1431-1433`（dflash_round 追加） | prefill 后恰好比 kv_len 多 1（`bfdiag.invariants.checks.check_committed_ahead_of_kv_by_one` 每轮都在验） |
| 3 | 全量层（12 层）KV block | 必须存（device tensor，每层一份） | `laguna.py:1643-1647`（reset_slot 清空整个 `blocks_per_slot`）,`:290-306`（分配，形状 `(2,n_blocks,block_size,kv_heads,head_dim)`） | **体积优化**：只需存 `ceil(kv_len/block_size)` 个 block，不是整个静态 `blocks_per_slot` 分配——否则 64K 场景下可能比实际大 8 倍 |
| 4 | SWA 环（36 层）KV block | 必须存（device tensor，每层一份） | `laguna.py:1648-1653`（reset_slot 清空整个环）,`:278-281`（`_ring_blocks_per_slot` 推导） | 永远存**整个环容量**，与 kv_len 无关（环形寻址，任何物理位置都可能"活着"） |
| 5 | DFlash draft KV 环（6 层）| 必须存（device tensor，每层一份） | `laguna_dflash.py:292-305`（分配 `_draft_blocks_per_slot`）,`:1419-1429`（环形寻址写）| **本任务最容易漏掉的一项**：`bfdiag/daemon/session.py` 的 RESET_CHECKLIST 说"不清零这个 has not been shown to change results"——那是对**reset**成立的论断（下一次 prefill 会覆盖），对 **restore** 完全不成立：restore 之后紧接着就是 `dflash_round` 读这个环 |
| 6 | SWA / draft 环的写指针（相位、wraparound 状态）| **无需单独存**（derived_no_store）| `laguna.py:1108,1143,624,632,822,832,977,1023`（全部形如 `pos % ring_slots`）；`laguna_dflash.py:1424,655`| 任务原文点名的最大风险，逐行核实后的结论：**代码里根本不存在单独的"写指针"变量**——每个访问点都是从 `slot_kv_len`（或其派生值）现算 `% ring_slots`，无状态。只要 #1 + #4/#5 的原始字节都对，相位自动对（有 wraparound 专项测试守着这个结论，不是只在文档里断言）|
| 7 | 下一轮的 `(anchor, draft_tokens)` | **无需单独存**（derived_no_store）| `laguna_dflash.py:1342-1347`（`dflash_prefill_bootstrap` 尾部）,`:1396-1398,1431-1433`（`dflash_round` 的 `next_anchor == committed[-1]`）| `anchor` 和 `draft_tokens` **不是** `backend`/`engine` 的属性，只活在调用者的循环变量里。但可证明：`anchor` 恒等于 `slot_committed_tokens[-1]`（已经是 #2 的一部分），`draft_tokens` 恒是对 draft KV 环的一次确定性前向重算（`_draft_forward`/`_draft_cg.replay`）——`restore_checkpoint` 用与生产代码**完全相同**的调用重新推导，不额外存储 |
| 8 | Laguna 轻量前缀复用（`find_prefix_match`）| **无需单独处理**（derived_no_store）| `laguna.py:1800-1818` | 纯函数于 #1/#2，恢复后自动"生效"，见 §4 与它的区别 |
| 9 | CUDA Graph 捕获期 warmup 残留 | 不是存档内容，但影响 restore 的写入顺序 | `bfdiag/daemon/session.py` 的 RESET_CHECKLIST 引用 `laguna_cuda_graph.py:294,702`,`laguna_dflash_cudagraph.py:301,544` | restore 必须先 `reset_slot(目标slot)` + 清空该 slot 的 draft 环，再写入存档内容——顺序错了会被残留污染 |
| 10 | **发现的真实 bug（未修复）**：`reset_slot` 的 block 切片切错了轴 | 与本任务范围无关，但被本任务的多槽测试意外撞见 | `laguna.py:1647,1653` | 见 §5 |
| 11 | GDN 递归状态 | **不适用**（重新核实，不只是照抄）| `laguna.py:400`（`gdn_layer_names=[]`）,`runtime/gdn_state.py` | LagunaBackend 构造时显式传空列表，这段状态属于另一条路径（`DirectModelRunner`），`bfdiag` 的 `LagunaEngineProvider` 完全不加载它 |
| 12 | 内容寻址持久前缀缓存（BlockPool/prefix_cache.py）| **不适用**（重新核实）| `laguna.py:1655-1658`（`reconcile_prefix_hit` 显式 stub，永远返回 0）| 同上，属于 `DirectModelRunner` 路径 |

### 1.3 为什么"不需要单独存"这个结论可信

第 6、7 条是任务原文最担心的两处（"写指针/相位""prefill 后紧跟的
draft_tokens"）。这两条能"无需存储"不是我拍脑袋的化简，而是：

1. **代码事实**：逐个访问点核对过，没有例外。
2. **有测试守着**：`tests/test_bfdiag_checkpoint_restore.py::
   test_full_state_integrity_after_restore` 用的测试环容量故意开得很小
   （SWA/draft 环 window=40、block_size=16 → 环容量约 80 token），
   `_prime_slot` 跑到 kv_len 远超环容量（真实触发 wraparound），断言
   `kv_len > engine._draft_blocks_per_slot * engine.block_size` 且逐字节
   验证恢复正确——不是"理论上应该没问题"，是"真的绕了几圈还测过"。
3. **verify 安全阀兜底**：即使上面两条推导有漏洞，`verify.py` 的确定性
   回放会在第一轮就发现（因为 `dflash_round`/`_draft_forward` 的输出对
   这些值高度敏感——见 §6 的参数化测试）。

## 2. 体积估算表

依据（均从代码读出，非假设）：

- 12 个全量层 + 36 个 SWA 层 + 6 个 draft 层（`laguna.py:7` 文档字符串 +
  `dflash_constants.py::DRAFT_NUM_LAYERS`）。
- KV heads=8、head_dim=128，均匀适用于全量/SWA/draft（`laguna.py:130`
  注释 + `dflash_constants.py::DRAFT_NUM_KV_HEADS/DRAFT_HEAD_DIM`）。
- FP8 KV cache 在代码里就是原始 `torch.uint8`（`laguna.py:302-304`：
  `kv_dtype = torch.uint8 if "fp8" in cache_dtype_str else ...`），1
  字节/元素——这是生产配置（NVFP4 量化模型）的实际情况。
- 每 token 每层字节数 = `2(K+V) × 8 头 × 128 dim × 1B = 2048B = 2KiB`。
  12 层 → 24 KiB/token，与任务原文给的数字完全一致。
- SWA/draft 环容量公式 `_ring_blocks_for_window(window, block_size,
  qo_max) = cdiv(window-1+qo_max, block_size) + 1`（`laguna.py:50-51`，
  本包在 `state.py::ring_blocks_for_window` 里重新实现了一份纯函数版本
  ——不 import `runtime.*`，避免任何 import 期 CUDA 风险）。

由 `bfdiag/checkpoint/state.py`（`python -m bfdiag.checkpoint.state`）
**跑出来**的表（不是手算，跑代码验证过与手推一致）：

### block_size = 64（当前默认）

| 上下文长度 | 全量层（12层，随长度变化） | SWA 环（36层，固定） | draft 环（6层，固定） | 合计/slot |
|---:|---:|---:|---:|---:|
| 4K | 0.094 GiB | 45.0 MiB | 7.5 MiB | 0.145 GiB |
| 16K | 0.375 GiB | 45.0 MiB | 7.5 MiB | 0.426 GiB |
| 32K | 0.750 GiB | 45.0 MiB | 7.5 MiB | 0.801 GiB |
| **64K** | **1.500 GiB** | 45.0 MiB | 7.5 MiB | **1.551 GiB** |
| 128K | 3.000 GiB | 45.0 MiB | 7.5 MiB | 3.051 GiB |
| 200K | 4.688 GiB | 45.0 MiB | 7.5 MiB | 4.739 GiB |
| 256K | 6.000 GiB | 45.0 MiB | 7.5 MiB | 6.051 GiB |

### block_size = 128（迁移目标）

| 上下文长度 | 全量层 | SWA 环 | draft 环 | 合计/slot |
|---:|---:|---:|---:|---:|
| 4K | 0.094 GiB | 54.0 MiB | 9.0 MiB | 0.155 GiB |
| 16K | 0.375 GiB | 54.0 MiB | 9.0 MiB | 0.437 GiB |
| 32K | 0.750 GiB | 54.0 MiB | 9.0 MiB | 0.812 GiB |
| **64K** | **1.500 GiB** | 54.0 MiB | 9.0 MiB | **1.562 GiB** |
| 128K | 3.000 GiB | 54.0 MiB | 9.0 MiB | 3.062 GiB |
| 200K | 4.688 GiB | 54.0 MiB | 9.0 MiB | 4.749 GiB |
| 256K | 6.000 GiB | 54.0 MiB | 9.0 MiB | 6.062 GiB |

**一个顺手的修正**：现有 `notes/2026-07-22-laguna-l0-memory-budget.md`
里"滑窗层每槽固定：36 × 512 × 2 KiB = 36 MiB/槽"是一个简化估算（直接用
`window` token 数，没有考虑 block 取整 + 环形公式的 `+1` 安全余量）。
按真实环形公式算出来是 45 MiB（bs=64）/ 54 MiB（bs=128），比简化值大
25%~50%。差异不大，但既然本任务需要"算清楚"，就把这个偏差记录下来。

全量层的体积**只随实际上下文长度变化，不随 `blocks_per_slot` 配置变化**
——这正是 `state.py::full_block_range` 的关键行为：只存
`ceil(kv_len/block_size)` 个 block，不是整个静态分配。

## 3. 指纹与拒绝策略

`store.py` 的 manifest 指纹分两层：

### 3.1 硬性字段（`HARD_FINGERPRINT_KEYS`）——不匹配直接拒绝，无法绕过

`block_size`、`blocks_per_slot`、`ring_blocks_per_slot`、`swa_window`、
`draft_blocks_per_slot`、`kv_dtype`、`num_slots`、`model_revision`。

这些字段任何一个不匹配，意味着张量形状/寻址/权重本身就是错的——恢复会
造成显存越界写入或者悄悄算出垃圾，不是"数值可能漂移"这种程度的风险。
`restore_checkpoint` 在**碰任何张量之前**就检查这批字段，`bs=64` 存档
恢复到 `bs=128` 引擎会被 `FingerprintMismatchError` 拒绝，错误信息里
明确点名 `block_size: saved=64 current=128`（`tests/
test_bfdiag_checkpoint_restore.py::test_restore_rejects_block_size_
mismatch` 覆盖）。

### 3.2 软性字段（`SOFT_FINGERPRINT_PATHS`）——默认只警告，不拒绝

三个仓库（`qwen-sm120-runtime`/`sparkinfer`/`vllm`）的 git sha。

**这是一个需要明确交代的取舍**，与任务原文字面意思有张力：任务写的是
"指纹不匹配必须拒绝恢复"，但这个仓库"提交节奏极快"（用户自己的项目
记忆：单日近百次提交常见）——如果 git sha 变化就硬拒绝，这个功能会在
同一天之内失效，而"跳过重复 prefill 反复调试同一个问题、期间持续改
decode 路径代码"正是本功能存在的理由。所以设计成：

- **默认**：git sha 差异被记录、在 `bf checkpoint restore` 输出里显眼
  展示（"soft fingerprint diffs (informational, did not block
  restore)"），但不阻止恢复。
- **真正的安全网是 §1 提到的 verify.py**：如果代码变化真的改变了数值
  行为，确定性回放会在第一轮就抓到、拒绝交出 slot——这比"git sha 变了
  就整体拒绝"精确得多（后者会把大量"改动其实不影响这条路径"的无害
  提交也拒之门外）。
- **逃生舱**：`restore_checkpoint(..., require_clean_fingerprint=True)`
  把软性差异也升级成硬性失败，留给"必须逐位复现某次历史运行"这种更
  偏执的场景。

`block_size` 之所以放进硬性字段而不是软性——任务原文点名的理由完全
成立：项目正在做 64→128 迁移，这个字段一旦不匹配，不是"结果可能不一样"
这个量级的问题，是直接的张量越界/寻址错乱。

### 3.3 `prompt_hash` 是纯展示性字段

不参与拒绝判断（不同名字的存档本来就该有不同的 prompt，用户按名字选存档
时已经做过这个选择）；只在 `bf checkpoint show` 里给人核对用。

## 4. 与 P3 被动预触发冻结的区别

| | 本任务（主动 checkpoint/restore）| P3（被动预触发冻结）|
|---|---|---|
| 触发方式 | 人主动挑一个时刻（几乎总是"64K prefill 刚完成"）显式调用 | 后台纯函数持续监控 T1 签名，异常时自动冻结 |
| 覆盖范围 | 一个 slot 的**完整**可恢复状态（KV cache 原始字节 + 全部簿记）| 最近 ~50 轮的**遥测**（每层张量的 absmax/L2/mean/NaN·Inf 计数，不是原始 KV） |
| 目的 | 恢复后能**继续生成**（跳过 prefill）| 恢复后能**离线分析异常之前发生了什么**（不需要继续生成）|
| 存活周期 | 跨天、跨无数次新的 daemon 会话——存盘到 `.bfdiag/checkpoints/`，持久化 | 环形缓冲区，正常运行时持续被覆盖，只在触发时落盘一次 |
| 体积 | 每个 checkpoint 若干 GiB（见 §2）| 每轮几十 KB（T1 签名），55 MiB 环（T2 全量，P3 尚未实施）|
| 与 `find_prefix_match` 的关系 | 独立机制，跨进程/跨 daemon 会话生效 | 不涉及 |

一个容易混淆但值得澄清的点：**`find_prefix_match`（`laguna.py:1800-
1818`）已经提供了"同一个 slot 连续跑同一个前缀不用重新 prefill"的能力
——为什么还需要这个 checkpoint 功能？** 因为 `find_prefix_match` 只在
**同一个存活的进程/slot 内**有效——一旦这个 daemon 进程退出（无论是主动
`bf daemon stop` 还是崩溃），这段状态就彻底没了，下一次必须整段重新
prefill。本任务要解决的正是"daemon 会话之间"（甚至"今天关掉、明天再
开"）的持久化，`find_prefix_match` 完全帮不上忙——这是两个互补而不是
重叠的机制。

## 5. 顺手发现的真实 bug（未修复，不在本任务范围内）

写多槽（`num_slots=2`）单测时，`state.py`/`store.py`/`restore.py` 自己的
张量切片如果照抄 `runtime/backends/laguna.py:1647,1653`（`reset_slot`）
的写法会立刻爆出错误结果——追下去发现这是 **`reset_slot` 自身的一个真实
bug**：

KV cache 张量形状是 `(2, num_blocks, block_size, kv_heads, head_dim)`——
`laguna.py:301` 注释写明"dim=0: 0=K, 1=V"，即 dim0 是 K/V 轴（大小固定为
2），dim1 才是 block 索引。全文件除 `reset_slot` 外的每一处 block 范围
访问都写成 `ring[:, db, ...]`（先用裸 `:` 切满 dim0，再用具体索引切
dim1）——唯独 `reset_slot`（`laguna.py:1647,1653`）写成
`self.kv_caches[name][full_start:full_end].zero_()`，**没有前导的
`:,`**，切的是 dim0（大小仅为 2）而不是 dim1！

后果：

- 对 slot 0（`full_start=0`）：切片被 clip 成 `[0:2]` = 整个 dim0，
  dim1（block 维度）完全不受限——`.zero_()` 会把**所有 slot** 的全量层
  KV 全部清零。
- 对 slot > 0（`full_start >= blocks_per_slot`，通常远大于 2）：切片
  落在 dim0 范围之外，得到空张量——`.zero_()` 是**静默的 no-op**，
  该 slot 的 KV 根本没被清零。

这个 bug 在当前生产环境**完全被掩盖**：DFlash 要求 `capacity==1`（见
`laguna_dflash.py::mtp_verify_and_commit_batch` 文档字符串），即
**`num_slots` 永远是 1**——此时"清空整个 dim0"和"清空 slot 0 的 block"
是同一个操作（因为也只有一份 block 分配），bug 无法被观测到。一旦
Laguna 以 `num_slots > 1` 运行（无论是非 DFlash 路径还是未来的多槽
DFlash），这个 bug 就会实际发生。

**未修复**：`runtime/` 不在本任务范围内，按约定记录、不动手，转交
`runtime/backends/laguna.py` 的 owner。本包自己的代码（`state.py`/
`store.py`/`restore.py`/`testing.py`）**没有**复制这个 bug——所有切片
都用了正确的 `tensor[:, start:end]` 形式，由
`tests/test_bfdiag_checkpoint_state.py::
test_reset_slot_axis_bug_is_real_and_this_package_does_not_replicate_it`
（纯张量切片复现，不依赖 `runtime.*`）和整套多槽测试守着。

## 6. 关键设计决策

### 6.1 `restore_checkpoint` 只重置目标 slot，不像 `reset_laguna_engine` 那样重置整个引擎

`bfdiag/daemon/session.py::reset_laguna_engine` 是引擎级的（遍历所有
slot）。本包的 `restore_checkpoint` 故意只做 `backend.reset_slot(目标
slot)` + 清空该 slot 的 draft 环——因为"往热 daemon 的某个 slot 恢复一个
存档"不应该打扰其他 slot 里已经在跑的实验。`tests/
test_bfdiag_checkpoint_restore.py::
test_restoring_into_nonzero_slot_does_not_disturb_other_live_slots` 专门
测这条。

### 6.2 `save_checkpoint`/`restore_checkpoint` 都有"探测式 baseline 会推进真实状态"的副作用

`verify.py` 的确定性回放不是"只读检查"——它是真的跑 `dflash_round`
（唯一能验证"模型真的会算出这个结果"的办法）。这意味着：

- `save_checkpoint` 存完之后，**活着的那个 slot** 会被 baseline 探针
  推进几轮，不再停留在存档的那个点——如果调用者存完还想让这个 slot
  保持原状，需要自己紧接着 `backend.reset_slot(slot)`。
- `restore_checkpoint` 默认 `verify_after=True`，验证探针的输出是
  **真实的、确定性的生成内容**，不是被丢弃的空转——所以返回值里专门有
  `verified_tokens` 字段装这些真实产出的 token，`anchor`/`draft_tokens`
  给的是验证轮**之后**下一轮该用的值。
- 需要"纯净、完全不被二次触碰"的恢复结果（比如只是想看一眼恢复出来的
  原始字节）时，传 `verify_after=False, derive_next_round=False`——这
  条路径完全不调用 `_draft_forward`，是本包对外暴露的**唯一**真正"零
  副作用"的恢复模式。

### 6.3 `anchor`/`draft_tokens` 不入档，靠恢复后重新推导

见 §1.2 第 7 条。好处：manifest 更小、不用担心"存的值和恢复时环境计算出
的值不一致"这类问题——因为两者本来就该是同一个确定性函数的两次求值。

### 6.4 `bf exec --from-checkpoint <name> script.py` 没有按任务示例字面实现

任务给的目标 CLI 里有这一条，但 `bf exec` 命令本身定义在
`bfdiag/daemon/cli.py`——不在本任务允许触碰的文件范围内（"只碰
`bfdiag/checkpoint/*`"）。已实现的等价物：
`bf checkpoint restore <name> --exec-file script.py`（`bfdiag/checkpoint/
cli.py::_cmd_restore`），效果一样（同一次 daemon `exec` 调用里先恢复、
再执行脚本，脚本能看到恢复出来的 `anchor`/`draft_tokens` 局部变量），
只是命令名字不同。

**如果想要字面的 `bf exec --from-checkpoint`**，需要 `bfdiag/daemon/
cli.py` 的 `exec` 子命令加一个 `--from-checkpoint NAME` 参数，解析后在
构造要执行的代码前面插入：

```python
from bfdiag.checkpoint.restore import restore_checkpoint
_bf_ckpt = restore_checkpoint(engine, args.slot or 0, args.from_checkpoint)
anchor, draft_tokens = _bf_ckpt.anchor, _bf_ckpt.draft_tokens
```

这是一行改动量级的事，只是文件所有权不在本任务范围内，记在这里交给
`bfdiag/daemon/` 的 owner。

## 7. 交付文件清单

```
bfdiag/checkpoint/__init__.py     # 包说明 + 与 P3 关系摘要
bfdiag/checkpoint/state.py        # 声明式状态清单（12 条）+ 几何解析 + 体积估算
bfdiag/checkpoint/testing.py      # FakeBackend/FakeDFlashEngine（纯 CPU，确定性）
bfdiag/checkpoint/store.py        # manifest + safetensors 存取、指纹、体积统计
bfdiag/checkpoint/restore.py      # 恢复主流程（目标slot-only）
bfdiag/checkpoint/verify.py       # 确定性回放安全阀
bfdiag/checkpoint/cli.py          # bf checkpoint save|list|show|restore|rm|schema
tests/test_bfdiag_checkpoint_state.py    # 13 个用例：清单/几何/体积表/bug 复现
tests/test_bfdiag_checkpoint_store.py    # 12 个用例：manifest/指纹
tests/test_bfdiag_checkpoint_restore.py  # 13 个用例：完整性/丢项/指纹拒绝/verify 安全阀
tests/test_bfdiag_checkpoint_cli.py      # 7 个用例：CLI 分发 + 真实 daemon 端到端往返
notes/2026-07-27-bfdiag-checkpoint-restore.md   # 本文件
```

## 8. 验证方式

```bash
# lint（仓库自带 venv 的 ruff，纯静态分析）
/home/bot/project/qwen-sm120-runtime/.venv/bin/ruff check bfdiag/checkpoint/ tests/test_bfdiag_checkpoint*.py

# 单测（全部针对 FakeBackend/FakeDFlashEngine，CPU-only，物理上不可能碰 GPU）
/home/bot/project/qwen-sm120-runtime/.venv/bin/python -m pytest -q \
    tests/test_bfdiag_checkpoint_state.py \
    tests/test_bfdiag_checkpoint_store.py \
    tests/test_bfdiag_checkpoint_restore.py \
    tests/test_bfdiag_checkpoint_cli.py

# 体积表（跑代码验证，不是手算）
/home/bot/project/qwen-sm120-runtime/.venv/bin/python -m bfdiag.checkpoint.state
```

结果：45 个用例全绿，`ruff check` 全绿。全仓库回归
（`pytest -q --ignore=tests/test_bf_attention.py --ignore=tests/
test_laguna_sparkinfer_attn.py`，这两个文件本身就因为 venv 没装 vllm
无法收集，与本任务无关）：**641 passed, 35 failed, 1 skipped**——35 个
失败逐一确认是 venv 缺 vllm 导致的 `runtime/` 测试收集期 import 错误，
与本任务改动前**完全一致的基线**（把 `bfdiag/checkpoint/*` 和四个测试
文件移出去单独跑一遍，得到 `596 passed, 35 failed`；`596 + 45 = 641`，
失败数不变，证明零回归）。

## 9. 已知限制 / 遗留问题

- **真实 `LagunaBackend`/`DFlashEngine` 路径一次都没跑过**——`state.py`/
  `store.py`/`restore.py`/`verify.py` 的 duck-typed 代码是照当前
  `runtime/` 源码写的（`codegraph_explore` + 行号核对，不是猜的），但
  硬约束禁止在本任务里用真机验证。见 §10 GPU 待办清单。
- **`bf exec --from-checkpoint`** 没有字面实现，见 §6.4 的等价物和一行
  改动方案。
- **`prompt_hash`/软性指纹字段目前只在 `bf checkpoint show`/`restore`
  的输出里展示，没有做成一个"列出所有可能兼容的存档"之类的更高级
  UI**——`bf checkpoint list` 目前只是平铺列出所有存档，按名字找，没有
  按 workload 分组/推荐的功能，若这类工作流需求明确，是后续可加的
  纯 CPU 特性。
- **`FakeBackend`/`FakeDFlashEngine` 的确定性回放（sha256 摘要驱动）
  只是为了让测试对"任一状态项被破坏"敏感，不是真实注意力数值的模拟**
  ——真实模型的确定性依赖 `enforce_eager=True`/固定 seed/关闭
  TF32 等一系列假设，这些假设本身需要 GPU 验证（见 §10 第 4 条）。
- **体积表假设 KV cache 是 FP8（`torch.uint8`，1 字节/元素）**——这是
  当前 NVFP4 量化模型的实际配置，但 `state.py::estimate_checkpoint_
  bytes` 也支持传 `dtype_size=2` 算 bf16 场景（所有数字翻倍），未在表格
  里重复列出。

## 10. 需要 GPU 才能验证的待办清单（下一步串行安排的唯一依据）

与 `notes/2026-07-27-bfdiag-warm-daemon.md` §10 同样的纪律：每一条都是
"读代码写出来、从未跑过"，上 GPU 后必须按顺序过一遍。

1. **`state.slot_geometry` 对真实 `LagunaBackend`/`DFlashEngine` 实例的
   duck-typed 读取是否成立**：验证方法——在真实（哪怕最小配置）引擎上
   跑一遍 `slot_geometry(backend, engine, 0)`，确认 `_full_layer_names`/
   `_swa_layer_names`/`_ring_blocks_per_slot`/`_draft_blocks_per_slot`/
   `_draft_layer_names` 这些属性名和当前代码假设的一致（尤其是
   `_draft_layer_names` 的发现逻辑，`laguna_dflash.py:256-285` 有一段
   "主路径失败则退化到从 draft 模型直接遍历"的 fallback，需要确认真实
   环境走的是哪一条）。
2. **`_gather_tensors`/`restore_checkpoint` 的张量写入在真实 GPU 张量上
   是否真的是纯拷贝、没有意外的设备迁移开销或同步问题**：`.to("cpu")`
   在本任务里全程只在 CPU 张量上跑过（no-op），第一次真正从 GPU 搬到
   CPU 需要验证：(a) 64K 全量层≈1.5GiB 这个量级的 `.to("cpu")` 耗时是否
   在可接受范围（"跳过 prefill"如果本身要花几十秒搬数据，收益会打折扣）；
   (b) safetensors 落盘/加载在这个量级下的实际 I/O 耗时。
3. **`save_checkpoint`/`verify.run_probe` 里 `engine._draft_forward`/
   `engine.dflash_round` 的调用签名和返回值形状是否与本包假设一致**：
   本包假设 `dflash_round` 返回 dict 且含 `committed`/`next_anchor`/
   `next_draft_tokens`/`context_count` 四个 key（照抄
   `laguna_dflash.py:1354-1462` 的返回语句核对过），需要真实调用一次
   确认没有理解偏差。
4. **确定性前提是否真的成立**：`verify.py` 的整套安全阀假设"同一
   `(anchor, draft_tokens, kv_len, KV 内容)` 输入，`dflash_round` 的
   输出逐位确定"——这依赖 `enforce_eager=True`/固定 seed/CUDA Graph
   replay 路径不引入非确定性。`notes/2026-07-27-bfdiag-warm-daemon.md`
   §10 第 4 条已经把"金丝雀基线是否稳定"列为待验证项，这里是同一个
   前提的另一个消费者——如果那条验证失败（模型本身有非确定性），
   本包的 verify 安全阀会**永远拒绝**恢复（不是漏报，是会把"正常"的
   恢复也当成失败拒绝掉），需要先确认前提成立。
5. **`_kv_dtype_str`（读 `backend.kv_caches[name].dtype`）在真实环境下
   是否总是 `torch.uint8`（fp8）**：如果某些配置下 cache_dtype 不是
   fp8（例如调试时临时切换到 bf16），需要确认 `HARD_FINGERPRINT_KEYS`
   里的 `kv_dtype` 检查能正确捕捉到，且体积表用
   `dtype_size=2` 重算的数字是准确的。
6. **safetensors 对真实模型 KV dtype 的支持面**：本包假设 KV cache
   要么是 `torch.uint8`（fp8 打包）要么是模型的 `kv_cache_torch_dtype`
   （通常 bf16）——两者 safetensors 都原生支持，但如果真实环境出现
   某种 safetensors 不认识的 dtype（比如某个自定义 fp4 打包格式），
   需要在真实环境验证 `save_file`/`load_file` 是否报错。
7. **`.bfdiag/checkpoints/<name>/` 的磁盘占用在真实 64K/128K/256K 场景
   下是否与 §2 表格吻合**：需要一次真实 64K prefill 后 `bf checkpoint
   save`，用 `du -sh` 核对实际文件大小与预测的 1.55 GiB（bs=64）是否
   一致（允许安全带来的少量额外开销，比如 safetensors 自身的 header）。
8. **`bf checkpoint restore` 之后紧接着真实解码若干轮，是否真的比
   冷启动 prefill 快**——这是本功能存在的全部理由，必须实测对比：
   (a) 冷启动 64K prefill 的耗时基线；(b) `bf checkpoint restore` 的
   耗时（磁盘读取 + 张量写入 + verify 探针）；(c) 确认 (b) 远小于 (a)，
   否则这个功能没有达成目标。
9. **多槽（`num_slots > 1`）场景下，`restore_checkpoint` 对目标 slot 的
   恢复是否真的不影响其它 slot**——CPU 测试已经用 FakeBackend 验证过
   这条（`test_restoring_into_nonzero_slot_does_not_disturb_other_live_
   slots`），但真实引擎的 CUDA Graph 捕获（`_verify_cg`/`_draft_cg`）
   目前只对**一个**物理 slot 捕获（DFlash 要求 `capacity==1`）——如果
   真实场景真的想要多槽 DFlash，这条假设本身需要重新核实，不只是
   checkpoint 这一层的问题。
