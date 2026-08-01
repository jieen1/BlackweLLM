# A3 缓存协调者：可实施设计

> 编制日期：2026-08-02 · 基线 commit：`d87c7ef`（`main`，Track A 第 1–5 步已完成）
> 分支：`work/a3-design-20260802`（worktree `/home/bot/project/qsr-w-a3`）——**本文档不合并进 main**。
>
> 范围：本文档只出设计。不改 `runtime/`、`server/`、`tests/` 的任何生产/测试代码；
> 本文档里出现的 dataclass、方法签名、测试骨架**全部是说明性代码块**，是"这样写"的提案，
> 不是已落地的实现。
>
> 记号约定：**【事实】**= 亲自读代码/读上游源码核实过的陈述，带 `file:line`；
> **【判断】**= 设计取舍，可被推翻；**【待拍板】**= 半径大到不该由本文档单方面定案的项，
> 集中在 §8，带选项和推荐但不代选。

---

## 0. 方法论：跟"转述"的区别

本文档的每一条 vLLM/SGLang 引用都是本轮**重新核实**的行号（`~/.venvs/vllm025/lib/
python3.12/site-packages/vllm/`、`/home/bot/project/sglang`），不是照抄
`notes/2026-08-01-hybrid-cache-prior-art.md` 的转述——那份笔记质量很高，但重新核实后发现
它有**两处需要纠正**的过度概括（§3、§4 开头），本文档在指出它们的同时保留它做对的部分。
本仓库侧的引用全部来自本轮直接 `Read`/`grep` 的输出。

---

## 1. 现状盘点（事实，非判断）

### 1.1 Laguna 今天实际跑的前缀缓存，不是 `architecture.md` §2.4 描述的那套

**【事实，且是一处需要先纠正的文档错误】** `docs/architecture.md:125` 说现状是"内容寻址：
块级哈希 + 引用计数 + LRU 驱逐（`runtime/block_pool.py`）"。这与真实代码不符：

- `grep -n "hash\|ref_cnt\|LRU\|lru" runtime/backends/laguna.py` **零匹配**。
- `runtime/backends/laguna.py:39` 是这个文件唯一一处 `from runtime.block_pool import ...`，
  只导入 `ChunkedPrefillState`，不导入 `BlockPool`/`Block`/`hash_block_tokens`。
- `server/app.py:1248-1249` 自己的注释直接否认了 BlockPool 假设："LagunaBackend uses static
  block allocation (num_slots × blocks_per_slot), not a dynamic BlockPool."
- `docs/qwen36-rebuild-spec.md:144` 独立佐证："`LagunaBackend` 从不构造 `BlockPool`，
  从不调用 `cache_block`/`touch`/`hash_to_block`"。
- 真正驱动线上准入的三个方法，全部是**同槽（per-slot）token 列表线性比较**，向下取整到
  `block_size`，没有跨槽内容寻址、没有 `ref_cnt`、没有共享 LRU 池：
  - `reconcile_prefix_hit`（`laguna.py:2179-2220`）：遍历所有槽的 `_prefix_cache_tokens[s]`
    （每个槽**上一次**跑完时保存的 token 历史），逐 token 比较，取最深命中槽，写入
    `_pending_prefix_hits[best_slot]`。
  - `find_best_slot_for_prompt`（`laguna.py:2222-2259`）：同样的比较逻辑，但只在给定的
    `free_slots` 里找，返回 `(best_slot, hit_depth)`。
  - `find_prefix_match`（`laguna.py:2531-2549`）：单槽版本，比较 `slot_committed_tokens[slot]`
    （当前活跃槽的实时历史），供 DFlash 内部用（见下）。
- 命中记录用两个并行数组：`_prefix_cache_tokens[slot]` / `_prefix_cache_kv_len[slot]`，
  在 `reset_slot`（`laguna.py:2157-2177`）里写入，有一条防止"二次 reset 冲掉缓存"的显式判断
  （`:2171-2174`）。
- 调用点：`server/engine.py:1090-1106`。两处细节：
  1. `find_best_slot_for_prompt` 返回的 `_hit` 在 `engine.py:1091` **被丢弃**，随后
     `engine.py:1106` 对每个 prompt 再单独调一次 `reconcile_prefix_hit` 拿真正的命中深度。
  2. `engine.py:1090` 用 `hasattr(self.runner, "find_best_slot_for_prompt")` 做能力探测，
     这跟 `architecture.md` §3.5.3 想消灭的 `try/except AttributeError` 反模式是姊妹形态。
     协调者落地时应顺手换成 `capabilities.prefix_cache` 查询。
- `runtime/backends/protocol.py:192,194-198` 已把这两个方法收进协议，由 `prefix_cache`
  能力位治理（`CAPABILITY_MEMBERS`，`:235`）。今天两者的返回类型是裸 `int` /
  `tuple[int, int]`（slot, hit_depth）。
- `find_prefix_match` **不在协议的 13 个成员里**，被 `runtime/backends/laguna_dflash.py:1379`
  （DFlash 投机路径内部）和 `bfdiag/workloads.py:2362`（诊断）直接调用，绕开 `ServerEngine`。
  `bfdiag/daemon/session.py:138,145` 明确称它"Laguna's own lightweight per-slot
  prefix-cache reuse"——概念同属一个家族，但不受 `prefix_cache` 能力位治理，是协议之外的
  第三个消费者，改协议形状时容易漏看，见 §7-b。

### 1.2 `BlockPool` + `_on_evict_block` 钩子：孤儿基础设施，不是死代码，但也不是活代码

**【事实】** `runtime/block_pool.py` 的 `BlockPool` 类（`:270` 起）——哈希索引
（`hash_to_block`）、`ref_cnt`、intrusive LRU free queue（`FreeBlockQueue`）、
`_on_evict_block` 钩子——**在生产路径里从未被实例化**。唯一的生产/参考构造点是
`oracle/qwen36_vllm/direct_model_runner.py:441,895`（已退役、只读的 oracle）；其余全部在
`tests/test_block_pool.py` 和 `benchmarks/prefix_cache_*_check.py`（单测/基准）。

`_on_evict_block` 只被两处赋值：`direct_model_runner.py:590`
（`self.block_pool._on_evict_block = self.evict_gdn_checkpoint`）和
`benchmarks/prefix_cache_eviction_check.py:196`（基准里手动挂 stub）。
`runtime/backends/laguna.py`、`server/engine.py` 对 `_on_evict_block` 零匹配。
`_ssm_spec_row`/`_physical_slot`（`block_pool.py:23-79`）同理：被 `tests/test_block_pool.py`
测试纯数学正确性，但唯二的调用方是 `oracle/qwen36_vllm/{metadata_builders,cuda_graphs,
backends/qwen36,direct_model_runner}.py`（全部只读）。

**【判断】** 这不是"S4 时代写坏的死代码"，是"提取之后失去了活调用方的正确代码"：`BlockPool`
是 commit `8ec9cd3`（"B5 模块化 Domain 1：block_pool 提取"）从 `direct_model_runner.py`
纯移动出来的；那个运行器（当时经 vLLM 服务 Qwen3.6）后来又整体搬进 `oracle/qwen36_vllm/`
（`git log` 确认：commit `a9cb932`，"Isolate retired Qwen runtime from Laguna
distribution"）。提取先发生、退役后发生，退役没有动 `block_pool.py`，因为它已经被泛化出去了
——这正是这些钩子"还在、没用、结构完好、但比它们服务的模型更晚被写下日期"的原因。

### 1.3 一处活文档间的矛盾："残迹"这个词本身可能已经过时

**【事实】** `docs/roadmap.md:158`（S4 条目）、`docs/architecture.md:129-131`、以及本任务
原话（`docs/implementation-plan.md:195`）都把 `_on_evict_block`/`evict_gdn_checkpoint`
称为"残迹"（措辞暗示待清理）。但 `docs/qwen36-rebuild-spec.md:561-565`——编制日期
**2026-08-02**，比 `implementation-plan.md` 的"2026-08-01 二次修订"新一天——明确写：
"`runtime/block_pool.py` 里没有需要清理的 GDN 专属残留代码……两者都是干净、可直接复用的
挂钩，不是'死代码待删'……它们是**休眠但设计良好的挂钩**"。`docs/README.md` 的"文档纪律"
把三份文档全部列为"反映当前状态"的活文档——这不是转述之间打架，是仓库里两份都自称权威的
文档互相矛盾，且更新的一份明确指出更旧的一份措辞有问题。

**【判断】** 我读了 `_on_evict_block`（`block_pool.py:323-329,343-361`）、`_ssm_spec_row`
（`:45-79`）、`_physical_slot`（`:23-24`）的实现后同意 `qwen36-rebuild-spec.md` 的结论：三者
都是通用、正确、不含任何"专属于已废弃方案"的错误逻辑的原语。所以"清掉 S4 的 GDN 残迹"应
拆成两件不冲突的事：**(a)** 改文档措辞（§8 决策点 1）；**(b)** 不删除、不重写这三个原语，
留给 §7 决定何时接线。

### 1.4 本仓库已有一份完整的参考实现，比 vLLM/SGLang 更贴近我们的寻址方案

**【事实】** `oracle/qwen36_vllm/prefix_cache.py:131-166` 的 `reconcile_prefix_hit` 已经
实现了 `(A, G, L=G)` 两资源协调，逐字对应我们要设计的 `(kv_hit, state_hit)`：左到右哈希
匹配算 `A`（attention 命中）；从 `A` 向左找最深的、**哈希一致**的 GDN checkpoint 边界算
`G`；`L = G`（恒 `<= A`）。`A>0, G=0` 视为 compute miss，不是"部分命中"
（`prefix_cache.py:135-139` 注释原话）。

`oracle/qwen36_vllm/gdn_state.py:183-226` 的 `evict_gdn_checkpoint`/
`_evict_gdn_checkpoints_for_budget` 已经实现"独立字节预算 + LRU + 双向 lockstep"：

- **正向**：`direct_model_runner.py:590` 把 `evict_gdn_checkpoint` 挂到
  `BlockPool._on_evict_block`——KV 块被驱逐时联动丢配套 checkpoint。
- **反向**：`gdn_state.py:205-209`——checkpoint 自己的字节预算触发驱逐时，*如果*配套 KV
  块 `ref_cnt == 0` 就顺手丢它的 hash；*如果* `ref_cnt > 0`，**只丢 checkpoint，不动 KV
  块**，注释原话："losing only the checkpoint, which merely turns a future would-be hit
  into a safe compute miss (L = G <= A still holds)"。

**【判断】** 这份代码不能被 import（硬约束），但可以被抄读——`qwen36-rebuild-spec.md` 自己
也是这个用法。下面 §3、§4 的很多设计答案直接引用这份实现，而不是重新发明。

### 1.5 一处此前没被任何文档描述过的已有基础设施：`ArchitectureSpec.needs_two_cache_families`

**【事实，重要】** `runtime/architecture.py:56-64`（`LayerSpec`）已经有一个 `cache: str`
字段，取值 `CACHE_PAGED_KV`/`CACHE_RECURRENT`（`:47-48`），字段文档字符串原话：
"This is the field `SlotResourceManager` (step 7) is built around: a checkpoint mixing
both is what makes two cache families necessary rather than hypothetical."
（`architecture.py:59-62`）。`ArchitectureSpec`（`:96-142`）据此暴露：

```python
# runtime/architecture.py:126-142（已存在，原样引用）
@property
def paged_kv_layers(self) -> tuple[int, ...]:
    return tuple(layer.index for layer in self.layers if layer.cache == CACHE_PAGED_KV)

@property
def recurrent_layers(self) -> tuple[int, ...]:
    return tuple(layer.index for layer in self.layers if layer.cache == CACHE_RECURRENT)

@property
def needs_two_cache_families(self) -> bool:
    return bool(self.paged_kv_layers) and bool(self.recurrent_layers)
```

**这条改变了本设计的一个关键决策**（见 §8 决策点 3）：Track A 第 3 步（A1 ModelSpec，已落地，
`tests/test_architecture_spec.py` 覆盖）**已经**预留了"这个 checkpoint 是否需要两类缓存"
这个布尔判定，而且它天然属于**模型架构事实**（从 `config.json` 解析而来），不是**backend
执行层能力**（`BackendCapabilities` 描述的是"这个 backend 实现了什么"，不是"这个模型结构
上需不需要什么"）。这意味着协调者判断"要不要实例化第二个分配器"时，理应先问
`ArchitectureSpec.needs_two_cache_families`，而不是凭空在 `BackendCapabilities` 里新加
一个字段——尽管两者最终数值上应该一致（§8 决策点 3 展开这个取舍）。

`runtime/model_registry.py:80`：`IMPLEMENTED_BACKENDS = frozenset({"laguna"})`——今天
只有一个 backend，`needs_two_cache_families` 对 Laguna checkpoint 的解析结果为 `False`
（单一 `CACHE_PAGED_KV` 层类型），这是 §5"零行为变更"论证的又一层证据，且是**已经跑过真实
checkpoint 验证**的证据（`tests/test_architecture_spec.py`），不是本文档新提出的假设。

### 1.6 协议现状（`runtime/backends/protocol.py`，已核实）

**【事实】**

- `BackendCapabilities`（`:60-79`）：5 个冻结布尔字段；`prefix_cache` 治理
  `reconcile_prefix_hit`/`find_best_slot_for_prompt`（`CAPABILITY_MEMBERS`，`:235`）。
- `BackendSnapshot`/`SlotSnapshot`/`PrefixSnapshot`（`:83-126`）已落地（Track A 第 2 步，
  commit `f24f5ad`）。
- 协议模块自己的文档字符串（`:42-46`）写明重命名"stays available to take whenever call
  sites next change for an unrelated reason (naturally: **step 7's A3 coordinator**,
  or whichever step first touches these three call sites again)"——协议作者已经预见到
  本步骤会改这两个方法的签名；改它是"顺势"，不是"破坏契约"。
- 会红的具体测试清单（改协议形状后，不是猜的）：`tests/test_backend_protocol.py`
  （`TestContractShape`/`TestConformanceChecker`/`TestLagunaConformance`）+
  `tests/test_laguna_server_integration.py:143-164`
  （`test_reconcile_prefix_hit_cold_miss`/`_warm_match`，`:150,162` 直接断言裸 `int`）。

### 1.7 两个跟本设计相关、重新核实过的数字（不沿用旧文档）

**【事实】** `notes/prefix-cache-design.md`（旧文档，"Status: DESIGN, not yet built"，写于
vLLM-based `direct_model_runner.py` 时代）用的 `block_size=16`、`chunk_size=8192` 是那个
时代的数字。今天的真实值：`server/app.py:97` `SERVER_BLOCK_SIZE` 默认 **64**（不是 16）；
`runtime/backends/laguna.py:276` `_prefill_chunk_tokens` 默认 **8192**（跟旧文档的数字
一致，但是重新核实过的）。`8192/64=128` 整除，今天两个默认值之间没有对齐问题——但这条必须
在 Track B 真正选定 GDN checkpoint 边界策略时**重新核实**，不能假设"当年算过就一直成立"
（坑 #2，§6）。

### 1.8 一处需要立刻记下来、容易被忽略的既有分歧：`RESERVED_PHYSICAL_SLOTS`

**【事实，重要】**

```
runtime/block_pool.py:20:                  RESERVED_PHYSICAL_SLOTS = 1
runtime/backends/laguna.py:53:             RESERVED_PHYSICAL_SLOTS = 0
runtime/backends/laguna_cuda_graph.py:41:  return slot  # RESERVED_PHYSICAL_SLOTS = 0
```

**同名常量，两个模块，两个不同的值。** `docs/qwen36-rebuild-spec.md:131,603,677` 已经把
这个分歧记录在案，结论是 `=1`"**应废弃（待核实）**——花了四轮调试才坐实这是'vLLM 调度器
从不产出物理索引 0'这一 vLLM 特定事实，不是硬件事实；Laguna 保留 0 个物理槽也能跑"。

**【判断】** 这是给 §2 INV-A3-8 的一个具体的、已经在文档里出现过的反面教材：如果协调者
或未来的 state 分配器不假思索地沿用 `block_pool.py` 的 `=1` 约定，会在**与今天 Laguna
实际使用的约定不一致**的情况下引入一个新常量——这正是本设计通篇强调的"不要转述别处的数字"
原则要防的那类错误，恰好在这个仓库自己身上已经发生过一次（且已被记录、未被修正）。A3 落地
时应该显式核实新分配器该用哪个值，不能默认抄 `block_pool.py` 的 `=1`。

---

## 2. 问题 1：协调者持有哪些不变量，破坏时的可观测症状是什么

| # | 不变量 | 来源 | 破坏时的可观测症状 | 今天有没有断言 |
|---|---|---|---|---|
| **INV-A3-1** | 资源引用一致：任何被引用的资源不会被两个互不知情的活槽同时持有 | 对应旧 INV2/INV9 | **不是崩溃**——是某个请求的输出**因为另一个请求的写入而改变**，没有异常。只能靠信号探针测试（每槽打标记 token）或 bit-exact 回归发现——这正是"很多 token 之后才显形"的字面含义 | 否 |
| **INV-A3-2**（核心） | `state_hit <= kv_hit` 恒成立（`L = G <= A`） | 对应旧 INV3/R5；`oracle/prefix_cache.py:131-166` 已实现 | 若违反：协调者告诉调度层可以跳过比实际拥有有效递归状态更多的 token → 续写的 suffix 从错误的 GDN 状态起步 → **贪心输出在很多 token 之后偏离预期，没有异常、没有崩溃**，只能靠 `bf diff` 在几千 token 之后发现质量/接受率回归 | 否。`bfdiag/invariants/checks.py` 今天的六个 `check_*` 函数（`check_kv_len_monotonic`/`check_no_duplicate_ids`/`check_aux_hidden_alignment`/`check_cg_replay_slot_consistency`/`check_accepted_bound`/`check_page_table_covers_seqlen`）一个都不覆盖跨资源一致性 |
| **INV-A3-3** | 驱逐双向 lockstep，且方向不对称：KV 驱逐**必然**级联丢配套 state checkpoint；state 驱逐**只允许**在配套 KV 块 `ref_cnt==0` 时顺手清它的哈希索引，绝不允许回收活引用的 KV 内存 | 对应旧 INV3/R5；`gdn_state.py:183-209` 双向注释块 | 正向断了：未来的命中以为某段 KV 区域有有效 state，但那段内存其实已经分给了别人——同 INV-A3-2 的静默污染症状。**反向若被错误地做成对称**（state 驱逐强行回收活 KV）：活请求的 KV 被抽走，表现为崩溃（非法内存访问/形状错误）或更糟——静默错误的 attention 输出。`gdn_state.py:196` 的注释（"`L = G <= A` still holds"）正是为了防止这种对称化 |
| **INV-A3-4** | 有存活引用（`ref_cnt>0` 或活槽占用）的资源永不被任一分配器驱逐 | 对应旧 INV9/R4 | 并发准入+驱逐压力下的崩溃（读到错误张量，use-after-free 形态）；只在负载下才能复现——这正是为什么 `notes/prefix-cache-design.md` 自己的 `admission_under_pressure` 测试（移植进 `benchmarks/prefix_cache_eviction_check.py:501` 附近）故意在其它槽活跃时强制驱逐，而不是孤立测试驱逐 |
| **INV-A3-5** | 投机/草稿 token 永不写入任一缓存；被拒绝的草稿不能污染未来的命中 | 对应旧 INV4 | 未来某请求"命中"了一段实际从未被接受过的 token——输出在命中边界处偏离一个冷参照，偏离量正好等于被拒绝的草稿内容。只能靠"命中路径输出 vs 冷路径参照"比较发现，不是靠形状检查 |
| **INV-A3-6** | 无递归层的 backend（今天的 Laguna）必须恒有 `state_hit == kv_hit`——没有第二资源就没有分歧的可能 | 新增，§1.5 的 `needs_two_cache_families=False` 直接对应 | 若违反：Laguna 开始跳过比过去更少的 prefill——**纯性能回归，没有正确性信号**，只能靠 A6 的吞吐/bit-exact 门禁事后发现。给每个 `needs_two_cache_families=False` 的 backend 写一条断言这个等价关系的单测，成本很低，应该先写（§7） |
| **INV-A3-7** | CUDA Graph 重放对"哪些块/行来自命中 vs 新计算"保持无感知 | 对应旧 INV5 | 图里烤进了错误形状/过期地址；表现为非法内存访问，或更糟——重放时"看起来正确但实际错误"的输出。Laguna 的 decode 图今天不碰递归状态，这条对 Laguna 是空集；一旦 Track B 真正把递归状态行接入图重放，这是全新问题（§8 决策点 4） |
| **INV-A3-8** | 新分配器采用的"保留物理地址"约定必须显式核实，不能默认继承 `block_pool.py` 的 `RESERVED_PHYSICAL_SLOTS=1` | 对应旧 INV7；§1.8 的具体分歧 | `block_pool.py` 保留 1 个物理槽，但 `LagunaBackend` 自己的约定是 0 个（§1.8）。这条不是"可能会破坏"的假设性风险——它**已经**是仓库里存在的一个真实分歧，`qwen36-rebuild-spec.md` 已经记录但未解决。如果协调者/新分配器盲目继承 `=1`，而实际数据用的是 `=0` 的寻址假设，后果是本项目历史上已经真实发生过一次的"物理索引 0 读写错误状态，100% 确定性错误输出"（`block_pool.py:17-19` 自己的注释） |

**【判断】** INV-A3-2 和 INV-A3-3 是这张表里真正难查的两条，且今天完全没有自动化断言能在
它们被破坏时报警。这直接决定了 §7 必须有一步专门把它们变成 `bfdiag/invariants/checks.py`
里的显式 `check_*` 函数（哪怕现在 KV-only 场景永远不会触发，也要先把断言点位就位）——这是
对 `implementation-plan.md` C8 纪律（"没红过的门禁，能不能构造一个让它红的输入"）的提前
应用。

---

## 3. 问题 2：`(kv_hit, state_hit)` 不相等时怎么办 —— 核心难点

### 结论

**取 `state_hit`（它恒 `<= kv_hit`，即"取 min"），且必须 block-aligned。**
`[state_hit, kv_hit)` 区间的 KV 尽管物理上还在，但当作 compute miss 处理：那段位置从未被
"发布"（append-only + immutable-published 规则），从 `state_hit` 续写前缀是安全的，不需要
垃圾回收或部分覆盖逻辑。

**举例：KV 命中 900，state 只命中 400 → 有效复用长度 = 400。**

- **不能整体退化到 0**：`[0,400)` 的 attention 命中是真实、可验证的（同一份 chained hash
  锁定），丢弃它没有正确性收益，只有性能损失。
- **不能用 900**：`[400,900)` 这段区间根本不存在"可以直接读、跳过 forward"的递归状态——
  GDN 层不是分块可组合的，它是单个累积标量，只能从上一个真实 checkpoint 边界顺序重算。
  用 900 会导致 GDN 层从位置 400 开始，拿着"400 之后没更新过"的状态去处理 500 个新 token，
  产生错误但不崩溃的结果——正是 INV-A3-2 被破坏时的症状。

### vLLM/SGLang 实际怎么处理——本轮重新核实后的修正

**【事实，修正 #1】** `notes/2026-08-01-hybrid-cache-prior-art.md` 的整体框架把这个问题
描述成"两个上游都返回两个数字"。重新核实 vLLM 的**调度器实际读到的值**后，这个概括**不
成立**：

- `vllm/v1/core/kv_cache_coordinator.py:630`，`HybridKVCacheCoordinator.find_longest_
  cache_hit`，返回类型 `tuple[tuple[list[KVCacheBlock], ...], int]`——**第二个元素是单个
  int**（"The number of tokens of the longest cache hit"），由一个迭代不动点算法算出
  （`:642-680` 起的 `while True` 循环："Each attention type either accepts the current
  candidate length or reduces it. If any type reduces the length, restart checks over
  all types."）。这是**调度器真正消费的值**——`kv_cache_manager.py:161`
  （`self.block_pool = self.coordinator.block_pool`）确认所有 `SingleTypeKVCacheManager`
  子类（包括 `MambaManager`）共享同一个 `coordinator`，这个不动点循环就是让所有组"同意"
  一个数字的机制。
- **确实存在**一个返回"每组各自命中长度"的方法——`find_longest_cache_hit_per_group`
  （`kv_cache_coordinator.py:742`）——但它唯一的调用点是 `scheduler.py:697-698`，**被
  `self.connector is not None and self.has_mamba_layers` 严格限定**
  （`scheduler.py:690-691`），即只服务**分离式 prefill（KV-connector）的跨节点传输量估算**
  （`scheduler.py:715`：Mamba 状态传输量不管 KV 侧命中与否都要估），不是本地调度的通用路径。
  这大概率就是 vLLM 0.26.0 号称的"partial prefix-cache hit support for hybrid models"
  的种子，但**本机没有 0.26.0 checkout，未能核实**这个猜测。
- **结论**：vLLM 的通用调度路径**不是**"暴露两个数字给调度器"，是"内部收敛成一个数字
  （已经是更保守的那一侧）"——跟我们自己退役代码的做法（`L=G<=A`，直接推导而非迭代求解，
  因为两组场景下不需要通用不动点算法）殊途同归。

- **【事实，修正 #2】** 该笔记 §5 的标题"Eviction: two budgets, two accountings"配合其
  开篇"Both projects solved this problem in public"的整体框架，容易让人以为"两个独立预算"
  是 vLLM 和 SGLang 的共识。重新核实后：**这是 SGLang 的设计，不是 vLLM 的**。vLLM 的
  `BlockPool` 是**一个共享、无类型区分的池**：所有 `SingleTypeKVCacheManager`（含
  `MambaManager`）持有同一个 `self.block_pool` 引用（`kv_cache_manager.py:161`），用的是
  一个全局 LRU free queue、一个 `num_gpu_blocks` 地址空间，靠 `unify_hybrid_kv_cache_
  specs`（`kv_cache_utils.py:1403`起）把每组的 `page_size_bytes` 统一到公共值，让一个整数
  block id 能合法索引进任意组的张量。vLLM **没有**资源类型感知的独立预算。

  SGLang 才是真正做了"两个数字、两个预算"的那家：`MatchResult`
  （`/home/bot/project/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py:155-190`）
  同时携带 `device_indices`（定义 KV 命中长度）和 `mamba_branching_seqlen`（"the mamba
  radix cache branching point, which is the longest page-aligned position that could've
  been cache hit if there exists a mamba state"）——**两个独立字段，不是收敛后的单一值**。
  `schedule_policy.py:140-141` 把 `mamba_branching_seqlen` 原样搬到请求对象上，"怎么用这
  个差值"的决策被推迟到下游 Mamba 专用代码，不在匹配那一刻就解决。

### 我们该怎么选

**【判断】** 在**类型层面**照 SGLang 的做法——真正暴露两个数字（满足任务原话"返回
`(kv_hit, state_hit)` 二元组"的字面要求，也给 `/metrics`/bfdiag/A6 的"前缀命中率不回归"
门禁一个具体可观测的东西："`state_hit < kv_hit` 多久发生一次、差多少"是一个真实信号，
能告诉 Track B GDN checkpoint 粒度是不是太粗）。在**调度层面**照 vLLM 的收敛路径和我们
自己退役代码的做法——只有一个"有效数字"驱动"跳过多少 prefill"的决策，没有任何一个被
核实过的系统（vLLM、SGLang、我们自己的退役代码）会在**调度逻辑**里对两个数字的差值分支
处理，差值只用于观测，不用于分支：

```python
@dataclass(frozen=True)
class PrefixHit:
    """kv_hit: 最长的、KV 物理存在且可引用的 block-aligned 前缀长度。
    state_hit: 不超过 kv_hit 的、有匹配递归状态 checkpoint 的最长边界。
    不变量（INV-A3-2）：0 <= state_hit <= kv_hit 恒成立。
    无递归层的 backend 恒有 state_hit == kv_hit（INV-A3-6）。
    """
    kv_hit: int
    state_hit: int

    def __post_init__(self) -> None:
        if self.state_hit > self.kv_hit:
            raise ValueError(
                f"INV-A3-2 violated: state_hit={self.state_hit} > kv_hit={self.kv_hit}"
            )

    @property
    def effective(self) -> int:
        """调度层应该用来跳过 prefill 的长度。"""
        return self.state_hit
```

选 dataclass 而非裸 `tuple[int, int]`，理由跟 `BackendCapabilities` 选 dataclass 而非
字符串一样（`protocol.py:63-68`）：`.kv_hit`/`.state_hit` 不会像 `result[0]`/`result[1]`
一样被意外调换顺序，不变量能写成 reviewer 真的会读到的 docstring/`__post_init__` 断言，
而不是散落在调用点的隐含约定。

state 分配器的搜索范围本身应该受 `kv_hit` 约束（边界 `Lc <= kv_hit`），而不是"两边独立算
完再比较"——这正是 `oracle/prefix_cache.py:157`（`for boundary_blocks in range(matched_
blocks, 0, -1)`）已经实现的顺序：先算 `A`，再在 `[0, A]` 范围内反向找 `G`。

---

## 4. 问题 3：驱逐时两类资源的预算怎么分，谁先被驱逐

### 三种已知设计，不是两种

重新核实后，现在能看到的实际是**三个**先例，不是"两个上游都这样做"：

| 设计 | 预算 | 代表 |
|---|---|---|
| 单一共享池，无类型区分，全局 LRU | 无独立预算——靠统一 `page_size_bytes` 让一个 id 空间覆盖所有组 | vLLM（§3 修正 #2） |
| 两个独立预算，双向不对称 lockstep | KV 按块数；state 按字节数；state 驱逐绝不回收活 KV | SGLang（`EvictParams(num_tokens, mamba_num)`/`EvictResult`，`base_prefix_cache.py:83-98`；hit-test `hi_mamba_radix_cache.py:357`） |
| 两个独立预算，双向不对称 lockstep（同上一行的形状，独立实现） | 同上 | 本仓库已退役的 `oracle/gdn_state.py`（§1.4） |

### 结论：选第二/第三种，不选 vLLM 的共享池

**【判断】** 理由：KV 的天然记账单位是"块数"（随前缀长度线性增长）；GDN state 的天然记账
单位是"checkpoint 个数 × 固定字节数"（每个 checkpoint 大小不随前缀长度变化——
`notes/prefix-cache-design.md:280-284` 给的具体数字"~151MB/checkpoint"是**旧硬件/旧配置
下的数字**，Track B 落地时必须用当前 checkpoint 的真实层数/维度重算，不能直接抄）。
vLLM 的"统一 `page_size_bytes`"技巧要求所有资源类型能被压缩进同一个字节粒度的 block——
这对"每前缀恒定 151MB、跟长度无关"的 GDN state 来说会很不自然（把一个固定大小的东西硬套
进"跟着前缀长度增长的块"抽象，是削足适履）。独立预算避免了这个问题，且本仓库自己已经有
一份验证过的实现可以抄读。

### 谁先被驱逐：两条独立 LRU + 双向不对称 lockstep

不是"KV 满了先动 KV，state 满了先动 state"这种全局排序问题，是两条独立 LRU 各自跑，
用**不对称**的双向 lockstep 保证两边不互相说谎：

- **正向（KV 触发）**：一个 KV 块被自己的 LRU/预算驱逐 → **必然**级联丢弃同键的 state
  checkpoint。`gdn_state.py:183-198` 已经这么写。
- **反向（state 触发，预期**更频繁**，因为 state 预算通常远小于 KV 池）**：state
  checkpoint 被自己的字节预算驱逐 → **不**级联丢弃 KV 块，除非该块此刻 `ref_cnt==0`。
  `gdn_state.py:205-209` 已经这么写，注释原话："merely turns a future would-be hit into
  a safe compute miss"。

这个不对称跟 SGLang 的双态设计（`mamba_evicted`/`mamba_backuped`，`:357`）结论一致——
**一个节点的 KV 可以在没有 state 的情况下继续存活**，但**state 不能在没有 KV 哈希锚点的
情况下独立存在**。方向性是设计的一部分，不是疏忽。

**两个预算同时告急时先动哪个**：【事实，本轮未重新核实，标注清楚】
`notes/2026-08-01-hybrid-cache-prior-art.md` 提到 SGLang `HybridCacheController` 有
"KV 池先、额外池后完成才做"的固定顺序，这条本轮**没有**独立重新核实（不同于上面已核实的
`EvictParams`/`hi_mamba_radix_cache.py:357`），按笔记转述先记在这里，若要写进实现，落地前
需重新对照 `hybrid_cache_controller.py` 源码核实。【判断】即使这条不算数，独立推荐 KV
先（自己的 LRU 决定），state 后：KV 是更大、竞争更激烈的资源（本项目 96GB 卡上 KV 与投机
scratch 已经在抢紧张预算，`implementation-plan.md` §7.6 F2），state checkpoint 按已知
数量级（每个几十~百 MB）相对便宜，更适合作为"安全阀"而非"主战场"。

### 谁"更值钱"因而更该被保留

**【判断】** 不引入折算汇率。state checkpoint 是固定字节数、KV 块大小随 `block_size` 和
层数变化，两者的"每字节命中收益"在不同工作负载下差异很大，任何固定汇率都会在某类负载上
系统性偏袒另一类。独立预算 + 各自 LRU 是"不需要预判负载分布就不会错"的选择。

---

## 5. 问题 4：怎么做到对今天只有 KV 缓存的 Laguna 零行为变更

### 为什么这次的"零行为变更"比听起来更容易做到

**【事实，§1.1/1.5 的直接推论】** Laguna 的活代码路径（`reconcile_prefix_hit`/
`find_best_slot_for_prompt`/`find_prefix_match`）**从不调用** `BlockPool`。这意味着
A3 的新机器（协调者、可能的第二分配器）和 Laguna 的既有机器**今天就不共享调用图**——
"零行为变更"不是"需要小心工程才能保证的约束"，接近"结构上自动成立"。而且
`ArchitectureSpec.needs_two_cache_families`（§1.5）对 Laguna checkpoint 解析结果已经是
`False`，并已经过真实 checkpoint 验证（`tests/test_architecture_spec.py`）——这不是本
文档提出的新假设，是已经落地并测试过的事实。

### 论证链

1. 协议要改的两个方法（`reconcile_prefix_hit`/`find_best_slot_for_prompt`）在 Laguna 侧
   的语义变化，只需要让返回值等价于 `PrefixHit(kv_hit=hit, state_hit=hit)`——数值逐字节
   不变。
2. 是否要**改变现有方法的返回类型**，还是**新增一个只在 `needs_two_cache_families=True`
   时才需要的成员**，是本设计里我判断不出唯一正确答案的一个具体分叉——见 §8 决策点 2，
   带两个选项和各自代价，不在这里代选。
3. 协调者作为一层薄适配：对 `needs_two_cache_families=False` 的 checkpoint（今天的
   Laguna），协调者直接转发给 backend 自己的 `reconcile_prefix_hit`，不实例化第二个
   分配器，不触碰 `BlockPool`。真正的第二资源分配器只在 Track B 落地 GDN state 时才
   实例化。

### 具体验证门禁（形式化"零行为变更"，不是口头保证）

- **贪心 bit-exact**（已有的迁移不变量 #1，`architecture.md` §3.3）。
- 现有测试 `test_reconcile_prefix_hit_cold_miss`/`_warm_match`
  （`tests/test_laguna_server_integration.py:143-164`）改成断言 `.effective` 值不变，
  **加上**一条新断言 `result.state_hit == result.kv_hit`——这条本身就是"Laguna 零行为
  变更"的形式化证明，能在 bit-exact 跑完之前先红。
- `find_prefix_match`（DFlash 内部消费者）**不**在本次协议改动范围内，但要有一条测试
  显式核实"没有受影响"，不能只是口头说"应该没事"（§7-b）。

---

## 6. 六条坑逐条对照

对照 `notes/2026-08-01-hybrid-cache-prior-art.md` §6 的六条修改建议（标注哪些本轮
独立重新核实过、哪些沿用笔记原话）：

**1. 不统一分配器。**【已核实，双侧】已经是设计前提。两个独立分配器 + 协调者：KV 分配器
= 现有 Laguna 同槽方案（是否升级到 `BlockPool` 跨槽共享是独立决策，§8 决策点 5，不属于
本次范围）；state 分配器 = Track B 落地时新建的专用 checkpoint 池（对齐
`oracle/gdn_state.py` 的形状，重新实现而非 import oracle）。

**2. block size 对齐 + 反向搜索。**【已核实】§3 已给出"取 `state_hit`，且 state 分配器
搜索范围受 `kv_hit` 约束"的规则。对齐今天没有现成问题（§1.7：64 与 8192 整除），但这是
**当前默认值凑巧对齐**，不是设计上保证永远对齐——【待验证】Track B 真正选定 GDN checkpoint
边界策略时必须重新核实这两个常数的整除关系，不能只在设计文档里 assert 一次就当永远成立。

**3. 投机保守释放。**【已核实，vLLM；SGLang 本轮未找到对应机制，标为未找到而非不存在】
vLLM 的规则（§2 INV-A3 表，`single_type_kv_cache_manager.py:1150-1153`）：释放前减去整个
`num_speculative_tokens` 窗口。SGLang 的 `mem_cache/` 层本轮搜索
`speculative`/`draft`/`EAGLE` 关键词未找到对应机制——**这是"没找到"，不是"确认不存在"**，
可能在没覆盖到的投机 worker 代码里。本仓库退役代码对 state 侧给出了一个更强的替代方案：
`_ssm_spec_row`（`block_pool.py:45-79`）给每个投机候选一条**专属、永不共享**的 state
行，被拒绝的候选的行永不再被读——不需要保守释放预算，直接消除这一类问题。【判断】A3 对
state 侧优先复用这个"专属行"方案（已存在、已单测），vLLM 的"减去投机窗口"规则只留给 KV
侧的块计数场景（Track B 问题，不是 A3 协调者本身要解决的）。

**4. 同轮不可跨请求借用。**【判断，结构性豁免，已核实我们自己的 restore 机制】vLLM 的
"fake shortage，推迟一轮"是因为它的 GDN state 存在于一个共享池里。本仓库固定槽架构下，
每个新请求拿到的是独立物理槽的独立 state buffer，不存在这种竞争。唯一可能出现"同轮多个
请求碰同一份资源"的场景是跨槽 restore——但 `restore_gdn_state`（`gdn_state.py:213-219`，
`torch._foreach_copy_`）做的是**拷贝**，不是指针别名，两个请求在同一轮都从同一份
checkpoint 拷贝恢复不会冲突。这是本仓库既有设计决定带来的结构性豁免，记录"为什么不需要"
比设计一个不需要的机制更重要。这条本身是一个**调度层**约束（谁能不能在同一轮被准入），
不是协调者该拥有的判断——协调者只需要暴露足够信息（比如"这个前缀的 state 边界需要一个
全新的槽"），不该自己拥有准入决策（`ServerEngine` 今天完全没有这个概念，因为它只管一类
资源）。

**5. 独立驱逐预算 + 双向 lockstep。**【已核实，SGLang 与本地代码；vLLM 明确不这样做，
见 §4】答案见 §4。

**6. MTP 是否带 GDN。**【事实，已有结论，不是本文档要回答的问题】
`docs/qwen36-rebuild-spec.md:537-541` 的 B-6 结论："6 个本地 checkpoint 的 `mtp.*`
张量清一色 `self_attn.*`+`mlp.*`，零 GDN"。这条**没有**消除"投机窗口的保守释放"这条难点
（target 的 48 层 GDN 在 verify 时还是要跑、还是要回滚），但确实说明 MTP draft 本身不带
GDN，跟 vLLM 的"current draft models don't have mamba layers"
（`single_type_kv_cache_manager.py:1225-1226`）一致，可以直接复用 vLLM
`reachable_block_mask` 的简化假设。

---

## 7. 分步实施计划

| 子步 | 内容 | 行为变更 | 门禁 | GPU |
|---|---|---|---|---|
| **7-a** | 类型层：`PrefixHit`、协议成员/能力位形状（按 §8 决策点 2/3 的拍板结果），`check_conformance` 更新。INV-A3-6 单测：每个 `needs_two_cache_families=False` 的 backend 在新形状下行为等价 | 无 | `tests/test_backend_protocol.py` 扩展 + `test_reconcile_prefix_hit_*` 改断言（§1.6 已列出会红的具体测试） + ruff | ❌ |
| **7-b** | `engine.py` 调用点更新：`hasattr` 探测换成 `capabilities.prefix_cache` 查询；显式核实 `find_prefix_match`（DFlash 内部消费者）未受影响，补一条覆盖测试而不是口头保证 | 无（重构） | 影子一致性单测 + 一条新的 `find_prefix_match` 回归测试 | ❌ |
| **7-c** | state 分配器骨架：新模块（不是 `block_pool.py`，呼应坑 #1），固定槽、不分页，寻址复用已有的 `_ssm_spec_row`/`_physical_slot`（§1.8 的 `RESERVED_PHYSICAL_SLOTS` 分歧在此显式核实，不默认继承 `=1`）。还没有模型接到它，纯记账逻辑，可用假数据测，参照 `benchmarks/prefix_cache_eviction_check.py` 的纯 Python 检查方法论（`lockstep_eviction`/`refcnt_never_evicted`/`byte_budget` 等，`:96-317`）——搬方法，不 import oracle | 无（没有调用方） | 新单测，CPU-only，无 torch | ❌ |
| **7-d** | 协调者骨架：owns INV-A3-1~5，对 `needs_two_cache_families=False` 的 backend 纯转发（§5 第 3 点） | 无（`needs_two_cache_families` 对今天每个生产 checkpoint 都是 `False`） | 影子一致性单测（协调者转发结果与直接调用 backend 逐字节相等）+ 全套不变量单测（信号探针、压力下准入、字节预算、双向 lockstep——方法论同 7-c） | ❌ |
| **7-e** | 不变量断言接线：把 §2 的 INV-A3-2/3/4/8 写成 `bfdiag/invariants/checks.py` 里新的 `check_*` 函数，跟着 `check_no_duplicate_ids` 的既有模式 | 无 | 单测 + 故意构造违反场景断言真的 `raise`（呼应 C8 纪律） | ❌ |
| **7-f** | 文档措辞修正（§1.3 冲突）——待拍板是否由本任务执行，见 §8 决策点 1 | 无（纯文档） | 无 | ❌ |
| **7-g** | 切换生效：把 7-a 到 7-e 串起来，`ServerEngine` 真正走协调者而不是直连 backend | **有** | `implementation-plan.md` 已定的四条：bit-exact + 接受率 + 前缀命中率不回归 + C-LIVE | ✅ |
| 7-h（Track B 窗口，超出本次范围，仅标记依赖） | 第二资源真正实例化，`(kv_hit, state_hit)` 分裂逻辑落地 | 有 | Track B 自己的门禁 | ✅ |

**为什么这样切**：7-a 到 7-f 全部零 GPU、全部行为不变或影子模式，可以在 GDN kernel 落地
之前全部完成（呼应 A1/A2/A5 前 4 步的既有纪律）。只有 7-g 是真正"有行为变更"的一步，且
此时改动面已被前面步骤压缩到最小——`implementation-plan.md` 标的"有，半径最大"这个风险
标签，准确地说只属于 7-g，不是整个第 7 步。

---

## 8. 待拍板事项（不代选）

### 决策点 1：§1.3 的文档措辞冲突怎么处理

**背景**：`roadmap.md`/`architecture.md` 说"残迹"（暗示待清理），`qwen36-rebuild-spec.md`
（更新一天）说"休眠但设计良好"。
- **(a)** 保留原措辞，按字面执行——删除/重写 `_on_evict_block` 等。代价：丢弃已验证正确
  的代码，纯为措辞一致重新发明。
- **(b)** 改 `roadmap.md`/`architecture.md` 措辞为"休眠原语"，A3 决定何时复用。代价：
  需要碰不属于本任务范围的活文档。

**推荐 (b)**。理由见 §1.3。由谁改、什么时候改，留给你定。

### 决策点 2：`reconcile_prefix_hit` 的返回类型怎么改——原地改，还是新增一个成员

**背景**：今天 `reconcile_prefix_hit -> int` 是 `CAPABILITY_MEMBERS["prefix_cache"]`
治理的协议成员，唯一调用方是 `engine.py:1106`。
- **(a) 原地改**：把返回类型改成 `PrefixHit`，Laguna 的实现内部包一层
  `PrefixHit(hit, hit)`，同一提交里更新 `engine.py` 的调用点（读 `.effective`）。
  好处：一个方法只有一种含义，不产生"两个名字做同一件事"的长期维护面；协议文档自己
  （`protocol.py:42-46`）已经预告"这三个调用点下一次被摸到时"适合改名/改形状，本次正是
  那个时机。代价：这是一次真正的签名变更，理论上任何未来实现了 `prefix_cache=True` 的
  backend 都必须跟着改。
- **(b) 新增成员**：保留 `reconcile_prefix_hit -> int` 不动，新增一个只在
  `needs_two_cache_families=True`（或新能力位）时才要求实现的成员（如
  `reconcile_hybrid_hit -> PrefixHit`）。好处：今天唯一的实现者（Laguna）完全不用碰这个
  方法，改动面最小，严格加法。代价：未来如果 `Qwen36Backend` 同时需要 `prefix_cache=True`
  和第二资源，会同时实现两个语义相近的方法，增加一层认知负担；且"哪个是权威值"需要额外
  文档说明。

**【判断】倾向 (a)**：协议本身已经在文档里预告了这个时机，且今天只有一个实现者、一个
调用方，变更成本很低，不必为一个还不存在的第二 backend 的假设性需求预先加一层。但这条
不是"半径小到我能替你定"的问题——如果 Track B 落地时发现 (a) 的迁移成本比预想的高（比如
第二个 backend 也需要单纯的 `int` 语义作为向后兼容），(b) 随时可以退回去做，代价是 7-a
这一步返工。**如果你倾向 (b) 以保留更大灵活性，直接告诉我，不影响后面 7-c/7-d/7-e 的
设计**。

### 决策点 3："要不要两类缓存"这个信号该放在哪一层

**背景**：§1.5 发现 `ArchitectureSpec.needs_two_cache_families` **已经存在**（Track A
第 3 步落地，专为本步骤预留）。这跟"新增 `BackendCapabilities` 布尔字段"的直觉方案是
两个不同层面的信号：`ArchitectureSpec` 描述**模型结构事实**（从 config.json 解析，与
backend 实现无关）；`BackendCapabilities` 描述**backend 执行层能力**（这个类现在支不
支持某个方法族）。
- **(a)** 协调者只读 `ArchitectureSpec.needs_two_cache_families` 来决定要不要实例化
  第二分配器，不在 `BackendCapabilities` 新增字段。好处：不重复一个已经存在的信号，
  `ArchitectureSpec` 是"权威"来源（Registry 解析出来，构造 backend 之前就知道）。代价：
  `BackendCapabilities` 的"一个 backend 的能力=一份 frozen dataclass"这个心智模型出现
  一个例外——第二资源相关的能力不在这里查，查的人需要知道去 `ArchitectureSpec` 找。
- **(b)** 两个都要：`ArchitectureSpec` 保留原样（权威来源），`BackendCapabilities` 也
  加一个派生字段（构造时从 `ArchitectureSpec` 抄一份），给需要"只查 backend 能力就够"的
  调用方（比如 `/metrics`）一个统一入口。代价：两份数据，需要保证构造时永远同步，增加
  一个"两者不一致"的新失效模式（虽然可以用一条构造期断言消灭）。

**推荐 (a)**，理由：本设计通篇强调"不要发明已经存在的东西"（§1.3、§1.4、§1.5 都是这个
主题的具体案例），`(b)` 引入的"两份数据必须同步"本身就是一个新的、不必要的不变量。但
`/metrics` 目前怎么读能力（§1.6）如果依赖 `BackendCapabilities` 已经是既定接口约定，
可能需要一个小的适配层——这个细节留给 7-a/7-d 落地时处理，不影响这里的架构选择。

### 决策点 4：CUDA Graph 状态中立捕获——新问题，不能从 Laguna 借

Track B/B2 范围，不在本次 A3 决定，列在这里只是不让它在两份文档之间失踪：
`docs/qwen36-rebuild-spec.md:135` 已经指出 Laguna 的 decode 图从不碰递归状态，它的
warmup 复用安全论证不能直接搬过去。退役代码的方案（永久保留 `2 × batch_size` 个 warmup
槽，`cuda_graphs.py:87-130`）可以作为起点，但那份文档自己也说这是"全新问题，无可抄"。

### 决策点 5：A3 是否应该在本轮把 Laguna 的 KV 侧切换到 `BlockPool`（跨槽共享）

- **(a)** 不切换：A3 只处理"加第二类资源"的协调形状，KV 侧维持现状。零行为变更，风险
  最小，符合 `implementation-plan.md` 对第 7 步的判断——"在 Track B 的递归状态到来之前
  它没有真实消费者"。
- **(b)** 顺手切换，一次性把两块基础设施都用上。代价：真正的行为变更（引入跨请求 KV
  共享，今天不存在），不是 Track B/GDN 真正需要的东西，会把"爆炸半径最大的一步"的半径
  进一步放大。

**推荐 (a)**。"切到 `BlockPool`"是一个独立的、有自己收益论证的性能特性，应该走 Track F
或独立评估，不该被 A3 这个结构性协调任务捎带上。

---

## 9. 门禁记录

- **ruff check**（`~/.venvs/vllm/bin/python -m ruff check .`）：`All checks passed!`
- **pytest，CPU-only 环境**（全新 venv，只装 `.[dev]`，不装 torch，模拟 CI job 1）：
  `794 passed, 127 skipped`
- **pytest，CPU-torch 环境**（`~/.venvs/vllm`，`torch==2.13.0a0+gitcf30153`，模拟 CI
  job 2）：`1179 passed, 3 warnings`
- 本文档不改任何 `.py` 文件——上面两条基线在改动前后不会变化；记录它们是为了确认
  worktree 本身处于干净可用状态。系统自带的 `/usr/bin/python3` 环境**不能**用来模拟
  CI job 1（会假报一条 torch 相关测试失败，因为它混了 `~/.local` 下的部分包，不是干净的
  `.[dev]` 安装——跟既有认知一致：CI 只装 dev extras，本地必须另建干净 venv 才能准确
  模拟）。
- **doc-link 检查**：仓库目前没有自动化的 doc-link checker（已搜索 `scripts/`、
  `tests/`、`.github/workflows/ci.yml`，均无匹配）。手动核对本文档内的相对链接，全部
  指向本轮已确认存在的文件：`../notes/2026-08-01-hybrid-cache-prior-art.md`、
  `../notes/prefix-cache-design.md`、同目录下的 `architecture.md`/`roadmap.md`/
  `implementation-plan.md`/`qwen36-rebuild-spec.md`、
  `../docs/archive/2026-07-30-architecture-two-tenant.md`——均确认存在。

---

## 10. 引用索引（便于复核）

- `runtime/backends/laguna.py:39`（唯一 block_pool import）、`:53`
  （`RESERVED_PHYSICAL_SLOTS=0`）、`:2157-2177`（`reset_slot`）、`:2179-2220`
  （`reconcile_prefix_hit`）、`:2222-2259`（`find_best_slot_for_prompt`）、`:2531-2549`
  （`find_prefix_match`）
- `runtime/backends/protocol.py:29-46`（命名债务/预告）、`:60-79`
  （`BackendCapabilities`）、`:192,194-198,235`（`prefix_cache` 治理的成员）
- `runtime/block_pool.py:17-24`（`RESERVED_PHYSICAL_SLOTS=1`/`_physical_slot`）、
  `:23-79`（`_ssm_spec_row`）、`:270-361`（`BlockPool`/`_on_evict_block`/`_evict_one`）
- `runtime/architecture.py:47-48`（cache kind 常量）、`:56-64`（`LayerSpec.cache`）、
  `:126-142`（`paged_kv_layers`/`recurrent_layers`/`needs_two_cache_families`）
- `runtime/model_registry.py:80`（`IMPLEMENTED_BACKENDS`）
- `server/app.py:97`（`SERVER_BLOCK_SIZE`默认 64）、`:1248-1249`
- `server/engine.py:1090-1106`
- `oracle/qwen36_vllm/gdn_state.py:183-226`、`oracle/qwen36_vllm/prefix_cache.py:131-166`、
  `oracle/qwen36_vllm/direct_model_runner.py:590`
- `benchmarks/prefix_cache_eviction_check.py:1-66`（方法论）、`:96-317`（纯 Python 检查）
- `tests/test_backend_protocol.py`、`tests/test_laguna_server_integration.py:143-164`、
  `tests/test_block_pool.py`、`tests/test_architecture_spec.py`
- `docs/roadmap.md:158`、`docs/architecture.md:125,129-131`、
  `docs/implementation-plan.md:195`、`docs/qwen36-rebuild-spec.md:131,144,537-541,561-565,603,677`
- `notes/prefix-cache-design.md`（§3.4、§3.9、§4）
- vLLM `~/.venvs/vllm025/.../v1/core/single_type_kv_cache_manager.py:1026,1042,1064-1065,
  1069-1073,1074-1079,1143,1150-1153,1186,1199-1206,1225-1226`；
  `kv_cache_coordinator.py:161(kv_cache_manager.py),514-680,630,742`；`sched/scheduler.py:690-698,715`；
  `kv_cache_utils.py:1403`
- SGLang `/home/bot/project/sglang/.../srt/mem_cache/allocator/mamba.py:30-35`；
  `base_prefix_cache.py:83-98,155-190`；`hi_mamba_radix_cache.py:357,713-742`；
  `managers/schedule_policy.py:140-141`
