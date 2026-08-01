# eager verify vs CG verify：真实数值分歧（结构机制 + oracle 定性均已确认，见文末两节续查）

**状态（2026-08-02 两轮续查后更新）：**

1. **结构机制已定位并 GPU 确认**：两条路径在 sparkinfer 的 verify-mode split-KV 规划上，
   对同一个真实 kv_len 算出了不同的 KV 分块数（CG 冻结在 1 块，eager 4~16 块），11 个探
   测点里 10 个的"块数是否一致"与笔记原表的"是否分歧"完全对应。见「2026-08-02 续查：结构
   性根因定位」。
2. **oracle 定性已确认，且推翻了第一轮"怀疑 merge kernel 有 bug"的猜测**：建了一个不经过
   sparkinfer 任何 split-KV 代码路径的稠密 causal attention oracle，在单层注意力输出这一
   粒度上直接测——eager 的真实输出、"CG 式"(冻结 chunk size 但用 eager 执行)的输出，两者
   都以 cos≥0.999997 逼近 oracle，在 kv_len=64/400/500 上全部成立。**块数不一致本身不是
   "错误"，两种块数在注意力算子这一层都算对了。** 全模型 logits 层面看到的真实 argmax
   翻转、26.7 的 logit 差，现在看更可能是 **MoE 路由对这类微小差异的放大效应**，不是
   split-KV merge kernel 的正确性缺陷。见「2026-08-02 续查第二轮：oracle 三方对拍」。

按协调者"先想清楚数值判据"的要求：这属于协调者自己列的第三类结论——"三方都在 0.99999+
但 argmax 仍翻转"——**严重性判定为"近似平局位置的翻转"，不是"某条路径的注意力算错了"**，
但下游后果(生产可能真的选错 token)并未因此消失，只是机制变了(MoE 放大 vs 注意力本身算
错)。两节续查的原始证据都保留，不覆盖。

## 背景

修 C-1 的容量 bug（见 [`2026-08-01-c1-c2-gpu-investigation.md`](2026-08-01-c1-c2-gpu-investigation.md)）之后，DFlash 的 eager verify 回退（`DFlashEngine._forward_verify_with_aux`）终于能跑完一次真实调用而不抛异常了。按计划做"贪心位精确交叉验证"（拿同样的输入分别跑 CG-verify 和 eager-verify，比对 logits）时，发现两者在长上下文时输出**不一致，而且不是浮点噪声级别的差异**。

**这比容量 bug 本身更严重**：容量 bug 修复前，eager verify 崩给 `ValueError`——响亮、可见、会被重试挡住。容量 bug 修复后，eager verify 能跑，但跑出来的是**错的 token**——静默，且和"正确"输出在表面上没有任何区别。修复把一个响亮失败变成了沉默失败。

**直接后果**：`QSR_DFLASH_REQUIRE_CG` 的默认值已经从 `"0"`（降级但响亮）改成 `"1"`（拒绝启动）——见 `runtime/backends/laguna_dflash.py` 里 `self._require_cg` 赋值处的大段注释，理由原文搬过去了，这里不重复。

## 触发面确认（读代码，非 GPU 实测）

`_forward_verify_with_aux` 在 `laguna_dflash.py` 里恰好 3 个调用点（`generate_verify_only`/`dflash_round`/`mtp_verify_and_commit_batch` 的等价形态），全部是同一种写法：

```python
if self._verify_cg is not None:
    ... CG replay ...
else:
    ... eager 回退 ...
```

`self._verify_cg` 只在 `_capture_verify_cg()`（`_init_cuda_graph()` 内，构造期调用一次）里被赋值——成功赋成真实对象，失败赋 `None`。**全仓库搜索确认没有第二处赋值**，构造完成后这个值终身不变。没有任何按 shape/batch/kv_len 走的旁路条件。`_partial_verify_cgs`（`bfdiag/daemon/provider.py` 里用 `getattr(..., {})` 防御性引用）在 `runtime/` 里**根本不存在**，是给未来功能占位的引用，不是今天的第二条触发路径。

## `QSR_DFLASH_REQUIRE_CG` 默认值改动的真机端到端验证

用 `scripts/blackwellm_ctl.sh start`（真实 HTTP 服务，不是 `bf daemon`）在本机验证了三种场景，全部符合预期：

1. **默认配置（无任何覆盖）**：服务正常起来（`verify_cg`/`draft_cg` 都正常捕获，跟以往一致）。冷启动 curl `/metrics` 确认 `blackwellm:dflash_cg_captured{graph="draft"} 1` / `{graph="verify"} 1`；发一个真实 `/v1/chat/completions` 请求（200，正常返回），再 curl 一次确认 `requests_completed_total` 计数、`dflash_cg_captured` 数值都稳定不变。
2. **`QSR_DFLASH_DEBUG_FORCE_CG_FAIL=verify`,不覆盖 `QSR_DFLASH_REQUIRE_CG`（新默认值 `1` 生效）**：服务**启动失败**——traceback 从 `_do_capture` 一路原样往上传（`_attempt_cg_capture` → `_capture_verify_cg` → `_init_cuda_graph` → `DFlashEngine.__init__` → `LagunaBackend.enable_dflash` → `ServerEngine._load_laguna_model`），FastAPI 的 `lifespan` 捕获到后打 `Application startup failed. Exiting.`。之后 `pgrep` 确认没有 `server.app` 进程，curl `/health` 拿到 `000`（连接被拒），`nvidia-smi` 确认显存已经完全释放（回到 ~1.5GB 基线,没有半加载状态残留)。**"拒绝启动"这条路径端到端确认生效。**
3. **`QSR_DFLASH_DEBUG_FORCE_CG_FAIL=verify QSR_DFLASH_REQUIRE_CG=0`（显式选择退回旧行为)**：服务正常起来,启动日志里能看到 `logger.error`（带 `exc_info`)明确写着"verify CUDA Graph capture failed -- degrading to verify's eager fallback...";curl `/metrics` 确认 `{graph="verify"} 0`、`{graph="draft"} 1`（只有 verify 被强制失败，draft 不受影响);发一个短 prompt 的真实请求,200 正常返回（**这只证明"没崩",不代表输出正确**——这次调查已经证明 eager verify 在 kv_len≥400 时输出会跑偏，这个请求的上下文远小于那个阈值,所以短请求成功不能反证那个分歧不存在)。

三个场景加起来把"新默认值真的按预期工作"这件事从"读代码觉得应该对"变成了"真机确认过"。

**结论：eager verify 今天只有一条触发路径——verify CG 在启动期捕获失败（或 `QSR_VERIFY_CUDA_GRAPH=0` 主动关闭）。** 这是一个**潜伏风险**，不是"正在发生的活跃故障"：本次会话里每一次冷启动，verify CG 和 draft CG 都捕获成功（daemon 启动日志反复确认），今天的生产服务没有理由正在通过这条路径服务任何请求。风险在于：一旦某次启动 verify CG 捕获失败，在这个数值 bug 修好之前，`QSR_DFLASH_REQUIRE_CG=1` 的默认值会让服务拒绝启动而不是带着这个缺陷继续跑。

## 实测证据

GPU：RTX PRO 6000 Blackwell Max-Q。worktree `work/gpu-20260801`。`bf daemon` 生产等价配置（`block_size=64`、`blocks_per_slot=4096`、`num_slots=3`，CUDA Graph/DFlash 默认开）。

### 方法

两次独立测试，结论一致：

1. **同 slot 先后调用**：`backend.prefill_with_aux(0, ...)` 建立真实 kv_len，然后先调 `engine._verify_cg.replay_with_aux(0, verify_tokens, kv_len)`，再调 `engine._forward_verify_with_aux(0, verify_tokens, kv_len, 16)`，比较两次的 16×vocab logits。
2. **双 slot 隔离**（排除"第一次调用写了 KV，第二次调用读到被污染的状态"这个假设）：slot 0 和 slot 1 各自跑**完全相同**的 prefill（先 assert 两个 slot 的 `prefill_with_aux` 首 token 一致、kv_len 一致），CG 走 slot 0，eager 走 slot 1，两者互不接触。

两种方法结果一致，说明不是测试脚本的同 slot 副作用问题。

### 结果（双 slot 隔离，`verify_tokens = [bonus_token] + [11]*15`，16 个位置逐行比对）

| kv_len | 逐位 bit-exact | argmax 是否一致 | 峰值 raw logit 差 |
|---:|---|---|---:|
| 64 | ✅ 完全一致 | ✅ | 0 |
| 400 | ❌ | ❌（多个位置选错 token） | 5.80 |
| 500 | ❌ | ❌ | **26.69** |
| 510 | ❌ | ❌ | 7.58 |
| 511 | ❌ | ❌ | 8.44 |
| 512 | ❌ | ❌ | 7.90 |
| 513 | ❌ | ❌ | 6.77 |
| 520 | ❌ | ❌ | 5.28 |
| 600 | ❌ | ❌ | 5.62 |
| 1000 | ❌ | ❌ | 7.31 |
| 2016 | ❌ | ❌ | 5.20 |

**分界点在 64 和 400 之间，不是 512（SWA window）**——512 前后（510/511/512/513）没有观察到任何"跨过窗口边界就变化"的特征，差异幅度在整个 400+ 区间大致同一量级、没有单独在 window 边界处跳变或突变。

第 0 行（对应 anchor/bonus token 自身的预测，只依赖已缓存的上文，不依赖同批次其它 15 个 verify token 之间的因果注意力）在多个 trial 里也观察到不一致（例如 kv_len=511 时 argmax_cg[0]=923 而后续 kv_len 也有第 0 行不一致的情形），所以**不能简单归因为"只有 token 间因果注意力那部分算错"**——需要更细的定位才能下结论。

## 已排除的假设

- **SWA window（512）对齐问题**——最初的怀疑，因为最早一版非隔离测试里 kv_len=2016/4097 都在 window 之上。**排除**：kv_len=400（远小于 512，window 根本不截断任何东西）已经不一致；510/511/512/513 四个紧贴边界的点没有表现出任何"跨过边界才变化"的模式。
- **测试脚本的同 slot 副作用（第一次调用写 KV，污染第二次调用读到的状态）**——排除：双 slot 隔离测试（两个 slot 各自独立 prefill，互不接触）复现了同样的分界点和幅度。
- **positions 不一致**——排除：读代码确认两条路径都用 `arange(kv_len, kv_len+num_tokens)`，逐行核对一致。
- **我这次 C-1 容量修复本身引入的问题**——排除的把握没有前两条那么绝对，但可能性很低：容量修复只改了 `SparkinferPrefillWorkspace` 建 workspace 时的 scratch buffer 大小（`max_work_items`/`max_partial_rows`），不改变给定 `(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode, window_left)` 时 `create_paged_plan` 实际算出的调度本身——该调度是这几个入参的纯函数，与 workspace 的 scratch 容量大小无关。而且这条 eager 路径在这次修复之前**从未成功跑完过一次完整调用**（每次都在 `_ensure_capacity` 报错崩掉），所以这个数值分歧很可能是一直存在、从未被观察到的——不是这次修复"引入"的，是这次修复"揭露"的。

## 还没查、留给下一个人的方向

- **逐层定位**：用 `bf divergence`（`docs/diagnostics-guide.md` 里现有工具）在 kv_len=64（一致）和 kv_len=400（不一致）之间找第一个发散的层/子模块。
- **split-KV 归并怀疑**：`create_paged_plan(mode="verify", enable_cuda_graph=False)` 在这些 kv_len 下 `split_kv=True`（本次调查另一部分测过，见 C-1 笔记里"real eager planner" 一节的 7 点扫描），意味着攻击力按多个 KV chunk 切分后要做部分结果归并。CG 路径捕获时用的是**同一个 kernel**、但**不同的调度构造方式**（`enable_cuda_graph=True` 下的 `create_paged_plan`，走 `_ensure_capacity` 自动扩容而不是本次修复引入的"预先按最大容量跑一次真实 plan"）。两条路径的 split-KV 归并顺序/切块边界如果不同，理论上可能产生这个量级的数值差异——**但这只是一个待验证的假设，不是结论**。
- **第 0 行也不一致**这一点值得单独确认：如果连"只依赖已有上文、不依赖同批 15 个 draft token 互相因果注意力"的那一行都对不上，说明问题可能不在"16 个 token 内部怎么互相看"，而在更基础的"怎么读已有 KV"这一层——这会大幅收窄排查范围。
- 排查时优先复用双 slot 隔离这个方法（已确认排除了副作用类假设），不要走回同 slot 先后调用的路子。

## 相关

- 触发面判断、`QSR_DFLASH_REQUIRE_CG` 默认值改动：`runtime/backends/laguna_dflash.py`（`_attempt_cg_capture` 与 `DFlashEngine.__init__` 里 `self._require_cg` 的注释）。
- C-1 容量修复本身（不是这份笔记的主题，只是让这条路径第一次跑得动）：[`2026-08-01-c1-c2-gpu-investigation.md`](2026-08-01-c1-c2-gpu-investigation.md)。

---

## 2026-08-02 续查：结构性根因定位

`work/diverge-20260802`，先做零 GPU 代码对比、把假设收窄到一条后才上卡（GPU 用于验证
一个不需要加载模型的纯 planner 探针，见下）。

### 结论先行

两条路径都会调用 sparkinfer 的 `planner.create_paged_plan(mode="verify", ...)`，但
**调用时机、`enable_cuda_graph`、`cache_seqlens` 三者都不同**，导致两条路径对**同一个
真实 kv_len**算出的 split-KV **KV 分块数（`num_chunks_kv`）系统性不同**：

- **CG**（`LagunaCudaGraphVerify._init_workspaces`，`laguna_cuda_graph.py:709-724`）在
  construction 期只调用一次 `create_paged_plan`，且故意用**整个 workspace 的最大容量**
  当 `cache_seqlens`（`max_kv = (blocks_per_slot+16)*block_size-1`），不是真实 kv_len。
  算出的 `plan.kv_chunk_size` 被写进一个设备端 buffer（`kv_chunk_size_ptr`），此后**每次
  replay 都原样读回，从不重算**——sparkinfer 里唯一能在 replay 时重新按真实 kv_len 算
  chunk size 的机制是 `PagedAttentionWorkspace._uses_laguna_verify_analytic_schedule` /
  `update_prefill_graph_work_metadata_triton` 的 `ADAPTIVE_CHUNKING` 分支，而它硬编码只
  在 `cta_tile_q==64`（等价于 verify 查询窗口正好 8 token、GQA6）且 `page_size==128` 时
  才生效——本项目的真实配置是 `NUM_QUERY_PER_REQ=16`（`dflash_constants.py:9`，不是 8）、
  生产 `block_size=64`（`server/app.py` 的 `SERVER_BLOCK_SIZE` 默认，不是 128），两条都
  不满足，所以这条"自适应重算"分支在这个部署上**从未生效过**——CG 的 chunk size 对整个
  进程生命周期都是那个"给最大容量算出来的"冻结值。
- **eager**（`SparkinferPrefillWorkspace.forward`，`laguna_sparkinfer_attn.py:499-509`，
  被 `_forward_verify_with_aux` 经 `SparkinferAttentionImpl.forward` 调用）在**每次真实
  调用时都重新算 plan**，`enable_cuda_graph=False`，`cache_seqlens` 是**真实当前
  kv_len**，且预算（`max_batch_size_if_split`）基本不设上限，所以二分搜索
  （`_prefill_binary_search_kv_chunk_size`）总能选到 sparkinfer 允许的最小 chunk size
  （`min_kv_chunk_size = max(128 // page_size, 1)`）。

两边的 `force_split_kv` 都是 `True`（`mode=="verify"` 时的默认值，
`planner.py:1853-1854`，与 `enable_cuda_graph` 无关），所以 `split_kv` 标志本身在两条路
径上一直相同——**真正分歧的是 chunk *size*，进而是 chunk *数量***：chunk 数量为 1 时，
"split" 退化成单块、单次 online-softmax、无 merge，数值上应等价于非 split；chunk 数量
>1 时，才会真正跑多块 + 跨块 merge 的代码路径。

### 怎么确认的（GPU，但不需要加载模型）

`create_paged_plan` 是纯规划函数,只读张量的 shape/dtype/device 和
`cache_seqlens`/`cu_seqlens_q` 的整数值,不读 K/V 内容——不需要真实权重、不需要
Laguna router、不需要完整 `LagunaBackend`。新增
`bfdiag/workloads.py::diagnose_dflash_verify_split_kv_chunking(engine)`
（永久诊断函数，不是一次性脚本，遵照本仓库 `bf exec` 投热引擎的约定）：直接用真实生产
形状（`num_q_heads=48, num_kv_heads=8, head_dim=128,
kv_dtype=float8_e4m3fn, block_size=64, blocks_per_slot=4096, window_left=-1`,
`NUM_QUERY_PER_REQ=16`）构造 synthetic CUDA 张量,分别调用一次"CG 式"（
`enable_cuda_graph=True, cache_seqlens=[max_kv]`,精确复刻
`LagunaCudaGraphVerify._init_workspaces`)和 11 次"eager 式"（
`enable_cuda_graph=False`,`cache_seqlens=[kv_len]`,kv_len 取上表原始探测点
64/400/500/510/511/512/513/520/600/1000/2016)。

实测数字（`bf ls` run_id `618876c9dcaa`/`4a65871ca2ee`，两次独立运行结果一致，
`bf diff` 确认 config 与两个关键指标零漂移）：

**勘误（2026-08-02 续查第二轮发现并修正）**：下面两个数字最初用错了 head count——
误信了 sparkinfer 自己代码里"Laguna q=8/GQA6 verifier"那条注释里写的
`num_q_heads=24, num_kv_heads=4`，那其实是 sparkinfer 给 **TP=2 分片**场景标注的形状提示，
本项目 TP=1、真实形状是 `docs/model-support.md` 写明的 `48 Q heads / 8 KV heads`
（full-attention 层，`gqa_group_size=6` 两边算出来一样，只是绝对头数不同）。用正确头数
重跑（`run_id` 见下一轮的 `4b31962b7894` 之前的 split-KV 探针复算）：`cg_kv_chunk_size`
从 17600 变成 **37632** token（`chunk_pages` 从 275 变成 **588**）。**结论(哪些 kv_len
不一致)完全不变**——37632 token 依然远超所有已测长度，CG 冻结值套到任何已测 kv_len
块数还是 1，下表逐点对照关系原样成立，只是这两个绝对数字本身是错的，这里更正,不隐藏。

- CG 冻结的 `kv_chunk_size=37632` token（`chunk_pages=588`，用整块容量
  `max_kv=263167` 算出来的，且 `cta_tile_q=16` 不是 sparkinfer 那个 M64 特快路径的 64）。
- eager 每次都选到 `kv_chunk_size=128` token（sparkinfer 允许的最小值，2 页）。

| kv_len | eager 算出的 `num_chunks_kv` | CG 冻结值套到这个 kv_len 会是多少块 | 是否不一致 | 原笔记该点是否分歧 |
|---:|---:|---:|:---:|:---:|
| 64 | 1 | 1 | 否 | 否（bit-exact） |
| 400 | 4 | 1 | **是** | 是 |
| 500 | 4 | 1 | **是** | 是（26.7 那个点） |
| 510 | 4 | 1 | **是** | 是 |
| 511 | 4 | 1 | **是** | 是 |
| 512 | 4 | 1 | **是** | 是 |
| 513 | 5 | 1 | **是** | 是 |
| 520 | 5 | 1 | **是** | 是 |
| 600 | 5 | 1 | **是** | 是 |
| 1000 | 8 | 1 | **是** | 是 |
| 2016 | 16 | 1 | **是** | 是 |

**11 个探测点，"块数是否一致"与原表"是否分歧"逐点完全对应**（1 个一致对应
bit-exact，10 个不一致对应 10 个分歧点，包括 26.7 那个尖峰）。CG 一侧无论 kv_len 多大，
只要不超过 chunk size 对应的 ~37632 token（远超所有已测长度)，块数永远是 1——这也正好
呼应上一轮笔记里 `laguna_cuda_graph.py` 类文档注释自己写的"图用的是非 split 的 extend
规划,尽管这在逻辑上是一次 verify forward"：`split_kv` 标志位是 `True`，但块数恒为 1
让它在实践中等价于非 split。

**这也顺带解释了上一轮"排除 SWA window 512 对齐"为什么排除得对**：SWA 层是独立的
group，它的 CG 容量上限是 ring buffer 大小（远小于 full-attention 层的
`blocks_per_slot=4096`），分歧真正跟着走的是 **full-attention 层组的容量**（4096
blocks≈262144 token），跟 512 这个 SWA 窗口毫无关系——原笔记"510/511/512/513 没有
表现出跨窗口才变化的模式"这条观察，在这个新根因下是预期结果，不是巧合。

### 还没定论的部分（诚实说明，不是回避）

块数不一致本身是**结构性事实**（GPU 探针直接读出来的规划器数字，不是推测），但它到底
是不是"数值分歧的**唯一**放大因素"——即 sparkinfer 的多块 split-KV merge kernel 在真
实跑起来时是否存在正确性缺陷——我**没有**在这次续查里直接确认。原因：

- 本 worktree 目前**加载不了 Laguna 模型**——`runtime/laguna_router.py:21` 的
  `TARGET_SM = "sm_120a"` 与 `Makefile`（`build-laguna-router` 目标，commit `d9b635e`
  "Fix SM120 router gencode family mismatch" 之后）生成的 manifest 里
  `"target_sm": "sm_120f"` 不一致，`LagunaRouterLibrary.load()` 直接抛
  `LagunaRouterError: Laguna router target mismatch: expected sm_120a, got sm_120f`，
  daemon 启动即崩（`.bfdiag/daemon.log` 有完整 traceback）。`git show d9b635e --
  runtime/laguna_router.py` 确认那次提交只改了 `Makefile`，没有同步改这个常量——这是一个
  **与本次分歧调查无关、但看起来会挡住这个分支上任何人加载 Laguna 模型的独立问题**，值
  得单独排期，我没有动它（改一行常量能绕过，但那是"改无关代码"，超出这次任务授权，留给
  你判断由谁修、怎么测）。
  这也是为什么本次确认停在"sparkinfer planner 输出的数字"这一层，没能再往下做一次端到
  端 logits 级别的交叉验证（本来想在原笔记表中"64 和 400 之间"找一个 eager 刚好从 1 块
  变 2 块的 kv_len，比如 129~256 附近，直接测那里是否已经开始分歧，作为比"块数相关"更强
  的因果证据——这个实验被上面的 router 问题挡住了，没做）。
- 块数不一致**不等于** merge 一定错——如果 sparkinfer 的多块 merge（log-sum-exp 合并）
  实现是对的，块数不同应该只造成 bf16/fp8 舍入顺序级别的噪声（历史上这个仓库测过的正确
  kernel 都在 cos=0.999999 量级，见 `notes/2026-07-27-bfdiag-oracle-divergence.md` 的
  阈值论证表）。但原笔记测到的是**真实 argmax 翻转、单点 logit 差 26.7**——这已经远超"仅
  仅是舍入顺序不同"的量级。我倾向于认为这指向 merge kernel（或其与 fp8 K/V descale 的
  交互）在真正被多块路径覆盖时存在缺陷，而不是良性噪声被意外放大——但这只是基于量级的
  推断，不是我直接证实的 kernel 级证据。这部分是 sparkinfer 自己的代码（`/home/bot/
  project/sparkinfer`），按约定我只读不改；有能力判定的应该是 sparkinfer 团队，或者拿到
  一个真实 dense-attention oracle（`bf divergence` 设计的用途，但它自己的 GPU 采集路径
  也从未跑过，见 `notes/2026-07-27-bfdiag-oracle-divergence.md` 第5节）后端到端比对。

**已排除**（本次续查新增，不重复上一轮已排的）：
- **不是** `cta_tile_q` 不同——两条路径对这个真实形状（`packed_qo_len=96`）都算出
  `cta_tile_q=16`，不是 sparkinfer 那个为 `NUM_QUERY_PER_REQ=8` 调的 M64 特快路径（该
  路径需要 `packed_qo_len==48` 且 `page_size==128`，两条都不满足）。
- **不是** `split_kv` 标志位本身不同——两条路径的 `force_split_kv` 都是 `True`
  （`mode=="verify"` 时的 sparkinfer 默认值，`enable_cuda_graph` 不参与这个判断）。真正
  不同的是 chunk size/数量,不是这个布尔位。

### 修复方案（未实施，报给你判断这轮修还是单独排期）

三个方向，风险递增，但"确定性"也递增：

1. **让 eager 也退化成单块**（改 `laguna_sparkinfer_attn.py` 里 eager 调
   `create_paged_plan` 那处，传一个足够大的 `fixed_split_size` 或
   `disable_split_kv=True`，强制 eager 在 verify 模式下也只用 1 块）。
   - 范围最小、风险最低——只改本仓库一个调用点的一个参数，不碰 sparkinfer。
   - 能让两条路径**互相一致**，但**不能证明"单块"这个算法本身相对真实 dense attention
     是对的**——如果单块路径本身有别的、更隐蔽的问题，这个修法只是让两条路径"一起错"而
     不再报警。
2. **让 CG 也能按真实 kv_len 重算 chunk size**（把 `ADAPTIVE_CHUNKING` 那条门槛放宽到
   本项目的真实形状，或者在 `update_prefill_graph_replay_metadata` 里补一条适配本项目
   `cta_tile_q=16` 的重算路径）。
   - 这会让 CG 也走多块 merge——如果 sparkinfer 的多块 merge 真的有缺陷，这个修法反而
     会把缺陷带进生产路径（今天生产路径靠"冻结成单块"意外躲开了它）。
   - 需要改 sparkinfer 或在 `laguna_cuda_graph.py` 里重新实现一套等价逻辑，工作量和风险
     都明显更大，而且没有回避掉"到底哪个块数是对的"这个前置问题。
3. **先做一次端到端 oracle 对拍**（真实 dense attention/bf16 参考 vs 两条路径各自的输
   出），确定单块和多块哪个（如果有任一个）更接近真值，再决定往 1 还是往 2 修。
   - 最扎实，但工作量最大——需要先解决 router 加载问题，可能还需要 `bf divergence` 补
     上它自己从未跑过的 GPU 采集路径（见上），或者一个独立的纯 sparkinfer 单测（构造已
     知答案的小 attention 问题，绕开整个 Laguna 模型）。

### 是否影响 `QSR_DFLASH_REQUIRE_CG` 默认值

**不影响，而且这次续查的发现进一步支持保留 `1`（拒绝启动）这个默认值。** 现在有了更具
体的机制：eager 一旦被真正触发（verify CG 捕获失败或被主动关闭），在 kv_len 大到需要
2 块以上时会走一条**生产路径几乎从未跑过**的 sparkinfer 代码（多块 split-KV merge）——
这条代码路径本身尚未确认对错。在查清楚"单块/多块哪个对"之前，让服务在 CG 捕获失败时继
续跑而不是拒绝启动，等于把一条未经生产验证的代码路径悄悄送上线。

### 本次改动

- `bfdiag/workloads.py`：新增 `diagnose_dflash_verify_split_kv_chunking(engine)`——永久
  诊断函数（不是一次性脚本），后续任何人都能直接 `bf exec` 调用复核上表数字，或换
  `probe_kv_lens` 探测新的 kv_len。
- 本文件：更新头部状态行，追加本节。上一轮的原始证据与排除清单未做任何修改。

**下一节（第二轮续查）用真实 GPU oracle 数据部分推翻了上面"多块 merge kernel 本身尚未确
认对错"这句话背后的怀疑方向——不是删掉这段，是接着往下查清楚了，见下。**

---

## 2026-08-02 续查第二轮：oracle 三方对拍（协调者指派，方向 3）

### 背景：协调者的指令

上一节交完之后，协调者：(a) 确认并修复了本节撞到的 router `TARGET_SM` 不匹配问题（main
commit `95f0db2`，本 worktree 已 `git merge origin/main` 拉取，问题已解决，不再是阻塞
项）；(b) 明确要求**先做 oracle 判定哪边对**，不是直接二选一押注方向 1 或方向 2——理由是
"CG 用整个 workspace 最坏容量规划出来的调度恰好对真实长度也正确，这件事不该被默认，需要
被证明"。同时给了三条判定分支和一条数值判据（"三方都在 0.99999+ 但 argmax 翻转"要单独
定性，别和真实 bug 混为一谈）。

### 方法：为什么不能直接照搬"整模型三方对比"

协调者最初建议的做法是：拿一个不走 paged/split-KV 的稠密 attention oracle，替换掉**整个
模型**的注意力实现，跑一次完整 verify forward，在**最终 logits**这一层跟 CG/eager 三方
对比。第一次尝试就是这么做的（见下面"先做的、后来发现有问题的整模型三方对比"），但结果
不可信,理由和过程记录在那一小节,不删,只是不能拿它下结论。

**结论：整模型 logits 层面的三方对比被 MoE 路由的放大效应污染，测不出这次要测的信号。**
改用**单层注意力算子输出**这一粒度的三方对比，绕开 MoE，才拿到干净的答案（见"改进方法"
一节）。

### 先做的、后来发现有问题的整模型三方对比

新增 `bfdiag/dense_attention_oracle.py::DenseCausalOracleImpl`——从零手写的 fp32
matmul→mask→softmax→matmul，完全不调用 sparkinfer 的 `create_paged_plan`/
`paged_attention_forward`，只支持这次调查需要的范围（`window_left=-1` 全注意力组、
batch=1、fp8 e4m3 KV）。配套 `bfdiag/workloads.py::diagnose_dflash_verify_dense_oracle`：
把这个 impl 临时挂到全注意力层组的 `BFAttention.impl` 上，跑一次真实 `_forward_verify_
with_aux`，拿到"oracle 版"整模型 logits，再分别与 CG/eager 的整模型 logits 比较。

GQA 头映射的校准：sparkinfer 真实 CuTe kernel 的 query-head→kv-head 映射规则在 Python 
层不可见，所以在 kv_len=64（已证明 CG/eager 之间 bit-exact 的点）用两种候选映射
（`repeat_interleave`/HF 标准 vs `repeat`/blocked）各跑一次 oracle，选择与 CG/eager 共识
更接近的一个——这个校准机制本身就是为了不要在没验证过映射对不对的情况下直接下结论,而它
恰好在第一次跑的时候就报警了。

**实测（`run_id=4b31962b7894`，daemon 加载真实 Laguna-S-2.1，`provider=laguna
block_size=64 blocks_per_slot=4096 num_slots=3`）：**

| | kv_len=64 | kv_len=400 | kv_len=500 |
|---|---:|---:|---:|
| cg vs eager cosine | 0.99999999999999989 | 0.9505 | 0.9612 |
| cg vs oracle cosine | 0.8449 | 0.9564 | 0.9446 |
| eager vs oracle cosine | 0.8449 | 0.9569 | 0.9509 |

**校准步骤本身就是报警信号**：kv_len=64 是上一轮证明过 CG 与 eager bit-exact
（cos=0.999999999999999989）的点。**在这个点上,oracle（无论哪种 GQA 映射,`interleaved`
=0.8449,`blocked`=0.1693,均远低于共识）同时偏离 CG 和 eager 两者**——CG 和 eager 独立
实现、通过不同代码路径，却在这一点上完全一致，如果 oracle 在这个双重印证过的点上还对不
上，最合理的解释是 **oracle 自己有 bug**，不是"CG 和 eager 恰好共享同一个错误"。没有在这
个信号面前继续往下比 kv_len=400/500（那样会是在一个未验证正确的 oracle 上盖结论），转去
定位 oracle 的 bug——定位过程见下一节。

### 定位 oracle 的 bug：不是 oracle 数学本身，是"整模型换 12 层"这个操作本身

单层隔离测试（不接回模型，纯 Python 里对同一份真实捕获的 Q/K/V/KV-cache/attn_metadata
分别跑"真实 eager kernel"和"oracle 数学"，两者互不影响对方）：

```
layer0.scale: 0.08838834764831845  (== 128**-0.5，跟假设一致)
real_output vs oracle_output (仅 layer 0, kv_len=64):
  max abs diff: 0.0009765625
  cosine:       0.999998927116394
```

**oracle 数学本身是对的**（GQA 映射用 `interleaved`，跟 HF/vLLM 的 `repeat_kv` 惯例一
致）。问题出在"把 oracle 同时接到全部 12 个全注意力层、跑一次完整 48 层模型"这个操作
上——即使每一层 oracle 自己的输出跟真实 attention 只差 0.999999 量级的噪声，48 层里有 47
层是 MoE（"layer 0 为稠密"，其余 47 层 top-10/256 路由），这个仓库自己的历史标定数据早
就记录过 **MoE 单层的"正确基线"本身只有 cos≈0.95~0.97**
（`notes/2026-07-27-bfdiag-oracle-divergence.md` 阈值表）——远比 attention 层的
0.999999 噪声大。12 层全注意力同时换成一个数学上等价但字节不同的实现,产生的噪声会被
后面 47 层 MoE 的路由决策(离散的 top-10 选择,不是连续函数)放大——这就是为什么哪怕
oracle 数学本身没问题,整模型 logits 层面的三方对比还是测不出"块数不一致"这个信号:噪声
底噪(MoE 放大 x "任意实现差异")已经比信号(块数不一致的具体影响)大。

**这本身是一条独立、值得记录的方法论发现**：CG 与 eager 之间任何"看起来只是等价重写"的
差异（哪怕舍入误差量级），一旦同时触及多层全注意力，經过 MoE 就可能在最终 logits 上炸开
成看起来很严重的分歧——这解释了为什么第一轮"结构性分歧"调查里 26.7 这么大的 logit 差不
一定意味着 split-KV merge kernel 本身错得那么离谱。

### 改进方法：单层注意力算子输出，绕开 MoE

对**同一个真实、未经任何改动的 eager verify forward**，用一个 capture 包装器 hook 住全
注意力组第一层（`model.layers.0.self_attn.attn`，`num_q_heads=48 num_kv_heads=8`，确认
是这次调查关心的那个 full-attention 组），拿到该层真实的 `(query, key, value, kv_cache,
attn_metadata, 真实输出)`。**不改模型、不接回任何东西**——纯粹在这些捕获到的张量上，另外
起算三份独立的"如果用这份输入,分别用 CG 式/eager 真实/oracle 会算出什么":

- **eager 真实**：就是刚才捕获到的真实输出，来自生产代码，未改动。
- **"CG 式"**：既然 CG 的真实 CUDA Graph replay 完全绕开 Python 的 `impl.forward` 分发
  （替换 `.impl` 对它毫无影响，第一轮笔记已经确认过这点），没法直接 hook。改用等价手
  段：直接调 `sparkinfer.attention.paged.planner.create_paged_plan(..., enable_cuda_
  graph=False, fixed_split_size=<CG 冻结的 chunk_pages=588>)`——把 CG 冻结的分块大小,通
  过 `fixed_split_size` 这个参数,固定给一次**非图捕获**的普通 eager 执行。这在数学上精
  确复现 CG 会算出来的分块方式（`force_split_kv` 对 `mode="verify"` 恒为 `True`，跟
  `enable_cuda_graph` 无关，见第一轮笔记，所以 `fixed_split_size` 之后两边分块策略完全
  一致），但不需要真的建一次 CUDA Graph。
- **oracle**：`DenseCausalOracleImpl`，直接喂同一份捕获张量。

**实测（真实 GPU，layer `model.layers.0.self_attn.attn`）：**

| kv_len | eager_vs_oracle cos | cg_style_vs_oracle cos |
|---:|---:|---:|
| 64 | 0.999999 | 0.999999 |
| 400 | 0.999997 | 0.999999 |
| 500 | 0.999998 | 0.999999 |

**在单层注意力算子这个粒度上，eager 的真实多块输出和"CG 式"单块输出，都以 0.999997 以上
的 cosine 逼近稠密 oracle——两者都在这个仓库历史标定过的"正确 kernel 变体"量级
（≥0.999991，见 `notes/2026-07-27-bfdiag-oracle-divergence.md`），彼此之间也互相印证。**
块数不一致（1 块 vs 4 块）这个第一轮确认的结构性事实，在**注意力算子输出**这一层，并不
对应任何超出正常 bf16/fp8 舍入噪声的数值差异——两种分块方式对这个真实输入来说都"算对了"。

### 判定（对照协调者给的三条分支）

**落在协调者列的第三类："三方都在 0.99999+，但（全模型层面）argmax 仍翻转"——是近似平局
位置的翻转，不是某条路径的注意力本身算错了。** 不是"CG 对 eager 错"，也不是"eager 对 CG
错"（那两条分支都要求单层输出本身就有真实、非噪声级的差异，这次没有测到）。全模型 logits
层面 26.7 的 logit 差、真实 argmax 翻转，最可能的机制是：CG 和 eager 各自在**多个**全注
意力层上都带着这类量级的实现差异（不只是块数——分块方式变了,内部的浮点求和顺序、
fp8 反量化的具体算子调用顺序都会跟着变,块数只是这次能明确抓到、能复现、能解释边界位置
的那一个具体机制)，这些差异逐层累积后，被后面 47 层 MoE 的离散路由决策放大，才炸成了全
模型层面看起来很严重的分歧。

**没有做到、如实说明**：

- 只验证了全注意力组的**第一层**（`model.layers.0`），不是全部 12 层。原计划扫全部 12
  层 × 3 个 kv_len，但用于复现"CG 式"分块（`fixed_split_size` + 手搭
  `PagedAttentionWorkspace.for_fixed_capacity` + `build_paged_attention_binding` +
  `paged_attention_forward`）的那段辅助代码疑似有容量估算的 bug——两次尝试跑更大范围时
  分别撞到 CUDA OOM 和 `cudaErrorIllegalAddress`（后者是真正的越界写，把 daemon 的
  CUDA context 直接搞坏,必须整个重启进程才能恢复），具体细节见下面"过程中的两次 GPU 故
  障"。这段辅助代码只存在于本次会话的 scratchpad 调试脚本里，从未提交进仓库，不留作遗
  留 bug，但也意味着"12 层里只有第 1 层被真正验证过"这句话是诚实的限定，不是可以忽略的
  细节——不能排除更深的层（尤其是排在后面、输入已经被前面层的微小差异间接影响过的层）表
  现不同。这是本轮续查唯一没有闭环的地方，留给下一个人（或下一次会话）用一个先修好容量
  估算再跑、或者干脆换用"只调 `create_paged_plan` 读 `.new_batch_size` 不实际执行 kernel"
  这种零风险方式去扫完剩下 11 层。
- kv_len=64 这一点上，本节表格里 `eager_vs_oracle`/`cg_style_vs_oracle` 的具体数字确认
  可信（跟上面"定位 oracle bug"那节独立、更早的单层校验交叉印证，两次不同脚本、两次独立
  运行给出同一个 0.999999），但同一次运行里另一个字段（当时打印成"eager_num_chunks_kv"
  的那一列）实际读的是"CG 式"分支自己的分块数，标签写错了，不是真的又测出了一个新事实——
  已在下一次运行里修正标签,但那次运行在修正后的第一次真实调用上就撞上了下面说的
  `cudaErrorIllegalAddress`，没有拿到修正后的真实 eager 分块数打印结果。这不影响上面表
  格里两个 cosine 数字的可信度（那两个数字的计算路径没有受这个标签 bug 影响），只是意味
  着"eager 在 layer 0、kv_len=400/500 时真实用了几块"这个具体数字，本轮没有单独用这层
  真实数据重新打印确认——但第一轮笔记的 `diagnose_dflash_verify_split_kv_chunking`
  已经确认这是纯 shape/dtype 决定的规划结果、与具体是哪一层无关，所以间接确信度仍然很高。

### 过程中的两次 GPU 故障（如实记录，不是绕过不提）

1. **CUDA OOM**：第一次尝试对全部 12 层 × 3 个 kv_len 做"CG 式"重建时，capture 包装器把
   每一层的完整 `kv_cache`（该层的整块 3-slot 缓存，约 1.6GB）都 `.clone()` 了一份——12
   层叠加约 19GB，跟已经占了约 92GB/97GB 的模型本身叠在一起直接 OOM。改成只保留引用、不
   clone 之后解决（`kv_cache` 在同一次调用范围内不会被并发改写，保留引用是安全的）。
2. **`cudaErrorIllegalAddress`**：单层版本改进后，第一次调用序列（kv_len=64/400/500 各
   跑一次"CG 式"重建）顺利跑完并给出了上面这张表；同一个 daemon 进程里紧接着的下一次调
   用（哪怹是一次新 exec、同一个长驻 CUDA context）在**执行到真正的 kernel 之前**（还在
   `canary` 自检阶段）就报了非法内存访问——说明前一批调用已经悄悄写坏了某块内存，只是延
   迟到下一次 kernel launch 才炸出来。嫌疑指向 `cg_style_output` 这段辅助代码里手搭的
   `PagedAttentionWorkspace.for_fixed_capacity(max_work_items=..., max_partial_rows=...)`
   容量估算——用的是"这次单独调用刚好需要多少"而不是这个仓库其余地方一直强调的"按声明容量
   ,永不重新估算"的正确模式,大概率某处差了一点导致 CuTe kernel 写出界。停止继续用这段
   代码本身,把 daemon 完整 stop/start 一次以清空 CUDA context 后恢复健康,继续验证
   `eager_vs_oracle`/`cg_style_vs_oracle` 两个数字的可信度不受影响（前一批调用的输出在
   写坏内存之前已经拷回 CPU）。**这段代码只存在于 scratchpad，未提交，是本次调查的诊断
   脚手架的 bug,不是生产代码的 bug**。

### 更新后的修复方案取舍（叠加在第一轮"修复方案"之上，不是替换）

第一轮列的三个方向（1. 让 eager 塌成单块；2. 让 CG 按真实 kv_len 重算；3. 先做 oracle）
里，方向 3 现在做完了，答案是"两种块数在注意力算子层面都对"。这把方向 1/2 的风险评估改
写了一部分：

- **方向 1（让 eager 也塌成单块）**：不再有"如果单块本身就是那个隐藏 bug，两条路径会
  一起错"的顾虑——单层测试已经证明单块（CG 式）和多块（eager 式）在注意力这一层都逼近
  稠密真值。这个方向现在风险更低、更值得优先做——它消除的是"两条路径的其中一个具体、已
  确认的实现差异来源"，虽然不保证消除全部差异（见下）。
- **方向 2（让 CG 按真实 kv_len 重算）**：结论不变，仍然是更大工作量、更高风险，而且现
  在多了一个理由不优先做：既然两种块数本身都不算错，"让 CG 也用另一种同样算对的块数"换
  不来任何正确性收益，纯粹是多做工。
- **新增顾虑（不属于原三个方向，是本轮新发现）**：**即使修好块数这一个具体差异，也不能
  保证 eager 和 CG 从此在 full-model 层面完全一致**——因为 MoE 对任何微小实现差异的放大
  效应是系统性的，不是这一个 bug 独有。块数只是这次能明确抓到、能复现的一个具体来源；分
  块方式变了会带来的其它连带差异（内部求和顺序、descale 调用顺序等）可能还有别的、这次
  没抓到的小尾巴。修方向 1 预期会**显著缩小**分歧，但"完全消除"需要更强的保证（比如证明
  eager 和 CG 在这条路径上逐算子字节一致，而不只是"数学上等价、cos 够高"）——这个更强的
  保证这轮没有验证，留作后续如果真的要修时的验收标准之一。

### 是否影响 `QSR_DFLASH_REQUIRE_CG` 默认值（更新，不覆盖第一轮结论——结论不变，理由变了）

**仍然不影响，继续支持保留 `1`。** 第一轮的理由是"多块 merge kernel 本身尚未确认对错，
必须先查清楚"——这个理由现在被本轮 oracle 结果部分否定了（多块 merge 本身在注意力层面是
对的）。但支持保留 `QSR_DFLASH_REQUIRE_CG=1` 的结论不受影响，理由换成了更扎实的一条：
**MoE 对 eager/CG 之间任何微小实现差异的放大效应是真实、已被 GPU 数据证实的**——eager 一
旦被真正触发，即使这次揭示的具体块数差异被修好，也不能保证它和 CG 在 full-model 输出上
完全一致（见上一节"新增顾虑"）。在有这条放大机制的前提下,继续让服务在 CG 捕获失败时"降
级但响亮"（`QSR_DFLASH_REQUIRE_CG=0`）而不是拒绝启动，风险不比第一轮估计的低。

### 本次改动（第二轮）

- `bfdiag/dense_attention_oracle.py`（新文件）：`DenseCausalOracleImpl`——从零手写、不
  经过 sparkinfer 任何 split-KV 代码路径的稠密 causal attention 参考实现,支持 GQA 头映
  射校准（`interleaved`/`blocked`）。放在 `bfdiag/` 而不是 `oracle/`：`oracle/` 是另一
  条独立调查（vLLM-vs-engine oracle）的只读参考树，这个类是这次调查专用的，需要能被继
  续编辑的地方。
- `bfdiag/workloads.py`：新增 `diagnose_dflash_verify_dense_oracle(engine, kv_lens=...)`
  ——整模型三方对比（已确认被 MoE 放大污染，不建议用它下结论，但函数本身保留：如果日后
  MoE 放大这条假设本身需要重新验证，或者需要一个"全模型层面到底差多少"的整体量级参考，
  这个函数仍然有用，只是不能单独拿它的输出去判定"哪条路径对"）。
- 本文件：追加本节。第一轮的原始证据、排除清单、以及第一轮"修复方案"/"是否影响
  REQUIRE_CG"两节均未做任何删改，只在本节里叠加、更新推理链条。
- GPU 锁：`diverge-agent` 已在两次故障之间正确 acquire/release（`/tmp/gpu_lock.sh`），
  当前锁空闲。daemon 已在故障后完整重启并确认恢复健康，未遗留异常进程或显存占用
  （`nvidia-smi` 确认恢复到基线 ~450MiB）。
