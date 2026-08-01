# eager verify vs CG verify：真实数值分歧（未根因，独立立项）

**状态：未根因，不要现在开始查。** 这份笔记是把已有证据和排除清单存档，交给下一个专门查这个的人（或未来的我）。查法预计需要 `bf divergence` 逐层排查，是独立的调查范围，超出这次 C-1 容量修复的任务边界。

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
