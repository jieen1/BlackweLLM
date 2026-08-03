# 历史实现（vLLM 时代的 Qwen3.6-27B runtime）完整测绘

日期：2026-08-03 · 状态：🟢 只读调研，零 GPU · 模型：`unsloth/Qwen3.6-27B-NVFP4`（**与今天同一个 checkpoint**）

> **这份文档的目的**：让人**不必再回老树一次问一个问题**。它把历史实现的能力清单、
> 每一个实测数字连同其精确配置、当年已知但没修的问题、计划过但没建的东西，以及
> 「历史构件 → 今天等价物 → 状态」的逐条映射，一次性摊平。
>
> **与 [`../docs/qwen36-rebuild-spec.md`](../docs/qwen36-rebuild-spec.md) 的关系**：那份做的是
> `oracle/qwen36_vllm/` 的**逐模块可移植性判定**（能不能搬、怎么搬），本文档做的是
> **能力与数字的测绘**（做过什么、量到多少、什么口径、今天缺什么）。两份互补，
> 冲突处以本文档为准的地方会显式标出（§0.3）。

---

## 0. 先读这三条纠偏（不读会得出错误结论）

### 0.1 `/home/bot/project/qsr-hist`（`8f5c195`）**不是历史实现的终点**

`8f5c195` 的日期是 **2026-07-21**。历史实现（vLLM 路径）此后又演进了 **9 天、22 个提交**
才在 **2026-07-30（`a9cb932`）** 被剥离。只读 `qsr-hist` 会漏掉这一整段。

| 提交 | 日期 | 加了什么（本调研关心的） |
|---|---|---|
| `684d035` | 07-22 | 完整 256K 上下文 + 最大并发 |
| `ead0a98` | 07-22 | 代码规范 / CI 工具 |
| `b4e624d` | 07-22 | **GPU 采样（temperature/top-k/top-p）**、`compat_vllm.py` 单点收口、**引擎轮级 watchdog（200 轮无进展强制回收）**、**cancel() + SSE 断连**、MTP 接受直方图 |
| **`a8bd167`** | **07-22** | **增量 chunked prefill + prefill/decode 解耦调度 —— 就是 M1 / INV8 Phase B** |
| `8ec9cd3`/`36254b5`/`044940d`/`5886742` | 07-22 | B5 模块化：`block_pool` / `metadata_builders` / `cuda_graphs` / `mtp_accept` 拆出单文件 |
| `160babd`/`b5397d8` | 07-23 | E1：`ModelSpec` 冻结架构参数；MTP 方法抽到 `Qwen36Backend` |
| `7c143ec` | 07-23 | **C2 logprobs**（verify/decode 路径返回 per-token logprobs + top-k） |
| `97c3e0c` | 07-23 | D2 指标：MTP accept / prefix cache / KV 路径埋点 |
| `994ae5c`/`5a993ce` | 07-23 | B5：`PrefixCacheOps`、`GdnStateManager` 抽出 |
| `e66d254` | 07-26 | `block_size` 16 → **64**（sparkinfer page 布局） |
| `ff4d858`/`a9cb932` | 07-30 | 隔离进 `oracle/qwen36_vllm/`，退出生产路径 |

**最重要的后果**：`notes/2026-07-20-inv8-chunked-hit-prefill-plan.md` 把「Phase B 跨步交织」
写成未建成 —— 那是 07-20 的状态。**07-22 的 `a8bd167` 把它建成了**（见 §3.3）。
任何「M1 从来没做过」的结论都是只看 `qsr-hist` 造成的。

### 0.2 历史实现今天并没有丢

- **完整最终态代码**：`oracle/qwen36_vllm/`（11 个模块，约 8047 行）
  —— `direct_model_runner.py`(2017) / `backends/qwen36.py`(2159) / `cuda_graphs.py` /
  `metadata_builders.py` / `gdn_state.py` / `prefix_cache.py` / `attention_compat.py` /
  `vllm_compat.py` / 4 个 `nvfp4_*_patch.py` / `triton_norm_ops.py` / `gemma_norm_patch.py`。
- **完整历史记录**：`PROGRESS.md`（4266 行）**没有被删除，是被 rename** 成
  `docs/archive/2026-07-20-PROGRESS.md`；`项目实施规划.md` → `docs/archive/2026-07-18-项目实施规划-qwen36-only.md`。
- **所有 notes**：`qsr-hist/notes/` 的 15 个文件**全部**在今天的 `notes/` 里（同名同内容，
  仅 `2026-07-21-kernel-comprehensive-review.md` 与 `prefix-cache-design.md` 两个有后续修改）。

⇒ **不需要去 `qsr-hist` 找代码**。要读历史实现，读 `oracle/qwen36_vllm/`（更新、更完整）；
要读历史记录，读 `docs/archive/2026-07-20-PROGRESS.md`。
`qsr-hist` 唯一不可替代的用途：看 07-21 那个时间点的**未拆分单文件** `runtime/direct_model_runner.py`（6170 行）。

### 0.3 需要修正 `qwen36-rebuild-spec.md` 的一处内部矛盾

该文档 §1.4 的 chunked-prefill 那两行**自相矛盾**：先正确地引用
`direct_model_runner.py:1817-1938`（`prefill_chunked_step` 中途返回 `False` 让引擎跑一轮 decode），
随即写「连 oracle 自己的 Phase B（跨步交织）也**从未完工**」并引 INV8 计划文档为据。
**后半句过期**：那份计划写于 07-20，`a8bd167`（07-22）实现了它。
正确表述是：**oracle 里有一份可工作的跨步交织实现，今天两个 backend 都把它 stub 掉了**（§1 M-1）。

---

## 1. 🔴 今天缺失清单（最高优先，按当年实测收益排序）

> 判据：历史实现有、且有实测收益；今天的自研路径（`runtime/`）没有或退化。
> 「历史收益」一栏的口径见 §4，不要跨口径引用。

| # | 缺什么 | 历史位置 | 今天位置 / 状态 | 当年实测收益 |
|---|---|---|---|---|
| **M-1** | **MTP 路径的 CUDA Graph**（verify 图 + draft step 图 + padded step0） | `oracle/qwen36_vllm/cuda_graphs.py`（`CapturedBatchDecodeGraph` `:21-689`、`CapturedMTPDraftStepGraph` `:692-1119`）；调用点 `backends/qwen36.py:1985+`、`:671-902` | **完全没有**。`runtime/backends/qwen36_mtp.py` 全文件零 `CUDAGraph`（`:150-153` 自述）；`runtime/backends/qwen36.py:931-1012` 只捕 plain decode，而 `server/engine.py:566-577` 说明 MTP 轮**根本不走**那条路 | W1-S：**+140.9%**（27.464→66.152）后又 **+74.06%**（78.565→136.750）。**今天的对照事实**：`8d32c2a` 实测开 MTP 后 decode 从 28.0 → 7.80 tok/s（**0.28×**），正是因为 MTP 让已捕获的 decode 图变得不可达 |
| **M-2** | **跨槽批处理的 verify/draft**（一次 forward 覆盖全部活跃槽，含 ragged qo_len） | `backends/qwen36.py::mtp_verify_and_commit_batch`(`:1985-2159`)、`_mtp_sync_and_propose_batch`(`:671-902`)、`_mtp_forward_batch`(`:453-596`) | `runtime/backends/qwen36.py:679-716` 是**逐槽串行的 dict comprehension**；模型图 batch=1（`qwen36_mtp.py:113-123` 自述） | 单槽→跨槽批处理当年 **+43%**（11.60→16.61 W1-S） |
| **M-3** | **GDN 投机状态的 K+1 行寻址（零回滚）** —— `verify_batch_spec` + `build_gdn_metadata_spec_batch` + `_ssm_spec_row` | `oracle/.../direct_model_runner.py::verify_batch_spec`(`:1581`)、`metadata_builders.py:524-596`、`block_pool.py:45-79` | 今天用 **snapshot/restore**：`runtime/model/qwen36_model.py::spec_forward`(`:771+`) 逐位置克隆 K+1 份 GDN 快照，`commit_spec_snapshot`(`:1010-1041`) 按 m 选一份 `copy_` 回去。`runtime/recurrent_state_pool.py::spec_row`(`:90-108`)、`runtime/block_pool.py:45-79` **实现了但零调用** | Phase 2 用这个机制删掉 snapshot+recompute-forward：**+18.76%**（66.152→78.565）。当年 recompute 分支命中 **84.4% 的轮次、约 56% 墙钟** |
| **M-4** | **跨步交织 chunked prefill**（prefill 不再独占引擎轮次） | `oracle/.../direct_model_runner.py::prefill_chunked_begin/_step`(`:1731-1938`)，`ChunkedPrefillState` | **两个 backend 都是 stub**：`runtime/backends/qwen36.py:534-590`（`prefill_chunked_step` 直接 `return True`，docstring 写 "One-shot"）、`laguna.py:2413-2415`。引擎侧调度骨架**还在**（`server/engine.py:442`、`:1306`、`:1390`），`:607` 注释直言 "unused: Qwen36Backend prefill is one-shot" | `a8bd167` 验证：长 prefill 进行中，短请求 TTFT=0.68s **不被阻塞**。当年判定这是 128K/c=4 端到端差距的 **60-70%**（TTFT 25.7s vs 原生 4.4s）。**注意**：intra-admission 分块（Phase A）只买到 **−10.7% TTFT**，不是这条 |
| **M-5** | **内容寻址前缀缓存**（块哈希链 + 跨请求/跨槽共享 + refcount + LRU 驱逐 + GDN checkpoint 联动） | `oracle/qwen36_vllm/prefix_cache.py`(357行)、`gdn_state.py::materialize_gdn_checkpoint/checkpoint_view/evict_gdn_checkpoint`、`backends/qwen36.py::mtp_prefill_with_cache`(`:1635-1983`) | `runtime/block_pool.py`（518 行，机制完整）**生产路径零调用方**（grep `BlockPool(` 在 `runtime/`+`server/` 为空）。今天 Qwen36 用 **O(prompt) 同槽线性 token 比较**（`qwen36.py:720-834`）+ **每槽一份滚动 recurrent checkpoint**（`_maybe_checkpoint` `:862-899`），预算只有 **2 份 ≈154 MiB**（历史是 8 GiB）；KV 侧**没有驱逐**（静态每槽连续页，无跨槽共享） | 单槽 64K exact-repeat：cold TTFT 16744.5ms → warm 178.8ms = **93.67×**（P3.4）。跨请求命中的价值全部来自这条 |
| **M-6** | **同轮 fan-out fork**（≥2 个同轮请求共享前缀只算一次） | `backends/qwen36.py::mtp_prefill_fanout_batch`(`:1318-1549`) | 无 | 无独立 e2e 数字（只有正确性验证：N=2/3/4，head block `ref_cnt` 2/3/4，零串扰） |
| **M-7** | **会话亲和 warm-continue**（零 restore 续接） | `backends/qwen36.py::mtp_prefill_warm_continue`(`:1560-1633`) | 协议**预留了坑没实现**：`runtime/backends/protocol.py:22-27,202-209`，`--session-affinity` 静默降级 | e2e 证明机制（`session_warm_continuations` +1 且不占 `prefix_cache_hits`），无吞吐数字 |
| **M-8** | **生产默认并发** | `qsr-hist/server/engine.py:129-143`：`capacity` 默认 4、`num_slots=16`、`blocks_per_slot=512` | `server/app.py:91,96`：`QSR_SERVER_CAPACITY` 默认 **1**、`QSR_SERVER_NUM_SLOTS` 默认 **2** | 所有历史 headline 都是 **c=4**。今天的 c=1 数字与历史 c=4 数字**不可直接比** |
| **M-9** | **split-KV 并行度**（从每槽容量派生固定 split，CUDA-graph 安全） | `oracle/.../direct_model_runner.py:681-684`：`_DECODE_TARGET_SPLITS_PER_REQ = 32`（128K 下 → 每请求 32 个 split，才凑够 CTA） | **不只是缺旋钮，是结构上没有**：sparkinfer 的 planner 自己决定；`mode="extend"` **硬编码 `split_kv=False, disable_split_kv=True`**（`runtime/model/qwen36_model.py:1149-1155`、`runtime/backends/laguna_sparkinfer_attn.py:186-189`）；qwen36 decode 直接声明 `max_partial_rows=0`（`qwen36_model.py:1227`，"no split-KV merge buffer"）。全仓库唯一可能 split 的路径是 Laguna 的 `mode="verify"` | 首次引入 **+13.1%**（16.61→18.78）。**血的教训**：图里遗留的 `TARGET_SPLITS=16` 与 eager 的 64 不一致，直接把接受率从 70.29% 变成 76.67% 的**假象**。⚠️ **历史在 128K/c=4 下靠 32 splits/请求才把 CTA 数拉起来（batch=4 时不 split 只有 16 CTA / 188 SM）。今天走长上下文时这条要重新验证** |
| **M-12** | **跨槽向量化的采样 / accept-reject**（一次 GPU argmax + 一次 `.tolist()` 覆盖全 batch） | `[hist]:1239-1305` `determine_accept_reject_batch`；draft 续步也是「一次 argmax + 一次 tolist」 | 逐槽 Python 循环 + 每槽一次 `.item()` **设备同步**：`runtime/backends/qwen36.py:625-633`、`laguna.py:2190-2201`。`runtime/mtp_accept.py:74-144` 的向量化批版**实现了但未接线** | 当年这是 Phase 3 明确的优化项之一（把 `num_reqs × k` 次 host 往返压成 1 次）。`capacity=1` 时看不见，**是 capacity>1 的第一堵墙** |
| **M-10** | **无损性回归锚点**（同一 fixture 的 bit-identical 不变量） | `total_committed_tokens=4116` + `draft_acceptance_rate_pct=70.29204431017119`，跨 20+ 轮改动逐字节一致 | 今天有 `bfdiag`/`bf diff`（更强的框架），但**没有一个等价的 Qwen3.6 数值锚点** | 这是当年每次改动都能立刻发现回归的原因 |
| **M-11** | **真实工作负载长跑驱动** | `benchmarks/mtp_sustained_realistic_workload_check.py`（63.5 分钟 / 7720 轮 / 758 次真实准入，内存逐字节平坦） | 文件还在（`benchmarks/mtp_sustained_realistic_workload_check.py`），但 import 的是 `oracle.qwen36_vllm.*`，**对自研路径不可用** | 唯一一次「真实变化内容」的稳态吞吐测量：**~26.3-26.7 accepted tok/s**（c=4，混合长度） |

### 1.1 今天有、历史没有的（不要往回搬）

- **`bfdiag` / `bf diff` / run record / flight recorder**：当年没有，是今天相对当年的净改进。
- **CUDA Graph 捕获成败的可观测性（C7-2）**：`cg_status`、`/debug/stats`、
  Prometheus `blackwellm:dflash_cg_captured`、`decode_graph_replays` 计数器。当年捕获失败是静默的。
- **GPU 侧融合的 CG 元数据重建**（`runtime/kernels/cg_decode_metadata.py`，B=1 零 H2D）：
  当年 oracle 自己承认「未做到完全无分配」（只做到 pinned staging + numpy 视图）。
- **反向 loader 检查**（`warn_on_unconsumed_tensor_families`）：正是它把 `unsloth/` 的
  16 个 `k_scale`/`v_scale` 报出来的（`notes/2026-08-03-fp8-kv-cache.md`）。
- **`ArchitectureSpec` / `ModelBackend` 协议 / `check_conformance`**：当年没有模型抽象层。
- **sparkinfer 首次真实形状 JIT 的 `PagedPlanBudget` 解法 + 启动 warmup**：当年没这个问题，
  但今天已经解掉了（`runtime/model/qwen36_model.py:1233-1240`，`server/engine.py:686-694`）。
- **请求超时（600s）+ 客户端取消 + SSE 断连回收**：历史只有 watchdog。
- **`PersistentSeed`**：修掉了「每 token 重新播种」这类历史从未遇到（因为只有贪心）的问题。

---

## 2. 映射表：历史构件 → 今天等价物 → 状态

状态取值：**等价** / **不同** / **缺失** / **主动放弃**。行号：历史侧一律引 `oracle/qwen36_vllm/`
（最终态；`qsr-hist` 的行号只在标注 `[hist]` 时使用），今天侧引 `runtime/`、`server/`。

### 2.1 投机解码 / MTP

| 历史构件 | 今天 | 状态 |
|---|---|---|
| 独立 draft 模型加载（`load_eagle_model` → `Qwen3_5MTP`） | checkpoint 内 `mtp.*` 头，自建 `Qwen36MTPHead`（`runtime/model/qwen36_model.py:2503-2581`），`Qwen36MTPLayer` `:2450-2500` | **不同**（更好：不再借 vLLM 的类） |
| 每轮 draft = step0 teacher-forced 重同步 + K−1 自回归步，**全部跨槽批处理**（`backends/qwen36.py:671-902`、`:598-670`） | `Qwen36MTPEngine._draft_loop`（`runtime/backends/qwen36_mtp.py:184-202`）：**K 次串行 `mtp_step`，batch=1，无 step0 重同步概念** | **不同 + 退化**（M-2） |
| verify = 1 次跨槽 `verify_batch_spec`，qo_len=k+1（`direct_model_runner.py:1581`） | 每槽 2 次 forward：anchor 推进 `[1,1]` + `verify_forward` `[1,K]`（`qwen36_mtp.py:289-297`） | **不同**（多一次 anchor forward；当年 anchor 的 KV 写入折在 verify 里） |
| accept/reject：贪心最长前缀 + 一个 recovery/bonus token，向量化 `determine_accept_reject_batch`（`[num_reqs, k+1]` 一次 argmax + 一次 `.tolist()`） | `runtime/mtp_accept.py:26-56`（`determine_accept_reject_from_predictions`），算法**一模一样**；向量化的 `determine_accept_reject_batch`（`:74-144`）实现了但**未接线** | **等价**（算法）/ **缺失**（批版接线） |
| 非贪心（采样）投机 | 历史：**不支持**，`mtp_verify_and_commit*` 无条件贪心（2026-07-17 明确记为「非零温度对比的前置条件」） | 今天有 `sample_accept_reject`（`runtime/mtp_accept.py:250-330`，Leviathan/Chen 拒绝采样），但 `draft_probs` 是 one-hot（`qwen36_mtp.py:304-307`）⇒ 退化成「以 p(x) 概率接受」。`mtp_accept.py:270-275` 自己标注了这个缺口 | 今天**更前进但未完成**；不是回归 |
| GDN 递归状态的接受/拒绝提交 | 见 M-3 | **不同**（回到 snapshot 风格） |
| attention KV 的拒绝处理 | 两边都是**整数截断，不擦写**（历史：位置寻址，拒绝位永不再读；今天 `qwen36_model.py:2798-2801`） | **等价** |
| draft 头自身 KV 的回滚 | 历史：draft 模型无 GDN，纯截断 | 今天 `qwen36_mtp.py:341` `cache.seq_len = round_mtp_start + m` | **等价** |
| K 的默认值 | K=3（`num_speculative_tokens=3`） | K=**4**（`server/app.py:164`） | **不同**（比较接受率时必须换算） |
| MTP 开关默认 | 生产 server 默认**开**（历史 headline 全部带 MTP） | `QSR_SERVER_ENABLE_MTP` 默认 **`"0"`**（`server/app.py:163`） | **不同** |
| 每轮 resync（把已接受的内部行用 target 真实 hidden 重写） | 历史：step0 就是 teacher-forced 重同步，**始终执行** | `qwen36_mtp.py:373-417`，默认关；且 **`self.model.mtp_resync_step` 在 `runtime/model/qwen36_model.py` 里根本不存在** ⇒ 打开即 `AttributeError`（唯一定义在 `tests/test_qwen36_mtp_engine.py:155` 的 stub） | **缺失且是 bug** |

### 2.2 CUDA Graph

| 历史 | 今天 | 状态 |
|---|---|---|
| verify forward（target，qo_len=k+1，batch 1..num_slots//2）**总是捕获** | **无**（MTP 轮不走图） | **缺失**（M-1） |
| draft K−1 continuation 步（qo_len=1）**总是捕获**；`replay_incremental` 跨页才重建 page indices | **无** | **缺失** |
| draft step0：ragged 时 **pad 到 max_qo_len** 再捕获（2026-07-20），replay 后 `index_select` 取回各槽真行 | **无** | **缺失** |
| 纯 decode（qo_len=1）捕获 | **有且更完整**：`runtime/backends/qwen36.py:972-1012`，`1..num_slots` **每个 batch size 一张图**、共享 `graph_pool`；replay `:652` | **今天更好** |
| prefill **从不捕获** | 同 | **等价** |
| GDN snapshot/restore **从不捕获** | 今天 spec 快照在 `verify_forward` 内、不在图里 | **等价（都在图外）** |
| 捕获期 warmup 槽污染问题（GDN 递归状态**非幂等**，不能拿真实槽热身） | 历史两阶段解法：先「永久保留 `2×batch_size` 槽」→ 后 `precapture_cuda_graphs()` 用真实槽热身、捕获后立即 `reset_slot`（2026-07-20 深夜，**解除了长上下文 OOM**） | 今天：`qwen36.py:941-946,965-970` 捕获后 `finally` **无条件清零全部槽**；`server/engine.py:340-357` `cg_extra=1` | **等价（同一思路，今天更省）** |
| 捕获失败 = 静默退 eager | `cg_status` + `/debug/stats` + Prometheus + 整组丢弃并卸载 driver（`qwen36.py:950-964`） | **今天更好** |
| `_fill_buffers` 优化轨迹：临时 tensor → pinned staging → numpy 视图 → page indices 缓存 | 今天 `runtime/kernels/cg_decode_metadata.py`（GPU 侧融合，B=1 零 H2D） | **今天更好** |

### 2.3 Prefill / 调度

| 历史 | 今天 | 状态 |
|---|---|---|
| intra-admission chunked prefill（`chunk_size` 默认 **8192**，`_DEFAULT_PREFILL_CHUNK_SIZE`，`metadata_builders.py:146`）；target + draft **锁步**分块；GDN 用 `has_initial_state` 续接 | `runtime/backends/qwen36.py:472-494` 循环调用 `self.model(...)`，块大小 `_prefill_chunk_tokens()`（`:496-528`，**8192**，再按 `2**31-1 / (2*intermediate)` 封顶）；**draft 侧不分块**（prefill 后一次性 `draft_after_prefill`） | **不同**（同样是分块前向，但 draft 侧机制不同） |
| hit 路径的 ragged suffix 分块（INV8 Phase A） | 无 hit-suffix 概念（同槽复用后直接前向剩余 suffix） | **不同** |
| **跨步交织**（`prefill_chunked_begin/_step`） | stub。⚠️ 引擎侧 `else: self._pending_prefill = ...`（`server/engine.py:1416-1420`）**因此是不可达死代码**——两个 backend 都恒返回 `done=True` | **缺失**（M-4） |
| sparkinfer 首次真实形状 JIT 编译 | 历史无此问题（vLLM kernel 预编译） | 今天**已显式解决**：`PagedPlanBudget` 让 `cta_tile_q` 来自声明容量而非活的 `qo_len`（`runtime/model/qwen36_model.py:1233-1240`、`laguna_sparkinfer_attn.py:175-200`），加启动 warmup（`server/engine.py:686-694`，`QSR_SERVER_WARMUP_PAGED_ATTENTION` 默认 `"1"`）。修之前一次真实请求内会付 26-37 秒编译 | 今天**新增的问题、已修** |
| TTFT 仪表 | benchmark 侧测（`native_warm_compare.py`、`mtp_w1s_our_runtime_perf.py`） | server 侧 `TTFT_BUCKETS` 直方图存在（`server/metrics.py:36,84-103`），但**只在两条 SSE 流式路径上记录**（`server/app.py:956`、`:1756-1762`），非流式不记；且 `tracing.py:150` 的 `prefill_done()` **只有测试在调**，⇒ `/debug/traces` 的 `avg_prefill_ms` **恒为 0** | **部分缺失（M-4 的度量前提没有）** |
| `chunk_size` + 真 ragged batch → `NotImplementedError`（`backends/qwen36.py:1043-1050`） | 今天没有这个组合（逐槽处理），问题**结构性消失** | **主动放弃/消失** |
| 连续批处理引擎：`ServerEngine._step_sync` 先准入（阻塞 prefill）再一轮 verify | 今天 `server/engine.py:1218+` 结构相同，但**先 decode 再推进一个 prefill chunk**（骨架已支持交织，缺 backend 实现） | **等价 + 骨架更前进** |
| 准入容量检查带 `self.K` token 余量（`capacity_ok`，2026-07-19 修的真实 bug） | `server/engine.py:773-774` 同样保留 K 余量 | **等价** |
| 无 `max_num_batched_tokens` 概念（原生 vLLM 有 8192） | 同样没有（全仓库零命中：`max_num_batched_tokens`/`token_budget`/`max_num_seqs`） | **等价（都缺）** |
| watchdog：200 轮无进展强制回收（`b4e624d`，07-22） | **保留了**：`server/engine.py:1723-1760`，`watchdog_max_stale_rounds` 默认 **200**（`:285`），`find_stale_slots` `:102-113` | **等价** |
| 无请求超时 | **今天多了**：`request_timeout_s` 默认 **600.0**（`server/engine.py:286`，回收在 `:1690-1721`） | 今天更好 |
| 无抢占/重排队 | 同样没有（`preempt` 全仓库零命中）；等待队列是纯 FIFO（`server/engine.py:447,1215,1337`） | **等价（都缺）** |
| 超容量请求在 prefill 中途抛 `RuntimeError`，**整批一起崩** | 结构性缓解：准入前 `capacity_ok`（`server/engine.py:773-774`，`prompt_len + max_tokens + K ≤ 131072`），且 qwen36 逐槽 prefill，一个槽抛错不会带塌同批其它槽 | **今天更好（但仍非优雅拒绝）** |

### 2.4 KV cache / 分页 / 递归状态

| 历史 | 今天 | 状态 |
|---|---|---|
| `kv_cache_dtype` 默认 **`"fp8_e4m3"`**（`oracle/.../direct_model_runner.py:258`，`qsr-hist/runtime/direct_model_runner.py:1192`）—— **当年就是生产默认** | `QSR_QWEN36_FP8_KV`，2026-08-03 起**默认开**（`runtime/model_loading.py:228-280`）。省 ~12.3 GiB | **已追平**（曾长期缺失，见 `notes/2026-08-03-fp8-kv-cache.md`） |
| k_scale/v_scale：走 vLLM 自己的加载逻辑 | `unsloth/` 有 16+16 个 bf16 标量，今天已消费 | **等价** |
| `block_size`：16 → **64**（`e66d254`，07-26） | Laguna 页大小 = `block_size` 默认 **64**（`server/app.py:97`，只接受 64/128）；**qwen36 页大小固定 128**（`runtime/model/qwen36_model.py:333`），`block_size` 在 qwen36 里**只当前缀缓存/checkpoint 的对齐粒度**（`qwen36.py:205-214` 要求能整除 128） | **不同**（由 sparkinfer 决定） |
| `blocks_per_slot`：benchmark 2560 / server 512（后升 4200，67200 token 上限） | `QSR_SERVER_BLOCKS_PER_SLOT` 默认 **2048** ⇒ 2048×64 = **131072 token/槽**（`server/app.py:109`、`server/engine.py:387`） | **不同（今天上限更高）** |
| 保留物理块/槽 0（`RESERVED_PHYSICAL_SLOTS=1`） | `runtime/block_pool.py:17-24` 沿用 `=1`，但 Laguna 本地 `=0` | **应废弃**（根因是「vLLM 调度器从不产出物理索引 0」这个 vLLM 事实，四轮调试才坐实；不是硬件事实） |
| split-KV：`_DECODE_TARGET_SPLITS_PER_REQ=32`，`kv_split_size` 从每槽容量派生 | 无等价旋钮 | **缺失**（M-9） |
| GDN 递归状态：每槽 `1 + num_speculative_tokens` 行（`allocate_fixed_slot_kv_caches`，`ssm_rows_per_slot`） | `runtime/recurrent_state_pool.py`（271 行），`spec_row`（`:90-108`）实现了但未用 | **不同 + 部分缺失** |
| GDN checkpoint 池（按字节预算 + LRU + hash 标签防错配）`gdn_state.py`，预算默认 **8 GiB**（`gdn_checkpoint_byte_budget: int = 8 * 2**30`） | `runtime/recurrent_state_pool.py:134-270`（`register`/`touch`/`get_by_hash`/`evict`/`evict_for_budget`）+ `runtime/backends/qwen36.py:862-930`。**预算只有 2 份 checkpoint**（`DEFAULT_CHECKPOINT_BUDGET_MULTIPLE = 2`，`qwen36.py:127,234-238`），每份 ~77 MiB ⇒ 约 **154 MiB**，且是每槽一份滚动 checkpoint | **不同（今天预算小 50×，只够同槽复用，不支持跨请求命中）** |
| `gpu_memory_utilization=0.85` 真实进入 vLLM 的 KV 池 profiling | **今天是死旋钮**：`server/engine.py:281,398` → `build_laguna_config` → `SelfBuiltModelConfig` 后**再无读取者**；qwen36 完全看不到它。`server/app.py:100-103` 的注释声称有 `profile_kv_cache_blocks`，**该函数不存在** | **缺失（且文档谎报）** |
| — | `QSR_SERVER_KV_CACHE_DTYPE` 同样是死旋钮：`server/engine.py:393` 存下来后全仓库零读取 | **缺失（旋钮无效）** |
| GPU 常驻 snapshot 双缓冲 + `torch._foreach_copy_`（Phase 1，把 89-117ms/轮 的 D2H/H2D 变成 D2D） | 今天 spec 快照直接在 GPU 上 clone（`spec_forward`） | **等价（都不过 CPU）** |
| `gpu_memory_utilization` 默认 0.85（server） | 无等价全局旋钮（按 max_seq_len 推导） | **不同** |

### 2.5 采样 / 输出

| 历史 | 今天 | 状态 |
|---|---|---|
| 贪心（argmax）——**唯一路径**，直到 `b4e624d`（07-22）加入 temperature/top-k/top-p | `runtime/sampling.py`（237 行）：`temperature`(默认 0.0) / `top_k`(0) / `top_p`(1.0) / `seed`；`PersistentSeed`(`:24-89`) 修掉了「每 token 重新播种 ⇒ 每个位置抽同一个数」的 N3 bug | **今天更完整**（历史 `runtime/sampling.py` 122 行是同一份的祖先） |
| —（历史同样不支持） | `min_p` / `repetition_penalty` / `frequency_penalty` / `presence_penalty` / `logit_bias` **全仓库零命中**；`n != 1` 在 `server/app.py:553-556` 拒绝 | **等价（都缺）** |
| logprobs / top-k 替代项（`7c143ec`，07-23） | `runtime/logprobs.py:15-71`（GPU 上 log_softmax+topk，只把小结果 `.cpu()`）+ `return_logprobs` 贯穿协议 | **等价**（同一血统） |
| 采样是否跨槽批处理 | 历史：draft/accept 路径**跨槽向量化**（一次 argmax + 一次 `.tolist()`） | 今天：**逐槽 Python 循环 + 每槽一次 `.item()` 同步**（`qwen36.py:625-633`、`laguna.py:2190-2201`）；唯一批处理路径是贪心 CUDA-Graph decode（argmax 烘进图里） | **退化**（见 M-12） |
| 结构化输出 / grammar | 无 | `runtime/structured_output.py` 存在，但 grammar slot 列表**永远为空**（`server/engine.py:1473-1486`），`response_format` 的 json_object/json_schema 在 `server/app.py:585-618` 被拒 | 今天多，但未接线 |

### 2.6 已经「休眠就位」的（重建时不要重新发明）

| 机制 | 位置 | 状态 |
|---|---|---|
| `_ssm_spec_row` GDN×投机行寻址 | `runtime/block_pool.py:45-79`；`runtime/recurrent_state_pool.py:90-108` | 原样存在，零调用 |
| `BlockPool`（free-list + refcount + LRU + 内容索引） | `runtime/block_pool.py:270-519` | 完整，生产零调用 |
| `_on_evict_block` 驱逐回调挂钩 | `runtime/block_pool.py:326-329` | 值为 `None`，等 `evict_gdn_checkpoint` 接进去 |
| `determine_accept_reject_batch` 向量化批版 | `runtime/mtp_accept.py:74-144` | 实现+有测试，未接线 |
| `warm_continue` 能力位 | `runtime/backends/protocol.py:22-27,202-209` | 协议留坑，无实现 |
| `ChunkedPrefillState` + 引擎交织调度 | `runtime/block_pool.py:153-183`、`server/engine.py:442,1306,1390` | 契约与调度在，backend 是 stub |

---

## 3. 能力清单：历史实现做过什么，在哪

> 行号：`oracle/qwen36_vllm/`（最终态）优先；`[hist]` 前缀表示引 `qsr-hist/runtime/direct_model_runner.py`（`8f5c195`）。

### 3.1 投机解码 / MTP（K=3）

- **draft 模型加载**：`get_model()` + `load_eagle_model()` 借 vLLM 的 `Qwen3_5MTP` /
  `Qwen3_5MultiTokenPredictor`（`qwen3_5_mtp.py`）。⚠️ 早期文档写成 `Qwen3NextMTP` 是**错的**，
  已在 2026-07-17 更正 —— checkpoint 的 `model_type` 是 `qwen3_5`，字段名是 `mtp_num_hidden_layers`。
- **draft 必须每步同步**：结论来自逐行读 vLLM `execute_model`（`:1114`、`:1456-1479`、`:582-623`）
  与 `_prepare_prefill_inputs_kernel`（`speculator.py:510-519`，"shift input_ids by one"）。
  精化后的结论：**必须在每轮状态机里，但调用点可以收在一个漏斗**，不必散进每个 forward 入口。
  → 这正是今天 `qwen36_mtp.py` 模块 docstring 里那条 (token, hidden) 偏移一位的同一件事。
- **单槽原语**：`_mtp_forward`(`backends/qwen36.py:91-201`)、`_mtp_sync_and_propose`(`:203-260`)、
  `mtp_prefill`(`:262-306`)、`mtp_verify_and_commit`(`:308-442`)。
- **批版**：`_mtp_forward_batch`(`:453-596`)、`_mtp_run_continuation_steps`(`:598-670`)、
  `_mtp_sync_and_propose_batch`(`:671-902`)、`mtp_verify_and_commit_batch`(`:1985-2159`)。
- **accept/reject**：`determine_accept_reject`(`[hist]:1215`)、`determine_accept_reject_batch`(`[hist]:1239-1305`)
  —— 一次 `argmax` + `cumprod` 累积 AND + 一次 `.tolist()`，`k+1` 位置一次算完。
- **step0 图化的门**（`[hist]:3552-3555`）：`enable_cudagraph and max(num_new_tokens_list) <= _MAX_DECODE_QO_LEN(16)`；
  ragged 时 pad 到 `max_qo_len`（重复末 token + expand 末 hidden 行），replay 后 `index_select` 取真行。
  **`_MAX_DECODE_QO_LEN` 这道门是承重的**：曾差点把真实 prefill 路由进 decode kernel 图。
- **GDN 回滚三代**：
  1. Option A `snapshot_gdn_state`/`restore_gdn_state`（`[hist]:2711/:2809`，48 层，`logits_exact_equal=true`）——
     初版走 pageable D2H/H2D，**89-117ms/轮**；
  2. Phase 1 改 GPU 常驻固定地址 + D2D `copy_`（~604MB VRAM）；
  3. Phase 2 直接**删掉回滚**：`_ssm_spec_row`(`[hist]:84`) 给每槽 K+1 个专用 SSM 行，
     `build_gdn_metadata_spec_batch` 用 `num_accepted_tokens` 选行，
     **GDN 每个候选位置的输出本来就是因果有效的，只有状态 commit 才需要知道接受数**。
     `snapshot/restore` 保留但生产 verify 路径不再调用。

### 3.2 CUDA Graph 覆盖

见 §2.2 表 + `notes/2026-07-17-post-ragged-round-next-steps.md` §9.3（历史自己写的「什么被捕获、为什么」权威表）。
关键工程细节：

- **capture 期不能用真实槽热身**（GDN 递归状态非幂等）。这条是 2026-07-17 独立评审抓出来的
  **真实缺陷**，量化证据：老写法下 `logits max_abs_diff=7.93`、`cosine=0.55`、
  GDN `conv_max_diff=45.8`、`ssm_max_diff=12.5`。signal-probe（只看解码文本）**抓不到**。
  修完后 `cudagraph_eager_parity_check` 给出 `max_abs_diff=0.0` / `cosine=1.0` / 48 层全 0.0。
- **capture 区内禁止 `torch.cuda.synchronize()`** → `_forward_no_sync()`。
- **每次 replay 一次 device-wide sync 是反效果的**，已删。
- **图内常量必须与 eager 一致**：`TARGET_SPLITS` 事故（§2.2）。

### 3.3 Prefill

- **cold 分块**（uniform 长度）：`mtp_prefill_batch`(`backends/qwen36.py:904-1310`)，
  `chunk_size` 默认 8192；target 与 draft **锁步**分块；GDN 靠 `has_initial_state` 续接；
  块边界做 GDN checkpoint。正确性：同一 16384-token prompt 用 `None/4096/8192/1024` 四种切法，
  anchor + 全部 K=3 draft token **精确一致**，GDN layer-0 逐字节 0.0（证明寻址正确），
  48 层累积是 bf16 round-trip 的良性漂移。
- **hit 分块**（ragged suffix，INV8 Phase A）：三路门 —— monolithic（suffix ≤ chunk）/
  uniform 分块 / ragged 逐槽分块；`effective_chunk = chunk_size // num_slots` 把每次
  forward 的总 token 数压到 ~8192（对齐原生 `max_num_batched_tokens`）。
- **跨步交织（Phase B / M1）**：`prefill_chunked_begin/_step`（`direct_model_runner.py:1731-1938`，`a8bd167` 07-22）。
  默认 chunk **512 tokens**；短 prompt（≤chunk）或 ragged batch 走 monolithic 快路径；
  hit 槽在 `begin` 里立即 restore，之后只增量推进 suffix；引擎每轮 **先跑 decode 再推进一个 chunk**。
  提交自述的验证：长 prefill 进行中短请求 TTFT=0.68s。
  ⚠️ 提交自己标注 `Not-tested: 128K+ 真实长 prefill 的 ITL 恶化量化`，
  且 `Directive: chunk_size >2048 需重测 ITL 恶化是否 <10%`。**没有 128K 的 TTFT 前后对比数字**。
- **prefill 永远 eager**（从不捕图）；理由：真实长 prefill 是 compute-bound，
  Phase 0 实测 prefill 段 `utilization.gpu` 已 85-99%。
- **last-position-only logits**：`logits_last_position_only`（默认 False，只有 prefill 传 True）。
  修的是把 `[65536, 248320]` bf16（~30.3 GiB）算两遍、只读 4 行的浪费。

### 3.4 前缀缓存（P0→P4b，历史最完整的一条线）

设计文档 `notes/prefix-cache-design.md`（854 行，INV1-INV9 + R1-R6 风险登记），
实施日志 `notes/prefix-cache-implementation-log.md`（1811 行）。

- **P0** block-table 间接层（行为逐字节不变）
- **P1** `BlockPool`：FIFO free-list + `ref_cnt`，按需 `_ensure_blocks`，`reset_slot` 归还
- **P2** 同轮 fan-out fork：leader 算一次共享前缀 → 兄弟 `reference()` attention 块 +
  跨槽 `restore_gdn_state(allow_cross_slot=True)` + 只续算自己的 suffix
- **P3.1** 持久内容寻址：`hash_block_tokens`（blake2b-128 链式，`extra_keys=(kv_cache_dtype,)`
  保证 fp8 与 nvfp4 KV 不撞），`BlockPool.hash_to_block/cache_block/get_cached_block/touch`；
  GDN checkpoint 池带 `hash_value` 标签，**错前缀 restore 会被拒绝而不是被用**；
  发布只发**完整已提交块**（不发部分尾块、不发未接受的 draft）
- **P3.2** 驱逐；**P3.3** 统一入口 `mtp_prefill_with_cache` + 全量不变量门
- **P3.4** 长上下文性能门（见 §4.5）
- **P4a** server 集成（`enable_prefix_cache=True` 默认开）+ 命中率埋点
- **P4b** 会话亲和 warm-continue（默认关，TTL 30s）
- **核心规则**：`L = G ≤ A` —— attention 侧能复用到 `A`，GDN 只有块对齐的
  checkpoint `G`，取两者的下界。**这就是今天 `qwen36.py:720+` 注释里
  「先算 attention 侧，再向下搜 recurrent checkpoint」的出处。**

### 3.5 调度与准入

- `ServerEngine`（`qsr-hist/server/engine.py`，689 行）：单引擎线程（`_engine_thread_main`/`_step_sync`），
  每轮：过期回收 retained 槽 → 准入（阻塞 prefill）→ 一轮 `mtp_verify_and_commit_batch`。
- 默认：`capacity=4`、`num_slots=16`、`block_size=16`、`blocks_per_slot=512`、
  `kv_cache_dtype="fp8_e4m3"`、`enable_cudagraph=True`、`enable_prefix_cache=True`、
  `enable_session_affinity=False`、`gpu_memory_utilization=0.85`、`idle_sleep_s=0.005`。
- `num_slots` 下限：生产 `capacity + (capacity if cudagraph)`；非生产 `3*capacity + ...`。
  **今天 `server/engine.py:340-357` 是同一条公式的后代。**
- 无 token 预算（`max_num_batched_tokens`）概念 —— 这正是原生 vLLM 有而我们没有的东西。

### 3.6 GDN / 递归状态

- 64 层 = 16 full-attention + **48 GDN**；MTP 头是 **1 层 full-attention，无 GDN**
  （`[hist]:1502` 有 `raise RuntimeError("unexpected GDN layer in MTP draft model")` 做断言）。
- 每层每 token：`in_proj` GEMM → `causal_conv1d_update` → `fused_sigmoid_gating_delta_rule_update`
  （spec）/ `fused_recurrent_gated_delta_rule_packed_decode`（非 spec）→ `out_proj`，约 9 次 launch/层。
- **`gdn_attn.py` 的真实 builder 在 spec-decode 共存时会把非 spec decode 重分类成 prefill**
  ⇒ `causal_conv1d_update` 的非 spec 分支在 MTP 生产下**接近死代码**。
- 死 spec 行污染：`reset_slot` 故意不清零张量，K=3 时 conv 的 [3,4,5] 行会留旧值 ——
  **已证明良性**（下轮 decode 先写后读，token 逐字节一致），但它坑过一次测试方法论。

### 3.7 采样

- 07-22 之前：**无条件贪心**。
- `b4e624d` 起：`runtime/sampling.py`（122 行）temperature/top-k/top-p，
  约束是「greedy 路径 bit 级不变」；**明确拒绝**了「采样 + MTP 联合投机」（正确性风险）。
- `7c143ec` 起：verify/decode 返回 per-token logprobs + top-k。

---

## 4. 所有实测数字（含精确配置与不可比警告）

> ⚠️ **本节是本文档最容易被误引的部分**。每张表先写口径，跨表引用前先看 §4.7。

### 4.0 口径定义（先读）

| 代号 | 定义 | 用它的脚本 |
|---|---|---|
| **口径 A** | `total_accepted_tokens / (TTFT + decode_wall)`，c=4 聚合 | `prefix_cache_warm_throughput_check.py`、`native_warm_compare.py` |
| **口径 B** | 稳态 decode（**扣掉 TTFT**）的 accepted tok/s | `decode_step_profile.py` |
| **口径 C** | W1-S：4096 in / 256 out、c=4、K=3、冻结 n=16 fixture、greedy，`total_accepted / wall`（含 prefill，但 4K prefill 占比小） | `mtp_w1s_our_runtime_perf.py --batched --cudagraph` |
| **口径 D** | cold 单批 accepted tok/s（**含完整 prefill**；长上下文下 prefill 主导） | D1 sweep / `w1s_native_bench.py` |

**A 与 C/D 不可互换**；**A 与 B 差一个 TTFT**（128K/c=4 时 TTFT 25.7s 摊在 256 个输出 token 上，
足以把比值从 >1 拉到 0.718）。今天 `notes/2026-08-03-performance-gap-vs-historical.md`
的「按每步读取量折算有效带宽」是正确的第三种可比法。

### 4.1 口径 C —— W1-S（4096in/256out, c=4, K=3, n=16 冻结 fixture, greedy）

对照的「原生」= **vLLM server + `--attention-backend CUSTOM`（即我们自己的 SM120 kernel）+ MTP**，
`ignore_eos=true`。**不是 FlashInfer。**

| 阶段 | accepted tok/s | 相对原生 144.54 | 关键改动 | 日期 |
|---|---:|---:|---|---|
| 单槽串行、eager | 11.60 | 12.46× 慢 | — | 07-17 |
| 跨槽批处理 | 16.61 | 8.7× 慢 | `+43%` | 07-17 |
| + split-KV 修复 | 18.78 | 7.7× 慢 | `+13.1%` | 07-17 |
| + ragged recompute 批处理 | 18.54 | 7.8× 慢 | **持平**（预测的 2.28× 没兑现） | 07-17 |
| + GPU 常驻 GDN snapshot（Phase 1） | **27.464** | 5.26× 慢 | `+48.1%` | 07-17 |
| + CUDA graph MTP 轮（Phase 3） | **66.152** | 2.185× 慢 | `+140.9%` | 07-17/18 |
| + Phase 2 spec-decode GDN（eager verify） | **78.565** | 1.840× 慢 | `+18.76%` | 07-18 |
| + verify CUDA graph 重新接上 | **136.750** | **1.057×** | `+74.06%` | 07-18 |
| + `torch.set_grad_enabled(False)` | **142.504** | **1.014×** | `+3.4%` | 07-18 |
| 后续回归门复测区间 | 142.5 / 147.656 / 148.193 / 147.931 / 165.730 / 156.939 / **162.638** | 0.89–1.15× | 代码逐字节未变 | 07-18~19 |

**这条线的三个必读注意**：
1. 142–166 的散布**不是代码差异**：`total_committed_tokens=4116` 与
   `draft_acceptance_rate_pct=70.29204431017119` 在全部测量里逐字节一致。归因是 Max-Q 热态
   （独立审计连测 3 次的散布只有 0.38 tok/s，56°C→73°C）。**引用时应引区间，不引单次。**
2. `136.750 → 142.504` 那一步的 `gpu_busy_s_summed_across_slots` 从 44.47s 降到 26.58s，
   **committed token 数完全一致（4116）** —— 这是「图化确实减少了 GPU 流上的耗时」的直接证据。
3. 批处理后 TTFT 从 693ms 变 ~2900ms **不是回归**：批版把整批共享 prefill 的成本记到每个槽头上，
   单槽版只记自己那次调用。**口径不同，不可比。**

### 4.2 口径 A —— 长上下文 warm 前缀命中（P + 10240-token 新 suffix, c=4, greedy, max_tokens=256, 3 次重复）

对照的「原生」= `launch_test_server.py --baseline-flashinfer`，**FLASHINFER backend（main + MTP 两侧）**，
`--kv-cache-dtype fp8_e4m3 --enable-prefix-caching --max-num-seqs 4 --max-model-len 262144
--enable-chunked-prefill --max-num-batched-tokens 8192`。**与 §4.1 的「原生」不是同一个东西。**

| 时点 | 128K/c=4 我们 | 128K 原生 | 64K/c=4 我们 | 64K 原生 | 关键改动 |
|---|---:|---:|---:|---:|---|
| 07-20 基线（**无 CUDA Graph**，hit 路径未分块） | 83.24 | 146.85 | — | 222.17 | — |
| + INV8 Phase A（hit suffix 分块 8192 + `effective_chunk`） | **105.4** (0.718×) | 146.85 | 115.4 (0.519×) | 222.17 | TTFT 28769→25694ms（**−10.7%**） |
| + `SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL=1` | 119.3 (0.812×) | 146.85 | 114.4 | 222.17 | kernel env |
| 07-20 晚（同上，另一批测量） | 104.74 | 146.85 | 121.52 | 222.17 | — |
| 07-20 深夜（原生重测） | 120.8 (**0.975×**) | 123.8 | 132.0 (0.696×) | 189.7 | split-KV 确认 |
| **+ CUDA Graph 内存修复**（`precapture_cuda_graphs`，不再翻倍 num_slots） | **154.7 (1.25×)** | 123.8 | **201.4 (1.06×)** | 189.7 | **+28.1% / +52.5%** |
| + pinned staging buffer | 157.3 (1.27×) | 123.8 | 205.7 (1.08×) | 189.7 | 显存 94.6→91 GiB |
| 3 次测量最终 | ~122（105–133 波动） | 117.4 | **200.6** | 175.2 (**1.145×**) | — |
| + padded draft step0 CUDA Graph | **159.5** | 117.4 | 194.3 | 175.2 | +20-23%（128K） |
| + numpy `_fill_buffers` | **166.3 (1.42×)** | 117.4 | — | — | +4.3% |
| + Triton RMSNorm mixed-dtype 修复 | **176.45** | — | — | — | **+6.5%** |
| + `replay_incremental` | 176.78 | — | — | — | 噪声内 |
| + `SM120_DECODE_VTRANSPOSE_ELIM=1` | 180.77 | — | — | — | +2.3% |
| + splits 64→32 | 181.95 | — | — | — | +0.65% |
| + page indices 缓存 | **183.43** | — | **236.69** | — | +0.81% |
| **+ KWIDE + V272 kernel（终值）** | **222.44** warm / 226.03 cold | — | **未测** | — | **+21.3%** |

**这张表的五个必读注意**：
1. **终值 222.44 是 128K；64K 的终值不存在。** `236.69` 是 **KWIDE 之前**（183.43 那一代）的 64K 值，
   PROGRESS.md 自己把「64K/c=4 benchmark 验证」列为**待运行**。`docs/qwen36-rebuild-spec.md`
   把 236.69 当 64K 基线是对的，但要知道它落后于 128K 的终值一代。
2. **「0.718×」是 07-20 上午的瞬时状态，不是历史终态**。同一脚本到 07-21 读到的是
   222.44 vs 原生 117.4 ≈ **1.89×**。今天 `notes/2026-08-02-qwen36-historical-performance-record.md`
   写「端到端报告值当时落后（0.718×）」的「当时」两个字是承重的。
3. **原生侧自己波动很大**：128K warm 在 117.4–146.85 之间，64K warm 在 175.2–222.17 之间，
   同一天不同批次能差 20%。**任何我们/原生比值都必须同批次。**
4. PROGRESS.md:4243 的「vs native vLLM cold: **68x faster**」是 **warm 比 cold**，
   **不可比**，不要引用。
5. 07-20 那一批全部是 `enable_cudagraph=False`（长上下文下 num_slots 翻倍会 OOM）。
   **「CUDA Graph 内存修复」那一行是整条曲线上最大的单项收益，且它修的是我们自己的限制，不是 kernel。**

### 4.3 口径 B —— 稳态 decode（扣 TTFT）

| 场景 | 数字 | 来源 / 注意 |
|---|---|---|
| 128K/c=4，**推导值** | 我们 ≈262 agg tok/s ≈15 ms/step；原生 ≈197 ≈20 ms/step | `notes/2026-07-20-comprehensive-optimization-plan.md` §0。⚠️ **这是从 105.4 / 146.85 与 TTFT 反推出来的，不是直接测量。** |
| 128K/c=4 warm，eager，直接测 | 130.9 ms/step，109.5 tok/s，`kv_len=141312` | `decode_step_profile.py`，07-20 |
| 128K/c=4，开 CUDA Graph | steps/sec 11.47（eager 9.49），accepted tok/s **147.44**（eager 122.01） | 07-20 深夜 |
| 128K/c=4，numpy 优化后 | steps/sec 13.36，**174.30 accepted tok/s（稳态）**，committed/step 13.05（4 槽合计），acceptance 0.5210，step 74.87ms | 07-20，**这是最可靠的一个稳态数字** |

### 4.4 口径 D —— cold 形状扫描（含完整 prefill）

| 上下文 / c=4 | 我们 | 原生 | 配置 | 日期 |
|---|---:|---:|---|---|
| 16K | 25.137 → 46.394 → 58.638 → **67.232** | 121.960 | 依次为：原始 / last-position logits 修复 / +cudagraph / +chunk 8192 | 07-18/19 |
| 32K | 29.522 | 32.941 | `--batched --cudagraph`，峰值 84.6% | 07-18 |
| 64K（chunked） | 13.386 / 13.950 | 10.800 | `--blocks-per-slot 5120 --chunk-size 8192`，峰值 51.6% 平坦 | 07-19 |
| 128K（chunked） | 5.014 | 3.270 | | 07-19 |
| 200K（chunked） | 2.434 | 2.598 | 两侧峰值 79.5% | 07-19 |
| 256K（chunked） | 1.557 | 0.580 | **两侧均可行**，峰值 82.8%，无 OOM | 07-19 |
| 64K/c=1 | 10.290 | 9.117 | 无 cudagraph | 07-18 |
| 64K/c=2 | 11.498 | 14.484 | 无 cudagraph | 07-18 |

**注意**：`--blocks-per-slot` 在不同 cell 之间是变的（2560 / 5120 / 262656），
`--chunk-size` 也是（None / 8192）。**跨 cell 比较前先核对这两个值。**

### 4.5 前缀缓存

**warm 命中吞吐（suffix=10240, max_tokens=256, gpu_mem_util=0.85, 无 cudagraph, 07-20）**

| P | c | cold TTFT | warm TTFT | TTFT 加速 | warm tok/s | cold tok/s | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|
| 64K | 1 | 17,133 ms | 3,962 ms | 4.32× | 41.15 | 28.89 | 52.9 GiB |
| 64K | 4 | 16,881 ms | 15,681 ms | 1.08× | 114.28 (agg) | 97.02 | 62.5 GiB |
| 128K | 1 | 48,230 ms | 6,141 ms | 7.85× | 33.05 | 40.78 | 76.4 GiB |
| 128K | 4 | 48,877 ms | 28,769 ms | 1.70× | 83.24 (agg) | 100.27 | 92.9 GiB |
| 200K | 1 | 131,034 ms | 8,566 ms | 15.30× | 10.34 | 10.99 | **97.3 GiB ⚠️** |

⚠️ 200K/c=1 溢出到 CPU 共享内存被节流 ⇒ 那个 8.57s 的 warm TTFT **可能比正常配置更慢**，
根因是 `_prefill_cold_with_populate` 未分块。**c=4 那两行的 TTFT 加速只有 1.08/1.70×，
正是 INV8 未解除时 hit-suffix 不分块的直接后果。**

**单槽长上下文（P3.4，64K，exact-repeat）**：cold TTFT 16,744.5ms；
warm turn2（P+64）**178.8ms → 93.67×**；turn3（P+128）190.8ms → 87.75×。
GDN layer-0 committed 行**逐字节** 0.0，8 轮解码 20/20 token 相同。
R8 内存轨迹 [30458, 30608, 30608, 30608] MiB（漂移 0.49%）；
R9 哈希开销：4K=0.579ms（占 cold TTFT 0.0035%）、64K=9.74ms（0.058%）。

**server e2e（P4a）**：2534-token prompt，turn2 命中 L=2528（= `block_align_down(2534−1)`），
TTFT 1.25s → 0.91s（1.38×，短 prompt 下本就有限）。

**原生 vLLM 自己的 APC**：256K byte-identical 重复 775.9s → 49.6s = **~15.4×**（重启复测排除污染）；
128K warm 场景 2.14M 命中 / 9.08M 查询，cold 91.1s → warm 4.4s = **20.6×**。

### 4.6 接受率 —— 🔴 这组数字被一个 benchmark bug 污染过，必读

**先说结论**：2026-07-21 发现 `benchmarks/native_warm_compare.py` 的 Prometheus 抓取有
**双重计数 bug** —— `"num_accepted_tokens" in metric_name` 同时匹配了主计数器
`vllm:spec_decode_num_accepted_tokens_total` 和 per-position 计数器
`vllm:spec_decode_num_accepted_tokens_per_pos_total`，把 `delta_accepted` **膨胀约 2×**。
修复：第 101 行加 `"per_pos" not in metric_name`。

**修正后（128K/c=4 warm）**：原生真实接受率 **64.2%**（mean_acceptance_length **2.926**）
vs 我们 **66.7%**（tokens/step **3.0**）—— **我们略优于原生**。

⇒ **所有基于「原生 4.852 vs 我们 4.0/3.3」的推论都作废**，包括
`notes/2026-07-20-comprehensive-optimization-plan.md` §1A 那张表里
「接受长度占差距 20-25%」这一行，以及今天
`notes/2026-08-02-qwen36-historical-performance-record.md` 转述的同一行。
**接受长度当年不是我们的短板。**（我们自己的 3.0–3.3 tokens/step 这个绝对值仍然成立，
今天 `notes/2026-08-03-performance-gap-vs-historical.md` 拿它当历史基线是对的；
错的只是「原生更高」这个比较。）

**其余接受率测量（全部注明口径，彼此不可直接比）**：

| 值 | 口径 | 日期 | 备注 |
|---|---|---|---|
| 原生 70.38% vs 我们 67.25% | 4096in/256out, c=4, 双方 greedy，per-draft-token 接受率 | 07-17 | 修掉两个混淆项后的第一次公平比；3.1pp |
| 原生 76.81% vs 我们 82.31% | 同上但样本扩大 | 07-17 | **方向翻转** ⇒ 说明不是稳定机制差 |
| 原生 94.46% vs 我们 82.31% | 4 请求 × 2000 输出 token（形状对齐） | 07-17 | 12.15pp，但**有效样本量只有 4 条轨迹** |
| 原生 79.51% vs 我们 73.06% | 冻结 fixture n=16、256 输出 | 07-17 | 6.45pp，~1.6 combined SE |
| **原生 72.59%（n=128）vs 我们 71.25%（n=64）** | 同冻结 fixture | 07-17 | **1.34pp，< 1 combined SE ⇒ 统计上不可区分。这是这条线的收敛答案。** |
| 我们 **70.29204431017119%** | 4K/c=4 W1 headline | 07-18~20 | 跨 20+ 轮改动逐字节复现的锚点 |
| acceptance 0.5171 / 0.5210；committed/step 12.85 / 13.05 | 128K/c=4 warm，**4 槽合计** | 07-20 | 除以 4 ≈ 3.2 tokens/step/槽 |
| **50.3%**（约每轮 2 token） | 128K/c=4 warm，与 222.44 tok/s 同批 | **07-21（终值）** | |

⚠️ **一个长期存在的测量陷阱（当年花了两轮才定位）**：随机 token 输入 + 无重复惩罚的贪心长生成
会退化成重复模式，**两侧模型都能轻易预测自己的重复** ⇒ 接受率被机械抬高。
同一 run 的前 341 个 draft 是 67.25%，后 ~1966 个是 **~84.9%**。原生也一样
（短请求 76.81% → 4×2000 输出 94.46%）。**任何接受率数字必须连生成深度一起引用。**

### 4.7 显存

| 场景 | 值 | 日期 / 注意 |
|---|---:|---|
| 128K/c=4 warm | 92.9 GiB（分块前）→ 90.7 GiB（Phase A 后）→ 94.6 GiB（CG）→ 91 GiB（pinned）→ 95.5 GiB（padded step0） | 07-20/21，卡是 97887 MiB |
| 64K/c=4 warm | 63–65 GiB | 07-20 |
| 200K/c=1 cold-populate（未分块） | **97.3 GiB，溢出被节流** | 07-20 |
| 256K/c=4 cold chunked | 82.8% 峰值，无 OOM | 07-19 |
| 200K/c=4 | **两侧均不可行（>95 GiB）** | 07-20 |
| GDN snapshot 缓冲（4 槽） | ~604 MB | 设计值 |
| 32K prefill 瞬时激活 | ~22 GB（持久 ~39 GB 之上）；4 次连续 prefill **完全平坦 63353 MiB** | 07-17 |
| KV 每 token | 2×4 heads×256×1B(fp8) = **2 KB/token/层**；16 层 × batch4 @131K ≈ **8.4 GB 每 verify 步读取** | |

### 4.8 质量

| 指标 | 值 | 配置 | 日期 |
|---|---|---|---|
| MMLU-Pro | **84.54%**（官方 86.2，−1.7pp，在 414 题 ±3.5% 抽样噪声内） | 414 题分层抽样，thinking，5-shot CoT，greedy，max_tokens=32768，零截断 | 07-22 |
| HumanEval | 我们 44.5%（73/164）vs stock vLLM 43.3%（71/164） | greedy，evalplus，**max_tokens=768（已知会低估，两侧同等）** | 07-21 |
| HumanEval+ | 我们 43.3%（71/164）vs 42.7%（70/164） | 同上 | 07-21 |
| AIME26 / GPQA | **无记录**（测过但撤销） | | |

**端到端生成质量门（Phase A，07-18）**：8 个 prompt 贪心 256+ token，对照 **不开投机的原生 vLLM
（同 CUSTOM backend、同 fp8 KV）**：2/8 逐 token 完全一致；其余 6 个各在**恰好一个**位置分叉，
且**全部**是真实 near-tie（margin 0.125–0.625 logit 单位，远低于 `NEAR_TIE_LOGIT_MARGIN=2.0`），
分叉后的续写全部流畅切题。

### 4.9 Profiling 分解（⚠️ 全部是 vLLM kernel 的分解，见 §6）

| 时点 / 场景 | Attention | GEMM | GDN | 其他 |
|---|---:|---:|---:|---|
| 07-15，batch=1，无 MTP，全模型 nsys | 1.5% | **76.0%** | 8.0% | 采样 3.7% / quant 2.0% |
| 07-20，128K/c=4 warm，eager | **78.0%** | 6.8% | 1.7% | 13.5% |
| 07-20，同上，原生 FlashInfer | 60.1% | 25.1% | 2.6% | 12.2% |
| 07-20，128K/c=4，开 CUDA Graph | **87.3%** | 1.7% | 1.5% | |
| 07-21，Triton RMSNorm 之后 | 52.0%（32.83ms） | 33.8%（21.38ms） | 2.5% | RMSNorm 3.2% / copy 3.4%；总 CUDA 63.26ms + CPU 7.03ms = 70.29ms/步 |
| 07-22（`notes/2026-07-22-a1a-gdn-profiling.md`，eager，c=1） | 3.5%（4K）/ 28.2%（128K） | **71.1%（4K）/ 53.7%（128K）** | **5.1% / 3.9%** | |

**跨行不可直接比**：batch/并发/上下文/是否开图/kernel 版本全都不同。
唯一稳定的结论是 **GDN 从来不超过 ~8%，从来不是瓶颈**。

### 4.10 更早期的、已被后续工作推翻的数字（别再引用）

- 「GPU-busy% ≈ 95-101%，所以没有 launch gap 可挤」—— **口径错误**。Phase 0 的 nsys 台账证明
  busy% 是一个**跨度**（≈墙钟），真正的 kernel-active 只有 **8.8-10.2%**，
  无 kernel 的空隙占 **66.6-72.5%**。这解释了「95% busy 却 30% utilization」的长期矛盾。
- 「recompute fallback 值 ~2.28× 提速」—— **投影错误**。按 recompute 槽数分桶比较墙钟时，
  没扣掉「recompute 轮本来就 commit 更少 token」。真实修完是**持平**（18.78→18.54）。
- 「原生比我们快 12.5×」（07-17）—— 那是单槽串行、无图、无 Phase 1/2 的状态，早已作废。

---

## 5. 当年已知的问题：修了的 vs 仍然开着的

### 5.1 已修（附根因，今天可能重演的用 ⚡ 标出）

| 问题 | 根因 | 修法 |
|---|---|---|
| ⚡ 输出全错（Phase 3 初期，20/20 确定性失败） | 手写 metadata 把逻辑槽当物理索引，**vLLM 调度器从不把物理索引 0 分给真实请求** | `RESERVED_PHYSICAL_SLOTS=1` + `_physical_slot()`。⚠️ **这是 vLLM 事实不是硬件事实**，今天不该照抄 |
| ⚡ 每轮 ~25 MiB 显存单调增长（1107 轮后 69 GB allocated / 99.3% 卡） | **从没关过 autograd** —— 每次 eager forward 建一棵永不释放的图 | `torch.set_grad_enabled(False)` 放在 `__init__` 第一行 |
| ⚡ 16K/c=4 峰值 99.2% 显存、慢 4.85× | prefill 把 `[65536, 248320]` bf16（~30.3 GiB）算了**两遍**（target + draft），只读 4 行 | `logits_last_position_only`（默认 False，只 prefill 传 True）⇒ 显存 99.2%→55.4%，TTFT 34.0s→12.5s |
| ⚡ 接受率从 70.29% 莫名变 76.67% | CUDA Graph 类里有自己一份**过期的** `TARGET_SPLITS=16`，与 eager 的 64 不同 ⇒ attention reduction 顺序变 ⇒ near-tie 翻转 | 图里直接读 runner 的 `decode_fixed_*` |
| ⚡ 图 replay 与 eager 差 `max_abs_diff=7.93`、`cosine=0.55` | `capture()` 的 3 次 warmup 跑在**后面还要 replay 的同一批槽**上，GDN 递归状态**非幂等** | 专用 disposable warmup 槽；后改为 `precapture` + 立即 reset |
| ⚡ `build_gdn_metadata_batch` 快路径走错分支 | 判定写成 `isinstance(qo_len, int) and qo_len == 1`（**按类型**），`[1]` 这个列表落到 chunked 路径 | 改成 `all(qo == 1 for qo in qo_lens)`（**按值**） |
| `_mtp_sync_and_propose` 第 2 个探索步之后 draft 全错 | 探索循环把冻结的 `slot_draft_sync_len` 当每步的 `prior_kv_len`，而真实写位置在推进 | `_mtp_forward` 收显式 `prior_kv_len`；循环维护 `running_prior_kv_len` |
| `build_attention_metadata_batch` 把 chunked prefill 路由进 decode kernel | 无条件 `decode_qo_len = qo_len` | 加 `is_decode` 参数（默认 True，只 prefill 传 False） |
| `reset_slot` 不清 `slot_draft_sync_len`/`slot_pending_draft_tokens` | — | 补清 |
| GDN snapshot 的 generation 计数器不绑槽、restore 后不标记已消费 | — | 加 slot-id tag + consumed 标记 |
| eager 批路径完全没有 split-KV（`max_num_splits==1`） | `kv_split_size` 从活的 kv_len 现推 | 从每槽容量派生固定值 |
| server 第一个请求就崩 | `apply_chat_template(tokenize=True)` 返回 `BatchEncoding` 不是 list | `return_dict=False` |
| 补全少了第一个 token | `committed_tokens` 从来不含 prefill 产出的 anchor | 准入时 `committed_tokens=[anchor]` |
| 请求正好落在容量边界会中途崩 | `capacity_ok()` 零余量，MTP K=3 前瞻会瞬时超 | 加 `self.K` token 余量 |
| Triton RMSNorm **完全没生效**（161 个 `GemmaRMSNorm` 全部回退原生 PyTorch） | `forward_native` 里 `self.weight.float() + 1.0` 让 weight 变 fp32、x 是 bf16，`supports_args` 要求同 dtype | 去掉 dtype 约束，kernel 内统一 fp32 ⇒ **+6.5% e2e** |
| 「cold prefill 对分配历史敏感」的 GDN 异常（47.015625 / 58.765625） | **诊断脚本自己的 bug**：把 hit 的 @19200 克隆去比 cold 的**解码后活体** @19235/@19245 | 同位置比较后 `(0.0, 0.0)` 逐字节一致。**不是 runtime 缺陷** |
| 接受率对比「原生更高」 | `native_warm_compare.py` Prometheus 双重计数（§4.6） | 加 `"per_pos" not in metric_name` |

### 5.2 当年知道、但**没有修**（今天可能继承）

| 开放项 | 详情 | 今天状态 |
|---|---|---|
| **`chunk_size` + 真 ragged batch** | `mtp_prefill_batch` 直接 `raise NotImplementedError`（`oracle/.../backends/qwen36.py:1043-1050`），到剥离为止一直在 | 今天逐槽处理，问题结构性消失 |
| **ragged prefill × CUDA Graph 从未同时跑过** | 所有 ragged/准入测试都 `enable_cudagraph=False`。论证（prefill 恒 eager）成立但**未实证** | 今天 MTP 完全不走图，问题形态不同 |
| **mid-flight 准入（batch 变大）× CUDA Graph 从未同时跑过** | 图按 `len(slots)` 查表，**变大**方向从未走过；且 cudagraph 要求准入槽落在前 `concurrency` 个逻辑索引内 | **仍然相关**：今天 `qwen36.py:974` 每个 batch size 一张图，方向问题类似 |
| **`mtp_async_arrival_check.py` 的 7.9375 logit 分歧** | 📛 **仓库内两处记载互相矛盾**：`PROGRESS.md`（07-18 条目）宣称已关闭、"每个 passed 门都绿了"；`notes/2026-07-19-comprehensive-audit-and-forward-plan.md` §3.2 亲自重跑得到 `passed: false`（`near_tie_margin: 7.9375`，round 13）。**本文档不裁决，两条都记下** | 该脚本今天仍在 `benchmarks/`，import `oracle.*` |
| **请求级错误处理缺失** | 超容量请求在 prefill 中途抛 `RuntimeError`，**整批 forward 一起崩**，而不是只拒这一个 | **[待核实]** 今天是否有等价保护 |
| **compute-sanitizer 干净门从未满足** | 4 次尝试全部卡/淹在 `_warmup()` 里。两个**先于本项目存在**的缺陷：① `causal_conv1d_fn` 进程内**首次调用返回全零**；② `qwen_gdn_linear_attn.py::_output_projection` 未初始化内存读（100 例） | 都是 vLLM/Triton 侧，今天不适用 |
| **进程内跨 rep 显存增长** | 45→72→80/95 GB，不影响正确性、退出即释放，**未解释** | grad 修复后基本消失，但从未单独复查 |
| **无小时/天级连续运行证据** | 最长 63.5 分钟 / 7720 轮（内存逐字节平坦）。SIGTERM 中断导致事后正确性判定没跑完 | 今天 Track C 的 C5（24h soak）就是这条 |
| **200K/c=4 双方均不可行** | >95 GiB KV，不是我们的实现问题 | 结构性，仍成立 |
| **`_prefill_cold_with_populate` 未分块** | 导致 200K/c=1 峰值 97.3 GiB 溢出 | 今天 prefill 分块了，但缓存 populate 路径不存在 |
| **FlashInfer decode 在 qo_len=4 下正确性差** | `benchmarks/flashinfer_decode_feasibility.py`：qo_len=1 cos=0.9997，**qo_len=4 cos=0.13**，未查 | 无关（今天不用 FlashInfer） |
| **`mtp_sustained_realistic_workload_check` 的 pool_size/capacity 排队失配** | 测试参数化问题，不是 runtime bug | 脚本仍在 |
| **投机 + 非贪心采样** | 明确拒绝（"M1 阶段正确性风险过高，先分离"） | 今天做了一半（one-hot draft_probs，`mtp_accept.py:270-275` 标注） |

---

## 6. 计划过但没建的（连同当年自己的理由与预期收益）

来源：`notes/2026-07-20-comprehensive-optimization-plan.md`（Q1-Q4 / M1-M3 / L1-L3）与
`docs/archive/2026-07-20-PROGRESS.md` 末尾的路线图。

| 项 | 内容 | 当年预期 | 最终结局 | 今天有没有 |
|---|---|---|---|---|
| **Q1** 缩小 warmup 槽的 KV 分配 | 让长上下文也能开 CUDA Graph | +10-20% | **被更好的方案取代**（`precapture_cuda_graphs` 用真实槽热身后 reset），实测 +28-52% | 今天等价（`cg_extra=1`） |
| **Q2** ragged step0 padding 图化 | step0 在 70% 接受率下几乎每轮都退 eager | +3-6% | **建成**（07-20），实测 128K **+20-23%**（远超预期） | ❌ 缺失 |
| **Q3** `NATIVEFP8` decode kernel | env 开关 A/B | +5-15% | **建成**，+13%（128K）/+24%（64K） | 不适用（vLLM kernel） |
| **Q4** 分别报告稳态与含 TTFT | 修口径 | 0 | 部分做了（§4.3） | 今天 `bfdiag` 更强 |
| **M1** **跨步交织 chunked prefill** | 允许「部分 prefill」槽，每 `_step` 推进一个 chunk | TTFT 25.7s → ~5-7s；128K 105 → 140+ | **建成**（`a8bd167`，07-22），但**只有短 prompt 不被阻塞的定性验证，没有 128K TTFT 数字** | ❌ stub（M-4） |
| **M2** decode attention 路由到 FlashInfer | 当年以为 SM120 kernel 慢 5.8× | +15-30% | **取消**：微基准显示 SM120 NATIVEFP8(split=64) 1.561ms/层 vs FlashInfer(tensor_cores) 1.528ms/层，**只差 2%**；且 FlashInfer AOT 不支持 GQA group_size=6 | 不适用 |
| **M3** 把 K 个 draft 提议合成一次 tree-attention 前向 | 每步 4 次前向 → 2 次 | +5-10% 稳态 | **从未建** | ❌（今天是 K 次串行 + 1 次 anchor + 1 次 verify，更差） |
| **L1** 48 层 GDN 融合（~9 kernel/层） | 上限 8% | **从未建**；后续 profiling 把 GDN 压到 3.9-5.1%，优先级进一步降低 | ❌，且**不该做** |
| **L2** NVFP4 GEMM 权重布局 / quant-dequant 融合 | 上限 76% | 部分（`106e2e5` A2 自研 GEMM，声称比 vLLM 快 17.4%） | 今天走 sparkinfer |
| **L3** 接受长度恢复（"draft/target kernel 数值分歧"） | 20-25% 的差距 | **问题不存在**：那个差距是 §4.6 的 benchmark bug | 不适用 |
| TMA（`cp.async.bulk`）集成进 decode kernel | 带宽 674 → 741.9 GB/s（+10.1%，已用微基准验证） | 预期 e2e +5-6% | **只写了 `csrc/decode_tma_load.cuh` 头文件，从未集成** | 不适用（vLLM kernel） |
| Adaptive split（verify 用 32、draft 用 128） | +0.5% | 分析过，未建 | 不适用 |
| `torch.compile` kernel 融合 | +1-3% | 未研究 | ❌ |
| KV cache 压缩（让 200K/c=4 可行） | — | 未建 | ❌ |
| INV8 硬化第 3 步（分块 `_prefill_cold_with_populate` 与 `warm_continue`） | 让 200K/c≥2 可测 | 未建 | ❌ |

---

## 7. 不可迁移的结论（别把死路口再走一遍）

历史实现坐在 **vLLM fork**（`/home/bot/vllm`，`vllm/v1/attention/backends/sm120_gqa.py`，1060 行）
+ **自研 CUDA kernel**（`/home/bot/project/sm120-flash-attention/`）上。
今天走 **sparkinfer**。以下结论**全部关于那套 kernel，与今天无关**：

### 7.1 attention kernel（全部不适用）

- ❌ 「SM120 decode attention 比 FlashInfer 慢 **5.8×**，是最大的单项可优化点」（07-20）
  —— 后来同一份文档系列自己推翻：**在正确 split 配置下只差 2%**，
  再后来微基准说我们**快 FlashInfer 13.4%**（1.63-1.71ms @ 676-710 GB/s vs FI 1.94ms @ 597 GB/s）。
  三个互相矛盾的数字并存，**唯一安全的做法是对 sparkinfer 重测，一个都不搬**。
- ❌ 占用率路线全封死：`__launch_bounds__(256,2)` → 慢 1.79×（寄存器 spill）；
  `setmaxnreg.inc` 硬上限 224 而 compute 需 255；V3 warp-specialized（126 regs，2 CTA/SM）
  **速度与 V2 持平**（纯内存受限，额外 `__syncthreads` 抵消收益）。
- ❌ split 数调优：微基准说 split=2048 比 4096 快 17%，**端到端却退化 30%**（166.3→115.9）。
  教训（**这条是可迁移的**）：**split/tile 调优必须端到端验证，微基准会骗人**。
- ❌ FP4 KV：标量版慢 12×；移植 tensor-core 后 cos=0.999993 且比标量快 4×，
  但**仍比 FP8 慢 3×**。裁决：**FP4 KV 的价值在容量（2× KV cache），不在速度**。
  （**这条结论本身可迁移**：FP4 是 compute-bound，e2m1 无硬件解码指令。）
- ❌ KWIDE + V272（16 字节 `cp.async` + V smem stride 260→272 对齐）：微基准 1.56×，
  **e2e +21.3%（183.43 → 222.44）**。这是历史终值的最后一跃，**完全是 kernel 层的**，
  今天没有对应物。⇒ **222.44 这个基线里有一大块是今天不可能自动获得的。**
- ❌ 全部 `SM120_*` env 变量（`USE_V2_DECODE_KERNEL`、`USE_V2_DECODE_NATIVEFP8_KERNEL`、
  `DECODE_VTRANSPOSE_ELIM`、`PREFILL_V2_KWIDE`、`QO1_NATIVEFP8_MIN_KV`、`MINBLOCKS`）。

### 7.2 GEMM / 量化（不适用）

- ❌ 「GEMM 占 76% GPU 时间」—— batch=1、无 MTP、vLLM 自己的 NVFP4 kernel 上测的。
- ❌ 四个 `nvfp4_*_patch.py`：全部是往 **vLLM 的 kernel 注册表/优先级列表**里打的猴子补丁。
  sparkinfer 的 `dense_gemm` 是 SM120/121 上唯一的 warp-MMA 引擎，选择歧义**结构性不存在**。
- ❌ `nvfp4_cudnn_patch.py` 号称 +12.6% 但**默认关闭、自承对真实权重不 bit-exact**。

### 7.3 vLLM/CUTLASS/Triton 侧的缺陷（不适用）

- ❌ `causal_conv1d_fn` 首次调用返回全零（Triton 冷启动）。
- ❌ CUTLASS SM120 pingpong GEMM 的 `racecheck` RAW hazard（100 例）——
  真实定位过，但**被证明不是错误输出的根因**（禁用该 kernel 后输出仍错，只是错法不同）。
- ❌ `_C.abi3.so` 缺 `rms_norm`/`fused_add_rms_norm` 符号 ⇒ Triton RMSNorm 的存在理由。
- ❌ `RESERVED_PHYSICAL_SLOTS=1`（vLLM 调度器约定）。
- ❌ `_MAX_DECODE_QO_LEN=16`（绑定 vLLM `SM120GQAMetadataBuilder` 的测试范围）。

### 7.4 「原生」对照本身的可比性

- §4.1 的原生 = **vLLM + 我们自己的 CUSTOM SM120 backend + MTP**
  ⇒ 那条线量的是 **scheduler / runtime 的差异**，不是 kernel 的差异。
- §4.2 的原生 = **vLLM + FLASHINFER backend（main + MTP 两侧）**
  ⇒ 那条线量的是 **kernel + scheduler 一起的差异**。
- **两条线的「原生」数字不能互相引用。**

### 7.5 ✅ 反过来，这些是**可迁移**的（关于工作负载与算法，不关于 kernel）

1. **诊断结构**：长上下文下 TTFT/prefill 调度 > 接受长度 > 稳态 decode 步 —— 这个**排序**成立。
2. **GDN 从来不是瓶颈**（≤8%，后测 3.9-5.1%）。
3. **GDN 递归状态非幂等** ⇒ 任何 warmup/重放机制都不能用真实请求槽。
4. **GDN 的每位置输出对 K+1 个候选都是因果有效的，只有状态 commit 需要知道接受数** ⇒ 零回滚可行。
5. **跨槽批处理与 CUDA Graph 是两个独立且都很大的杠杆**（+43% 与 +141%/+74%）。
6. **微基准 ≠ 端到端**（split 调优 −30% 的教训）。
7. **near-tie 方法论**：`NEAR_TIE_LOGIT_MARGIN=2.0`；GDN layer-0 必须逐字节相等作为寻址证明，
   48 层累积用 near-tie。fp8/批次非结合性下逐字节相等**不是可达的标准**（原生 vLLM 自己也做不到）。
8. **接受率必须连生成深度一起报**（重复退化会机械抬高接受率 18-24 个点）。
9. **散布必须归因**：热态能造成 142-166 的散布；正确性锚点逐字节一致才能证明不是代码问题。

---

## 8. 附录：要再查什么，去哪查

| 想知道 | 去哪 |
|---|---|
| 历史实现的**代码**（最终态） | `oracle/qwen36_vllm/`（**不要**去 `qsr-hist`，那是 07-21 的旧版） |
| 历史实现的**逐轮记录**（4266 行，本文档数字的一手来源） | `docs/archive/2026-07-20-PROGRESS.md` |
| 07-21 那个时间点的**单文件 6170 行版本** | `/home/bot/project/qsr-hist/runtime/direct_model_runner.py` |
| 前缀缓存的**不变量与风险登记**（INV1-9 / R1-6） | `notes/prefix-cache-design.md` |
| 前缀缓存的**逐阶段实施日志** | `notes/prefix-cache-implementation-log.md` |
| MTP 从零到能跑的**完整推导链**（含四次调试大回合） | `notes/direct-model-runner-design.md`（4067 行） |
| Phase 0-3 的**再诊断与逐阶段结果** | `notes/2026-07-17-post-ragged-round-next-steps.md`（1894 行，§9.3 是「什么被捕获」权威表） |
| 07-20 的**瓶颈分解与 Q/M/L 计划** | `notes/2026-07-20-comprehensive-optimization-plan.md`（⚠️ §1A「接受长度 20-25%」已作废，见 §4.6） |
| INV8 Phase A/B 的**设计** | `notes/2026-07-20-inv8-chunked-hit-prefill-plan.md`（⚠️ Phase B「未建成」已过期，见 §0.1） |
| kernel 层的**全部裁决**（全部不适用，见 §7） | `notes/2026-07-21-kernel-comprehensive-review.md`、`notes/2026-07-20-kernel-optimization-plan.md` |
| 独立审计视角的**开放项清单** | `notes/2026-07-19-comprehensive-audit-and-forward-plan.md` |
| 「cold prefill 分配敏感」那条**假警报**的完整根因 | `notes/2026-07-20-cold-prefill-rootcause-plan.md`（+ `-investigation.md` 是提出问题那份） |
| `oracle/` 的**逐模块可移植性判定** | `docs/qwen36-rebuild-spec.md`（⚠️ §1.4 一处内部矛盾见 §0.3） |
| 今天 Qwen3.6 的**实测现状** | `notes/2026-08-03-performance-gap-vs-historical.md`、`-fp8-kv-cache.md`、`-cudagraph-vs-eager-decode-throughput.md`、`-mtp-serving-gpu-verification.md` |

### 8.1 本文档里仍然 [待核实] 的东西

1. `a8bd167` 的跨步交织**在 128K 上的 TTFT 前后对比** —— 提交自己写 `Not-tested`，
   仓库内也找不到。**「M1 建成了」是代码事实；「M1 兑现了预期的 5-7× TTFT 下降」没有证据。**
   而且今天**连量它的仪表都是坏的**（`prefill_done()` 无生产调用方，见 §2.3）。
2. `mtp_async_arrival_check.py` 到底是绿是红（§5.2，仓库内两处记载互相矛盾）。
3. 128K 长上下文下，sparkinfer 在 **无 split-KV**（`max_partial_rows=0`）时 decode attention 的
   CTA 数与带宽利用率 —— 历史在同一形状下需要 32 splits/请求才把 CTA 从 16 拉到 ~512（M-9）。
4. 历史 GDN checkpoint 预算 8 GiB vs 今天 ~154 MiB 的差距，对真实多轮会话命中率的影响（§2.4）。

### 8.2 已在本轮核实、可以从「未知」里划掉的

- ✅ **watchdog 保留了**（`server/engine.py:1723-1760`，200 轮），并且今天还多了 600s 请求超时。
- ✅ **超容量今天在准入前就被拒**（`capacity_ok`），不再是「整批 forward 一起崩」。
- ✅ **`PROGRESS.md` 没有被删**，是 rename 成 `docs/archive/2026-07-20-PROGRESS.md`。
- ✅ **跨步交织在历史里是建成了的**（`a8bd167`，07-22），INV8 计划文档的「未完工」已过期。
- ✅ **接受长度当年不落后于原生** —— 「原生 4.85 vs 我们 4.0」是 benchmark 双重计数 bug（§4.6）。

### 8.3 一个容易踩的上下文事实

今天 server 的**默认 checkpoint 是 Laguna**（`server/app.py:89`，`poolside/Laguna-S-2.1-NVFP4`），
backend 由 `config.json` 解析决定（`server/app.py:395`）。
本文档所有「今天」的 Qwen3.6 结论指的是 `backend="qwen36"` 这条路径，
它**不是默认启动路径**；`QSR_SERVER_ENABLE_MTP` 也默认关。
拿「今天实测 xx tok/s」和历史比之前，先确认那个数字是 Qwen3.6 还是 Laguna 的
（这个坑仓库已经踩过两次：`c53bd7c` 的 MMLU-Pro 84.5% 归属、
`notes/2026-08-02-qwen36-historical-performance-record.md` 里的五条 Laguna 吞吐）。
</content>
</invoke>
