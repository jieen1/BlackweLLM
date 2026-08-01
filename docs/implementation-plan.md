# BlackweLLM 实施计划清单（2026-08-01 编制 · P0 已按最新状态重排）

> 基线：`main @ 619a09d`。上游文档 [`roadmap.md`](roadmap.md) 的基线是 `ce21eb5`，
> 其后 main 已推进 **49 个提交**，路线图里若干"未解决"条目实际已经关闭。
> 本文档 = roadmap 的**执行视图**：状态已按当前代码核实，条目按优先级排成一条可执行的序列。
>
> roadmap.md 仍是**目标与理由**的权威来源；本文档只回答"下一个动作是什么、谁卡着谁"。
> 标 **[待办·开发执行]** 的条目需要 GPU / 真机 / 下载权重，编制者不代跑。

---

## 0. 相对 roadmap 的状态修正（核实过，非推测）

roadmap §1.2 / §1.3 / §4 里这些条目**已经关闭**，排期时不要再计入：

| roadmap 条目 | 文档状态 | 实际状态（`main @ 619a09d`） |
|---|---|---|
| T0-7 仓库卫生 | 🔴 未做 | 🟢 **基本完成**：`benchmarks/` 136 → 108 个脚本（删 29 个零引用脚本 + vLLM bit-exact 家族，加 README 规约）；根日志 / `build/` / `evalplus_results/` **全部已被 `.gitignore` 覆盖**（`.gitignore:6,24,25,28,29`）——它们是磁盘杂物，不是仓库污染。真正待处理的只有 1 个 untracked 文件和 4 个已合并分支 |
| T0-8 N6 全套件 flaky | 🔴 未做 | 🟡 **已处理但未根因**：`914e3f2` 把断言精确到目标行 + 失败时打印原始输出与存活线程。~28 次真机重跑（GPU 40–60% + 4 核满载）**零复现**，与简报的 3/5 有明显差距，根因未定 |
| T0-8 N7 Protocol 不符 | 🔴 未做 | 🟢 已修（`76bcf3e`） |
| N2 `stop` 序列未实现 | 🔴 | 🟢 已实现，含跨 token 边界匹配（`0cd772c`） |
| N3 `seed` 每 token 重播种 | 🔴 | 🟢 已修为每请求推进单一 generator（`0700b25`） |
| N1 结构化输出是空壳 | 🔴 最严重 | 🟡 **危险性已消除、功能缺口仍在**：`b2d73cb` 改为显式拒绝 `json_object`/`json_schema`，不再静默失败。`GrammarState.apply_mask` 依旧无调用点 → **降级为普通功能开发，不再是 P0 事故** |
| N4 / C0 bfdiag 隔离保证失效 | 🔴 优先 | 🟡 **主体已完成**：审计成文（341 行），逐模块判定 real / fake / honest-split，删掉伪装成回归门禁的合成张量 demo，新增真调用 `restore_checkpoint`、`reset_laguna_engine` 的测试。剩余见 C0R |
| **Track A 设计定稿**（M1 交付物） | — | 🟢 **已完成 2026-08-01**：草案（`architecture.md` §3.1–3.4）已升级为可实施规格 **§3.5**——13 个成员的完整签名、观测层只读快照、能力查询形状、bfdiag 耦合处置、8 步迁移顺序与回滚点 |
| **A2 工作量** | "`LagunaBackend` 有 50+ 公开方法" | 🟢 **被高估**：50 是**方法总数**（含私有），公开只有 **24 个 + 2 个 property**；而 `ServerEngine` 实际只调用其中 **13 个**。协议倒推的输入是这 13 个 |
| **N8**（本次定稿中发现） | — | 🔴 **`--session-affinity` 100% 失效**：`engine.py:971` 调用的 `mtp_prefill_warm_continue` 只存在于 `oracle/qwen36_vllm/`，`LagunaBackend` 没有且无 `__getattr__` 转发。异常被 `try/except` 吞掉 → 永远静默回退冷 prefill，`session_warm_continuations` 恒为 0，测试零覆盖。详见 `architecture.md` §3.5.6 |

### 0.1 roadmap 里没有、但必须进 P0 的一条

`main` 最新三个提交（`2c06355` / `d9e52ce` / `619a09d`）是**同一天内三个只有真机活服务器才能暴露的 bug**：

- `/metrics` 冷启动窗口 500（`get_stats()` 短路返回 2 键，渲染读 6 键）；
- `/metrics` 忙时 500（`slot_kv_len` 是 list，端点当 mapping 读，**同一端点当天第二次挂**）；
- 引擎丢唤醒 —— 会话第二轮必挂（对话客户端每次都恰好落在那 152 ms 窗口）；
- 附带：Laguna 的 chat template **确实注入 `<think>`**，源码注释写反了，导致 `</think>` 被当正文首字符送出。

**1100 条测试全部看不见它们**，因为它们依赖的是部署事实而非代码逻辑。→ 新增 **P0-B C-LIVE**。

**而且这不是三个孤立的 bug**：两次 `/metrics` 500 的共同根因是**观测层直接读执行层内部**——
`server/app.py` 至今仍在读 `runner._prefix_cache_kv_len` / `runner._prefix_cache_tokens`
两个**私有属性**，加上 `slot_kv_len` 的 list 形状假设。这正是 A2 协议缺位的症状。
→ **A2 不只是重构，它关掉的是一个已经咬了两次的故障类**（见 P0-E/A2）。

---

## 1. P0 的定义与退出条件

**P0 = 在动核心执行路径之前必须就位的护栏 + 那条动核心执行路径的主线本身。**

P0 完成 = 下面五条同时成立：

1. 仓库无已合并分支残留、无待决 untracked 文件；
2. 存在一道**活服务器冒烟门禁**，且能在三个已知 bug 的父提交上变红；
3. **D3（GPU CI 形态）与 D6（主线 checkpoint）已拍板**；
4. Track A 设计从草案升级到**可实施**（有签名、有迁移顺序、有回滚点）；
5. Laguna 跑在新抽象上，**贪心 bit-exact + 性能不低于基线 3% + 接受率不回归**。

**关键路径**（唯一一条长串行链，决定整体交付日期）：

```
P0-C 拍板 D3 ─┐
              ├─→ P0-D 设计定稿 ─→ A1 ─→ A2 ─→ A3 ─→ A4 ─→ A5 ─→ A6 验收
P0-B C-LIVE ──┘                                                      │
P0-A 卫生（不卡任何人，可随时插）                                     ↓
P0-C 拍板 D6 ────────────────────────────────────→ Track B 才能起步
```

**体量**：P0-A/B/C 约 1 周；P0-D 约 1 周；A1–A6 约 1.5 个月。合计 ≈ M1 剩余 + M2。

---

## 2. P0-A · 仓库卫生收尾（约 0.5 天，不卡任何人）

比 roadmap 预估的小得多——`.gitignore` 已经覆盖了绝大部分，剩下的是分支残留。

- [x] **A-1** 删已并入 main 且**无活动状态**的分支/worktree（2026-08-01 执行）：
  - ✅ `fix/engine-lost-wakeup`（`d9e52ce`，无 worktree）
  - ✅ `fix/live-thinking-and-metrics`（`2c06355`，worktree `…-fix` 干净）→ worktree + 分支已删
  - ⏸️ `fix/metrics-busy-500`、`worktree-laguna-mid-conversation-system` —— **当时有未提交改动，未动**；worktree 侧后由用户自行处理
- [ ] **A-2** 处置 `perf/repro-2ce5-baseline-20260730`（`060fabb`，**未并入**，含 1 个性能提交 "Cache loop-invariant values in generate_verify_only decode loop"，作者自评"perf 影响在噪声内 ~0.5%"）→ 归入 Track F/F7 评估，或明确废弃
- [ ] **A-3** 处置 untracked 文件 `benchmarks/repro_prefix_cache_slowdown.py`：按 `benchmarks/README.md` 规约决定收编、转 `bf exec`、还是删（在主工作区，留给持有者操作）
- [ ] **A-4**（可选）主工作区磁盘杂物：9 个 `*.log`、`build/`、`evalplus_results/` —— **已被 gitignore，不影响仓库**，纯占盘
- [ ] **A-5** N6 结案规则落文进 `AGENTS.md` 或 `diagnostics-guide.md`：**若再复现，直接贴带线程列表的失败输出，不许重跑**；若 M2 结束前零复现，从 roadmap 移除该条目

**验收**：`git branch` 只剩 `main` + 活跃工作分支；`git status` 干净。

---

## 3. P0-B · C-LIVE 活服务器冒烟门禁（约 3 天）

依据 §0.1。目标不是覆盖率，是**把"只有部署事实才能证伪的假设"变成自动化断言**。

### 3.1 现成基础比预期好——这是"收编"不是"新建"

- `scripts/blackwellm_ctl.sh`（300 行）已有 `start / stop / restart / status / logs / config / relay`
- `tests/` 已有三个端到端脚本：`test_api_compat.py`、`test_real_world.py`、`test_e2e_256k_longctx.py`
- 但它们**被 `tests/conftest.py:20-24` 的 `collect_ignore` 排除在收集之外**，因为是脏脚本：
  module-level 执行真实网络 I/O 和后台线程，`test_real_world.py` 甚至在 module scope 直接 `sys.exit(...)`

→ **正确动作是把这三个脚本改造成可被门禁调用的形态，而不是从零写第四个。**
排除它们是对的（不该污染 `pytest -q`），但排除之后它们就没人跑了，等于三份已写好的端到端覆盖被闲置。

### 3.2 实施

- [ ] **B-1** 把三个 e2e 脚本的副作用收进函数（消灭 module-level I/O 与 `sys.exit`），保持 `collect_ignore` 不变，改为通过显式 marker 或独立入口调用
- [ ] **B-2** 冒烟脚本主体：起真服务 → 断言下列各项 → 干净关停
  - [ ] `/metrics` 在**冷启动窗口**（无任何请求完成时）返回 200 且六个聚合键齐全
  - [ ] `/metrics` 在**长请求进行中**（`engine.active` 非空）返回 200 —— 当天第二次挂的形状
  - [ ] **背靠背两轮对话**，第二轮在前一轮响应结束后 **< 200 ms** 发出 —— 丢唤醒 bug 的精确窗口
  - [ ] OpenAI + Anthropic 双协议 × 流式/非流式 × 工具调用各一条
  - [ ] thinking 契约：在**模板注入 `<think>` 的真实模型**上断言 `content` 首字符不是 `</think>`
  - [ ] `/v1/completions` 逐字返回（不套 chat template）—— 2026-07-27 事故的守门断言
- [ ] **B-3** 挂进 `blackwellm_ctl.sh` 与 Makefile，成为**每次动 `server/` 或 `runtime/backends/` 后的必跑项**
- [ ] **B-4** [待办·开发执行] 首次真机运行，结果记进 bfdiag run record

**验收（硬标准）**：三个已修 bug 中的每一个，在其修复提交的**父提交**上跑该脚本都能变红。
达不到这条，说明门禁没覆盖到真正的失败面，不算完成。

---

## 4. P0-C · 两个卡住后续轨道的拍板（需要人决定）

- [ ] **C-1 D3 · GPU CI 形态**（RK7）：(a) 自托管 runner / (b) 本地 pre-push 门禁 + 人工签核 / (c) 只在里程碑人工全量跑
  → **卡 C4 位精确门禁 → 卡 A6 的"零回归"验收**。不定，Track A 就没有可执行的验收标准，A1–A5 写完也无法宣布完成
- [ ] **C-2 D6 · Qwen3.6 主线 checkpoint**：带 MTP 的社区文本版 vs 官方 NVFP4 版 → **卡 B0 起步**
- [ ] （不卡任何轨道，可延后）**D4 重命名时机**、**D5 `oracle/qwen36_vllm/` 处置**

---

## 5. P0-D · Track A 设计：草案 → 可实施 ✅ 已完成（2026-08-01）

**产出：[`architecture.md` §3.5 实施规格](architecture.md)**。全部零 GPU 完成。

- [x] **D-1 `ModelBackend` 协议的完整签名** → §3.5.1：13 个成员逐条签名 + 调用频次；`LagunaSlotState`（frozen，文档已自称"只读服务端视图"）可直接提升为协议的 `SlotState`；`ChunkedPrefillState` 保持不透明
- [x] **D-2 观测层只读快照** → §3.5.2：定位到 `app.py:673`、`:676` 两处**私有属性直读** + `:1182-1221` 的 list 形状假设；定稿 `snapshot() -> BackendSnapshot`（三个 frozen 类型），观测层只拿值不拿引用，切换完成后删掉 `_slot_kv_len()` 容错helper
- [x] **D-3 能力查询形状** → §3.5.3：定为 `capabilities` property 返回 frozen `BackendCapabilities`（五个布尔）。**否决字符串式 `supports("...")`**：拼错不报错、无法静态检查。可直接序列化进 bfdiag run record 与 `/metrics`，同时是 C2 分级降级的判定输入
- [x] **D-4 bfdiag 耦合处置** → §3.5.4：4 个模块直接 import 执行层，**其中 `workloads.py` 摸的是私有成员** `_physical_slot` / `_ring_prefix_reuse_is_safe`；逐项给了处置
- [x] **D-5 迁移顺序与回滚点** → §3.5.5：**8 步，A3 从第 3 位移到第 7 位**——它爆炸半径最大（touch 前缀缓存）且在 Track B 递归状态到来前**没有真实消费者**，先做等于承担最大风险换零收益。每步标了行为变更、门禁、回滚点、是否需要 GPU
- [x] **D-6 成文进 `architecture.md`**（M1 交付物）→ 新增 §3.5.1–3.5.6

**定稿过程中发现 N8**（§3.5.6）：`--session-affinity` 100% 失效，见 §0 表格与 §6 下方。

---

## 6. P0-E · Track A 实施（M1→M2，约 1.5 月，主线）

**执行顺序按 [`architecture.md` §3.5.5](architecture.md) 定稿的 8 步**，不是 roadmap 的 A1→A6 编号顺序：
按爆炸半径从小到大，零行为变更的步骤排在前面。**前 4 步完全不需要 GPU。**

| # | 步骤 | 行为变更 | 门禁 | GPU |
|---|---|---|---|---|
| 1 | [ ] **A2-shadow** 定义 Protocol（D-1 的 13 个成员 + D-3 能力查询），静态+运行时断言 `LagunaBackend` 满足它，**不改调用点** | 无 | 类型检查 + 一致性单测 | ❌ |
| 2 | [ ] **A2-观测** `snapshot()` 落地，`app.py:673/676` 两处私有直读改走它，删 `_slot_kv_len()` | 无（同值） | 单测 + C-LIVE metrics 两条 | ❌ 写 |
| 3 | [ ] **A1 ModelSpec（影子）** 按 §3.2-A 九个字段族解析 `config.json`，断言结果与当前硬编码值逐字段相等，暂不驱动任何东西。含 RK8：显式拒绝带 vision tower 的权重 | 无 | 影子一致性单测 | ❌ |
| 4 | [ ] **A5 Registry（影子）** 路径 → `(spec, backend, loader, 投机策略)`，断言等于今天的硬编码选择 | 无 | 影子一致性单测 | ❌ |
| 5 | [ ] **切换** Registry 成为唯一真相源；删 `engine.py:188` `MODEL`、`:190` `BACKEND`、`app.py:81` `SERVER_MODEL_BACKEND` | **有** | 贪心 bit-exact + C-LIVE | ✅ |
| 6 | [ ] **A4 加载器 adapter** 拆出 compressed-tensors；公共部分（分片流式读取、参数全覆盖断言、KV scale post-load）不变 | 有（同权重） | 逐张量校验和相等 + bit-exact | ✅ |
| 7 | [ ] **A3 SlotResourceManager** `block_pool` 升级为槽位资源管理器，两类资源联动驱逐；清掉 S4 的 GDN 残迹（`evict_gdn_checkpoint`）。递归状态部分随 Track B 落地 | **有，半径最大** | bit-exact + 接受率 + 前缀命中率不回归 + C-LIVE | ✅ |
| 8 | [ ] **A6 验收**（硬门禁，**依赖 C-1 拍板**） | — | 见下 | ✅ |

**A6 验收四条**（[待办·开发执行]，均需真机）：
- [ ] 贪心输出 **bit-exact**
- [ ] 性能不低于基线 3%（fox-64K 353–368 / fox-4K 353–357 / galaxy-4K 395–401 / code-4K 341–359 tok/s）
- [ ] 接受率不回归（96.3–100%）
- [ ] C-LIVE 冒烟通过；比数前先 `bf diff` 判可比性（2026-07-27 教训）

**风险 RK3**：动核心执行路径，Laguna 是唯一生产模型。**顺序不能颠倒：先 P0-B 后第 5 步。**

### 6.1 N8 · `--session-affinity` 静默失效（需拍板）

`engine.py:971` 调 `mtp_prefill_warm_continue`，`LagunaBackend` 没有该方法（只在 `oracle/qwen36_vllm/`），
异常被 `try/except` 吞掉 → 每次都静默回退冷 prefill。默认关闭，但 `--session-affinity` 是文档化的 CLI 开关。
完整证据见 [`architecture.md` §3.5.6](architecture.md)。

- [ ] **N8 拍板**：(a) 为 Laguna 实现 warm-continue ／ (b) 删除该 flag 与相关调度路径 ／ (c) 启动期拒绝该 flag，把静默降级变成显式失败
- **倾向 (c) 作为立即动作**（零 GPU、可当天落地），(a)/(b) 待 Track A 的能力查询就位后重新评估
- [ ] 无论选哪个：补上 `warm_continue` / `session_warm` 的测试覆盖（当前**零覆盖**）

---

## 7. P0 之后：其余轨道（优先级与顺序）

```
P1  Track B Qwen3.6-27B  B0→B1→B2→B3   ←── M2→M4 主线，串行，2.5–3 月
P1  Track D 易用性 D1→D6                ←── M2→M5，与 B 全程并行
P1  Track C 稳定性 C0R→C6               ←── M1→M6 贯穿，不设独立里程碑
P1  Track G 25B-A3B                     ←── M4→M5，前置=拿到 config
P2  Track E 兼容性 E2→E1→E3→E4          ←── M3→M6，与 B/D 并行
P2  Track F 性能                        ←── 机会主义，永不抢占 C/D
P2  Track H 发布 0.2.0                  ←── M5→M6
```

### 7.1 P1 · Track B · Qwen3.6-27B（M2→M4，串行）

**单点最大风险 RK1**：GDN 占 48/64 层，引入**第二类缓存**（conv + ssm state），
要和固定槽位、前缀缓存、CUDA Graph、投机解码四套机制全部对接。
`docs/archive/2026-07-30-architecture-two-tenant.md` §6.2 是可复用先验。

**B0 事实基线**（M2，2 周，全部 [待办·开发执行]）
- [ ] B0-1 变体清点选型，定主线 checkpoint（**依赖 C-2 拍板**）
- [ ] B0-2 modelopt NVFP4 的 tensor 命名与 scale 语义**逐项确认，不猜**
- [ ] B0-3 sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 / page_size ∈ {64,128} / fp8 KV` 下的正确性与吞吐
- [ ] B0-4 GDN 方案三选一：① FLA v0.5.2 `gated_delta_rule` ② 从 `oracle/qwen36_vllm/` 移植 ③ 自研。**建议先 ① 拿正确性，profiling 说话后再决定 ③**
- [ ] B0-5 GDN 递归状态更新是否 **CUDA Graph capture-safe**（决定 B2 可行性）
- [ ] B0-6 mrope-interleaved 在纯文本下能否退化为标准 1D RoPE
- [ ] B0-7 容量测算：64 层 / 256K / 96 GB 的 KV + 递归状态显存账 → context × 并发可行域

**B1 正确性优先**（M2→M3，1 月）：eager、batch=1、无图、无投机、无前缀缓存
- [ ] GDN 层（conv1d state + gated delta rule + 输出门）· Full attention（sparkinfer paged）· 稠密 SwiGLU（NVFP4）· RoPE partial 0.25 + mrope · modelopt 加载 · 注意力输出门控
- **门禁**：与 HF transformers 贪心**逐 token 对齐**（≥ 3 工作负载 × 512 token）；逐层 logits 余弦相似度进 bfdiag

**B2 服务化**（M3，1 月）
- [ ] 固定槽位 + 连续批处理 · 递归状态纳入槽位生命周期 · CUDA Graph（依赖 B0-5）· 前缀缓存联动驱逐（A3 的第一个真实用户）· 并发 ≥ 2
- **门禁**：双协议回归全绿 + **C-LIVE 通过** + 与 B1 eager 贪心 bit-exact

**B3 性能与投机**（M4，1 月）
- [ ] MTP draft/verify 含 **GDN 递归状态推测回滚**（本轨道最难）· GDN kernel 调优 · FP8 KV · 128K/256K 容量与吞吐
- **门禁**：接受率与吞吐进 bfdiag 基线；与上游框架同 prompt 同参数 A/B

### 7.2 P1 · Track D · 易用性（M2→M5，与 B 并行）

- [ ] **D1 单命令启动** `blackwellm serve <model-path-or-id>`，自动推导槽位与块数（当前 `[project.scripts]` 只有 `bf`）
- [ ] **D2 显存规划器** → 消灭 S7 的 `QSR_SERVER_CAPACITY`/`NUM_SLOTS`/`BLOCKS_PER_SLOT`/`PRODUCTION` 四变量耦合陷阱
- [ ] **D3-impl 启动前置检查**：`runtime/preflight.py` 已有 738 行九项校验，扩到 SM120 检测 / 显存 / CUDA driver / sparkinfer 版本 / checkpoint 完整性 / 架构是否受支持，全部在加载权重之前
- [ ] **D4 配置文件**：YAML 取代十几个环境变量（环境变量降级为覆盖手段）
- [ ] **D5 命名统一**：`QSR_` → `BWLLM_`（**当前 370 处引用**），带一个版本兼容期
- [ ] **D6-docs 文档三件套**：安装部署 / 配置调参 / 故障排查

**排序**：D1+D2+D3-impl 在 M3 一批交付（共同构成"一条命令起服务"），D4/D5/D6 在 M5 收口。

### 7.3 P1 · Track C · 稳定性（贯穿）

- [ ] **C0R** C0 审计残余：① `bfdiag/` 的 **code_ref 行号系统性漂移**（约定改为引用符号名；从命中率最高的 5 个文件起：`checkpoint/state.py`、`daemon/session.py`、`invariants/checks.py`、`shapes/attention.py`、`determinism.py`）② `cold_capacity.py` / `det_cli.py` 未逐行核实 ③ 两条无调用点的不变量（`aux_hidden_alignment`、`cg_replay_slot_consistency`）接线或明确标记
- [ ] **C1 故障面清单**：显存不足 / 槽位卡死 / 图捕获失败 / kernel JIT 失败 / 长请求超时 / 客户端断连 / tokenizer 边界 / 非法采样参数 / 并发抢占 —— 每项要有检测点、指标、日志、用户可见错误、恢复动作
- [ ] **C2 分级降级**：CUDA Graph → eager；投机 → 非投机；前缀缓存命中 → 冷 prefill，每级出指标（**接口形状来自 P0-D 的 D-3 能力查询**）
- [ ] **C3 看门狗覆盖**：补覆盖测试 + 故障注入
- [ ] **C4 确定性与可复现**：补 bit-exactness 回归门禁并纳入 CI（**依赖 C-1 拍板**；是 A6 验收的实现载体）
- [ ] **C5 长稳测试**：24h soak [待办·开发执行]
- [ ] **C6 崩溃可诊断**：进程级异常留下 bfdiag run record

**插入时机**：C0R 与 C-LIVE 一起在 M1；C1/C2 随 Track A 落地；C3/C4 在 M3；C5/C6 在 M6 发布门禁前。

### 7.4 P1 · Track G · Qwen3.6-25B-A3B（M4→M5）

前置：**先拿到 `config.json`**（本地无 checkpoint，架构参数全未知，RK4 —— 在此之前时间是占位不是承诺）。

- [ ] **G0** [待办·开发执行] 拉 config：专家数 / top-k / 是否 hybrid / 是否带 MTP
- [ ] **G1** 若 Hybrid GDN + MoE：GDN 复用 Track B；MoE 走 sparkinfer `moe.fused_moe`
- [ ] **G2 router kernel 泛化**（S6）：`runtime/laguna_router.py:20-21` 的 `EXPERTS = 256` / `TOP_K = 10` 是模块级常量，`:184-185`、`:205-206` 直接依赖 → 定泛化还是特化
- [ ] **G3** [待办·开发执行] `moe.fused_moe` 在非 256/top-10 形状下的可用性与性能（SM120 上 MoE 已测定带宽饱和）

### 7.5 P2 · Track E · 兼容性（M3→M6）

- [ ] **E2 采样 + 投机共存**（消灭 S8）：`temperature > 0` 当前直接退化成无投机自回归，需要拒绝采样 / typical acceptance。**这是最明显的功能缺口，排在 E1 之前**
- [ ] **E1-N1 结构化输出真正实现**：`GrammarState.apply_mask` / `apply_mask_batch` 接进采样环。**优先级已从"P0 事故"降为普通功能**——`b2d73cb` 后是显式拒绝而非静默失败
- [ ] **E1-剩余**：`n>1`、usage token 统计准确性
- [ ] **E3 客户端验证矩阵** [待办·开发执行]：openai-python / anthropic-sdk / Claude Code / Cline / Roo / OpenWebUI / LiteLLM，结果做成兼容性表
- [ ] **E4 reasoning 正确暴露**：OpenAI `reasoning_content` 已接（`c86858a`）。**Anthropic 侧维持非标准事件，除非拿到合法签名来源**——`f13fd4a` 的生产事故明确禁止无签名重发 thinking block

### 7.6 P2 · Track F · 性能（机会主义）

**纪律**：只在有明确 roofline 依据、且不损害 Track C/D 的前提下做；必走 `bf diff` + 接受率/质量门禁。

- [ ] **F1** TURBO_ATTN 质量回归修复（per-head descale / Hadamard 旋转 / 自适应切换）：收益 +6%，但 code 接受率 97.8% → 58.6%，当前默认关闭
- [ ] **F2** FA4 技法用于 prefill / extend（TMA、persistent scheduler、FP8 softmax）
- [ ] **F3** FP8 attention `num_stages ≥ 2`（SMEM 36 KB « 99 KB）
- [ ] **F4** sparkinfer 剩余 9 处 gate（`7a1d69d` 只放宽 13 处中的 4 处，其余是**刻意留下**的 scope 限制）
  - ⚠️ **硬风险**：`planner.py` 的 grid occupancy 预算常量按 `num_kv_heads=4` 推导，用到 8 **需重新推导**，不是放宽谓词就够
  - ⚠️ sparkinfer 源码改动写清楚交给该团队，不直接改
- [ ] **F5** MoE 输出中心并行（Warp Decode 类），2–4 周，长期备选
- [ ] **F6** GDN kernel 自研（依赖 B0/B2 profiling）
- [ ] **F7** 评估 `perf/repro-2ce5-baseline-20260730` 上未合并的 `060fabb`（见 P0-A/A-2）

### 7.7 P2 · Track H · 发布 0.2.0（M5→M6）

- [ ] **H1 发布门禁**：CI 绿（两个 job）· C-LIVE 通过 · 24h 长稳（C5）· 两个模型系列质量回归 · 文档三件套 · **依赖可从公开源安装（即 sparkinfer 上游化，RK2 未解，当前钉 `origin/master @ 0844a4f`）**
- [ ] **H2 素材纪律**：只发实测数字，标注硬件/配置/复现命令；不做 apples-to-oranges 对比
- [ ] **H3** `0.2.0` = "多模型 + 生产可用"的第一个公开版本

---

## 8. 里程碑对齐

| 里程碑 | 时间 | 对应条目 | 说明 |
|---|---|---|---|
| **M1** | 2026-08 | **P0-A + P0-B + P0-C + P0-D**，A1/A2 起步 | Track 0 实际已近清零，M1 富余产能应投入 C-LIVE 与设计定稿 |
| **M2** | 2026-09 | A3–A6 落地 + B0 事实基线 + B1 起步 + D1/D2 起步 | Laguna 零回归是硬门禁 |
| **M3** | 2026-10 | B1 验收 + B2 服务化 + D1/D2/D3-impl 交付 + C1/C2 | 一条命令能起服务 |
| **M4** | 2026-11 | B3 性能与 MTP + G0/G1 | 25B-A3B 正确性对齐 |
| **M5** | 2026-12 | G2/G3 + D4/D5/D6 收口 + E3 客户端矩阵 | 三个模型系列同一套配置流程 |
| **M6** | 2027-01 | C5/C6 + Track H 全项 | 24h soak + 发布 checklist |

---

## 9. 阻塞与依赖速查

| 被卡的事 | 卡它的 | 状态 |
|---|---|---|
| A6 零回归验收 | **P0-C/C-1 GPU CI 拍板** → C4 位精确门禁 | 🔴 未拍板 |
| B0 起步 | **P0-C/C-2 主线 checkpoint 拍板** | 🔴 未拍板 |
| ~~A1–A6 开工~~ | ~~P0-D 设计升级到可实施~~ | 🟢 **已解锁**：规格在 `architecture.md` §3.5，第 1–4 步可立即开工且不需要 GPU |
| P0-E 第 5–8 步 | **P0-B C-LIVE** + GPU | 🔴 C-LIVE 未开始 |
| N8 处置 | 拍板 (a)/(b)/(c) | 🔴 未拍板，倾向 (c) |
| Track B 全部 | Track A（A1–A5）完成 | 🔴 未开始 |
| B2 CUDA Graph | B0-5（GDN 是否 capture-safe） | 🔴 未验证 |
| C2 分级降级的接口 | P0-D/D-3 能力查询形状 | 🔴 未定 |
| Track G 排期 | G0 拿到 config | 🔴 本地无 checkpoint |
| H1 发布 | sparkinfer 上游化（RK2） | 🔴 仍钉私有 fork |

---

## 10. 不做清单（不再重复讨论）

多卡 TP/PP/EP · 多机 · SM120 以外架构 · 视觉/多模态输入 · 训练/微调/LoRA 热加载 ·
AWQ/GPTQ/INT4 等其他量化 · 通用 HF 架构自动支持 · 复活 `oracle/qwen36_vllm/`。理由见 [`roadmap.md`](roadmap.md) §3。

---

## 11. 配套文档

- [`roadmap.md`](roadmap.md) — 目标、理由、风险登记、待拍板事项（**权威**）
- [`architecture.md`](architecture.md) — 现状架构 + **目标架构与五个关键抽象（Track A 设计草案，§3）**
- [`model-support.md`](model-support.md) — 模型支持矩阵 + 接入新模型的操作指南
- [`diagnostics-guide.md`](diagnostics-guide.md) — bfdiag 使用指南（必读）
- [`../notes/2026-08-01-bfdiag-assertion-audit.md`](../notes/2026-08-01-bfdiag-assertion-audit.md) — C0 审计全文
- [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md) — fork 差异与 9 处未放宽 gate 的安全性分析
