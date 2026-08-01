# BlackweLLM 实施计划清单（2026-08-01 编制 · P0 已按最新状态重排）

> 基线：`main @ 619a09d`。上游文档 [`roadmap.md`](roadmap.md) 的基线是 `ce21eb5`，
> 其后 main 已推进 **49 个提交**，路线图里若干"未解决"条目实际已经关闭。
> 本文档 = roadmap 的**执行视图**：状态已按当前代码核实，条目按优先级排成一条可执行的序列。
>
> roadmap.md 仍是**目标与理由**的权威来源；本文档只回答"下一个动作是什么、谁卡着谁"。
> 标 **[待办·开发执行]** 的条目需要 GPU / 真机 / 下载权重，编制者不代跑。
>
> **2026-08-01 二次修订**：基线推进到 `main @ 6acc4ba`（含 `235f51e` JIT 编译修复）。
> 消化 [`investigation-queue.md`](investigation-queue.md) §A/§D 的调研结论；拍板 §4 的
> C-1（GPU CI 形态）、C-2（Qwen3.6 主线 checkpoint）、§6.1 的 N8（`--session-affinity`）
> 三项，均已解锁下游条目。新增 Track F 的 F1/F2（见 §7.6）与 Track C 的冷启动路径审计
> （见 §7.3）。

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
| **N8**（本次定稿中发现） | — | 🟡 **`--session-affinity` 100% 失效，处置已拍板**：`engine.py:971` 调用的 `mtp_prefill_warm_continue` 只存在于 `oracle/qwen36_vllm/`，`LagunaBackend` 没有且无 `__getattr__` 转发。异常被 `try/except` 吞掉 → 永远静默回退冷 prefill，`session_warm_continuations` 恒为 0，测试零覆盖。详见 `architecture.md` §3.5.6。**2026-08-01 已拍板 (c)**：启动期直接拒绝该 flag，实现见 §6.1，尚待落地（零 GPU） |

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
3. ~~**D3（GPU CI 形态）与 D6（主线 checkpoint）已拍板**~~ —— ✅ **2026-08-01 两条均已拍板**（§4），
   D3 的落地机制（`make gate-local`）与 D6 衍生的 vision 张量过滤任务尚未开工；
4. Track A 设计从草案升级到**可实施**（有签名、有迁移顺序、有回滚点）；
5. Laguna 跑在新抽象上，**贪心 bit-exact + 性能不低于基线 3% + 接受率不回归**。

**关键路径**（唯一一条长串行链，决定整体交付日期）：

```
✅P0-C 拍板 D3 ─┐
              ├─→ P0-D 设计定稿 ─→ A1 ─→ A2 ─→ A3 ─→ A4 ─→ A5 ─→ A6 验收
P0-B C-LIVE ──┘                                                      │
P0-A 卫生（不卡任何人，可随时插）                                     ↓
✅P0-C 拍板 D6 ───────────────────────────────────→ Track B 才能起步
```

**体量**：P0-A/B/C 约 1 周；P0-D 约 1 周；A1–A6 约 1.5 个月。合计 ≈ M1 剩余 + M2。

---

## 2. P0-A · 仓库卫生收尾（约 0.5 天，不卡任何人）

比 roadmap 预估的小得多——`.gitignore` 已经覆盖了绝大部分，剩下的是分支残留。

- [x] **A-1** 删已并入 main 且**无活动状态**的分支/worktree（2026-08-01 执行）：
  - ✅ `fix/engine-lost-wakeup`（`d9e52ce`，无 worktree）
  - ✅ `fix/live-thinking-and-metrics`（`2c06355`，worktree `…-fix` 干净）→ worktree + 分支已删
  - ⏸️ `fix/metrics-busy-500`、`worktree-laguna-mid-conversation-system` —— **当时有未提交改动，未动**；worktree 侧后由用户自行处理
- [ ] **A-2** 处置 `perf/repro-2ce5-baseline-20260730`（`060fabb`，**未并入**，含 1 个性能提交 "Cache loop-invariant values in generate_verify_only decode loop"，作者自评"perf 影响在噪声内 ~0.5%"）→ 归入 Track F/F9 评估，或明确废弃
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

- [x] **C-1 D3 · GPU CI 形态**（RK7）—— ✅ **2026-08-01 已拍板：(b) 本地 pre-push 门禁 + 人工签核**。
  理由：这台机器只有一块 GPU（RK5），自托管 runner（选项 a）会和开发实时抢卡，而"一次验证以分钟计"
  正是本项目全部效率问题的根源——引入一个还要排队等 runner 的额外瓶颈是在加剧病灶，不是治它。
  选项 (c)（只在里程碑人工全量跑）门禁太松，位精确回归会在里程碑之间的一个月里悄悄漂移而没人发现。
  **具体机制**（待落地，见 §7.3 Track C/C4）：一个 `make gate-local`（或 pre-push hook）跑 C-LIVE 冒烟 +
  贪心 bit-exact 探针，touch `runtime/backends/` 或 `server/` 的分支推送前必跑；PR 模板加一条人工签核勾选项
  （"我已在本机跑过 gate-local 并附结果"），没有勾选不合并。这解锁了 **C4 位精确门禁 → A6 验收** 的执行路径。
- [x] **C-2 D6 · Qwen3.6 主线 checkpoint** —— ✅ **2026-08-01 已拍板：官方 `nvidia/Qwen3.6-27B-NVFP4`**
  - ⚠️ **roadmap 对这个选择的描述前提不成立**（2026-08-01 读权重索引核实）：它写的是"社区版能做投机 / 官方版需另找 MTP 层"，
    实际**两份都带 MTP**——`nvidia/Qwen3.6-27B-NVFP4` 有 15 个 `mtp.*` 张量（`mtp.fc.weight`、`mtp.layers.0.*`），
    `sakamakismile/...-Text-NVFP4-MTP` 同样 15 个。
  - 真正的取舍是：**官方 provenance + 需排除 333 个 vision 张量**（`language_model_only=False`，带 `vision_config`）
    vs **社区量化 + 天生文本版**（`language_model_only=True`，0 个 vision 张量，单文件）。
  - **理由**：排除 333 个 vision 张量是一次性的机械过滤工作（按 tensor 名前缀跳过 `vision.*`），
    不是架构级的工程风险；而衍生模型（微调版、下一代 Qwen）迟早都会带 vision tower，这个过滤器不管选哪份
    checkpoint 都要写一次，官方选择不省这个工。反过来，provenance 是不可逆的：发布时"官方 NVFP4"比
    "社区量化"站得住，而 B1 阶段一旦跑通逐 token 对齐，两份 checkpoint 换着测的成本很低。
    **社区文本版 `sakamakismile/...-Text-NVFP4-MTP` 留作交叉验证 baseline**，不是弃用。
  - 附带事实：四个本地 Qwen3.6 变体里 **unsloth 那份是 compressed-tensors，不是 modelopt**——
    量化格式必须逐 checkpoint 读，不能按架构推断。
  - ⚠️ **这个决定对 A1 提出一个新要求**（2026-08-01 定稿时发现，roadmap RK8 需要同步更新）：
    `architecture.md` §3.2-A 现在的措辞是"带 vision tower 的 checkpoint 不受支持，直接拒绝"（RK8）；
    但官方 checkpoint **恰好带 vision tower**，我们要的是"接受 checkpoint，但只加载语言模型部分、跳过
    `vision.*` 张量"，不是整体拒绝。`validate_text_only` 的语义要从"config 里出现 `vision_config` 就报错"
    改成"允许 `vision_config` 存在，但要求 loader 处于 `language_model_only=True` 模式并断言零 vision 张量
    被加载"。这条留给 A1 落地时处理（不改 `architecture.md`，此处只记录设计要求），已加进 B0-1。
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

> 进度：**第 1–4 步全部完成**（`4ed5a7b`、`f24f5ad`、`a1287c5`），**全程零 GPU**。
> torch 环境 1151 passed（起点 1100），无 torch 环境 852 passed / 93 skipped，ruff 两关全过。
> **第 5 步（Registry 成为唯一真相源）是第一个需要真机的步骤**——它改行为，门禁是贪心 bit-exact。
> 在拿到 GPU 之前，剩余的零 GPU 工作是 P0-B 的 C-LIVE 脚本编写（B-1/B-2/B-3）。

| # | 步骤 | 行为变更 | 门禁 | GPU |
|---|---|---|---|---|
| 1 | [x] **A2-shadow** ✅ `4ed5a7b`：`runtime/backends/protocol.py`（torch-free）+ `BackendCapabilities` + `check_conformance`；`LagunaBackend` 加 `capabilities`。**零调用点改动**。门禁经两种注入漂移验证会变红 | 无 | 类型检查 + 一致性单测 | ❌ |
| 2 | [x] **A2-观测** ✅ `f24f5ad`：`snapshot()` 落地，`/metrics` 与 `/debug/stats` 改走契约，删 `_slot_kv_len()`；补上 `/metrics` **此前完全没有的路由级测试**（冷启动 + 忙时两态） | 无（同值） | 单测 ✅ + C-LIVE metrics 两条（待 GPU） | ❌ 写 |
| 3 | [x] **A1 ModelSpec（影子）** ✅ `a1287c5`：`runtime/architecture.py`（torch-free）解析层类型序列/FFN/RoPE/量化/MTP/**每层缓存需求**，含 RK8 的 `validate_text_only`。影子断言对真实 checkpoint 成立 | 无 | 影子一致性单测 ✅ | ❌ |
| 4 | [x] **A5 Registry（影子）** ✅ `a1287c5`：`runtime/model_registry.py`，路径 → `(spec, backend, loader, 投机策略)`，对 Laguna 解析出 `laguna / compressed_tensors / dflash` = 今天的硬编码选择 | 无 | 影子一致性单测 ✅ | ❌ |
| 5 | [ ] **切换** Registry 成为唯一真相源；删 `engine.py:188` `MODEL`、`:190` `BACKEND`、`app.py:81` `SERVER_MODEL_BACKEND` | **有** | 贪心 bit-exact + C-LIVE | ✅ |
| 6 | [ ] **A4 加载器 adapter** 拆出 compressed-tensors；公共部分（分片流式读取、参数全覆盖断言、KV scale post-load）不变 | 有（同权重） | 逐张量校验和相等 + bit-exact | ✅ |
| 7 | [ ] **A3 协调者**（**不是**统一分配器，2026-08-01 更正）：两个独立分配器 + 协调者持有不变量；前缀匹配返回 `(kv_hit, state_hit)`；逐资源驱逐预算与账目；清掉 S4 的 GDN 残迹。**动工前必读** [`hybrid-cache-prior-art`](../notes/2026-08-01-hybrid-cache-prior-art.md)（vLLM + SGLang 真源码先例，含 6 条会被踩中的坑） | **有，半径最大** | bit-exact + 接受率 + 前缀命中率不回归 + C-LIVE | ✅ |
| 8 | [ ] **A6 验收**（硬门禁，**依赖 C-1 拍板**） | — | 见下 | ✅ |

**A6 验收四条**（[待办·开发执行]，均需真机）：
- [ ] 贪心输出 **bit-exact**
- [ ] 性能不低于基线 3%（fox-64K 353–368 / fox-4K 353–357 / galaxy-4K 395–401 / code-4K 341–359 tok/s）
- [ ] 接受率不回归（96.3–100%）
- [ ] C-LIVE 冒烟通过；比数前先 `bf diff` 判可比性（2026-07-27 教训）

**风险 RK3**：动核心执行路径，Laguna 是唯一生产模型。**顺序不能颠倒：先 P0-B 后第 5 步。**

### 6.1 N8 · `--session-affinity` 静默失效 —— ✅ 已拍板 (c)，待落地

`engine.py:971` 调 `mtp_prefill_warm_continue`，`LagunaBackend` 没有该方法（只在 `oracle/qwen36_vllm/`），
异常被 `try/except` 吞掉 → 每次都静默回退冷 prefill。默认关闭，但 `--session-affinity` 是文档化的 CLI 开关。
完整证据见 [`architecture.md` §3.5.6](architecture.md)。

**2026-08-01 已拍板：(c) 启动期拒绝该 flag**，把静默降级变成显式失败。理由：
`mtp_prefill_warm_continue` 只存在于已退役的 `oracle/qwen36_vllm/`，这不是"暂时缺实现"，
是"调用一个属于另一个已截肢子系统的方法"——`try/except` 把这个错误配置伪装成了正常运行
（指标恒为 0、零测试覆盖，三年都不会有人发现）。真要做 warm-continue，等 Track A 的能力查询
（§3.5.3 `BackendCapabilities.warm_continue`）落地后重新评估 (a)，那时候的实现成本和收益都更清楚；
现在花力气实现它，地基（协议/能力查询）还没打，等 A2 落地后很可能要重写。(b)（直接删 flag）
则会丢掉 P4b 已写好的调度逻辑，代价大于 (c)，且不排除后续真的要做 warm-continue。

**落地清单**（零 GPU，可当天完成）：
- [ ] `--session-affinity` 在启动期查 `capabilities.warm_continue`，为 `False` 时直接报错拒绝启动
  （而不是等到运行期某次 `try/except` 才发现），错误信息指向本节
- [ ] 删除或改写 `engine.py:971-978` 的 `try/except Exception` 兜底——它存在的唯一理由（掩盖
  `AttributeError`）随着启动期拒绝而消失；调用点应假定 `mtp_prefill_warm_continue` 存在
- [ ] 补上 `warm_continue` / `session_warm` 的测试覆盖（当前**零覆盖**）：至少一条测试断言
  `--session-affinity` 在当前 `LagunaBackend` 上启动期报错，而不是运行期悄悄退化
- [ ] (a)（为 Laguna 实现 warm-continue）留作 Track A 完成后的重新评估项，不在本次范围内

---

## 7. P0 之后：其余轨道（优先级与顺序）

```
P1  Track B Qwen3.6-27B  B0→B1→B2→B3   ←── M2→M4 主线，串行，2.5–3 月
P1  Track D 易用性 D1→D6                ←── M2→M5，与 B 全程并行
P1  Track C 稳定性 C0R→C6               ←── M1→M6 贯穿，不设独立里程碑
P1  Track G 25B-A3B                     ←── M4→M5，前置=拿到 config
P2  Track E 兼容性 E2→E1→E3→E4          ←── M3→M6，与 B/D 并行
P2  Track F 性能                        ←── 机会主义，永不抢占 C/D；F1/F2 例外，升 P1，见 §7.6
P2  Track H 发布 0.2.0                  ←── M5→M6
```

### 7.1 P1 · Track B · Qwen3.6-27B（M2→M4，串行）

**单点最大风险 RK1**：GDN 占 48/64 层，引入**第二类缓存**（conv + ssm state），
要和固定槽位、前缀缓存、CUDA Graph、投机解码四套机制全部对接。
`docs/archive/2026-07-30-architecture-two-tenant.md` §6.2 是可复用先验。

**B0 事实基线**（M2，2 周，全部 [待办·开发执行]）
- [x] B0-1 变体清点选型，定主线 checkpoint —— ✅ **已拍板（见 §4/C-2）：官方 `nvidia/Qwen3.6-27B-NVFP4`**，
  社区文本版 `sakamakismile/...-Text-NVFP4-MTP` 留作交叉验证。**衍生任务（未开工）**：
  - [ ] B0-1a 写一个按 tensor 名前缀跳过 `vision.*` 的加载过滤器（333 个张量，机械工作，一次写好可复用于
    任何带 vision tower 的衍生 Qwen3.6 checkpoint）
  - [ ] B0-1b A1 的 `validate_text_only` 语义要跟着调整（见 §4/C-2 的附带发现）：从"config 里有
    `vision_config` 就整体拒绝"改成"允许 `vision_config` 存在，但要求 loader 处于
    `language_model_only=True` 并断言零 vision 张量被实际加载"——这条不改 `architecture.md`，
    在 A1 落地时一并处理
- [ ] B0-2 modelopt NVFP4 的 tensor 命名与 scale 语义**逐项确认，不猜**
- [ ] B0-3 sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 / page_size ∈ {64,128} / fp8 KV` 下的正确性与吞吐
- [ ] B0-4 GDN 方案三选一：① FLA v0.5.2 `gated_delta_rule` ② 从 `oracle/qwen36_vllm/` 移植 ③ 自研。**建议先 ① 拿正确性，profiling 说话后再决定 ③**
- [ ] B0-5 GDN 递归状态更新是否 **CUDA Graph capture-safe**（决定 B2 可行性）
- [ ] B0-6 mrope-interleaved 在纯文本下能否退化为标准 1D RoPE
- [ ] B0-7 容量测算：64 层 / 256K / 96 GB 的 KV + 递归状态显存账 → context × 并发可行域
- [ ] **B0-8 · Qwen3.6 的 MTP 层是否带 GDN**（`investigation-queue.md` B-6，**另一 agent 正在查，本文档不预判结论**）：
  vLLM 注释 "draft models have no mamba layers, so no eagle shift"——若我们的 MTP 层也不含 GDN，
  **B3 最难的一项（GDN 递归状态推测回滚）直接不存在**。查法：读 checkpoint `config.json` 里 MTP 段的
  `layer_types` + `mtp.*` 张量名，零 GPU。**应在 B3 排期前答掉**，见下方 B3 的两个分支

**B1 正确性优先**（M2→M3，1 月）：eager、batch=1、无图、无投机、无前缀缓存
- [ ] GDN 层（conv1d state + gated delta rule + 输出门）· Full attention（sparkinfer paged）· 稠密 SwiGLU（NVFP4）· RoPE partial 0.25 + mrope · modelopt 加载 · 注意力输出门控
- **门禁**：与 HF transformers 贪心**逐 token 对齐**（≥ 3 工作负载 × 512 token）；逐层 logits 余弦相似度进 bfdiag

**B2 服务化**（M3，1 月）
- [ ] 固定槽位 + 连续批处理 · 递归状态纳入槽位生命周期 · CUDA Graph（依赖 B0-5）· 前缀缓存联动驱逐（A3 的第一个真实用户）· 并发 ≥ 2
- **门禁**：双协议回归全绿 + **C-LIVE 通过** + 与 B1 eager 贪心 bit-exact

**B3 性能与投机**（M4，1 月）—— **[待验证] 两个分支，取决于 B0-8 的结论，不预判**：
- 若 B0-8 结论 = MTP 含 GDN：MTP draft/verify 含 **GDN 递归状态推测回滚**（本轨道最难）
- 若 B0-8 结论 = MTP 不含 GDN：MTP draft/verify 退化为标准 verify（无递归状态回滚），
  **本轨道最难的一项直接消失**，B3 体量应下修
- [ ] GDN kernel 调优 · 128K/256K 容量与吞吐（两个分支都要）
- [ ] **KV dtype 待定**：`investigation-queue.md` C-2 正在测 NVFP4 KV vs FP8 KV 在我们卡上的 prefill/decode
  对比（另一 agent 进行中）。上游第三方在 RTX PRO 5000 上的数字（NVFP4 KV prefill 慢 1.7–1.8×，
  decode 更快）不是我们的卡也不是我们的形状，**只作为倾向 FP8 KV 的参考，不作为决定**——等 C-2 本机
  结果回来再定，本文档暂标 **[待验证]**，不写死"FP8 KV"
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

> **2026-08-02 补充**：C0R–C7 的分期、新增 C8/C9、以及"如何证明新门禁真的会红"的通用
> 方法论，已详细展开进 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §3/§4/§5——
> 本节保留可勾选的执行清单，细节和调度理由不重复。

- [ ] **C0R** C0 审计残余：① `bfdiag/` 的 **code_ref 行号系统性漂移**（约定改为引用符号名；从命中率最高的 5 个文件起：`checkpoint/state.py`、`daemon/session.py`、`invariants/checks.py`、`shapes/attention.py`、`determinism.py`）② `cold_capacity.py` / `det_cli.py` 未逐行核实 ③ 两条无调用点的不变量（`aux_hidden_alignment`、`cg_replay_slot_consistency`）接线或明确标记
- [ ] **C1 故障面清单**（**2026-08-02 按 GPU 窗口拆成三段，见 `e2e-and-quality-plan.md` §3.1**）：
  - [ ] C1-backend：显存不足 / CUDA Graph 捕获失败 / kernel JIT 失败 / 并发抢占——蹭 P0-E 第 5–8 步窗口
  - [ ] C1-protocol：客户端断连 / tokenizer 边界 / 非法采样参数 / 长请求超时——与 Track E 的 E3 共享同一个热身服务器
  - [ ] C1-slot：槽位卡死——随 A3（第 7 步）落地同窗口验证
  每项要有检测点、指标、日志、用户可见错误、恢复动作
- [ ] **C2 分级降级**：CUDA Graph → eager；投机 → 非投机；前缀缓存命中 → 冷 prefill，每级出指标（**接口形状来自 P0-D 的 D-3 能力查询**）。三个触发点各自蹭对应子系统落地时的 GPU 窗口，不单独申请
- [ ] **C3 看门狗覆盖**：补覆盖测试 + 故障注入。**大部分故障注入用 Python 层 mock 掉 CUDA 异常即可**，不需要真实 OOM；只有少量场景需要真机确认，见缝插针
- [ ] **C4 确定性与可复现**：补 bit-exactness 回归门禁并纳入 CI。**C-1 已拍板**（§4，(b) 本地 pre-push
  + 人工签核）——实现载体是 `make gate-local` + PR 签核勾选项，是 A6 验收的门禁本体
- [ ] **C5 长稳测试**：24h soak [待办·开发执行]。**唯一真正需要独占一整天 GPU、不能蹭窗口的条目**——
  排在 M3 末（Track B 的 B1/B2 验收完成、B3 性能冲刺前的天然间隙）与 M6 末（发布前）两个检查点，
  需要提前和 Track B 协调时间，不能假设"顺路"
- [ ] **C6 崩溃可诊断**：进程级异常留下 bfdiag run record
- [ ] **C7 冷启动 / 首次真实形状路径审计**（本批新增，2026-08-01）：`235f51e` 修的是"每个未见过的
  page-table 宽度都触发 30–100s 重编译"，根因是**编译缓存键覆盖了整个模型该走的 layer-group，
  但预热只覆盖了主模型的 extend/decode 两种 mode**。该修复自己的提交记录留了一个明确未闭合的口子
  （见 [`../notes/2026-08-01-prefill-shape-buckets-root-cause.md`](../notes/2026-08-01-prefill-shape-buckets-root-cause.md)
  "已知缺口"一节），这条不是我猜的，是原作者自己写的 not-tested：
  - [ ] C7-1 DFlash 的 eager verify 回退路径（`_forward_verify_with_aux`，`mode="verify"`）**不在
    `warmup_paged_attention_shapes()` 的覆盖范围内**——它对 `SparkinferPrefillWorkspace._key()` 是独立
    于 extend/decode 的第三个 contract。生产配置下 `DFlashEngine.__init__` 同步捕获 verify/draft CUDA
    Graph，理论上不该走到这条 eager 路径，但**没有直接证据证明本机 100% 命中 CG**——CG 捕获成功只打
    `logger.info`，这个仓库默认日志配置下 info 级别被静默丢弃，只有失败才可见（`logger.warning`）。
    验证方法见该 notes 文件"已知缺口"一节给出的具体步骤（给 `_forward_verify_with_aux`/`_draft_forward`
    接诊断日志，故意让 verify CG 捕获失败或强制走 eager 分支，复现一次）。**如果坐实，修法与 `235f51e`
    完全同构**：给 `enable_dflash()` 声明 `mode="verify"` 的 `prefill_capacity_by_window_left`，warmup
    补一次 `mode="verify"` 的 dummy 调用
  - [ ] C7-2 把 CUDA Graph 捕获**成功**的可观测性从"默认不可见"提到可查询——不是把日志级别拉到
    warning 就完事（那只是把噪音换个位置），是把"这次运行 verify/draft CG 到底捕获成功没有"变成
    `snapshot()`（§3.5.2）或 `/debug/stats` 里的一个字段，跟 `capabilities.cuda_graph` 一起构成
    Track C2 分级降级判断真正需要的信号
  - [ ] C7-3 这不只是 DFlash 一处的问题：`investigation-queue.md` C-1（flashinfer #3255）指向的是同一
    类别——warmup/autotune 是否用**生产真实形状**而不是 autotuner 的第一个合成小形状。B0-3（sparkinfer
    paged attention 在 `head_dim=256/gqa_group=6` 下的验证）应显式包含这条检查，不能只测正确性
  - **为什么现在记录、但不立即安排 GPU 时间**：C7-1/C7-3 都需要真机复现，且是**假设未坐实**（`235f51e`
    的作者本人也说"两个假设都没彻底坐实"）；C7-2（可观测性）不需要额外 GPU，可以和 P0-E 第 5 步
    的第一次真机验证捆一起做，零增量 GPU 成本
  - **2026-08-02 交叉引用**：Track E 的 E3（SDK 矩阵）每个客户端第一次对新代码发请求都天然处于
    冷启动窗口，某些 SDK 的默认超时可能比 JIT 停顿更短——E3 的完成判据应包含"首请求"场景，
    是 C7 的又一个验证入口，见 §7.5/E3
- [ ] **C8 门禁可信度周期审计**（本批新增，2026-08-02）：把 N4/C0 揪出的"从不调用真实函数、一直是
  绿的假门禁"变成常设动作，不是一次性事故处理。每两个里程碑（M2/M4/M6）抽 12–15 个现有测试/门禁，
  逐条回答"它真的红过吗""如果没红过，能不能构造一个会让它红的输入"，答不出来的记入门禁债务清单。
  首轮（M2）优先审计 `bfdiag/checkpoint`（N4 事发模块群，验证修复后的状态经得起这两个问题）。
  方法见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §3.2。零 GPU
- [ ] **C9 质量回归**（本批新增，2026-08-02）：MMLU-Pro（分层子集）+ evalplus HumanEval+/MBPP+ 对
  **Laguna** 跑一次——现有 harness（`benchmarks/official/mmlu_pro_eval.py` + `quality_regression.py`，
  `92f8b34`，2026-07-22）从建成起没有指向过当前生产模型，指向的是已退役的 Qwen3.6/vLLM。
  **[待验证]**：`roadmap.md` §0 引用的"Laguna-S-2.1 MMLU-Pro 84.5%"疑似是那次旧评测（84.54%）的
  误引，需要 C9-a 跑出真数字来核实或替换。
  - [ ] C9-a [待办·开发执行] M2，蹭窗口：对现有 harness 做存在性验证，跑一次小规模 Laguna 子集
  - [ ] C9-b M2→M3：包装成 bfdiag run record，纳入 `bf diff` 可比性纪律，锁定回归基线
  - [ ] C9-c [待办·开发执行] M3→M4（Track B B1/B2 落地后）：加一份 Qwen3.6-27B 覆盖，独立基线
  - [ ] C9-d [待办·开发执行] M6 发布前：全量或接近全量规模跑一次，作为 H1 的一项输入
  方法见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §3.3
- **风险登记**：见 `roadmap.md` §6 RK9（本批新增，"冷启动/首次真实形状路径系统性覆盖不足"是一个模式，
  不是一次性 bug；这条把它显式列成风险而不是留在一条已关闭 commit 的尾巴里）

**插入时机**：C0R 与 C-LIVE 一起在 M1；C1/C2 随 Track A 落地；C3/C4 在 M2→M3；C5/C6 在 M6 发布门禁前
（C5 另有 M3 末检查点）；C7-2 随 P0-E 第 5 步捆绑；C7-1/C7-3 在拿到 GPU 窗口后见缝插针，不单独申请
专用时段；C8 每两个里程碑一次（M2/M4/M6），零 GPU；C9 M2 起首次基线，随 Track B 推进加 Qwen3.6 覆盖。

### 7.4 P1 · Track G · Qwen3.6-25B-A3B（M4→M5）

前置：**先拿到 `config.json`**（本地无 checkpoint，架构参数全未知，RK4 —— 在此之前时间是占位不是承诺）。

- [ ] **G0** [待办·开发执行] 拉 config：专家数 / top-k / 是否 hybrid / 是否带 MTP
- [ ] **G1** 若 Hybrid GDN + MoE：GDN 复用 Track B；MoE 走 sparkinfer `moe.fused_moe`
- [ ] **G2 router kernel 泛化**（S6）：`runtime/laguna_router.py:20-21` 的 `EXPERTS = 256` / `TOP_K = 10` 是模块级常量，`:184-185`、`:205-206` 直接依赖 → 定泛化还是特化
- [ ] **G3** [待办·开发执行] `moe.fused_moe` 在非 256/top-10 形状下的可用性与性能（SM120 上 MoE 已测定带宽饱和）

### 7.5 P2 · Track E · 兼容性（M3→M6）

> **2026-08-02 补充**：`docs/api-layer-design.md` §5/§7 已经把 N1/N2/N3/`n>1`/usage token
> 五条逐项核实过；下面按核实后的真实状态重排。分期细节、每步"如何证明测到了协议层"的
> 方法、GPU 窗口调度，见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2/§4/§5。
> **已关闭，不再列入本节待办**：N2 `stop`（已接通）、N3 `seed`（`PersistentSeed` 已接通）、
> `n>1`（`server/app.py:447` 已显式 400 拒绝，`api-layer-design.md` §5.3——不是"仍待核查"）。

- [ ] **E2 采样 + 投机共存**（消灭 S8）：`temperature > 0` 当前直接退化成无投机自回归
  （`_greedy_accept_reject`，`runtime/backends/laguna_dflash.py:76`），需要拒绝采样 /
  typical acceptance。**这是最明显的功能缺口，排在 E-N1 之前**
  - [ ] E2-a（M2，零 GPU）：拒绝采样/typical-acceptance 算法本身的正确性，CPU 合成分布验证
  - [ ] E2-b [待办·开发执行]（M2→M3，蹭 P0-E 剩余窗口或 Track F F1 窗口）：接进
    `laguna_dflash.py` 的验证步骤；完成判据 = 接受率进 bfdiag 基线 + 采样分布统计学匹配
    非投机路径（KS 检验或等价方法，不是肉眼比较）
- [ ] **E-N1 结构化输出真正实现**（原 E1-N1，重新核实后拆分）：`GrammarState.apply_mask` /
  `apply_mask_batch` 逻辑本身没问题，真正阻塞是解码循环没有可用掩码注入点——admission
  阶段裸 `argmax`、CUDA Graph 贪心重放把贪心烤进 graph、eager 贪心分支绕过
  `sample_from_logits`；默认 `temperature=0.0` 使得最常见请求形态恰好总走这三条不可达
  路径（`docs/api-layer-design.md` §7.1）。**已明确排除**：只接通 `temperature>0` 时唯一
  可达的窄缝——比完全不接更危险（默认场景看起来接上了但仍不受约束）。
  - [ ] E-N1-a（M2，零 GPU，决策备忘）：把两个中间态选项（等全量修复 vs 显式限定
    `temperature>0`-only）写清楚交给需要拍板的人，不代为决定
  - [ ] E-N1-b [待办·开发执行]（拍板后，M3→M4，需 GPU，**前提是 Track A 对
    `laguna.py`/`laguna_cuda_graph.py` 的改动已稳定**——文件归属边界）：实施选定方案；
    完成判据 = N≥20 个样本、覆盖 ≥3 种 schema 复杂度，100% 通过 JSON Schema 校验
- [ ] **E3 客户端 SDK 一致性矩阵**（本批扩充为结构性资产）：现有 `test_api_compat.py`/
  `c_live_smoke.py` 全部手搓 `http.client`，**没有一个用真实厂商 SDK 解析响应**。
  分四步，便宜且本机已装的先做：
  - [ ] E3-a [待办·开发执行]（M2）：openai-python（2.34.0）+ anthropic-sdk-python
    （0.99.0）**[待验证：具体安装在哪个 venv]**——非流式/流式 chat、一次工具调用、一次
    故意触发的 400/422，确认 SDK 把错误体映射成正确的异常类型。首次真机运行蹭 P0-E 第 5
    步或 C-LIVE B-4 的窗口。产出锁定基线（bfdiag run record 或等价 transcript）
  - [ ] E3-b [待办·开发执行]（M3）：LiteLLM——协议归一化层，能捕获两个原生 SDK 都不会
    触发的归一化 bug
  - [ ] E3-c [待办·开发执行]（M4）：Claude Code 本身跑一次真实编码会话（吃自己的狗粮）
  - [ ] E3-d [待办·开发执行]（M5）：Cline / Roo + OpenWebUI（下游集成，排最后）
  - 完成判据 + "如何证明会红"（历史 bug 父提交回放，同 C-LIVE B-4 标准）+ 与 RK9 的交叉
    检查（真实 SDK 默认超时可能比 JIT 停顿更短），见
    [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2.4
- [ ] **E4 reasoning 正确暴露**：OpenAI `reasoning_content` 已接（`c86858a`）。**Anthropic
  侧维持非标准事件，除非拿到合法签名来源**——`f13fd4a` 的生产事故明确禁止无签名重发
  thinking block。**建议补一条零成本常驻回归测试**：断言 Anthropic 流式路径不产出未签名
  的规范 `thinking` block，写一次不需要里程碑节奏
- [ ] **E5 chunked input-logprob 默认开启**（本批新增，来自 SGLang v0.5.16，`investigation-queue.md` D-8）：
  我们已有 logprobs 路径，双协议都暴露 `top_logprobs`；长 prompt 的峰值显存是真问题（Track F 的
  F1/F2 也在争这块显存）。**小而自足，不依赖 Track A**——建议提前到 M2 跟 F1 的窗口扫测一起做，
  不用等 Track E 默认的 M3 窗口
- [ ] **usage token 两个小缺口**（2026-08-02 从"待核查"降级为"已知的两个小缺口"，
  `api-layer-design.md` §5.5）：
  - [ ] `usage.completion_tokens_details.reasoning_tokens` 细分字段缺失——独立、小，可随时排期
  - [ ] `<usage>` 标签剥离流式/非流式语义不统一——需先拍板统一到哪种语义，再改

### 7.6 P2 · Track F · 性能（机会主义，但 F1/F2 例外）

**纪律**：只在有明确 roofline 依据、且不损害 Track C/D 的前提下做；必走 `bf diff` + 接受率/质量门禁。

**2026-08-01 本批新增两条例外**：F1/F2 从"机会主义/P2"提升到 **P1**，排进 M2。理由——不是因为它们
比 Track A 更急，是因为它们**不依赖 Track A**、成本低、且直指本项目当前两个最硬的约束（吞吐上限、
显存上限），有本机实测数据支持，不是纯粹的"顺手试试"：
- 接受率实测 96.3–100%（`roadmap.md` §1.1，2026-07-31/08-01 复现）但 `NUM_SPECULATIVE_TOKENS` 固定 15——
  这个组合说明限制吞吐的很可能是窗口本身，不是接受率上限（F1）。
- Laguna 权重 66.8 GB ≈ 67 GB（59.5 GB MoE + 7.3 GB non-MoE，`notes/2026-07-29-gpu-memory-audit.md`），
  96 GB 卡上留给 KV + 投机 scratch + 其它的预算很紧。**协调者在本轮任务中的实时汇报（2026-08-01）**：
  生产服务实测显存 94.2 / 97.9 GB（**98.8% 占用**）——比 2026-07-29 那份静态审计（1 slot/131K 配置，
  76.0/95.6 GB，79.5%）紧得多，两者配置不同（并发/上下文长度未知，暂不能直接对比），但方向一致：
  **投机 scratch 在跟 KV 抢一块越来越紧的预算**（F2）。

**F1 · DFlash 固定窗口 → 加宽/自适应 verify 窗口**（来自 `investigation-queue.md` D-2，SGLang/vLLM 的
DSpark）。**分两步，先做便宜的那步**：
- [ ] F1-1（便宜，先做）：不实现自适应控制器，先把 `NUM_SPECULATIVE_TOKENS` 从 15 静态调大（16→24→32
  等），用 `bf diff` 测各工作负载的 tok/s 与接受率。如果这一步就有明显收益且接受率不掉，F1-2 的
  优先级立刻下降——没必要为了一个静态调参就能拿到的收益去接一套新的置信度估计
- [ ] F1-2（若 F1-1 见顶但接受率仍然很高才做）：置信度驱动的自适应窗口（DSpark 风格）。
  **警告不是白捡**：vllm #49369 报告 DSpark 在某些负载上比不开投机还慢，必须按工作负载分别 A/B，
  不能默认全开
- **调度**：不需要 Track A，可以现在做；但需要真机 GPU 时间，**排进 M2 时优先蹭 P0-E 第 5 步或
  C-LIVE 的 GPU 窗口，不单独申请专用时段**——本机只有一块 GPU，任何需要 GPU 的验收项天然串行
  （见 `roadmap.md` §6 RK5 的更新）

**F2 · 投机 scratch 显存优化**（来自 `investigation-queue.md` D-3，ReplaySSM Ring Spec-Verify 报告
11.5 GB → 1.8 GB——**别人的卡、别人的形状，不能当我们的数字用**）：
- [ ] F2-0 先补一次带日期来源的显存审计（当前并发/上下文配置下的真实占用，与协调者汇报的
  94.2/97.9 GB 对齐或更新），确认 DFlash 投机 scratch 实际占用多少、KV 实际还剩多少余量——没有这一步，
  "值不值得做"无法判断
- [ ] F2-1 读 ReplaySSM 的 ring-buffer 技巧具体怎么把 scratch 降下来的，映射到我们自己的
  draft/verify CUDA Graph scratch 分配（`laguna_dflash_cudagraph.py`），判断有多少是**调度/复用层面**
  能拿到（我们做），有多少要动 sparkinfer 的 kernel 内部（转 SparkInfer，按 `AGENTS.md` 规矩写清楚
  交接，不直接改源码）
- [ ] F2-2 若调度层面就有收益：实现 + `bf diff` 判可比性 + 接受率/bit-exact 回归门禁
- **为什么不是"机会主义"**：显存是这个项目的硬约束；F2-0/F2-1 的结论应该喂给 A3 协调者设计
  （P0-E 第 7 步，见 `architecture.md` §3.5.5）——投机 scratch 迟早要变成 A3 管理的资源类型之一，
  不要各自为战

**D-5（Hybrid SWA+full DFlash drafters + 投机专用 kv_cache_dtype）——已核实，无需新工作**：
`investigation-queue.md` D-5 说"直接对口"，读代码后发现**不完全对**：①投机专用 `kv_cache_dtype`
**已经是现状**——`laguna_dflash.py` 里 draft KV cache 显式按 `# Self-allocated: FP8 as uint8` 分配，
与主模型该层自己的 dtype 选择独立，不是新工作；②"hybrid (SWA+full) 的 drafter"这个 vLLM 新能力，
我们的 draft 模型走的是另一条路——固定 6 层、全 SWA（window=512）、bf16 权重，KV cache 只有
0.007 GB（`notes/2026-07-29-gpu-memory-audit.md`），已经靠"用一个更小的专用 draft 模型"达到了跟
"hybrid 主模型 KV dtype 独立"类似的省显存效果，没有证据表明改成 hybrid drafter 会更好。
**结论：D-5 从待办清单移除，不是降级，是核实后发现已经做到。**

- [ ] **F3** TURBO_ATTN 质量回归修复（per-head descale / Hadamard 旋转 / 自适应切换）：收益 +6%，但 code 接受率 97.8% → 58.6%，当前默认关闭
- [ ] **F4** FA4 技法用于 prefill / extend（TMA、persistent scheduler、FP8 softmax）。
  **T0 触发条件**（`investigation-queue.md` D-6）：FlashAttention 维护者已合入 sm120 PR（#2413）且有
  面向 5090 的 TMA + warp specialization PR 在做（#2440），但 FA4 本体上不了 SM120（缺 tcgen05/TMEM），
  当前 sm120 路径只有 FP16/BF16、`main` 部分路径仍报错、在 5090 上比 FA2 **慢约 5%**。**保持观察，
  不要提前动**——触发条件是"那批 PR 落到 main 且在 sm120 上跑赢 FA2"，到那时才从"自己移植"变成
  "评估采纳"
- [ ] **F5** FP8 attention `num_stages ≥ 2`（SMEM 36 KB « 99 KB）
- [ ] **F6** sparkinfer 剩余 9 处 gate（`7a1d69d` 只放宽 13 处中的 4 处，其余是**刻意留下**的 scope 限制）
  - ⚠️ **硬风险**：`planner.py` 的 grid occupancy 预算常量按 `num_kv_heads=4` 推导，用到 8 **需重新推导**，不是放宽谓词就够
  - ⚠️ sparkinfer 源码改动写清楚交给该团队，不直接改
- [ ] **F7** MoE 输出中心并行（Warp Decode 类），2–4 周，长期备选
- [ ] **F8** GDN kernel 自研（依赖 B0/B2 profiling）
- [ ] **F9** 评估 `perf/repro-2ce5-baseline-20260730` 上未合并的 `060fabb`（见 P0-A/A-2）
- [ ] **F10** NVFP4 per-token online MoE 量化（`investigation-queue.md` D-7，vLLM v0.26.0 + CuTe-DSL
  MXFP4）：Laguna 是 256 专家 NVFP4 MoE，直接可比。**kernel 形状 → 写清楚交给 SparkInfer 团队评估，
  不直接改其源码**（按 `AGENTS.md` 规矩）——本条目"我们做"的部分只是写一份技术提案文档

### 7.7 P2 · Track H · 发布 0.2.0（M5→M6）

- [ ] **H1 发布门禁**：CI 绿（两个 job）· C-LIVE 通过 · 24h 长稳（C5）· 两个模型系列质量回归 · 文档三件套 · **依赖可从公开源安装（即 sparkinfer 上游化，RK2 未解，当前钉 `origin/master @ 0844a4f`）**
- [ ] **H2 素材纪律**：只发实测数字，标注硬件/配置/复现命令；不做 apples-to-oranges 对比
- [ ] **H3** `0.2.0` = "多模型 + 生产可用"的第一个公开版本

---

## 8. 里程碑对齐

| 里程碑 | 时间 | 对应条目 | 说明 |
|---|---|---|---|
| **M1** | 2026-08 | **P0-A + P0-B + P0-C + P0-D**，A1/A2 起步 | Track 0 实际已近清零，M1 富余产能应投入 C-LIVE 与设计定稿；P0-C 的 D3/D6/N8 三项拍板已完成 |
| **M2** | 2026-09 | A3–A6 落地 + B0 事实基线（含 B0-8）+ B1 起步 + D1/D2 起步 + **F1-1/F2-0 机会窗口**（蹭 A6/C-LIVE 的 GPU 时段）+ E5 | Laguna 零回归是硬门禁；B0-8/C-2/C-3 三条 [待验证]，由并行 agent 产出，回来后再定 B3 分支与 KV dtype |
| **M3** | 2026-10 | B1 验收 + B2 服务化 + D1/D2/D3-impl 交付 + C1/C2 | 一条命令能起服务 |
| **M4** | 2026-11 | B3 性能与 MTP + G0/G1 | 25B-A3B 正确性对齐 |
| **M5** | 2026-12 | G2/G3 + D4/D5/D6 收口 + E3 客户端矩阵 | 三个模型系列同一套配置流程 |
| **M6** | 2027-01 | C5/C6 + Track H 全项 | 24h soak + 发布 checklist |

---

## 9. 阻塞与依赖速查

| 被卡的事 | 卡它的 | 状态 |
|---|---|---|
| ~~A6 零回归验收~~ | ~~**P0-C/C-1 GPU CI 拍板**~~ → C4 位精确门禁 | 🟢 **已拍板 (b)**：本地 pre-push 门禁 + 人工签核；`make gate-local` 机制待落地（§7.3/C4） |
| ~~B0 起步~~ | ~~**P0-C/C-2 主线 checkpoint 拍板**~~ | 🟢 **已拍板**：官方 `nvidia/Qwen3.6-27B-NVFP4`；B0-1 衍生出 vision 张量过滤任务，见 §7.1 |
| ~~A1–A6 开工~~ | ~~P0-D 设计升级到可实施~~ | 🟢 **已解锁**：规格在 `architecture.md` §3.5，第 1–4 步可立即开工且不需要 GPU |
| P0-E 第 5–8 步 | **P0-B C-LIVE** + GPU | 🔴 C-LIVE 未开始；F1-1（窗口扫测）可蹭同一 GPU 窗口一起做 |
| ~~N8 处置~~ | ~~拍板 (a)/(b)/(c)~~ | 🟡 **已拍板 (c)**：启动期拒绝该 flag，实现清单见 §6.1，尚未落地（零 GPU） |
| Track B 全部 | Track A（A1–A5）完成 | 🔴 未开始 |
| B0-8 GDN 是否存在于 MTP | `investigation-queue.md` B-6（另一 agent 在查） | 🟡 进行中，**不预判**；决定 B3 的两个分支（§7.1） |
| B3 KV dtype 选型 | `investigation-queue.md` C-2（另一 agent 在查） | 🟡 进行中，**不预判**；上游数字仅供参考，不当结论用 |
| B2 CUDA Graph | B0-5（GDN 是否 capture-safe） | 🔴 未验证 |
| C2 分级降级的接口 | P0-D/D-3 能力查询形状 | 🔴 未定 |
| Track G 排期 | G0 拿到 config | 🔴 本地无 checkpoint |
| H1 发布 | sparkinfer 上游化（RK2） | 🔴 仍钉私有 fork |
| RK6/H1 "可从公开源安装" | `investigation-queue.md` C-3（PyTorch 2.13.0 wheel 是否带 `sm_120`，另一 agent 在查） | 🟡 进行中，**不预判** |

---

## 10. 不做清单（不再重复讨论）

多卡 TP/PP/EP · 多机 · SM120 以外架构 · 视觉/多模态输入 · 训练/微调/LoRA 热加载 ·
AWQ/GPTQ/INT4 等其他量化 · 通用 HF 架构自动支持 · 复活 `oracle/qwen36_vllm/`。理由见 [`roadmap.md`](roadmap.md) §3。

---

## 11. 配套文档

- [`roadmap.md`](roadmap.md) — 目标、理由、风险登记、待拍板事项（**权威**）
- [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) — Track C/Track E 的详细分期、GPU 窗口调度、
  "反复审查"节奏机制、每条新门禁"如何证明会红"的方法论
- [`architecture.md`](architecture.md) — 现状架构 + **目标架构与五个关键抽象（Track A 设计草案，§3）**
- [`model-support.md`](model-support.md) — 模型支持矩阵 + 接入新模型的操作指南
- [`diagnostics-guide.md`](diagnostics-guide.md) — bfdiag 使用指南（必读）
- [`../notes/2026-08-01-bfdiag-assertion-audit.md`](../notes/2026-08-01-bfdiag-assertion-audit.md) — C0 审计全文
- [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md) — fork 差异与 9 处未放宽 gate 的安全性分析
