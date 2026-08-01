# BlackweLLM 实施计划清单（2026-08-01 编制）

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
| T0-7 仓库卫生 | 🔴 未做 | 🟢 **主体已完成**：`benchmarks/` 136 → 107 个脚本（删 29 个零引用一次性脚本 + vLLM bit-exact 家族，加 README 规约）；分支 21 → 8，worktree 6；`.ruff_cache/` 已 gitignore；无 tracked 的 `.bak`/`.orig`/`*.log`/`build/` |
| T0-8 N6 全套件 flaky | 🔴 未做 | 🟡 **已处理但未根因**：`914e3f2` 把断言精确到目标行 + 失败时打印原始输出与存活线程。~28 次真机重跑（GPU 40–60% + 4 核满载）**零复现**，与简报的 3/5 有明显差距，根因未定 |
| T0-8 N7 Protocol 不符 | 🔴 未做 | 🟢 已修（`76bcf3e`） |
| N2 `stop` 序列未实现 | 🔴 | 🟢 已实现，含跨 token 边界匹配（`0cd772c`） |
| N3 `seed` 每 token 重播种 | 🔴 | 🟢 已修为每请求推进单一 generator（`0700b25`） |
| N1 结构化输出是空壳 | 🔴 最严重 | 🟡 **危险性已消除、功能缺口仍在**：`b2d73cb` 改为显式拒绝 `json_object`/`json_schema`，不再静默失败。`GrammarState.apply_mask` 依旧无调用点 → **降级为普通功能开发，不再是 P0 事故** |
| N4 / C0 bfdiag 隔离保证失效 | 🔴 优先 | 🟡 **主体已完成**：审计成文（`notes/2026-08-01-bfdiag-assertion-audit.md`，341 行），逐模块判定 real / fake / honest-split，删掉伪装成回归门禁的合成张量 demo，新增真调用 `restore_checkpoint`、`reset_laguna_engine` 的测试。**剩余见 C0R** |

### 0.1 roadmap 里没有、但必须进计划的一条

`main` 最新三个提交（`2c06355` / `d9e52ce` / `619a09d`）是**同一天内三个只有真机活服务器才能暴露的 bug**：

- `/metrics` 冷启动窗口 500（`get_stats()` 短路返回 2 键，渲染读 6 键）；
- `/metrics` 忙时 500（`slot_kv_len` 是 list，端点当 mapping 读，**同一端点当天第二次挂**）；
- 引擎丢唤醒 —— 会话第二轮必挂（对话客户端每次都恰好落在那 152 ms 窗口）；
- 附带：Laguna 的 chat template **确实注入 `<think>`**，源码注释写反了，导致 `</think>` 被当正文首字符送出。

**1100 条测试全部看不见它们**，因为它们依赖的是部署事实而非代码逻辑。
→ 新增 **C-LIVE**（下方 P0-2）。在 Track A 这种动核心执行路径的重构**之前**建立这道门禁，
否则重构会批量生产同一类 bug 而门禁看不见。

---

## 1. 执行顺序总览

```
P0  ①T0-R 卫生收尾  ②C-LIVE 活服冒烟门禁  ③拍板 D3/D6（卡后续轨道）
        ↓
P0  ④Track A 模型抽象层 A1→A6  ←──── M1→M2 主线，1.5 月，其它一切的前置
        ↓
P1  ⑤Track B Qwen3.6-27B  B0→B1→B2→B3   ←── M2→M4 主线，串行，2.5–3 月
P1  ⑥Track D 易用性 D1→D6                ←── M2→M5，与 B 并行
P1  ⑦Track C 稳定性 C0R→C6                ←── M1→M6 贯穿，不设独立里程碑
P1  ⑧Track G 25B-A3B                      ←── M4→M5，前置=拿到 config
P2  ⑨Track E 兼容性 E1→E4                 ←── M3→M6，与 B/D 并行
P2  ⑩Track F 性能                         ←── 机会主义，永不抢占 C/D
P2  ⑪Track H 发布 0.2.0                   ←── M5→M6
```

**贯穿性主线**：Track A（抽象层）→ Track B（Qwen3.6）是唯一的长串行链，决定整体交付日期。
**真并行**：Track D 与 Track B 全程可并行（不同人、不同文件面）；Track C 贯穿但按需插入。
**关键路径上的两个拍板**：**D3（GPU CI 形态）卡 A6 验收门禁**，**D6（主线 checkpoint）卡 B0 起步**。

---

## 2. P0 · 立刻做（M1 收尾，约 1 周）

### ① T0-R · Track 0 卫生收尾

- [ ] **T0-R1** 主工作区 untracked 残留清理：根目录 9 个 `*.log`、`build/`、`evalplus_results/` 归位或删除（均未被 git 跟踪，属工作区污染，不影响仓库）
- [ ] **T0-R2** 清理 6 个 worktree 中的失效项（含 `.claude/worktrees/laguna-mid-conversation-system`）与 8 个分支里已合并的
- [ ] **T0-R3** `benchmarks/` 第二轮分流：107 个脚本按 README 规约再筛一遍，能转 `bf exec` 的转掉
- [ ] **T0-R4** N6 结案规则落文：**若再复现，直接贴带线程列表的失败输出，不许重跑**；若 M2 结束前零复现，从 roadmap 移除该条目

**验收**：`git status` 干净；`git branch` 只剩活跃分支。**体量**：2 天。

### ② C-LIVE · 活服务器冒烟门禁（本计划新增，P0）

依据 §0.1。目标不是覆盖率，是**把"只有部署事实才能证伪的假设"变成自动化断言**。
基础设施已存在：`scripts/blackwellm_ctl.sh`（start/stop/status/logs）。

- [ ] **C-LIVE-1** 冒烟脚本：起真服务 → 断言下列各项 → 干净关停
  - [ ] `/metrics` 在**冷启动窗口**（无任何请求完成时）返回 200 且六个聚合键齐全
  - [ ] `/metrics` 在**长请求进行中**（`engine.active` 非空）返回 200 —— 这条正是当天第二次挂的形状
  - [ ] **背靠背两轮对话**，第二轮在前一轮响应结束后 **< 200 ms** 发出 —— 丢唤醒 bug 的精确窗口
  - [ ] OpenAI + Anthropic 双协议 × 流式/非流式 × 工具调用各一条
  - [ ] thinking 契约：在**模板注入 `<think>` 的真实模型**上断言 `content` 首字符不是 `</think>`
  - [ ] `/v1/completions` 逐字返回（不套 chat template）—— 2026-07-27 事故的守门断言
- [ ] **C-LIVE-2** 挂进 `blackwellm_ctl.sh` 与 Makefile，成为**每次动 `server/` 或 `runtime/backends/` 后的手动必跑项**
- [ ] **C-LIVE-3** [待办·开发执行] 首次真机运行并把结果记进 bfdiag run record

**验收**：三个已修 bug 中的每一个，在其修复提交的父提交上跑该脚本都能变红。**体量**：3 天。

### ③ 拍板 · 卡住后续轨道的两项

- [ ] **D3 GPU CI 形态**（RK7）：(a) 自托管 runner / (b) 本地 pre-push 门禁 + 人工签核 / (c) 只在里程碑人工全量跑
  → **卡 C4 位精确门禁 → 卡 A6 的"零回归"验收**。不定，Track A 就没有可执行的验收标准
- [ ] **D6 Qwen3.6 主线 checkpoint**：带 MTP 的社区文本版 vs 官方 NVFP4 版 → **卡 B0 起步**
- [ ] （可延后）**D4 重命名时机**、**D5 `oracle/qwen36_vllm/` 处置** —— 不卡任何轨道，随 Track D / Track B 收尾时决定

---

## 3. P0 · Track A 模型抽象层（M1→M2，约 1.5 月，主线）

一切"多模型"的前置。设计目标不是通用性，是**让接入第 N 个模型的成本可预测**。
**原则**：协议由现有实现倒推，不预设未来；逐步切换，每步可回滚。

- [ ] **A1 `ModelSpec` 升级为架构描述**（当前仅 88 行、只有层名列表 + MTP 开关）
  - 从 `config.json` 解析：层类型序列、每层注意力/线性注意力/MLP 类型、RoPE 配置、量化格式、MTP 配置、**每层的缓存需求（分页 KV vs 递归状态）**
  - 不支持的架构在**加载权重之前**报错，不是跑到一半 NaN
  - **RK8**：显式拒绝带 vision tower 的权重，给明确错误（Qwen3.6 文本版与多模态版共用架构名）
- [ ] **A2 Backend 协议**：把 `LagunaBackend` 的 50+ 公开方法收敛成显式协议（prefill / chunked prefill / decode / decode_batch / reset_slot / prefix 匹配与回放 / spec-decode 生命周期 / CUDA Graph 捕获）
- [ ] **A3 缓存资源抽象**：`block_pool` 从"KV 分页器"升级为"槽位资源管理器"，统一管理分页 KV（长度相关）与递归状态（长度无关、每槽固定）；前缀缓存驱逐对两类资源联动。顺带清掉 S4 的 GDN 残迹（`evict_gdn_checkpoint` 等当前无活代码路径）
- [ ] **A4 加载器抽象**：compressed-tensors / modelopt 两套 NVFP4 布局的 tensor 命名与 scale 语义拆成两个 adapter，公共部分（分片流式读取、参数全覆盖断言、KV scale post-load）不变
- [ ] **A5 模型注册表 + 自动识别**：checkpoint 路径 → 读 config → 匹配架构 → 选 backend + loader + spec 策略。**消灭 `ServerEngine.MODEL = "poolside/Laguna-S-2.1-NVFP4"`（`server/engine.py:188`）、`BACKEND = "laguna"`（:190）、`SERVER_MODEL_BACKEND`（`server/app.py:81`）**
- [ ] **A6 Laguna 迁到新抽象，零回归**（硬门禁，**依赖 D3 拍板**）
  - [ ] 贪心输出 **bit-exact**
  - [ ] 性能不低于基线 3%（基线：fox-64K 353–368 / fox-4K 353–357 / galaxy-4K 395–401 / code-4K 341–359 tok/s）
  - [ ] 接受率不回归（96.3–100%）
  - [ ] 比数前先 `bf diff` 判可比性 —— 2026-07-27 教训
  - [ ] [待办·开发执行] 上述四条均需真机

**风险 RK3**：这是动核心执行路径的重构，Laguna 是唯一生产模型。**先 C-LIVE 后 A**，且每步可回滚。

---

## 4. P1 · Track B · Qwen3.6-27B 接入（M2→M4，约 2.5–3 月，串行）

**单点最大风险 RK1**：GDN 占 48/64 层，且引入**第二类缓存**（conv state + ssm state），
要和固定槽位、前缀缓存、CUDA Graph、投机解码四套已有机制全部对接。
`docs/archive/2026-07-30-architecture-two-tenant.md` §6.2 是可复用先验。

### B0 · 事实基线（M2，约 2 周）—— 全部是 [待办·开发执行]

- [ ] **B0-1** 本地 4 个变体清点选型，确定主线 checkpoint（**依赖 D6 拍板**）
- [ ] **B0-2** modelopt NVFP4 的 tensor 命名与 scale 语义**逐项确认，不猜**
- [ ] **B0-3** sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 / page_size ∈ {64,128} / fp8 KV` 下的正确性与吞吐（planner 有分支，无本机实测记录）
- [ ] **B0-4** GDN 方案三选一：① FLA v0.5.2 `gated_delta_rule`（chunk + fused_recurrent 两条 Triton 路径）② 从 `oracle/qwen36_vllm/` 移植 ③ 自研。**建议先 ① 拿正确性，profiling 说话后再决定要不要 ③**
- [ ] **B0-5** GDN 递归状态更新是否 **CUDA Graph capture-safe**（决定 B2 可行性）
- [ ] **B0-6** mrope-interleaved 在纯文本输入下能否退化为标准 1D RoPE
- [ ] **B0-7** 容量测算：64 层 / 256K / 96 GB 下 KV + 递归状态显存账 → context × 并发可行域

**验收**：一份事实基线文档，每项写成"实测值 + 复现命令"。

### B1 · 正确性优先（M2→M3，约 1 月）

刻意放弃全部性能：eager、batch=1、无 CUDA Graph、无投机、无前缀缓存。

- [ ] GDN 层（conv1d state + gated delta rule + 输出门）
- [ ] Full attention 层（走 sparkinfer paged）
- [ ] 稠密 SwiGLU MLP（NVFP4）—— 需要稠密 NVFP4 GEMM 路径
- [ ] RoPE：partial_rotary_factor 0.25 + mrope-interleaved
- [ ] modelopt 权重加载（A4 的第二个 adapter）
- [ ] 注意力输出门控（swish gate）

**验收门禁**：与 HF transformers 参考实现贪心**逐 token 对齐**（≥ 3 工作负载 × 512 token）；逐层 logits 余弦相似度进 bfdiag。

### B2 · 服务化（M3，约 1 月）

- [ ] 接入固定槽位调度 + 连续批处理
- [ ] 递归状态纳入槽位生命周期（reset / 复用 / 看门狗回收）
- [ ] CUDA Graph 捕获（decode 路径；依赖 B0-5 结论）
- [ ] 前缀缓存：递归状态 checkpoint 与 KV 块**联动驱逐**（A3 的第一个真实用户）
- [ ] 并发 ≥ 2

**验收**：HTTP 端到端双协议回归全绿 + **C-LIVE 冒烟通过** + 与 B1 eager 路径贪心 bit-exact。

### B3 · 性能与投机（M4，约 1 月）

- [ ] MTP draft / verify（自带 1 层），含 **GDN 递归状态的推测回滚**（本轨道最难的一处）
- [ ] GDN kernel 调优或自研（依据 B0/B2 profiling）
- [ ] FP8 KV
- [ ] 长上下文 128K / 256K 容量与吞吐

**验收**：接受率与吞吐进 bfdiag 基线；与上游框架同 prompt 同参数 A/B。

---

## 5. P1 · Track D · 易用性（M2→M5，与 Track B 全程并行）

现状"必须读源码才能正确启动"，目标"读一页文档就能上线"。

- [ ] **D1 单命令启动** `blackwellm serve <model-path-or-id>`，自动推导槽位与块数（当前 `[project.scripts]` 只有 `bf`，无服务入口）
- [ ] **D2 显存规划器**：给定「模型 + 目标上下文 + 目标并发」算出配置并校验，或给定显存反推可行域 → **消灭 S7 那个 `QSR_SERVER_CAPACITY`/`NUM_SLOTS`/`BLOCKS_PER_SLOT`/`PRODUCTION` 四变量耦合陷阱**
- [ ] **D3-impl 启动前置检查**：`runtime/preflight.py` 已有 738 行九项校验，需扩到 SM120 检测 / 显存 / CUDA driver / sparkinfer 版本 / checkpoint 完整性 / **架构是否受支持**，全部在加载权重之前，错误信息带修复建议
- [ ] **D4 配置文件**：YAML 取代十几个环境变量（环境变量降级为覆盖手段）
- [ ] **D5 命名统一**：`QSR_` → `BWLLM_`（**当前 370 处引用**），带一个版本的兼容期与弃用警告；包目录 `qwen-sm120-runtime` → `blackwellm` 时机见拍板 D4
- [ ] **D6-docs 文档三件套**：安装部署 / 配置调参 / 故障排查

**排序建议**：D1 + D2 + D3-impl 在 M3 一批交付（它们共同构成"一条命令起服务"），D4/D5/D6 在 M5 收口。

---

## 6. P1 · Track C · 稳定性（M1→M6 贯穿，不设独立里程碑）

核心思路：**把每一种失败都变成一个有名字、有指标、有降级路径的已知状态**。

- [ ] **C0R C0 审计残余**（主体已完成，见 §0）
  - [ ] `bfdiag/` 的 **code_ref 行号系统性漂移**：约定改为"能引用符号名就别引用行号"；从命中率最高的 5 个文件起（`checkpoint/state.py`、`daemon/session.py`、`invariants/checks.py`、`shapes/attention.py`、`determinism.py`）
  - [ ] `cold_capacity.py` / `det_cli.py` 两个模块**未逐行核实**（当前标"未核实"而非"已排除"）
  - [ ] 两条定义了但 `runtime/` 无调用点的不变量（`aux_hidden_alignment`、`cg_replay_slot_consistency`）：接线或明确标记为未接线
- [ ] **C1 故障面清单**：显存不足 / 槽位卡死 / CUDA Graph 捕获失败 / kernel JIT 失败 / 长请求超时 / 客户端断连 / tokenizer 边界 / 非法采样参数 / 并发抢占 —— 每项要有检测点、指标、日志、用户可见错误、恢复动作
- [ ] **C2 分级降级**：CUDA Graph → eager；投机 → 非投机；前缀缓存命中 → 冷 prefill。每级出指标（部分已有，需成体系）
- [ ] **C3 看门狗覆盖**：已有 stale slot 回收，补覆盖测试 + 故障注入
- [ ] **C4 确定性与可复现**：per-request seed 已有；补 bit-exactness 回归门禁并纳入 CI（**依赖 D3 拍板**；是 A6 验收的实现载体）
- [ ] **C5 长稳测试**：24h soak，监控显存碎片、host 内存、槽位分布、指标漂移 [待办·开发执行]
- [ ] **C6 崩溃可诊断**：进程级异常留下 bfdiag run record，不是只有一行 traceback

**插入时机**：C0R 与 C-LIVE 一起在 M1 做完；C1/C2 随 Track A 落地（新抽象层正好是加降级钩子的时机）；C3/C4 在 M3；C5/C6 在 M6 发布门禁前。

---

## 7. P1 · Track G · Qwen3.6-25B-A3B（M4→M5）

**前置：先拿到 `config.json`**。本地无 checkpoint，架构参数全部未知 —— 在此之前**时间是占位不是承诺**（RK4）。

- [ ] **G0** [待办·开发执行] 拉 `config.json`，确认：专家数 / top-k / 是否 hybrid / 是否带 MTP
- [ ] **G1** 若为 Hybrid GDN + MoE：GDN 复用 Track B 成果；MoE 走 sparkinfer `moe.fused_moe`（NVFP4 已支持）
- [ ] **G2 router kernel 泛化**（S6）：`runtime/laguna_router.py:20-21` 的 `EXPERTS = 256` / `TOP_K = 10` 是模块级常量，且在 `:184-185`、`:205-206` 被直接依赖 → 决定是泛化还是为新形状再做一个特化
- [ ] **G3** [待办·开发执行] sparkinfer `moe.fused_moe` 在非 256/top-10 形状下的可用性与性能（SM120 上 MoE 已测定为带宽饱和，泛化的性能代价未知）

---

## 8. P2 · Track E · 兼容性（M3→M6，与 B/D 并行）

- [ ] **E1-N1 结构化输出真正实现**：`runtime/structured_output.py` 的 `GrammarState.apply_mask` / `apply_mask_batch` 接进 `server/engine.py` 的采样环。**优先级已从"P0 事故"降为"普通功能"** —— `b2d73cb` 后是显式拒绝而非静默失败，客户端不再拿到假的 JSON 保证
- [ ] **E1-剩余** 仍待核查：`n>1`、usage token 统计准确性
- [ ] **E2 采样 + 投机共存**（消灭 S8）：当前 `temperature > 0` 直接退化成无投机自回归。需要拒绝采样 / typical acceptance。**这是当前最明显的功能缺口**，建议排在 E1-N1 之前
- [ ] **E3 客户端验证矩阵** [待办·开发执行]：openai-python / anthropic-sdk / Claude Code / Cline / Roo / OpenWebUI / LiteLLM，每个跑真实会话，结果做成兼容性表进文档
- [ ] **E4 reasoning 正确暴露**：OpenAI `reasoning_content` 已接（`c86858a`）。**Anthropic 侧维持非标准事件，除非拿到合法签名来源** —— 见 roadmap §1.4，`f13fd4a` 记录的生产事故明确禁止无签名重发 thinking block

**E 轨内部排序**：E2 → E1-N1 → E1-剩余 → E3 → E4（条件满足才做）。

---

## 9. P2 · Track F · 性能（机会主义，M3→M6）

**纪律**：只在有明确 roofline 依据、且不损害 Track C/D 的前提下做。
任何性能改动必须走 `bf diff` 判可比性 + 接受率与质量回归门禁。

- [ ] **F1** TURBO_ATTN（FP8 QK MMA）质量回归修复：per-head descale / Hadamard 旋转 / 自适应 FP8-BF16 切换。收益 +6%，但 code 工作负载接受率 97.8% → 58.6%，当前默认关闭
- [ ] **F2** FA4 技法用于 prefill / extend（TMA、persistent scheduler、FP8 softmax）
- [ ] **F3** FP8 attention 的 `num_stages ≥ 2`（SMEM 36 KB « 99 KB，有余量）
- [ ] **F4** sparkinfer 剩余 9 处未放宽的 gate（`7a1d69d` 只放宽了 13 处中的 4 处；其余是**刻意留下**的 scope 限制，不是遗漏）
  - ⚠️ **硬风险**：`planner.py` 的 grid occupancy 预算常量按 `num_kv_heads=4` 推导，用到 8 **不是放宽谓词就够，需要重新推导**
  - ⚠️ 按既定约束：sparkinfer 源码改动写清楚交给该团队，不直接改
- [ ] **F5** MoE 输出中心并行（Warp Decode 类方案），2–4 周量级，长期备选
- [ ] **F6** GDN kernel 自研（依赖 B0/B2 profiling 结论）

---

## 10. P2 · Track H · 发布 0.2.0（M5→M6）

- [ ] **H1 发布门禁全项**
  - [ ] CI 绿（两个 job）
  - [ ] C-LIVE 冒烟通过
  - [ ] 24h 长稳通过（C5）
  - [ ] 两个模型系列的质量回归通过
  - [ ] 文档三件套齐备（D6-docs）
  - [ ] 依赖可从公开源安装 —— **即 sparkinfer 上游化完成（RK2 未解，当前钉 `origin/master @ 0844a4f`）**
- [ ] **H2 素材纪律**：只发实测数字，标注硬件 / 配置 / 复现命令；不做 apples-to-oranges 对比
- [ ] **H3** 版本 `0.2.0` = "多模型 + 生产可用"的第一个公开版本

---

## 11. 里程碑对齐（体量校准）

| 里程碑 | 时间 | 本清单对应条目 | 说明 |
|---|---|---|---|
| **M1** | 2026-08 | ①T0-R ②C-LIVE ③拍板 + Track A 设计定稿（A1/A2 设计） | Track 0 实际已近清零，M1 富余产能应投入 A1/A2 设计与 C-LIVE |
| **M2** | 2026-09 | A3–A6 落地 + B0 事实基线 + B1 起步 + D1/D2 起步 | Laguna 零回归是硬门禁 |
| **M3** | 2026-10 | B1 验收 + B2 服务化 + D1/D2/D3-impl 交付 + C1/C2 | 一条命令能起服务 |
| **M4** | 2026-11 | B3 性能与 MTP + G0/G1 | 25B-A3B 正确性对齐 |
| **M5** | 2026-12 | G2/G3 服务化 + D4/D5/D6 收口 + E3 客户端矩阵 | 三个模型系列同一套配置流程 |
| **M6** | 2027-01 | C5/C6 + Track H 全项 | 24h soak + 发布 checklist |

**M1 的实际余量**：Track 0 只剩约 1 周的收尾，比 roadmap 编制时预估的"0.5 个月"少得多。
建议把释放出的产能全部投入 **Track A 的设计定稿**（A1/A2），因为它是唯一的长串行链起点。

---

## 12. 阻塞与依赖速查

| 被卡的事 | 卡它的 | 状态 |
|---|---|---|
| A6 零回归验收 | **D3 GPU CI 形态拍板** → C4 位精确门禁 | 🔴 未拍板 |
| B0 起步 | **D6 主线 checkpoint 拍板** | 🔴 未拍板 |
| Track B 全部 | Track A 完成（A1–A5） | 🔴 未开始 |
| B2 CUDA Graph | B0-5（GDN 是否 capture-safe） | 🔴 未验证 |
| B3 GDN 状态回滚 | B2 递归状态生命周期 | — |
| Track G 排期 | G0 拿到 config | 🔴 本地无 checkpoint |
| H1 发布 | **sparkinfer 上游化**（RK2） | 🔴 仍钉私有 fork |
| 一切重构的安全网 | **C-LIVE** | 🔴 本计划新增 |

---

## 13. 不做清单（不再重复讨论）

多卡 TP/PP/EP · 多机 · SM120 以外架构 · 视觉/多模态输入 · 训练/微调/LoRA 热加载 ·
AWQ/GPTQ/INT4 等其他量化 · 通用 HF 架构自动支持 · 复活 `oracle/qwen36_vllm/`。

理由见 [`roadmap.md`](roadmap.md) §3。

---

## 14. 配套文档

- [`roadmap.md`](roadmap.md) — 目标、理由、风险登记、待拍板事项（**权威**）
- [`architecture.md`](architecture.md) — 当前架构与目标架构
- [`model-support.md`](model-support.md) — 模型支持矩阵 + 接入新模型的操作指南
- [`diagnostics-guide.md`](diagnostics-guide.md) — bfdiag 使用指南（必读）
- [`../notes/2026-08-01-bfdiag-assertion-audit.md`](../notes/2026-08-01-bfdiag-assertion-audit.md) — C0 审计全文
- [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md) — fork 差异与 9 处未放宽 gate 的安全性分析
