# BlackweLLM 路线图（2026-08 → 2027-01）

> 编制日期：2026-08-01 · 基线 commit：`ce21eb5` · 本文档取代
> [`docs/archive/2026-07-26-roadmap-vllm-removal.md`](archive/2026-07-26-roadmap-vllm-removal.md)
>
> 本文档中所有"现状"数字均为 2026-08-01 在本仓库实测所得，来源在正文标注。
> 标注 **[待验证]** 的条目是尚未在本机跑过的假设，不作为决策依据，只作为待办。
>
> **2026-08-01 二次修订**（基线推进到 `6acc4ba`）：消化
> [`investigation-queue.md`](investigation-queue.md) §D 的上游调研结论，重排优先级；
> §7 的 D3（GPU CI 形态）、D6（Qwen3.6 主线 checkpoint）拍板，加上 N8
> （`--session-affinity`，见 [`implementation-plan.md`](implementation-plan.md) §6.1）拍板；
> 新增风险 RK9（冷启动/首次真实形状路径）。本轮另有三条 [待验证] 事项由并行 agent 在查
> （B0-8 GDN、KV dtype 选型、torch wheel 是否带 `sm_120`），本文档不预判其结论。

---

## 0. 定位变更（这是本次路线图重写的原因）

**旧定位**：只服务 `Laguna-S-2.1-NVFP4` 一个模型，把 SM120 上的极限性能榨干，
拿一个漂亮的数字去发布。

**新定位**：**Blackwell SM120 单机推理运行时**。硬件面收窄到极致（只有
SM120、只有单机），换取在这个窄面上把**稳定性、易用性、模型兼容性**做到
生产可用；性能从"主线目标"降级为"机会主义优化"。

变更依据：Laguna-S-2.1 的模型能力经评测后判断为一般（MMLU-Pro 84.5%，
STEM 强、人文弱），继续在它身上做深度优化的边际收益不足以支撑一次发布。
但为它建起来的这套东西——自研 SM120 执行栈、固定槽位调度、CUDA Graph
生命周期、前缀缓存、双协议 API、bfdiag 诊断平台——是**与模型无关的资产**，
值得围绕它重新组织目标。

### 收窄的硬件合同（不再讨论，作为公理）

| 维度 | 合同 | 含义 |
|---|---|---|
| GPU 架构 | **仅 SM120 / CC 12.0** | 不做 SM90/SM100/消费级以外的兼容；启动即检测，不匹配直接拒绝启动 |
| 拓扑 | **单机、单进程** | 无 TP / PP / EP / 多机；`world_size=1` 是硬编码前提，不是配置项 |
| 卡数 | **单卡优先，多卡不在本路线图内** | 96 GB 单卡是容量规划基准 |
| 权重精度 | **NVFP4 优先，FP8 次之** | SM120 无 BF16 tensor core，BF16 权重不是一等公民 |
| KV 精度 | **FP8 e4m3 优先** | |

这个合同的价值在于**它允许我们删代码**：任何为"未来可能的多卡/多架构"
保留的抽象，都应该被删掉，而不是留着长草。

### 北极星指标（按优先级）

1. **能跑起来**：拿到一个受支持的 checkpoint 路径，一条命令启动，不需要人肉算
   `blocks_per_slot`。
2. **不会崩**：24 小时连续压测无 slot wedge、无显存泄漏、无需重启。
3. **输出可信**：与参考实现（HF transformers / 上游框架）在贪心解码下 token 级对齐，
   有回归门禁看着。
4. **够快**：在上述三条成立的前提下，再谈 tok/s。

---

## 1. 现状盘点（2026-08-01 实测）

### 1.1 已经建成的（真资产）

| 资产 | 状态 | 证据 |
|---|---|---|
| vLLM 完全剥离 | ✅ 生产路径零 vLLM 依赖 | `runtime/model_loading.py` / `runtime/laguna_config.py` 自建；vLLM 仅存在于 `oracle/`，已排除出 wheel |
| 自研模型图 | ✅ Laguna 全栈自建 | `runtime/model/`（decoder / linear / embedding / attention 占位 / RoPE） |
| 固定槽位连续批处理 | ✅ | `server/engine.py`，独立引擎线程持有 CUDA context，asyncio 侧无锁 deque + pipe 唤醒 |
| CUDA Graph 生命周期 | ✅ decode / draft / verify 三类图 | `runtime/backends/laguna_cuda_graph.py`（1106 行）、`laguna_dflash_cudagraph.py` |
| 前缀缓存 | ✅ 内容寻址 + 引用计数 + LRU | `runtime/block_pool.py`；同槽 KV 复用 + SWA ring 重建 |
| DFlash 投机解码 | ✅ 接受率 96.3–100% | `runtime/backends/laguna_dflash.py`（1707 行） |
| OpenAI + Anthropic 双协议 | ✅ 含流式 / 工具调用 / logprobs | `server/formats/` |
| Prometheus 指标 | ✅ `blackwellm:*` 命名空间 | `server/metrics.py` |
| bfdiag 诊断平台 | ✅ 飞行记录仪 / run record / 可比性判定 / 热引擎 | `bfdiag/`，CLI `bf`，见 [`diagnostics-guide.md`](diagnostics-guide.md) |
| 自研 SM120 kernel | ✅ router（.cu）+ RoPE / RMSNorm / KV scatter（Triton） | `runtime/kernels/` |

**Laguna 当前性能**（2026-07-31 实测，**2026-08-01 在当前 SparkInfer fork HEAD 上复现确认**，
analytic decode 路径，无 TURBO）：

| 工作负载 | tok/s | 接受率 |
|---|---|---|
| fox-64K | 353–368 | 96.9% |
| fox-4K | 353–357 | 96.3–97.0% |
| galaxy-4K | 395–401 | 100% |
| code-4K | 341–359 | 97.8% |

> README 里的 222 / 267 tok/s 是旧数字，已在本次文档整理中更正。复现数据、过程、
> 以及过程中发现的两个诊断链路问题见
> [`../notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md`](../notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md)。

### 1.2 曾经是红灯的（Track 0 止血，2026-08-01 处理）

R1–R6、R8 已在 2026-08-01 的 Track 0 批次里解决，保留在表里是为了记录**问题的形状**——
下一次同类问题该往哪看。R7、R9 仍然开着。

| # | 问题 | 证据 | 状态 |
|---|---|---|---|
| R1 | **CI 是红的** | 原诊断是 `tests/test_swa_scratch_lifecycle.py` 等裸 `import torch` 导致 pytest 收集期 ImportError。**实测后修正：流水线根本没走到 pytest**——`ruff check .` 这一步就先红了（`benchmarks/quick_check.py` 自 `e6793bc` 起有 4 个未使用 import）。而且真正违反 CPU-only 契约的模块不是 3 个而是 **5 个**，其余 4 个是经 `bfdiag.workloads` 摸到 sparkinfer、需要 fastapi、`monkeypatch.setattr` 的字符串目标会真的 import sparkinfer 子模块——**只有真跑才暴露，grep 看不见** | ✅ 已解决 |
| R2 | **4 个测试失败** | 装了 torch 后 `926 passed, 4 failed`：3× `test_bfdiag_ring.py::TestVerifyOnlyTrace`（假 backend 缺 `block_size`/`device`，测试替身漂移，非生产 bug）；1× thinking 契约冲突 | ✅ 已解决 |
| R3 | **thinking 标签契约自相矛盾** | `d52a3b1` "Strip thinking tags from all API responses" 与断言"保留 think 标签"的测试同时存在于 main；该 commit 写着 "Tested: unit tests pass"，但那个测试当时就是红的 | ✅ 已定契约，见 §1.4 |
| R4 | **thinking 剥离逻辑有误伤风险** | 根因比"贪婪"更准确：两条正则**没有锚定**，把文本里任何位置的 `<think>`/`</think>` 都当成删除信号。`_ORPHAN_CLOSE_RE = r"\A.*?</think>"` 删掉任何 `</think>` 之前的全部内容 | ✅ 已解决 |
| R5 | **sparkinfer 的性能补丁不可复现** | 2026-07-31 的 gating 放宽**从未提交到任何分支**，工作区被清后丢失（所以按分支做 pickaxe 搜索找不到）。已从悬空提交 `1e306d7`/`ec8bb1eb` 恢复，rebase 到 upstream `3bd3a2e`，现为 `jieen1/sparkinfer` `origin/master` 的 `7a1d69d`/`0844a4f` | ✅ 已解决，见 [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md) |
| R6 | **torch 版本合同不一致** | `pyproject.toml` 钉 `torch==2.11.0`；实测环境 `2.13.0a0`；sparkinfer 要求 `>=2.12` | ✅ 钉 `torch==2.13.0` |
| R7 | **Qwen3.6 支持已被摘除** | `ff4d858` / `a9cb932` 把 Qwen3.6 + DirectModelRunner 整体移入 `oracle/qwen36_vllm/`（8370 行，仍依赖 vLLM）；`ServerEngine.__init__` 对 `backend != "laguna"` 直接抛 `ValueError` | 🔴 未动——不是"退化"是"截肢"，走 Track A/B 重新接入 |
| R8 | **文档全面过期** | `AGENTS.md` 指名的 4 个模块都已不存在；README 英文段说"Currently optimized for Qwen3.6-27B"，中文段说"当前生产模型为 Laguna-S-2.1" | ✅ 已解决 |
| R9 | **仓库卫生** | `server/engine.py.bak` / `.orig`、`runtime/backends/laguna.py.bak`、根目录 9 个 `*.log`、`build/`、21 个残留分支 / worktree | 🟡 部分——sparkinfer 侧的 `blackforge-main` 已删，仓库自身待清 |

**这一批的方法论教训**（值得比修复本身更认真地记住）：

- **三个分支各自全绿、合起来是红的。** api 分支在装了 fastapi 的环境里验证，ci 分支在没装的环境里验证，直到合并才暴露 `test_format_regression.py` 违反了它自己 docstring 声明的 CPU-only 契约。**并行分工必须配一次真实的合并验证**，否则每个分支的"绿"都是局部的。
- **诊断要跑，不能读。** R1 的原始诊断（我方）是错的——错在只看代码不看流水线实际死在哪一步。
- **提交信息里的 `Tested:` 是有约束力的。** `d52a3b1` 声称 "unit tests pass" 却带着红测试进了 main，直接制造了 R2/R3。

### 1.3 这一批新发现的问题（尚未解决）

| # | 问题 | 证据 | 归属 |
|---|---|---|---|
| N1 | **结构化输出是空壳** | `runtime/structured_output.py` 的 `GrammarState.apply_mask()` / `apply_mask_batch()` 在 `server/engine.py` 里**从未被调用**。`json_object` / `json_schema` 请求会被正常接受，但**完全不约束生成**——静默失败，客户端拿到的是普通文本 | Track E |
| N2 | **`stop` 序列完全未实现** | 两套协议都是 | Track E |
| N3 | **`seed` 语义可疑** | 每个 token 重新播种，而不是推进同一个 generator | Track E |
| N4 | **bfdiag 的隔离保证可能已经失效** | `bfdiag/checkpoint/state.py` 有一条 `"bug_found_not_fixed"` 手册条目 + 专门的回归测试，指向 `laguna.py:1647,1653` 的张量轴错误。但真实的 `reset_slot` 已被重写（现在 1945-1965），为前缀缓存保留而**完全不再清零 KV 内存**；那两个行号现在指向另一个函数。连带问题：`bfdiag/checkpoint/restore.py` 明确依赖 `reset_slot` 清掉 checkpoint 范围外的残留来保证恢复隔离性。那个回归测试仍然绿，因为它**从不调用真实函数**，只在合成张量上复现抽象的切片 bug 模式 | Track C（**优先**——诊断平台自己说谎比一般 bug 危险） |
| N5 | **Anthropic 侧拿不到规范形态的 reasoning** | 见 §1.4 | Track E |
| N6 | **全套件下的 flaky** | `test_bfdiag_record.py::test_cli_ls_labels_an_unfinished_record_running`。已缩窄：单文件 8/8 过；bfdiag 子集 + 12 路 CPU 满载 3/3 过；只在**全套件**且机器有 GPU 负载时出现（3/5）。排除了两个显而易见的猜测——标签逻辑是 `finished_at is None → "running"`，**与时间无关**，不是老化阈值；`default_store()`/`bfdiag_dir()` 每次调用都重读环境变量，不是缓存 store。结论：来自 `tests/test_bfdiag_*` 之外某个测试的跨测试副作用 | Track 0 收尾 |
| N7 | **`FakeEngineProvider.load` 与 Protocol 不符** | 没接 `EngineProvider` 声明、`LagunaEngineProvider` 实现了的 `on_stage` 参数。当前休眠（调用点没传），改了就炸 | Track C |

### 1.4 thinking / reasoning 契约（D1 已定案）

**契约**：`content` / `text` 永不包含 reasoning；OpenAI 侧走 `reasoning_content`（delta / message）；
`QSR_REASONING_MODE=expose|strip`，默认 `expose`。判定规则从"对最终文本跑正则"改成
**生成流上的锚定状态机**——只有当 `<think>` 是生成文本的第一个字符时才认定存在 reasoning 段，
`StreamProcessor` 是这条规则的唯一实现，非流式路径复用同一个状态机。

**Anthropic 侧是非标准的**，这是一个有据可查的取舍而非疏忽：`f13fd4a`（2026-07-22）记录了一次
真实生产事故——Claude Desktop 会校验 thinking block 的加密签名，伪造的 32 位十六进制签名被拒后，
客户端**静默丢弃后续所有 content block，包括 tool_use**，用户的工具选择返回 "(no content)"。
那次修复留下了明确指令：`Do NOT re-add thinking block emission without a valid signature source`。
签名是服务端加密产物，我们造不出来。所以 Anthropic 侧发的是非标准的
`reasoning_content_delta` 事件 + 顶层字段，而不是规范的 `thinking` content block。

**可推翻的条件**：拿到合法签名来源。在那之前不要"顺手改回规范形态"——那正是 `f13fd4a` 修掉的 bug。

### 1.5 结构性短板（不是 bug，是设计债）

| # | 短板 | 具体表现 |
|---|---|---|
| S1 | **模型是硬编码的，不是配置** | `ServerEngine.MODEL = "poolside/Laguna-S-2.1-NVFP4"`；`BACKEND = "laguna"` 且拒绝其他值；`server/app.py` 里 `SERVER_MODEL_BACKEND = "laguna"` |
| S2 | **没有 backend 协议** | `LagunaBackend` 有 50+ 公开方法，`ServerEngine` 直接调用，没有任何接口约束。加第二个模型时无从下手 |
| S3 | **ModelSpec 是空壳** | `runtime/model_spec.py` 88 行，只有层名列表和 MTP 开关；不描述层类型序列、RoPE 类型、量化格式、MLP 类型 |
| S4 | **只有一类缓存** | `block_pool.py` 只管 paged KV；GDN/SSM 递归状态的挂钩（`evict_gdn_checkpoint` 等）是 Qwen3.6 时代留下的**残迹**，当前 Laguna 无 GDN 层，这条路径没有活代码 |
| S5 | **加载器只认一种量化格式** | 只支持 compressed-tensors（Laguna）；Qwen3.6 NVFP4 是 modelopt 格式 |
| S6 | **router kernel 写死 Laguna** | `runtime/laguna_router.py`：`EXPERTS = 256`、`TOP_K = 10` 是模块级常量 |
| S7 | **容量配置要人肉算** | 启动要同时设对 `QSR_SERVER_CAPACITY` / `NUM_SLOTS` / `BLOCKS_PER_SLOT` / `PRODUCTION`，四者有耦合约束，算错就是 OOM 或白白浪费显存 |
| S8 | **采样与投机互斥** | `temperature > 0` 直接退化成无投机自回归解码；只有贪心走完整 MTP 流水线 |
| S9 | **benchmarks/ 已经失控** | 136 个脚本，绝大多数是一次性诊断残留（bfdiag 的存在就是为了取代它们，但旧脚本没清） |
| S10 | **环境变量前缀仍是 `QSR_`** | 产品叫 BlackweLLM，目录叫 `qwen-sm120-runtime`，变量叫 `QSR_`，三套命名 |

---

## 2. 目标模型清单

### 2.1 本路线图覆盖

| 模型 | 架构 | 优先级 | 备注 |
|---|---|---|---|
| `Laguna-S-2.1-NVFP4` | MoE + SWA/Full 注意力 | P0（保持不回归） | 现有唯一生产模型，是重构的**回归基准** |
| `Qwen3.6-27B`（NVFP4 / 文本版） | Hybrid GDN + Full 注意力，稠密 MLP | P1 | 本地已有 4 个 checkpoint 变体 |
| `Qwen3.6-25B-A3B` | **[待验证]** 推测为 Hybrid + MoE | P1 | 本地无 checkpoint，需先拉 config |
| 上述两者的衍生微调版 | 同上 | P2 | 只要 `config.json` 架构字段一致即应自动可用 |

### 2.2 Qwen3.6-27B 架构事实（读自本地 `nvidia/Qwen3.6-27B-NVFP4` 的 `config.json`）

```
architectures : Qwen3_5ForConditionalGeneration   (model_type: qwen3_5)
num_hidden_layers : 64  =  48 linear_attention  +  16 full_attention  (interval 4)
hidden_size       : 5120        intermediate_size : 17408  (稠密 MLP，非 MoE)
num_attention_heads : 24        num_key_value_heads : 4    (GQA group = 6)
head_dim          : 256         partial_rotary_factor : 0.25
attn_output_gate  : True        output_gate_type : swish
linear_*          : conv_kernel_dim 4 / key_head_dim 128 × 16 heads
                    / value_head_dim 128 × 48 heads / ssm dtype fp32
rope              : mrope interleaved, mrope_section [11,11,10], theta 1e7
mtp_num_hidden_layers : 1       max_position_embeddings : 262144
vocab_size        : 248320      quant_method : modelopt (NVFP4) + fp8 kv
vision_config     : 存在（多模态）— 本路线图只做文本版
```

### 2.3 与 Laguna 的差异矩阵（这就是工作量的来源）

| 维度 | Laguna-S-2.1 | Qwen3.6-27B | 差距 |
|---|---|---|---|
| 注意力层构成 | 48 层全是注意力（36 SWA + 12 Full） | 16 层 Full + **48 层 GDN 线性注意力** | 🔴 **GDN 目前 0% 覆盖** |
| head_dim | 128 | 256 | 🟡 sparkinfer planner 有 `head_dim>=256 & gqa_group<=8` 分支，**[待验证]** 实测 |
| FFN | MoE 256 专家 top-10 + 共享专家 | 稠密 SwiGLU | 🟢 更简单，但需要稠密 NVFP4 GEMM 路径 |
| 注意力输出 | 普通 | **门控输出**（swish gate） | 🟡 模型图改动 |
| RoPE | yarn，partial 0.5，分层不同 theta | mrope interleaved，partial 0.25 | 🟡 需要 RoPE 变体 |
| 滑窗 | sliding_window 512（SWA ring KV） | 无滑窗 | 🟢 可省掉整套 SWA ring 机制 |
| 量化格式 | compressed-tensors | **modelopt** | 🟡 加载器分支 |
| 投机解码 | DFlash（独立 draft 模型） | **MTP**（1 层，在 checkpoint 内） | 🔴 不同机制，且 GDN 状态回滚是难点 |
| 词表 | 100352 | 248320 | 🟢 |

**结论**：GDN（48/64 层）是 Qwen3.6 支持的**单点最大风险**。它不只是一个 kernel，
而是引入了**第二类缓存**——长度无关的递归状态（conv state + ssm state），
它要和固定槽位、前缀缓存、CUDA Graph、投机解码这四套已有机制全部对接。
`docs/archive/2026-07-30-architecture-two-tenant.md` §6.2 记录过当年 Qwen3.6 时代
对这个问题的解法，是可复用的先验。

---

## 3. 不做清单

明确写下来，是为了让"要不要顺手支持一下"这个问题不再重复出现。

| 不做 | 理由 |
|---|---|
| 多卡 TP / PP / EP | 硬件合同外；引入的抽象会污染整个执行栈 |
| 多机 | 同上 |
| SM120 以外的架构 | 项目的全部价值来自架构专用化 |
| 视觉 / 多模态输入 | Qwen3.6 有 vision tower，本路线图只做文本版；多模态是另一条产品线 |
| 训练 / 微调 / LoRA 热加载 | 纯推理运行时 |
| AWQ / GPTQ / INT4 等其他量化格式 | 除非某个目标模型只有这种格式；不做通用量化框架 |
| 通用 HF 架构自动支持 | 每个架构显式接入，宁可少而正确 |
| 把 `oracle/qwen36_vllm/` 复活 | 它依赖 vLLM，与"零 vLLM"合同冲突。Qwen3.6 走新抽象层重新接入，旧代码只作参考读物 |

---

## 4. 轨道与优先级

九条轨道（0、A–H），按优先级排列。轨道内部是有序的，轨道之间大量并行。

### Track 0 · 止血（P0，M1 内必须清零）

把 §1.2 的红灯全部解决。这是所有后续工作的前置——在一个 CI 红、
测试红、依赖不可复现的仓库上做架构重构，等于没有护栏。

- ✅ **T0-1 CI 恢复绿灯**。**两条路都走了**：保留 CPU-only job 作为契约守门人
  （5 个违规模块改 `pytest.importorskip`），另加一个装 CPU torch 的 job 扩大覆盖面。
  第二个 job 有个非显然的前提：光装 torch 不够——多个模块假定"有 torch 就有
  numpy/safetensors/huggingface_hub/transformers/triton"，这在完整 GPU 环境里成立，
  对裸 torch wheel 不成立，最小可用集是 `.[dev,serving]` + CPU torch wheel + triton。
  另外修掉 `benchmarks/quick_check.py` 的 4 个未使用 import——`ruff check .` 是 CI
  的第一步，它红着，流水线从来没走到 pytest。
- ✅ **T0-2 修 bfdiag_ring 失败**：确认是测试替身漂移而非生产 bug
  （`DFlashEngine.__init__` 合法地从 backend 取 `device`/`block_size`/`_draft_blocks_per_slot`，
  测试用 `object.__new__` 绕过了构造函数）。顺带审了全部假对象，`test_bf_attention.py` /
  `test_cudagraph_buffers.py` 的同类写法对得上真实类，无需改动。
- ✅ **T0-3 / T0-4 thinking 契约定稿并重做**：见 §1.4。
- ✅ **T0-5 sparkinfer 补丁**。**结论与原计划不同**：补丁不是"未上游"，是**从未提交、已丢失**。
  已从悬空提交恢复、rebase 到新 upstream、合入 `jieen1/sparkinfer` 的 `origin/master`
  （fork 归我们所有，`upstream` 才是另一个团队的——原计划把这两者搞混了）。
  启动期校验保留：`check_sparkinfer_analytic_decode_gate` 用真实生产形状去探活的 gate，
  探测到关闭报 **warning 而非 fatal**（性能显著下降但功能正常，不该拦住启动）。
- ✅ **T0-6 依赖版本合同统一**：`torch==2.13.0`；补上三个漏声明的直接依赖
  （`huggingface_hub` / `nvidia-cutlass-dsl` / `triton`）；`transformers` 从 `serving` 移到 `cuda`；
  sparkinfer 钉 `origin/master @ 0844a4f`；`runtime/preflight.py` 九项启动期校验，
  接在 `server/app.py:main()` 的 `uvicorn.run` 之前。
- 🔴 **T0-7 仓库卫生**：未做。删 `.bak` / `.orig` / 根目录日志；清理已合并分支与
  `.claude/worktrees/` 残留；`benchmarks/` 分流（保留的进 `benchmarks/`，
  一次性诊断残留删除或转为 `bf exec` 脚本）。
- 🔴 **T0-8 收尾**（本批新增）：N6 的全套件 flaky 定位；N7 的 Protocol 不符。

**体量**：约 0.5 个月，其中 T0-1～T0-6 已于 2026-08-01 完成。

### Track A · 模型抽象层（P0，M1→M2）

这是"支持更多模型"的全部前置条件，也是"易用性"的根。设计目标不是通用性，
是**让接入第 N 个模型的成本可预测**。

- **A1 `ModelSpec` 升级为架构描述**：从 checkpoint 的 `config.json` 解析出
  层类型序列、每层的注意力/线性注意力/MLP 类型、RoPE 配置、量化格式、
  MTP 配置、缓存需求（每层需要 paged KV 还是递归状态）。
  校验前置：不支持的架构在**加载前**报错，不是跑到一半 NaN。
- **A2 Backend 协议**：把 `LagunaBackend` 的公开面收敛成一个显式协议
  （prefill / chunked prefill / decode / decode_batch / reset_slot /
  prefix 匹配与回放 / spec-decode 生命周期 / CUDA Graph 捕获）。
  先用 Laguna 做唯一实现，**协议由现有实现倒推**，不预设未来。
- **A3 缓存资源抽象**：`block_pool` 从"KV 分页器"升级为"槽位资源管理器"，
  统一管理两类资源——分页 KV（长度相关）与递归状态（长度无关、每槽固定）。
  前缀缓存的驱逐必须对两类资源联动（这正是 §1.5-S4 里那些残迹当年要解决的问题）。
- **A4 加载器抽象**：compressed-tensors / modelopt 两套 NVFP4 布局的
  tensor 命名与 scale 语义分离成两个 adapter，公共部分（分片流式读取、
  参数全覆盖断言、KV scale post-load）保持不变。
- **A5 模型注册表 + 自动识别**：给定 checkpoint 路径 → 读 config → 匹配架构
  → 选 backend + loader + spec 策略。`ServerEngine` 不再有 `MODEL` 常量。
- **A6 Laguna 迁到新抽象，零回归**：门禁 = 贪心输出 bit-exact + 性能不低于
  基线 3%（bfdiag run record 对比，`bf diff` 判可比性）。

**体量**：约 1.5 个月。**风险**：这是一次动到核心执行路径的重构，
Laguna 的性能与位精确是硬约束，必须逐步切换而非一次性替换。

**2026-08-01 补充**（`investigation-queue.md` D-4，vLLM v0.26.0 "每 KV-cache group 选不同 attention
backend；滑窗作为显式 backend capability"）：这条**验证了 A1/A2 的设计方向，不改变它**——A1 本就按
层类型序列描述架构（full / sliding / linear-attention 逐层区分），Qwen3.6 的 16 full + 48 GDN 混合
正是这个设计要接住的形状。**唯一的具体补充**：`BackendCapabilities`（§3.5.3）目前只有五个布尔标志，
没有把"滑窗"显式建模成一等能力——当前 Laguna 的 SWA 是通过层类型隐式处理的。A1/A2 落地时应把滑窗
参数（窗口大小、per-layer 是否滑窗）提升成 `ModelSpec`/能力查询里可查询的字段，而不是留在模型图内部。
不新开条目，作为 A1/A2 实现时的一条设计备忘。

### Track B · Qwen3.6-27B 重建（P1，M2→M4，有参考实现）

**这不是"接入一个陌生模型"，是"重建一个曾经在 vLLM 上跑通、有实测数字的实现"。**
`oracle/qwen36_vllm/` 有 8047 行、11 个模块的参考代码；`docs/archive/2026-07-20-PROGRESS.md`
等处有当年的真实吞吐/接受率/质量/显存数字。完整的逐模块判定（可直接搬/需改写/已被取代/
应废弃，逐项标新位置）、验收基线的完整来源表、以及在今天 Track A 抽象上的重建设计，见
[`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md)——本节只摘要结论，细节与行号引用一律
以那份文档为准。

**2026-08-02 关键纠偏**（读完 `oracle/qwen36_vllm/backends/qwen36.py` 全部 2159 行后确认）：
**模型数学本身（GDN 层 forward、mrope-interleaved RoPE、`attn_output_gate`、稠密 SwiGLU
MLP、modelopt NVFP4 反量化）完全不在这份参考代码里**——它当年活在 vLLM pip 包自己的
`Qwen3_5ForConditionalGeneration`/`Qwen3_5MTP` 类里，从未被 vendor 进本仓库，`get_model()`
只是现场把它借来用。**oracle 里真正能复用的是编排层与状态管理层**：GDN checkpoint
快照/恢复（`gdn_state.py`，466 行，判定为最高价值文件）、GDN 状态×投机解码的行寻址方案
（`_ssm_spec_row`，**已经原样存在于 `runtime/block_pool.py:45-79`，未被删除，休眠等接线**）、
accept/reject 判定算法（**已经完全移植完成**，就是今天的 `runtime/mtp_accept.py`）、块哈希
前缀缓存骨架。这意味着 B1（模型图 + 正确性）比原计划更接近纯新写，B0/B2/B3（状态管理 /
CUDA Graph 编排 / 投机回滚机制）比原计划有更多现成参照——工作量没有减少，是**性质变了**。

分四段，每段有独立的验收，不允许"边写边猜"。

#### B0 · 事实基线（M2，约 2 周）

- ✅ **主线 checkpoint 已拍板（见 §7 D6）：官方 `nvidia/Qwen3.6-27B-NVFP4`**。本地其余变体：
  `unsloth/...`（compressed-tensors，不是 modelopt——量化格式必须逐 checkpoint 读，不能按架构推断）、
  `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`（留作交叉验证 baseline）、`morosystems/ThinkingCap-...`。
  衍生任务：写一个按 tensor 名前缀跳过 `vision.*` 的加载过滤器（333 个张量，一次写好可复用）；
  A1 的 `validate_text_only` 语义要跟着调整为"允许 `vision_config` 存在但断言零 vision 张量被加载"
  （见 RK8；`runtime/architecture.py:292-319` 已经是这个语义，非空文档承诺）。
- Tensor 清单与 modelopt scale 语义逐项确认（不猜命名）——`git show a9cb932^:runtime/model/nvfp4_linear.py`
  可取回一份当年针对 compressed-tensors 格式写的、几乎完工的 NVFP4 Linear 原型（权重侧张量操作
  `swizzle_blockscale` 等可直接搬），但参数命名要按 modelopt 真实 checkpoint 重新逐项确认，
  见 `qwen36-rebuild-spec.md` §1.9/§3.4。
- **[待验证]** sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 /
  page_size ∈ {64,128} / fp8 KV` 下的正确性与吞吐——planner 里已有对应分支，
  但没有本机实测记录。**验证时应同时检查 warmup/autotune 是否覆盖了这个真实形状**
  （见 RK9、`investigation-queue.md` C-1，同一类"首次真实形状才暴露代价"的问题）。
- GDN 方案选型三选一，**倾向已加强为"先 1，晚点再看 3"**：
  1. 依赖 `flash-linear-attention`（本地已有 v0.5.2；本轮新确认 `fla.ops.gated_delta_rule` 的
     `chunk`/`fused_recurrent` 两条路径本地均可 `import` 成功、且无需 `causal_conv1d`——但**从未在
     SM120 上实跑**，`investigation-queue.md` §F 记录的 Blackwell 相关 bug 全部是 B200/SM100，
     无 SM120 记录，"未验证"不等于"已知能跑"）；
  2. 从 `oracle/qwen36_vllm/` 的 vLLM 路径移植（**注：GDN 层 forward 本身不在这份代码里**，
     真正能移植的是 `gdn_state.py` 的状态管理，不是算子本身，见 `qwen36-rebuild-spec.md` §1.0）；
  3. 自研 Triton kernel。
  **[待验证，本轮新增数据支持]**：`notes/2026-07-22-a1a-gdn-profiling.md` 实测 GDN 48 层合计
  decode GPU 时间占比恒定在 **3.9%–5.1%**（4K 与 128K 上下文均如此，NVFP4 GEMM 才是主导，
  71.1%→53.7%）——GDN kernel 本身**不是**性能瓶颈，支持"先 1 拿正确性，profiling 说话后再决定
  要不要 3"这条既有建议，不改变它。
- ✅ **Qwen3.6 的 MTP 层是否带 GDN**（`investigation-queue.md` B-6）——**已确认：不带**。
  6 个本地 checkpoint 变体的 `mtp.*` 张量清一色 `self_attn.*`+`mlp.*`，零 `linear_attn.*`。
  **但这不消除 B3 的 GDN 回滚项**——vLLM 那条"draft models have no mamba layers"的注释指的是
  draft 模型自己的递归状态，草稿侧因此确实可以少做（不需要为 MTP 头单独管理 conv/ssm 状态或做
  eagle-shift 类操作），但**主模型的 48 个 GDN 层在 verify 时照样跑完整 64 层前向**，被拒 token
  照样污染了不可逆的递归状态更新，回滚问题原样保留在 B3，只是范围从"草稿+主模型两侧"收窄到
  "只有主模型侧"。详见 `notes/2026-08-01-b6-mtp-gdn-verification.md`；B3 不再需要按"带/不带 GDN"
  写两个分支，只有一个分支（见下）。
- **[待验证，另一 agent 在查，本文档不预判]** NVFP4 KV vs FP8 KV 在我们卡上的对比
  （`investigation-queue.md` C-2）：上游第三方在 RTX PRO 5000 上的数字（NVFP4 KV prefill 慢
  1.7–1.8×）不是我们的卡也不是我们的形状，只作参考，不作决定。
- 容量测算：64 层 / 256K 上下文 / 96 GB 下的 KV + 递归状态显存账，
  给出 context × 并发的可行域。**旧参照数字**（vLLM 执行路径下测的，不能直接当新框架的数字用，
  仅作方向参考）：128K/c=4/warm 约 90.7–92.9 GiB，64K/c=4/warm 约 63–65 GiB，256K/c=4/cold
  可行（82.8% 峰值，无 OOM），200K/c=4 两侧均不可行（>95GB）——完整来源见
  `qwen36-rebuild-spec.md` §2.4/§2.5。

**验收**：一份事实基线文档，把上述每项写成"实测值 + 复现命令"。

#### B1 · 正确性优先（M2→M3，约 1 个月）

刻意放弃所有性能：eager、batch=1、无 CUDA Graph、无投机、无前缀缓存。
**性质提醒**：以下五项里的 GDN 层/RoPE/MLP/门控/modelopt 加载**全部是新写代码**
（`oracle/qwen36_vllm/` 不含模型数学，只含编排层，见本节开头的纠偏），不要按"移植"的
工作量估计排期。

- GDN 层（conv1d state + gated delta rule + 输出门）
- Full attention 层（走 sparkinfer paged）
- 稠密 SwiGLU MLP（NVFP4）
- RoPE：partial_rotary_factor 0.25 + mrope-interleaved
- modelopt 权重加载
- 注意力输出门控

**验收门禁**：与 HF transformers 参考实现在贪心解码下**逐 token 对齐**
（至少 3 个工作负载 × 512 token）；逐层 logits 余弦相似度记录进 bfdiag。
**质量验收基线**（Qwen3.6-vLLM 时代实测，2026-07-21/22，完整来源见
`qwen36-rebuild-spec.md` §2.3）：MMLU-Pro **84.54%**（vs 官方 86.2，−1.7pp 噪声内）；
HumanEval **44.5%** / HumanEval+ **43.3%**（vs 同权重 stock vLLM 43.3%/42.7%，无退化）。
**⚠️ 这三个数字目前在 `docs/model-support.md:49` 里被错标成 Laguna 的当前质量数字**——
引用时以本节和 `README.md` 的历史数字表为准，不要拿 `model-support.md` 当独立确认，
详见 [`../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md)。

#### B2 · 服务化（M3，约 1 个月）

- 接入固定槽位调度 + 连续批处理
- 递归状态纳入槽位生命周期（reset / 复用 / 看门狗回收）——协调者设计（对应 A3）见
  `qwen36-rebuild-spec.md` §3.3，六条修改（不做统一分配器、两数字前缀匹配、投机保守释放、
  同轮不可跨请求借用、逐资源驱逐预算、块大小对齐）来自
  [`hybrid-cache-prior-art.md`](../notes/2026-08-01-hybrid-cache-prior-art.md)
- CUDA Graph 捕获（decode 路径；GDN 的状态更新是否 graph-safe 是关键 **[待验证]**——
  本轮确认这条**没有可抄的参照**：Laguna 的 decode 图从不触碰递归状态，其 warmup 复用天然
  安全，oracle 当年的解法是保留 `2×batch_size` 专用 warmup 槽（GDN 状态非幂等，不能用真实
  请求槽热身），这个设计要在自建 CUDA Graph 骨架上重新验证，不是照抄就对，见
  `qwen36-rebuild-spec.md` §3.5、§6.1 判定的**第一难点**）
- 前缀缓存（含递归状态 checkpoint 与 KV 块的联动驱逐——A3 的第一个真实用户，
  `BlockPool._on_evict_block` 挂钩已就位但值为 `None`，是否直接接线还是被协调者新设计取代
  留给 A3 落地时拍板）
- 并发 ≥ 2
- **前置条件（本批新增）**：`bfdiag/daemon/provider.py` 目前直接持有具体的 `LagunaBackend`/
  `DFlashEngine` 类型，B2 的验收依赖 bfdiag（run record / `bf diff`），需要 Track A 把
  `bfdiag` 的 provider 改成按协议持有，**应在 B2 开始前完成**，不是可以顺手拖到 B2 期间做的小事
  （见 `architecture.md` §3.5.4，`qwen36-rebuild-spec.md` §3.6）

**验收**：HTTP 端到端，OpenAI + Anthropic 双协议回归全绿；
与 B1 的 eager 路径贪心 bit-exact。

#### B3 · 性能与投机（M4，约 1 个月）

**只有一个分支**（B0 的 B-6 结论已定：MTP 头本身无 GDN，但主模型侧回滚问题原样保留，
不再按"带/不带 GDN"写两个分支）：

- MTP draft / verify（Qwen3.6 自带 1 层 MTP，草稿侧因头部无 GDN 而少一块状态管理），
  含**主模型 48 层 GDN 递归状态的推测回滚**——寻址方案（`_ssm_spec_row`）与 accept/reject
  判定算法已经现成可用（见本节开头），真正要重新解决的只是把这两者接到自建 CUDA Graph +
  自建模型图上，与 D-3（ReplaySSM Ring Spec-Verify）合并排期
- GDN kernel 调优或自研（依据 B0/B2 的 profiling；本轮数据显示 GDN 恒占 <5.1% decode 时间，
  优先级应低于 NVFP4 GEMM 与 attention 调优，见 B0）
- **KV dtype 待定**：[待验证]，取决于 B0 里 NVFP4 KV vs FP8 KV 的本机对比结果，不写死 "FP8 KV"
- 长上下文（128K / 256K）容量与吞吐

**验收**：接受率与吞吐进 bfdiag 基线；与上游框架同 prompt 同参数做 A/B。
**性能验收基线**（Qwen3.6-vLLM 时代实测终值，完整轨迹与噪声说明见
`qwen36-rebuild-spec.md` §2.1/§2.2）：

| 指标 | 历史基线（终值） | 配置 / 日期 |
|---|---|---|
| 吞吐（128K, c=4, warm） | **222.44 tok/s** | MTP K=3，2026-07-21，`PROGRESS.md:4239-4244` |
| 吞吐（64K, c=4, warm） | **236.69 tok/s**（更可信）/ 267 tok/s（较低置信度，仅架构文档回声） | 同上，2026-07-21 |
| MTP 接受率（128K, c=4, warm） | **50.3%**（约每轮 2 token） | 与 222.44 tok/s 同批测量，2026-07-21 |
| 256K, c=4（cold, chunked） | 1.557 tok/s，双方可行，82.8% 峰值显存 | 2026-07-19 |

**新实现打不平这些数字就是退步，但对比前必须先确认口径一致**——这些数字全部是在 vLLM
执行路径下测的（含 vLLM scheduler/ForwardContext 开销），新框架走 Track A 抽象后开销分布
不同，理论上有改善空间但不是承诺；且接受率在当年不同测量批次间本身有 3+pp 波动（含一次
已定位的计数 bug），不要用单次数字判定退步/进步，按仓库纪律先 `bf diff` 再比较。

### Track C · 稳定性（P1，贯穿 M1→M6）

不是一个阶段，是一条持续的轨道。核心思路：**把每一种失败都变成一个
有名字、有指标、有降级路径的已知状态**。

- **C0 诊断平台自身的可信度**（本批新增，**优先于本轨道其它条目**）：N4 揭示的问题是
  `bfdiag/checkpoint` 依赖一个已经不成立的前提（`reset_slot` 会清零 KV），而守护它的
  回归测试从不调用真实函数，所以一直是绿的。**一个会说谎的诊断平台比没有诊断平台更危险**——
  它让错误结论带着"有测试保证"的权重传播。要做的：审计 `bfdiag/` 里所有"对真实
  backend 行为的断言"，区分哪些真的在验证真实代码、哪些只是在合成数据上复现抽象模式；
  后者必须显式标注成"模式演示"而不是"回归门禁"。
- **C1 故障面清单**：显存不足、槽位卡死、CUDA Graph 捕获失败、kernel JIT 失败、
  长请求超时、客户端断连、tokenizer 边界、非法采样参数、并发抢占。
  每一项要有：检测点、指标、日志、用户可见错误、恢复动作。
- **C2 分级降级**：CUDA Graph → eager；投机 → 非投机；前缀缓存命中 → 冷 prefill。
  每级降级都要出指标（现在部分已有，需成体系）。
- **C3 看门狗覆盖**：已有 stale slot 回收，需要覆盖测试 + 故障注入。
- **C4 确定性与可复现**：per-request seed 已有；补 bit-exactness 回归门禁，
  纳入 CI。**D3 已拍板 (b)**：本地 pre-push 门禁 + 人工签核，落地机制（`make gate-local` +
  PR 签核勾选项）见 [`implementation-plan.md`](implementation-plan.md) §7.3。
- **C5 长稳测试**：24h soak，监控显存碎片、host 内存、槽位分布、指标漂移。
- **C6 崩溃可诊断**：进程级异常要留下 bfdiag run record，不是只有一行 traceback。
- **C7 冷启动 / 首次真实形状路径审计**（本批新增，见 RK9）：`235f51e` 修的是"每个未见过的
  page-table 宽度都触发 30–100s JIT 重编译"，但修复自己的提交记录留了一条明确未闭合的口子——
  DFlash 的 eager verify 回退路径不在启动期预热覆盖范围内，而且 CUDA Graph 捕获**成功**的
  可观测性目前是 0（只有失败会显式可见）。这不是孤立 bug，是"首次遇到真实形状/真实路径的代价被
  系统性低估"这一模式的又一个实例——`investigation-queue.md` C-1（sparkinfer warmup/autotune
  是否用真实形状）与 B0-3 的验证范围都属于同一类别。详细任务拆解见
  [`implementation-plan.md`](implementation-plan.md) §7.3/C7。

### Track D · 易用性（P1，M2→M5）

现状是"必须读源码才能正确启动"。目标是"读一页文档就能上线"。

- **D1 单命令启动**：`blackwellm serve <model-path-or-id>`，自动推导槽位与块数。
- **D2 显存规划器**：给定「模型 + 目标上下文 + 目标并发」算出配置并校验；
  或给定显存反推可行域。这直接消灭 §1.5-S7 那个四变量耦合陷阱。
- **D3 启动前置检查**：SM120 检测、显存、CUDA / driver、sparkinfer 版本、
  checkpoint 完整性、架构是否受支持——**全部在加载权重之前**，
  失败给出人能读懂的、带修复建议的错误。
- **D4 配置文件**：YAML 配置取代十几个环境变量（环境变量保留为覆盖手段）。
- **D5 命名统一**：`QSR_` → `BWLLM_`，带一个版本的兼容期与弃用警告；
  包目录 `qwen-sm120-runtime` → `blackwellm` 的重命名时机需要拍板（见 D4）。
- **D6 文档三件套**：安装部署、配置调参、故障排查。

### Track E · 兼容性（P2，M3→M6）

- **E1 API 面补齐审计**：2026-08-01 已做了第一轮，结果比预期差，**这几条现在是
  已确认的功能缺口而不是待核查项**（见 §1.3）：
  - **N1 结构化输出是空壳**——`json_object` / `json_schema` 被接受但完全不约束生成，
    静默失败。这是最严重的一条：客户端以为拿到了 JSON 保证，实际没有。
    优先级应高于本轨道其它条目。
  - **N2 `stop` 序列完全未实现**（两套协议）。
  - **N3 `seed` 每 token 重新播种**而非推进同一个 generator。
  - 错误码语义已修（FastAPI 曾把错误体双重包进 `{"detail": ...}`，两套协议的
    规范形状都不匹配）。
  - 仍待核查：`n>1`、usage token 统计准确性。
- **E2 采样 + 投机共存**（消灭 §1.5-S8）：`temperature>0` 时的投机验证
  （拒绝采样 / typical acceptance），这是当前最明显的功能缺口。
- **E3 客户端验证矩阵**：openai-python、anthropic-sdk、Claude Code、
  Cline / Roo、OpenWebUI、LiteLLM——每个跑一遍真实会话，
  把结果做成一张兼容性表放进文档。
- **E4 reasoning 内容的正确暴露**：OpenAI 的 `reasoning_content` /
  Anthropic 的 `thinking` block，而不是一刀切删除（与 T0-3 同一件事的下游）。
- **E5 chunked input-logprob 默认开启**（本批新增，`investigation-queue.md` D-8，来自
  SGLang v0.5.16）：削峰值显存，我们已有 logprobs 路径、双协议都暴露 `top_logprobs`。
  **小而自足，不依赖 Track A，也不用等 M3**——建议提前排进 M2，跟 Track F 的 F1 窗口扫测
  蹭同一个 GPU 验证窗口一起做（见 Track F）。

### Track F · 性能（P2，机会主义，M3→M6 —— 但两条例外见下）

**降级为机会主义轨道**：只在有明确 roofline 依据、且不损害 Track C/D 的
前提下做。已知的候选方向（来自 2026-07-31 的研究记录，以及本批 2026-08-01 消化的
`investigation-queue.md` §D 上游调研）：

**2026-08-01 本批新增两条例外，从"机会主义/P2"提升到 P1、排进 M2**——不是因为它们比
Track A 更急，是因为它们**不依赖 Track A、成本低，且直指本项目当前两个最硬的约束**
（吞吐上限、显存上限），有本机实测数据支撑，不是纯粹的"顺手试试"：

- **接受率 96.3–100% 但投机窗口固定**（D-2）：Laguna 的接受率实测 96.3–100%（§1.1，
  2026-07-31/08-01 复现），而 DFlash 的 `NUM_SPECULATIVE_TOKENS` 固定为 15——这个组合
  说明**限制吞吐的很可能是窗口本身，不是接受率上限**。第一步是便宜的：不实现自适应
  控制器（DSpark 风格），先把窗口从 15 静态调大，`bf diff` 测吞吐与接受率；只有静态调大
  见顶了才上置信度驱动的自适应窗口——vllm #49369 报告 DSpark 在某些负载上比不开投机还慢，
  不是白捡，必须按工作负载分别 A/B。
- **显存是硬约束，且比想象中紧**（D-3）：Laguna 权重 66.8 GB ≈ 67 GB（59.5 GB MoE +
  7.3 GB non-MoE，`notes/2026-07-29-gpu-memory-audit.md`），96 GB 卡上给 KV + 投机
  scratch 的预算很紧。协调者在本轮任务中的实时汇报：生产服务实测显存 94.2/97.9 GB
  （**98.8% 占用**）；2026-07-29 的静态审计（1 slot/131K 配置）测得的是 76.0/95.6 GB
  （79.5% 占用），两次配置不同、不能直接对比，但方向一致——**投机 scratch 在跟 KV 抢一块
  越来越紧的预算**。ReplaySSM Ring Spec-Verify 报告的 11.5 GB → 1.8 GB 是别人的卡、
  别人的形状，不能当我们的数字用，第一步是补一次带日期来源的本机审计，再判断这个技巧
  有多少能在我们自己的调度/scratch 复用层面拿到（我们做），有多少要动 sparkinfer 的
  kernel 内部（转 SparkInfer，写清楚交接，不直接改源码）。这条的结论应该喂给 Track A 的
  A3 协调者设计——投机 scratch 迟早要变成 A3 管理的资源类型之一。
- **调度纪律**：这两条不需要 Track A，可以现在做；但都需要真机 GPU 时间，**应优先蹭
  P0-E 第 5 步或 C-LIVE 的 GPU 窗口，不单独申请专用时段**——本机只有一块 GPU，任何需要
  GPU 的验收项天然串行（RK5），这也是 D3 选 (b) 而不是 (a) 的同一条理由。
- 完整任务拆解（F1/F2 的分步骤清单）见 [`implementation-plan.md`](implementation-plan.md) §7.6。

**已核实、从待办移除的一条**：`investigation-queue.md` D-5（hybrid SWA+full DFlash
drafter + 投机专用 `kv_cache_dtype`）读代码后发现**不完全对**——投机专用 `kv_cache_dtype`
已经是现状（draft KV cache 按 FP8/uint8 分配，与主模型该层自己的 dtype 选择独立）；
"hybrid drafter"这个 vLLM 新能力我们走的是另一条路（固定 6 层全 SWA 的专用小 draft
模型，KV cache 只有 0.007 GB），已经用不同手段达到类似的省显存效果。**这不是降级，
是核实后发现已经做到**，见 [`implementation-plan.md`](implementation-plan.md) §7.6。

其余仍是机会主义、不设强制时间表：

- TURBO_ATTN（FP8 QK MMA）的质量回归修复：per-head descale / Hadamard 旋转 /
  自适应 FP8-BF16 切换。收益 +6%，但 code 工作负载接受率会从 97.8% 掉到 58.6%，
  当前默认关闭。
- FA4 技法用于 prefill / extend 路径（TMA、persistent scheduler、FP8 softmax）。
  **T0 触发条件**（`investigation-queue.md` D-6，**保持观察，不要提前动**）：FlashAttention
  维护者已合入 sm120 PR（#2413，"WIP"），并有面向 5090 的 TMA + warp specialization PR
  在做（#2440，正是这里计划要移植的技法）。但 FA4 算法本体上不了 SM120（缺 tcgen05/TMEM）；
  当前 sm120 路径只有 FP16/BF16、`main` 上部分路径仍报错、在 5090 上比 FA2 **慢约 5%**。
  触发条件是"那批 PR 落到 main 且在 sm120 上跑赢 FA2"——到那时才从"自己移植"变成
  "评估采纳"。
- FP8 attention 的 `num_stages≥2`（SMEM 36 KB « 99 KB，有余量）。
- MoE 输出中心并行（Warp Decode 类方案），2–4 周量级，长期备选。
- GDN kernel 自研（依赖 Track B 的 profiling 结论）。
- **NVFP4 per-token online MoE 量化**（`investigation-queue.md` D-7，vLLM v0.26.0 +
  CuTe-DSL MXFP4）：Laguna 是 256 专家 NVFP4 MoE，直接可比。**kernel 形状 → 写清楚交给
  SparkInfer 团队评估，不直接改其源码**（按 `AGENTS.md` 规矩）——这条我们自己要做的部分
  只是写一份技术提案文档，不是实现。
- **sparkinfer 里还有 9 处未放宽的 gate**（本批发现）：`7a1d69d` 只放宽了 13 处中的 4 处
  （decode / prefill 的 analytic graph dispatch 谓词），其余 9 处——verify-graph 识别、
  各 CTA trait 选择分支、SWA budget、graph-replay 路径选择——是当初**刻意留下**的
  scope 限制，不是遗漏。逐项清单与安全性分析见
  [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md)。
  **动它之前必须知道的一条硬风险**：`planner.py` 的 grid occupancy 预算常量是按
  `num_kv_heads=4` 推导的，用到 8 不是放宽谓词就够，需要重新推导。

**纪律**：任何性能改动必须走 `bf diff` 判可比性 + 接受率与质量回归门禁，
2026-07-27 那次"两个不可比的接受率被当成打平证据、损失一整天"的教训写在
[`diagnostics-guide.md`](diagnostics-guide.md) 里。

### Track G · Qwen3.6-25B-A3B（P1，M4→M5）

前置：**先拿到 `config.json`**。当前本地无此 checkpoint，架构参数全部未知。

预期工作（**[待验证]**，以拿到 config 为准）：

- 若为 Hybrid GDN + MoE：GDN 复用 Track B 成果；MoE 走 sparkinfer `moe.fused_moe`
  （NVFP4 已支持），但 **router kernel 需要泛化**——现在
  `runtime/laguna_router.py` 的 `EXPERTS=256 / TOP_K=10` 是模块级常量。
- 若专家数 / top-k 与 Laguna 不同，需决定：泛化现有 kernel，还是为新形状
  再做一个特化。SM120 上 MoE 已被测定为带宽饱和，泛化的性能代价 **[待验证]**。

### Track H · 发布（P2，M5→M6）

- 发布门禁：CI 绿 + 长稳通过 + 两个模型系列的质量回归 + 文档三件套齐备 +
  依赖可从公开源安装（即 sparkinfer 上游化完成）。
- 素材纪律：只发实测数字，标注硬件 / 配置 / 复现命令；不做 apples-to-oranges 对比。
- 版本：`0.2.0` 作为"多模型 + 生产可用"的第一个公开版本。

---

## 5. 里程碑（月度体量）

| 里程碑 | 时间 | 交付 | 验收 |
|---|---|---|---|
| **M1** | 2026-08 | Track 0 全清；Track A 设计定稿；文档基线（本次） | CI 绿、测试全绿、依赖可复现、抽象层设计评审通过 |
| **M2** | 2026-09 | Track A 落地，Laguna 迁移零回归；Track B0 事实基线（含 B0-8 GDN-in-MTP 结论）；Track B1 起步；Track F 的 F1-1/F2-0 机会窗口测试（蹭 A6/C-LIVE 的 GPU 时段）；Track E 的 E5 | Laguna 贪心 bit-exact + 性能不低于基线 3%；Qwen3.6 事实基线文档 |
| **M3** | 2026-10 | Qwen3.6-27B B1 正确性验收 + B2 服务化；Track D 第一批（D1/D2/D3） | Qwen3.6 逐 token 对齐；HTTP 端到端双协议绿；一条命令能起服务 |
| **M4** | 2026-11 | Qwen3.6-27B B3 性能与 MTP；25B-A3B B0/B1 | 接受率与吞吐进基线；25B-A3B 正确性对齐 |
| **M5** | 2026-12 | 25B-A3B 服务化；Track D 收口（D4/D5/D6）；Track E 客户端矩阵 | 三个模型系列同一套配置流程；兼容性表全绿 |
| **M6** | 2027-01 | 长稳、发布门禁、`0.2.0` | 24h soak 通过；发布 checklist 全项 |

**并行度提示**：Track C（稳定性）和 Track F（性能）没有独立的里程碑，
它们是贯穿的；每个里程碑的验收里都含有对应条目。

---

## 6. 风险登记

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| RK1 | **GDN 是最大未知数**：kernel 性能未知、递归状态与投机/前缀缓存/CUDA Graph 的交互复杂 | 可能拖垮 M3/M4 | B0 阶段先做可行性验证再承诺时间；先用 FLA 拿正确性，把"性能"和"能跑"解耦 |
| RK2 | **sparkinfer 本地补丁未上游** | 发布阻塞；换机器复现不出性能 | T0-5 优先；在合入前用版本钉 + 启动校验让问题显性 |
| RK3 | **抽象层重构回归 Laguna 性能** | 唯一的生产模型退化 | A6 的 bit-exact + 性能门禁作为硬约束；逐步切换、每步可回滚 |
| RK4 | **25B-A3B 配置未知** | Track G 无法排期 | 尽早拉 config；在此之前 Track G 的时间是占位而非承诺 |
| RK5 | **单 GPU、无并行的开发环境** | 每次验证成本以分钟计，是迭代速度的硬上限；**2026-08-01 补充**：任何需要 GPU 的验收项（A6 bit-exact、C-LIVE、F1/F2 的实测、B0/B3 的 profiling）**天然串行，不能靠并行 agent 压缩工期**——本轮已出现多个并行 agent 同时想起服务的风险，协调者已加互斥锁 + 显存守卫应对。这条也是 D3 选 (b) 而不是 (a) 的直接支撑：单卡机器上一个自托管 runner 本身就是又一个要排队抢卡的进程，不会绕开这条串行约束，只会再制造一个抢卡方 | 严格执行 bfdiag 三条法则（不写一次性脚本、比数前先 `bf diff`、失败先读 trace）；GPU 验收任务按优先级排队，不并发申请 |
| RK6 | **依赖链漂移**（torch / cutlass-dsl / sparkinfer / transformers） | 静默变慢或变错 | T0-6 版本合同 + 启动期校验 + CI 锁定；`investigation-queue.md` C-3（PyTorch 2.13.0 wheel 是否带 `sm_120`）**另一 agent 在查，[待验证]，不预判**——若带，自编译要求终结，直接解这条风险 |
| RK7 | ~~**GPU CI 缺失**~~ | 位精确与性能门禁只能人工跑 | ✅ **2026-08-01 已拍板 (b)**：本地 pre-push 门禁 + 人工签核（理由见 §7 D3）。RK5 补充的"GPU 验收天然串行"是这个选择成立的前提——自托管 runner（选项 a）不解决串行问题，只是把它挪到另一个进程里，还多了排队开销。机制落地见 [`implementation-plan.md`](implementation-plan.md) §7.3/C4 |
| RK8 | **Qwen3.6 多模态字段** | 文本版 checkpoint 与多模态版共用架构名，加载器可能误判 | A1 的架构校验要显式拒绝带 vision tower 的权重，给明确错误。**2026-08-01 更新**：D6 拍板选了official `nvidia/Qwen3.6-27B-NVFP4`——这份 checkpoint **本身带 vision tower**，所以"拒绝带 vision tower 的权重"这条规则要改窄：不是"config 里出现 `vision_config` 就整体拒绝"，是"接受该 checkpoint，但要求 loader 显式处于 `language_model_only=True` 模式，断言零 vision 张量被实际加载"。这条留给 A1 落地时处理，不改 `architecture.md`，见 [`implementation-plan.md`](implementation-plan.md) §4/C-2 与 §7.1/B0-1b |
| RK9 | **冷启动/首次真实生产形状路径系统性覆盖不足**（本批新增，2026-08-01） | `235f51e` 修的是"每个未见过的 page-table 宽度都触发 30–100s JIT 重编译"，而这个修复自己的提交记录留了一条**尚未坐实的同类缺口**：DFlash 的 eager verify 回退路径（`mode="verify"`）不在启动期预热覆盖范围内，且 CUDA Graph 捕获**成功**的可观测性目前是 0（只有失败会打 warning，成功只打 info，默认日志配置下不可见）——这不是一次性 bug，是一个模式："首次遇到的真实形状/真实路径"这一类代价一直系统性地被低估，直到真机流量把它暴露出来。`investigation-queue.md` C-1（sparkinfer warmup/autotune 是否用真实形状，另一 agent 在查）与 B0-3（sparkinfer paged attention 的验证范围）都属于同一类别 | 见 [`implementation-plan.md`](implementation-plan.md) §7.3/C7：C7-1（DFlash verify 路径预热覆盖，需 GPU 复现）、C7-2（CUDA Graph 捕获成功可观测性，可蹭 P0-E 第 5 步零增量 GPU 成本）、C7-3（呼应 investigation-queue C-1，纳入 B0-3 的验证范围）|

---

## 7. 待拍板事项

这些是需要人做决定、不该由实现者顺手选一个的分叉点。

| # | 议题 | 选项 |
|---|---|---|
| ~~**D1**~~ | ~~thinking / reasoning 的产品契约~~ | ✅ **已定案 2026-08-01**：按协议暴露 + `QSR_REASONING_MODE` 开关。Anthropic 侧因签名不可伪造而采用非标准事件，理由与可推翻条件见 §1.4 |
| ~~**D2**~~ | ~~CI 与 torch 的关系~~ | ✅ **已定案 2026-08-01**：两条都要——CPU-only job 守契约，CPU-torch job 扩覆盖 |
| ~~**D3**~~ | ~~**GPU CI 形态**~~ | ✅ **已定案 2026-08-01：(b) 本地 pre-push 门禁 + 人工签核**。理由：这台机器只有一块 GPU（RK5），自托管 runner（选项 a）本身也要抢卡，不解决"GPU 验收天然串行"这条约束，只是换一个进程排队；而"一次验证以分钟计"正是本项目全部效率问题的根源（RK5），选项 (a) 会再制造一个抢卡方，不是治它。(c)（只在里程碑人工全量跑）门禁太松，位精确回归会在里程碑之间悄悄漂移而没人发现。落地机制（`make gate-local` + PR 签核勾选项）见 [`implementation-plan.md`](implementation-plan.md) §4/C-1、§7.3/C4 |
| **D4** | **重命名时机** | 包目录 `qwen-sm120-runtime` → `blackwellm`、环境变量 `QSR_` → `BWLLM_`：随 Track D 一起做，还是推到 `0.2.0` 发布前一次性做 |
| **D5** | **`oracle/qwen36_vllm/` 的处置** | (a) 保留为只读参考（当前）；(b) Track B 完成后整体删除；(c) 现在就删，需要时从 git 历史取 |
| ~~**D6**~~ | ~~**Qwen3.6 主线 checkpoint**~~ | ✅ **已定案 2026-08-01：官方 `nvidia/Qwen3.6-27B-NVFP4`**。前提已被更正——两份候选（官方版、社区版 `sakamakismile/...-Text-NVFP4-MTP`）都带 15 个 `mtp.*` 张量，真正的取舍不是"谁能投机"，是 **provenance vs 过滤 333 个 vision 张量**。排除 vision 张量是一次性机械工作（按 tensor 名前缀跳过 `vision.*`），衍生模型（微调版、下一代 Qwen）迟早都会带 vision tower，这个过滤器无论选哪份 checkpoint 都要写；反过来 provenance 不可逆，发布时官方来源比社区量化站得住。社区文本版留作交叉验证 baseline，不弃用。<br>**衍生影响**：这个决定要求 A1 的 `validate_text_only`（RK8）从"检测到 `vision_config` 就整体拒绝"改成"接受该 checkpoint，但要求 loader 处于 `language_model_only=True` 并断言零 vision 张量被实际加载"，见 RK8 与 [`implementation-plan.md`](implementation-plan.md) §7.1/B0-1 |
| **N8**（原属 [`implementation-plan.md`](implementation-plan.md) §6.1） | **`--session-affinity` 静默失效** | ✅ **已定案 2026-08-01：(c) 启动期拒绝该 flag**。它调的 `mtp_prefill_warm_continue` 只存在于已退役的 `oracle/qwen36_vllm/`，异常被 `try/except` 吞掉 → 每次静默回退冷 prefill、指标恒为 0、零测试覆盖——把静默降级变成显式失败。真要做 warm-continue，等 Track A 的能力查询（§3.5.3 `BackendCapabilities.warm_continue`）落地后再评估 (a) 才划算；现在做，协议地基未定，很可能要重写。落地清单见 [`implementation-plan.md`](implementation-plan.md) §6.1 |

---

## 8. 待验证清单

以下条目在本文档编制时**没有本机实测记录**，一律不作为决策依据。
它们是 Track B0 的主要内容。

- [ ] sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 / fp8 KV` 下的正确性与吞吐
- [ ] FLA `gated_delta_rule` 在 SM120 上能否跑通、速度如何（chunk 与 fused_recurrent 两条路径）
- [ ] GDN 递归状态更新是否 CUDA Graph capture-safe
- [ ] Qwen3.6 modelopt NVFP4 的 tensor 命名与 scale 语义逐项确认
- [ ] Qwen3.6-27B 在 96 GB 上的 context × 并发可行域（含递归状态显存）
- [ ] mrope-interleaved 在纯文本输入下能否退化为标准 1D RoPE
- [ ] `Qwen3.6-25B-A3B` 的 `config.json`（专家数 / top-k / 是否 hybrid / 是否带 MTP）
- [ ] sparkinfer `moe.fused_moe` 在非 256/top-10 形状下的可用性与性能
- [ ] 现有 4 个失败测试各自的"正确期望"是什么（尤其 thinking tag 那个）
- [ ] **（本批新增）** Qwen3.6 的 MTP 层是否带 GDN（`investigation-queue.md` B-6，另一 agent 在查）——
  决定 Track B3 走哪个分支
- [ ] **（本批新增）** NVFP4 KV vs FP8 KV 在我们卡上的 prefill/decode 对比（`investigation-queue.md`
  C-2，另一 agent 在查）——决定 B3 的 KV dtype 选型
- [ ] **（本批新增）** PyTorch 2.13.0 PyPI wheel 是否带 `sm_120`（`investigation-queue.md` C-3，
  另一 agent 在查）——影响 RK6 与 H1"可从公开源安装"
- [ ] **（本批新增）** 当前生产配置下的真实显存占用带日期来源的审计（Track F/F2-0）——协调者本轮
  实时汇报的"94.2/97.9 GB，98.8%"与 2026-07-29 静态审计的"76.0/95.6 GB，79.5%"配置不同、
  未经交叉验证，需要一次新的、注明并发/上下文配置的 bfdiag run record
- [ ] **（本批新增）** DFlash 的 eager verify 回退路径（`mode="verify"`）是否真的会在生产流量下
  被打到，以及 CUDA Graph 捕获成功的可观测性缺口（见 RK9 / `implementation-plan.md` §7.3/C7）
- [ ] **（本批新增）** `NUM_SPECULATIVE_TOKENS` 从 15 静态调大是否能在不损失接受率的前提下提升吞吐
  （Track F/F1-1）

---

## 9. 与本文档配套的其他文档

- [`architecture.md`](architecture.md) — 当前架构与目标架构
- [`model-support.md`](model-support.md) — 模型支持矩阵 + 接入新模型的操作指南
- [`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md) — Track B 重建规格：`oracle/qwen36_vllm/`
  逐模块判定与新位置映射、Qwen3.6-vLLM 时代验收基线、在 Track A 抽象上的重建设计、风险清单
- [`diagnostics-guide.md`](diagnostics-guide.md) — bfdiag 使用指南（仍然有效，必读）
- [`archive/README.md`](archive/README.md) — 已归档文档索引及归档原因
- [`../notes/README.md`](../notes/README.md) — 116 篇调查记录的分类索引
