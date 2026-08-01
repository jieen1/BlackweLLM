# BlackweLLM 路线图（2026-08 → 2027-01）

> 编制日期：2026-08-01 · 基线 commit：`ce21eb5` · 本文档取代
> [`docs/archive/2026-07-26-roadmap-vllm-removal.md`](archive/2026-07-26-roadmap-vllm-removal.md)
>
> 本文档中所有"现状"数字均为 2026-08-01 在本仓库实测所得，来源在正文标注。
> 标注 **[待验证]** 的条目是尚未在本机跑过的假设，不作为决策依据，只作为待办。

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

**Laguna 当前性能**（2026-07-31 实测，analytic decode 路径，无 TURBO）：

| 工作负载 | tok/s | 接受率 |
|---|---|---|
| fox-64K | 353–368 | 96.9% |
| fox-4K | 353–357 | 96.3–97.0% |
| galaxy-4K | 395–401 | 100% |
| code-4K | 341–359 | 97.8% |

> README 里的 222 / 267 tok/s 是旧数字，已在本次文档整理中更正。

### 1.2 现在是红灯的（必须先止血）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| R1 | **CI 是红的** | `tests/test_swa_scratch_lifecycle.py` 裸 `import torch`；CI 只装 `.[dev]`（无 torch）→ 收集期就 ImportError。`test_bfdiag_cold_capacity.py` / `test_bfdiag_ring.py` 同样问题 | "CPU-only 单元测试"这条护栏实际已失效 |
| R2 | **4 个测试失败** | 装了 torch 后：`926 passed, 4 failed` | 3× `test_bfdiag_ring.py::TestVerifyOnlyTrace`（fake backend 缺 `block_size`/`device`，fixture 过时）；1× `test_laguna_server_integration.py::test_laguna_chat_response_preserves_generated_think_tags` |
| R3 | **thinking 标签契约自相矛盾** | `d52a3b1` "Strip thinking tags from all API responses" 与断言"保留 think 标签"的测试同时存在于 main；该 commit 的 message 写着 "Tested: unit tests pass"，但那个测试当时就是红的 | 产品行为未定义 |
| R4 | **thinking 剥离逻辑有误伤风险** | `server/formats/thinking.py` 的 `_ORPHAN_CLOSE_RE = r"\A.*?</think>"` 会把响应里**任何** `</think>` 之前的全部内容删掉；`_UNCLOSED_THINK_RE = r"<think>.*\Z"` 会把 `<think>` 之后全部删掉 | 让模型"讲解 `<think>` 标签"这类请求会被静默截断——对代码模型是高频场景 |
| R5 | **依赖的 sparkinfer 是本地改过的** | 2026-07-31 的性能提升来自改 sparkinfer 的 gating 判定（`attention/paged/_forward.py`、`planner.py`），这些改动在 `/home/bot/project/sparkinfer` 本地，未上游 | 换一台机器 / 装 PyPI 版 sparkinfer 就复现不出当前性能；发布阻塞项 |
| R6 | **torch 版本合同不一致** | `pyproject.toml` 钉 `torch==2.11.0`；实际测试环境是 `2.13.0a0`；sparkinfer README 要求 `torch>=2.12` | 按 pyproject 装出来的环境跑不了 |
| R7 | **Qwen3.6 支持已被摘除** | `ff4d858` / `a9cb932` 把 Qwen3.6 + DirectModelRunner 整体移入 `oracle/qwen36_vllm/`（8370 行，仍依赖 vLLM）；`ServerEngine.__init__` 对 `backend != "laguna"` 直接抛 `ValueError` | 不是"退化"，是"截肢"——要恢复等于重写 |
| R8 | **文档全面过期** | `AGENTS.md` 指名的 4 个模块（`direct_model_runner.py` / `compat_vllm.py` / `metadata_builders.py` / `cuda_graphs.py`）都已不存在；README 英文段说"Currently optimized for Qwen3.6-27B"，中文段说"当前生产模型为 Laguna-S-2.1"，互相矛盾 | 本次整理已处理 |
| R9 | **仓库卫生** | `server/engine.py.bak` / `.orig`、`runtime/backends/laguna.py.bak`、根目录 9 个 `*.log`、`build/`、21 个残留分支 / worktree | |

### 1.3 结构性短板（不是 bug，是设计债）

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

七条轨道，按优先级排列。轨道内部是有序的，轨道之间大量并行。

### Track 0 · 止血（P0，M1 内必须清零）

把 §1.2 的红灯全部解决。这是所有后续工作的前置——在一个 CI 红、
测试红、依赖不可复现的仓库上做架构重构，等于没有护栏。

- **T0-1 CI 恢复绿灯**：裸 `import torch` 改 `pytest.importorskip`；或者反过来，
  承认 CI 需要 torch 并在 CI 装 CPU 版 torch。**需要拍板选哪条**。
- **T0-2 修 3 个 bfdiag_ring 失败**：fake backend fixture 补 `block_size` / `device`。
  顺带审一遍：还有多少测试用的是已经跟真实 backend 脱节的假对象。
- **T0-3 thinking / reasoning 契约定稿**（见 §7 待拍板 D1），然后让代码和测试一致。
- **T0-4 thinking 剥离逻辑重做**：从"正则删除"改为"基于生成状态机切分"——
  服务端知道 `<think>` 是不是模板注入的、知道 `</think>` 出现的位置，
  不该退化到对最终文本做贪婪正则。
- **T0-5 sparkinfer 本地改动上游化**：整理 2026-07-31 的 gating 放宽改动，
  写清楚交给 sparkinfer 团队（本仓库不直接改 sparkinfer 源码）；
  在上游合入前，用一个明确的版本钉子 + 启动期校验，让"跑在未打补丁的
  sparkinfer 上"变成一个响亮的错误而不是静默变慢。
- **T0-6 依赖版本合同统一**：torch / sparkinfer / nvidia-cutlass-dsl / transformers
  各钉一个实测通过的版本，写进 `pyproject.toml`，并在启动期校验。
- **T0-7 仓库卫生**：删 `.bak` / `.orig` / 根目录日志；清理已合并分支与
  `.claude/worktrees/` 残留；`benchmarks/` 分流（保留的进 `benchmarks/`，
  一次性诊断残留删除或转为 `bf exec` 脚本）。

**体量**：约 0.5 个月。

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
  前缀缓存的驱逐必须对两类资源联动（这正是 §1.3-S4 里那些残迹当年要解决的问题）。
- **A4 加载器抽象**：compressed-tensors / modelopt 两套 NVFP4 布局的
  tensor 命名与 scale 语义分离成两个 adapter，公共部分（分片流式读取、
  参数全覆盖断言、KV scale post-load）保持不变。
- **A5 模型注册表 + 自动识别**：给定 checkpoint 路径 → 读 config → 匹配架构
  → 选 backend + loader + spec 策略。`ServerEngine` 不再有 `MODEL` 常量。
- **A6 Laguna 迁到新抽象，零回归**：门禁 = 贪心输出 bit-exact + 性能不低于
  基线 3%（bfdiag run record 对比，`bf diff` 判可比性）。

**体量**：约 1.5 个月。**风险**：这是一次动到核心执行路径的重构，
Laguna 的性能与位精确是硬约束，必须逐步切换而非一次性替换。

### Track B · Qwen3.6-27B 接入（P1，M2→M4）

分四段，每段有独立的验收，不允许"边写边猜"。

#### B0 · 事实基线（M2，约 2 周）

- 本地已有变体清点与选型：`nvidia/Qwen3.6-27B-NVFP4`、`unsloth/...`、
  `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`（含 MTP 层，单文件）、
  `morosystems/ThinkingCap-...`。确定**主线 checkpoint**（倾向文本版 + 带 MTP）。
- Tensor 清单与 modelopt scale 语义逐项确认（不猜命名）。
- **[待验证]** sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 /
  page_size ∈ {64,128} / fp8 KV` 下的正确性与吞吐——planner 里已有对应分支，
  但没有本机实测记录。
- **[待验证]** GDN 方案选型三选一：
  1. 依赖 `flash-linear-attention`（本地已有 v0.5.2，`fla/ops/gated_delta_rule`
     有 chunk + fused_recurrent 两条 Triton 路径）；
  2. 从 `oracle/qwen36_vllm/` 的 vLLM 路径移植；
  3. 自研 Triton kernel。
  建议先 1 拿正确性，profiling 说话后再决定要不要 3。
- 容量测算：64 层 / 256K 上下文 / 96 GB 下的 KV + 递归状态显存账，
  给出 context × 并发的可行域。

**验收**：一份事实基线文档，把上述每项写成"实测值 + 复现命令"。

#### B1 · 正确性优先（M2→M3，约 1 个月）

刻意放弃所有性能：eager、batch=1、无 CUDA Graph、无投机、无前缀缓存。

- GDN 层（conv1d state + gated delta rule + 输出门）
- Full attention 层（走 sparkinfer paged）
- 稠密 SwiGLU MLP（NVFP4）
- RoPE：partial_rotary_factor 0.25 + mrope-interleaved
- modelopt 权重加载
- 注意力输出门控

**验收门禁**：与 HF transformers 参考实现在贪心解码下**逐 token 对齐**
（至少 3 个工作负载 × 512 token）；逐层 logits 余弦相似度记录进 bfdiag。

#### B2 · 服务化（M3，约 1 个月）

- 接入固定槽位调度 + 连续批处理
- 递归状态纳入槽位生命周期（reset / 复用 / 看门狗回收）
- CUDA Graph 捕获（decode 路径；GDN 的状态更新是否 graph-safe 是关键 **[待验证]**）
- 前缀缓存（含递归状态 checkpoint 与 KV 块的联动驱逐）
- 并发 ≥ 2

**验收**：HTTP 端到端，OpenAI + Anthropic 双协议回归全绿；
与 B1 的 eager 路径贪心 bit-exact。

#### B3 · 性能与投机（M4，约 1 个月）

- MTP draft / verify（Qwen3.6 自带 1 层 MTP），含**GDN 递归状态的推测回滚**
- GDN kernel 调优或自研（依据 B0/B2 的 profiling）
- FP8 KV
- 长上下文（128K / 256K）容量与吞吐

**验收**：接受率与吞吐进 bfdiag 基线；与上游框架同 prompt 同参数做 A/B。

### Track C · 稳定性（P1，贯穿 M1→M6）

不是一个阶段，是一条持续的轨道。核心思路：**把每一种失败都变成一个
有名字、有指标、有降级路径的已知状态**。

- **C1 故障面清单**：显存不足、槽位卡死、CUDA Graph 捕获失败、kernel JIT 失败、
  长请求超时、客户端断连、tokenizer 边界、非法采样参数、并发抢占。
  每一项要有：检测点、指标、日志、用户可见错误、恢复动作。
- **C2 分级降级**：CUDA Graph → eager；投机 → 非投机；前缀缓存命中 → 冷 prefill。
  每级降级都要出指标（现在部分已有，需成体系）。
- **C3 看门狗覆盖**：已有 stale slot 回收，需要覆盖测试 + 故障注入。
- **C4 确定性与可复现**：per-request seed 已有；补 bit-exactness 回归门禁，
  纳入 CI（GPU 侧需要一台可用机器，**需要拍板 GPU CI 怎么做**，见 D3）。
- **C5 长稳测试**：24h soak，监控显存碎片、host 内存、槽位分布、指标漂移。
- **C6 崩溃可诊断**：进程级异常要留下 bfdiag run record，不是只有一行 traceback。

### Track D · 易用性（P1，M2→M5）

现状是"必须读源码才能正确启动"。目标是"读一页文档就能上线"。

- **D1 单命令启动**：`blackwellm serve <model-path-or-id>`，自动推导槽位与块数。
- **D2 显存规划器**：给定「模型 + 目标上下文 + 目标并发」算出配置并校验；
  或给定显存反推可行域。这直接消灭 §1.3-S7 那个四变量耦合陷阱。
- **D3 启动前置检查**：SM120 检测、显存、CUDA / driver、sparkinfer 版本、
  checkpoint 完整性、架构是否受支持——**全部在加载权重之前**，
  失败给出人能读懂的、带修复建议的错误。
- **D4 配置文件**：YAML 配置取代十几个环境变量（环境变量保留为覆盖手段）。
- **D5 命名统一**：`QSR_` → `BWLLM_`，带一个版本的兼容期与弃用警告；
  包目录 `qwen-sm120-runtime` → `blackwellm` 的重命名时机需要拍板（见 D4）。
- **D6 文档三件套**：安装部署、配置调参、故障排查。

### Track E · 兼容性（P2，M3→M6）

- **E1 API 面补齐审计**：逐项核对 OpenAI / Anthropic 规范——
  `n>1`、`stop` 序列、`seed`、usage 统计准确性、错误码语义、
  structured output（`runtime/structured_output.py` 已有骨架，需验证是否真正生效）。
- **E2 采样 + 投机共存**（消灭 §1.3-S8）：`temperature>0` 时的投机验证
  （拒绝采样 / typical acceptance），这是当前最明显的功能缺口。
- **E3 客户端验证矩阵**：openai-python、anthropic-sdk、Claude Code、
  Cline / Roo、OpenWebUI、LiteLLM——每个跑一遍真实会话，
  把结果做成一张兼容性表放进文档。
- **E4 reasoning 内容的正确暴露**：OpenAI 的 `reasoning_content` /
  Anthropic 的 `thinking` block，而不是一刀切删除（与 T0-3 同一件事的下游）。

### Track F · 性能（P2，机会主义，M3→M6）

**降级为机会主义轨道**：只在有明确 roofline 依据、且不损害 Track C/D 的
前提下做。已知的候选方向（来自 2026-07-31 的研究记录）：

- TURBO_ATTN（FP8 QK MMA）的质量回归修复：per-head descale / Hadamard 旋转 /
  自适应 FP8-BF16 切换。收益 +6%，但 code 工作负载接受率会从 97.8% 掉到 58.6%，
  当前默认关闭。
- FA4 技法用于 prefill / extend 路径（TMA、persistent scheduler、FP8 softmax）。
- FP8 attention 的 `num_stages≥2`（SMEM 36 KB « 99 KB，有余量）。
- MoE 输出中心并行（Warp Decode 类方案），2–4 周量级，长期备选。
- GDN kernel 自研（依赖 Track B 的 profiling 结论）。

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
| **M2** | 2026-09 | Track A 落地，Laguna 迁移零回归；Track B0 事实基线；Track B1 起步 | Laguna 贪心 bit-exact + 性能不低于基线 3%；Qwen3.6 事实基线文档 |
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
| RK5 | **单 GPU、无并行的开发环境** | 每次验证成本以分钟计，是迭代速度的硬上限 | 严格执行 bfdiag 三条法则（不写一次性脚本、比数前先 `bf diff`、失败先读 trace） |
| RK6 | **依赖链漂移**（torch / cutlass-dsl / sparkinfer / transformers） | 静默变慢或变错 | T0-6 版本合同 + 启动期校验 + CI 锁定 |
| RK7 | **GPU CI 缺失** | 位精确与性能门禁只能人工跑 | 需要拍板（见 D3）：自托管 runner，还是本地 pre-push 钩子 + 人工签核 |
| RK8 | **Qwen3.6 多模态字段** | 文本版 checkpoint 与多模态版共用架构名，加载器可能误判 | A1 的架构校验要显式拒绝带 vision tower 的权重，给明确错误 |

---

## 7. 待拍板事项

这些是需要人做决定、不该由实现者顺手选一个的分叉点。

| # | 议题 | 选项 |
|---|---|---|
| **D1** | **thinking / reasoning 的产品契约** | (a) 服务端一律剥离（当前代码行为）；(b) 按协议暴露——OpenAI `reasoning_content` / Anthropic `thinking` block；(c) 由请求参数控制。影响 Track E4 与所有下游 agent 客户端的行为 |
| **D2** | **CI 与 torch 的关系** | (a) 坚持 CPU-only、所有 torch 测试 `importorskip`；(b) CI 装 CPU 版 torch，扩大可测面。当前是"声称 (a)、实际两者都不成立" |
| **D3** | **GPU CI 形态** | (a) 自托管 runner；(b) 本地 pre-push 门禁 + 人工签核；(c) 只在里程碑节点人工全量跑 |
| **D4** | **重命名时机** | 包目录 `qwen-sm120-runtime` → `blackwellm`、环境变量 `QSR_` → `BWLLM_`：随 Track D 一起做，还是推到 `0.2.0` 发布前一次性做 |
| **D5** | **`oracle/qwen36_vllm/` 的处置** | (a) 保留为只读参考（当前）；(b) Track B 完成后整体删除；(c) 现在就删，需要时从 git 历史取 |
| **D6** | **Qwen3.6 主线 checkpoint** | 带 MTP 的文本版 vs 官方 NVFP4 版——前者能做投机但来源是社区量化，后者官方但需另找 MTP 层 |

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

---

## 9. 与本文档配套的其他文档

- [`architecture.md`](architecture.md) — 当前架构与目标架构
- [`model-support.md`](model-support.md) — 模型支持矩阵 + 接入新模型的操作指南
- [`diagnostics-guide.md`](diagnostics-guide.md) — bfdiag 使用指南（仍然有效，必读）
- [`archive/README.md`](archive/README.md) — 已归档文档索引及归档原因
- [`../notes/README.md`](../notes/README.md) — 116 篇调查记录的分类索引
