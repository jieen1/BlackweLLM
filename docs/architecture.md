# BlackweLLM 架构：现状与目标

> 编制日期：2026-08-01 · 基线 commit：`ce21eb5` · 本文档取代
> [`docs/archive/2026-07-30-architecture-two-tenant.md`](archive/2026-07-30-architecture-two-tenant.md)
>
> 旧文档描述的是"vLLM 剥离进行中 + 双租户（Qwen3.6 / Laguna）"的中间态。
> 那个中间态已经结束：vLLM 剥离完成，Qwen3.6 租户被摘除。本文档描述
> **剥离完成后的真实现状**（第 2 章）与**支持多模型所需的目标架构**（第 3 章）。

---

## 1. 项目定位：一台专用推理机

BlackweLLM 不是通用推理框架。它的每一个设计决策都建立在一组**收窄到不能再窄
的硬件与部署合同**上，收窄本身就是它的价值来源。

| 维度 | 合同 |
|---|---|
| GPU 架构 | 仅 SM120 / CC 12.0（RTX PRO 6000 Blackwell、RTX 5090） |
| 拓扑 | 单机、单进程、单卡；`TP=PP=EP=1` 是编译期前提，不是运行期配置 |
| 权重 | NVFP4 优先，FP8 次之 |
| KV | FP8 e4m3 优先 |
| 并发 | 固定槽位，量级 1–8 |
| 模型 | 逐个显式接入，不做通用架构自动支持 |

**这条合同带来的直接后果**：SM120 没有 wgmma、没有 BF16 tensor core，
所以通用框架里那些为 SM90/SM100 写的快路径在这里全部落到慢路径。
自己写一条只走 SM120 的路，就是全部的性能来源。同样地，`world_size=1`
让整个分布式抽象层可以不存在——不是"简化"，是"删除"。

**这条合同带来的直接义务**：既然放弃了通用性，就必须在窄面上把
"能跑、不崩、结果对"做到通用框架的水准以上，否则收窄没有意义。
这就是 [`roadmap.md`](roadmap.md) 把稳定性/易用性提到性能之前的原因。

---

## 2. 现状架构（2026-08-01）

### 2.1 分层

```
                          HTTP 客户端
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  server/app.py — FastAPI                                  │
  │    /v1/chat/completions  /v1/completions  /v1/messages    │
  │    /v1/models  /health  /metrics  /debug/stats            │
  │    server/formats/ — openai · anthropic · stream          │
  │                      tools · thinking · content           │
  └───────────────────────────┬──────────────────────────────┘
                              │  asyncio 侧：无锁 deque + os.pipe() 唤醒
  ┌───────────────────────────▼──────────────────────────────┐
  │  server/engine.py — ServerEngine（独立引擎线程持 CUDA ctx）│
  │    准入 · 固定槽位分配 · 分块 prefill · 连续批处理         │
  │    前缀命中判定 · 会话亲和 · 看门狗 · 超时/取消            │
  │    ⚠ MODEL / BACKEND 是类常量，写死 Laguna                │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  runtime/backends/laguna.py — LagunaBackend（2461 行）    │
  │    槽位状态 · KV cache 拥有权 · attention metadata 构造   │
  │    SWA ring KV · CUDA Graph 生命周期 · MoE patch          │
  │    ├─ laguna_cuda_graph.py     decode / verify 图         │
  │    ├─ laguna_dflash.py         DFlash 投机引擎            │
  │    ├─ laguna_sparkinfer_attn.py  paged attention 适配     │
  │    ├─ laguna_sparkinfer_moe.py   fused MoE 适配           │
  │    └─ bf_attention.py          KV 写入 + attention 分派   │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  runtime/model/ — 自建模型图                              │
  │    laguna_model · laguna_decoder · laguna_dflash_model    │
  │    plain_linear · plain_embedding · plain_attention       │
  │  runtime/model_loading.py — 自建权重加载（流式 safetensors）│
  │  runtime/laguna_config.py  — 自建运行期配置                │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  kernel 层                                                │
  │    sparkinfer（外部）— paged attn · fused MoE · GEMM      │
  │    runtime/kernels/ — laguna_router_sm120.cu（自研）       │
  │                      nvfp4_gemm_sm120.cu                  │
  │                      rope / rms_norm / kv_scatter（Triton）│
  └───────────────────────────────────────────────────────────┘

  横切：bfdiag/（飞行记录仪 · run record · 可比性判定 · 热引擎守护进程）
        bfprobe/（探针）· server/metrics.py（Prometheus）
```

### 2.2 线程模型

一个进程，两个执行域：

- **asyncio 域**：FastAPI 处理 HTTP、解析协议、构造 `GenerationRequest`，
  推入 `collections.deque`（append/popleft 在 CPython 下是 GIL 原子的），
  再往 `os.pipe()` 写一个字节唤醒引擎线程。asyncio 侧**永不阻塞**。
- **引擎线程**：独占 CUDA context，跑 `_step_sync()` 循环——排空请求队列 →
  处理取消 → 回收过期保留槽 → 准入 → 分块 prefill → 一轮 decode/verify →
  提交 token → 回填 future / 流式通道。所有 GPU 操作都在这条线程上，
  因此不需要 CUDA 侧的锁。

结果回传通过 `loop.call_soon_threadsafe` 解析 asyncio future，
流式则通过 `StreamChannel` 推送。

### 2.3 固定槽位调度

不是通用 paged 调度器，是**固定槽位**：

- 启动时按 `capacity` 分配 N 个逻辑槽，每槽预留 `blocks_per_slot × block_size` 个 token 位。
- 一个请求占一个槽，从 prefill 到生成结束不迁移。
- 槽位不足则排队等待，不做抢占。
- CUDA Graph 预热槽按 backend 计算：Laguna 非 DFlash 路径需要 `+1`；Qwen
  和 DSV4 都在接客前用真实槽完成捕获并重置，因此服务层 `+0`。Qwen pool
  自己仍保留一行**逻辑** scratch 供 graph 安全路径使用：legacy 模式下它
  对应完整物理行；dynamic arena 下页表行地址固定但初始全指向 null，页面只在
  捕获/使用时按需借出，不再常驻一整行。动态持久前缀由 arena 的
  `CACHED_REF0` bundle 持有，也不再依赖固定 scratch 物理行。

**优点**：地址固定 → CUDA Graph 可以捕获整轮 decode；无块迁移 → 无碎片。
**代价**：`capacity` / `num_slots` / `blocks_per_slot` 三者耦合，配错就是
OOM 或者显存白扔——这正是 [`roadmap.md`](roadmap.md) Track D 要消灭的问题。

Qwen 另有 opt-in 的全局 page-bundle arena（`QSR_QWEN_KV_MODE`）：
`strict` 按所有可服务槽的最大上下文加 null/COW watermark 建池；`elastic`
用 `QSR_QWEN_KV_POOL_BYTES` 指定物理预算。两者目前都执行保守的
full-sequence reservation：slot 分配和 KV 预算必须同时成功才开始 prefill，
不足时请求留在 waiting queue；完成、取消、超时、异常和 reset 都释放未兑现的
tail reservation。`legacy` 仍是默认回滚路径，直到 4×256K GPU 矩阵完成。

> ⚠️ **已知不一致**：`runtime/block_pool.py` 里 `RESERVED_PHYSICAL_SLOTS = 1`，
> `runtime/backends/laguna.py` 里同名常量 `= 0`，两处各有一份 `_physical_slot()`。
> 当前因为 Laguna 侧取 0 而没有实际分歧，但这是两套并存的槽位编址约定，
> 应在 Track A 的缓存抽象里合一。

### 2.4 前缀缓存

⚠️ **本节此前描述错了现状，2026-08-02 更正。** 原文写的是"内容寻址：块级哈希 +
引用计数 + LRU 驱逐（`runtime/block_pool.py`）"——那是 `block_pool.py` 里实现的机制，
但**生产路径一行都没调用它**。

**实际生效的机制**是同槽 KV 复用：`LagunaBackend.find_best_slot_for_prompt()` +
`reconcile_prefix_hit()`（`runtime/backends/laguna.py:2179`/`:2222`）把命中的前缀 KV
留在原槽，只对超出前缀的部分重新 prefill，并重建 SWA ring 窗口。`server/app.py:1249`
的注释已经写明这一点（"LagunaBackend uses static block allocation … not a dynamic
BlockPool"），等于代码自己否认了本节原来的说法。

**`block_pool.py` 的真实状态**（518 行）：`BlockPool` / `hash_block_tokens` /
`FreeBlockQueue` **生产调用方为零**；`runtime/` 与 `server/` 里唯一的提及是上面那句
说明它没被用的注释。生产代码从该模块只 import 一个 `ChunkedPrefillState` dataclass
（`laguna.py:39`）。但它**有 44 个通过的测试**（`tests/test_block_pool.py` +
`tests/test_invariants.py`）。

这个组合对 §3.5.5 第 7 步（A3 缓存协调者）很重要：它既不是"可以直接建在上面的可靠
地基"，也不是"没人测过的死代码"，而是**测试充分但无人调用**——测试编码的是当初作者的
假设，不是今天的生产现实。第 7 步若要复用它，必须先用真实生产路径验证，不能拿测试
全绿当作它能工作的证据。

`block_pool.py` 里仍保留着 GDN checkpoint 的联动驱逐挂钩（`evict_gdn_checkpoint`
等）。**措辞更正（2026-08-02，决策点 1 拍板 (b)）**：此前本节和 `roadmap.md` S4 都把它
叫"残迹"，那个词暗示"待清理"，是错的。准确的说法是**休眠原语**——它当年为 Qwen3.6 写成、
经过验证、有测试，只是当前 Laguna 没有 GDN 层所以没有活代码路径。Track A 第 7 步会把它
变回活的，因为 Qwen3.6 正需要它。按"残迹"字面执行（删掉重写）等于丢弃已验证正确的代码，
纯为措辞一致而重新发明。

### 2.5 投机解码（DFlash）

Laguna 走 DFlash：一个独立的 draft 模型（`Laguna-S-2.1-DFlash-NVFP4`）
产生 K 个草稿 token，主模型一次 M=K+1 宽的 verify forward 批量验证。

- draft / verify 各有独立的 CUDA Graph，捕获在共享 scratch 上，
  并发时按槽位重新寻址后逐槽顺序 replay。
- 接受/拒绝判定在 `runtime/mtp_accept.py`（纯函数，可 CPU 测试）。
- 贪心（T=0）走完整投机流水线；`temperature > 0` **退化为无投机自回归**。

### 2.6 CUDA Graph

三类图：M=1 decode 图（按 batch 形状捕获）、DFlash draft 图、DFlash verify 图。
捕获时机在服务开始接客之前，捕获会把 dummy 数据写进某个槽的物理 KV 区间，
所以必须在任何真实请求拿到槽位之前完成，并在之后重置该槽。

### 2.7 依赖面

| 依赖 | 角色 | 状态 |
|---|---|---|
| `sparkinfer` | paged attention / fused MoE / blockscaled GEMM | **本地 editable 安装，且带未上游的 gating 补丁**（见 roadmap R5） |
| `torch` | 张量与 CUDA | pyproject 钉 2.11.0，实测环境 2.13.0a0，sparkinfer 要求 ≥2.12 —— **三者不一致** |
| `transformers` | tokenizer / AutoConfig（Laguna 需 `trust_remote_code`） | |
| `huggingface_hub` | 本地快照解析 | |
| `fastapi` / `uvicorn` | HTTP | |
| `vllm` | **仅 `oracle/`**，已排除出 wheel（`pyproject` 的 `packages.find` 不含 oracle） | 生产路径零依赖 |

### 2.8 现状的结构性短板

一句话概括：**这是一台为一个模型手工装配的机器，没有装配线**。

| 短板 | 位置 |
|---|---|
| 模型身份硬编码 | `ServerEngine.MODEL` / `.BACKEND`、`app.py:SERVER_MODEL_BACKEND` |
| 无 backend 接口 | `ServerEngine` 直接调 `LagunaBackend` 的 50+ 方法 |
| `ModelSpec` 不描述架构 | 只有层名列表 + MTP 开关，88 行 |
| 只有 KV 一类缓存 | 递归状态（GDN/SSM）无一等公民地位 |
| 加载器只认 compressed-tensors | modelopt 格式无路径 |
| router kernel 写死 256 专家 / top-10 | `runtime/laguna_router.py` 模块级常量 |
| 命名三套并存 | 产品 BlackweLLM / 目录 qwen-sm120-runtime / 变量 `QSR_` |

---

## 3. 目标架构：装配线

目标不是"通用化"，而是**让接入第 N 个模型的成本可预测**。判定标准：
接入一个新架构应该只需要写"模型描述 + 模型图 + 加载器 adapter"三样东西，
不需要碰调度器、不需要碰服务层、不需要碰缓存管理。

### 3.1 分层（目标）

```
  server/app.py               协议层  —— 与模型无关
        │
  server/engine.py            调度层  —— 与模型无关
        │                     准入 · 槽位 · 批处理 · 前缀 · 看门狗
        │                     通过 ModelBackend 协议访问执行层
        ├──────────────── ModelRegistry ────────────────┐
        │   checkpoint 路径 → 读 config.json            │
        │   → 匹配架构 → 选 (spec, backend, loader)     │
        │                                                │
  ┌─────▼──────────────────────────────────────────────┐│
  │  ModelBackend 协议（执行层接口）                    ││
  │    prefill / prefill_chunked_* / decode(_batch)     ││
  │    reset_slot / find_prefix_match / 前缀回放        ││
  │    投机生命周期 / CUDA Graph 捕获 / slot_state      ││
  └─────┬───────────────────────┬──────────────────────┘│
        │                       │                        │
  LagunaBackend           Qwen36Backend            （未来）│
        │                       │                        │
  ┌─────▼───────────────────────▼──────────────────────┐ │
  │  SlotResourceManager（缓存资源层）                  │ │
  │    分页 KV（长度相关）  +  递归状态（长度无关）      │ │
  │    引用计数 · LRU · 两类资源联动驱逐                │ │
  └─────┬──────────────────────────────────────────────┘ │
        │                                                 │
  ┌─────▼──────────────────────────────────────────────┐ │
  │  模型图 runtime/model/<arch>/                       │ │
  │  加载器 runtime/loading/<quant-format>.py ◄─────────┘ │
  └─────┬──────────────────────────────────────────────┘
        │
  kernel 层（sparkinfer + runtime/kernels/）—— 按能力而非按模型组织
```

### 3.2 五个关键抽象

#### A. `ModelSpec` —— 架构描述（不是层名列表）

从 checkpoint 的 `config.json` 解析成一个冻结的架构描述，
**它是加载前校验的唯一依据**：

- 层类型序列：每层是 full-attention / sliding-attention / linear-attention（GDN）
- 每层的 FFN 类型：dense SwiGLU / MoE（专家数、top-k、共享专家）
- 注意力参数：q/kv 头数、head_dim、是否输出门控、滑窗大小
- 线性注意力参数：conv kernel、key/value 头数与维度、状态 dtype
- RoPE：类型（default / yarn / mrope）、partial_rotary_factor、分层 theta
- 量化：格式（compressed-tensors / modelopt）、KV scheme
- 投机：无 / MTP（层数）/ 独立 draft 模型
- **缓存需求**：每层需要分页 KV、递归状态、还是两者都不要

不支持的架构在**加载权重之前**失败，给出具体到字段的错误
（例如"带 vision tower 的 checkpoint 不受支持，请使用文本版"）。

#### B. `ModelBackend` 协议 —— 执行层接口

由现有 `LagunaBackend` **倒推**得到，不预设未来。核心方法族：

| 方法族 | 内容 |
|---|---|
| prefill | `prefill` / `prefill_sampled` / `prefill_with_aux` / `prefill_chunked_begin` / `prefill_chunked_step` |
| decode | `decode` / `decode_sampled` / `decode_batch` / `decode_batch_sampled` |
| 槽位 | `reset_slot` / `slot_state` / `find_best_slot_for_prompt` |
| 前缀 | `find_prefix_match` / `reconcile_prefix_hit` / `prepare_exact_prefix_replay` / `continue_prefill_with_aux` |
| 投机 | `has_speculative_decode` / `enable_*` / `verify_and_commit_batch` |
| 图 | `capture_decode_cuda_graph` |

协议**不承诺**所有 backend 都实现全部能力：投机、前缀缓存、CUDA Graph
都是可选能力，通过能力查询暴露，调度层据此降级。这也正好是
Track C 的"分级降级"所需要的形状。

#### C. `SlotResourceManager` —— 两类缓存资源

> ⚠️ **2026-08-01 更正：不要"统一"分配器。** 读了 vLLM `0.25.1` 与 SGLang 的真实实现后，
> 两家都**刻意让两个分配器保持分离、在其上做协调**——SGLang 的 `MambaSlotAllocator`
> 明写 "we ... do NOT inherit the KV base class"。本节下面的"统一管理"措辞是误导，
> A3 应当是**协调者**而不是统一分配器。完整先例分析、以及另外五条会被踩中的坑
> （前缀搜索方向相反、块大小对齐、投机窗口的保守释放、同一轮内不可跨请求借用递归状态、
> 逐资源驱逐预算）见
> [`../notes/2026-08-01-hybrid-cache-prior-art.md`](../notes/2026-08-01-hybrid-cache-prior-art.md)。

当前 `block_pool` 只管分页 KV。目标是统一管理两类资源：

| 资源 | 特征 | 生命周期 |
|---|---|---|
| **分页 KV** | 大小随序列长度增长，块级共享 | 内容寻址 + 引用计数 + LRU |
| **递归状态**（conv + ssm） | 大小与序列长度**无关**，每槽固定 | 快照/恢复，与 KV 块**联动**驱逐 |

联动是关键：一个前缀的 KV 块被驱逐时，同一前缀边界上的递归状态快照
必须同步失效，否则会用一个前缀 A 的递归状态去续接前缀 B 的 KV，
这是那种"许多 token 之后才显形"的最难查的一类 bug。
`block_pool.py` 里现存的 `evict_gdn_checkpoint` 挂钩就是当年为此而设。

#### D. 加载器 adapter —— 按量化格式分层

公共部分不变（分片流式读取以约束 host 内存、参数全覆盖断言、
KV scale post-load）；差异部分收进 adapter：

| 格式 | 用于 | 差异点 |
|---|---|---|
| compressed-tensors | Laguna | `weight_packed` / `weight_scale` 命名、`config_groups` 语义 |
| modelopt | Qwen3.6 NVFP4 | `hf_quant_config.json`、global scale、不同的 KV scale 键 |

#### E. `ModelRegistry` —— 自动识别

`serve <checkpoint>` → 读 `config.json` → 匹配 `architectures` +
`model_type` + 量化格式 → 返回 `(ModelSpec, Backend 类, Loader adapter,
默认投机策略)`。`ServerEngine` 从此没有 `MODEL` 常量。

### 3.3 迁移不变量（重构过程中绝对不能变的）

在 Track A 把 Laguna 迁到新抽象的过程中，以下三条是硬约束，
任何一条破了就回滚：

1. **贪心输出 bit-exact**：同 prompt 同参数，token 序列与迁移前完全一致。
2. **性能不低于基线 3%**：以 bfdiag run record 为准，`bf diff` 判可比性。
3. **接受率不回归**：DFlash 接受率维持在 96%+ 区间。

### 3.4 目标架构下 Qwen3.6 需要新增的东西

| 组件 | 说明 |
|---|---|
| GDN 层实现 | conv1d state + gated delta rule + 输出门；kernel 来源待定（FLA / 移植 / 自研） |
| 递归状态资源类型 | 接入 `SlotResourceManager`（见 C） |
| 稠密 SwiGLU MLP | NVFP4 权重，走 blockscaled GEMM |
| 门控注意力输出 | `attn_output_gate: True` |
| RoPE 变体 | partial 0.25 + mrope-interleaved |
| modelopt 加载 adapter | 见 D |
| MTP 投机引擎 | 1 层 MTP，含递归状态回滚 |

**不需要新增**（可直接复用）：固定槽位调度、连续批处理、分页 KV、
CUDA Graph 框架、协议层、指标、bfdiag。
**可以不用**：SWA ring KV（Qwen3.6 无滑窗）、MoE 路径（27B 是稠密）、
router kernel（同上）。

---

## 3.5 实施规格（P0-D 定稿 · 2026-08-01）

§3.2 给的是形状，本节给的是**照着能写代码的那一层**。所有事实核实自
`main @ 619a09d`，行号会漂移，引用符号名优先。

### 3.5.1 协议的真实面：12 个成员，不是 50

`ServerEngine` 通过 `self.runner` 访问执行层，**全仓库唯一入口**。
`LagunaBackend` 有 50 个方法（其中公开 24 个 + 2 个 property），
但 `ServerEngine` 只用到 **12 个**——协议应当由这 12 个倒推，而不是照抄 24 个。

（B3，2026-08-03：曾经是 13，含 `enable_dflash`。第二个投机解码 backend
——`Qwen36Backend` + MTP——落地后发现 `enable_dflash` 是纯粹的 LOAD-TIME
装配调用，从未经过 `_step_sync` 那条常驻调度路径，且 qwen36 自己的等价物
`enable_mtp` 签名真的不同（多一个 `enable_resync` 关键字、返回 `None` 不是
`bool`）——不是换个名字就能套进同一个协议成员。已从 `CAPABILITY_MEMBERS`
移出；见 `runtime/backends/protocol.py` 里 `CAPABILITY_MEMBERS` 自己的
docstring。）

| 成员 | 调用次数 | 签名（`runtime/backends/laguna.py`） |
|---|---|---|
| `reset_slot` | 15 | `(slot: int) -> None` |
| `slot_state` | 6 | `(slot: int) -> LagunaSlotState` |
| `prefill` | 1 | `(slot: int, prompt_ids: list[int]) -> int` |
| `prefill_chunked_begin` | 1 | `(slots: list[int], prompts_per_slot: list[list[int]], chunk_size: int = 512) -> ChunkedPrefillState` |
| `prefill_chunked_step` | 1 | `(state: ChunkedPrefillState) -> bool` |
| `decode_batch_sampled` | 1 | `(slot_ids, token_ids, kv_lengths, params_list: list[SamplingParams], *, return_logprobs=False, top_logprobs=0) -> list[int] \| tuple[list[int], list[dict]]` |
| `find_best_slot_for_prompt` | 1 | `(token_ids: list[int], free_slots: list[int]) -> tuple[int, int]` |
| `reconcile_prefix_hit` | 1 | `(token_ids: list[int]) -> int` |
| `has_speculative_decode` | 1 | **`@property` → `bool`**（不是方法；`engine.py` 按值传给 `classify_decode_slots`） |
| `mtp_verify_and_commit_batch` | 1 | `(slots, anchors: dict[int,int], drafts: dict[int,list[int]], *, return_logprobs=False, top_logprobs=0) -> dict[int, dict]` |
| `capture_decode_cuda_graph` | 1 | `() -> int \| None` |
| **`mtp_prefill_warm_continue`** | 1 | ⚠️ **`LagunaBackend` 没有这个方法** —— 见 3.5.6 |

每个 backend 自己的"怎么打开投机解码"方法（`LagunaBackend.enable_dflash` /
`Qwen36Backend.enable_mtp`）保留各自诚实的名字和签名，不再是协议成员——
`_load_laguna_model`/`_load_qwen36_model` 本来就已经是分开的、各管各的
load-time 方法，这里只是把同一种不对称也应用到"怎么打开投机解码"这一步。

伴随的两个数据类型也属于协议面：

- `LagunaSlotState`（frozen）：`kv_len: int`、`committed_tokens: tuple[int, ...]`、
  `is_fresh` property。文档字符串已经写明它是"只读的服务端视图"——**这正是协议应有的形状，
  可以直接提升为 `SlotState`**，改名去掉 `Laguna` 前缀即可。
- `ChunkedPrefillState`：分块 prefill 的不透明句柄，`begin` 产出、`step` 消费。
  协议层应保持它不透明（`TypeVar` 绑定到具体 backend），不下沉字段。

### 3.5.2 观测层的只读状态视图（关掉 `/metrics` 那条故障类）

`server/app.py` 现在**直接读执行层内部**，三处：

| 位置 | 读的东西 | 问题 |
|---|---|---|
| `app.py:673` | `runner._prefix_cache_kv_len[i]` | **私有属性** |
| `app.py:676` | `runner._prefix_cache_tokens` | **私有属性** |
| `app.py:1182-1221` | `slot_kv_len`，经 `_slot_kv_len()` 容错读取 | 公开属性，但**形状假设**（list vs mapping）不属于任何契约 |

2026-08-01 当天 `/metrics` 两次 500 都出自这里：一次是聚合字典的键数不一致，
一次是把 list 当 mapping 读。`619a09d` 加的 `_slot_kv_len()` 容错helper 是正确的止血，
但它治的是症状——**根因是观测层没有一个属于自己的契约**。

**定稿**：协议提供一个只读快照方法，观测层只准读它：

```
SlotSnapshot   (frozen)  slot: int · kv_len: int · is_fresh: bool
PrefixSnapshot (frozen)  slot: int · cached_kv_len: int · cached_tokens: int
BackendSnapshot(frozen)  slots: tuple[SlotSnapshot, ...]
                         prefix: tuple[PrefixSnapshot, ...]
```

`snapshot() -> BackendSnapshot` 是协议成员。观测层拿到的是**值**，不是引用，
形状由协议保证，backend 换实现不会再把 `/metrics` 带下去。
`_slot_kv_len()` 容错helper 在切换完成后删除——它存在的理由随之消失。

### 3.5.3 能力查询：frozen dataclass，不是字符串

§3.2-B 说"投机 / 前缀缓存 / CUDA Graph 都是可选能力，通过能力查询暴露"，未定形状。

**定稿**：协议暴露 `capabilities` property，返回 frozen dataclass：

```
BackendCapabilities (frozen)
    speculative_decode: bool      # 决定 mtp_* 是否可调；backend 自己的开关方法
                                   # （enable_dflash / enable_mtp）不在协议面里
    prefix_cache:       bool      # 决定 reconcile_prefix_hit / find_best_slot 是否可调
    cuda_graph:         bool      # 决定 capture_decode_cuda_graph 是否可调
    chunked_prefill:    bool      # 决定 prefill_chunked_* 是否可调
    warm_continue:      bool      # 见 3.5.6
```

选它而不选 `supports("spec_decode")`：字符串拼错不会报错、无法静态检查、
IDE 补全不到；dataclass 三样都成立，且**可以直接序列化进 bfdiag run record 与 `/metrics`**，
让"这次运行到底开了什么"变成有记录的事实。这同时是 Track C2 分级降级的判定输入——
降级路径读的就是这五个布尔。

**约束**：能力为 `False` 时，对应方法族**不允许被调用**（协议不要求实现），
调度层在调用前查能力，而不是 `try/except AttributeError`。3.5.6 正是后者的代价。

### 3.5.4 第三个消费者：bfdiag 的耦合点

协议不只有 `ServerEngine` 一个消费者。`bfdiag/` 有 4 个模块直接 import 执行层，
其中**两处摸的是私有成员**：

| 模块 | 依赖 | 处置 |
|---|---|---|
| `bfdiag/daemon/provider.py` | `LagunaBackend`、`DFlashEngine` | 改为按协议持有，热引擎因此能托管任意 backend |
| `bfdiag/sensitivity/measure.py` | `LagunaBackend`、`DFlashEngine` | 同上 |
| `bfdiag/workloads.py` | **`_physical_slot`**、**`_ring_prefix_reuse_is_safe`**（私有）、`DRAFT_WINDOW` | 私有依赖必须显式化：要么提升为协议成员，要么在 bfdiag 侧独立复现并标注（`shapes/` 已有这个先例且是刻意的） |
| `bfdiag/shapes/__init__.py`、`bfdiag/divergence/thresholds.py` | `dflash_constants` 常量 | 常量依赖，风险低，保持 |

**这条不做的代价**：诊断平台会在重构中静默失效——正是 C0 审计（`notes/2026-08-01-bfdiag-assertion-audit.md`）
刚揭示过的那类问题，且当时的结论是"会说谎的诊断平台比没有诊断平台更危险"。

### 3.5.5 迁移顺序与回滚点

roadmap 只说"逐步切换而非一次性替换"，未给顺序。**定稿的原则是按爆炸半径从小到大，
把零行为变更的步骤排在前面，让最危险的一步在护栏最强的时候做。**

注意这与 roadmap 的 A1→A6 编号顺序**不同**：A3（`SlotResourceManager`）被移到最后。
理由是它touch 前缀缓存这套最微妙的机制，而在 Track B 的递归状态到来之前它**没有真实消费者**——
先做它等于承担最大风险换取零收益。

| # | 步骤 | 行为变更 | 门禁 | 回滚 | **需要 GPU？** |
|---|---|---|---|---|---|
| 1 | **A2-shadow**：定义 Protocol，静态+运行时断言 `LagunaBackend` 满足它，不改调用点 | 无 | 类型检查 + 一致性单测 | revert 单文件 | ❌ |
| 2 | **D-2 状态视图**：`snapshot()` 落地，`app.py` 三处私有读改走它，删 `_slot_kv_len` | 无（同值） | 单测 + **C-LIVE metrics 两条断言** | revert | ❌ 写；✅ C-LIVE |
| 3 | **A1 ModelSpec（影子模式）**：解析 Laguna 的 `config.json`，断言其结果与当前硬编码值逐字段相等，暂不驱动任何东西 | 无 | 影子一致性单测 | revert | ❌ |
| 4 | **A5 Registry（影子模式）**：路径 → `(spec, backend, loader)`，断言等于今天的硬编码选择 | 无 | 影子一致性单测 | revert | ❌ |
| 5 | **切换**：Registry 成为唯一真相源，删 `ServerEngine.MODEL` / `BACKEND` / `SERVER_MODEL_BACKEND` | **有** | **贪心 bit-exact** + C-LIVE | 回到第 4 步提交 | ✅ |
| 6 | **A4 加载器 adapter**：拆出 compressed-tensors adapter | 有（同权重） | 权重逐张量校验和相等 + bit-exact | 回到第 5 步提交 | ✅ |
| 7 | **A3 协调者**（**不是**统一分配器 —— 见 §3.2-C 更正）：两个独立分配器 + 一个持有不变量的协调者；前缀匹配返回 `(kv_hit, state_hit)`；逐资源驱逐预算。**动工前必读** [`hybrid-cache-prior-art`](../notes/2026-08-01-hybrid-cache-prior-art.md) | **有，爆炸半径最大** | bit-exact + 接受率 + 前缀命中率不回归 + C-LIVE | 回到第 6 步提交 | ✅ |
| 8 | **A6 验收** | — | 三条迁移不变量（§3.3）全过 | — | ✅ |

**关键结论：第 1–4 步完全不需要 GPU**（协议一致性、config 解析、注册表解析都是纯 CPU）。
在没有 GPU 的窗口里，Track A 可以一路推进到第 4 步结束——那已经是 A1/A2/A5 的主体。

### 3.5.6 N8：`--session-affinity` 当前 100% 失效（本次定稿中发现）

`server/engine.py:971` 调用 `self.runner.mtp_prefill_warm_continue(slot, prompt_ids, prior_len)`。
该方法**只存在于 `oracle/qwen36_vllm/`**（`backends/qwen36.py:1560`、`direct_model_runner.py:1991`），
是 Qwen3.6 被截肢时留下的残肢。`LagunaBackend` 没有它，也没有 `__getattr__` 转发（已核实）。

调用点被 `try/except Exception` 包着（`engine.py:972-978`），所以：

- 每次 warm-continue 尝试抛 `AttributeError` → 被吞 → 记一条 `warm-continue failed` 日志
  → 回退到 `reset_slot` + 重新排队 → `session_warm_fallbacks` +1；
- 输出**仍然正确**，只是永远拿不到 warm 续接的加速；
- `session_warm_continuations` 恒为 0；
- `tests/` 对 `warm_continue` / `session_warm` **零覆盖**（已核实）。

默认关闭（`engine.py:203` 默认 `False`，`QSR_SERVER_ENABLE_SESSION_AFFINITY` 默认 `"0"`），
但 `--session-affinity` 是**面向用户的文档化 CLI 开关**（`app.py:1116,1127`），
`benchmarks/server_e2e_check.py:98` 也会打开它。

**这正是 3.5.3 那条约束存在的理由**：用 `try/except` 兜底可选能力，
换来的是一个"看起来在工作、实际永远走降级路径"的功能。改用能力查询后，
这类问题在调用前就被拦住，而不是变成日志里的一行异常。

**三个选项，需要拍板（本文档不代选）**：

| 选项 | 含义 | 代价 |
|---|---|---|
| (a) 为 Laguna 实现 warm-continue | 恢复功能 | 需要设计 + GPU 验证；收益取决于会话场景占比 |
| (b) 删除该 flag 与相关代码路径 | 承认它不属于当前产品面 | 丢掉 P4b 那批已写好的调度逻辑 |
| (c) 启动期拒绝该 flag（能力查询为 `False` 时报错） | 最小改动，把静默降级变成显式失败 | 功能仍缺，但不再骗人 |

**倾向 (c) 作为立即动作**（零 GPU、可当天落地、消除"静默降级"），
(a)/(b) 作为 Track A 完成后基于能力查询重新评估的事项。

---

## 4. 观测与诊断

`bfdiag`（CLI `bf`）是这个仓库排查问题的**默认入口**，不是可选工具：
飞行记录仪常态开启、run record 留证据、`bf diff` 判两次运行可比性、
热引擎守护进程支持秒级迭代。

设计动机是硬约束：**这台机器只有一块 GPU、不能并行，一次真实验证以分钟计**，
所以唯一的效率杠杆是"每次 GPU 运行能榨出多少信息"。

完整用法见 [`diagnostics-guide.md`](diagnostics-guide.md)——**在写任何诊断代码之前读它**。
`bfdiag` 是纯标准库、被 `runtime/` 模块级导入，因此必须保持无第三方依赖、
无导入期副作用。

Prometheus 指标在 `blackwellm:*` 命名空间下覆盖三个维度：
速度（e2e 延迟、TTFT、每输出 token 时间、吞吐计数）、
稳定性（运行/等待请求数、按结束原因的成功计数、按状态码的错误计数、KV 利用率）、
准确性（投机 prefill 的 bootstrap 校验、前缀缓存命中率）。
逐指标参考见 [`../server/README.md`](../server/README.md)。

---

## 5. 工程护栏

| 护栏 | 状态 |
|---|---|
| `ruff check .` 全仓绿 | ✅ 强制 |
| `ruff format --check` 生产包 | ✅ 强制 |
| 单元测试（CPU-only） | ✅ 绿——CI 的契约守门人，无 torch 环境下必须零收集错误 |
| 单元测试（CPU torch） | ✅ 绿——扩大覆盖面的第二个 job |
| 启动期环境校验 | ✅ `runtime/preflight.py`，九项，接在模型加载之前 |
| CI（push + PR） | ✅ 绿（2026-08-01 恢复）。一个已知 flaky：见 [`roadmap.md`](roadmap.md) §1.3 N6 |
| 位精确回归门禁 | 有脚本，无自动化（GPU CI 缺失，待拍板 D3） |
| 性能回归门禁 | bfdiag run record + `bf diff`，人工触发 |
