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
- 额外的物理槽用于 CUDA Graph 预热（非 DFlash 路径 +1）。

**优点**：地址固定 → CUDA Graph 可以捕获整轮 decode；无块迁移 → 无碎片。
**代价**：`capacity` / `num_slots` / `blocks_per_slot` 三者耦合，配错就是
OOM 或者显存白扔——这正是 [`roadmap.md`](roadmap.md) Track D 要消灭的问题。

> ⚠️ **已知不一致**：`runtime/block_pool.py` 里 `RESERVED_PHYSICAL_SLOTS = 1`，
> `runtime/backends/laguna.py` 里同名常量 `= 0`，两处各有一份 `_physical_slot()`。
> 当前因为 Laguna 侧取 0 而没有实际分歧，但这是两套并存的槽位编址约定，
> 应在 Track A 的缓存抽象里合一。

### 2.4 前缀缓存

内容寻址：块级哈希 + 引用计数 + LRU 驱逐（`runtime/block_pool.py`）。
命中后走**同槽 KV 复用**——把命中的前缀 KV 留在原槽，只对超出前缀的部分
重新 prefill，并重建 SWA ring 窗口。

`block_pool.py` 里仍保留着 GDN checkpoint 的联动驱逐挂钩（`evict_gdn_checkpoint`
等），那是 Qwen3.6 时代的设计残迹：**当前 Laguna 没有任何 GDN 层，这条路径
没有活代码**。Track A 的缓存抽象会把它变回活的——因为 Qwen3.6 需要它。

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
| 单元测试 | ⚠️ **当前红**——见 [`roadmap.md`](roadmap.md) §1.2 R1/R2 |
| CI（push + PR） | ⚠️ **当前红**——CPU-only 契约已失效 |
| 位精确回归门禁 | 有脚本，无自动化（GPU CI 缺失，待拍板） |
| 性能回归门禁 | bfdiag run record + `bf diff`，人工触发 |

