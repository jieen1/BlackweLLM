# 模型支持矩阵与接入指南

> 编制日期：2026-08-01 · 最后更新：2026-08-21
>
> 这份文档回答两个问题：**现在支持哪些模型**，以及**接入一个新模型要做什么**。
> 排期见 [`roadmap.md`](roadmap.md)，架构设计见 [`architecture.md`](architecture.md)。

---

## 1. 支持矩阵

| 模型 | 状态 | 说明 |
|---|---|---|
| `poolside/Laguna-S-2.1-NVFP4` | ✅ **生产** | DFlash 投机、前缀缓存、CUDA Graph 全开 |
| `Qwen3.6-27B`（NVFP4 文本版，`unsloth/Qwen3.6-27B-NVFP4`） | ✅ **可服务**（2026-08-05） | 自研 `qwen36` 后端；MTP K=3 + MTP/decode CUDA Graph + 持久前缀缓存 + FP8 KV；质量基线已复现 |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` + DFlash2 | ✅ **原生接入，显式 opt-in**（2026-08-21） | SM120 原生 GGUF Q/K kernel、F32 full-attention、DFlash2 draft；target decode、draft、fixed/ragged verify CUDA Graph 均已实机捕获。服务默认改为 packed 权重 + 仅 `M≥32` transient BF16/cuBLAS prefill：同口径 4K fresh A/B warm TTFT `3.7755→1.1630 s`、wall `4.1234→1.5194 s`，decode 约 `89.63→87.73 tok/s`（测量噪声内持平），接受率 `28/31`、输出 SHA 不变；resident BF16 仍可显式回退（历史 `81.905 tok/s`），现有 NVFP4+DSpark 历史基线为 `232.045 tok/s`、`0.6365 s`。完整质量/长上下文/并发矩阵仍待独立门禁，见 [`notes/2026-08-20-qwen38-q6-dflash2-performance.md`](../notes/2026-08-20-qwen38-q6-dflash2-performance.md)、[`notes/2026-08-21-qwen38-q6-sglang-tc-blockm.md`](../notes/2026-08-21-qwen38-q6-sglang-tc-blockm.md) 与 [`notes/2026-08-21-qwen38-q6-transient-prefill.md`](../notes/2026-08-21-qwen38-q6-transient-prefill.md) |
| `Qwen3.6-25B-A3B` | 🔴 **计划中**（M4→M5） | 架构参数尚未确认 |
| 上述两者的社区微调 / 再量化衍生版 | 🔴 计划中 | 目标是 `config.json` 架构字段一致即自动可用 |
| 其他一切 | ❌ 不支持 | 见 [`roadmap.md`](roadmap.md) §3「不做清单」 |

### 硬件与格式合同（所有模型共同前提）

- GPU：SM120 / CC 12.0，单卡单机
- 权重：NVFP4 优先，FP8 次之
- KV：FP8 e4m3 优先
- 无 TP / PP / EP，无多机，无视觉多模态输入

---

## 2. Laguna-S-2.1（当前生产）

```
architectures  : LagunaForCausalLM        model_type: laguna
层数           : 48  =  36 sliding_attention  +  12 full_attention
hidden_size    : 3072      head_dim : 128
注意力头        : sliding 层 72 头 / full 层 48 头，num_key_value_heads 8
sliding_window : 512
FFN            : MoE 256 专家 top-10 + 共享专家（layer 0 为稠密）
                 moe_intermediate_size 1024 · routed_scaling 2.5 · per-head gating
RoPE           : full 层 yarn（theta 5e5, factor 32, partial 0.5）
                 sliding 层 default（theta 1e4, partial 1.0）
量化           : compressed-tensors · nvfp4-pack-quantized · fp8 KV scheme
词表 / 上下文   : 100352 / 262144
投机           : 独立 draft 模型 Laguna-S-2.1-DFlash-NVFP4
```

**当前性能**（2026-07-31 实测，2026-08-01 复现确认，analytic decode，无 TURBO）：
4K 约 353–401 tok/s、64K 约 353–368 tok/s、接受率 96.3–100%。

**质量**：⚠️ **无 Laguna 的评测数据。** 本仓库从未对 Laguna 跑过 MMLU-Pro /
HumanEval+ —— `evalplus_results/official/` 下三份结果的 `model` 字段都是
`qwen3.6`（2026-07-22）。此处原先引用的 84.54% / HumanEval+ 打平，是
**Qwen3.6-27B** 在已退役的 vLLM 执行路径上的成绩，被误标到了 Laguna 名下。
补测 Laguna 的质量基线已排为 Track C 的 C9。

---

## 3. Qwen3.6-27B（当前支持）

### 3.0 支持状态（2026-08-05）

`unsloth/Qwen3.6-27B-NVFP4` 已通过自研 `qwen36` 后端可服务（`server.app`，
生产路径零 vLLM）：MTP K=3、MTP anchor/draft/sync/verify 与 decode CUDA Graph
全部 capture 并在真实服务路径回放、persistent prefix cache、FP8 e4m3 KV。
质量基线在同参数下复现（MMLU-Pro 414 精确复现 84.54%；tool/agent/longctx
均为 1.000；HumanEval 768 在 ±3.9pp SE 内），证据见
[`notes/2026-08-05-qwen36-quality-rerun.md`](../notes/2026-08-05-qwen36-quality-rerun.md)。
一条命令启动 best 配置（3 × 256K 槽位）：

```bash
bash scripts/run_qwen36_quality.sh server start best
```

历史 vLLM 时代的实现仅保留在 `oracle/qwen36_vllm/` 作离线参考，不再可服务。

### 3.1 架构事实（读自本地 `nvidia/Qwen3.6-27B-NVFP4` 的 `config.json`）

```
architectures : Qwen3_5ForConditionalGeneration     model_type: qwen3_5
层数          : 64  =  48 linear_attention（GDN）  +  16 full_attention（interval 4）
hidden_size   : 5120        intermediate_size : 17408（稠密 SwiGLU，非 MoE）
注意力         : 24 q 头 / 4 kv 头（GQA group 6）· head_dim 256
                attn_output_gate: True · output_gate_type: swish
线性注意力     : conv_kernel_dim 4 · key 16 头 × 128 · value 48 头 × 128
                mamba_ssm_dtype: float32
RoPE          : mrope interleaved · mrope_section [11,11,10]
                theta 1e7 · partial_rotary_factor 0.25
量化           : modelopt · NVFP4 · kv_cache_scheme fp8
词表 / 上下文   : 248320 / 262144
投机           : mtp_num_hidden_layers = 1（MTP 在 checkpoint 内）
其他           : 含 vision_config（多模态）—— 本项目只做文本版
```

### 3.2 本地已缓存的候选 checkpoint

| 仓库 | 量化 | 分片 | 备注 |
|---|---|---|---|
| `nvidia/Qwen3.6-27B-NVFP4` | modelopt NVFP4 | 3 片 | 官方量化，含 vision |
| `unsloth/Qwen3.6-27B-NVFP4` | modelopt NVFP4 | — | 社区量化 |
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | modelopt NVFP4 | 单文件 | **文本版 + 含 MTP 层**，投机解码的候选主线 |
| `morosystems/ThinkingCap-Qwen3.6-27B-NVFP4` | modelopt NVFP4 | — | 微调衍生 |
| `Qwen/Qwen3.6-27B-FP8` | FP8 | — | 对照/回退 |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | AWQ | — | 格式不在支持范围 |

主线 checkpoint 已定为 `unsloth/Qwen3.6-27B-NVFP4`
（`scripts/run_qwen36_quality.sh` 的 `MODEL_SNAPSHOT`；2026-08-03 起
`resolve_checkpoint` 放行该仓库，质量与 serving 均以其为准）。

### 3.3 与 Laguna 的差异（= 工作量来源）

| 维度 | Laguna | Qwen3.6-27B | 需要做什么 |
|---|---|---|---|
| 层构成 | 48 层全注意力 | **48 层 GDN** + 16 层 Full | 🔴 GDN 全新，含第二类缓存 |
| head_dim | 128 | 256 | 🟡 验证 sparkinfer paged 支持 |
| FFN | MoE 256/top-10 | 稠密 SwiGLU | 🟢 更简单，需稠密 NVFP4 GEMM |
| 注意力输出 | 普通 | swish 门控 | 🟡 模型图 |
| RoPE | yarn，分层 theta | mrope interleaved，partial 0.25 | 🟡 RoPE 变体 |
| 滑窗 | 512（SWA ring KV） | 无 | 🟢 整套 SWA 机制可省 |
| 量化格式 | compressed-tensors | modelopt | 🟡 加载器 adapter |
| 投机 | 独立 DFlash draft | 内置 MTP 1 层 | 🔴 新引擎 + 递归状态回滚 |

---

## 4. Qwen3.6-25B-A3B（目标，参数待确认）

**当前本地没有该 checkpoint，架构参数全部未知。** 在拿到 `config.json`
之前，任何关于它的工作量估计都是猜测。

需要首先确认的字段：

- [ ] `architectures` / `model_type`（是否与 27B 同族）
- [ ] `layer_types`：是否同为 GDN + full attention 混合，比例如何
- [ ] MoE 参数：`num_experts` / `num_experts_per_tok` / 是否有共享专家 / `moe_intermediate_size`
- [ ] 量化格式与是否有 NVFP4 版本
- [ ] `mtp_num_hidden_layers`
- [ ] 是否含 vision tower

**已知的一处硬约束**：`runtime/laguna_router.py` 的 `EXPERTS = 256` /
`TOP_K = 10` 是模块级常量，自研 router kernel 只对 Laguna 的形状特化。
若 25B-A3B 的专家数/top-k 不同，需要在"泛化 kernel"与"再做一个特化"
之间做选择。SM120 上 MoE 已被测定为带宽饱和，泛化的性能代价待实测。

---

## 5. 接入一个新模型要做什么

> ⚠️ 下面描述的是 **Track A 抽象层落地之后**的目标流程。
> 在此之前（当前状态），接入新模型意味着重写 `ServerEngine` 和写一个
> 全新的 backend——这正是 Track A 存在的理由。

### 5.1 判定：这个模型在不在范围内

依次检查，任何一条不满足就**先停下来讨论**，不要开始写代码：

1. 有 NVFP4 或 FP8 权重吗？（没有 → 不在范围内）
2. 是纯文本模型，或有纯文本变体吗？（只有多模态 → 不在范围内）
3. 单卡 96 GB 装得下吗？（装不下 → 不在范围内，本项目不做多卡）
4. 它的层类型在已支持集合内吗？
   （full attention / sliding attention / GDN linear attention / dense SwiGLU / MoE）
5. 它的 RoPE 类型在已支持集合内吗？
6. 它的量化格式在已支持集合内吗？（compressed-tensors / modelopt）

第 4–6 条出现新类型是正常的，那就是这次接入的主要工作量，
但要**在开始前就把它识别出来并单独排期**，而不是写到一半才发现。

### 5.2 六步流程

| # | 步骤 | 产出 | 验收 |
|---|---|---|---|
| 1 | **事实基线** | 一份文档：config 全字段、tensor 清单与命名、量化 scale 语义、显存测算、与最接近的已支持模型的差异矩阵 | 每一项都是"实测值 + 复现命令"，没有"应该是" |
| 2 | **`ModelSpec` 描述** | 架构描述 + 加载前校验 | 拿一个不支持的 checkpoint 喂进去，在加载权重前报出具体到字段的错误 |
| 3 | **模型图 + 加载器** | `runtime/model/<arch>/` + 必要时新的量化 adapter | 权重全覆盖断言通过（没有任何参数没被 checkpoint 赋值） |
| 4 | **正确性优先跑通** | eager · batch=1 · 无图 · 无投机 · 无前缀缓存 | 与 HF transformers 参考实现贪心逐 token 对齐（≥3 个工作负载 × 512 token）；逐层 logits 余弦相似度进 bfdiag |
| 5 | **服务化** | 接入固定槽位调度、CUDA Graph、前缀缓存、并发 | HTTP 端到端双协议回归绿；与第 4 步的 eager 路径贪心 bit-exact |
| 6 | **性能与投机** | 投机引擎、kernel 调优、长上下文容量 | 吞吐与接受率进 bfdiag 基线；与上游框架同 prompt 同参数 A/B |

**顺序不可交换**。第 4 步之前不谈性能——历史上这个仓库最贵的几次教训
（block_size 128 接受率、fused_kv_scatter 的 value stride、
FP8 舍入平局）都是"性能改动跑在正确性门禁前面"造成的。

### 5.3 每步都必须遵守的诊断纪律

这台机器一块 GPU、不能并行、一次验证以分钟计。所以：

1. **不要再往 `benchmarks/` 里写一次性脚本**——那里已经有 136 个、
   零复利。实验投给热引擎：`bf exec <script>`。
2. **比较任何两个数字之前先 `bf diff`**——它会打印配置差异并拒绝把
   不可比的两次运行称为可比。
3. **失败时先读已有 trace，不要重跑**——飞行记录仪常态开启，
   失败那次的逐轮历史已经在盘上。

完整说明见 [`diagnostics-guide.md`](diagnostics-guide.md)。

### 5.4 温冷引擎的边界（用错会得到看起来合理的假数字）

| 用热引擎（`bf exec`）验证 | 必须冷启动新进程验证 |
|---|---|
| 稳态 decode 性能 | 冷 prefill 性能 |
| 接受率 | OOM / 显存压力上限 |
| 数值实验 | `block_size` / `capacity` / `gpu_memory_utilization` / `max_model_len` / 量化后端 |

右列全是**加载期**参数，模型载入时就固定了，热引擎里改它们不会生效，
但也**不会报错**——这是这个仓库里最容易产出假结论的一类操作。

---

## 6. 已知的跨模型陷阱

按历史教训整理，接入新模型时逐条对照：

| 陷阱 | 症状 | 参考 |
|---|---|---|
| 参数未被 checkpoint 赋值 | 随机初始化的权重，输出在很多 token 之后才劣化 | 加载器的全覆盖断言必须无豁免列表 |
| KV scale 未从 checkpoint 拷进运行期 buffer | 输出乱码 / 退化重复 / 接受率异常 | `runtime/model_loading.py:_apply_kv_cache_scale_post_load` |
| 页大小 / 块大小变更 | 接受率悄悄下降 | `notes/2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md` |
| KV scatter 的 stride 写错 | 长上下文才显形 | `notes/2026-07-27-fused-kv-scatter-value-stride-bug-ROOT-CAUSE-FOUND.md` |
| FP8 舍入平局 | 合成随机数据测不出，需穷尽式真实数据 + 精确有理数比较 | 同上目录 |
| CUDA Graph 捕获污染槽位 | 首个真实请求拿到脏 KV | 捕获必须在接客前完成并重置该槽 |
| 递归状态与 KV 块驱逐不同步 | 用前缀 A 的状态续接前缀 B 的 KV | 见 [`architecture.md`](architecture.md) §3.2-C |
| 温冷引擎混用 | 数字看起来合理但无意义 | 见 §5.4 |
