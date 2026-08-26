# Qwen3.8-Flash-Next（Qwen4 架构预览）day-0 接入调研

日期：2026-08-26（发布前数小时）。**纯调研，未写码。**

问题：Qwen 预告北京时间 2026-08-26 23:00（15:00 UTC）开源
`Qwen/Qwen3.8-Flash-Next`（125B-A6B 多模态 MoE，基于下一代 Qwen4 架构），
并同步发布 `Qwen3.8-Flash-Next-FP8`。官方明说这是**提前放出架构改进、让推理
运行时为 Qwen4 全家族做 day-0 准备**（HF 页 "A Preview of the Qwen4
Architecture"；HN 上 Qwen 社区也如此解读）。我们要评估：权重什么格式、
vLLM/SGLang 有无支持代码、我们的 runtime 要补什么。

## 1. 已核实事实（V）

- ModelScope pre-release API（`/api/v1/models/Qwen/Qwen3.8-Flash-Next`）：
  `PlannedArtifacts = [Qwen/Qwen3.8-Flash-Next, Qwen/Qwen3.8-Flash-Next-FP8]`，
  `ReleaseDate = 2026-08-26T15:00Z`，介绍仅一句："基于下一代 Qwen4 架构所构建的
  多模态 MoE 模型"。FP8 变体页面此刻尚无独立条目（API 返回获取失败）。
- HF 页：倒计时至 8 月 26 日，1572 人订阅；无任何文件（权重未上传）。
- 规格（社区核实，转引自 `JP-devv/humble-b70-llm` 的 pre-release 评估，原始来源
  r/LocalLLaMA `1vxwu4g` 与 NVIDIA DGX forum t/381228）：
  - 主模型 **125B**，激活 **~6B**；
  - 另有 **51B "Engram" n-gram 条件记忆参数，独立于 125B 之外**——即
    Kimi-Δ 式 "Conditional Memory via Scalable Lookup"（arxiv 2601.07372）
    的哈希查表嵌入，每 token 仅 KB 级读取；
  - 官方量化：BF16 base + **FP8**；社区 NVFP4（vcruz305）与 GGUF 阶梯的
    占位仓库已建、尚空；
  - unsloth 承诺 **llama.cpp day-0 支持**（reddit `1vxybmy`）。
- 同族参照物：`Qwen/Qwen3.8-2.4T-A95B`（config.json 已实拉）=
  `Qwen3_5MoeForCausalLM`，`model_type=qwen3_5_moe_text`：
  `layer_types` 为 3×linear_attention + 1×full_attention 循环
  （`full_attention_interval=4`），`head_dim=256`、`num_key_value_heads=4`、
  `attn_output_gate=true`（swish）、`partial_rotary_factor=0.25`、
  `num_experts=512`、`num_experts_per_tok=10`、shared expert、
  `mtp_num_hidden_layers=1`、`max_position_embeddings=262144`、
  `vocab_size=248320`。Flash-Next 大概率沿用这套几何（层数/宽度不同）。

## 2. 权重格式预判（U，待发布后 10 分钟内核实）

- 载体：safetensors（base BF16 + FP8 两份），config.json + index.json 标准 HF 布局；
  GGUF 由 unsloth 跟进。
- `architectures` 大概率是**新字符串**（`Qwen4ForCausalLM` /
  `Qwen3_8FlashNextForCausalLM` 一类；csghub-server 测试代码里已出现
  `"Qwen4ForCausalLM"` 占位，但无任何推理框架注册过它）。
- Engram 部分预期复用 LongCat-Flash 同源的 config 键：
  `ngram_vocab_size_ratio`（表大小 = ratio × vocab）、`emb_split_num`（k）、
  `emb_neighbor_num`（n）；哈希为 `pow(vocab, delta, m + 2*(i*k+j) + 1)`
  （vLLM `longcat_flash_ngram.py:37-120` 已实读）。
- 发布夜第一件事：拉 `config.json` 与 `model.safetensors.index.json`，
  把本节所有 U 转成 V——尤其 engram 的**出厂 dtype**（BF16 则表 ~102 GB，
  必须引擎侧重量化或 host 存放；FP8/INT4 则 ~24–55 GB）。

## 3. 生态支持现状（2026-08-26 实测）

**结论：截至发布前，vLLM / SGLang / transformers 主分支均未合并任何
Qwen4 / Flash-Next 支持代码**（`gh search` 代码与 PR 双向核实），
支持 PR 预计发布后数小时内出现。但**三块积木全部已存在**：

| 积木 | vLLM（本机 0.27.1 已装 + main） | SGLang main | 其他 |
|---|---|---|---|
| 混合 GDN+full 注意力 | `qwen3_next.py`、`qwen3_5.py`（注册 `Qwen3NextForCausalLM`、`Qwen3_5(Moe)ForCausalLM`） | `qwen3_next.py`、`qwen3_5*.py` | transformers 有 `qwen3_next`、`qwen3_5(_moe)` modeling；llama.cpp 有 `src/models/qwen3next.cpp` + MTP 接线 |
| n-gram embedding（engram） | **已有**：`longcat_flash_ngram.py`（`LongcatFlashNgramForCausalLM`）+ `csrc/libtorch_stable/ngram_embedding_kernels.cu`（CUDA 哈希 kernel） | **已有**：`kernels/jit/csrc/ngram_embedding.cuh` + `ngram_embedding_manager.py` + `layers/n_gram_embedding.py`（LongCat-Flash 已服务） | 机制与 Flash-Next 同源（同一篇论文路线） |
| MTP 头 | `qwen3_next_mtp.py`、`qwen3_5_mtp.py` | 同名文件 | llama.cpp conversion/qwen.py 统一 MTP 接线 |

即：Flash-Next 的支持代码 = 既有混合注意力 backbone + 既有 MoE + 既有 ngram
embedding 的组合 + 新 config 注册，各框架的接入成本都不高；**我们自研栈的
成本取决于下面 §4 的缺口**。

## 4. 对本仓库（96 GB RTX Pro 6000、TP=1）的缺口分析

已有资产（直接复用）：混合 GDN+full backbone（qwen36/qwen38 后端，正在服务
Qwen3.8-27B）、FP8 KV、NVFP4 加载与 native quantizer
（`notes/2026-08-23-qwen38-modelopt-fp4-quantizer.md`）、`nvfp4_gemm_sm120.cu`、
GGUF 路径、MTP K=3、DFlash2/DSpark、persistent prefix cache、
vision 张量跳过过滤器先例（roadmap 衍生任务）。

缺口（按阻断程度）：

1. **混合 backbone 上的 MoE**——当前只有 Laguna 后端有 MoE，且
   `laguna_router.py` EXPERTS=256 硬编码、路由是 SM120 自研 kernel；
   qwen 后端没有 MoE 层。需要把 sparkinfer fused MoE（或同款）接进
   qwen38 图，router 容量扩到 512 experts / top-10（家族推断值）。
   这比路线图 M5（Qwen3.6-25B-A3B）更大、更急。
2. **Engram（51B n-gram 表）**——全新组件。显存算术（96 GB 卡）：
   - FP8 官方权重：125 + 51 = 176 GB，**直接出局**；
   - 主模型 NVFP4（~0.47 B/param ≈ 59 GB）+ engram INT4（~24 GB）≈ 83 GB +
     lm_head/MTP/KV/激活 ≈ **可行但紧**；
   - engram 放 host RAM 是更稳的路线（每 token 仅 KB 级读取，带宽不是问题，
     但需要引擎支持——vLLM 的 LongCat ngram 实现是纯 GPU
     `VocabParallelEmbedding`，**没有现成 host-offload 开关**，这条要自研）。
3. **多模态**——官方是多模态模型，但路线图明确文本优先；沿用既有
   "允许 vision_config 存在、断言零 vision 张量加载" 策略即可，不阻断。
4. **MoE + MTP/GDN 状态回滚**——既有难点（D-3/ReplaySSM 同族问题），
   DFlash2/DSpark 在 qwen38 target 上的经验可迁移，但 verify 成本随
   专家数上升。

质量预期参考（HN 社区，非官方）：`sqrt(125×6) ≈ 27.3`，即能力对标
Qwen3.8-27B；官方定位是运行时参考实现而非旗舰。

## 5. 发布夜行动清单

1. 拉 `config.json`：`model_type`/`architectures`、`layer_types`、
   `num_experts`/top-k、engram 键、MTP 深度、`max_position_embeddings`、
   vision_config 有无。
2. 拉两份 `model.safetensors.index.json`：张量命名（决定权重映射能复用多少
   `qwen3_5_moe` 逻辑）、engram 出厂 dtype 与精确字节数 → 敲定 §4 显存路线。
3. 盯 vllm/sglang/transformers 的支持 PR（发布后数小时内会来），对照其
   config 解读校准我们的实现。
4. 决策点：engram 上 GPU（INT4）还是 host RAM；NVFP4 自量化复用
   2026-08-23 quantizer 流程的可行性。
5. 然后才是实施：qwen38 后端 + MoE 层 → engram → MTP/DFlash2 接线 →
   质量门（按 `docs/model-support.md` §6 的陷阱清单走）。

## 6. 发布前已完成的准备（2026-08-26 下午，发布前实测）

1. **FN0 探针 PASS：b12x fused MoE 直接接受 Flash-Next 家族几何**
   （`scripts/fn0_probe_b12x_moe_e512.py`，合成权重，对照法排除了探针自身
   错误）：E=512 / top-10 / hidden 8192 / intermediate 2048（2.4T-A95B
   config 推断值）在 M∈{1,4,8,32,64} 全部通过——**expert GEMM 不需要
   sparkinfer 侧改动**。关键测量：每 MoE 层 NVFP4 专家权重驻留
   **13.5 GiB**；prepare 期间 staging 峰值 ~60 GiB（装载器必须逐层
   del+empty_cache，Laguna 生产已有此模式）。探针踩坑记录：
   `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT` /
   `SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE` 必须在 b12x import 前设置；
   scale/alpha 约定必须镜像 `prepare_sparkinfer_layer`（用 1.0 占位会
   非法访存）。**装载期峰值（2026-08-26 三合成层实测）**：生产模式
   （逐层 prepare 后 del+empty_cache）下每层峰值仅高出驻留
   **~14.5 GiB**（早期 61.5 GiB 读数是未 empty_cache 的分配器缓存
   虚高）；最坏峰值 ≈ 全模型驻留 + 14.5 ≈ 74 GiB < 96 GiB——
   **装载策略无需改动**，逐层释放纪律保持即可。
2. **路由语义钉死，且与 Laguna 不同**：Qwen MoE 家族（transformers
   `Qwen3NextTopKRouter` 参考 + vLLM qwen3_next 一致）是
   **softmax(FP32) → top-k → renormalize**；Laguna 的
   `laguna_router_sm120.cu` 是 sigmoid + correction bias，**不可复用**。
   注意 2.4T-A95B checkpoint 带 `e_score_correction_bias` 张量但
   vLLM qwen3_next 并不消费它（装载时跳过）——Flash-Next 落地时需按
   其支持 PR 复核。
3. **新路由已实现并测试**：`runtime/kernels/qwen_moe_router.py`
   （Triton，softmax+top-k，tie-break 取低专家号，输出权重按 logits
   dtype 舍入、ids int32，支持调用方 arena）+ `tests/test_qwen_moe_router.py`
   17 项全绿（含 512/256/200/128 专家、精确平局、renormalize 开关、
   arena、空批、契约校验）。已验证的数值事实：kernel softmax 对
   torch softmax 最大相对差 ~8e-7；bf16 输出舍入是主要误差源（~两个
   bf16 码位），近简并专家的选择次序不保证与 torch.topk 逐位一致
   （概率差 <1e-6 量级，语义等价）。
4. **张量命名部分锁定**：Laguna 实际 checkpoint 确认家族命名为
   `model.layers.N.mlp.experts.E.{gate,up,down}_proj.{weight_packed[U8,
   N,K/2], weight_scale[F8_E4M3, N,K/16], weight_global_scale,
   input_global_scale}`；Flash-Next 若沿用 qwen3_5_moe 布局则装载映射
   可复用。待发布后以真实 index.json 复核。

尚未做（等发布）：真实 config 解析与权重映射、质量门、端到端服务接线。

**发布前已实现并测试的组件（均绿）：**

| 组件 | 文件 | 测试 |
|---|---|---|
| softmax→topk→renorm Triton 路由（家族契约，非 Laguna sigmoid） | `runtime/kernels/qwen_moe_router.py` | `tests/test_qwen_moe_router.py` 17 |
| MoE 层模块（gate→路由→b12x 专家+shared expert，CUDA-Graph 安全） | `runtime/model/qwen38_moe.py` | `tests/test_qwen38_moe.py` 5（含与生产 prepare 位级一致 + 图重放位级一致） |
| 几何参数化专家装载/准备（家族命名） | `runtime/backends/qwen38_sparkinfer_moe.py` | 上 + `tests/test_qwen38_moe_loading.py` 2（合成 safetensors） |
| 离线 NVFP4 权重量化（checkpoint 三元组，自量化→b12x 全链） | `runtime/nvfp4_weight_quant.py` | `tests/test_nvfp4_weight_quant.py` 5 |
| Engram n-gram 哈希表 + 融合嵌入 | `runtime/model/engram.py` | `tests/test_engram.py` 7 |

关键实现结论：
- `SparkinferMoEOutputArena` 参数化了 `hidden_size`（Laguna 默认不变，
  向后兼容）；
- NVFP4 权重 gs 约定 = `2688/amax`（大值，compressed-tensors/Laguna 族），
  dequant `code*sf/gs`，量化器吃同一个大值——方向是实测钉死的
  （反向 nRMSE≈1.0）；
- engram 哈希 = 每 embedder 模 `m+2r+1` 上的 base-vocab 多项式，回看被
  序列起点/负标记/回看中的 EOS 截断，与 SGLang `.cuh` 逐位一致。

全量套件（本改动前后各一次）：`2673 passed, 23 skipped`；ruff check/format 全绿。
注：`/tmp/ci-sim` 解释器本机不存在，用系统 `/usr/bin/python3`（无 torch）
做 collect-only 核验，唯一 collection error 是
`tests/test_server_perf_grid_observability.py` 依赖 `aiohttp`——**与本改动
无关的既有问题**。

## 来源

- modelscope.cn pre-release API、HF 预告页（2026-08-26 实拉）
- `Qwen/Qwen3.8-2.4T-A95B/raw/main/config.json`（实拉）
- `JP-devv/humble-b70-llm:docs/qwen-flash-next.md`（社区 pre-release 评估，
  V/I/U 分级严谨，本文 V 级规格转引自它）
- HN 49432317 讨论（质量预期、day-0 定位解读）
- 本机 `~/.venvs/vllm`（vllm 0.27.1）registry + `longcat_flash_ngram.py` 实读
- `gh search` 对 vllm/sglang/transformers/llama.cpp 的代码与 PR 检索
  （2026-08-26，零 Qwen4/Flash-Next 命中）
- IT之家 2026-08-25 报道（发布时间与双版本确认）
