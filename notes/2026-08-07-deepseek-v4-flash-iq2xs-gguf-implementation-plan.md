# DeepSeek-V4-Flash-0731（GGUF IQ2_XS）接入实施方案

> 编制日期：2026-08-07 · 状态：🟢 方案已定，待实施
> 触发：用户指令 —— 在本 runtime 上运行
> `bullerwins/DeepSeek-V4-Flash-0731-GGUF` 的
> `DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf`（87.9 GB / 81.86 GiB）。
> 官方参考材料已落地在 [`dsv4flash-ref/`](dsv4flash-ref/)（官方 inference 引擎
> model.py/kernel.py/convert.py、encoding 文档与 4 组 tokenizer 测试向量、
> config.json、模型卡）。本文是 `docs/model-support.md` §5 六步流程的实例化。

---

## 0. TL;DR

- **可行，且注意力 kernel 层基本是现成的**：本项目自有 sparkinfer fork 已含
  `attention.compressed_mla` / `attention.sparse_mla` / `attention.nsa_indexer`，
  其常量（nope 448 FP8 / rope 64 BF16 / 单 latent KV / 窗口 128 / topk 512 /
  `attn_sink`）与 DeepSeek-V4-Flash **逐项吻合**（本就该 fork 为 DSV4 系列写的）。
  本机 cutlass-dsl 已升 4.6.0，三者 `is_supported()==True`（仍需功能实测，见 R3）。
- **真正的缺口**有四个：① GGUF 容器加载器（现在只认 safetensors）；
  ② IQ2_XS / Q8_0 权重的 dequant-GEMM kernel（sparkinfer MoE 通路只吃 NVFP4）；
  ③ DSV4 特有的模型图部件（HCA 压缩器、indexer、mHC 超连接、hash 路由、
  sqrtsoftplus top-6 路由）；④ 服务层（tokenizer 是自定义 encoding，非标准模板）。
- **为什么必须是 GGUF/IQ2_XS**：284B 参数在 96 GB 卡上，原生 MXFP4+FP8（≈162 GB）
  和 NVFP4 转码（≈142 GB）都装不下；IQ2_XS（81.86 GiB）是唯一能装下的量化档。
  这**推翻** 2026-07-26 路线图里"不做 GGUF/IQ kernel"的决定，理由见 §1.3。
- **投机解码**：GGUF 主文件不含 MTP/DSpark 张量（已从 header 证实）；
  DSpark 伴生模型（10.9 GB，block-5 草稿）是二期事项，结构上是"小 DFlash"，
  复用 DFlash 引擎骨架与通用的 `mtp_accept.py`。
- **验证 oracle 链**：官方 reference 代码（逐部件数值）→ 本地 llama.cpp
  （master 已支持 `deepseek4` 架构，端到端贪心/top-k-logit 对齐）→ bfdiag 基线。

---

## 1. 背景与决策前提

### 1.1 模型事实（来源：官方 config.json + GGUF header + arXiv 2606.19348）

DeepSeek-V4-Flash-0731：**284B 总参 / 13B 激活**，1M 上下文，MIT。
架构名 `deepseek_v4` / GGUF arch `deepseek4`。

| 项 | 值 |
|---|---|
| 层数 | 43（另 3 层 DSpark/MTP 阶段，GGUF 不含） |
| hidden | 4096；vocab 129280；rms_eps 1e-6 |
| MoE | 256 routed + 1 shared，top-6，inter 2048，`sqrtsoftplus` 打分 + noaux_tc 选择偏置 + renorm × route_scale 1.5，swiglu clamp ±10 |
| hash 路由 | 前 3 层：专家选择由 `tid2eid[token_id]`（[129280,6] int32）预定，权重仍取自 gate logits |
| 注意力 | MLA 变体：q_lora 1024 → 64 头 × head_dim 512（latent，448 nope + 64 rope），单 latent KV，o_groups 8 × o_lora 1024，每头 `attn_sink` 偏置 |
| 混合注意力 | `compress_ratios` 逐层：层 0,1,40,41,42 = 0（纯滑窗 128，base rope 无 YaRN）；层 2,4,…,38 = 4（CSA：滑窗 + indexer top-512/压缩位）；层 3,5,…,39 = 128（HCA：滑窗 + 全部 seq/128 压缩位） |
| 压缩器 | 每 ratio 个 token 做学习门控池化（wkv/wgate + APE + softmax），ratio-4 带 overlap；输出 nope FP8（block-64 ue8m0）+ rope BF16；indexer 专用压缩器额外做 Hadamard 旋转 + FP4 模拟量化 |
| indexer | q 来自 q latent（wq_b 1024→64×128），Hadamard + FP4 模拟；weights_proj [4096→64] BF16；对压缩 K 打分取 top-512 |
| mHC 超连接 | hc_mult 4：每层 attn/ffn 前后各做一次 `(2+4)*4=24`-mix，Sinkhorn 20 迭代投影（fp32）；最终 hc_head 收敛到 logits |
| RoPE | YaRN factor 16（65536→1M），theta 10000；压缩 KV 用 theta 160000；纯滑窗层不 YaRN |
| 原生量化 | dense FP8 128×128 block + ue8m0 scale；专家 MXFP4（block-32 e8m0） |
| 采样 | temp 1.0，top_p 0.95（agent）/1.0；thinking/encoding 走 `encoding_dsv4.py` |
| DSpark | block_size 5，noise token 128799，target layers [40,41,42]，markov rank 256，confidence head |

### 1.2 GGUF 文件事实（已解析 header：1328 张量，63 KV）

| 族 | 数量/形状 | 类型 | 体积 |
|---|---|---|---|
| `blk.L.ffn_{gate,up,down}_exps` | [256, 2048, 4096] 等（GGUF 维序反转） | **IQ2_XS**（74/32 bpw） | ≈74.6 GiB |
| dense 投影（q_a/q_b/kv/o_a/o_b、shexp、embed、lm_head、hc_fn、compressor 等） | 661 个 | Q8_0（8.5 bpw） | 7.19 GiB |
| norms / attn_sinks / ape / hc base+scale | 492 个 | F32 | ~0.03 GiB |
| `ffn_gate_inp`（gate 权重） | 43 × [256,4096] | BF16 | 0.08 GiB |
| `ffn_gate_tid2eid` | 3 × [6,129280] | I32 | 0.009 GiB |
| `exp_probs_b.bias`（层 3+） | [256] | F32 | — |
| **合计** | | | **81.86 GiB**（与 README 一致） |

关键确认：**无 `mtp.*`/`dspark.*` 张量**；tokenizer 完整嵌入（gpt2 BPE +
merges + chat template，`tokenizer.ggml.pre = joyai-llm`）；
`deepseek4.attention.compress_ratios` 为 46 项（43 层 + 3 MTP 阶段）。

### 1.3 为什么推翻"不做 GGUF/IQ kernel"（2026-07-26 决定）

当时的前提是"权重 NVFP4 优先、FP8 次之"且目标模型 ≤30B 级。DeepSeek-V4-Flash
改变了约束：**284B 参数 × 任何 ≥4 bpw 的格式都 > 96 GB**。装得下的唯一档位是
≤2.6 bpw 的专家量化（IQ2_XS 档，KLD 0.60 / top-token 一致率 75%，是 bullerwins
Pareto 集里唯一 <90 GB 的点）。因此：
- GGUF 不是"兼容层扩张"，而是**这个模型唯一的入口格式**；
- IQ2_XS/Q8_0 dequant-GEMM 是**容量约束的直接推论**，不是格式偏好；
- 该决定仅对 DSV4-Flash 接入放行；不改变"新模型优先 NVFP4/FP8"的一般政策。

### 1.4 显存测算（RTX PRO 6000 Blackwell，95.6 GiB 可用）

| 项 | 估算 |
|---|---|
| 权重（packed，常驻不反量化） | 81.86 GiB |
| KV/槽（128K ctx）：窗口环 43×128×~584B + ratio-4 层 19×32K×584B + ratio-128 层 19×1K×584B + indexer 19×32K×68B | ≈0.45 GiB |
| KV/槽（1M ctx） | ≈3.3 GiB |
| 压缩器 decode 状态、激活、workspace、CUDA context、CG | ~4 GiB |
| **2 槽 × 128K 合计** | **≈86.8 GiB（余量 ~9 GiB）** |

红线：**禁止任何"懒反量化成 BF16 常驻"的路径**（Qwen3.6 dequant-cache 教训，
`notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`）。所有量化权重必须以
packed 形态参与计算（kernel 内 dequant）。

---

## 2. 关键决策（D1–D10）

| # | 决策 | 选项 | 拍板 | 理由 |
|---|---|---|---|---|
| D1 | 权重容器 | (a) GGUF 直读 (b) 离线转 safetensors | **(a)** | 转码要多一份 88 GB 盘 + 一遍全量读写；GGUF header 即元数据；流式单张量读取可守住 23 GiB 主机内存 |
| D2 | 专家 GEMM | (a) IQ2_XS dequant-GEMM（Triton→CUDA） (b) 转 NVFP4 (c) 反量化 BF16 | **(a)** | (b)≈142 GB、(c)≈250 GB，均超卡；(a) 对 GGUF artifact 无损，与 llama.cpp oracle 同输入 |
| D3 | dense GEMM（Q8_0） | (a) Q8_0 dequant-GEMM (b) Q8_0→FP8 转换复用 fp8_linear | **(a)** | (b) 引入二次量化误差，破坏与 oracle 的贪心对齐目标；Q8_0 反量化极简 |
| D4 | 注意力 kernel | (a) sparkinfer fork 的 compressed_mla/sparse_mla/nsa_indexer (b) 自研 | **(a)** | fork 中就是为 DSV4 写的，`attn_sink`/512topk/128 窗口全是一等参数；"sparkinfer 是唯一注意力 kernel"是仓库政策 |
| D5 | KV 布局 | 按 `compressed_reference.py` 页格式：每 token [448 FP8][64 BF16 rope][8 UE8M0 scale]，页 256 token；窗口环 + 压缩区 + indexer 区三域分层池 | **照此** | 与 fork kernel 的内存池契约一致（SGLang 兼容布局）；分层池有 Qwen36SlotPool/Laguna 环先例 |
| D6 | 路由 kernel | (a) 参数化 laguna_router_sm120.cu (b) 新写 dsv4 路由 kernel | **(b)，先 Triton 后 CUDA** | Laguna kernel 是位精确资产（oracle 钉死），不动它；DSV4 需要 sqrtsoftplus+top6+route_scale+hash(input_ids)，差异过大 |
| D7 | 投机 | 一期不做；二期 DSpark（DFlash 骨架 + Markov/confidence head） | **分期** | DSpark 权重不在主 GGUF；先保证主模型贪心对齐 |
| D8 | 对齐 oracle | 官方 reference（部件级）+ llama.cpp（端到端） | **双 oracle** | reference 验证数学语义；llama.cpp 消费同一 artifact，给出端到端贪心/top-k-logit 基线 |
| D9 | tokenizer | 从 GGUF 抽取 vocab/merges 生成 HF tokenizer 目录 + 服务层 vendoring `encoding_dsv4.py` | **照此** | 官方明确不用 Jinja 模板；4 组官方测试向量作 fixture |
| D10 | 前缀缓存 | 一期关；压缩条目是 token 块的确定函数，hash 可复用 block_pool 的 blake2b 链 | **延后** | 先对齐正确性；异构缓存联动是 Qwen3.6 级工程量 |

---

## 3. 分阶段实施

> 严格遵循 `docs/model-support.md` §5 的顺序：**事实基线 → ModelSpec → 模型图+加载器
> → 正确性（eager/batch=1/无图/无投机/无前缀缓存）→ 服务化 → 性能与投机**。
> 每阶段给出验收门禁；未过门禁不进下一阶段。

### Phase 0 · 资产与工具（进行中）

- [x] 主文件 + imatrix 经 hf-mirror 下载中（wget setsid 后台，~15-20 MB/s）
      → `/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/`
- [x] 官方参考落盘 `notes/dsv4flash-ref/`（含 encoding 4 组测试向量）
- [x] GGUF header 解析（本文 §1.2；脚本可沉淀为 `loader/inspect_gguf.py`）
- [ ] 构建本地 llama.cpp（master 已含 `LLM_ARCH_DEEPSEEK4`）：
      `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`；跑通
      `llama-cli -m <gguf> -p "..." --temp 0`，作为端到端 oracle
- [ ] sparkinfer fork 三个 DSV4 op 的**功能实测**：在 fork 内跑
      `tests/attention/test_compressed_mla.py`、`test_sparse_mla.py`、
      `test_attention_nsa_indexer_*.py`（先处理 fork 当前未提交的
      tensor_fp8_channel_linear WIP —— 开工前 `git status` 确认分支节奏）
- [ ] 安装 `fast_hadamard_transform`（官方 reference 的依赖；tilelang 0.1.9 已在）

### Phase 1 · 事实基线与离线参考（~1 周）

产出：`notes/2026-08-XX-dsv4flash-fact-baseline.md` + 离线工具。
1. `loader/inspect_gguf.py`：张量清单/类型/偏移/字节统计，进 CI-sim 可跑的
   纯 stdlib 解析器（bfdiag 无第三方依赖原则）。
2. **IQ2_XS / Q8_0 CPU 参考反量化器**（Python/numba 均可，离线工具不限依赖）：
   IQ2_XS 块格式按 ggml 定义（256 元素超块，8×32 子块，2-bit 查表 + 子块 scale），
   用 imatrix 文件与 llama.cpp 的 dequant 做交叉验证。
3. **部件级数值 oracle**：用 GGUF 反量化出的单层权重（Q8_0/F32 部件可精确反量化）
   初始化官方 `inference/model.py` 的 Compressor / Indexer / Block(hc) 模块，
   随机输入下对比我们实现的输出（fp32 参考，容差按 §4 门禁）。
   重点：overlap_transform、APE 加在 softmax 前/后的顺序、rope 去旋转时机、
   sinkhorn 的行列归一化方向、hash 路由"权重仍取自 logits"的语义。
4. tokenizer 抽取：GGUF → HF `tokenizer.json`（gpt2 BPE）+ special tokens；
   `pre=joyai-llm` 的 pre-tokenize 正则从 llama.cpp 源码核对；
   用 4 组官方 encoding 测试向量做 round-trip 验证。
5. 显存实测：权重全量上 GPU 的占用（冷进程），确认 §1.4 测算。

**门禁**：IQ2_XS 反量化与 llama.cpp 逐张量 bit-exact（CPU 侧）；tokenizer
round-trip 4/4；显存实测与测算差 <5%。

### Phase 2 · GGUF 加载器 + 模型图（~2 周）

1. `runtime/loading/gguf.py`：mmap + 逐张量 yield `(name, tensor)`，遵守
   `iterate_safetensors_checkpoint` 的流式契约（23 GiB 主机内存红线）；
   张量直接上 GPU，不做主机侧 BF16 中间态。IQ2_XS/Q8_0 以 packed 形态保存
   （torch.uint8 视图 + 形状/scale 元数据对象），**不反量化**。
2. `runtime/model_registry.py`：新增 `DeepseekV4ForCausalLM` family →
   backend `deepseek_v4`；GGUF 的 loader 判定走**文件后缀分支**
   （`resolve_checkpoint` 目前只认目录+config.json，需在 `server/app.py::lifespan`
   前加文件感知解析；`runtime/loading/__init__.py` 预留的"第二个格式触发 dispatch"
   在此兑现）。
3. `runtime/architecture.py`：从 GGUF KV 生成 `ArchitectureSpec` 的 producer
   （layer_types 逐层：`sliding_attention`×5 + `csa_attention`×19 + `hca_attention`×19，
   或复用现有词表 + 每层 compress_ratio 字段 —— A1 备忘里"滑窗参数提升为可查询字段"
   在此一并落地）。
4. `runtime/model/dsv4_model.py`（新模型图）：
   - 线性层分型：`IQ2XSLinear`（专家）、`Q8_0Linear`（dense）、`PlainLinear`（BF16 gate_inp）、
     `F32Param`（norms/sinks/ape/hc）——接口与 `_make_linear` 分派模式一致；
   - 每层：HC(pre) → attn_norm → MLA(compressor/indexer) → HC(post) → HC(pre) →
     ffn_norm → MoE(gate/hash + 6 routed + 1 shared, swiglu clamp) → HC(post)；
   - `load_weights` 消费 GGUF 命名（`blk.L.attn_q_a` 等），全覆盖断言零豁免。
5. 无 KV 缓存的 eager 前向（给定外部 KV 注入的占位）跑通形状。

**门禁**：权重全覆盖断言通过；单 token 前向与官方 reference（同权重同输入，
无缓存的纯计算部分）logits 余弦 > 0.99999；`tools/verify_no_vllm_laguna.py` 绿。

### Phase 3 · 缓存与注意力后端，贪心对齐（~3-4 周，最大不确定区）

1. `runtime/model/dsv4_slots.py`：每层三域缓存池（窗口环 / 压缩区页 256 /
   indexer 区）+ 压缩器 decode 状态（kv_state/score_state，每槽固定）+
   scratch 行（沿用 Qwen36SlotPool 模式；放宽其 uniformity asserts 是**本阶段
   的主要结构改动**，按 prior-art 笔记用"分离分配器 + 协调层"，不做统一池）。
2. KV 写 kernel：新 Triton kernel 按 `compressed_reference.py` 页布局写
   latent KV（nope→FP8 block-64 ue8m0 + rope BF16 + scale 字节）；
   窗口环写入与压缩器产出的压缩条目写入分开。
3. 注意力接线：
   - 层 0,1,40-42（ratio 0）：compressed_mla 仅窗口部分（或 paged SWA）；
   - ratio-4 层：nsa_indexer 出 top-512 → sparse_mla/compressed_mla；
   - ratio-128 层：顺序索引全压缩位 → sparse_mla；
   - `attn_sink` 从 GGUF 载入（Laguna 已有 sink 参数布线先例，
     `laguna_decoder.py:256`，此处不再丢弃）。
4. 路由：Triton 版 DSV4 router（gate GEMM 走模型图；kernel 做 sqrtsoftplus +
   偏置选择 + top-6 + renorm × 1.5；hash 层吃 input_ids 查 tid2eid）。
   CUDA Graph 兼容性按现有纪律：arena 预分配、定形不定值。
5. mHC：Triton 融合 kernel（linear → sinkhorn → pre/post），fp32 内部；
   先用 torch 实现打通，再融合（decode 每 token 1720+ 次小 launch 不可接受）。
6. 端到端：`load_deepseekv4_model()` orchestrator + `DeepseekV4Backend`
   （`ModelBackend` 协议 12 成员 + `BackendCapabilities`，
   `check_conformance` 过）；eager、batch=1、无图、无前缀缓存。

**门禁（正确性红线）**：与 llama.cpp 同 prompt 贪心逐 token 对齐，
≥3 个工作负载 × 512 token；不要求 100%（不同 kernel 库累加序差异），
但要求 top-1 一致率 ≥ 99%、top-5 logit 一致率 ≥ 99.9%、无系统性漂移
（`docs/model-support.md` §4 的量化 kernel 验收口径）；逐层 logits 余弦进 bfdiag；
`bf trace` 全绿。

### Phase 4 · 服务化（~2-3 周）

1. `ServerEngine._load_model` 第三分支 `_load_deepseek_model`；tokenizer 来源
   改为 Phase 1 产出的 HF 目录（engine.py:460-479 的 `AutoTokenizer` 假设）；
   preflight 增加 GGUF 分支。
2. 采样与格式：temp 1.0/top_p 0.95 默认；`server/formats/` 增加 DSV4 encoding
   适配（vendoring MIT 的 `encoding_dsv4.py`，thinking/reasoning_effort 三档映射）；
   EOS=1、无 BOS 添加。
3. decode CUDA Graph：compressed_mla 的 `for_contract`/replay-state 模式
   （Qwen36 `Qwen36DecodeGraphAttention` 是模板）；压缩器状态的 per-slot 缓冲
   在捕获前分配；捕获在空槽窗口内完成（现有纪律）。
4. 槽位/容量：2 槽 × 128K 起步（余量 ~9 GiB）；`QSR_SERVER_*` 参数映射；
   显存审计走 `notes/2026-08-02-gpu-memory-audit.md` 的方法。
5. 双协议 HTTP 回归绿；与 Phase 3 eager 路径贪心 bit-exact（同 kernel 配置下）。

### Phase 5 · 性能与投机（持续）

1. IQ2_XS dequant-GEMM 调优（Triton → 手写 CUDA/tilelang；目标 decode ≥ 100 tok/s）。
2. prefill 分块（indexer/压缩器的 chunked 语义）与长上下文容量实测（128K → 1M）。
3. **DSpark**：下载 `DeepSeek-V4-Flash-0731-DSpark.gguf`（10.9 GB）；
   3 阶段 DSparkBlock（自带 MoE + HC）走 DFlash 骨架：main_proj 注入
   （替代 DFlash 的 per-layer context-KV 预计算）、noise token ≈ MASK、
   verify 批 6 token（anchor+5）、Markov head（rank-256 logits 偏置，块内自回归）
   + confidence head；accept 复用 `mtp_accept.py`（K=5 直接映射）。
4. 前缀缓存：压缩条目哈希接入 block_pool 的链式 blake2b；异构缓存联动按
   Qwen3.6 双向非对称 lockstep（INV-A3-3）模式。

---

## 4. 验证计划（oracle 链与纪律）

1. **部件级**：官方 reference（tilelang kernel + 我们反量化的权重）→ 我们的
   Triton 实现；fp32 参考容差：hc/compressor 1e-4，注意力 logit 1e-3。
2. **端到端**：llama.cpp（同 artifact、greedy、temp 0）≥3 工作负载 × 512 token；
   指标：top-1 ≥99%、top-5 ≥99.9%、KLD 进 bfdiag run record。
3. **诊断纪律**：实验一律 `bf exec`（不新增 benchmarks/ 脚本）；数字比较先
   `bf diff`；失败先读 trace；温冷引擎边界按 model-support.md §5.4。
4. **回归**：CI-sim（torch-free）+ 全量 venv 套件保持绿；新增单测不依赖权重
   （GGUF 解析器用合成 GGUF fixture）。

## 5. 风险登记

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| R1 | 显存余量仅 ~9 GiB | CG 捕获尖峰/碎片 OOM | packed-only 常驻；空槽捕获；显存审计；必要时降为 1 槽或缩上下文 |
| R2 | IQ2_XS Triton GEMM 性能未知 | decode 不达预期 | decode 是带宽瓶颈，dequant 开销可吸收；Phase 5 再上 CUDA |
| R3 | sparkinfer 三个 DSV4 op 在本卡**未功能验证**（is_supported 只是版本地板） | Phase 3 地基塌 | Phase 0 先跑 fork 自带测试；不行就在 fork 内修（fork 可直接编辑） |
| R4 | `joyai-llm` pre-tokenizer 行为不明 | tokenizer 对齐失败 | 以 llama.cpp 实现为基准 + 官方 4 组测试向量 round-trip |
| R5 | noaux_tc 平局/偏置语义细节 | 贪心漂移 | 以 reference `Gate.forward` 逐行为准；topk 平局取最小索引 |
| R6 | mHC sinkhorn fp32 数值与性能 | 慢/漂移 | 融合 kernel 全 fp32；与 reference 对拍 20 迭代结果 |
| R7 | 主机内存 23 GiB vs 88 GB 文件 | 加载 OOM | mmap + 逐张量流式（契约已明确） |
| R8 | ratio-128 层在 1M ctx 的索引构造 | 长上下文卡壳 | 顺序索引 buffer 预生成；page 256 对齐 |
| R9 | sparkinfer fork 有未提交 WIP（tensor_fp8_channel_linear） | 分支节奏冲突 | 开工前按 AGENTS.md 查 git status；DSV4 工作另起 worktree |
| R10 | 政策反转的文档债 | 后人困惑 | 本文 §1.3 + roadmap 对应条目引用本文 |

## 6. 工作量与里程碑

| 阶段 | 估算（单人） | 出口 |
|---|---|---|
| P0 资产/工具 | ~2 天（下载占墙钟 ~1.5h） | llama.cpp oracle 可用；fork op 实测过 |
| P1 事实基线 | ~1 周 | 反量化 bit-exact；tokenizer 4/4 |
| P2 加载器+模型图 | ~2 周 | 全覆盖断言；无缓存前向对拍 |
| P3 缓存+注意力+对齐 | ~3-4 周 | 贪心对齐三工作负载 |
| P4 服务化 | ~2-3 周 | HTTP 双协议 + CG，与 eager bit-exact |
| P5 性能+DSpark | 持续 | tok/s 基线进 bfdiag；接受率 |

合计到"可服务（eager 正确、CG decode、双协议）"：约 8-10 周。

## 7. 附录

### 7.1 GGUF → reference 参数映射（节选）

| GGUF | reference（inference/model.py） |
|---|---|
| `blk.L.attn_q_a/q_b/attn_q_a_norm` | `attn.wq_a/wq_b/q_norm` |
| `blk.L.attn_kv/attn_kv_a_norm` | `attn.wkv/kv_norm` |
| `blk.L.attn_output_a/b` | `attn.wo_a/wo_b`（o_a 按组视图 [8,1024,4096]） |
| `blk.L.attn_sinks` | `attn.attn_sink` |
| `blk.L.attn_compressor_{kv,gate,ape,norm}` | `compressor.wkv/wgate/ape/norm` |
| `blk.L.ffn_gate_exps/up_exps/down_exps` | `experts[*].w1/w3/w2`（E 维融合） |
| `blk.L.ffn_*_shexp` | `shared_experts.w1/w3/w2` |
| `blk.L.ffn_gate_inp` | `gate.weight` |
| `blk.L.exp_probs_b.bias` | `gate.bias` |
| `blk.L.ffn_gate_tid2eid` | `gate.tid2eid`（层 0-2） |
| `blk.L.hc_attn_fn/base/scale` 等 | `hc_attn_fn/hc_attn_base/hc_attn_scale` |
| `token_embd/output/output_norm/output_hc_*` | `embed/head/norm/hc_head_*` |

### 7.2 采样与 encoding

- temp 1.0；top_p 0.95（agent）/1.0；EOS=1；不加 BOS。
- `encode_messages(messages, thinking_mode=...)`；reasoning_effort ∈ {low, high, max}
  （映射细节以 `encoding_dsv4.py` 为准，Phase 4 落地时核对）。

### 7.3 下载与文件

- 权重：`/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/`（wget setsid 后台，
  断点续传；日志 `/home/bot/models/wget_main.log`）
- 参考：`notes/dsv4flash-ref/`；arXiv:2606.19348；
  官方仓库 `deepseek-ai/DeepSeek-V4-Flash-0731`（hf-mirror 可达）
- DSpark 伴生（Phase 5 再下）：`DeepSeek-V4-Flash-0731-DSpark.gguf` 10.9 GB
