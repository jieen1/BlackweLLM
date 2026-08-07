# DeepSeek-V4-Flash-0731（GGUF IQ2_XS）事实基线

> 编制日期：2026-08-07 · 状态：🟢 有效
> [`docs/model-support.md`](../docs/model-support.md) §5.2 第 1 步的产出：
> 每一项都是实测值 + 复现命令，没有"应该是"。接入方案见
> [`2026-08-07-deepseek-v4-flash-iq2xs-gguf-implementation-plan.md`](2026-08-07-deepseek-v4-flash-iq2xs-gguf-implementation-plan.md)，
> 官方参考材料在 [`dsv4flash-ref/`](dsv4flash-ref/)。

## 1. Checkpoint 身份

| 项 | 值 | 复现 |
|---|---|---|
| 文件 | `/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf` | hf-mirror wget -c（2026-08-07 下载完成） |
| 大小 | 87,899,450,784 B（payload 81.86 GiB + header 5,334,816 起数据区） | `python -m loader.inspect_gguf <file>` → `[COMPLETE]` |
| 伴随 | imatrix 470,338,784 B（802×512-token 校准，bullerwins 量化用） | 同目录 |
| GGUF | version 3，arch `deepseek4`，1328 张量，63 KV | 同上 |
| 上游 | `deepseek-ai/DeepSeek-V4-Flash-0731`（arXiv:2606.19348：284B 总参/13B 激活/1M ctx，MIT） | hf-mirror 可达 |

## 2. 权重构成（inspect_gguf 实测）

| 类型 | 张量数 | 体积 | 承载 |
|---|---:|---|---|
| IQ2_XS | 129（43 层 × gate/up/down_exps） | 74.58 GiB | routed experts，[256, …] 融合存法 |
| Q8_0 | 661 | 7.19 GiB | 全部 dense 投影、embed、lm_head、hc_fn、compressor wkv/wgate |
| BF16 | 43 | 0.08 GiB | 每层 `ffn_gate_inp`（router 权重 [256,4096]） |
| I32 | 3 | 0.01 GiB | 层 0-2 `ffn_gate_tid2eid` [6,129280] 哈希路由表 |
| F32 | 492 | ~0.03 GiB | norms、attn_sinks[64]、APE、hc base/scale、exp_probs_b.bias |

**无 mtp/dspark 张量**（投机权重在单独的 10.9 GB DSpark 伴生 GGUF 中）。

### 2.1 张量形状约定（三方核验，0 失配）

- **GGUF dims = GGML ne 顺序，dims[0] 是连续（最快）维；torch shape = dims 反转。**
  例：`ffn_gate_inp` dims (4096, 256) → torch [256, 4096] = Linear(4096→256).weight；
  `ffn_gate_exps` dims (4096, 2048, 256) → torch [256, 2048, 4096] = 专家主序 [E, inter, hidden]。
- 核验方法（复现命令）：`python tools/verify_dsv4_tensor_map.py` —— 按 llama.cpp
  `src/models/deepseek4.cpp:78-145` 的 create_tensor 声明手工转录全部 1304 个期望
  ne，与实际文件逐一比对：**1304/1304 存在且形状 0 失配**；参考 model.py 的参数
  形状（Linear.weight=[out,in]、ape=[ratio, coff×512]、hc_fn=[24, 16384] 等）同向一致。
- indexer 张量实际命名（此前猜测有误，已纠正）：`blk.L.indexer.proj.weight`、
  `blk.L.indexer.attn_q_b.weight`、`blk.L.indexer_compressor_{kv,gate,ape,norm}.weight`。
- 张量族清单：全局 6（token_embd/output/output_norm/output_hc_{fn,base,scale}）；
  每层固定 20（attn_norm/sinks/q_a/q_a_norm/q_b/kv/kv_a_norm/output_a/output_b、
  hc_{attn,ffn}_{fn,base,scale}×2、ffn_norm/gate_inp/{gate,up,down}_exps/
  {gate,up,down}_shexp）；ratio≠0 层 +4（attn_compressor_{kv,gate,ape,norm}）；
  ratio=4 层再 +6（indexer.proj/attn_q_b + indexer_compressor_{kv,gate,ape,norm}）；
  层 0-2 有 ffn_gate_tid2eid，层 3+ 有 exp_probs_b.bias。合计 6+43×20+41×4+21×6+3 = 1328 ✓。

## 3. 反量化语义（已位精确验证）

- Q8_0：`{fp16 d; int8 qs[32]}`，34 B/32 元素；`y = d*q`。
- IQ2_XS：`{fp16 d; uint16 qs[32]; uint8 scales[8]}`，74 B/256 元素；
  每个 qs 码低 9 位 → `iq2xs_grid`（512×uint64，8 字节即 8 个幅值），
  高 7 位 → `ksigns_iq2xs`（128 个符号掩码）；子块 delta = `d*(0.5+nibble)*0.25`。
- 验证：`tools/gguf_dequant_golden.c` 链接 llama.cpp 官方
  `dequantize_row_{q8_0,iq2_xs}`，16 随机块/格式 **fp32 位精确一致**
  （`tests/test_gguf_dequant_golden.py`）。码表逐字抽自 llama.cpp 79bba02a。
- 真实张量探针：`tools/gguf_tensor_probe.py`（Q8_0 attn_q_a 值域 ±0.078、
  IQ2_XS gate_exps 落在特征离散网格 ±0.1 —— 形态正常）。
- **符号零语义**：C 参考按 `(db*grid)*(±1)` 计算，零幅值带符号位时产生 −0.0；
  任何实现必须在浮点域取负（纯 Python 参考已按此修正并仍与 llama.cpp 位精确）。
- numpy 向量化反量化（`runtime/loading/gguf.py::dequantize_{q8_0,iq2_xs}_packed`）
  已通过与纯 Python 位精确参考的逐位对比（`tests/test_gguf_loader.py`），
  可用于批量离线工具；**位精确基准链**：llama.cpp 官方 C 实现
  （`tools/gguf_dequant_golden.c` 链接验证）⇔ 纯 Python 参考
  （`loader/gguf_dequant.py`）⇔ numpy 向量化版。

## 4. 架构要点（config.json + GGUF KV + 官方 reference 三方一致）

43 层；hidden 4096；vocab 129280；MoE 256+1/top-6/sqrtsoftplus/noaux_tc 偏置/×1.5/swiglu clamp ±10；
MLA 变体（q_lora 1024 → 64 头 × 512 latent，448 nope FP8 + 64 rope BF16，单 latent KV，
o_groups 8 × o_lora 1024，每头 attn_sink）；逐层 compress_ratios（**GGUF 与 HF config
均为 46 项 = 43 主层 + 3 MTP 阶段**）：主层 = `[0,0] + [4,128]×20 + [4]`，即
**层 0,1 = 0（纯窗口 128）；偶数层 2..42 = 4（21 层，带 indexer top-512）；
奇数层 3..41 = 128（20 层，全压缩位）**；末尾三个 0 属 DSpark 阶段，不是层 40-42
（层 40-42 在文件中带压缩器，层 40/42 带 indexer，实测证实）；
压缩器 = 门控池化（wkv/wgate + APE + softmax，ratio-4 带 overlap），输出 nope FP8 block-64 ue8m0；
indexer 压缩器带 Hadamard 旋转 + FP4 模拟；mHC mult 4 / sinkhorn 20 / fp32；
YaRN（factor 16，65536→1M，theta 10000；压缩 KV 用 theta 160000；纯窗口层不 YaRN）；
前 3 层哈希路由（tid2eid，权重仍取 logits）。
逐行语义基准：`dsv4flash-ref/inference/model.py`（本机可 import，ModelArgs 接受 0731 全形状）。

## 5. tokenizer（已证一致）

- 官方 `tokenizer.json`（BPE，128000 词 + 1283 added = 129280 id 空间）已落盘
  `dsv4flash-ref/`；GGUF 内嵌版与官方 id 空间内逐字节一致
  （`tests/test_gguf_tokenizer_consistency.py`；HF 计数的 129283 是 bos/eos/pad 重复声明）。
- pre-tokenizer：`joyai-llm` = DEEPSEEK3_LLM 正则集（llama.cpp `llama-vocab.cpp:320`），
  与 tokenizer.json 的 Split 序列一致；clean_spaces=false；add_bos=false；EOS=1。
- chat 不用 Jinja：官方 `encoding/encoding_dsv4.py` + 4 组测试向量在
  `dsv4flash-ref/encoding/tests/`。

## 6. 显存与容量（RTX PRO 6000 Blackwell，95.6 GiB 可用）

- 权重 packed 常驻：81.86 GiB（禁止 BF16 反量化缓存，余量红线见方案 §1.4）。
- llama.cpp 实测（冷进程，2048 ctx，全层 offload）：**85,909 MiB**，EXIT 0。
- KV 测算（21 个 ratio-4 层 + 20 个 ratio-128 层 + 43 层窗口环 128×584 B；
  压缩条目 584 B = 448 FP8 + 64×2 rope BF16 + 8 scale；indexer K 68 B/条）：
  **单槽 156K ≈ 0.58 GiB；单槽 256K ≈ 0.92 GiB**（此前按 19/19 层算得偏低，
  已按 21/20 更正）。余量 97,887 − 85,909 ≈ 11.7 GiB → 256K 理论 ~12 槽、
  保守 10 槽；156K 理论 ~19 槽、保守 12-16 槽。本 runtime 实测留待 Phase 3。

## 7. Oracle 基线

- **llama.cpp**（build-sm120，79bba02a，SM120 CUDA）：加载、greedy、thinking 模式
  生成全部正常。首测性能（2K ctx，96 tok，未调优）：**prefill 19.4 t/s / decode 15.7 t/s**。
  日志 `/home/bot/models/llama_smoke.log`。
  llama.cpp 加载器约束（deepseek4.cpp:44-56）：compress_ratios 项数 ≥ 层数、
  值域 {0,4,128}、gating 必须 sqrtsoftplus、SWA pattern 0（全层窗口）。
- **sparkinfer fork DSV4 算子**：compressed_mla/sparse_mla/mla_compressed/nsa_indexer
  本机 **67/67 通过**（日志 `/home/bot/models/sparkinfer_dsv4_tests.log`）。
- 官方 reference（tilelang 0.1.9）可 import；`fast_hadamard_transform` 安装失败，
  需要时用 torch Sylvester-Hadamard 等价替换（只影响 indexer 路径）。

## 7.5 部件级语义对拍（tests/test_dsv4_reference_parts.py，GPU 实测）

用真实 GGUF 权重初始化官方 reference 模块，与我们独立写的实现对拍：

- **Gate（打分路由层）**：`softplus(logits).sqrt()` → 偏置只进 top-k **选择**、
  不进权重 → 无偏分数 gather → renorm × 1.5。**位精确一致**（indices + weights
  全等）。不变量：每 token 权重和恒为 1.5（route_scale）。
- **Gate（哈希层 0-2）**：`indices = tid2eid[input_ids]`（[129280,6]），
  **但 gate logits 照常计算、权重仍从中 gather**（"跳过选择，不跳过 gate"）；
  **位精确一致**。
- **hc_split_sinkhorn**（tilelang kernel vs torch 复现，fp32）：mixes[24] =
  pre(4) | post(4) | comb(16)；`pre = sigmoid(m·scale[0]+base)+eps`，
  `post = 2·sigmoid(m·scale[1]+base)`，comb = softmax(行)+eps → 列归一 →
  19 轮（行归一 → 列归一）。容差内一致（归约序差异）。
  ⚠️ **已验证的微妙事实：循环以列归一化收尾，终态列和≈1、行和自由漂移
  （实测 0.92–1.08），并非双随机矩阵——实现时不要"修正"成对称归一。**
- 尚未对拍：Compressor（overlap/APE/decode 状态机）、Indexer、注意力聚合
  （Phase 2/3 随模型图落地时补对拍）。

## 8. 尚未实测（后续阶段的门禁项）

- 本 runtime 内的权重上卡占用与 KV 预算（Phase 2/3，冷进程）。
- 贪心逐 token 对齐（Phase 3：我们的 kernel vs llama.cpp，同 artifact）。
- pre-tokenize/encoding 端到端 round-trip（Phase 4 用官方 4 组向量）。
- DSpark 伴生模型（Phase 5，尚未下载）。
