# DSV4 服务化端到端验证与 bf16-q prefill+indexed 内核缺陷（2026-08-09）

状态：🟢 服务化可用（缺省内核）；🔴 bf16-q 模式的 prefill+indexed 流有内核缺陷（fork，待修）。

## 1. 端到端服务验证（真实 82 GB GGUF）

启动：`QSR_SERVER_MODEL_PATH=<...>.gguf` + `QSR_DSV4_PREFILL_ROWS=32`，
`--port 8001 --no-cudagraph --no-prefix-cache`。engine ready，91.9 GiB 常驻。

请求验证：

| 请求 | 输出 | 结果 |
|---|---|---|
| "what is 2+2? Answer in one word." | reasoning: "four" | ✅ 正确 |
| "Write the numbers from one to five..." | reasoning: "1, 2, 3, 4, 5" | ✅ 正确 |
| "capital of France" | reasoning: "Paris" | ✅ 正确 |
| "Count from 1 to 3 and then write DONE" | reasoning: "1, 2, 3, DONE" | ✅ 正确 |

reasoning/content 分流正确：模型把答案放在未闭合 `<think>` 块内然后 EOS
（IQ2_XS checkpoint 的固有行为——聊天模式编码器在 `<｜Assistant｜>` 后加
`</think>`，模型仍自行开 `<think>`），服务器按 `find_reasoning_span` 规则
（必须以 `<think>` 开头才算 reasoning）把整段放进 `reasoning_content`。

## 2. 服务路径构成

- registry `deepseek_v4` 解锁（`IMPLEMENTED_BACKENDS`）
- `ServerEngine._load_deepseek_model` 第三分支；tokenizer 用官方
  `tokenizer.json`（`QSR_DSV4_TOKENIZER_DIR`，EOS=1 无 BOS）
- 消息编码走 vendored 官方 `encoding_dsv4.py`（`thinking_mode="chat"`）
- 后端多槽 kernel-path：每槽 43 层 `Dsv4AttnKernelLayer`（权重与 eager 模型共享）
- chunked prefill：MLA scratch 按 `QSR_DSV4_PREFILL_ROWS`（默认 32）行规划，
  长 prompt 按 min(max_q_rows, window)=32 行分块；压缩器/索引器中间序列
  状态机逐 token 推进，与顺序 oracle bit-exact（`tests/test_dsv4_chunked_prefill.py`）

## 3. 🔴 bf16-q 模式的 prefill+indexed 流缺陷（fork kernel）

**现象**：`SPARKINFER_MLA_DSV4_BF16_Q=1` 下，真实模型 layer 2（ratio-4）prefill
输出 NaN；隔离探针 14 行 SWA-only 干净（cos 0.9999965），加上 indexed（压缩）
流后 cos 崩到 **0.4366**（无 NaN 但严重错误，全模型里传播成 NaN）。

**范围**：`s1_qk_nope_dsv4_bf16`（bf16-Q QK）/ `s6_xv_nope_dsv4_bf16`
（bf16-PV XV）对 indexed 候选流的 prefill（多行）处理错误；单 token decode
正常（此前 CG 探针 0.999996 验证过）。

**根因已隔离（2026-08-09 复测）**：`SPARKINFER_MLA_SM120_NUM_SPLITS=1` 强制
单 split 后 bf16-q 的 14 行 prefill+indexed **cos=1.0（完美）**；auto splits
（>1）时 cos=0.19。同形状 fp8 路径 splits=auto 和 =1 均 cos=1.0。即
**bf16-q 变体在 num_splits>1（split-K 多 CTA 部分归约）下产生错误的
partial O/LSE**，而 fp8 路径正确。单 split 时（小行数或强制）bf16-q 正确。
疑点：`s6_xv_nope_dsv4_bf16` 自建 sm_p_full 的 barrier 与 fp8 路径的
double-buffer/split 调度在 num_splits>1 时可能冲突；`s1_qk_nope_dsv4_bf16`
的 global Q 读本身正确（fp8 对照佐证 split 机制本身没问题）。

**影响**：服务用**缺省内核**（bf16_q=0）已正确（上述端到端 ✅）；bf16-q
数值改进模式对 serving 不可用，直到 fork 修好 split-K>1 路径。

**待修方向**：修 fork 的 bf16-q QK/XV 在 split-K>1 下的 partial 归约
（先复现最小 split=2 用例，再查 s6 自建 sm_p_full 与 fp8 barrier/缓冲
切换的冲突）。

## 4. 遗留

- bf16-q prefill+indexed 修复（fork，待排期）
- 与 llama.cpp 端到端贪心对齐（质量基线）
- 性能优化：IQ2_XS/Q8_0 的 kernel 内 dequant（原生计算，不做 BF16 反量化常驻）
