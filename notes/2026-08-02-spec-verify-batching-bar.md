# 批处理大投影被否决得太早：bit-exact 不是这里的判据

日期：2026-08-02 · 分支 `work/bitexact-bar-20260802`（**未合并 main**）·
测量环境：RTX PRO 6000 Blackwell Max-Q，`nvidia/Qwen3.6-27B-NVFP4` 真权重

> 本文档区分 **【实测】**（本仓库或本机源码里跑出/读到的）与 **【推算】**（有明确
> 假设和算式的外推）。表格数字未标注即为【实测】。

---

## 0. 结论先行

`notes/2026-08-02-gdn-spec-forward-batching.md` 与 `docs/implementation-plan.md` §7.1 B3
把 `in_proj_qkv`/`in_proj_z`/`out_proj` 留在逐位置循环里，理由是批处理"破坏
bit-exact"。**这个理由不成立，三条独立证据：**

1. **它要保的那个 bit-exact 在 `verify_forward` 里早就不存在了。** 同一个
   `verify_forward` 对全部 64 层调用 `layer.mlp(hidden_states)`、
   `layer.input_layernorm(hidden_states)`、`layer.self_attn(...)`，输入是整块
   `[1, K, hidden]`——正是"输出维几千的批量 `F.linear`"。实测这些**全部不 bit-exact**。
   两层实盘堆叠后（**用未改动的 shipped `spec_forward`**）：layer 1 的递归状态与顺序
   解码差 **0.0117**、**72.7% 元素不同**。而批处理三个大投影只让 `spec_forward` 自己的
   输出动 **≤0.00049**。**被保护的东西，上一行代码已经花掉了。**
2. **两种舍入没有一个"更对"。** 对同一批 BF16 输入和同一份反量化 BF16 权重做 FP64
   点积作为真值：逐位置循环的 RMSE **恰好等于**"把 FP64 真值直接舍成 BF16"的 RMSE
   （6 位有效数字相同，即**它就是正确舍入**）；批量 GEMM 在唯一有差别的那个投影上是
   **0.066 ULP vs 0.047 ULP**（输出量纲的 ULP）。**两者都在半 ULP 的表示极限之内**，
   排不出高下——这与 `docs/b1-correctness-criterion.md` §1 判定原 B1 门禁"要一个不存在的
   东西"是同一件事。
3. **上游两家都批处理，且都不要求 bit-exact。**（§4，带源码行号）

**递归状态是否该用不同标准？答案是"是，但结论仍然是不需要 bit-exact"**（§3）：
一次**满 1 ULP** 的状态扰动跑 96 步只长到 **2×**，绝对值 **0.00098**——而 runtime 自己
每一步都做的 fp32→bf16 状态回舍，96 步累积到 **0.00162**。**这个改动往状态里注入的噪声，
比设计已经故意注入的噪声还小（0.60×）。**

**代价与收益**：单层 `spec_forward` **K=8 3.41ms → 1.69ms（2.02×）**、
**K=16 5.76ms → 1.83ms（3.15×）**（另一次运行 2.40×/3.70×，同进程对比稳定）。

---

## 1. `verify_forward` 里的 bit-exact 早就没了（这是最关键的一条）

`runtime/model/qwen36_model.py` 的 `Qwen36TextModelSelfBuilt.verify_forward`
（L2121-2143）对每一层做：

```python
hidden_states = layer.input_layernorm(hidden_states)      # [1, K, 5120] 整块
...  spec_forward(...)  或  layer.self_attn(...)          # 整块
hidden_states = layer.mlp(hidden_states)                  # [1, K, 5120] 整块
```

`Qwen36MLP.forward` 是 `down_proj(silu(gate_proj(x)) * up_proj(x))`，
`intermediate_size = 17408`。**这就是"输出维几千的批量 `F.linear`"本身。**

### 1.1 实测：这些 sublayer 批处理后都不 bit-exact（K=16，真权重）

| 子模块 | 输出维 | bit_exact | max_abs_diff | 差异元素占比 |
|---|---:|:--:|---:|---:|
| `layers.0.mlp`（整体） | 5120 | ❌ | **0.0078125** | **57.1%** |
| `layers.0.mlp.gate_proj` | 17408 | ❌ | 0.00195 | 26.9% |
| `layers.0.mlp.down_proj` | 5120 | ❌ | 0.00195 | 0.2% |
| `layers.0.input_layernorm` | 5120 | ❌ | 0.0078125 | <0.05% |
| `layers.3.self_attn.q_proj` | 12288 | ✅ | 0 | 0% |
| `layers.3.self_attn.k_proj` | 1024 | ❌ | 0.000244 | <0.05% |
| `layers.3.self_attn.v_proj` | 1024 | ❌ | 7.6e-6 | <0.05% |
| `layers.3.self_attn.o_proj` | 5120 | ❌ | 0.000977 | 0.1% |

⚠️ 顺带纠正原 note 的两条：`q_proj` 输出维 12288 **却是 bit-exact 的**——所以
"输出维 512 是分界线"不是一条可依赖的规律，它取决于 cuBLAS 为具体 (M, N, K) 选到哪个
kernel；而 `Qwen36RMSNorm` 也**不是**逐行绝对安全的（原 note 说 norm 批处理 bit-exact，
那是对 `Qwen36RMSNormGated` 在 `head_v_dim=128` 上测的，`input_layernorm` 在 5120 上
会漏出来）。

### 1.2 实测：两层堆叠后，shipped 代码的递归状态就已经不 bit-exact

把 layer 0/1（都是 `linear_attention`）连同真实 MLP 与真实 layernorm，按
`verify_forward` 的顺序堆起来，跑两遍——**一遍整块 K=16（verify 方式），一遍逐 token
（顺序解码方式）**，`spec_forward` 用的是**未改动的 main 代码**：

| 对象 | bit_exact | max_abs_diff | 差异元素占比 |
|---|:--:|---:|---:|
| layer 0 `recurrent_state`（走完 K 之后） | ❌ | 0.00195 | 3.4% |
| layer 0 `conv_state` | ✅ | 0 | 0% |
| **layer 1 `recurrent_state`** | ❌ | **0.01172** | **72.7%** |
| **layer 1 `conv_state`** | ❌ | **0.03125** | **62.7%** |
| 两层输出 hidden | ❌ | 1.0 | 54.8% |

- layer 0 的差异来自 `input_layernorm` 批处理（§1.1 最后一行），不是投影。
- layer 1 的差异来自 layer 0 的 MLP 批处理，量级比投影批处理大 **24×**。
- 48 个 GDN 层里，**只有 layer 0 的输入没被前面的批量 MLP 污染**。

**因此 `docs/implementation-plan.md` §7.1 B3 里那句"回滚后状态与顺序非投机解码
`max_abs_diff=0.0`"，只在"单独一层、两边喂完全相同的输入"这个人造条件下成立，
在 `verify_forward` 里不成立。** 这不是 bug（B3 判据本来也不要求 bit-exact，
见 `docs/b1-correctness-criterion.md` §7），但它意味着**用 bit-exact 否决投影批处理
是在保护一个已经不存在的性质**。

复现：`scripts/b3_probe_batching_bar.py`（Part A / Part E）。

---

## 2. 两种舍入哪个"更对"——FP64 oracle

对同一份 BF16 输入 `x` 与同一份反量化后的 BF16 权重 `W`，用 **FP64** 算
`x @ W.T` 作为真值，比较三种 BF16 结果（K=16，真权重）：

| 投影 | 输出维 | RMSE 逐位置 | RMSE 批量 | RMSE `_bmm_project` | **RMSE 理想舍入（不可避免下界）** |
|---|---:|---:|---:|---:|---:|
| `in_proj_qkv` | 10240 | 1.77794e-4 | 1.77795e-4 | 1.77794e-4 | **1.77794e-4** |
| `in_proj_z` | 6144 | 1.83003e-4 | 2.59173e-4 | 1.83003e-4 | **1.83003e-4** |
| `in_proj_b` | 48 | 1.03936e-4 | 1.48964e-4 | 1.03936e-4 | **1.03936e-4** |
| `out_proj` | 5120 | 1.96261e-4 | 1.96261e-4 | 1.96261e-4 | **1.96261e-4** |

换算成"输出量纲的 ULP"：

| 投影 | 逐位置 | 批量 | 理想舍入 |
|---|---:|---:|---:|
| `in_proj_qkv` | 0.0455 | 0.0455 | 0.0455 |
| `in_proj_z` | 0.0468 | **0.0663** | 0.0468 |
| `in_proj_b` | 0.0532 | **0.0763** | 0.0532 |
| `out_proj` | 0.0063 | 0.0063 | 0.0063 |

读法：

- **逐位置循环的 RMSE 与"直接把 FP64 真值舍成 BF16"的 RMSE 完全相同**——它是正确舍入，
  没有额外误差。这一点对逐位置循环是加分项，必须如实写出来。
- **但批量 GEMM 的额外误差只有 0.02 ULP 量级，且四个投影里有两个（含最大的
  `in_proj_qkv`）连这点差别都没有。** 半 ULP 是任何 BF16 表示的误差下界；
  两个都落在下界的 1/6 以内的实现，谈不上谁"正确"。
- `_bmm_project` 在这两个大投影上与逐位置**逐位相同**（RMSE 到 6 位有效数字一致）
  ——这一点原 note 的结论（"bmm 在 >512 就不 bit-exact"）在这组输入上没复现出来，
  但它不影响本文结论，因为本文主张的正是"不需要 bit-exact"。

---

## 3. 递归状态该不该用不同的标准

**协调者的直觉是对的：这两类对象确实不该一刀切。** 投影的前向输出被消费一次就没了；
递归状态跨步存活，误差会被后续所有 token 继承。所以专门测了它。

### 3.1 一次 1 ULP 扰动，96 步之后

给 `recurrent_state` **每个元素**注入 **±1 个完整 BF16 ULP**（随机符号；这是
"一次舍入之差"能造成的最大扰动，是任何归约顺序变化的上界），然后跑 96 步普通解码：

| | step 1 | step 16 | step 32 | step 64 | step 96 |
|---|---:|---:|---:|---:|---:|
| `max_abs_diff`(state) | 0.000488 | 0.000732 | 0.000732 | 0.000977 | **0.000977** |
| 相对 max\|S\| | 0.685% | 0.877% | 1.136% | 0.637% | 0.752% |

**96 步只长到 2.0×，而且是随机游走式的、不是指数发散。**

### 3.2 对照：runtime 自己每一步都在注入的噪声

`Qwen36GatedDeltaNet.forward` 的 `state.recurrent_state.copy_(last_state)`
把 FLA 内部的 FP32 结果**每步回舍成 BF16**（B0-4/B0-7 的刻意设计，对齐 transformers）。
把同一条轨迹用 FP32 状态再跑一遍作为对照：

| | step 1 | step 96 |
|---|---:|---:|
| bf16 状态 vs fp32 状态 | 0.000121 | **0.00162**（相对 1.240%） |

**结论：一次满 1 ULP 的状态扰动，跑 96 步后是 0.00098；
runtime 自己的每步回舍，同样 96 步后是 0.00162。比值 0.60×。**

即：**批处理往递归状态里注入的噪声，比这个设计已经故意注入、并且 B1-R 全绿通过的噪声
还要小。** 递归状态确实值得单独审，审完的结论仍然是"不需要 bit-exact"。

⚠️ 一个必须写清楚的量级对照：`docs/b1-correctness-criterion.md` §5.3 里
`gdn-state-decay:1e-2`（每步乘 0.99）是**红**的。它看起来"也只有 1%"，但那是**每步
同向复利**——96 步后状态被压到 0.38 倍（62% 误差），与这里"一次性、随机符号、
96 步后 0.75%"完全不是一个东西。同一节里 `gdn-state-decay:1e-3` 是**绿**的，
理由是"1.0 附近 BF16 的 ULP = 0.39%，0.1% 舍回原值"——**本改动的扰动正是 1 ULP 量级，
即那条绿线所在的量级。**

---

## 4. 上游怎么做（本机源码，不是回忆）

读的是本机两棵源码树：vLLM `/home/bot/vllm`，SGLang `/home/bot/project/sglang`。

### 4.1 verify 就是一次批处理前向，M = Σ(k_i+1)

- **vLLM**：`vllm/v1/worker/gpu_model_runner.py:2185-2216` 把投机 token 排进
  **同一次** `self.model(...)`，`logits_indices` 从这一次前向里 gather 出 K+1 个采样位置；
  `:2802-2810` 的注释给了 flatten 后的行号布局示例。
  `:840` 直接写 `self.uniform_decode_query_len = 1 + self.num_spec_tokens`。
  **即：verify 的 GEMM 行数是 Σ(k_i+1)，非投机 decode 是 num_reqs——上游早已接受
  行数相关的归约顺序差异。**
- **SGLang**：`srt/model_executor/forward_batch_info.py:89-115` 把
  `ForwardMode.TARGET_VERIFY` 归类为 **extend**（prefill 形状）而不是 decode；
  `srt/layers/attention/hybrid_linear_attn_backend.py:396-405` 的
  `verify_query_start_loc` 以 `draft_token_num` 为步长。

### 4.2 正确性判据：全是分布/准确率，没有一处 bit-exact

- vLLM `tests/v1/e2e/spec_decode/test_spec_decode.py`：贪心采样下
  **EAGLE 只要求 60% 的 prompt 文本与非投机参照相同**（`:484`
  `assert matches > int(0.6 * len(ref_outputs))`）；**MTP 要求 80%**
  （`:31, :891-893`，`MTP_SIMILARITY_RATE = 0.8`，注释原文 "Heuristic: expect at least
  80% of the prompts to match exactly"）；ngram/suffix 只看 GSM8K 准确率门槛。
  Medusa 只看接受率 `assert acceptance_rate >= 0.198`。
  该文件里的 `torch.equal` **全部作用在整数索引/元数据张量上**，没有一处作用在激活上。
- SGLang 最严的一条是 `python/sglang/test/kits/spec_server_kits.py:130-183` 的
  `SpecParityKit`：4 条短 prompt × 48 token 的**输出文本**必须与非投机参照相同——
  **而这条在 XPU 与 CPU 后端被显式关闭**，理由写在注册处：
  `test/registered/spec/eagle/test_spec_eagle_parity.py:16-21`
  `disabled="EAGLE3 numerical parity mismatches on XPU"`，
  `test/registered/cpu/test_spec_eagle_parity_cpu.py:10-14`
  `disabled="EAGLE3 numerical parity mismatches on CPU intel_amx"`。
- **与我们架构最接近的一条外部锚**：SGLang 对 **Qwen3-Next（GDN 混合）+ MTP** 的门禁是
  KL 散度阈值——`test/registered/models_e2e/test_qwen3_next_models_mtp.py:22-23, 49-50`：
  `kl_div_thres = 0.0035`（topk=1）/ `0.008`（topk=4 树形），
  外加 `gsm8k_accuracy_thres = 0.93`。**即：同架构、同投机方式，上游要的是 KL < 3.5e-3，
  不是 bit-exact。**

### 4.3 vLLM 官方文档把这件事写死了

`docs/features/speculative_decoding/README.md:186-218`：

> 1. **Theoretical Losslessness** — Speculative decoding sampling is theoretically
>    lossless **up to the precision limits of hardware numerics**. Floating-point errors
>    might cause slight variations in output distributions …
> - **Batch Size and Numerical Stability**: Changes in batch size may cause variations
>   in logprobs and output probabilities …

`docs/usage/faq.md:20-29` 更直接地点名：

> the same requests might be batched differently due to … **batch expansion in
> speculative decoding**. These batching variations, combined with numerical instability
> of Torch operations, can lead to slightly different logit/logprob values at each step.

**vLLM 的 "lossless" 明确限定在算法层（rejection sampling 的代数），从不承诺数值层。**

### 4.4 递归状态回滚：上游连"回滚"都不做，直接按 index 取

这一条对我们最有用，因为它正面回答"递归状态要不要 bit-exact"。

- **vLLM**：在**同一次批处理 verify 前向里**把 K+1 个候选位置的状态全部算出来写进
  K+1 个 slot，接受后按 `num_accepted_tokens - 1` 直接**索引选取**：
  `vllm/model_executor/layers/mamba/ops/mamba_ssm.py:333-354`
  （`init_token_idx = tl.maximum(num_accepted - 1, 0)`）、
  `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:103-116`（GDN 门控同款）、
  `vllm/v1/attention/backends/mamba_attn.py:50-52` 的注释原文
  `# Number of accepted tokens for each spec sequence (for loading correct checkpoint)`。
  `causal_conv1d.py:836-854` 有一段把机制讲得最清楚的注释（conv_state 的滑窗滚动）。
- **SGLang**：`srt/speculative/spec_utils.py:660-701` 的
  `commit_mamba_states_after_verify` + `hybrid_linear_attn_backend.py:1010-1048` 的
  `fused_mamba_state_scatter_with_mask`，存储是
  `memory_pool.py:541-554` 的 `[num_layers, size+1, speculative_num_draft_tokens, ...]`。
- **两家都没有任何"恢复出来的状态要与顺序解码相等"的检查。** 而且它们的 verify 走的是
  chunk/并行扫描形式，**恢复出来的状态本来就与逐 token 递归的结果不同**，照样发布。

### 4.5 批不变性：两家都承认不成立，都是默认关闭的可选项

- vLLM `VLLM_BATCH_INVARIANT`，默认 `False`（`vllm/envs.py:89`），文档标 beta；
  `model_executor/layers/batch_invariant.py:907-919` 的注释直接点名 split-k 与
  reduced-precision reduction 是根因。**其 determinism 测试套件里没有任何投机解码。**
- SGLang `--enable-deterministic-inference` 默认 `False`；而且
  `srt/arg_groups/speculative_hook.py:601-606` **直接把它和投机的 rejection sampling
  判为互斥并抛错**。

---

## 5. 全模型实测：logits 上的 gap error 与 τ

> 这一节是本文的主证据：把批处理版投影接上，跑 B1-R 的 R1 指标，
> **参照系是我们自己的非投机路径**（`docs/b1-correctness-criterion.md` §7 B3 的规定）。

本次运行（一次模型加载，3 个工作负载 × 224 个 verify 位置 = **672 步**，K=16）：

- `sequential` —— 普通贪心解码，每次 1 token（B1 已证明的路径）。**它是下面所有比较的 oracle。**
- `verify_shipped` —— 把同一串 token 按 K=16 分块喂进 `verify_forward`，用**今天的**
  `spec_forward`（三个大投影逐位置循环），每块 `commit_verify(accepted_count=K)`。
- `verify_batched` —— 完全相同，只是那三个投影批处理。

步锁（两侧被强制走同一串 token）与 `accepted_count=K` 一起，把"接受率"这个混淆项彻底
移除：两条 verify 路径走的 verify 块调度**完全相同**。

### 5.1 三组比较，全部 672 步

| 指标 | **shipped vs 顺序解码**（今天已在跑的） | **batched vs 顺序解码** | **batched vs shipped**（本改动自己的足迹） | **τ（B1-R §6.1 校准阈值）** | B1-R 控制组（我们 vs HF） |
|---|---:|---:|---:|---:|---:|
| `median_gap_error` | 0.125 | **0.125** | 0.125 | 0.25 | 0.125 |
| `p90_gap_error` **（主）** | 0.250 | **0.250** | 0.250 | **0.5** | 0.250 |
| `p99_gap_error` | 0.375 | **0.375** | 0.375 | 6.0 | 3.375 |
| `max_gap_error` | 0.750 | **0.625** | 0.875 | 20.0 | 10.56 |
| `mean_kl_topk` **（主）** | 2.59e-4 | **2.87e-4** | 2.85e-4 | **5e-3** | 1.58e-3 |
| `max_tie_slack_ulps` **（主）** | 1.0 | **1.0** | 1.0 | **32** | 8 |
| `disagreement_rate` | 0.74% | 0.89% | 0.45% | 3% | 0.77% |
| `p90_logprob_error` | 0.1875 | 0.2112 | 0.1942 | 0.5 | 0.250 |

**逐负载**（p90 / p99 / mean KL / 翻转数）：

| 负载 | shipped vs 顺序 | batched vs 顺序 | batched vs shipped |
|---|---|---|---|
| prose | 0.1250 / 0.2500 / 4.05e-4 / 1 | 0.1875 / 0.2500 / 3.98e-4 / **0** | 0.1875 / 0.2500 / 3.78e-4 / 1 |
| code | 0.1875 / 0.2500 / 1.08e-4 / 3 | 0.1250 / 0.2500 / 1.07e-4 / 4 | 0.1250 / 0.2500 / 1.20e-4 / 1 |
| instruction | 0.2500 / 0.5000 / 2.65e-4 / 1 | 0.2500 / 0.4375 / 3.55e-4 / 2 | 0.2500 / 0.5625 / 3.56e-4 / 1 |

### 5.2 读法

1. **主判据 `p90_gap_error` 在三组里完全相同：0.250，bar = 0.5，余量 2.0×。**
   批处理**没有让任何一个分位数变差**——median/p90/p99 三个分位数三组一字不差，
   `max_gap_error` 反而从 0.750 降到 0.625。
2. **本改动自己的足迹（batched vs shipped，p90 = 0.250）与今天已经存在、已经被接受的
   那个足迹（shipped vs 顺序解码，p90 = 0.250）一样大。** 即：把这三个投影批处理，
   往 logits 里加的东西与 verify 相对顺序解码本来就有的东西同量级。
3. **全部 8 条 gap-error 类 bar 全绿，最小余量 2.0×，最大 32×。**
   （脚本报的 `passes: False` 只有两条原因，都是 `nll_relative_excess` 与
   `min_logits_cosine` "gated but was not measured"——那是 B1-R 的 R4/R3 两条腿，
   本脚本按设计不测。**没有任何一条 gap-error bar 被触发。**）
4. **翻转全部是真平局**：三组的 `max_tie_slack_ulps` 都是 **1.0 ULP**——
   意思是"我们的 logits 只要动 1 个 ULP 就复现 oracle 的排序"，而 BF16 表示不出比
   1 ULP 更小的移动。对比 B1-R 控制组（我们 vs HF）是 8 ULP，最弱可检出注入是 106 ULP。
5. **`docs/b1-correctness-criterion.md` §7 对 B2 提的"τ_B2 = τ_B1/10"这条更严的建议
   （同一份数学、同一张卡，噪声底应当远低于我们 vs HF）在这里得到了实测支持**：
   p99 3.375 → 0.375（9×）、max 10.56 → 0.75（14×）、mean KL 1.58e-3 → 2.6e-4（6×）、
   slack 8 → 1（8×）。**但把 τ/10 当作 bar 会把 shipped 路径一起判红**（它的 p90 也是
   0.250 > 0.05），所以那不是一条能区分本改动的 bar，只是印证了噪声底确实低一个量级。
6. **外部锚**：SGLang 对 **Qwen3-Next（同为 GDN 混合）+ MTP** 的门禁是
   `kl_div_thres = 0.0035`（topk=1）。我们三组的 `mean_kl_topk` 都在 **2.6e-4 ~ 2.9e-4**,
   **在其阈值内 12 倍以上**。（两者 KL 的定义口径不同——SGLang 比的是 prefill 重打分
   vs decode 的 logprob，我们比的是 top-K 并集上的 KL——只作数量级锚点，不作等价换算。）

⚠️ `max_drift_ratio` 三组都是 1.0，那是"步数不足两个 128 窗口"时的返回值
（每负载 224 步 < 256），**不是测出来的 1.0**。趋势这条腿本次没有信息量。

复现：
```bash
~/.venvs/vllm/bin/python -u scripts/b3_verify_batching_logit_agreement.py --k 16 --steps 224
```
原始数据 `.bfdiag/runs/b3_verify_batching_agreement.json`（未入库，600 步 × 3 组的逐步记录）。

### 5.3 顺带得到的全模型吞吐数（不是外推）

同一次运行里，每个负载 14 轮 K=16 的 verify（**全 64 层 + `lm_head` + 逐位置 top-1024
落盘**，不只是 GDN）：

| 负载 | verify shipped | verify batched | 加速 |
|---|---:|---:|---:|
| prose | 7.4s | 4.7s | **1.57×** |
| code | 7.3s | 5.2s | **1.40×** |
| instruction | 7.2s | 4.9s | **1.47×** |

换算成每层每轮：`(7.4-4.7)s / 14 轮 / 48 层 = 4.02ms`——
与单层探针实测的 `5.76 - 1.83 = 3.93ms`（K=16）**一致到 2%**。
**这是单层外推第一次被全模型数字交叉验证。**

---

## 6. MTP 加速比重估【推算，不是二次实测】

先把口径写清楚：`benchmarks/fixtures/qwen36_mtp_e2e_20260802.json` 是 **K=8** 且是在
**批处理落地之前**的代码上测的；`notes/2026-08-02-gdn-spec-forward-batching.md` 用
"每轮非 GDN 成本 = 实测每轮总时长 − 48 × 单层 spec_forward" 反推出非 GDN 项，本节沿用
**同一条算式和同一份 fixture**，只把 GDN 项换成本次实测的比值。

**输入**：
- fixture：prose 14 轮 / 11.765s / 非投机 4.152 tok/s；code 10 轮 / 10.200s / 非投机 5.159 tok/s，均 32 token。
- 那次会话的单层 K=8：OLD 6.975ms → 批处理后（= 今天的 main）4.087ms。
- 本次会话的单层 K=8：shipped 3.409ms → 全批处理 1.687ms，比值 **0.4948**。
  （绝对毫秒随 GPU 热状态漂移，所以用**同进程比值**而不是绝对值跨会话搬运。）
- 于是全批处理的等效值 = 4.087 × 0.4948 = **2.022ms/层**，48 层 = **97.1ms/轮**
  （今天的 main 是 196.2ms/轮）。

| prompt | 轮数 | 非 GDN/轮（反推） | fixture 实测 | 今天 main【推算】 | **全批处理【推算】** | **GDN 归零的上限【推算】** |
|---|---:|---:|---:|---:|---:|---:|
| prose | 14 | 505.5ms | 0.655× | 0.784× | **0.913×** | 1.086× |
| code | 10 | 685.2ms | 0.608× | 0.704× | **0.793×** | 0.905× |

**结论必须直说：批处理是真收益，但它不足以把 MTP 推过 1.0×。**

- prose 0.784× → **0.91×**，code 0.704× → **0.79×**。
- 再叠加已合入 sparkinfer 的多步融合 kernel（`1fd76d1`，对 `spec_forward` 再快 1.2–1.5×，
  K=8 下顺序递归约占 1.687ms 里的 0.4–0.6ms）：prose ≈ **0.94×**，code ≈ **0.81×**。
- **最右一列是硬上限**：假设 GDN verify 成本**归零**，prose 也只到 1.086×，
  **code 只到 0.905×——连 1.0× 都够不到。**

**所以"大投影不能批处理"从来不是 MTP 翻正与否的那个开关。** 批掉它们把 prose 可拿的
头寸（0.784 → 1.086 的空间）吃掉约 **60%**，但剩下的 505ms/685ms 每轮非 GDN 成本才是
决定项，它由三块构成（前两块在非投机路径上根本不存在）：

1. **MTP 链式起草** ~73ms/轮（K=8，`scripts/b3_probe_mtp_head.py` 实测单 step 9.18ms）；
2. **每轮末尾多跑一次完整顺序前向来推进 anchor** ≈ 240ms（prose）/ 194ms（code）——
   这是 `scripts/b3_mtp_e2e_acceptance_throughput.py` 的实现方式，**不是必然的**，
   把 anchor 折进 verify 块可以省掉；
3. **接受率本身**：prose 15.2%（2.21 token/轮）、code 35.0%（3.80 token/轮）。

**下一步该攻的是 2 和 3，不是 GDN。** 这与 `docs/implementation-plan.md` §7.1 B3 现在的
措辞（把大投影列为"MTP 翻正的真正门槛"）不一致，那句需要改。

⚠️ 本节是【推算】。**没有**重新跑全模型 e2e 基准来出第二个真实的 speedup 数字——
但与前一次外推不同的是，这次的 GDN 项比值有 §5.3 的全模型 verify 计时交叉验证
（每层每轮 4.02ms vs 单层 3.93ms，差 2%）。

---

## 7. 我没能验证的东西

1. **没有重新跑 `scripts/b3_mtp_e2e_acceptance_throughput.py` 出第二个真实 speedup。**
   §6 是算式，不是实测。要坐实需要一次带 MTP 头的全模型窗口。
2. **没有测采样（非贪心）路径。** `docs/b1-correctness-criterion.md` §7 给 B3 的第二条
   建议是"分布层面的 KS 检验"，本次只做了贪心步锁下的 logit 比较。
3. **`max_drift_ratio` 无信息**：每负载 224 步 < 2×128，该指标返回常量 1.0。
   要让趋势这条腿有力，需要每负载 ≥256 步。
4. **没有改 runtime 代码。** `spec_forward_batched` 是 `scripts/b3_probe_batching_bar.py`
   里的一份参照实现，逐行对齐 shipped 版本、只改三处投影，用于测量。
   真要落地需要把它写回 `Qwen36GatedDeltaNet.spec_forward` 并补测试。
5. **只测了 K=16（logits）与 K=8/16（单层计时）、batch=1、eager、无 CUDA Graph、
   ≤250 位置。** 长上下文与 CUDA Graph 捕获下的行为没有覆盖。
6. **`in_proj_z`/`in_proj_b` 的批量 GEMM 确实比逐位置多 ~0.02 ULP 的 RMSE**（§2）——
   本文主张这在半 ULP 的表示极限面前不可区分，但这是一个**判断**，不是"零差异"。
   如果将来某个下游对 1e-2 ULP 敏感，这条要重新审。
7. **`_bmm_project` 在 out=6144/10240 上本次测出与逐位置逐位相同**，与原 note "bmm 在
   >512 就失效"的结论不一致。没有去二分定界，也没有排除输入分布/cuBLAS 版本的影响——
   本文的结论不依赖它，但这个矛盾没有解决。
8. **两层堆叠探针用的是随机高斯输入，不是真实 hidden state**（§1.2）。
   全模型的 §5 才是真实输入下的最终判据；§1.2 的作用是定位机制，不是给量级定标。

---

## 8. 相关

- `scripts/b3_probe_batching_bar.py` —— §1/§2/§3 的单层探针（A/B/C/D/E 五部分）
- `scripts/b3_verify_batching_logit_agreement.py` —— §5 的全模型 gap error 测量
- `benchmarks/fixtures/b3_batching_bar_probe_20260802.json` —— §1/§2/§3 原始数据
- `docs/b1-correctness-criterion.md` §1/§5.3/§6.1/§7 —— 判据、噪声底、阈值、B3 参照系
- `notes/2026-08-02-gdn-spec-forward-batching.md` —— 被本文修正的那次结论
- `docs/implementation-plan.md` §7.1 B3 —— 需要按本文修正的两处措辞
