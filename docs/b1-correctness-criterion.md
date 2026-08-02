# B1 正确性判据（B1-R）：在"两个 bf16 实现必然在平局处分歧"下仍能抓 bug

> 编制：2026-08-02 · 分支 `work/b1-criterion-20260802`（**未合并 main**） ·
> 取代 `implementation-plan.md` §7.1 B1 门禁的字面写法
>
> 本文档区分两类陈述：**【实测】**= 本仓库或上游源码里读到/跑出的事实；
> **【判断】**= 我的设计取舍，可被推翻。凡是没有标注的表格数字都是【实测】。

---

## 0. 结论先行

**推荐判据 B1-R = 步锁强制解码下的 logit 一致性（R1）+ 平局审计（R2）+ 逐层余弦扫描（R3）+ 强制解码 NLL（R4），四条全绿才算过。**

原门禁"与 HF 贪心逐 token 对齐 ≥3 工作负载 × 512 token"**作废**，不是因为跑不过就放宽，
而是因为它要求的东西在 bf16 下不存在（§1）。B1-R 的核心变化是：

| | 原门禁 | B1-R |
|---|---|---|
| 两侧走的轨迹 | 各自自由生成，一旦分歧就是两条完全不同的文本 | **步锁**：两侧被强制走同一串 token |
| 每次运行产出的信息量 | 每个工作负载 **1 个数**（首次分歧位置） | 每个工作负载 **512 个数**（每步的 logit 差） |
| 对平局的态度 | 一次翻转即失败 | 翻转允许，但**翻转的置信度**必须在噪声底之内 |
| 没有翻转时 | 无信息（"过了"） | 仍然有 512 个连续量测 |

**实测结论（GPU，真权重，2026-08-02）：**

- **B1-R 可达**：当前实现在全部 11 条 bar 上通过，控制组 gap_error 中位数
  = **0.125 = 恰好 2 个 bf16 ULP**，NLL 与 HF 相差 **2.5e-5**（相对）。
- **B1-R 有区分力**：16 个注入配置里，7 个在噪声底之上的**全部变红**（每个红 5–10 条 bar），
  最弱的一个是"每 64 步漏更新一次 GDN 递归状态"。
- **6 个配置保持绿色是对的**，其中 2 个可证明是**恒等变换或格式舍不出来的量**（§5.5）。
- **方法合法性已实测**：用我们自己的 token 做步锁强制解码，与自由生成**逐位相同**（§5.1）。

⚠️ **B1-R 不覆盖两件事**：NVFP4 反量化（两侧共用同一份反量化结果）与长上下文（全部测量 ≤300 位置）。见 §8.2。

---

## 1. 为什么原门禁不可达（不是"当前跑不过"，是"数学上要不到"）

【实测，`notes/2026-08-02-b1-greedy-alignment-fails.md`】三个工作负载的首次分歧处，
两个候选 token 的 logit 间距分别是 **1、2、2 个 bf16 ULP**：

| workload | 我们选 | HF 选 | logit 间距 | 该量级 ULP | = 几个 ULP |
|---|---|---|---:|---:|---:|
| factual-short | 13901 | 1007 | 0.0625 | 0.0625 | **1** |
| math-short | 2441 | 11 | 0.1250 | 0.0625 | **2** |
| instruction-longer | 271 | 846 | 0.1250 | 0.0625 | **2** |

bf16 有 7 位显式尾数，在 `[8,16)` 区间 ULP = `2^(3-7) = 0.0625`。
**"两个候选相差 1 个 ULP"意味着它们是 bf16 能表示的最近的两个数。**
要求两个独立写出来的实现在这种比较上永远同侧，等价于要求它们逐位一致——
而逐位一致在两套不同 kernel、不同求和顺序下不可能，本仓库已经用自己的两条路径
证明过一次（`notes/2026-08-02-eager-verify-cg-verify-divergence.md`：CG verify 与
eager verify 都在单层注意力上以 cos≥0.999997 逼近稠密 oracle，全模型 logits 仍然翻转 argmax）。

而且贪心是自回归的：**一次翻转之后两侧走上完全不同的轨迹，后续 511 个 token 的比较不含信息**。
`overall_match_rate=0.3287` 这个数看着像"三分之一正确"，其实只是两段无关文本的偶然重合率。
**原门禁把一次 512 步的昂贵运行压缩成了 3 个数（首次分歧位置），其中 2 个还是良性平局。**

---

## 2. 上游怎么验一个新模型实现（【实测】本机源码，不是回忆）

读的是本机两棵真实源码树：vLLM `/home/bot/vllm` @ `e12b91b032`，
SGLang `/home/bot/project/sglang` @ `b296e1a503`。

### 2.1 vLLM：合入门禁只要求"能构造"，数值对比是可选项且带平局豁免

`vllm/docs/contributing/model/tests.md` 原文把 Model correctness 列在 **Optional Tests**，
Required 只有"能加载"（`tests/models/test_initialization.py`，1 层 dummy 权重，
注释明写 `# Avoid calling model.forward()`）。

真做数值对比时用 `check_logprobs_close`（`tests/models/utils.py:220-255`），核心循环：

```python
is_tok_mismatch = output_id_0 != output_id_1
if is_tok_mismatch or always_check_logprobs:
    ...
    assert output_id_0 in logprobs_elem_1, fail_msg
    assert output_id_1 in logprobs_elem_0, fail_msg
    ...
    # Break out since sequences will now diverge.
    break
```

即：**token 不一致是允许的，只要各自选的 token 在对方 top-5 里；然后立即 `break` 停止比较。**
`num_logprobs=5`、`max_tokens=32`、8 条 prompt。文本不一致只 `warnings.warn`，不 assert。

vLLM 对平局的处理方式是**消除触发条件而不是建模平局**：
- `tests/models/language/generation/test_common.py:131-138` 为 `TitanML/tiny-mixtral` 单独走
  native kernel，注释原文 "near-uniform router logits flip expert selection under ~1 ULP drift"；
- 同文件 `:146-153` 直接把 starcoder2 的 prompt 换掉，理由是自然语言 prompt 让代码模型的
  数字 logits 近乎均匀，"HF<->vLLM bf16 drift can reorder top-K"；
- `tests/models/quantization/test_fp8.py:44` 把 `max_tokens` 压到 **4**，注释
  "Due to low-precision numerical divergence, we only test logprob of 4 tokens"。

### 2.2 SGLang：逐值比 logprob，有明确的绝对容差

`python/sglang/test/runners.py:917-963` 的 `check_close_model_outputs`：

```python
assert torch.all(abs(hf_logprobs - srt_logprobs) < prefill_tolerance)
assert torch.all(abs(hf_logprobs - srt_logprobs) < decode_tolerance)
```

常量在 `test/registered/models/test_generation_models.py:48-57`：

```python
prefill_tolerance: float = 5e-2
decode_tolerance: float = 6e-2  # Increased to fix numerical error in issue #8614.
rouge_l_tolerance: float = 1
```

【实测的重要旁证】SGLang 还有一条 **KL 自洽门禁**（`python/sglang/test/kl_test_utils.py:98-117`），
把 `prompt + generated_ids` 重新 prefill，要求 prefill logprobs 与 decode logprobs 的平均 KL
低于阈值。它对 **Qwen3-Next-80B（同为 GDN 混合架构）** 的阈值是
`kl_div_thres = 0.002`（`test/registered/models_e2e/test_qwen3_next_models.py:38-44`）。
——**这说明"chunk 算法 vs recurrent 算法"的差异在平均 KL 上是 1e-3 量级**，
即使它在单点 logit 上能到 1.94。这个量级对本判据的 KL bar 是一个外部锚点。

### 2.3 两家真正的回归门禁都是任务准确率，不是 token 匹配

- vLLM GSM8K：`accuracy_threshold` 减 **绝对 0.08**（`tests/evals/gsm8k/test_gsm8k_correctness.py:169-185`）；
  lm-eval `DEFAULT_RTOL = 0.08`。
- SGLang GSM8K：阈值注释原文 "Thresholds are measured_score - 0.05"。
- vLLM 困惑度门禁：`PPL_TOL = 0.01`（相对，单边），`MAX_LENGTH = 1024`
  （`tests/models/language/generation_ppl_test/ppl_utils.py:18-19`），每个模型钉一个 HF 基线，
  例如 `Qwen/Qwen3-0.6B: hf_ppl=23.864173889160156`。

### 2.4 逐层比对：只有 SGLang 有，且是自己比自己

SGLang `srt/debug_utils/comparator/` 默认判据 `rel <= 0.001`，CI 里是
`test_nightly_precision_regression.py` 的日更漂移检查（对象是**自己的历史基线**，不是 HF）。
vLLM 里**不存在**任何逐层激活对比 harness。

### 2.5 对我们的直接含义

**原 B1 门禁比上游任何一家都严格得多**：vLLM 32 token + top-5 豁免 + 首次分歧即停；
SGLang 32 token + fp16 + prompt ≤100；我们要求 512 token 逐 token 全等。
**这不是"我们要求高"，是"我们要求了一个不存在的东西"。**

同时上游给了两条可以直接借用的东西：
1. SGLang 的 `|Δ logprob| < 5e-2/6e-2` —— 一个**外部校准过的**、量纲与我们主指标同类的常数；
2. vLLM 的 `PPL_TOL = 0.01` —— 困惑度作为"语义地板"的现成写法。

---

## 3. 判据 B1-R

### 3.1 R1（主判据）：步锁强制解码下的 gap error

**运行方式**：两侧被强制走**同一串 token**（HF 自己的贪心输出）。各自 prefill 一次 prompt，
然后每步喂 1 个 token，读该步 logits。**没有任何一侧的 argmax 能改变轨迹**，所以第 i 步的
比较在任何 i 上都是同前缀、同模式的对比。

**指标**：对每一步，取两侧各自 top-64 的并集为捕获集，再取两侧各自 top-8 的并集为比较集 `S`，
以**oracle 自己的 argmax** `a` 为锚点，

```
delta(i) = max_{t ∈ S} | (mine[t] - mine[a]) - (oracle[t] - oracle[a]) |
```

读法："两个实现对**可能成为下一个 token 的那几个候选**的相对分数，分歧多大？"

**为什么是相对分数而不是绝对 logit**：logits 整体加一个常数对 argmax 和 softmax 都是恒等变换，
把它算成"误差"是自找假阳性。锚点减法正好消掉它（这一条在测试里被钉死，
`test_constant_offset_is_invisible_by_construction`）。乘性缩放不会被消掉，仍然被测到。

### 3.2 R2（平局审计）：它是 R1 的推论，不是第二个可调旋钮

设我们的 argmax 是 `m1`、oracle 的是 `o1`（= 锚点 `a`），定义两个单边裕度：

```
mine_margin   = mine[m1]   - mine[o1]     >= 0
oracle_margin = oracle[o1] - oracle[m1]   >= 0
tie_slack     = mine_margin + oracle_margin
```

代入锚点定义后：

```
tie_slack = (mine[m1] - mine[a]) - (oracle[m1] - oracle[a])
```

——**这正是 `delta` 定义式里 `t = m1` 那一项（带符号）。** 因此恒有

> **0 ≤ tie_slack ≤ delta**，每一步都成立。

意义：`delta ≤ τ` **蕴含**"任何一次翻转的双边合计裕度都不超过 τ"。
所以"首次分歧必须是近似平局"这条判据是 R1 的**推论**，不需要单独定阈值、
也不会在"这次恰好没翻转"时失去约束力。`tie_slack` 换算成 ULP 后就是人能直接读的那个数：
**它是"我们的 logits 要移动多少才能复现 oracle 的排序"**，而 bf16 表示不出比 1 ULP 更小的移动。

代数推导在 `bfdiag/divergence/logit_agreement.py` 模块 docstring 里，
随机数据的性质检查在 `tests/test_bfdiag_logit_agreement.py::test_tie_slack_never_exceeds_gap_error_on_random_data`。

### 3.3 R1 的五条 bar，各自堵什么

| bar | 堵的失效模式 | 为什么单靠 `max_gap_error` 不够 |
|---|---|---|
| `max_gap_error` | 任何一步的置信分歧 | —— |
| `p99_gap_error` | 单点异常值（一次 kernel 抖动）不该独自否决整次运行 | max 对 1/1536 的离群点过敏 |
| `max_tie_slack_ulps` | 人类可读的"翻转有多离谱" | 是 R1 的推论，但报告里必须单独出现，否则没人看得懂 |
| `max_disagreement_rate` | **系统性小偏置**：每次翻转都在 τ 以内，但翻转频率远高于噪声应有的频率 | 见 §4 盲区 3 |
| `mean_kl_topk` | 概率质量整体搬家但 argmax 一次都不翻 | token 级比较对此完全失明 |
| `max_logprob_error` | 与 SGLang 5e-2/6e-2 可直接对比的外部锚 | 我们自己的数字只有一个样本 |
| `max_drift_ratio` | **GDN 递归状态的累积性 bug**：每步都在噪声底以下，但持续增长 | 步锁把自回归放大关掉了，趋势必须显式测 |

`max_drift_ratio` = 后 128 步 `gap_error` 中位数 ÷ 前 128 步中位数。
**这条是专门为这个架构加的**：GDN 的递归状态是唯一跨 decode 步存活的东西，
它的误差**累积**而不是平均掉。步锁在消除自回归放大的同时也消除了"错状态最终吐出乱码"这个
最容易看见的症状，所以趋势必须补回来。

### 3.4 R3：逐层余弦扫描（沿用已有 bfdiag，不新造）

`bfdiag/divergence/scan.py` + `thresholds.py` 已经在仓库里，
按子模块种类给出 layer-0 底线并按 `sqrt(depth)` 放宽（attention `cos≥0.9999`、
router `0.999`、MoE `0.95`）。B1-R 直接复用，作用有二：

1. **定位**：R1 红了之后告诉你是第几层、哪个子模块。注入实验里这一列的表现最直观。
2. **独立灵敏度**：浅层阈值远比 logits 层严格（layer-0 attention 要求 `cos≥0.9999`），
   一个只影响某一层的 bug 在被 64 层稀释进 logits 之前就会在这里被抓到。

限制【实测】：只对 **prefill** 的一次前向有效（capture 走 `capture_hidden_states=True`，
`seq_len>1` → chunk 分支）。它测不到 decode 路径，这正是 R1 存在的原因——
§5.4 用 5 个 GDN 递归状态注入实测坐实了这条：R3 对它们**全部失明**。

⚠️ 【实测，§5.6】**逐层 hidden_state 判定今天不可用作门禁**（在正确实现上必红，
且对注入 bug 零区分力），已定位到两个既有缺陷。**B1-R 只门禁 `logits` 这一项**：
控制组 cos=0.999873，drop-q-norm 0.995515，drop-k-norm 0.917519 —— 区分力充分。
逐层结果仅作定位线索使用，且 layer 63 的数字必须忽略。

### 3.5 R4：强制解码 NLL / 困惑度（语义地板）

对一段固定文本（本仓库用一段 ~270 token 的说明文，见
`scripts/b1_reference_trajectory.py::NLL_TEXT`）做**一次**全长前向，算
`-log p(token[i] | token[:i])` 的平均。两侧都是一次全长前向 → **两侧都走
`chunk_gated_delta_rule`**，模式匹配，比较合法。

它的价值不是精度，是**动态范围**：任何真实降级（RoPE 角度错、漏 norm、状态错）
都会让 NLL 显著上升，而这个量**不受平局翻转影响**（它读的是固定 target token 的概率，
不读 argmax）。判据形式抄 vLLM：相对超出 HF 基线不得超过 `PPL_TOL` 量级，单边。

> ⚠️ **不要把 R4 当成唯一判据。** 它只有一个标量，定位能力为零，
> 对"只在长上下文才发作"的 bug 不敏感（文本只有 253 token），
> 而且【实测，§5.4】**对全部 GDN 递归状态 bug 结构性失明**——
> 连"把状态完全冻结"都读不出来，因为单次前向里那个状态不会被再读一次。
> 这条已钉成测试 `test_the_nll_leg_is_blind_to_every_recurrent_state_bug`。

---

## 4. 盲区（明确写出来，不假装没有）

### 盲区 1：加性常数偏移——**故意的，不是缺陷**

`gap_error` 与 `logprob_error` 都对"所有 logits 整体加常数"完全失明。
这是设计意图：该变换对 argmax 与 softmax 都是恒等的，把它算成误差是制造假阳性。
（乘性缩放**不**在此列，会被测到。）

### 盲区 2：效应小于噪声底 τ 的 bug——**不可约，且已量化**

任何在 top-K 相对 logit 上效应始终 ≤ τ 的 bug 是不可见的。这与"逐位一致不可达"是同一件事的
另一面：参照系本身有噪声，噪声以下的东西测不到。
判据能给的最强承诺是：**它能抓到任何效应超过"两个正确 bf16 实现之间噪声底"的 bug。**

【实测，§5.5】这个底具体在哪里，已经扫出来了：

| bug 类型 | 可检出 | 检不出 | 检不出的原因 |
|---|---|---|---|
| GDN 状态漏更新 | 1/64 步 | 1/128、1/256 步 | 真·低于噪声底（步数越长越灵敏） |
| GDN 状态乘性衰减 | 1e-2 | 1e-3、1e-4 | **bf16 表示不出**（1.0 附近 ULP=0.0039），与判据无关 |
| RoPE theta 相对误差 | —（1e-2 都测不到） | 1e-2 ~ 1e-5 | ≤300 位置下相位影响 ≪1 弧度；**会随上下文增长** |
| RoPE 位置整体 +1 | —— | —— | **数学恒等变换**，绿色是正确答案不是漏检 |

### 盲区 3：协调者点名的那个——"整体偏移但每次都恰好在平局处翻转"

**是的，一个只看"分歧处是否平局"的判据对这个是盲的。** 具体构造：一个 bug 给某一类 token
的 logits 统一加 0.1，那么每次翻转的 `tie_slack` 都会落在 0.2 以内，
只要 τ ≥ 0.2 就每一次都被判为"良性平局"。

B1-R 用**三条**独立的 bar 堵这个洞，但没有一条是完全的：

1. **`p90_gap_error`（主）**：0.1 的系统偏移本身就是 0.1 的 gap error。
   控制组 p90 = 0.250，bar = 0.5——**一个 0.25 以上的系统偏移直接红**，
   不需要它翻转任何 argmax。这是最强的一条，而它成立的前提正是
   **τ 从噪声底反推（0.25×2），不是从"当前能过"反推（那会给出 10 以上）**。
2. **`mean_kl_topk`（主）**：系统偏移会改变整个分布形状，不只是排序。
   控制组 1.58e-3，最弱可检出注入 1.80e-2，bar 5e-3。
3. **`max_disagreement_rate`**：【实测，§6.2】**这条比预想的弱得多**。
   控制组 652 步只翻转 5 次（泊松 σ≈2.2），最弱可检出注入 11 次 ≈2.7σ；
   绿组最高 1.38% 与红组最低 1.69% 几乎贴住。
   **它保留但明确降级为"不可单独否决"**，见 §8.4——这正是"分离度不够就说明这条 bar
   在这个模型上不好用"的实测结论，不是把它藏起来。

### 盲区 4：反量化（NVFP4 nibble 打包）——结构性，与本判据无关

【实测】HF 侧的权重是**从我们自己反量化的张量拷过去的**（`b1_verify_greedy_alignment.py`
docstring 已写明；本判据的 phase 1 沿用同一条路径）。所以两侧权重逐位相同，
**任何反量化错误会在两侧同样地发生，B1-R 恒为绿**。
这不是可以调阈值解决的，需要一个独立 oracle，见
`notes/2026-08-02-b1-nvfp4-nibble-packing-unverified.md`。**B1-R 不覆盖这一项，
B1 验收不能只靠它。**

### 盲区 5：步锁本身抹掉的东西

步锁消除了自回归放大——这是它能给出 512 个可比样本的原因，也是它的代价：
"错一次就滚雪球"这个真实生产后果在 B1-R 里看不见。
**缓解**：`max_drift_ratio`（趋势）+ 自由生成首次分歧位置作为**报告项**（不作门禁，因为它是彩票）。

### 盲区 6：只测 batch=1、eager、无图、无投机、短上下文

B1 的定义域本来就是这样。长上下文（>2048）、并发、CUDA Graph、投机全部不在 B1-R 覆盖内，见 §6。

---

## 5. 判据有区分力吗——注入已知 bug 的实测

> **这一节是本文档的核心。** 判据抓不到注入的 bug 就是废的。
> 方法学依据：`e2e-and-quality-plan.md` §4 的三种"如何证明会红"里的第 3 种（已知坏输入回放）。

注入器：`bfdiag/divergence/qwen36_bug_injection.py`（可逆 context manager，
按属性形状发现模块、从不 `isinstance`，31 条 CPU 测试覆盖"注入生效"与"退出后逐项还原"）。
扫描脚本：`scripts/b1_forced_decode_agreement.py`。

**环境**：RTX PRO 6000 Blackwell Max-Q，`nvidia/Qwen3.6-27B-NVFP4` 真权重，
HF `Qwen3_5ForCausalLM` 参照（权重从我们自己的反量化张量拷过去），
3 工作负载 × 256 步 = **652 步**（`instruction-longer` 在 HF 自己的贪心下第 140 步吐 EOS，
见 §5.5），17 个配置，单次模型加载全程复用。

### 5.1 先决条件：步锁本身不扰动我们的执行路径（实测，非论证）

`--selfcheck`：先自由生成，再用**我们自己刚生成的 token** 做步锁强制解码，
逐步比对两次的 logits 张量。

```
selfcheck factual-short:      24 steps -- bit-exact
selfcheck math-short:         24 steps -- bit-exact
selfcheck instruction-longer: 24 steps -- bit-exact
```

**逐位相同。** 喂一个不是自己 argmax 的 token 只改变读哪一行 embedding，
不改变 dispatch、shape 或 kernel。这条是整个方法的合法性前提，现在是实测的，不是推理的。

**附带结论**：`gdn-state-decay:1e-3` 与 `gdn-state-decay:1e-4` 两个配置的汇总统计与控制组
**逐位相同**（见下表）——这同时证明了本运行时**跑到跑之间是确定性的**，
所以扫描表里任何两行的差异都可以完全归因于注入本身，没有运行噪声。

### 5.2 控制组：噪声底就在 bf16 的表示极限上

| 统计量 | 652 步 | 1164 步（跑到自然结束） |
|---|---:|---:|
| gap_error 中位数 | **0.125** | 0.125 |
| gap_error p90 | **0.250** | 0.250 |
| gap_error p99 | 3.375 | 2.125 |
| gap_error max | 10.56 | 10.56 |
| 平均 KL(top-K) | 1.58e-3 | 1.06e-3 |
| argmax 翻转数 / 率 | **5 / 0.77%** | 7 / 0.60% |
| 翻转的最大 tie_slack | **8 ULP** | 8 ULP |
| drift ratio | 1.50 | **1.00** |
| NLL 相对 HF | **-0.003%** | -0.003% |
| logits cosine | 0.999873 | 0.999873 |

**gap_error 中位数 0.125 = 恰好 2 个 bf16 ULP**（该量级 ULP=0.0625）。
5 次翻转的 tie_slack 分布：3 次 ≤2 ULP、1 次 ≤4 ULP、1 次 ≤8 ULP，中位数 **2 ULP**。
——这**用完全不同的方法独立复现了 §1 那张表的 1–2 ULP**。原判据失败的原因至此闭环：
噪声底（2–8 ULP）大于平局间距（1–2 ULP），所以翻转是必然事件，不是缺陷。

逐 64 步窗口的 gap 中位数：`0.125 0.125 0.188 0.188` / `0.125 0.125 0.062 0.125` /
`0.125 0.125 0.125` ——**平的，不随步数增长**。控制组没有累积漂移。

`ppl(ours) = 4.0676` vs `ppl(HF) = 4.0677`。**我们的前向数学与 HF 在 NLL 上相对差 2.5e-5。**

### 5.3 全扫描表（p50/p90 = gap_error 分位数，slack 单位 ULP）

| 注入 | p50 | p90 | p99 | max | 平均KL | 翻转 | 率 | slack | drift | NLL% | logits cos | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| **none（控制）** | 0.125 | 0.250 | 3.375 | 10.56 | 1.58e-3 | 5 | 0.77% | 8 | 1.50 | -0.003 | 0.999873 | ✅ 绿 |
| rope-positions-offset:1 | 0.125 | 0.250 | 3.06 | 11.72 | 1.48e-3 | 7 | 1.07% | 10 | 1.50 | +0.134 | 0.999890 | ✅ 绿 |
| rope-theta-rel:1e-5 | 0.125 | 0.250 | 4.44 | 10.28 | 1.70e-3 | 6 | 0.92% | 12 | 1.50 | +0.012 | 0.999873 | ✅ 绿 |
| rope-theta-rel:1e-4 | 0.125 | 0.250 | 2.81 | 10.41 | 1.51e-3 | 4 | 0.61% | 1 | 2.00 | +0.023 | 0.999902 | ✅ 绿 |
| rope-theta-rel:1e-3 | 0.125 | 0.250 | 3.19 | 13.09 | 1.99e-3 | 5 | 0.77% | 15.5 | 1.50 | +0.148 | 0.999840 | ✅ 绿 |
| rope-theta-rel:1e-2 | 0.125 | 0.250 | 4.06 | 11.84 | 2.66e-3 | 9 | 1.38% | 5 | 1.50 | -0.022 | 0.999895 | ✅ 绿 |
| gdn-state-decay:1e-4 | 0.125 | 0.250 | 3.375 | 10.56 | 1.58e-3 | 5 | 0.77% | 8 | 1.50 | -0.003 | 0.999873 | ✅ 绿 |
| gdn-state-decay:1e-3 | 0.125 | 0.250 | 3.375 | 10.56 | 1.58e-3 | 5 | 0.77% | 8 | 1.50 | -0.003 | 0.999873 | ✅ 绿 |
| gdn-stale-every:256 | 0.125 | 0.250 | 4.44 | 11.82 | 1.71e-3 | 8 | 1.23% | 6 | 1.50 | -0.003 | 0.999873 | ✅ 绿 |
| gdn-stale-every:128 | 0.125 | 0.375 | 4.44 | 11.82 | 1.71e-3 | 8 | 1.23% | 6 | 1.50 | -0.003 | 0.999873 | ✅ 绿 |
| **gdn-stale-every:64** | 0.250 | **0.812** | 9.03 | 14.53 | **1.80e-2** | 11 | 1.69% | **106** | 2.33 | -0.003 | 0.999873 | 🔴 **红（5 bar）** |
| gdn-stale-every:8 | 0.812 | 3.125 | 9.94 | 24.54 | 6.12e-2 | 41 | 6.29% | 185 | 2.40 | -0.003 | 0.999873 | 🔴 红（8） |
| gdn-stale-every:1 | 18.97 | 27.93 | 36.4 | 48.05 | 4.84 | 528 | 81.0% | 457 | 1.18 | -0.003 | 0.999873 | 🔴 红（8） |
| gdn-state-decay:1e-2 | 0.812 | 3.000 | 13.0 | 32.23 | 5.69e-2 | 34 | 5.21% | 258 | **5.17** | -0.003 | 0.999873 | 🔴 红（9） |
| gdn-state-decay:5e-2 | 3.000 | 8.438 | 13.6 | 24.19 | 2.15e-1 | 84 | 12.9% | 194 | 3.42 | -0.003 | 0.999873 | 🔴 红（9） |
| **drop-q-norm** | 1.188 | 4.000 | 9.97 | 16.00 | 6.28e-2 | 39 | 5.98% | 128 | 3.17 | **+3.324** | **0.995515** | 🔴 红（10） |
| **drop-k-norm** | 3.625 | 8.625 | 17.2 | 20.66 | 2.46e-1 | 94 | 14.4% | 169 | 2.04 | **+18.74** | **0.917519** | 🔴 红（10） |

倍数（相对控制组）：

| 注入 | p90 | 平均KL | 翻转率 | slack |
|---|---:|---:|---:|---:|
| 全部 ✅ 绿的配置（9 个） | 1.00–1.50 | 0.93–1.68 | 0.80–1.60 | 0.12–1.94 |
| gdn-stale-every:64（最弱可检出） | **3.25** | **11.4** | 2.20 | **13.3** |
| gdn-stale-every:8 | 12.5 | 38.6 | 8.20 | 23.1 |
| gdn-state-decay:1e-2 | 12.0 | 35.9 | 6.80 | 32.2 |
| drop-q-norm | 16.0 | 39.6 | 7.80 | 16.0 |
| drop-k-norm | 34.5 | 155 | 18.8 | 21.1 |

**绿组与红组之间没有重叠**：绿组 p90 最大 0.375，红组最小 0.812；
绿组平均 KL 最大 2.66e-3，红组最小 1.80e-2（6.8×间隔）；
绿组 slack 最大 15.5 ULP，红组最小 106 ULP（6.8×间隔）。

### 5.4 每条腿各自抓到什么——这是多腿判据的实证依据，不是设计洁癖

| 注入 | R1 步锁 | R1 drift | R3 logits cos | R4 NLL |
|---|:--:|:--:|:--:|:--:|
| drop-q-norm / drop-k-norm | 🔴 | 🔴/🟢 | 🔴 | 🔴 |
| gdn-stale-every:64 / 8 / 1 | 🔴 | 🟢 | **🟢 盲** | **🟢 盲** |
| gdn-state-decay:1e-2 / 5e-2 | 🔴 | 🔴 | **🟢 盲** | **🟢 盲** |

⚠️ **R3 和 R4 对全部 5 个 GDN 递归状态注入完全失明**，而且是**结构性**的：
两者都是**一次全长 prefill 前向**，注入破坏的是"写回状态供下一步读"，
在单次前向里那个状态根本不会被再读一次。实测：包括**把状态完全冻结**（`stale-every:1`）
在内的全部 GDN 注入，NLL 与控制组小数点后五位相同，logits cosine 完全相同。

**这就是 R1 必须存在的全部理由**——GDN 占 64 层里的 48 层，是 `implementation-plan.md` §7.1
点名的 **RK1 单点最大风险**。只留便宜的 R3/R4，等于对这个子系统零覆盖。
这条已经钉成测试（`test_only_the_step_locked_legs_catch_recurrent_state_bugs`），
以后谁想把 NLL 提成主判据，必须先删掉那条测试。

反过来，`drop-k-norm` 让 NLL 涨 **18.7%**、logits cosine 掉到 0.9176——
R4/R3 对结构性 bug 的动态范围远大于 R1，且成本低一个数量级。两类腿互补，缺一不可。

### 5.5 检测下限（不是断言的，是扫出来的）

- **GDN 状态漏更新**：256 步窗口下，1/64 可检出，1/128 与 1/256 检不出。
  下限落在两者之间。步数越长越灵敏（1164 步控制组的噪声更低）。
- **GDN 状态衰减**：1e-2 可检出，1e-3 与 1e-4 **不可检出——而且不是判据的问题**：
  递归状态是 bf16，1.0 附近 ULP = 2⁻⁸ = 0.0039（0.39%），
  乘 0.999 或 0.9999 直接舍回 1.0。两个配置与控制组**逐位相同**。
  **格式本身表示不出这个 bug**。
- **RoPE theta**：1e-2 到 1e-5 全部检不出。theta 只通过
  `inv_freq = theta^(-2k/rotary_dim)` 进入，在 ≤300 个位置上 1% 的 theta 变化
  对每个旋转维度的相位影响远小于 1 弧度。**这是 B1 上下文长度下的真实盲区，
  且会随上下文增长**——B2/B3 在 128K 下不能假设它仍然成立。
- **RoPE 位置整体偏移 +1**：**数学上是恒等变换**，不是漏检。RoPE 的注意力打分只依赖
  相对位置 `i-j`，所有位置同加一个常数后每个 `q_i·k_j` 一字不变。
  判据在这里保持绿色是**特异性（不误报）的正面证据**：一个在这里变红的判据是在把恒等变换报成缺陷。

### 5.6 顺带查出来的两个既有缺陷（R3 今天不能当门禁用）

跑控制组时 `scan_layers` 报 `first_divergent_layer=0`——**在一个正确的实现上**。逐一定位：

1. **`thresholds.py` 的 `HIDDEN_STATE.max_rel_abs_error = 0.02` 在真实 bf16 残差流上不可达。**
   控制组实测 `rel_max_abs_err = 0.3307`，而**同一层的 cosine 是 0.9999957**。
   这条 bar 在正确实现上必红、在注入实现上也是同样的红——**零区分力**。
2. **`qwen36_capture.py` 在最后一层比错了对象。** 逐层实测 cos：layer 0–62 全部 ≥0.99994，
   **layer 63 = 0.9748**。HF 的逐 token RMS 在 56–62 层是
   3.07/3.33/4.23/3.47/4.01/4.32/5.03，到 63 层**掉到 1.97**——这是最终 RMSNorm 的指纹。
   `Qwen3_5TextModel` 属于"把 final norm 应用到 `output_hidden_states` 最后一项"的那一类，
   所以我们拿**归一化前**的 layer-63 和 HF **归一化后**的比了。
   这正是该模块 docstring 自己写下的"上 GPU 第一件要验的事"，现在验了。

**两条都没有在本分支修**：修法都不能在没有 GPU 窗口的情况下复验，
而给一个诊断工具发未验证的改动，正是"门禁变成摆设"的起点。
因此 **B1-R 的 R3 只门禁 `logits` 那一项**（控制组 0.999873，drop-q-norm 0.995515，
drop-k-norm 0.917519 —— 区分力充分），逐层 hidden_state 判定**暂不作为门禁**，
只作为定位线索，且 layer 63 的结果必须忽略。修复列入 §8。

---

## 6. 阈值怎么定

**原则先写死（这三条在测量之前就提交了，`3c4c287`，防止事后凑数）：**

1. τ 从**控制组的噪声分布**反推，不从"当前能过"反推；
2. 必须**交叉验证外部锚**（上游同类常数）；
3. 每条 bar 必须报"控制组 → bar → 最弱可检出注入"的双侧余量，余量不足的 bar 明确降级。

### 6.1 最终常数（`bfdiag/divergence/logit_agreement.py::CALIBRATED_THRESHOLDS`）

| bar | 控制组 | 最弱可检出注入 | **阈值** | 控制侧余量 | 检出侧余量 |
|---|---:|---:|---:|---:|---:|
| `p90_gap_error` **（主）** | 0.250 | 0.812 | **0.5** | 2.0× | 1.6× |
| `mean_kl_topk` **（主）** | 1.58e-3 | 1.80e-2 | **5e-3** | 3.2× | 3.6× |
| `max_tie_slack_ulps` **（主）** | 8 | 106 | **32** | 4.0× | 3.3× |
| `p99_gap_error` | 3.375 | 9.03 | **6.0** | 1.8× | 1.5× |
| `median_gap_error` | 0.125 | 0.250 | **0.25** | 2.0× | — |
| `p90_logprob_error` | 0.250 | 0.79 | **0.5** | 2.0× | 1.6× |
| `max_drift_ratio` | 1.50 | 3.42 | **3.0** | 2.0× | 1.14× |
| `max_gap_error` | 10.56 | 14.53 | **20.0** | 1.9× | — |
| `max_disagreement_rate` | 0.0077 | 0.0169 | **0.03** | 3.9× | — |
| `max_nll_relative_excess` | -0.003% | +3.32% | **1%** | — | 3.3× |
| `min_logits_cosine` | 0.999873 | 0.9955 | **0.9995** | — | — |
| 步数 | — | — | 每负载 ≥128，合计 ≥600，≥3 负载 | | |

### 6.2 三条 bar 被显式降级（不是忘了调，是数据不支持）

- **`max_gap_error`：几乎没有区分力，只当"炸得很厉害"的兜底。**
  控制组的 max（10.56）**高于两个注入配置的 max**（10.28 / 10.41）——
  用 max 当主判据会把有 bug 的实现排在正确实现前面。
  这就是主 bar 从 max 改成 **p90** 的原因，也钉成了测试
  （`test_max_gap_error_alone_would_not_separate`）。
- **`max_disagreement_rate`：样本量下分布重叠，不可单独否决。**
  控制组 652 步只有 5 次翻转（泊松 σ≈2.2），最弱可检出注入是 11 次——约 2.7σ。
  绿组最高 1.38%、红组最低 1.69%，几乎贴住。它保留是因为**系统性偏置会把它抬起来**
  （§4 盲区 3 的主要探针），但绝不能是唯一红的那条。
- **`max_drift_ratio`：专用探针，普适力弱。** 它唯一无可替代的场景是
  `gdn-state-decay:1e-2`（5.17×，累积型状态误差），但它**抓不到最严重的那个注入**
  （`stale-every:1` drift 只有 1.18，因为那个 bug 从第一步就是满值、没有趋势）。

### 6.3 外部锚交叉验证（这是"不是拍脑袋"的第二条腿）

1. **SGLang 的 `decode_tolerance = 6e-2`**（`|Δ logprob|`，fp16，top-5）
   vs 我们控制组的 `p90_logprob_error = 0.250`（bf16，逐 token id，top-8 并集）。
   看起来我们大 4.2×——但 **bf16 的 ULP 比 fp16 粗 8 倍**（7 位 vs 10 位尾数）。
   折算到 fp16 等效精度：`0.250 / 8 = 0.031`，**落在 SGLang 的 5e-2 之内**。
   而且我们的统计口径更严（按 token id 逐一对比，SGLang 按排名比 top-5 的值）。
   → 我们的噪声底与上游同类实现**一致**，没有异常。
2. **SGLang 对 Qwen3-Next-80B（同为 GDN 混合架构）的 KL 门禁 = 2e-3**
   （`test_qwen3_next_models.py:38-44`）vs 我们控制组 `mean_kl = 1.58e-3`（652 步）
   / `1.06e-3`（1164 步）。**同一量级，且在其阈值之内。** 我们的 bar 取 5e-3。
3. **vLLM 的 `PPL_TOL = 0.01`**（相对、单边）→ 直接采用为 `max_nll_relative_excess`。
   我们控制组是 **-0.003%**，比阈值小三个数量级。

### 6.4 阈值不依赖拟合时用的步数

在拟合用的 256 步/负载之外，又把三个负载跑到自然结束（1164 步）复测控制组：
p99 3.375→**2.125**、平均 KL 1.58e-3→**1.06e-3**、翻转率 0.77%→**0.60%**、
drift 1.50→**1.00**。**全部变好。** 所以这套 bar 在门禁真正运行的长度上是偏保守的，
不是被短程拟合出来的。已钉成
`test_the_calibration_is_not_an_artefact_of_the_fitting_horizon`。

---

## 7. B2 / B3 能不能用同一套

### B2（固定槽位 + 连续批处理 + CUDA Graph）

原门禁写的是"与 B1 eager 贪心 **bit-exact**"。

【判断】**这条应该保留，但要加一条"不 bit-exact 时怎么办"的分支**，理由是本仓库已经实测过
一次同类情况：`notes/2026-08-02-eager-verify-cg-verify-divergence.md` 里 CG 与 eager 在
`kv_len=64` 时 bit-exact、在 ≥400 时不一致，根因是 split-KV 的 **chunk 数不同**
（CG 冻结成 1 块、eager 4~16 块），而两种块数在单层注意力上**都对**（对稠密 oracle
cos ≥ 0.999997）。也就是说：**B2 不 bit-exact 不一定是 bug，但一定需要一个机制解释。**

建议写法：

- **首选 bit-exact。** 同一台机器、同一份权重、同样的 eager 数学，差别只在调度，
  没有理由默认它会变。
- **一旦不 bit-exact**：(a) 必须给出机制解释（像 chunk 数那样具体到规划器输出的数字）；
  (b) 必须通过 B1-R 的 R1，且 **τ_B2 = τ_B1 / 10** ——因为 B2-vs-B1 是同一份数学在同一张卡上，
  它的噪声底应当远低于"我们 vs HF"的噪声底，用同一个 τ 等于什么都不查。
- **B2 需要一条 B1 没有的新轴：批不变性。** 同一 prompt 单独跑 vs 混在 batch=4 里跑。
  【实测旁证】vLLM 自己在 `test_common.py:197-200` 用
  `max_num_seqs=1 if current_platform.is_rocm() else 2`，注释是
  "Remove the effects of batch variance ... batch invariance is not yet supported"
  ——**上游明确承认批不变性不成立**。所以这条只能用 B1-R 的 gap error 度量，不能要求 bit-exact。

### B3（投机 / MTP）

**不能直接套，要另设。** 三个理由：

1. **正确性的定义变了**：投机解码的正确性是"输出分布与非投机路径相同"，
   不是"与 HF 相同"。参照系应当是**我们自己的非投机路径**。
2. **verify 走的是完全不同的代码路径**：一次 verify 把 16 个候选 token 整段跑一遍主模型，
   `seq_len=16` → GDN 走 chunk 分支、attention 走 verify-mode split-KV。
   这与 B1 的 `seq_len=1` 递归分支**不是同一个算法**——正是 §1 那条"不能拿两种 GDN 算法互测"的
   同一个陷阱，只是换了个位置。所以 B3 **不能**要求"投机结果与非投机 bit-exact"。
3. **GDN 状态回滚**：候选被拒时主模型的递归状态已被不可逆更新（`implementation-plan.md` §7.1 B3
   + `investigation-queue.md` D-3）。这是 B3 独有的正确性风险，B1-R 里没有对应的探针。

建议 B3 判据：
- **贪心**：MTP 开 vs 关，用 B1-R 的 R1 度量（步锁比较自己的两条路径），
  加一条"接受/拒绝后 GDN 状态与非投机路径的状态张量对比"的直接探针；
- **采样**：分布层面的 KS 检验（`e2e-and-quality-plan.md` §2.2 E2-b 已经这么写了，沿用）；
- **兜底**：GSM8K 之类任务准确率（上游两家真正的回归门禁都是这个，见 §2.3）。

---

## 8. 需要你拍板的

### 8.1 B1 门禁是否按 B1-R 改写、并据此判定 B1 通过？

**这是最重要的一条。** 按 B1-R 的校准阈值，**当前实现全绿**（§5.2/§5.3 控制组行）。
不是"放宽到能过"——同一套阈值让 7 个注入 bug 全红，而且最弱的那个（1/64 漏更新 GDN 状态）
也红在 5 条 bar 上。

| 选项 | 含义 | 我的看法 |
|---|---|---|
| **(a) 采纳 B1-R，B1 判为通过**（推荐） | 改写 `implementation-plan.md` §7.1 的 B1 门禁；`IMPLEMENTED_BACKENDS` 可以推进 | 证据链完整：判据可达、有区分力、阈值双侧有余量、外部锚一致。**但见 8.2 的两条前置** |
| (b) 采纳 B1-R，但 B1 暂不判通过 | 先补上 8.2 的缺口再判 | 更保守；代价是 Track B 卡住 |
| (c) 保留原字面门禁 | —— | 不建议：§1/§2 已证明它要的是不存在的东西，且比上游任何一家都严 |

**我推荐 (a)，但把 8.2 的两项列为 B1 的显式已知缺口写进验收记录**，而不是当作没有。

### 8.2 采纳 (a) 时必须一起记下的两个缺口（不是可选脚注）

1. **反量化未被覆盖。** HF 侧权重是从我们自己反量化的张量拷过去的，
   NVFP4 nibble 打包若错，两侧同样地错，B1-R 恒绿。
   见 `notes/2026-08-02-b1-nvfp4-nibble-packing-unverified.md`。
   **B1 通过 ≠ 权重加载正确。**
2. **长上下文未被覆盖。** 全部测量在 ≤300 个位置上。§5.5 已经实测到
   RoPE theta 类误差在这个长度下**天然不可检出**，而它的影响随位置增长。
   128K/256K 的正确性是 B2/B3 的问题，B1-R 不发言。

### 8.3 R3（逐层扫描）的两个既有缺陷谁来修、什么时候修

§5.6 定位了两条，都需要 GPU 复验，本分支**没修**：

- `thresholds.py::_BASE_THRESHOLDS[HIDDEN_STATE].max_rel_abs_error = 0.02` 不可达
  （正确实现实测 0.33）。这个常量是**与 Laguna 共用的**，改它会影响另一条链路，
  不该由本任务单方面动。
- `qwen36_capture.py` 把我们的 pre-final-norm layer-63 与 HF 的 post-final-norm 比。
  这条只影响 Qwen3.6，改动局部。

选项：(a) 随下一个 Qwen3.6 GPU 窗口顺手修 + 复跑本扫描；
(b) 先只修 capture 的 layer-63（局部、不碰 Laguna），thresholds 单独排期；
(c) 归入 `e2e-and-quality-plan.md` §3.2 的 C8 门禁债务清单。
**我倾向 (b) + (c)**：layer-63 是 Qwen3.6 独有、爆炸半径小；
`HIDDEN_STATE` 阈值是跨模型的标定问题，正是 C8 该处理的形状。

### 8.4 `max_disagreement_rate` 保留还是删掉

§6.2 已论证它在当前样本量下分布重叠。选项：
(a) 保留在 0.03（现状，作为系统性偏置的探针，但绝不单独否决）；
(b) 删掉，避免给人"它在把关"的错觉；
(c) 提高样本量（更多工作负载 / 更长步数）直到它有统计力。
**我倾向 (a)**，理由写在 `CALIBRATED_THRESHOLDS` 的注释里，
而且 §4 盲区 3 那个"系统性偏置恰好总在平局翻转"的场景，
目前只有它和 KL 两条正面探针。**(c) 是正解但要额外 GPU 预算，由你决定值不值。**

### 8.5 工作负载数量与步数

现状 3 负载 / 652 步（或跑满 1164 步）。
`instruction-longer` 在 HF 自己的贪心下第 140 步吐 EOS，
所以"每负载 512 步"这条**对当前工作负载集不可达**（与实现无关）。
B1-R 改成"每负载 ≥128 且合计 ≥600"。
若要更强的统计力（见 8.4），最省的做法是**加工作负载而不是加步数**——
更多不同 prompt 覆盖更多 entropy 形态，而同一条轨迹越往后熵越低（§6.4 实测）。

---

## 9. 复现

```bash
# phase 1：HF 参照轨迹（一次，~5 分钟：stage 851 个张量到磁盘、重建 HF、跑 3 条轨迹 + NLL）
~/.venvs/vllm/bin/python scripts/b1_reference_trajectory.py

# phase 2：步锁比较 + 注入扫描（一次模型加载扫全部配置，256 步/负载约 1.5 分钟/配置）
~/.venvs/vllm/bin/python -u scripts/b1_forced_decode_agreement.py --selfcheck --steps 256

# 门禁运行（跑满自然长度，约 4 分钟）
~/.venvs/vllm/bin/python -u scripts/b1_forced_decode_agreement.py --configs none
```

**纯 CPU、无需 GPU / 无需 checkpoint** —— 判据逻辑、注入器机制、以及
**"判据真的会红"这件事本身**（用 checked-in 的实测扫描结果回放）：

```bash
/tmp/ci-sim/bin/python -m pytest -q tests/test_bfdiag_logit_agreement.py \
                                   tests/test_bfdiag_qwen36_bug_injection.py \
                                   tests/test_b1_criterion_discriminative_power.py
```

最后一个文件把 §5.3 整张表存进
`tests/fixtures/b1_injection_sweep_2026-08-02.json`，
用**与线上运行同一个** `evaluate_summary()` 回放。
**任何人把某条 bar 放宽到不再分离，CI 会在 CPU 上红**，不需要 GPU、不需要 27B 权重、
不需要 40 分钟扫描。这是 `e2e-and-quality-plan.md` §4 三种"证明会红"方法里第 3 种
（已知坏输入回放）的落地形式。

## 10. 相关

- `notes/2026-08-02-b1-greedy-alignment-fails.md` —— 原门禁实跑失败与 ULP 实测
- `notes/2026-08-02-eager-verify-cg-verify-divergence.md` —— "两边都对也会翻转"的先例
- `docs/e2e-and-quality-plan.md` §4 —— "如何证明新门禁会红"的三种方法
- `bfdiag/divergence/logit_agreement.py` —— 指标与恒等式推导
- `bfdiag/divergence/qwen36_bug_injection.py` —— 注入器
