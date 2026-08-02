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
`seq_len>1` → chunk 分支）。它测不到 decode 路径，这正是 R1 存在的原因。

### 3.5 R4：强制解码 NLL / 困惑度（语义地板）

对一段固定文本（本仓库用一段 ~270 token 的说明文，见
`scripts/b1_reference_trajectory.py::NLL_TEXT`）做**一次**全长前向，算
`-log p(token[i] | token[:i])` 的平均。两侧都是一次全长前向 → **两侧都走
`chunk_gated_delta_rule`**，模式匹配，比较合法。

它的价值不是精度，是**动态范围**：任何真实降级（RoPE 角度错、漏 norm、状态错）
都会让 NLL 显著上升，而这个量**不受平局翻转影响**（它读的是固定 target token 的概率，
不读 argmax）。判据形式抄 vLLM：相对超出 HF 基线不得超过 `PPL_TOL` 量级，单边。

> ⚠️ **不要把 R4 当成唯一判据。** 它只有一个标量，定位能力为零，
> 而且对"只在长上下文才发作"的 bug 不敏感（文本只有 ~270 token）。

---

## 4. 盲区（明确写出来，不假装没有）

### 盲区 1：加性常数偏移——**故意的，不是缺陷**

`gap_error` 与 `logprob_error` 都对"所有 logits 整体加常数"完全失明。
这是设计意图：该变换对 argmax 与 softmax 都是恒等的，把它算成误差是制造假阳性。
（乘性缩放**不**在此列，会被测到。）

### 盲区 2：效应小于噪声底 τ 的 bug——**不可约**

任何在 top-K 相对 logit 上效应始终 ≤ τ 的 bug 是不可见的。这与"逐位一致不可达"是同一件事的
另一面：参照系本身有噪声，噪声以下的东西测不到。
**缓解**：R3 的浅层阈值比 logits 层严格若干个数量级；R4 在长程上另有一条正交通道。
**但请诚实理解这条**：如果一个 bug 的全部效应都藏在 τ 以下，B1-R 抓不到，
本文档不声称能抓到。判据能给的最强承诺是：**它能抓到任何效应超过"两个正确 bf16 实现之间
噪声底"的 bug**，而 §5 的注入实验量化了真实 bug 离这个底有多远。

### 盲区 3：协调者点名的那个——"整体偏移但每次都恰好在平局处翻转"

**是的，一个只看"分歧处是否平局"的判据对这个是盲的。** 具体构造：一个 bug 给某一类 token
的 logits 统一加 0.1，那么每次翻转的 `tie_slack` 都会落在 0.2 以内，
只要 τ ≥ 0.2 就每一次都被判为"良性平局"。

B1-R 用**三条**独立的 bar 堵这个洞，但没有一条是完全的：

1. **`max_disagreement_rate`**：纯对称噪声下的翻转率由模型自身的 top1/top2 间距分布决定，
   是一个可测的常数；一个有偏的误差会把它系统性抬高。这是最直接的探针。
   【判断】它的有效性取决于控制组翻转率与注入组翻转率的分离度——§5 的实测必须给出这个分离度，
   分离度不够就说明这条 bar 在这个模型上不好用。
2. **`mean_kl_topk`**：0.1 的系统偏移会改变整个分布形状，不只是排序。
3. **`max_gap_error` / `p99_gap_error`**：0.1 的偏移本身就是 0.1 的 gap error，
   如果 τ 定在噪声底附近（而不是"够大以容纳所有观测到的翻转"），它就直接红了。
   **这就是为什么 τ 必须从控制组的噪声分布反推，而不是从"当前能过"反推。**

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

**[本节在 GPU 运行后填入实测数字]**

---

## 6. 阈值怎么定

**[本节在 GPU 运行后填入实测数字与最终常数]**

原则（先写死，避免事后凑数）：

1. **τ 从控制组的噪声分布反推，不从"当前能过"反推。**
   `max_gap_error` 取控制组实测最大值 × 2（安全系数），`p99` 取控制组 p99 × 2。
2. **交叉验证外部锚**：`max_logprob_error` 必须与 SGLang 的 5e-2/6e-2 处在同一量级；
   如果我们的控制组实测远超 6e-2，说明**要么我们的实现真有问题，要么这个模型的
   数值敏感度确实高于 SGLang 模型库的中位数**——两种解释都必须写下来，不能默默调高阈值。
3. **注入实验反推分离度**：每条 bar 都要报"控制组值 / 最小可检测注入的值"的比值。
   比值 < 3 的 bar 视为**不可靠**，要么去掉，要么明确标注为"仅供参考、不单独否决"。
4. **`max_disagreement_rate`** 无法从第一性原理推，只能实测控制组翻转率 ρ₀ 后取
   `max(3ρ₀, ρ₀ + 3σ)` 量级；如果控制组 ρ₀ 已经很高，这条 bar 天然弱。

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

## 8. 需要拍板的

**[本节在 GPU 运行后补齐，取决于实测结果]**

---

## 9. 复现

```bash
# phase 1：HF 参照轨迹（一次，昂贵：要 stage 权重到磁盘再重建 HF）
~/.venvs/vllm/bin/python scripts/b1_reference_trajectory.py

# phase 2：步锁比较 + 注入扫描（一次模型加载，扫全部配置）
~/.venvs/vllm/bin/python scripts/b1_forced_decode_agreement.py --selfcheck

# 纯 CPU：判据逻辑与注入器本身
/tmp/ci-sim/bin/python -m pytest -q tests/test_bfdiag_logit_agreement.py \
                                   tests/test_bfdiag_qwen36_bug_injection.py
```

## 10. 相关

- `notes/2026-08-02-b1-greedy-alignment-fails.md` —— 原门禁实跑失败与 ULP 实测
- `notes/2026-08-02-eager-verify-cg-verify-divergence.md` —— "两边都对也会翻转"的先例
- `docs/e2e-and-quality-plan.md` §4 —— "如何证明新门禁会红"的三种方法
- `bfdiag/divergence/logit_agreement.py` —— 指标与恒等式推导
- `bfdiag/divergence/qwen36_bug_injection.py` —— 注入器
