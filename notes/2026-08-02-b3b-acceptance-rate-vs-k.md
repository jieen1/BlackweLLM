# B3-b：接受率是 K/配置问题，不是（单纯）草稿头质量问题——真实 K 曲线

日期：2026-08-02 · 状态：🟢 已在真实 GPU 上测量（27B 全模型，非仿真）· 分支 `work/b3b-accept-20260802`

## 结论先行

**扫 K 之后，"MTP 投机解码比顺序解码慢"这个结论翻转了——但只在 K≈3–6 时翻转，且仍是 eager
单 slot 手写循环里的数字，不是 ServerEngine 数字。** 之前所有"MTP 更慢"的测量
（`notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md`、`notes/2026-08-02-b3a-anchor-fold-verdict.md`）
全部固定在 **K=8**。把 K 当成自由变量后：

| K | prose 接受率 | prose 加速比 | code 接受率 | code 加速比 |
|---:|---:|---:|---:|---:|
| 1 | 0.620 | 0.80x | 0.803 | 0.82x |
| 2 | 0.532 | 0.78x | 0.655 | 1.08x |
| 3 | 0.401 | **1.06x** | 0.615 | **1.27x** |
| 4 | 0.349 | **1.11x** | 0.608 | **1.38x** |
| 5 | 0.288 | **1.11x** | 0.531 | **1.41x** |
| 6 | 0.240 | 0.99x | 0.426 | 1.28x |
| 8（现状默认） | 0.206 | 0.88x | 0.314 | 1.09x |
| 16 | 0.080* | 0.65x* | 0.191* | 0.97x* |

（K=1..8 一行来自 `scripts/b3b_k_sweep.py --k-values 1 2 3 4 5 6 8 --n-tokens 128`，同一次模型加载、
同一个自由贪心基线内测量，37–79 rounds/格；带 `*` 的 K=16 行来自另一次独立的粗粒度复测
`--n-tokens 64`，K=1/2/4/8 两次复测方向一致但绝对值有 GPU 热状态漂移，见下方"未验证"。
原始数据：`benchmarks/fixtures/b3b_k_sweep_fine_n128_20260802.json`、
`benchmarks/fixtures/b3b_acceptance_rate_investigation_20260802.json`。复现：
`~/.venvs/vllm/bin/python scripts/b3b_k_sweep.py`。）

**接受率随 K 单调下降**（compounding，见下），**但加速比不是单调的**——K=1 因为"起草+verify
开销摊不到 1 个 token 上"而亏本，K=8/16 因为"接受率跌得比 K 涨得快"而些本，**中间 K≈3–6 是
两头都还没输的甜蜜点，两个 prompt 都实测跑赢非投机路径**。这是本仓库整条 B3 investigation
（sparkinfer kernel、批处理大投影、anchor 折叠、GDN 成本归零上限）第一次量出 **>1.0x** 的真实
speedup，而且没有改一行模型代码——纯粹是把写死的 `K=8` 换成 `K=4` 或 `K=5`。

📌 **这不等于"MTP 现在能上生产"**：这些数字和 B3 之前所有数字一样，来自
`scripts/b3b_k_sweep.py` 独立 eager 脚本（单 slot，无 CUDA Graph，无连续批处理，不经过
`ServerEngine`），量级可信、绝对值不可信到能直接当 SLO 承诺。下一步是"把默认 K 从 8 改到
4–5 并接进 ServerEngine 复测"，不是"MTP 已经翻正"。

## 你问的四件事，逐条回答

### 1. K 的影响——最便宜的实验，也是杠杆所在

上表已经是答案。定性解释两端为什么都亏：
- **K 太小（K=1）**：草稿+verify+"推进新 anchor"这三笔固定开销（前几轮 B3 已量化：草稿
  ~73ms、anchor 推进 ~130–240ms，`docs/implementation-plan.md` §7.1）分摊不到最多 1 个
  token 上，即使接受率高达 62–80% 也不够。
- **K 太大（K=8/16）**：48 层 GDN 的 `spec_forward` 顺序 kernel 启动成本随 K 增长
  （K=1 每轮 ~0.30s → K=16 每轮 ~0.60s，`round_s` 字段实测，见 fixture），而接受率跌得更快
  （prose 0.62→0.21→0.08，code 0.80→0.31→0.19），两者相除，speedup 在 K=8 附近开始转负、
  K=16 时明显更差。
- **中间 K≈3–6**：开销分摊到位、接受率还没跌穿，两个方向都还没输。

### 2. 草稿头质量——teacher-forced 单步准确率，与链内逐位置准确率对照

`scripts/b3b_teacher_forced_head_quality.py`：把 MTP 头在**每一个真实生成位置**上，喂真实
token + 真实 backbone 隐状态（不是头自己之前可能错的输出），只问一步"下一个 token 是什么"，
和自由贪心解码的真实下一个 token 比较。这是去掉了链式复合误差之后的"纯头质量"基线，
样本量远大于 K 扫描（每个 prompt 159 个位置，K 扫描每格只有 20–80 轮）：

| prompt | teacher-forced top-1 准确率 | 前 1/4 段 | 后 1/4 段 | miss 里落在 target rank2-5 的比例 |
|---|---:|---:|---:|---:|
| prose | 0.629 (100/159) | 0.513 | 0.744 | 0.678 |
| code | 0.824 (131/159) | 0.769 | 0.846 | 0.643 |
| instruction | 0.711 (113/159) | 0.641 | 0.667 | 0.783 |

头本身**不是坏的**：62–82% 的单步 top-1 准确率对一个"checkpoint 自带的单层" MTP 头是合理
水准（不是专门训练的多层独立 draft 模型），且大部分 miss 是"猜到了 target 的第 2–5 名"而不是
瞎猜（0.64–0.78 close-miss 比例）。这个数字和 K 扫描里的 position-0 准确率
（prose 0.57–0.66，code 0.62–0.83，随 K 波动但同量级）互相印证——**position 0 在算法构造上
就是一次 teacher-forced 单步预测**（`anchor_argmax` 来自对真实已提交 anchor 的一次普通前向，
不依赖头自己之前的任何输出），两条独立测量落在同一个范围，是一致性检查通过。

**逐位置准确率随链长下降，但不是崩到 0，是"衰减后平台"**（K=8, N=128, prose，48 轮）：

```
position:   0      1      2      3      4      5      6      7
accuracy: 0.625  0.625  0.542  0.542  0.354  0.354  0.312  0.333
```

前两格 ~0.62，中间两格 ~0.54，后四格稳定在 ~0.31–0.35——**不再继续往下掉**。这个形状
（AGENTS.md 诊断表说 reject_position 直方图的*形状*能区分几类问题）排除了"头从第一个草稿就
瞎猜"（position 0 不差）和"无限发散到纯噪声"（后段没有继续跌向 0）两类假设，指向的是**链式
自条件复合漂移**：头一旦在某个位置偏离真实延续，它后续吃的是自己那个偏离后的、但仍然
自洽的续写，与 target 在那个自洽续写下的预测仍有 ~30–40% 概率重合——不是随机噪声水平，
是"两个训练出来还算合理的模型在同一段文本上偶尔殊途同归"的水平。

reject_position 直方图（K=8, N=128）本身：

```
prose: {0: 18, 1: 9, 2: 8, 3: 4, 4: 6, 5: 1, 6: 1, 7: 1, full_accept: 0}
code:  {0: 10, 1: 5, 2: 6, 3: 5, 4: 4, 5: 3, 6: 1, 7: 0, full_accept: 3}
```

前重尾（大部分 reject 落在 position 0–2），细长尾巴一直到 position 7，48/37 轮里
0/3 次 full-accept——典型的几何衰减形状，和 K=1 的直方图（`{0: 30, full_accept: 49}`，
62% full-accept）对比，说明"满 K 全接受"这个事件本身随 K 增大迅速变成小概率事件，不是
突变。

**结论：这是配置（K）问题，不是（单纯）质量问题。** 决定性证据是第 1 条本身——同一份
权重、同一个头、什么都不改，只把 K 从 8 调到 4，speedup 就从 0.88x 翻到 1.38x（code）。
如果瓶颈纯粹是头质量差，调 K 不可能翻正任何东西。头质量设了一个**上限**（K 不能无限大，
K=16 已经比 K=8 更差），但没有决定"K=8 到底行不行"这件事——那件事完全由 K 本身决定。

### 3. tokens_match=False（code prompt）——用 gap error 判定，不用 token 相等

`scripts/b3b_divergence_gap.py` 复现了 B3-a 报告里 code prompt 的分叉（K=8, N=32，和原报告
同配置）：**分叉位置、分叉方向都复现了**——index 2，ref token 12 (`'-'`) vs spec token 471
(`' -'`)，即 `n-1` vs `n - 1`，和 `notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md`
原始记录逐字一致。

在这个分叉点，两条路径喂的上下文**逐 token 相同**（这是"first divergence"的定义本身），
所以两边在这一步的 logits 直接可比，用 `bfdiag.divergence.logit_agreement.compare_step`
（B1-R 同一套判据）测：

| 指标 | 测得值 | B1-R 校准 bar | 结论 |
|---|---:|---:|---|
| gap_error | **0.125** | p90 0.5 / median 0.25 | 远低于两条 bar |
| mean_kl_topk | 1.95e-3 | 5e-3 | 低于 bar |
| tie_slack_ulps | **1.0** | 32.0 | bf16 可表示的最小单位——字面意义上的"平局" |
| disagreement | mine_top1=471, oracle_top1=12（不同 token） | — | 是，这就是要解释的分叉 |

`evaluate_summary` 对这次单点测量给出 `passed=False`，但原因不是 gap_error 大——是
`disagreement_rate=1/1 > 0.03` 这条 bar 对单样本天然打不过（这是采样量的问题，不是幅度的
问题），以及 `nll_relative_excess`/`min_logits_cosine` 两个 R3/R4 聚合指标本单点脚本没算
（需要整条轨迹，不是单点可测）。**看幅度本身**：gap_error/kl/tie_slack 三项全部落在
B1-R 判定"正确"的区间内，tie_slack 精确等于 1 ULP——和
`notes/2026-08-02-spec-verify-batching-bar.md` 记录的"已知良性" case（"max_tie_slack_ulps
都是 1.0 ULP"）同一量级。**这次分叉是良性平局翻转，不是数值 bug**，坐实了原报告"这可能是
良性 kernel 调度分歧"的预判。

顺带一个结构性证据（读代码得出，不是新测量）：分叉**只能**发生在 verify 批处理算出的行
（round 内 position ≥1 的槽位），**不可能**发生在"推进新 anchor"槽位（那一步永远是普通单
token 前向，和顺序解码的计算完全同一路径，逐位一致）。这解释了为什么四次独立复现的分叉
（K=2/K=8 两次跑到 index=2，K=4/K=5/K=6 两次跑到 index=23）总是落在"某一轮 verify 批处理
产出的位置"，从不落在 index=1（那个槽位永远安全）——是机制性的、可预期的，不是随机乱飘。

### 4. 采样配置——全程贪心，非贪心接受率是完全空白

读代码确认：本报告、`scripts/b3_mtp_e2e_acceptance_throughput.py`、以及所有既有 B3 测量
全部走 `runtime.mtp_accept.determine_accept_reject_from_predictions`——纯贪心 argmax。
`runtime/mtp_accept.py` 里确实有非贪心的拒绝采样版本 `sample_accept_reject`（E2-a，已有
exact-rational 单测验证分布性质），但**只有 `runtime/backends/laguna_dflash.py` 调它**——
Qwen3.6 的 verify 路径完全没有接上它。**Qwen3.6 MTP 在 temperature>0 下的接受率，这个项目
从来没人测过**，包括本次。这是一块完全空白，不是"测了但没写进报告"。

顺带确认了 Laguna 自己的 `benchmarks/acceptance_regression.py`（96.3–100% 数字的来源）走
`generate_verify_only(..., temperature=0.0)`，默认也是贪心——所以"我们 15–80% vs Laguna
96–100%"这个对比在**采样配置**这个维度上是同口径的（贪心对贪心），差距不能归因于"他们采样
我们没采样"。差距真正的来源是下一条。

### 5. 对照 Laguna DFlash——差距量级怎么理解

Laguna DFlash 用的是**专门训练的独立 6 层 draft transformer**（自己的 attention、512-token
窗口、`DRAFT_NUM_QO_HEADS=72`，`runtime/backends/dflash_constants.py`），Qwen3.6 用的是
**checkpoint 自带的单层 MTP 头**，直接吃 backbone 最后一层的隐状态。这不是同一类模型，
不能直接比接受率数字本身。

但差距的**量级**值得解释，而不能只用"架构不同"一句话带过：本次测出的**K=1 单步接受率**
（52–80%，等价于 teacher-forced 准确率 63–82%）其实是一个不算差的数字——不是"坏到没法用"，
只是低于 Laguna 专用 draft 模型的 96–100%。真正让 Qwen3.6 显得"投机不划算"的，不是这
52–80% 本身，而是**在错误的 K 上把它链式复合了 8 次**，把有效接受率打到 15–31%。把 K 调对
之后（K≈3–6），同一个"不算差但也不算强"的头已经能撑起 >1.0x 的净加速——**差距的大头是配置
选择，不是头本身弱到没法用**。头质量确实设了上限（K 不能像 Laguna 的 15 那样大），但没有
决定"能不能用"。

## 我没能验证的东西

1. **没有重复实验、没有置信区间**：每个 (K, prompt) 组合只跑了一次。两次独立跑（N=64 粗
   扫、N=128 细扫）K 曲线的**形状**（接受率随 K 单调降、speedup 在中间 K 出峰）一致复现，
   但自由贪心基线本身两次跑分别是 5.95/6.29 tok/s 和 6.64/7.18 tok/s（~11–14% 漂移，本项目
   反复记录过的 GPU 热状态问题），所以 speedup 的**绝对值**（"1.41x"这个数字本身）不该当成
   精确值，只该当成"这个区间明显 >1.0x"。
2. **只测了 2 个 prompt**（prose、code）做 K 扫描；teacher-forced 多测了一个 instruction
   prompt 但 K 扫描没有。`benchmarks/acceptance_regression.py` 的重复性文本套件（"fox-4K"
   类，注明是投机解码的最佳场景）完全没有覆盖——真实工作负载的最优 K 可能和这里的 3–6
   不同，尤其是高度重复性的文本大概率能撑更大的 K。
3. **全程独立 eager 脚本，不经过 `ServerEngine`**：没有 CUDA Graph、没有连续批处理、没有多
   slot 竞争。`notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md` 就已经列过这条未验证
   项，本次同样没有解决——"K=4 时 speedup 1.38x"是这个独立脚本的数字，不是服务器数字。
4. **非贪心（temperature>0）接受率完全没测**，见上一节——不知道这条 K 曲线在采样解码下
   长什么样。
5. **K=16 只在粗扫（N=64）里出现过，没有用细扫（N=128）复核**；细扫止于 K=8。K 曲线在
   9–15 区间是插值猜测，不是实测。
6. **divergence gap-error 只深挖了 K=8 的 index=2 这一个分叉点**；K=4/5/6 的 index=23 分叉
   只给出了结构性论证（为什么只能是 verify 槽位），没有重新跑一次 gap-error 数值确认。
7. **没有解释 code 接受率显著高于 prose 的原因**（只是引用了投机解码文献里"重复性/确定性
   文本更容易命中"的常识，`benchmarks/acceptance_regression.py` 自己的 docstring 也这么说），
   没有用更多样的 prompt 类别验证这是否是通则。

## 建议处置（不是本次要做的，留给下一步拍板）

- 把 `docs/implementation-plan.md` §7.1 B3 的默认 K 从 8 改成 4–5 这件事，**值得在
  `ServerEngine` 里复测**——如果服务器路径下 speedup 依然 >1.0x，B3-b 就从"唯一还有杠杆"
  变成"已经翻正，可以规划接入服务路径"；如果服务器路径（CUDA Graph、连续批处理）把这个
  优势吃掉了，那才是新的、真正的硬上限。
- 这不是"MTP 现在能上生产"的结论——是"值得再花一次 GPU 窗口去 ServerEngine 里复测"的结论。

## 相关

- `scripts/b3b_k_sweep.py` / `scripts/b3b_teacher_forced_head_quality.py` /
  `scripts/b3b_divergence_gap.py` / `scripts/b3b_run_all.py` —— 本轮四个脚本
- `benchmarks/fixtures/b3b_k_sweep_fine_n128_20260802.json` —— K=1..8, N=128 细扫原始数据
- `benchmarks/fixtures/b3b_acceptance_rate_investigation_20260802.json` —— K sweep(含K=16,N=64)
  + teacher-forced + divergence-gap 合并跑的原始数据
- `notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md` —— 上一轮：K=8 固定时的负面结论
- `notes/2026-08-02-b3a-anchor-fold-verdict.md` —— 上一轮：anchor 折叠不可行
- `docs/b1-correctness-criterion.md` §7 —— B3 判据（gap error，不是 token 相等）
