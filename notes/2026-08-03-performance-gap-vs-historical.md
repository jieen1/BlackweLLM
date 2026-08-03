# 与历史性能的差距：**接受长度确定退化 2.75×，原始 decode 尚未同口径对比**

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（**历史与今天是同一个模型**）
· 状态：🟡 一条已坐实、一条待测

## 已坐实：MTP 接受长度退化 2.75×

| | 每步接受 token |
|---|---:|
| 历史（`notes/2026-07-20-comprehensive-optimization-plan.md`，128K/c=4） | **3.3** |
| 今天（`b3_mtp_e2e_acceptance_throughput.py`，K=4） | **1.20**（prose）/ 1.67（code） |

实测对实测，不是估算。**这是真实退化，不是口径差异。**

**已定位到疑似根因**：`(token, hidden)` 配对错误——草稿头第一步必须把
**hidden(位置 P)** 配 **token(位置 P+1)**，而此前每一个 B3 脚本配的都是
hidden(P)+token(P)（"这个 token 自己前向产出的 hidden"，而非"预测出它的那个 hidden"）。
对照 vLLM 自己的原生 Qwen3.6 MTP kernel（`_prepare_prefill_inputs_kernel`，
注释即 "Shift target_input_ids by one"）与历史代码的 `shifted_input_ids` 均确认。
修复已合入（`work/mtp-serving-20260803`），**接受率的实际改善待 GPU 实测**。

⚠️ 顺带：`work/mtp-resync-20260802` 的提交信息把这个差异称作
"this file's own (unshifted) convention"——**当成约定，没认出是 bug**。

## 也坐实了：原始 decode 步慢约 1.6×（按有效带宽）

**同口径换算到"每步做多少工作、花多少时间"：**

| | 历史（128K, c=4） | 今天（短上下文, c=4） |
|---|---:|---:|
| 每步/轮耗时 | 262 接受tok/s ÷ (3.3×4) = **50.4 ms/round** | 68.59 ÷ 4 = **58.3 ms/step** |
| 每步读取 | ~20 GiB 权重 **+ 8.4 GB KV** ≈ 28.4 GiB | ~20 GiB 权重（短上下文 KV 可忽略） |
| **有效带宽** | **≈ 564 GB/s** | **≈ 343 GB/s** |

**今天只有历史的约 61%，而且历史那边还额外扛着 128K 的 attention。**

这与短上下文"更容易"的直觉一致：128K decode 每步比短上下文**多读** 8.4 GB，
本该更慢；实测反而更快，所以差距是真的，不是口径造成的。

⚠️ 折算里的唯一假设：一次 verify round 的成本 ≈ 一次 decode step。
在 128K 下这个假设是合理的——attention 成本由 KV 读取主导，
K=1 与 K=4 读的 KV 量相同。**但仍需 128K 实测坐实。**

🔴 **这推翻了同日 [`2026-08-03-stage4-kernel-levers-exhausted.md`](2026-08-03-stage4-kernel-levers-exhausted.md) 的结论。**
那份的推理是"CG 下已 kernel-bound（GPU 89% 忙）⇒ 只能让 kernel 更快，而四条路都堵死 ⇒ 没杠杆了"。
**"已 kernel-bound"不等于"kernel 已最优"**——同模型、同卡，历史实现跑到过 564 GB/s。
所以缺口不在"还有没有杠杆"，而在**当前 kernel 组合本身比历史那套慢**。
那份 note 的四条否定（W4A4/W8A8/`bf16_gemv`/无 BF16 稠密 GEMM）各自仍然成立，
**但"因此没得优化了"这个总结不成立**。

## 从历史文档带回的两条 decode 步构成

`notes/2026-07-20-comprehensive-optimization-plan.md` §3 的分解，对今天仍有指示意义：

1. **attention kernel 在 131K 下比 FlashInfer 慢 5.8×**，当时被列为"最大的单项可优化点"。
   ⚠️ 那是 vLLM 的 `sm120_gqa` kernel，今天走 sparkinfer，**不能直接搬结论**，
   但"长上下文下 attention 是主导项"这个**结构**要在 128K 重测时验证。
2. **draft 的 continuation 步落在慢的 CUDA-core kernel 上**——v2 tensor-core kernel
   排除了 `qo_len==1` 且 MMA 不支持 fp8，于是每步 2–3 次、每次扫完整 128K KV。
   **今天的 MTP draft 步是否也落在类似慢路径上，接受率修好之后必须查**，
   否则接受率上去了、draft 成本吃掉收益。

## 两处自我纠正（都在同一天、同一件事上）

1. 初读时把历史 262（**带 MTP、按接受 token 计**）直接对今天 68.59（**不带 MTP**），
   得出"慢 3.9×"。错在拿 MTP 数字比非 MTP 数字。
2. 纠正过头：折算成"不带 MTP 约 79 vs 68.59，差 15%"，并说"杠杆已用尽的结论不必推翻"。
   **也是错的**——那个比法把历史额外扛着的 128K attention 当成了免费的。
   按每步实际读取量换算成有效带宽才是可比的，结论是 **564 vs 343 GB/s，约 1.6×**。
