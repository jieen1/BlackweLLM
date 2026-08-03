# Qwen3.6 历史性能记录：哪些数字是真的、属于谁、今天还算不算数

日期：2026-08-02 · 状态：🟢 已核实(逐提交读正文与改动文件) · 零 GPU

## 为什么有这份文档

Track B 要把 Qwen3.6 在自研框架里重建出来，性能目标不能凭印象定。git 里有 677 个提交、
大量带 tok/s 的标题，但**标题极具误导性**：同一天里"超越 vLLM 3.2 倍"和"追平原生至 5% 差距"
都出现过，而且不少写着 vLLM 对比的提交改的是 `laguna_*.py`。

本仓库已经吃过一次亏：`c53bd7c` 更正了"Laguna MMLU-Pro 84.5%"——那个数字其实属于 Qwen3.6。
**这次是反方向的同一个坑**：把 Laguna 的性能数字当成 Qwen3.6 的。

判定方法：逐个提交读**正文 + 改动文件清单**，不看标题。

## 一、真正属于 Qwen3.6 的性能记录

主文档：[`2026-07-20-comprehensive-optimization-plan.md`](2026-07-20-comprehensive-optimization-plan.md)
——标题即 "Performance Gap Analysis & Optimization Plan — **Qwen3.6-27B** SM120 Runtime"，
模型明确写为 `unsloth/Qwen3.6-27B-NVFP4`。

### 核心结论（用户记忆的"明显超过 vLLM"，出处在这里，但有关键限定）

> 报告的 `tok/s` 是 `total_accepted / (TTFT + decode)`。128K/c=4 下我们 TTFT **25.7 s**
> vs 原生 **4.4 s**，光这一项就主导了 0.718× 这个比值。扣掉它（并发 4 墙钟）：
> **我们稳态 decode ≈ 262 agg tok/s ≈ 15 ms/step，原生 ≈ 197 agg tok/s ≈ 20 ms/step。**
> 我们的稳态 decode 步已经追平甚至快于原生；标题差距来自 TTFT（prefill 调度）与接受长度，
> **不是 decode kernel 循环**。

**准确表述**：
- ✅ **稳态 decode 快于 vLLM 约 1.33×**（15 ms/step vs 20 ms/step）——这是真的，是要守住的成果
- ❌ **端到端报告值当时落后**（0.718×）——因为 25.7 秒的 TTFT 被摊到区区 256 个输出 token 上

**"探索过的方向要做到"指的就是这个**：decode 循环已经证明能赢，输在 prefill 调度。

### 差距分解（128K/c=4，对原生）

| 成分 | 占差距 | 证据 |
|---|---:|---|
| **TTFT / prefill 调度**（缺跨步交错的 chunked prefill） | **60–70%** | 25.7 s vs 4.4 s |
| ~~**接受长度**（3.3 vs 4.85 tokens/step）~~ | ~~20–25%~~ | 🔴 **这一行作废**（2026-08-03 更正）——「原生 4.85」是 `benchmarks/native_warm_compare.py` 的 Prometheus 抓取**双重计数 bug**：`"num_accepted_tokens" in metric_name` 同时匹配了主计数器和 per-position 计数器，把原生**膨胀约 2×**（修复：第 101 行加 `"per_pos" not in metric_name`）。**修正后 128K/c=4：原生 64.2% / 2.926，我们 66.7% / 3.0 —— 我们略优于原生。** 见 `2026-08-03-historical-implementation-survey.md` §4.6 |
| 稳态 decode 步时 | 10–15% | 15 vs 20 ms/step，基本持平 |

### 稳态 decode 步内部构成

| 成分 | 占比 | 备注 |
|---|---:|---|
| **GEMM**（64 层 target + draft，NVFP4/FP8） | 40–50% | nsys：batch=1 时 GEMM 占 GPU 时间 76% |
| **Attention（当年的 SM120 kernel）** | 20–30% | ⚠️ 当年"比 FlashInfer 慢 5.8×"，见下方"已失效"一节 |
| **Draft 模型开销**（每步 3 次额外前向） | 12–18% | draft 走的是慢的 `qo_len=1` CUDA-core kernel |
| **GDN**（48 层） | 8–10% | in_proj + conv1d_update + delta-rule + gating + out_proj |
| Logits/采样（vocab 248320） | 6–9% | |
| Python/eager 开销 | 0%（有 CG）→ 15–25%（eager） | 当年的 warm bench 是 `enable_cudagraph=False` |

## 二、⚠️ 这份计划里哪些已经不适用

它测的是**已退役的架构**：`runtime/direct_model_runner.py`（5915 行单文件）+ vLLM fork 的
`sm120_gqa.py`（1060 行）+ `/home/bot/project/sm120-flash-attention/`。今天这些都不在生产路径上。

- ❌ **"SM120 attention kernel 比 FlashInfer 慢 5.8×，是最大单项可优化点"** —— 那个 kernel
  已被 **sparkinfer paged attention** 取代。B0-3 实测 sparkinfer 在 `head_dim=256/gqa=6`
  下正确且吞吐健康（`notes/2026-08-02-trackB-b0-gpu-facts.md`）。**这条不能直接搬**，
  必须对 sparkinfer 重测。
- ❌ 一切引用 `direct_model_runner.py` 行号的条目
- ✅ **仍然适用的是诊断结构**：TTFT 主导、接受长度次之、decode 步基本持平——这个**排序**
  是关于工作负载与算法的，不是关于某个 kernel 实现的

## 三、几条容易被误引的 Laguna 数字（不是 Qwen3.6）

逐个核实改动文件后确认全部属于 Laguna：

| 提交 | 标题声称 | 实际 |
|---|---|---|
| `f5d96b6` 07-23 | 75.4 tok/s **(3.6x vLLM)** | 改 `laguna.py`；对照组是**未开 CUDA Graph 的 vLLM** |
| `442fa91` 07-23 | **超越 vLLM 原生 3.2 倍** | 正文写明 `Laguna-S-2.1-NVFP4`；vLLM 原生 21.0 tok/s 是 eager |
| `e6e94d0` 07-23 | 追平原生至 **5% 差距** | **同一作者两小时后的公平对比**：均含 CG，vLLM 82.2 / 我们 78.2 |
| `812bd59` 07-24 | 479 tok/s **surpasses vLLM 376.9** | 改 `laguna_dflash.py`；**我们 333 token 的 agent 编码 prompt vs vLLM 64K 合成文本**——不可比 |
| `66d5913` 07-25 | 80.4 tok/s，**+14% vs vLLM 70.5** | 改 `laguna*.py`；vLLM 基线配置未在提交里写明 |

📌 **`e6e94d0` 是这组里唯一一次同配置对比，结论是我们慢 5%。** 其余"倍数"级声称的对照组
都是未开 CUDA Graph 的 vLLM，或不同工作负载。这正是 `AGENTS.md` 里 `bf diff` 那条规矩的
由来（2026-07-27 因为比较了两个不可比的接受率而损失一天）。

## 四、对 Track B / Track F 的直接含义

1. **B3 的性能目标应当锚定"稳态 decode ≥ vLLM"**，因为历史上这一项赢过（1.33×）。
2. **TTFT 是历史上最大的一块（60–70%）**，对应 M1「跨步交错的 chunked prefill」。
   ⚠️ 与今天已知的另一件事叠加会更糟：sparkinfer JIT **按每个不同 `(seq_len, cache_seqlens)`
   形状重编译**，B1 实测一个 8-token prompt 付了 ~24 秒（`implementation-plan.md` §7.1 B0-3）。
   **prefill 侧同时背着"调度不交错"和"每个新形状重编译"两个问题。**
3. ~~**接受长度（3.3 vs 4.85）当年归因于 kernel 数值分歧**~~ 🔴 **前提已作废**（见上表）：
   原生那个 4.85 是 benchmark 双重计数造成的，修正后我们（3.0）反而略优于原生（2.926）。
   **「接受长度落后于原生」这个问题当年就不存在。** 下面这段保留存证——今天换了 sparkinfer + FLA，
   这条要重测，不能沿用。
4. 报告口径必须同时给 **含 TTFT** 与 **稳态** 两个数，否则会重演"稳态赢了但标题输了"的误读。

## 相关

- [`2026-07-20-comprehensive-optimization-plan.md`](2026-07-20-comprehensive-optimization-plan.md)（原始计划，152 行，6 个瓶颈分析 + quick wins + M1/M2/M3）
- [`2026-08-02-trackB-b0-gpu-facts.md`](2026-08-02-trackB-b0-gpu-facts.md)（sparkinfer/FLA 的今日事实）
- [`2026-08-02-evaluation-artifact-provenance.md`](2026-08-02-evaluation-artifact-provenance.md)（同一类归属事故的另一例）
