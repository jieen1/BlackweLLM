# mtpfix：为什么今天 Qwen3.6 MTP 接受率远低于历史 K=3 记录——历史代码对照

> 🔴 **本文的核心比较不成立(2026-08-02 晚,用户提问后核实)。**
>
> 本文把历史的 "mean acceptance length ≈4.0(≈100% of cap)" 与今天的 "0.401(52–86% of cap)"
> 并列,并称差距是真的。**两次测量的负载完全不同,不可比:**
>
> | | 历史 ≈4.0 | 今天 0.401 |
> |---|---|---|
> | 脚本 | `benchmarks/native_warm_compare.py` | `scripts/b3b_k_sweep.py` |
> | 上下文 | **128K 前缀 + 10240 token 新后缀** | `MAX_SEQ_LEN = 512` |
> | 生成 token | 256 | **64** |
> | 并发 | **c=4** | c=1 |
> | 前缀缓存 | 温命中 | 无 |
>
> **上下文差 250 倍。** 而接受率强烈依赖负载:本仓库自己的
> `benchmarks/fixtures/acceptance_regression_*.json` 显示同一模型在不同负载上从
> **18%(ids-cycle-4K)到 100%(qa-*)**。
>
> **单位换算是对的(4.0 对 cap 4 = 100%,0.401 换成长度 2.2 对 cap 4 = 55%),
> 错的是把两个不同负载的数放在一起比。** 换算正确不等于比较成立。
>
> 这正是 `AGENTS.md` 里 `bf diff` 那条规矩要防的事——2026-07-27 曾因比较两个不可比的
> 接受率损失一整天。**协调者(我)在下发本任务时没有核实负载是否一致,是这次的直接原因。**
>
> **仍然成立的部分**(与该比较无关):
> - accept/reject 逻辑是历史代码的逐字节移植 —— 已排除
> - "历史每轮全量重同步 vs 今天只对齐 anchor" —— **实现差异真实存在**,值得独立评估
> - checkpoint 发布方不同(历史 47 个脚本用 `unsloth/`,今天用 `nvidia/`)—— 真实差异,未隔离
>
> **不再成立的部分**:任何以"历史 ≈100%"为目标或参照的推论。

日期：2026-08-02 · 分支 `work/mtp-accept-fix-20260802` · 只读参考 `/home/bot/project/qsr-hist-mtp`（commit `8f5c195`）

## 口径先对齐（这一步做错后面全错）

历史说的是 **mean acceptance length**（K=3 时上限 K+1=4，`PROGRESS.md`："our mean
acceptance length ≈ 4.0"）；今天 b3b 说的是**接受率**（每个草稿槽位的接受概率）。换算：
`mean_length = mean_accepted_per_round + 1 = accept_rate * K + 1`。统一换算到"每轮
committed 长度 / (K+1) 上限"这个口径：

| 来源 | K | accept_rate | mean_accepted/round | mean_length | mean_length/(K+1) |
|---|---:|---:|---:|---:|---:|
| 历史（`PROGRESS.md`，vLLM 原生执行栈，`unsloth/Qwen3.6-27B-NVFP4`） | 3 | — | — | **≈4.0** | **≈100%** |
| 今天（本次 mtpfix 实测，自研栈，`nvidia/Qwen3.6-27B-NVFP4`，prose） | 3 | 0.356 | 1.07 | 2.07 | 52% |
| 今天（本次 mtpfix 实测，自研栈，同上，code） | 3 | 0.815 | 2.44 | 3.44 | 86% |
| 今天（B3-b 独立复测，`notes/2026-08-02-b3b-acceptance-rate-vs-k.md`，prose） | 3 | 0.401 | 1.20 | 2.20 | 55% |
| 今天（B3-b 独立复测，同上，code） | 3 | 0.615 | 1.85 | 2.85 | 71% |

两次独立测量（本次 + B3-b）在同一 K=3 下互相印证（52–55% / 71–86%），且**都远低于历史的
~100%**。**结论：不是 K 的差异**（K=1 的第 1 项已由 B3-b 的真实 K 曲线答掉，见下方"已排除"）——
即使把 K 调回历史的 3，今天的接受率也只有历史的一半到六分之一。

## 已排除：K 本身（第 1 项，另一 agent 已在真实 K 曲线上答过，本次独立复测互相印证）

`notes/2026-08-02-b3b-acceptance-rate-vs-k.md` 的真实 K 曲线（K=1..16）+ 本次 mtpfix 自己在
`nvidia/Qwen3.6-27B-NVFP4` 上独立跑的 K=3/K=8 对照（`scripts/mtpfix_k_sweep_selfbuilt.py`，
`.bfdiag/runs/mtpfix_k_sweep_selfbuilt.json`）方向完全一致：接受率随 K 单调下降，但即使在历史
使用的 K=3 上，今天的接受率也远没有到~100%。**如果差距纯粹是 K 的问题，K=3 应该已经接近满
额——它没有。** 排除。

## 已排除：accept/reject 判据（第 3 项）

逐行对照：

| | 历史 | 今天 |
|---|---|---|
| 文件:行号 | `qsr-hist-mtp/runtime/direct_model_runner.py:1215-1236`（单个）/`1239-1305`（批量） | `runtime/mtp_accept.py:26-56`（`determine_accept_reject_from_predictions`，单个的核心，`determine_accept_reject`59-71 只是套壳）/`74-144`（批量） |
| 判据 | `predicted = verify_logits[p].argmax(-1)`；`predicted == draft_tokens[p+1]` 则接受，否则用 `predicted` 作为 recovery token，`rejected_at=p` | 完全相同：`predicted_tokens[p] == draft_tokens[p+1]` 则接受，否则 `predicted_tokens[p]` 作为 recovery，`rejected_at=p` |
| 全部接受时的 bonus token | `verify_logits[k].argmax(-1)` | `predicted_tokens[k]`（调用方把 `verify_argmax` 的第 k 项传进来，同一来源） |

`runtime/mtp_accept.py` 的模块 docstring 明确写着"从 direct_model_runner.py 提取的
determine_accept_reject* 纯函数……纯移动不改逻辑（B5 parity 门禁）"——**这是有意的字节级
移植，不是独立重写**。两边都是贪心 top-1 相等判据，没有温度/采样差异（`runtime/mtp_accept.py`
里确实另有非贪心的 `sample_accept_reject`，但今天和历史的 MTP 路径都没接它，全程
`temperature=0`——`notes/2026-08-02-b3b-acceptance-rate-vs-k.md` 第 4 节已经核实过这一点）。
**排除。**

## 有实证支持、但不足以解释全部差距：起草机制的"每轮重接地"缺失（第 2 项）

### 历史机制：每一轮都用 TARGET 的真实隐状态重新"冲刷"整段新提交范围

`_mtp_sync_and_propose`（`qsr-hist-mtp/runtime/direct_model_runner.py:3004-3056`）的 step 0
不是"起草第一个 token 那一步"这么简单——它是一次**多 token teacher-forced 前向**
（`_mtp_forward`，`2892-3002`，`num_new_tokens` 可以 >1），覆盖**上一轮全部新提交的范围**
（`committed_len` 个位置：包括本轮所有被接受的草稿 token，加上新 bonus/recovery token），
每一个位置的 `prev_hidden` 都直接来自 TARGET 在同一次 verify 前向里算出的真实隐状态
（`mtp_verify_and_commit`，`3104-3237`，尤其 `3174-3237` 的 docstring："Draft catch-up +
next-round propose, folded into ONE call"；`real_new_hidden = verify_hidden[:committed_len]`）。
换句话说：**哪怕这一轮有 3 个草稿 token 全部被接受，历史实现也会在下一轮开始前，用 TARGET
的真实隐状态重新计算这 3 个位置的 draft 头 KV，覆盖掉起草阶段（自条件链）写入的旧值**。
`_mtp_forward` 自己的 docstring 说得很直白："an exploratory step's positions are simply
overwritten by the next round's real sync call"。

批量版是 `_mtp_run_continuation_steps`（`3390-3457`，协调方指名要对照的那个）——它只负责
step 0 **之后**的 k-1 个自回归探索步（`prev_hidden = step_hidden`，链式，不接地），本身不做
重接地；真正做重接地的是它的姐妹方法 `_mtp_sync_and_propose_batch`（`3459+`）自己的 step 0
调用，逻辑和单槽版 `_mtp_sync_and_propose` 一致，只是批处理。

### 今天的机制：每轮只有 1 个位置（anchor）接地，其余永久保留自条件链的值

`Qwen36MTPHead.forward`/`mtp_step`（`runtime/model/qwen36_model.py:1943-1967`/`2332-2365`）
本身支持多 token（`forward` 的 `next_token_embeds`/`prev_hidden` 形状是
`[1, seq_len, hidden_size]`），但 `mtp_step` 把它包死成单 token（`[1,1]`），而
`scripts/b3_mtp_e2e_acceptance_throughput.py:114-188`（`speculative_decode`）的驱动逻辑
从不做多 token 重接地：

```
148  while len(committed) < n_tokens:
149      round_mtp_start = mtp_cache.seq_len
151      mtp_hidden = anchor_hidden          # 只有这一行来自 TARGET 的真实隐状态
152      next_input = anchor_token
153      for _step in range(k):
154          draft_token, mtp_hidden = model.mtp_step(next_input, mtp_hidden, ...)
                                        # step 1..k-1 全部链式吃自己上一步的输出
...
172      model.commit_verify(...)
173      mtp_cache.seq_len = round_mtp_start + m   # 只是把指针截断，不重写任何 K/V
178      new_anchor = decision["committed"][-1]
180      new_anchor_logits, new_anchor_hidden = _logits_for(model, new_anchor_tensor, state)
                                        # 单独一次 TARGET 前向，只为了下一轮的 anchor
```

`Qwen36ForCausalLMSelfBuilt.verify_forward` 的 docstring
（`runtime/model/qwen36_model.py:2101-2104`）其实点名了这条路：`post_norm_hidden`
"...(row-wise) to Qwen36MTPHead as the next round's prev_hidden once accept/reject picks
which row"——也就是说，`verify_hidden[i]`（TARGET 在 verify 那次前向里对第 i 个被接受草稿
位置算出的真实隐状态）本来就是现成的、可以用来重接地 position 1..m-1 的数据，但
`speculative_decode` 从未使用它做这件事——第 173 行 `mtp_cache.seq_len = round_mtp_start + m`
只是**截断指针**（对被拒绝/未验证的探索步是对的，等于历史的"overwrite"半句），**但对
0..m-2 这些已接受的位置，物理 K/V 仍然是起草阶段自条件链算出的旧值，从未被目标模型的真实
隐状态重写过**（historical 那半句"next round's real sync call" 在今天完全没有对应实现）。

**证据链**：
- `scripts/b3b_teacher_forced_head_quality.py` 的独立测量证实了这个"链式自条件复合漂移"的
  症状——K=8 时逐位置准确率 `0.625, 0.625, 0.542, 0.542, 0.354, 0.354, 0.312, 0.333`，
  "衰减后平台"形状，`notes/2026-08-02-b3b-acceptance-rate-vs-k.md` 第 55-104 行的归因是
  "头一旦在某个位置偏离真实延续，后续吃的是自己那个偏离后的、但仍然自洽的续写"——这正是
  "缺少每轮重接地"的可观测后果：一旦某个位置的隐状态因链式自条件偏离 TARGET 真实轨迹，
  后续所有位置的自注意力都会持续吃到这个偏离，而且**跨轮持续存在**（因为下一轮不会重新计算
  它），不像历史实现那样每轮开头被冲刷掉。
- 但**这条差异不足以解释全部差距**：`scripts/b3b_teacher_forced_head_quality.py` 测的是
  "完全零复合"的最佳情形（每一步都喂真实 token + 真实 backbone 隐状态，不吃头自己的任何
  历史输出）——这正是"如果每轮都像历史那样重接地，起草头能达到的单步上限"。测出来是
  **62.9%（prose）/ 82.4%（code）/ 71.1%（instruction）**，离历史隐含的 ~100% 还差一大截。
  也就是说：**就算把"每轮重接地"补全，今天这份自研栈 + nvidia 权重的单步接受上限也就在
  63–82%，不是 100%**。重接地缺失能解释"为什么会复合变差"，但不能解释"为什么单步基线本身
  就比历史低"。后者指向第 4 项。

**本次未做的追加实验**：把"重接地"补丁实现出来（用 `verify_hidden[i-1]` 作为
`mtp_step` 第 1..m-1 个位置的 `prev_hidden`，重放这些位置而不是只截断指针）、跑一次 K=3
对照，直接测出"补上重接地能不能让 K=3 从 52–86% 回到接近 100%"。设计已经想清楚（见上面的
逐位置映射），但受限于本次时间预算没有落地成代码。**这是本次报告里最大的一处未验证项**。

## 有强证据、可能是最大单一因素：执行栈 + checkpoint 发布方都不同（第 4 项）

### 执行栈：历史用 vLLM 原生 MTP 加载器，今天是从零手写的独立实现

`qsr-hist-mtp/runtime/direct_model_runner.py:1507-1523`：

```python
1510    if vllm_config.speculative_config is not None:
1511        from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
...
1515        self.mtp_model = load_eagle_model(self.model, vllm_config)
```

历史的草稿模型是 vLLM 自己的 `load_eagle_model`（vLLM 上游、成熟、被许多模型复用的
EAGLE/MTP 加载路径）加载出来的，vLLM 内部把这个 Qwen3.6 MTP 模块叫
`Qwen3_5MTP`/`Qwen3_5MultiTokenPredictor`（`direct_model_runner.py:2902/2959/3261/5811`
的 docstring 里反复出现这个类名——vLLM 沿用了 Qwen3.5 世代的类名，但加载的权重和跑的路径
是 Qwen3.6 checkpoint 自带的 `mtp.*` 张量，不是另一个模型）。target 模型走这个仓库自己的
SM120 CUSTOM attention backend，但 MTP 草稿模型的 forward/权重加载/位置编码全部是 vLLM 自己
的代码，没有一行是这个仓库手写的。

今天的路径（`runtime/model/qwen36_model.py`）类名本身就写明了区别——
`Qwen36ForCausalLMSelfBuilt`/`Qwen36TextModelSelfBuilt`/`Qwen36MTPHead`，"SelfBuilt" 后缀
不是随便起的：`runtime/model_loading.py` 整个文件没有一处 `import vllm`（`grep -n
"^from vllm\|^import vllm" runtime/model_loading.py` 零命中），backbone 前向、RoPE、
attention、MTP 头全部是这个仓库独立手写的实现，从未与 vLLM 自己的 Qwen3.6 MTP 实现做过
数值对照。`scripts/b3_probe_mtp_head.py` 自己的 docstring 也承认这一点："not a numerical-
correctness claim against any oracle -- there is no non-speculative Qwen3.6 MTP path to
compare against"——**这句话现在看是不准确的**：vLLM 自己的原生 Qwen3.6 MTP 路径就是一个
现成的 oracle，历史仓库已经跑过它、量出过 ~4.0/4，只是没人把它当作"这个自研 MTP 头对不对"
的验证基准去对照过。

### Checkpoint 发布方：历史用 unsloth，今天用 nvidia——两个不同的量化包

```
$ grep -rln 'MODEL\s*=\s*"unsloth' qsr-hist-mtp/benchmarks/*.py | wc -l
47
$ grep -rln 'MODEL\s*=\s*"nvidia' qsr-hist-mtp/benchmarks/*.py | wc -l
0
```

历史仓库（commit 8f5c195）**全部 47 个设置 `MODEL = "..."` 的 benchmark 脚本**都指向
`unsloth/Qwen3.6-27B-NVFP4`，**零个**指向 `nvidia/Qwen3.6-27B-NVFP4`。今天的
`scripts/b3_mtp_e2e_acceptance_throughput.py`/`scripts/b3b_*.py`/本次 mtpfix 的脚本全部用的是
`nvidia/Qwen3.6-27B-NVFP4`。两个 checkpoint 都在本机缓存里
（`~/.cache/huggingface/hub/models--{nvidia,unsloth}--Qwen3.6-27B-NVFP4`），架构声明相同
（`Qwen3_5ForConditionalGeneration`），`mtp.*` 张量名和数量也相同（都是 15 个，名字逐一相同，
见下），但这不代表权重数值相同或量化质量相同——两个发布方的 NVFP4 校准方法不同（unsloth 以
自己的动态量化/校准流程著称，nvidia 用自己的 ModelOpt 流水线），MTP 头只有一层、参数量小，
比 64 层 backbone 对量化误差的冗余度低得多，两种校准跑出来的头质量完全可能有实质差异。

**本次尝试直接验证（结果：被结构性问题挡住，不是"跑了发现一样"）**：
`scripts/mtpfix_unsloth_checkpoint_probe.py` 把 `MODEL_PATH` 换成
`unsloth/Qwen3.6-27B-NVFP4`，其余代码逐行不变，试图直接测"同一份自研代码，换成历史用的
checkpoint，K=3 接受率会不会回到 ~100%"。**加载阶段就失败**：

```
RuntimeError: load_qwen36_model: 168 parameter(s) never received a checkpoint tensor,
e.g. ['model.layers.0.mlp.down_proj.weight', ...]
```

原因：`unsloth/Qwen3.6-27B-NVFP4` 的 `config.json` 声明的是**混合精度量化**——
`quantization_config.config_groups.group_0` 是 int8/float8 动态量化，覆盖大部分投影层；
`group_1`（真正的 NVFP4）只覆盖 **56-63 层的 `mlp.(gate|up|down)_proj`**。而
`nvidia/Qwen3.6-27B-NVFP4` 是均匀 NVFP4（这个仓库的 `runtime.loading`/
`quantized_layers_map` 就是照着 nvidia 这份布局写的、测的）。两个"同名"checkpoint 实际上是
**不同的每层精度方案**，不是同一份权重换了个量化包装。这本身就是一条独立、扎实的发现：
**"nvidia" 和 "unsloth" 的 Qwen3.6-27B-NVFP4 不能互换**，今天这份自研 loader 只认识 nvidia
的均匀布局。

**结果**：这条"直接换 checkpoint 验证"的实验被结构性问题挡住，**没能测出"unsloth 的 MTP
头权重，跑在今天的自研代码上，接受率会不会更高"**。要解锁这个对照，需要先给
`runtime.loading` 加混合精度量化层的识别支持——这超出本次投机接受率调查的范围，留作后续
工作。

## 结论：配置 or 实现？——两者都是，且实现/checkpoint 权重更大

1. **不是配置（K）本身**：K=3 下今天的接受率（52–86%，两次独立测量互相印证）远低于历史
   ~100%，K 只解释了"接受率会不会随 K 变差"这条曲线的形状（B3-b 已经答过），不解释"同样
   K=3 为什么历史和今天差这么多"。
2. **是"执行栈 + checkpoint 发布方"这两个叠加的配置/环境差异**——证据最扎实：历史是
   vLLM 原生 `load_eagle_model` 加载器 + `unsloth/Qwen3.6-27B-NVFP4`；今天是从零手写的
   `Qwen36MTPHead` + `nvidia/Qwen3.6-27B-NVFP4`。这两者中至少一个（很可能两个都有贡献）
   造成了"就算做到零复合误差的单步上限也只有 63–82%，不是 ~100%"这个观测（teacher-forced
   头质量测量）。**这本身就是结论**——但受限于 unsloth checkpoint 的混合精度布局，本次没能
   把"checkpoint 发布方"这一个变量单独隔离验证；"执行栈"这个变量目前也只有间接证据（vLLM
   自己的实现从未被引用为 oracle 做数值对照），没有直接的"用 vLLM 原生路径复现今天这两个
   prompt，看接受率是否回到 ~100%"这一实验。
3. **是（次要、有实证但不完整的）实现细节**：今天的驱动逻辑（`speculative_decode`）每轮只
   重接地 1 个位置（anchor），历史每轮重接地整段新提交范围，这个差异会让链式自条件漂移
   "只发生一次就消失"（历史）变成"跨轮永久累积"（今天）——teacher-forced 测量证实了
   "衰减后平台"这个复合漂移的症状存在，但同一测量也证明了"就算全部重接地，单步上限本身
   也只有 63–82%"，所以这条差异是**放大器，不是唯一病因**。

## 我没能验证的东西

1. **"重接地"补丁的直接效果**：设计已经想清楚（用 `verify_hidden[i-1]` 重放 `mtp_step`
   第 1..m-1 个已接受位置，而不是只截断 `mtp_cache.seq_len`），但没有写代码验证补上它之后
   K=3 接受率能提高多少。这是本报告最大的未验证项，也是最值得下一次 GPU 窗口去做的实验。
2. **unsloth checkpoint 上的直接对照**：被 `runtime.loading` 不支持混合精度量化布局挡住，
   没能测出"同一份自研代码换成历史 checkpoint，接受率会不会变化"。
3. **vLLM 原生路径复现今天的两个 prompt**：没有在 vLLM 原生 `load_eagle_model` 路径上跑
   "Once upon a time..."/"def fibonacci..."这两个具体 prompt 做直接对照——历史的 ~4.0
   是在别的 prompt/工作负载上测的（`benchmarks/native_warm_compare.py` 的长上下文缓存场景，
   128K/64K + c=4），不是同一个 13/19 token 短 prompt。不能完全排除"prompt 本身的可预测性
   差异"是另一个未量化的混杂因素（`notes/2026-08-02-b3b-acceptance-rate-vs-k.md` 自己也
   在"我没能验证的东西"第 7 条提出了同样的疑问：code 接受率显著高于 prose，没有用更多样
   prompt 验证是否是通则）。
4. **今天的自研 backbone/MTP 头是否在其他方面（非接受率）与 vLLM 原生实现数值一致**：
   `notes/2026-08-02-b3b-acceptance-rate-vs-k.md` 已经用 gap-error（而非 token 相等）验证过
   verify 路径内部"extend 模式 vs decode 模式"的分叉是良性平局（tie_slack=1.0 ULP），但那是
   自己跟自己比，不是跟 vLLM 原生实现比——没有做过跨执行栈的 logit 级别数值对照。
5. **量化误差本身对 MTP 头的影响幅度**：没有单独测过"同一份 bf16（非量化）MTP 头权重"作为
   参照——无法区分"63–82% 的单步上限"里有多少是 NVFP4 量化误差贡献的，多少是架构/训练本身
   的上限。

## 相关

- 历史参考（只读，未改动）：`/home/bot/project/qsr-hist-mtp`（commit `8f5c195`），
  `PROGRESS.md`、`runtime/direct_model_runner.py`
- 本次新增脚本：`scripts/mtpfix_k_sweep_selfbuilt.py`（K 可配置的接受率复测，独立复现了
  B3-b 的"K=3 不解释差距"结论）、`scripts/mtpfix_unsloth_checkpoint_probe.py`
  （尝试换 checkpoint，被混合精度量化布局挡住，附完整报错）
- 上游已合并到 main 的相关工作：`notes/2026-08-02-b3b-acceptance-rate-vs-k.md`
  （真实 K 曲线、teacher-forced 头质量、divergence gap-error）、
  `notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md`（K=8 固定时的原始负面结论）
- `docs/b1-correctness-criterion.md` §7 —— B3 判据（gap error，不是 token 相等）
