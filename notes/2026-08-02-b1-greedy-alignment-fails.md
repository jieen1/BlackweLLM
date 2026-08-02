# B1 逐 token 对齐门禁：不通过（实测，非推断）

日期：2026-08-02 · 状态：🔴 门禁未达成，根因方向已收窄 · 执行：协调者直接跑

## 结果

`scripts/b1_verify_greedy_alignment.py`，3 工作负载 × 512 token，对照 HF
`Qwen3_5ForCausalLM`：

```
all_fully_aligned = False
overall_match_rate = 0.3287
```

| 工作负载 | prompt_len | 匹配率 | **首次分歧** | 绝对位置 |
|---|---:|---:|---:|---:|
| factual-short | 8 | 0.4431 | **218** | 226 |
| math-short | 5 | 0.0723 | **32** | 37 |
| instruction-longer | 17 | 0.8571 | **120** | 137 |

⚠️ **匹配率本身有误导性**：贪心是自回归的，一个 token 翻转后两侧走上完全不同的轨迹，
后续比较已无意义。**有信息量的只有首次分歧位置。**

## 一个自然猜想被排除了

直觉会先怀疑 NVFP4 反量化（本仓库已记录"nibble 打包顺序在这台机器上没有独立 oracle"）。
**但排除得掉**：本脚本里 HF 侧的权重是**从我们自己反量化的张量拷过去的**，两侧权重逐位相同。
反量化若有错，两侧会**同样地**错。

所以差异只能来自**前向数学**。这同时也是这个门禁的固有局限（脚本自己的 docstring 已写明）：
它验证前向，**不验证反量化**。

## 分歧处的实际内容（解码后）

**factual-short @218**：
```
我们:  ' same paragraph\n\n<think>\n\n</think>\n\nThe first president of the United'
HF  :  ' same year as the American Revolution. He was a farmer and a'
```
`248068/248069` = `<think>`/`</think>`，`271` = `'\n\n'`。
**我们在约 218 步后退化成吐控制 token 并重启回答**；HF 自然续写。

**instruction-longer @120**：我们随后是 `[248045, 271, 248045, 271, ...]` 交替循环
（`248045` = `<|im_start|>`）——同样是控制 token 退化。

**math-short @32**：两侧**都**跑去写阿塞拜疆语的《1984》内容（裸 `/v1/completions` 补全、
无 chat template，基座模型的正常行为），仅在 `' Or'` vs `','` 上分叉。
📌 **这一条不能当作 bug 证据**：低置信度区域的近似平局翻转，与另两条性质不同。

## 分歧位置没有结构规律

| workload | 绝对位置 | mod 64 | mod 128 |
|---|---:|---:|---:|
| factual-short | 226 | 34 | 98 |
| math-short | 37 | 37 | 37 |
| instruction-longer | 137 | 9 | 9 |

**不是页/块边界**（`block_size=64`）。排除了"跨页时 KV 寻址错"这一类假设。

## ✅ 已坐实（2026-08-02 晚，教师强制实测）：分歧全部发生在 bf16 的表示极限上

按上文列的判定方法实测——把 `prompt + oracle[:i]`（两侧在此之前逐 token 相同）喂进我们的
模型，读第 `i` 位的 logits：

| workload | 我们选 | HF 选 | logit 间距 | 该量级的 bf16 ULP | = 几个 ULP |
|---|---|---|---:|---:|---:|
| factual-short | 13901 | **1007** | 0.0625 | 0.0625 | **1** |
| math-short | 2441 | 11 | 0.1250 | 0.0625 | **2** |
| instruction-longer | 271 | 846 | 0.1250 | 0.0625 | **2** |

bf16 有 7 位显式尾数，在 8–16 区间 ULP = `2^3 × 2^-7 = 0.0625`。
**三处分歧的两个候选 token 相差 1–2 个 ULP，即 bf16 能表示的最近距离。**

⚠️ **这推翻了本文档上一版的定性。** 这不是"实现有 bug 导致退化"，而是
**近似平局翻转**——与 `2026-08-02-eager-verify-cg-verify-divergence.md` 的结论同一性质：
两个都正确的实现，在 1 ULP 的平局上必然会分道扬镳，之后自回归把差异放大成完全不同的文本。
分歧后出现的控制 token 循环是**走了另一支之后的结果**，不是原因。

### ❌ 一条我差点误报的"真问题"——已自我推翻，保留全过程

原本写的是"我们自己的增量路径与全前向不自洽，疑似 bug"。**不成立，理由在代码里。**

`factual-short` 那行值得单独看：**教师强制下我们的 top-1 是 1007，正是 HF 的选择**；
而自由生成时我们选了 13901。**同一前缀、同一模型、不同结果。**

`runtime/model/qwen36_model.py:451`：

```python
if state.has_previous_state and seq_len == 1:
    core_attn_out, last_state = fused_recurrent_gated_delta_rule(...)   # 增量 decode
else:
    core_attn_out, last_state = chunk_gated_delta_rule(...)             # 整段全前向
```

**这是 FLA 的两个不同算法**——数学等价、数值不同，而且 **HF 自己做完全相同的分派**。
所以"增量 vs 全前向"这个对比测的是**分块算法 vs 递归算法**，它们从来就不该逐位相同。
不是 bug。

（这也解释了 factual-short 那个反常观察：教师强制时我们的 top-1 是 1007、与 HF 一致，
而自由生成时是 13901——两次走的是不同的 GDN 算法。）

#### 但实测数据本身仍有价值，而且量级出乎意料

同一前缀下 `chunk` 与 `fused_recurrent` 的 logits 差异（factual-short，prompt 8 token）：

| 生成步数 | 绝对位置 | cosine | max_abs_err | mean_abs_err |
|---:|---:|---:|---:|---:|
| 1 | 9 | 0.99997658 | 0.1250 | 0.0249 |
| 50 | 58 | 0.99996507 | 0.1250 | 0.0176 |
| 100 | 108 | 0.99985880 | 0.1250 | 0.0227 |
| 180 | 188 | 0.99857700 | **1.1406** | 0.1442 |
| 184 | 192 | 0.99995351 | 0.3438 | 0.0274 |
| 188 | 196 | 0.99811924 | **1.9375** | 0.1880 |
| 218 | 226 | 0.99869859 | 0.9141 | 0.1117 |

**与 KV 页边界无关**：`block_size=64`，边界（绝对位置 64/128/192）处的偏差反而**低**
（0.125 / 0.234 / 0.344），非边界的 188/196 反而最高。所以不是分页寻址问题。

⚠️ **方法论结论**：两种 GDN 算法在 logits 上可以差到 **~1.9**（≈30 个 bf16 ULP）。
这意味着**在这个模型上不能用教师强制去验证自由生成路径**——两者走不同算法，
差异远大于 argmax 的平局间距（1–2 ULP）。任何"喂进已知前缀看下一个 token"的验证方法
对 Qwen3.6 都无效。

- [x] ~~B1-selfcheck：增量 vs 全前向应当一致~~ —— ❌ **前提错误，此项作废**（见上）

## 原先的解释（保留作记录，已被上文取代）

单层实测（`notes/2026-08-02-trackB-b0-gpu-facts.md` 与 B1 step 2）：

| 层 | cosine | max_abs_err |
|---|---:|---:|
| GDN | 0.99998832 | 9.8e-4 |
| **Attention** | 0.99998492 | **0.0156** |

attention 的 `max_abs_err = 0.0156` 在 bf16 下不算小。**逐层误差在自回归几百步后累积到
足以翻转 argmax**，翻转后进入退化区域——与观察到的"先一致一两百步、然后突然退化"一致。

**这是一个待验证的解释，不是结论。** 要坐实需要：
- [ ] 逐 token 记录两侧 top-1 与 top-2 的 logit 间距，看首次分歧是否发生在近似平局处
      （若是 → 累积误差假说成立；若 top-1 差距很大仍翻转 → 是别的更严重的问题）
- [ ] 定位 attention 层 0.0156 误差的来源（RoPE / q_norm/k_norm / 输出门 / o_proj 逐段拆）
- [ ] `math-short` 与另两条分开处理——它大概率是良性平局

## 对计划的影响

- **B1 门禁按字面写法未达成**，`IMPLEMENTED_BACKENDS` 保持 `{"laguna"}` 不翻。
- ⚠️ **门禁写法本身值得复审**：512 步逐 token 全对齐，要求两个独立实现（sparkinfer paged
  attention vs HF eager）在 bf16 下几百步不出现任何 argmax 翻转。本仓库刚在
  eager-vs-CG 那次证明过，**两个都正确的实现也会在近似平局处翻转**
  （`notes/2026-08-02-eager-verify-cg-verify-divergence.md`）。
  所以合理的判据可能是"首次分歧处必须是近似平局"，而不是"永不分歧"——**但这个判据要先被
  论证，不能因为当前跑不过就放宽**。

## 相关

- `scripts/b1_verify_greedy_alignment.py`（本次运行的脚本，含三处 OOM 修复史）
- `/home/bot/project/qsr-w-b1/.bfdiag/runs/b1_greedy_alignment.json`（原始结果）
- [`2026-08-02-eager-verify-cg-verify-divergence.md`](2026-08-02-eager-verify-cg-verify-divergence.md)（"两边都对但 argmax 翻转"的先例）
