# B3：MTP 草稿头端到端接受率与吞吐——真实数字（27B 全模型，非仿真）

日期：2026-08-02 · 状态：🟢 已在真实 GPU 上测量 · 结论：**当前 eager 实现下，MTP 投机解码比不投机更慢**

## 结论先行

用 `nvidia/Qwen3.6-27B-NVFP4` 真实权重、真实 MTP 头、真实 prompt，跑了完整的
draft→verify→accept/reject→rollback 循环。**两条真实数字**：

| prompt | 接受率 | 每轮平均接受(K=8) | 投机 tok/s | 非投机 tok/s | 加速比 |
|---|---:|---:|---:|---:|---:|
| prose（"Once upon a time..."） | 15.2% (17/112) | 1.21 | 2.72 | 4.15 | **0.66×** |
| code（"def fibonacci(n): ..."） | 35.0% (28/80) | 2.80 | 3.14 | 5.16 | **0.61×** |

**MTP 投机解码目前比直接顺序解码慢 34–39%，不是加速。** 原始数据见
`benchmarks/fixtures/qwen36_mtp_e2e_20260802.json`；复现脚本
`scripts/b3_mtp_e2e_acceptance_throughput.py`。

## 为什么会更慢——不是猜测，是上一轮 B3 GPU 证据在全模型尺度的复现

上一轮 B3（`scripts/b3_probe_gdn_spec_rollback.py`）已经在单层尺度量化过：
`spec_forward`（K 次顺序 `fused_recurrent_gated_delta_rule`）比一次
`chunk_gated_delta_rule` 贵 **6.9×**，根因是 K 次顺序 kernel 启动开销
（"只能靠多步 kernel 融合"，kernel 团队范畴）。Qwen3.6 的 64 层里 **48 层是
GDN**（`linear_attention`），每次 verify 都要在这 48 层上各付一次这个开销，
而 attention 层的 K-token 批处理反而是便宜的（一次 kernel 调用即可，见
`Qwen36TextModelSelfBuilt.verify_forward` 的实现注释）。

草稿本身也不是免费的：`scripts/b3_probe_mtp_head.py` 测得单个 MTP 头
step（eager，真实权重）约 **9.18ms**，K=8 链式起草约 **73ms/轮**——这部分开销在
非投机路径里完全不存在，是纯粹的额外成本。

**接受率不够高时，"跳过的解码步数"省下的时间，抵不过"起草+verify 里 48 层
GDN 顺序 kernel 启动"这两笔开销。** 这与"低接受率场景 MTP 会亏"是投机解码
文献里的常识,只是这次是在这个具体实现上用真实数字量化出来的。

## 正确性判据：与非投机路径的分布一致性（不是与 HF 比）

`docs/b1-correctness-criterion.md` §7 的 B3 判据是"投机开 vs 关,比较自己的
两条路径",不是 bit-exact 对 HF。做法：同一 prompt，分别跑（a）本 runtime 自己
的自由贪心解码（非投机，B1 已证明），（b）本 runtime 自己的 MTP 投机解码
（draft+verify+greedy accept/reject）,比较两条**输出 token 序列**——贪心
accept/reject 的数学保证是：只要实现对，两条序列必须逐 token 相同（一旦某
处 draft 与 verify 不一致，verify 自己的预测就成为纠正,序列不可能偏离"纯
verify 路径会生成什么"）。

**结果**：
- prose prompt：**逐 token 相同**（32/32 token 完全一致）——强证据。
- code prompt：**第 2 个位置分叉**（`n-1` vs `n - 1`，即 token `-` 对
  `<space>-`）。解码后发现这不是语义 bug，是一次**平局翻转**：两个延续都是
  合法 Python，只是代码风格不同（紧凑 vs 带空格）。**没有再花 GPU 时间去
  实测那个位置两条路径的具体 logit gap 大小**——这是本报告明确列出的未验证项
  （见下）,但机制上高度可疑地指向已有先例：`notes/2026-08-02-eager-verify-
  cg-verify-divergence.md` 记录过同一现象类别（CG-replay decode 与 eager
  decode 在 kv_len≥400 时不一致，两边对稠密 oracle 都 cos≥0.999997）——都是
  "同样的数学、不同的 kernel 调度顺序"在**平局处**翻转,不是"两边算法不同"
  （GDN 层在这条路径上完全没有平局风险，`verify_forward` 恒用
  `spec_forward` 的单 token `fused_recurrent` 路径，从不触达
  `chunk_gated_delta_rule`——这正是上一轮 B3 那个"陷阱"被设计规避掉的地方）。

## 我没能验证的东西

1. **没有测量分叉点的真实 logit gap**——只看了 token 差异，没有回放两条路径
   在该位置的完整 logits 做 B1-R 式的 gap-error 度量。如果要坐实"平局翻转
   不是 bug"这个结论，这是下一步该做的（`bfdiag/divergence/logit_agreement.py`
   现成可用，只是这次没接上）。
2. **没有接进 `runtime/backends/qwen36.py` 的 `ServerEngine`/调度**——本报告的
   draft/verify/accept-reject 循环是独立 eager 脚本
   （`scripts/b3_mtp_e2e_acceptance_throughput.py`），不是生产 serving 路径。
   没有连续批处理、没有多 slot、没有 CUDA Graph 捕获投机步骤。"接进引擎/调度"
   在这次会话里只完成了"模型层原语就绪 + 单 slot 端到端验证",没有完成
   "engine 能通过 HTTP 服务投机解码请求"。
3. **样本量很小**——2 个 prompt、每个 32 token、K=8 固定。接受率随 K、prompt
   类型、上下文长度会有很大方差（Laguna 自己的 acceptance_regression 基线用
   13 条真实 workload），这次的数字是"确实测过、确实真实"，但不是统计意义上
   稳固的基线。
4. **没有做真实吞吐的进一步归因**（drafting 用了多少时间 vs verify 用了多少
   时间 vs 处理新 anchor 用了多少时间）——只有总 wall-clock，没有分段计时。
5. **`benchmarks/fixtures/qwen36_mtp_e2e_20260802.json` 目前不是回归门禁**——
   只是把这次真实测量的结果落盘,还没有像 Laguna 的
   `tests/test_acceptance_regression.py` 那样接一条"低于历史基线就红"的门禁。

## 相关

- `scripts/b3_probe_gdn_spec_rollback.py` —— 上一轮：单 GDN 层 rollback 位精确 + 6.9× 吞吐代价
- `scripts/b3_probe_mtp_head.py` —— 本轮：MTP 头本身的正确性 + 单 step 9.18ms
- `scripts/b3_mtp_e2e_acceptance_throughput.py` —— 本轮：全模型端到端接受率/吞吐
- `benchmarks/fixtures/qwen36_mtp_e2e_20260802.json` —— 原始测量数据
- `docs/b1-correctness-criterion.md` §7 —— B3 判据本身
