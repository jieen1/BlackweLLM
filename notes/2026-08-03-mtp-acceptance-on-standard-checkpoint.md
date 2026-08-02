# MTP 接受率在标准 checkpoint 上重测：**checkpoint 假说被证伪**

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）· K=4（shipped 默认）
· 单卡 RTX PRO 6000 Blackwell Max-Q · `scripts/b3_mtp_e2e_acceptance_throughput.py`

## 背景：这个实验为什么直到今天才能做

历史仓库 47 个 benchmark 脚本**全部**指向 `unsloth/`，而今天的 B3 脚本指向 `nvidia/`。
于是长期存在一个悬而未决的假说：**今天偏低的接受率是不是因为换了 checkpoint 发布方？**

这个对照此前**做不了**——`scripts/mtpfix_unsloth_checkpoint_probe.py` 在加载阶段就死
（`168 parameter(s) never received a checkpoint tensor`），因为当时没有 mixed-precision
adapter。该 adapter 于本日合入（`ca50017`），阻塞解除。

（顺带：脚本本身也曾因 `_ROOT` 硬编码到已删除的 worktree 而完全跑不起来，
见 `tests/test_scripts_no_hardcoded_worktree_root.py`。）

## 结果

| prompt | 接受率 | 每轮平均接受 (K=4) | 投机 tok/s | 非投机 tok/s | e2e |
|---|---:|---:|---:|---:|---:|
| prose | 30.0% (18/60) | **1.20** | 5.01 | 6.02 | **0.83×**（净亏） |
| code | 41.7% (20/48) | **1.67** | 6.37 | 5.97 | 1.07× |

**正确性通过**：两个 prompt 的投机路径与非投机路径 committed token 序列**逐 token 相同**
（`token sequences match: True`）——accept/reject 算法本身是对的，这次测的纯粹是**收益**。

## 结论：不是 checkpoint 的锅

和 `nvidia/` 上的记录（[`2026-08-02-b3-mtp-e2e-acceptance-throughput.md`](2026-08-02-b3-mtp-e2e-acceptance-throughput.md)，K=8）对齐着看。
接受率本身随 K 变化，不能直接比；**"每轮平均接受"才是那个跨 K 相对可比的量**：

| | nvidia (K=8) | 标准 (K=4) |
|---|---:|---:|
| prose | 1.21 | **1.20** |
| code | 2.80 | 1.67 |

**prose 上两者几乎完全相同（1.21 vs 1.20）。** 草稿头无论在哪个 checkpoint 上，
平均也就蒙对 ~1.2 个 token。**checkpoint 发布方不解释接受率差距，这条假说到此为止。**

⚠️ code 那一行**不构成结论**：K=8 vs K=4，链更长本来就可能累计更多接受，两者不可比。
要判 code 需要同 K 重测。**prose 那一行是可比的，且它是决定性的。**

历史的 "~4.0/4 (K=3)" 在 unsloth 上**同样复现不出来**。那个数字与今天的测法本就
不可比（128K 前缀 / c=4 并发 vs 512 token 单槽），已于 2026-08-02 撤回；
本次结果进一步说明**换 checkpoint 也追不回它**，别再把它当靶子。

## ⚠️ 但这组数字描述的不是生产状态

**非投机基线 5.97–6.02 tok/s —— 这是 eager。** 本脚本是裸 forward 循环，不走服务路径，
不捕获 CUDA Graph。而服务路径开 CG 是 **28.85 tok/s**
（[`2026-08-03-cudagraph-vs-eager-decode-throughput.md`](2026-08-03-cudagraph-vs-eager-decode-throughput.md)）。

这对投机解码**不是无关的缩放**：MTP 的收益取决于"一次 verify 是否比 K 次顺序 decode 便宜"，
而 CG 恰好把 decode 那一侧改变了 4.7×。**分母变了，比值就不能沿用。**

所以：

- **接受率结论（checkpoint 不解释差距）成立**——接受率是模型/权重性质，与 CG 无关。
- **e2e 比值（0.83× / 1.07×）只对 eager 路径成立**，不能外推到生产。
  MTP 在 CG 路径上是赚是亏，**目前仍然未知**。

这是本 session 第二次踩到同一形状的问题：**基线跑在 eager 上，结论却被当成运行时的性质。**

## 下一步

1. 在 **CG 路径**上重测 MTP e2e（需要让脚本走服务路径，或在脚本里开捕获）——
   这是判断 MTP 去留的真正判据。
2. code 那一行同 K（K=4）重测 `nvidia/`，才能说 code 上是否有 checkpoint 差异。
3. 重同步 A/B（`work/mtp-resync-20260802` 的 `aed0e2d`）仍无数据；
   注意它同样会落在 eager 基线上，除非一并改。
