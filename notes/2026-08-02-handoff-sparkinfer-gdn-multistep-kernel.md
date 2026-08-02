# 交接给 SparkInfer:GDN 多步融合 kernel

日期：2026-08-02 · 状态：📤 待转交 · 提出方：BlackweLLM

> 本仓库的硬约束是**不直接改 sparkinfer 源码**，需要的改动写成交接单。这是一份。

## 一句话

Qwen3.6 的投机解码需要在一次调用里推进 K 步 GDN 递归并**物化每一步的中间状态**。
今天只能用 K 次顺序 `fused_recurrent_gated_delta_rule` 做到，**顺序调用的 kernel 启动
开销本身就占了约 6.8ms/层**，这是我们在自己这一侧无法消除的部分。

## 为什么需要它（实测，不是估计）

Qwen3.6-27B 的 64 层里 **48 层是 GDN**。投机解码的 verify 阶段要把 K 个候选 token
跑一遍主模型；候选被拒时递归状态必须回滚到接受点。

我们的做法是 `spec_forward()`：K 次顺序 `fused_recurrent_gated_delta_rule`，物化 K+1 个
状态快照，于是回滚是一次 `.copy_()` 而不是重算。**正确性已 GPU 证明**：K=16 下每个接受数
m ∈ {0,1,5,8,15,16}，回滚后状态与顺序非投机解码 `max_abs_diff=0.0` 逐位相同。

**代价**（单层，eager，实测）：

| | 耗时 |
|---|---:|
| 一次 `chunk_gated_delta_rule` | 1.8 ms |
| `spec_forward`（K=16 次顺序 `fused_recurrent`） | **12.6 ms** = **6.9×** |

拆解：

| 成分 | 耗时 | 归属 |
|---|---:|---|
| 顺序递归的 **kernel 启动开销** | **~6.8 ms** | **需要 sparkinfer**（本单） |
| conv1d / 投影 / norm / clone 每步重跑 | ~5.7 ms | BlackweLLM 自己做（进行中） |

**端到端后果**：MTP 投机比顺序解码**慢 34–39%**（prose 0.66×、code 0.61×，真实 27B、
真实 prompt、K=8）。即使我们把上面那 5.7ms 全部消掉，~6.8ms 仍会留在每一层每一轮上。

## 具体请求

一个能在**单次 kernel 调用**内推进 K 步 gated delta rule、并**输出每一步中间状态**的入口。
概念签名：

```
fused_recurrent_gated_delta_rule_multistep(
    q, k, v, g, beta,          # [B, K, H, D]  K 个位置一次给全
    initial_state,             # [B, H, Dk, Dv]
    output_all_states=True,    # ← 关键：要 K+1 个快照，不只是最终态
) -> (out, states)             # states: [B, K+1, H, Dk, Dv]
```

**关键点是 `output_all_states`**。现有的 `chunk_gated_delta_rule` 只给最终状态，
所以不能用于回滚；`fused_recurrent` 给中间状态但只能一步一步调。缺的正是两者的交集。

## 验证方式（我们这边已就绪）

`scripts/b3_probe_gdn_spec_rollback.py` —— 单层真实 GDN + 真实 FP8 权重，不加载整模型。
它已经在用 K 次顺序调用的版本上跑出 `max_abs_diff=0.0`。

**新 kernel 的验收判据就是同一个探针给出同样的 0.0**，外加耗时对比。
⚠️ **不接受"cos 很高"** —— 回滚路径上的任何数值差异都会在自回归中放大。

## 相关

- `runtime/model/qwen36_model.py` —— `spec_forward` / `commit_spec_snapshot` 的当前实现
- [`2026-08-02-b3-mtp-e2e-acceptance-throughput.md`](2026-08-02-b3-mtp-e2e-acceptance-throughput.md) —— 端到端 0.66×/0.61× 的原始测量
- [`2026-08-02-trackB-b0-gpu-facts.md`](2026-08-02-trackB-b0-gpu-facts.md) §B0-4 —— FLA 在本卡上的基线数字

## 对照:Laguna 为什么不需要这个

Laguna 的 DFlash 投机接受率 96.3–100%、确实有效，**因为 Laguna 没有 GDN 层**，
verify 完全不付这份代价。**"投机对 Laguna 有效，对 Qwen3.6 也该有效"不成立**——
差别就在这 48 层上。
