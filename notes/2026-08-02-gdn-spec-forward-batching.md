# GDN spec_forward 批处理优化——真实数字 + 一个硬性约束的发现

日期：2026-08-02 · 状态：🟢 已在真实 GPU 上验证（单层，真实权重）· 分支：`work/gdn-batch-20260802`

## 结论先行

`Qwen36GatedDeltaNet.spec_forward`（B3 的 MTP verify 路径）把与递归无关的
部分（conv1d 状态更新 / `in_proj_*` 投影 / norm / `clone()`）从"每步重跑"
改成"批处理一次"，**同时保持逐位相同**（bit-exact，不是"cos 很高"）。

- **单层实测（K=16，真实 checkpoint 权重）**：`spec_forward` 从 ~19.9ms 降到
  ~12.0ms，**1.5-1.75×**（同进程 old-vs-new 对比，跨次运行有 GPU 热状态带来
  的方差，见下）。
- **正确性**：`scripts/b3_gdn_batching_before_after.py`（21/21 项通过，
  old-vs-new 所有 snapshot/输出 `max_abs_diff=0.0`）+
  `scripts/b3_probe_gdn_spec_rollback.py`（16/16 项通过，rollback-vs-顺序解码
  `max_abs_diff=0.0`）。
- **但没有达到"批掉全部非递归部分"的原计划**：`in_proj_qkv`/`in_proj_z`/
  `out_proj` 这三个大投影**批处理后不再逐位相同**——这是本次会话最重要的
  发现，细节见下一节。这三个仍然留在 `for t in range(seq_len)` 循环里，
  是本次优化**没能消除**的成本。
- **端到端**：没有重新跑全模型 e2e 基准（硬约束，见下），只给出一个有明确
  假设、可复现算式的**估算**（不是第二次真实测量）：prose 从 0.655× 提到
  约 **0.78×**，code 从 0.608× 提到约 **0.70×**——**仍然 < 1.0×**，MTP 投机
  在当前 eager 实现下仍然是净亏，只是亏得没那么多了。

## 硬性发现：`torch.bmm` 能保 bit-exact，但只在小输出维度下

**背景**：原计划是把 `in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b`/
`out_proj` 这五个投影全部从"每步一次 `F.linear`"改成"一次 `F.linear` 处理
全部 K 个候选位置"。第一次尝试就直接翻车：

```
in_proj_qkv: batched-vs-perpos max_abs_diff = 0.00195312  bit_exact=False
in_proj_z:   batched-vs-perpos max_abs_diff = 0.00195312  bit_exact=False
in_proj_a:   batched-vs-perpos max_abs_diff = 0.00195312  bit_exact=False
in_proj_b:   batched-vs-perpos max_abs_diff = 0.000976562 bit_exact=False
```

（`conv1d`同一测试下 `max_abs_diff=0`，`bit_exact=True`——conv1d 本身没有
这个问题，见下一节。）

**根因**：BF16 GEMM（`F.linear`/`torch.matmul` 广播/`torch.einsum`，三者
全部测过，结果一致）的浮点归约顺序依赖行数 M——把 K 个候选位置一次算完
（M=K 的大 GEMM）和逐个算（K 次 M=1 的 GEMV）在 cuBLAS 上不是同一条代码
路径，会有 ~1-2 ULP 的舍入差异。这不是这个 checkpoint 的 FP8/NVFP4 反量化
特有的问题——用未量化的合成 BF16 权重复现了同样的现象。

**修复尝试与结果**：`torch.bmm` 加显式 batch 维度（每个候选位置一个独立
的 `[1,in]@[in,out]`，权重用 `.expand()` 广播、不复制）**在小输出维度下确
实逐位相同**：

```
=== default settings ===
small  (hidden=5120, out=16):    bmm bit_exact=True   max_abs_diff=0
medium (hidden=5120, out=128):   bmm bit_exact=True   max_abs_diff=0
medium2(hidden=5120, out=512):   bmm bit_exact=True   max_abs_diff=0
medium3(hidden=5120, out=2048):  bmm bit_exact=False  max_abs_diff=0.000976562
large  (hidden=5120, out=12288): bmm bit_exact=False  max_abs_diff=0.00195312
out_proj-ish(hidden=2048,out=5120): bmm bit_exact=False max_abs_diff=0.000976562
```

**断点在输出维度 512 到 2048 之间**，与量化算法无关（合成未量化 BF16 权重
复现），与 `torch.use_deterministic_algorithms(True)` 无关（试过，断点不变）。
真实 checkpoint 的各投影输出维度（Qwen3.6-27B-NVFP4，`text_config`）：

| 投影 | 输出维度 | 是否 ≤512（bmm 安全区）|
|---|---:|---|
| `in_proj_a`/`in_proj_b` | `num_v_heads` = 48 | ✅ 安全 |
| `in_proj_z` | `value_dim` = 128×48 = 6144 | ❌ 不安全 |
| `in_proj_qkv` | `conv_dim` = 2×2048+6144 = 10240 | ❌ 不安全 |
| `out_proj` | `hidden_size` = 5120 | ❌ 不安全 |

于是最终能安全批处理的只有 `in_proj_a`/`in_proj_b`（通过 `_bmm_project`，
`runtime/model/qwen36_model.py` 新增的模块级函数）——`in_proj_qkv`/
`in_proj_z`/`out_proj` 三个大投影，无论是普通批处理还是 `bmm` 技巧，都测出
了非零 diff，只能留在原来的逐步循环里，**没能批掉这部分成本**。

## 实际批掉的部分

1. **conv1d 状态更新**：一次 `F.conv1d(padding=0)` 覆盖全部 K 个新位置
   （复用 `forward()` 自己 `seq_len>1` 分支已经在用的窗口构造），每个中间
   `conv_state` snapshot 靠对同一张拼接张量切片拿到，不需要重新计算——
   实测逐位相同（因果卷积的滑动窗口本身不依赖总输入长度，这条对，测过）。
2. **`in_proj_a`/`in_proj_b`**：通过 `_bmm_project`（`torch.bmm` + 显式
   batch 维度），一次调用覆盖全部 K 个位置——实测逐位相同。
3. **`Qwen36RMSNormGated`（norm）**：批处理后一次调用——实测逐位相同（每
   行的 mean/rsqrt 归约只在 `head_v_dim` 这一个固定小维度上做，不跨行，
   没有 GEMM 那种行数依赖的归约顺序问题）。
4. **`clone()` 数量**：conv_state 的 K+1 个 snapshot 从"每步 clone 两次"
   改成对同一张拼接张量切片再 clone 一次；recurrent_state 从"每步
   `copy_()` 到已有 buffer 再额外 `clone()`"改成"每步分配一个新 buffer
   直接作为 snapshot"，省掉了一半的 clone。

`in_proj_qkv`/`in_proj_z`/`out_proj` 加上递归本身（`fused_recurrent_
gated_delta_rule`，K 次顺序调用，任务原本就说明这部分是 sparkinfer 范畴、
不是这次的任务）——这四类调用仍然留在 `for t in range(seq_len)` 循环里。

## 单层实测：优化前后

对齐 B3 报告"12.6ms vs 1.8ms，6.9×"的口径（单层、eager、K=16，同一段真实
checkpoint 权重、同一 anchor state），用同进程 old-vs-new 对比
（`scripts/b3_gdn_batching_before_after.py`，old 代码取自这次会话开始前的
`main` 分支）：

| 指标 | OLD（会话开始时） | NEW（本次优化后） |
|---|---:|---:|
| `spec_forward`，K=16 | 19.9ms（多次运行 15.4-20.6ms，GPU 热状态方差）| 12.0ms（多次运行 9.9-14.5ms）|
| vs `chunk_gated_delta_rule` | 8.6-11.5× 慢 | 5.5-7.4× 慢 |
| `spec_forward` / OLD `spec_forward` | — | **1.5-1.75×** 加速 |

注：任务原文给出的基线"12.6ms vs 1.8ms，6.9×"是上一轮会话测的；这次同一
份未改代码在这台共享 GPU 上重新测得 15.4-20.6ms（隔离进程复测：main 分支
未改的 `spec_forward` 单独测得 17.9ms，chunk 1.63ms，比值 10.99×）——绝对
毫秒数因 GPU 热状态/时钟频率有 15-20% 量级的运行间方差是这台卡上的已知事实
（同一 session 内三次测 OLD 都在 15-21ms 区间），但**同进程内的相对加速比
（1.5-1.75×）和"vs chunk 的倍数下降"（从 ~10× 降到 ~6×）这两个结论在多次
重复测量下是稳定的**。

K=8（对齐 e2e 报告的 K，见下一节）：OLD 6.975ms → NEW 4.087ms，同样
**1.71×** 加速。

## 端到端估算（不是重新测量——硬约束不让加载全模型）

这台机器的硬事实（`notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`）：
Qwen3.6 一次完整前向后常驻 19GB→54GB+，且没有显存旋钮管得住。本次会话被
明确要求"用单层探针，别加载整模型"，所以**没有**重新跑
`scripts/b3_mtp_e2e_acceptance_throughput.py` 那样的全模型 e2e 基准。

以下是一个**有清楚假设、可复现算式**的估算，不是第二次真实测量：

**假设**（每条都有理由，不是拍脑袋）：
1. GDN 48 层的 `spec_forward` 总成本 ≈ 48 × 单层实测值——线性外推，层间
   结构相同（权重不同但形状/算法相同），kernel 启动开销这类成本本身就是
   "每层一份"，线性外推合理。
2. 每轮 verify 里"非 GDN 部分"（起草头链式调用、16 层全注意力、MLP、
   采样/accept-reject 的 Python 开销）本次完全没碰，成本不变——这是直接
   的代码事实（本次 diff 只碰了 `runtime/model/qwen36_model.py` 里
   `spec_forward`/`_bmm_project`），不是假设。
3. **接受/拒绝的轨迹（每轮接受几个 token、总轮数）不变**——这不是假设，
   是本次 bit-exact 验证的直接推论：`spec_forward` 的输出值逐位不变，
   贪心 accept/reject 是输出值的确定性函数，值不变则判定不变。

**算式**（`benchmarks/fixtures/qwen36_mtp_e2e_20260802.json` 提供
`wall_s`/`rounds`/`tokens_per_sec`，本次测量提供单层 K=8 的 OLD/NEW 数字）：

```
old_gdn_per_round = 48 × OLD_spec_forward(K=8) = 48 × 6.975ms = 334.8ms
new_gdn_per_round = 48 × NEW_spec_forward(K=8) = 48 × 4.087ms = 196.2ms
non_gdn_per_round = (wall_s_spec × 1000 / rounds) − old_gdn_per_round   # 用假设2/3反推
new_round_ms       = non_gdn_per_round + new_gdn_per_round
new_wall_s         = new_round_ms × rounds / 1000
new_tokens_per_sec = n_tokens / new_wall_s
new_speedup        = new_tokens_per_sec / nonspec_tokens_per_sec
```

| prompt | 轮数 | 原 speedup | non-GDN/轮（反推）| **估算新 speedup** |
|---|---:|---:|---:|---:|
| prose | 14 | 0.655× | 505.5ms | **≈0.784×** |
| code | 10 | 0.608× | 685.2ms | **≈0.704×** |

**仍然 < 1.0×**——即便把 GDN 层里能批的都批了，MTP 投机在当前 eager 实现
下仍然比顺序解码慢（prose 从慢 34.5% 收窄到约慢 21.6%，code 从慢 39.2%
收窄到约慢 29.6%）。这个估算本身指向一个更硬的结论：**剩下的成本
（`in_proj_qkv`/`in_proj_z`/`out_proj` 三个大投影的逐位串行调用 + 递归本身
的 K 次顺序 kernel 启动）是决定性的**，而这两类成本——大投影的批处理受限
于 cuBLAS 的行数相关舍入行为，递归的顺序依赖是数学上的硬约束——都不是
runtime 侧能绕开的，前者理论上可能需要 sparkinfer/cuBLAS 层面的多行
GEMV-batch 原语，后者任务原文已经界定为 sparkinfer 范畴。

## 我没能验证的东西

1. **没有重新跑全模型 e2e 基准**——上面的"估算新 speedup"是算出来的，不是
   测出来的第二个真实数字。硬约束（这台机器的显存事实）不允许在本次会话
   里加载全模型验证。如果要坐实，下一步是真的跑一次
   `scripts/b3_mtp_e2e_acceptance_throughput.py`（换成这次优化后的代码），
   单独申请一次全模型 GPU 窗口。
2. **没有验证"非 GDN 部分成本不变"这个假设本身**——虽然从代码 diff 看
   （只碰了 GDN 层这一个文件的一个方法）这个假设几乎必然成立，但"用了多
   少 Python 端额外开销/CUDA context 影响"这类二阶效应没有实测排除。
3. **`_bmm_project` 的 bit-exact 断点（512-2048 之间）没有精确定界**——
   只测了几个离散点（16/128/512/2048/12288/5120 out），没有二分查找到
   精确的分界值；这个数字本身也很可能是 cuBLAS 版本/GPU 型号相关的，不
   是这个 runtime 能长期依赖的稳定常量（如果 PyTorch/cuBLAS 升级，断点
   可能变化——这是 `_bmm_project` 只对 `in_proj_a`/`in_proj_b` 这两个"小
   到几乎不会踩到断点"的投影使用它、而不是对所有投影都尝试它的原因之一）。
4. **只在 K=16 和 K=8、单层（layer 0）、单一随机种子（1234）下验证**——
   没有跑遍全部 48 个 GDN 层（虽然它们结构相同，只是权重不同，风险较低，
   但"权重相关的地方"[原文语境]严格说没有逐层验证过）。

## 相关

- `runtime/model/qwen36_model.py`：`Qwen36GatedDeltaNet.spec_forward`（改动
  本体）、`_bmm_project`（新增，bit-exact 批处理投影的机制）
- `scripts/b3_gdn_batching_before_after.py`：本次新增，old-vs-new 同进程
  对比（正确性 + 吞吐），支持命令行传 K（默认 16，用 `8` 对齐 e2e 报告）
- `scripts/b3_probe_gdn_spec_rollback.py`：沿用原有探针（只改了 `_ROOT`
  指向这次的 worktree），复验 rollback-vs-顺序解码逐位相同
- `notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md`：本次估算依据的
  原始 e2e 测量（未改动的旧代码）
- `notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`：为什么这次没有
  重新跑全模型 e2e
