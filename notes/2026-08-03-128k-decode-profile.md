# 128K 解码 profiling：attention 占 58.5%，而 FP8 KV **让它更慢**

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）·
120,000 token 上下文、c=1、**eager**（未开 CG）· 单卡 RTX PRO 6000 Blackwell Max-Q

## 为什么测这个

此前所有关于"今天是不是比历史慢"的讨论，都是拿二手数字在不同上下文长度之间做算术，
**换算了两次、错了两次**（见
[`2026-08-03-performance-gap-vs-historical.md`](2026-08-03-performance-gap-vs-historical.md)）。
历史记录的口径是 128K，那就在 128K 上直接测。

## 结果：长上下文是完全不同的瓶颈区间

```
context=120035   decode 171.50 ms/token = 5.83 tok/s
leaf GPU kernel  76.88 ms/step  (wall 171.50 → 45% busy)
```

| ms/step | 占比 | 调用/步 | kernel |
|---:|---:|---:|---|
| **44.969** | **58.5%** | 16 | **sparkinfer `PagedForwardKernel`（attention）** |
| 11.163 | 14.5% | 56 | `W4A16FusedMoeKernel` |
| 4.856 | 6.3% | 112 | `cutlass_80_wmma` |
| 4.541 | 5.9% | 88 | cuBLAS `gemvx` |
| 0.196 | **0.3%** | 48 | GDN 递归 |

**对照短上下文**（[`2026-08-03-decode-kernel-profile.md`](2026-08-03-decode-kernel-profile.md)）：
那里 attention 几乎不可见、BF16 通用 GEMM 占 45%、NVFP4 融合 MLP 占 35%。
**128K 下 attention 一项就 58.5%。两个区间的优化对象完全不同**，
短上下文的结论不能外推到长上下文，反之亦然。

GDN 在两个区间都是 0.3–0.6%，**再次确认它不是量级项**。

## 🔴 FP8 KV 在长上下文下是**减速**的（与我的预测相反）

预测过：KV 从 BF16 转 FP8 使 attention 读取减半
（16 层 × 2(K+V) × 4 头 × 256 × 2字节 × 120000 ≈ 7.86 GB → 3.93 GB），
因此应当是"长上下文最大的提速杠杆"。**实测把这个预测推翻了：**

| @128K, c=1, eager | BF16 KV | FP8 KV | |
|---|---:|---:|---|
| `PagedForwardKernel` | 44.969 ms | **53.636 ms** | **+19%** |
| leaf kernel 合计 | 76.88 ms | **85.94 ms** | +12% |
| decode 墙钟 | 171.50 ms/tok | 170.40 ms/tok | 持平 |
| GPU busy | 45% | 50% | |

**读的字节减半，kernel 反而慢 19%。** sparkinfer 的 FP8-KV attention 路径要么在 kernel
内部做反量化、要么走了没那么优化的分支，把省下的带宽吃回去还倒贴。

⚠️ **墙钟持平是假象，不能当"没影响"**：这轮是 eager，GPU 只有 45–50% 忙，
一半时间在等 CPU，kernel 变慢被空转掩盖了。
**在 CG 下（已 kernel-bound、GPU 89% 忙）这 19% 会直接变成吞吐损失。**

**结论修正**：
- FP8 KV 是**显存**杠杆（实测省 12.3 GiB，B1-R 每条判据 2–8× 余量），这条不变。
- FP8 KV **不是速度杠杆**；在长上下文下它**损害**速度。
- 短上下文下 KV 读取本就可忽略，所以那里它是速度中性的。
- **翻默认值之前，必须在 CG 下补一轮长上下文吞吐对照**——eager 下测不出来。

## 顺带：单次 forward 的硬上限 ≈ 61,681 token

一次性把 120,000 token 喂进 `model(...)` 会崩：

```
OverflowError: Value overflow: 4177920000 exceeds range of l
  sparkinfer/moe/_shared/kernels/w4a16/kernel.py:8594
```

`m × fc1_cols = 120000 × 34816 = 4.18e9`，超过 int32 上限 2,147,483,647，
所以单次 forward 的 token 数上限是 `2147483647 / 34816 ≈ 61,681`。

**生产不受影响**——`Qwen36Backend.prefill_chunked_begin` 按 512 分块。
但**任何直接调裸模型 API 喂长序列的脚本都会撞上**，本文的第一次运行就是这么崩的。
本测量改为按 8192 分块后正常（prefill 120,000 token 用 86.3s / FP8 KV 下 67.1s）。

## 方法

`torch.profiler` 导出 chrome trace，**只累加 `cat=="kernel"` 的叶子事件**。
不要对 `key_averages()` 求和——它把算子层与其派发的 kernel 层混在一起重复计数，
今天早些时候那样做得出过 88.99 ms/step 和 −157% 的空闲率，是无意义的。
