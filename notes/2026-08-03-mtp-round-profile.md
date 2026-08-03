# MTP 一轮的实测分解：**GPU 只有 31% 忙,瓶颈是主机侧拷贝**

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）· K=4 · 短上下文
· 单卡 RTX PRO 6000 Blackwell Max-Q

## 为什么测这个

MTP 开着比关着慢 **0.42×**（11.6 vs 27.4 tok/s），即使 anchor 与 draft 循环的
CUDA Graph 已经落地。此前对"剩下的时间去哪了"的说法一律是推理——
"almost certainly goes to `verify_forward`"。**本文是实测。**

## 阶段耗时（CUDA event,逐项单独测）

| 阶段 | ms |
|---|---:|
| **一整轮 MTP** | **276.8** |
| **`verify_forward`(K=4)** | **192.8 —— 占 70%** |
| 单次 plain decode(M=1,eager) | 160.8 |

⚠️ **这里最该注意的一条,和"MTP 不划算"的直觉相反：
verify 4 个 token 只花单次 decode 的 1.2 倍（192.8 vs 160.8）。**
一次 verify 抵 4 次 decode——**投机的前提在 GPU 侧是成立的。**

## 真正的瓶颈：一轮里 GPU 有 69% 的时间是闲的

```
leaf GPU kernel: 87.03 ms/round
墙钟             276.84 ms
→ 只有 31% 忙
```

CPU 侧标注（**不计入 kernel 总和**，`aten::to`/`_to_copy`/`copy_` 是同一条嵌套链）：

| CPU 标注 | ms/round |
|---|---:|
| `aten::copy_` | **113.3** |
| `aten::to` | 105.7 |
| `aten::_to_copy` | 104.6 |
| `sparkinfer::w4a16_fused_moe_launch` | 34.1 |
| `paged_workspace.plan_metadata_to_device` | 26.7 |
| `aten::linear` / `aten::matmul` | 23.8 / 21.5 |
| `FusedRecurrentFunction` | 14.9 |

**一轮约 113 ms 花在张量拷贝/dtype 转换上——占墙钟 41%,不是计算。**

kernel 侧构成（87.03 ms）没有异常：`W4A16FusedMoeKernel` 24.8ms/28.4%、
cuBLAS `gemvx` 16.7ms/19.2%、`cutlass_80_wmma` 15.0ms/17.2%——与短上下文
decode 的构成一致（[`2026-08-03-decode-kernel-profile.md`](2026-08-03-decode-kernel-profile.md)）。
**没有哪个 kernel 变慢,是它们之间的空隙太大。**

## 结论：不是接受率,是两处没进图

MTP 关闭时,decode **走捕获的图**,36.5 ms/token（27.4 tok/s）。
MTP 开启时,anchor 与 draft 步**已进图**（本日落地）,
但 **`verify_forward` 没有**——它是一轮里最贵的一步（70%）,
每轮原样再付一次 CUDA Graph 本该消掉的那份主机侧开销。

**接受率不是主因。** 就算接受率从 1.54 翻倍到 3.0,一轮仍要付这 190 ms 的 GPU 空转。

那 113 ms 的 `copy_` 指向历史测绘的 **M-3**
（[`2026-08-03-historical-implementation-survey.md`](2026-08-03-historical-implementation-survey.md)）：

> 今天用 **snapshot/restore**：`spec_forward` 逐位置克隆 **K+1 份 GDN 快照**（48 个 GDN 层）,
> `commit_spec_snapshot` 按 m 选一份 `copy_` 回去。
> 历史用 `_ssm_spec_row` 的 **K+1 行寻址,零回滚**——当年这个改动删掉了
> **命中 84.4% 轮次、占约 56% 墙钟**的重算,值 **+18.76%**。
> **`runtime/recurrent_state_pool.py::spec_row` 今天实现了,零调用。**

## 下一步（按实测排序,不按猜测）

1. **捕获 `verify_forward`**——一轮 70%。两个已定位的阻塞点：
   `Qwen36AttentionWorkspace` 硬编码 `enable_cuda_graph=False`
   （`runtime/model/qwen36_model.py:1261`）,而 sparkinfer **有**
   `prepare_prefill_graph_replay_state`/`update_prefill_graph_replay_metadata`
   （`sparkinfer/attention/paged/workspace.py:1325`,Laguna 的 verify 图正在用）；
   以及 GDN `spec_forward` 的图捕获安全性未验证。
2. **用 `spec_row` 的 K+1 行寻址替掉 GDN snapshot/restore**——直接冲着那 113 ms 去。

## 方法

`torch.profiler` 导出 chrome trace,**只累加 `cat=="kernel"` 的叶子事件**。
不要对 `key_averages()` 求和——它把算子层与其派发的 kernel 层重复计数,
今天早些时候那样做得出过 88.99 ms/step 与 −157% 空闲率,是无意义的。
CPU 标注单列,不进 kernel 总和。
