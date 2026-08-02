# 解码 kernel profiling：CG 下已是 kernel-bound，且 ¼ 的 kernel 时间跑在 Ampere 代 kernel 上

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）· 单卡 RTX PRO 6000
Blackwell Max-Q（cc 12.0）· `torch.profiler`，叶子 kernel 计时（`cat=="kernel"`），
20 步解码，`/tmp/gpu_lock.sh` 下独占

## 一句话

**CG 解码已经基本压到 kernel 上（GPU 89% 忙），继续提速必须让 kernel 本身更快；
而 kernel 时间里 24.8% 跑的是 `cutlass_80_wmma_tensorop`——为 SM80 编译的 Ampere 代
kernel，在一张 SM120 的卡上。**

## 时间去哪了

| | ms/step |
|---|---:|
| 叶子 GPU kernel 时间 | **30.73** |
| GPU memcpy / memset | 0.28 |
| **GPU busy 合计** | **31.01** |
| **CG 墙钟**（服务路径实测 28.848 tok/s） | **34.67** |
| eager 墙钟（本次 profiling 同一进程实测） | **150.24** |

两个结论直接落下来：

1. **CG 下 GPU busy / 墙钟 = 31.01 / 34.67 = 89%。已经是 kernel-bound。**
   再想快，只能让 kernel 更快，或者少做 kernel——不是调度问题了。
2. **eager 下 GPU 只有 31.01 / 150.24 = 21% 忙，79% 在等。**
   这就是 [CG vs eager 4.71×](2026-08-03-cudagraph-vs-eager-decode-throughput.md)
   的机制：CG 消掉的不是计算，是 CPU 侧的每步重建与启动间隙。

CPU 侧最大的一块是 paged attention 的元数据规划（**不计入上面的 kernel 总和**，
它们是 CPU 标注区间）：

| CPU 标注 | ms/step |
|---|---:|
| `paged_workspace.plan_metadata_to_device` | **24.28** |
| `paged_workspace.plan_metadata_scalar_updates` | 6.85 |
| `paged_workspace.plan_metadata_copy_buffers` | 1.46 |
| `paged_workspace.plan_metadata_zero_buffers` | 0.97 |
| `paged_workspace.runtime_copy_page_table` | 0.64 |

合计约 34.2 ms/step 的 CPU 侧元数据工作——在 eager 下它就是那 119ms 空转的主因。
CG 把它固化进图里，所以生产路径不付这笔；**但任何绕过 CG 的路径都会重新付**
（例如今天的 MTP 测量脚本，见
[`2026-08-03-mtp-acceptance-on-standard-checkpoint.md`](2026-08-03-mtp-acceptance-on-standard-checkpoint.md)）。

## kernel 构成（30.73 ms/step）

| ms/step | 占比 | kernel |
|---:|---:|---|
| 10.752 | **35.0%** | `W4A16FusedMoeKernel`（sparkinfer，NVFP4 打包直算） |
| 4.645 | 15.1% | `cutlass_80_wmma_tensorop_s161616gemm_bf16_32x32_128x1` |
| 4.347 | 14.1% | cuBLAS `gemvx` bf16 |
| 1.880 | 6.1% | cuBLAS `gemvx` bf16（第二形状） |
| 1.619 | 5.3% | `cutlass_80_wmma_tensorop_bf16_..._16x16_64x1` |
| 1.338 | 4.4% | `cutlass_80_wmma_tensorop_bf16_..._16x16_128x2` |
| 1.153 | 3.8% | elementwise copy |
| 0.187 | **0.6%** | `fused_recurrent_gated_delta_rule_fwd_kernel`（GDN 递归） |

### 两条可以直接行动的

**① 24.8% 的 kernel 时间跑在 SM80 kernel 上（7.602 ms/step，三个 `cutlass_80_wmma`）。**
名字里的 `80` 是编译目标 compute capability。这张卡是 cc 12.0。
**阶段四"压榨 SM120"第一次有了具体靶子**：这 7.6ms 是 BF16 GEMM/GEMV 路径，
即非 NVFP4 层反量化后走的 `F.linear`。加上两个 `gemvx`（6.23ms），
**BF16 通用路径合计 13.8 ms/step = 45%**，比 NVFP4 专用 kernel 那 35% 还多。

**② GDN 递归 kernel 只占 0.6%。**
这独立证实了 roadmap 里"硬上限已排除 GDN 是 MTP 的决定项"那条判断——
在解码步上 GDN 递归根本不是量级项。**别再往那儿投优化。**

## 方法与自我纠正

第一版把 `torch.profiler.key_averages()` 直接求和，得出 88.99 ms/step 和 −157% 的
gap。**那是错的**：`key_averages()` 把算子层与它派发的 kernel 层混在一起，
`sparkinfer::w4a16_fused_moe_launch` 和它的 cutlass kernel 都报 10.935 ms，
同一份工作计了两次。本文改从导出的 chrome trace 里只累加 `cat=="kernel"` 的事件，
每微秒 GPU 时间只归属一次；CPU 标注单列、不进 kernel 总和。

## 归属：按调用次数精确对上了

用同一份 trace 数每步的 kernel 调用次数，与 checkpoint 声明的层构成对照：

| kernel 组 | 次/step | ms/step | 对应什么 |
|---|---:|---:|---|
| `W4A16FusedMoeKernel` | **56.0** | 10.752 | **0–55 层的 MLP**（NVFP4，每层一次融合调用） |
| `cutlass_80_wmma` ×3 | 129.0 | 7.603 | ┐ |
| cuBLAS `gemvx` ×2 | 104.0 | 6.227 | ┘ 合计 **233**，全部是 FP8 层反量化后的 `F.linear` |

FP8 线性层的预期数量（读自 `config_groups.group_0` 的 targets 与 `layer_types`）：

```
full_attention q/k/v/o : 16 × 4 = 64
GDN 三个投影            : 48 × 3 = 144
lm_head                :          1
56–63 层 MLP           :  8 × 3 = 24
                         合计     233
```

**233 预期 = 233 实测，一个不差。** NVFP4 侧 56 预期 = 56 实测。
归属没有猜测成分：**每一次 BF16 通用 GEMM/GEMV 调用，都对应一个被反量化成 BF16 的
FP8 层**；cuBLAS 按形状把它们分给 `gemvx`（104 次）和 SM80 WMMA cutlass（129 次）。

## 阶段四的两根杠杆（合计 80% 的 kernel 时间）

**① FP8 层 45%（13.83 ms/step，233 次调用）—— 现在是反量化成 BF16 再算。**
接上 FP8 W8A8 kernel 可以让它们全程保持量化。此前有意推迟（单层 cosine 0.9996，
比 NVFP4 路径差 30–40×，需要专门一轮全模型验证），见
[`2026-08-03-nvfp4-raw-param-free-and-fp8-w8a8-probe.md`](2026-08-03-nvfp4-raw-param-free-and-fp8-w8a8-probe.md)。

**② NVFP4 层 35%（10.75 ms/step，56 次调用）—— 现在走 W4A16。**
标准 checkpoint 的 `group_1` 声明的是 **W4A4** 且 `input_global_scale` 实际发货，
所以 `sparkinfer.gemm.blockscaled` 路径是开着的。⚠️ 此前那次 `blockscaled.mm` 失败
是在 `nvidia/` checkpoint 上（那里 `input_activations: None`，动态量化激活没有
对应物）——**那个否决不适用于标准模型**，详见 `docs/roadmap.md` 阶段 4。

**GDN 不在名单上**：0.187 ms/step、0.6%。

## 未决

- 上面的 kernel 归属是在 **eager** 下采的（kernel 时长与是否被图重放无关，
  但 kernel **序列**在 CG 下可能不同）。若要精确到生产，需在捕获图上用 nsys 采。
- 两根杠杆动手前都必须先过 B1-R 的 gap-error 判据——上次 `blockscaled.mm` 就是栽在那里。
- 带宽 roofline：28.85 tok/s ≈ 582 GB/s 有效带宽，卡的峰值在 1.8 TB/s 量级。
  但既然已 kernel-bound，下一步应是看这些具体 kernel 各自离自己的 roofline 多远，
  而不是看整体比值。
