# Qwen3.8 Q6 packed TC：参考 SGLang 的按形状 M-tile 优化

> 日期：2026-08-21
> 状态：🟢 已通过 microbenchmark 和隔离端到端 A/B；默认 packed TC 路径采用 `auto`

## 背景与参考

本轮不是把 Q6_K_XL 改成 SGLang 的标准 `block_q6_K`。目标文件仍使用 runtime
自己的 `Q6_K_SPLIT` 行尾布局：Q6 block 的 code/scale 部分为 208 bytes，FP16
`d` 值集中在每行尾部，行 stride 为每 block 210 bytes。

参考的 SGLang 结构来自本地源码：

- `/home/bot/project/sglang/sgl-kernel/csrc/quantization/gguf/mmq.cuh`
- `/home/bot/project/sglang/sgl-kernel/csrc/quantization/gguf/mmvq.cuh`
- `/home/bot/project/sglang/sgl-kernel/csrc/quantization/gguf/vecdotq.cuh`
- `/home/bot/project/sglang/python/sglang/srt/layers/quantization/gguf.py`

SGLang 的可复用轴是批量行共享一份权重 tile。原来的 Triton packed TC kernel
固定 `BLOCK_M=8`，因此 4K prefill 会对每 8 行重新解码同一批 Q6 权重。此次
没有重写 Q6 解码契约，只将 M tile 做成 shape-aware：

```text
QSR_GGUF_TC_BLOCK_M=auto（默认）
  Q5_K, M >= 32 -> BLOCK_M=64
  Q6_K, M >= 64 -> BLOCK_M=64
  其它 M >= 32  -> BLOCK_M=32
  M < 32         -> BLOCK_M=8  # DFlash2 M=8 verify / 小批量
```

显式 `8/16/32` 仍可用于 A/B。`BLOCK_N=32`、4 warps、stages=1 保持不变；
`BLOCK_N=64/128` 没有通过门禁；Q6 M=8 的 auto stages=2 在 fresh A/B 中保留，
因为它相对 stages=1 只有低个位数收益但输出和接受率完全一致。

## Microbenchmark

形状为 `Q6_K_SPLIT, N=17408, K=5120`，同一输入和 packed payload，SM120：

| M | 固定 `BLOCK_M=8` | `auto` | 变化 |
|---:|---:|---:|---:|
| 8 | 0.1175 ms | 0.1164 ms | 基本不变 |
| 32 | 0.4334 ms | 0.1185 ms | 约 3.7× |
| 4096 | 61.9696 ms | 16.4729 ms | 约 3.8× |

同一输入下 M=8/16/32 的 `BLOCK_M=8/16/32` 输出完全一致；另有已有的
Q6 packed tensor-core reference gate（cosine 约 0.9999966，误差在既有容差内）。

## 隔离端到端 A/B

两次服务均使用 `127.0.0.1:18380`，完成后停止；没有重启或修改现有服务。
配置固定为同一 torch-nightly venv、Q6 checkpoint、tokenizer、4K 数字 filler、
c=1、32 output tokens、prefix cache off、DFlash2 K=7、CUDA Graph、Q6/Q8
split、packed weights。HTTP 基准使用现有
`benchmarks/server_perf_grid.py`，cold + 2 warm。

| 配置 | warm decode 1 | warm decode 2 | decode 均值 | TTFT 均值 | wall 均值 |
|---|---:|---:|---:|---:|---:|
| 旧固定 M=8 | 77.54 | 77.39 | **77.465 tok/s** | **15.3378 s** | **15.7410 s** |
| 强制 M=32（否决） | 67.83 | 68.47 | **68.150 tok/s** | **5.5802 s** | **6.0383 s** |
| `auto`：M=8/32 | 84.75 | 84.46 | **84.605 tok/s** | **5.4442 s** | **5.8138 s** |

相对旧固定 M=8，`auto` 的 warm TTFT 降低约 **64.5%**，decode 提升约
**9.2%**，请求级吞吐提升约 **171%**（2.03 → 5.505 tok/s）。强制 M=32
虽然预填充更快，但会压低 DFlash2 verify/decode，故没有采用全局 M=32。

质量/图门禁保持不变：两种配置 completion SHA 都是
`34850d3f903fd71918a3db8ba1dd257b20f56b3e49405b4893ed100322c28e85`，
DFlash2 为 28 accepted / 31 committed，draft、fixed verify、ragged verify
和 target decode CUDA Graph 均 captured。

产物：

- `/tmp/qwen38_q6_q5mmq_baseline_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_tc_bm32_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_tc_bmauto_4k32_perf_20260821.json`

## 实施边界

这项改动只改变 packed Q6/Q5/Q8 tensor-core decoder 的 M tile 选择；resident
BF16 路径不受影响，M=1 native GEMV 也不受影响，现有 NVFP4 服务不受影响。
它借鉴的是 SGLang 的 tile reuse 原则，而不是直接复制 SGLang 的 MMQ/MMVQ
ABI；此前的 SGLang-style Q6 MMQ 仍是单独的 M=8 opt-in 实验，fresh A/B 只有
低个位数收益，不能替代这个更高 ROI 的 prefill 复用优化。
