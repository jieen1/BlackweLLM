# Qwen3.8 Q6：transient prefill dequantization 实测

> 日期：2026-08-21
> 状态：🟢 已接入 Q6+DFlash2 server 默认；resident BF16 与全 packed 仍可回退

## 结论

Q6+DFlash2 的服务默认现在保持 GGUF packed 权重，只对真正的 prefill batch
（`M >= 32`）逐 projection 临时 dequantize 成 BF16，立即用 cuBLAS
`F.linear` 后释放。DFlash2 的 `M=8` eager warmup、CUDA Graph capture 和
replay 永远走 packed TC，避免了之前把 verify warmup 误判成 prefill 导致的
illegal memory access。

## Fresh A/B

固定配置：SM120、torch-nightly、同一 Q6_K_XL、Q6/Q8 split、packed weights、
CUDA Graph、DFlash2 K=7、4K raw completion、32 output tokens、c=1、prefix
cache off、`127.0.0.1:18380`。每次均为 fresh server，完成后停止；8300 和现有
服务没有触碰。

| 配置 | warm decode | warm TTFT | warm wall | 接受/提交 | completion SHA |
|---|---:|---:|---:|---:|---|
| packed TC，transient off | 89.96 / 89.29 tok/s | 3.7731 / 3.7760 s | 4.1207 / 4.1261 s | 28 / 31 | `34850d3f...c28e85` |
| transient，`M>=32` | 88.73 / 86.72 tok/s | 1.1628 / 1.1631 s | 1.5152 / 1.5235 s | 28 / 31 | `34850d3f...c28e85` |

跨两次 warm 的均值：TTFT 约下降 **68.9%**（3.7755 → 1.16295 s），请求
wall 约下降 **63.1%**（4.1234 → 1.51935 s）；decode 在测量噪声内持平，
没有拿 prefill 优化冒充 decode 加速。

服务默认复测（不显式设置 transient/dequant 环境）同样落到该路径：warm
TTFT `1.1953 / 1.1986 s`、wall `1.5665 / 1.5615 s`，Graph 全部 captured，
接受 histogram 为 4 个完整 K=7 round，输出 SHA 不变。

## 安全边界

- `M < 32` 直接返回 `None`，所以 DFlash2 verify 不会在 capture 前 side-stream
  warmup 中走 cuBLAS/transient 分支。
- graph capture 时 `torch.cuda.is_current_stream_capturing()` 仍强制 packed。
- 单 projection transient workspace 默认上限为 512 MiB；词表 `Q8_0`
  LM head 超过上限，继续使用 packed kernel。
- `QSR_GGUF_NATIVE_PREFILL_DEQUANT=0` 回到全 packed；
  `QSR_GGUF_DEQUANTIZE_WEIGHTS=1` 回到 resident BF16。

## 产物

- `/tmp/qwen38_q6_transient_guard32_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_q6dflash2_defaults_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_q8lm_tc_4k32_perf_20260821.json`（packed baseline）
