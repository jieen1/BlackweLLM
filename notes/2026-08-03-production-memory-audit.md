# 生产配置下的显存审计（标准模型，注明配置与日期）

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）·
单卡 RTX PRO 6000 Blackwell Max-Q，97,887 MiB

回应 `docs/roadmap.md` Track F/F2-0 那条：**此前的两组数字（"94.2/97.9 GB, 98.8%"
与 2026-07-29 静态审计的 "76.0/95.6 GB, 79.5%"）配置不同、未交叉验证。**
本文给的是**当前标准模型**的实测，配置逐项写明。

## 配置（这是关键，不写清楚的数字没有意义）

```
QSR_SERVER_MODEL_PATH   = unsloth/Qwen3.6-27B-NVFP4
QSR_SERVER_CAPACITY     = 1        # 单并发
QSR_SERVER_NUM_SLOTS    = 2        # capacity + 1（CG warmup 槽）
QSR_SERVER_ENABLE_DFLASH= 0        # qwen36 backend 不支持
max_context             = 131072 tokens/slot（默认）
```

外部 `nvidia-smi` 采样，不是 `torch.cuda` 计数器。

## 实测常驻

| 配置 | 常驻 |
|---|---:|
| capacity=1, CG **on** | **72.39 GiB** |
| capacity=1, CG off（eager） | 77.69 GiB |
| capacity=2, num_slots=3, CG on | ~82 GiB（并发扫描中观测） |
| capacity=4, num_slots=5, CG on | 未 OOM（96 GiB 内跑完） |

⚠️ **CG 比 eager 少用 5.30 GiB**，不是多用。

## 构成：那 72 GiB 是什么

服务端启动日志直接给出的：

| 项 | 每槽 | ×2 槽 |
|---|---:|---:|
| KV cache | 8192 MiB | **16.00 GiB** |
| recurrent state（GDN） | 75.8 MiB | 0.15 GiB |

权重侧按 checkpoint 真实张量尺寸算（读 safetensors 元数据，不是估算）：

| 项 | GiB |
|---|---:|
| FP8 层原件（237 个 `.weight`，10.73B 参数 @ 1 B） | 9.99 |
| **FP8 层的 BF16 反量化缓存（10.73B @ 2 B）** | **19.99** |
| NVFP4 层打包权重（168 个 `weight_packed`，7.49B 字节） | ~6.98 |

合计约 **53 GiB 权重侧 + 16.15 GiB KV/recurrent ≈ 69 GiB**，与实测 72.39 GiB
在采样与分配器开销的范围内对得上。

## 主要发现：反量化缓存只解决了一半

`free_nvfp4_raw_params()`（`runtime/model/qwen36_model.py`）在融合权重备好后释放
NVFP4 原始参数，把常驻从 76.34 压到 53.08 GiB。**但那只覆盖了 NVFP4 的 56 层 MLP。**

**FP8 那一侧没有对应处理**：`CompressedTensorsFP8ChannelLinear._ensure_ready()`
把 `_weight_bf16` **永久缓存**，而 `self.weight`（FP8 原件）**同时留在显存里**：

```python
def _ensure_ready(self) -> None:
    if self._weight_bf16 is None:
        self._weight_bf16 = dequantize_fp8_channel(self.weight.data, self.weight_scale.data)

def forward(self, x):
    self._ensure_ready()
    ...                       # 只用 _weight_bf16，不再碰 self.weight
```

于是这 237 个张量**两份同时常驻 29.98 GiB**，而 `forward` 只读其中的 BF16 那份。

**可回收约 9.99 GiB，且不需要任何 kernel 工作**——照搬 NVFP4 的做法，在
`_ensure_ready()` 之后释放 FP8 原件即可。

⚠️ **但先别急着做**：FP8 W8A8 的判据预演正在进行（阶段四杠杆①）。
若 W8A8 可用，就需要**保留 FP8 原件**并干脆不建 BF16 缓存——那是更大的一笔
（省掉整整 19.99 GiB，还顺带消掉 45% 的 BF16 GEMM kernel 时间）。
**等预演结论出来再定做哪一种**，否则可能刚释放完又要加回来。

## 顺带

- 237 是匹配 group_0 命名的 `.weight` 张量数；解码 profiling 里数到的是
  **233 次/步 kernel 调用**（[`2026-08-03-decode-kernel-profile.md`](2026-08-03-decode-kernel-profile.md)）。
  两者不必相等——每步都调用的层数与 checkpoint 里的张量数不是同一件事
  （例如 `lm_head` 的调用节奏与 decoder 层不同）。**没有把这 4 个的差异查清楚**，
  在此存疑而不是编一个解释。
- `max_context=131072` 是默认值，KV 因此占 8 GiB/槽。这是**配置选择而非缺陷**，
  但它是 72 GiB 里最大的单项之一，缩短上下文能直接换显存。
