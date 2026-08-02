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

**已实施**（FP8 W8A8 预演给出否定结论后，保留 FP8 原件已无意义）：
`CompressedTensorsFP8ChannelLinear.free_fp8_raw_weight()` +
`Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights()`，在 `load_qwen36_model`
的 warmup 之后调用。

**真机实测：释放 233 个 Linear，常驻 44,626 → 38,698 MiB，实收 5.79 GiB**，
forward 输出不变。

⚠️ **实收 5.79 GiB 而非预估的 9.99 GiB** —— 分配器把释放出来的块留作复用，
`nvidia-smi` 看到的降幅小于实际丢弃的存储量。NVFP4 那轮是同一现象
（丢了 ~9.15 GiB 存储，常驻只从 67.10 降到 64.58）。**这是预期行为，不是没生效**：
`weight.data.numel()` 确实为 0，节省会体现在后续分配的头寸上。

📌 **顺带解开前面那个存疑**：`free_fp8_raw_weights()` 报告 **233** 个 Linear，
与解码 profiling 数到的 **233 次/步 kernel 调用完全一致**。所以 233 是**模型里
FP8 Linear 模块的数量**，而 237 是我的正则在 checkpoint 里匹配到的张量数——
多出的 4 个没有对应模块。**以 233 为准。**

## 🔴 更大的一笔：标准 checkpoint 其实发了 FP8 KV scale，而我们没用

释放 FP8 原件那次真机运行里，**新加的反向检查在第一次真实运行就报了警**：

```
load_qwen36_model: 2 checkpoint tensor family/families reached no model
parameter or buffer: k_scale x16, v_scale x16
```

`runtime/model/qwen36_model.py` 的模块 docstring 写着"本 checkpoint 声明
`kv_cache_quant_algo: FP8` 但**发货零个 `k_scale`/`v_scale`**（B0-2）"，
并据此决定用 BF16 KV 绕开"该用什么 scale"。**实测这个前提对标准模型不成立**：

| checkpoint | k_scale | v_scale |
|---|---:|---:|
| `nvidia/`（B0-2 当时测的） | 0 | 0 |
| `unsloth/`（**标准模型**） | **16** | **16** |

标准 checkpoint 发的是完整的静态 per-tensor 对称 FP8 KV 方案
（`num_bits=8`、`strategy=tensor`、`symmetric=True`、`observer=static_minmax`），
每个 full_attention 层一份 `k_scale`/`v_scale`。**那个被推迟的问题，checkpoint 自己
给了答案，而我们没有接**——`load_qwen36_model` 不像 Laguna 路径那样调用
`apply_kv_cache_scale_post_load`，所以这 32 个张量一直无人认领。

**这笔比释放 FP8 原件大得多**：KV 是 **8192 MiB/槽**，是本审计里最大的单项，
FP8 KV 直接把它减半。num_slots=2 省约 8 GiB；capacity=4（num_slots=5）省约 20 GiB。
⚠️ BF16 KV 目前是正确的、在跑的；这是**未兑现的机会而不是 bug**，动手前必须过 B1-R。

## 顺带

- `max_context=131072` 是默认值，KV 因此占 8 GiB/槽。这是**配置选择而非缺陷**，
  但它是 72 GiB 里最大的单项之一，缩短上下文能直接换显存。
