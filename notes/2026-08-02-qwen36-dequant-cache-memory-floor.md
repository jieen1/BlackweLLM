# Qwen3.6 的显存底线由反量化缓存决定，不受任何显存旋钮控制

日期：2026-08-02 · 状态：🟢 已核实（两次实测触发）· 未修复

## 结论

`ModelOptFP8Linear` / `ModelOptNVFP4Linear`（`runtime/model/modelopt_linear.py:67-75`）
**惰性把权重反量化成 BF16，并永久缓存**：

```python
self._weight_bf16: torch.Tensor | None = None
...
if self._weight_bf16 is None:
    self._weight_bf16 = dequantize_fp8(self.weight.data, self.weight_scale.data)
return F.linear(x, self._weight_bf16, self.bias)
```

**后果**：一旦一次前向摸遍所有层，常驻占用从 **~19 GB（量化态）涨到 ~54 GB+**。

⚠️ **这个增长与 `blocks_per_slot` / `num_slots` / `GPU_MEM_UTIL` 全部无关。**
那三个旋钮管的是 KV cache 的分配预算，管不到权重缓存。

## 为什么这条值得单独成文

2026-08-02 用户要用一部分显存，我给所有 agent 下发了

```
QSR_SERVER_CAPACITY=1  QSR_SERVER_NUM_SLOTS=1
QSR_SERVER_GPU_MEM_UTIL=0.60  QSR_SERVER_BLOCKS_PER_SLOT=512
```

**这套配置对本问题完全无效**，GPU 仍然两次冲到 ~96.6 / 97.9 GB。
**"降低槽位和显存利用率"这个直觉在 Qwen3.6 上不成立**——它是本轮踩了两次的坑，
而不是一次孤立事故。

## 这是 B1 的一个**有意**设计选择，不是 bug

`modelopt_linear.py` 的文档写明了缓存是刻意的（避免每次前向重复反量化）。
B1 的范围是"eager、batch=1 的正确性"，在那个范围内它是对的。
**没被压力测过的是它在共享显卡上的显存后果。**

## 相关但不同的一条

`scripts/b1_verify_greedy_alignment.py` 曾因同一机制 OOM（详见该文件 docstring）：
它同时持有我们的量化模型 + HF 的 BF16 副本 + 我们自己缓存的 BF16 反量化结果，
= 19 + 54 + 54 = 127 GB。当时的处置是**逐模块反量化→拷贝→立即释放**。
那个处置对"拷权重"有效，**对"跑前向"无效**——前向必须持有全部反量化权重。

## 可能的处置方向（均未验证，未拍板）

1. **加载时一次性反量化并释放量化态** —— 占用变成恒定 ~54 GB，不再有"跑着跑着涨上去"
   的行为；代价是失去了量化态的低占用启动。
2. **可配置的缓存策略**（缓存 / 不缓存 / LRU 上限）—— 不缓存意味着每次前向重复反量化，
   吞吐代价未测。
3. **NVFP4 直接计算**（不反量化到 BF16）—— 需要 kernel 支持，属 sparkinfer 范畴。

⚠️ 方向 3 与 B3 记录的另一条吞吐观察相关：B1 当前每次前向都要反量化到 BF16，
这已被 B2 标为"绝对吞吐 ~4 tok/s 的主因，B3 owns it"。

## 对使用者的直接建议（在修好之前）

**Qwen3.6 在这张卡上跑任何完整前向，就要按 ~54 GB+ 常驻来规划**，不要指望
`GPU_MEM_UTIL` 能压住它。要在共享卡上做验证，用**单层探针**而不是整模型——
B3 的 `scripts/b3_probe_gdn_spec_rollback.py` 就是这么做的（一层真实 GDN + 真实 FP8
权重，不加载整模型），正是为了避开这条。

## 相关

- `runtime/model/modelopt_linear.py`（机制本身，含 B1 的设计说明）
- `scripts/b1_verify_greedy_alignment.py`（同一机制的另一次 OOM 与处置）
- `scripts/b3_probe_gdn_spec_rollback.py`（规避方式：单层探针）
