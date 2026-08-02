# W4A4 blockscaled 走不通：**过不了 B1-R**（负面结论，已定案）

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）·
分支 `work/w4a4-20260803` @ `d58dadb` · 单卡 RTX PRO 6000 Blackwell Max-Q

## 结论

**不可用。** 0–55 层 MLP 改走真 W4A4（`sparkinfer.gemm.blockscaled.mm`）在数值上
**明确劣于**现有 W4A16 路径，**B1-R 的校准 gap-error 判据全线不过**。
**生产路径未改动**——`Qwen36MLP.forward` 一行没动，W4A16 仍是唯一的 NVFP4 MLP 路径。

## 这次为什么值得试（前提是对的）

阶段四此前有条结论说"`blockscaled.mm` 不适用，因为它要两个操作数都量化而 checkpoint 是
weight-only"。**那条只对 `nvidia/` 成立**，我 2026-08-03 早些时候把它误推广成了
"所有 checkpoint"（见 `docs/roadmap.md` 阶段 4 的纠正）。标准 checkpoint 的
`group_1` 声明 `input_activations: num_bits=4`，且 `input_global_scale` 是**实际发货
的张量**。所以前提确实变了，值得一试。

**kernel 契约也确实对得上**（动手前先核过，不是试出来的）：`blockscaled.mm` 的 NVFP4
recipe（`Float4E2M1` 数据、`e4m3` scale、`sf_vec_size=16`）与 checkpoint 的
`tensor_group` / `group_size=16` 在权重与激活两侧**完全匹配**，没有格式不符。

**所以这不是"方案选错了"，是数值上就是差。**

## 判据与数字

| | 实测 | 判据 | |
|---|---:|---:|---|
| `median_gap_error` | **0.5** | 0.25 | ✗ |
| `p90_gap_error` | **0.875** | 0.5 | ✗ |
| `p90_logprob_error` | **0.875** | 0.5 | ✗ |
| 最差单负载 `mean_kl_topk` | **7–8e-3** | 5e-3 | ✗ |

还有一条比数字更说明问题：**`instruction` 负载在 25–65 步内就发散到溢出了诊断的
top-1024 捕获窗口**——**比 B1-R 自己校准集里的任何一个注入 bug 都差**，那些全部落在
top-64 以内。

单层对照（layer 5，真实权重，对 BF16 参照）：

| 路径 | cosine | max-abs-err |
|---|---:|---:|
| 现有 W4A16 融合 | **0.999984–0.999990** | — |
| W4A4 blockscaled | 0.988–0.989 | ~1e-4 |

单层看"0.988 也不算离谱"，但那是**比现有路径差约 30 倍**的误差，
在 56 层自回归里复合起来就是上表的结果。**这正是"cos 很高"不能当判据的又一个实例。**

⚠️ **没有测速度。** 正确性是决定性判据且已经不过，测速没有意义——
测出来快也不能用，反而容易变成"再放宽一点判据"的诱因。

## 一个真陷阱：两个 global scale 都要**直接用、不取倒数**

`blockscaled.mm` 的 `alpha = 1 / (gs_weight × gs_activation)`，其中
`weight_global_scale` 与 `input_global_scale` **都按原值代入，不取倒数**——
**与现有 W4A16 路径 `nvfp4_components_for_fuse()` 的约定相反**
（那里 `weight_global_scale == 1 / weight_scale_2`，要取倒数）。

这不是推导出来就算了的：4 种正负/倒数组合在真实 layer-5 权重上实测，**3 种直接爆**
（输出恒 0，或 >1e10），只有"两个都直接用"落在 BF16 参照的量级上。
**搞反了会得到一个看起来正常、输出却是垃圾的模型**——和当初 `"!!!!"` 那个 bug 同一形状。

## 顺带修掉的：`input_global_scale` 此前被静默丢弃

`CompressedTensorsNVFP4Linear` 原先**根本没有建 `input_global_scale` 这个 Parameter**，
所以 checkpoint 里这个张量一直没人接。现已加载，并由
`tests/test_loading_compressed_tensors_mixed_precision.py::TestNvfp4W4A4ComponentsForFuse`
钉住上面那条约定（对着相反的 W4A16 式约定做断言）。

**注意这暴露的是一类问题而不只是一个**：`assert_all_params_loaded` 是**单向**的——
它保证"每个模型参数都拿到了 checkpoint 张量"，但**不保证"每个 checkpoint 张量都被消费"**。
一个没人接的 scale 张量是完全静默的。这次无害（W4A16 本就不需要激活 scale），
但同一个盲区下一次未必无害。

## 留下什么

- `scripts/verify_nvfp4_w4a4_gemm_single_layer.py` —— 单层数值对照
- `scripts/verify_nvfp4_w4a4_gemm_full_model_gap.py` —— B1-R gap-error 全模型判据
- `CompressedTensorsNVFP4Linear.nvfp4_w4a4_components_for_fuse()` —— **惰性、未接入生产**，
  连同上面那条约定一起留档，**免得下一个人从头再踩一遍倒数陷阱**。

## 对阶段四的意义

W4A4 这根杠杆（NVFP4 层 35%，10.75 ms/step）**关掉了**。
剩下的是另一根：**FP8 层 45%（13.83 ms/step、233 次调用）现在反量化成 BF16 再算**。
⚠️ 那条同样有前车之鉴：FP8 W8A8 单层 cosine 0.9996，**比 NVFP4 路径差 30–40×**
——与本次 W4A4 的 0.988 vs 0.99999 是**同一个量级的劣化**，而本次证明了这个量级
的劣化足以打穿 B1-R。**在投入实现前，应当先只做一次单层→全模型的判据预演。**
