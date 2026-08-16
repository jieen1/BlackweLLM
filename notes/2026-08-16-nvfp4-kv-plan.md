# Qwen3.8 backbone NVFP4 KV 方案（2026-08-16）

状态：🟢 **有效（方案已勘察，待实施）**

## 目标与预期

backbone KV 从 FP8（1 B/元素）降到 NVFP4（0.5 B/元素，block-16：e2m1 codes +
e4m3 块 scale）。verify attention 占 B1 轮时 ~41%（融合前数据），KV 读流量减半
→ 预期 decode **+12~15%**（注意：MTP FP8 KV 的教训——接受率是门禁，见 §4）。

## 现状与先例（b12x 内已有完整 nvfp4 KV 参考）

| 组件 | 现状 | 可复用先例 |
|---|---|---|
| 写入侧 | `fused_kv_scatter`（runtime，fp8） | `b12x/attention/_shared/mla/kv_cache.py::ConcatAndCacheNvfp4MlaFp8RopeKernel`（e2m1x2 打包 `cvt.rn.satfinite.e2m1x2` + e4m3 scale，`quantize_and_pack_16_fast`） |
| 读取侧 | `b12x/attention/paged/forward_paged.py` PagedForwardKernel（fp8 平面 + per-layer 标量 descale） | `b12x/attention/_shared/mla/decode_math.py` / `prefill_mg.py`（`_ld_global_nvfp4_fp8_rope_bfloat2` 全局载入+解包+scale 解码）；`b12x/gemm/blockscaled`（fp4 MMA）；`b12x/quantization/nvfp4/_impl.py` |
| 打包/scale 语义 | — | MLA 的 reader 语义：`e2m1 * e4m3_decode(scale_byte) * s_t`（与 runtime 的 `nvfp4_quant.py` block-16 量化同族） |

## 改动面（按小步拆分）

- **S1（可行性）**：b12x 写 nvfp4-KV decode kernel 最小原型（M=1 固定形状），
  复用 mla 的解包/scale 语义，对照 fp8 路径做数值验证（`test_attention_paged_forward` 风格）
- **S2（接入）**：planner/gate 新增 nvfp4 kv_dtype 路径（`_FP8_KV_DTYPE` 平行分支）；
  workspace 的 descale 结构从 per-layer 标量改为 per-16-block（grouped scale view）
- **S3（写入侧）**：runtime `fused_kv_scatter` nvfp4 变体（block-16 量化 + swizzle，
  复用 `nvfp4_quant.py` 的量化核与 oracle 位级测试）
- **S4（门禁）**：128K 真实文本 MTP 接受率 A/B + 质量 suite

## 门禁（MTP FP8 KV 的教训，2026-08-16 实测）

digit filler 100% 接受率掩盖了 FP8 MTP KV 的真实文本回归（46.4%→36%，净亏）。
NVFP4 KV 直接影响 **target** logits（不只是 draft），质量风险更高，S4 不可跳过；
且 bench 需要增加真实文本 cell（当前 c1/c4 全 digit filler，见 notes 同日晚些记录）。

## 预期风险

- KV 4-bit 精度在 128K 上下文的注意力权重分布上可能不足（FP8 已是 1 字节折中）
- 解包/scale 的 smem/寄存器成本可能吃掉部分字节收益（需 S1 测真实 kernel 时间）
- TMA 平面布局需为打包格式重定义（`_paged_kv_tma_plane_layout` 家族）
