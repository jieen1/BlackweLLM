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

- **S1（可行性）✅ 完成 2026-08-16**：runtime 写入侧 nvfp4 往返验证
  （`quantize_dequantize_nvfp4_roundtrip` + `QSR_QWEN36_NVFP4_KV=1`，commit `3c272ab`）。
  128K 实测：图捕获全成功、decode 103.9 tok/s、输出与 FP8 分叉但质量相当
  （char 68 处开始不同、754/881 字符差异、无崩坏）；真实文本接受率 57.1% vs
  FP8 46.4%（FP8 两次独立测量均 46.4%，稳定）——**数值可行**。
- **带宽决策实验 ✅ 2026-08-16**（128K×45 层规模，Triton 纯读 kernel）：
  fp8 读 12.1 GB = 7.6 ms；nvfp4 读+解包 6.0 GB = 2.9 ms（解包被带宽-bound
  特性部分隐藏，等效吞吐 2.08 vs 1.59 TB/s）→ **每轮省 4.7 ms ≈ decode +12%**。
  S2 值得投入。
- **S2（读侧 kernel）⬜ 插入点已定位**（`b12x/attention/paged/forward_paged.py`）：
  1. `__init__` dtype 分支（`kv_is_fp8` 平级加 `kv_is_nvfp4`，line 3156/3430-3432）
  2. KV TMA 源/布局（`_make_paged_kv_tile_source_tensor`、`kv_tma_plane_mem_dtype`、
     `_get_paged_kv_tma_plane_layout`）
  3. smem 载入（`_issue_paged_kv_tma_copy_*`）后插入**解包 copy**（打包行→fp8 smem）
  4. QK/PV MMA 消费点（`tiled_mma_qk_tma`，line 5498+）不变——操作数改为解包后的 fp8 smem
  - 解包实现参考：`b12x/attention/_shared/mla/decode_math.py` 的
    `_ld_global_nvfp4_fp8_rope_bfloat2` / `_nvfp4_pair_bfloat2_mg`（e2m1 解包 +
    e4m3 scale 解码，cute 实现可直接移植）
  - 池布局：每 (page, token, kv_head) 行 144 B = 128 B e2m1 codes + 16 B e4m3 scales
    （head_dim=256 = 16 个 block-16）
- **S3（写入侧真 nvfp4 池）**：`fused_kv_scatter` nvfp4 变体（打包写入，不再解包回
  fp8）——S2 完成后替换 S1 的往返
- **S4（门禁）**：128K 真实文本 MTP 接受率 A/B + 质量 suite

## 门禁（MTP FP8 KV 的教训，2026-08-16 实测）

digit filler 100% 接受率掩盖了 FP8 MTP KV 的真实文本回归（46.4%→36%，净亏）。
NVFP4 KV 直接影响 **target** logits（不只是 draft），质量风险更高，S4 不可跳过；
且 bench 需要增加真实文本 cell（当前 c1/c4 全 digit filler，见 notes 同日晚些记录）。

## 预期风险

- KV 4-bit 精度在 128K 上下文的注意力权重分布上可能不足（FP8 已是 1 字节折中）
- 解包/scale 的 smem/寄存器成本可能吃掉部分字节收益（需 S1 测真实 kernel 时间）
- TMA 平面布局需为打包格式重定义（`_paged_kv_tma_plane_layout` 家族）
