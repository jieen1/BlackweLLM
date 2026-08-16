# Qwen3.8 decode：W8A8/W4A4 GEMM roofline 调查——decode 已至 DRAM 带宽地板（2026-08-16）

状态：🟢 **实测定案**。规划 §6.1 决策树"按 GPU ms 排前两名"指向 W8A8 GEMM
（33% 轮时）与 W4A4 MLP（21%）；本调查测量这两族的真实带宽利用率，判定
kernel 级调优空间，并给出规划层面的结论。

## 背景数字（来自同日 B1 kernel-family 归因）

B1 128K decode 轮 GPU 32.2 ms：W8A8 GEMM 33.0%（nsys 表观 10.64 ms/轮，
188 次/轮）、W4A4 MLP 20.8%（6.7 ms/轮）、attention 16.5%（已单独测得
79-89% DRAM 峰值，见 `2026-08-16-qwen38-b1-decode-kernel-attribution.md`）。

## 测量方法演进（bench 方法学教训，三次迭代）

1. **逐次 sync bench**：每次 GEMM 后 `cuda.synchronize()`。结果 208-253 µs
   ——被 sync 排空流水线严重放大，不可用。
2. **批量发射 + 6 份轮转权重**（试图打冷 L2）：出现**双峰**——同一形状
   有时 ~28 µs（~1140 GB/s）有时 ~47 µs（~660 GB/s）。根因不是硬件状态，
   是 **bench 自身缺陷**：轮转足迹 6×31.5 MB = 189 MB，而本卡 L2 ≈126 MB，
   **轮转缓冲部分常驻 L2**，命中迭代快、未命中迭代慢。"冷 bench" 不冷。
   此假象曾误导出"N32 tile 有效/无效"的矛盾读数。
3. **真冷 bench**：每次发射前用 300 MB `fill_` 冲刷 L2（远超 L2 容量），
   CUDA event 计时。双峰消失，数字稳定，M=1 与 M=4 一致（小 M 下纯权重
   流带宽主导）。以下结论只基于方法 3。

## 真冷测量结果（qsr_fp8_w8a8 生产 kernel，M=1/4 相同）

| GEMM 形状（N×K） | 次数/轮 | µs | GB/s | %峰值(1792) |
|---|---:|---:|---:|---:|
| attn.q_proj 12288×5120 | 16 | 57.3 | 1097 | 61% |
| attn.o_proj 5120×6144 | 16 | 30.7 | 1024 | 57% |
| gdn.in_proj_qkvz 16384×5120 | 48 | 65.5-67.4 | 1245-1280 | 70-71% |
| gdn.out_proj 5120×6144 | 48 | 30.5-30.7 | 1024-1032 | 57-58% |
| attn.k_proj / v_proj 1024×5120 | 32 | 15-16 | 280-350 | 16-20% |

大 GEMM 真冷带宽 1024-1280 GB/s；k/v_proj 因 N=1024 只有 16 个 CTA
（tile N=64）欠填充，但绝对量小（合计 ~0.5 ms/轮）。

W4A4 MLP 同法推算：8.4 GB 权重 / 6.7 ms ≈ 1254 GB/s ≈ 70% 峰值，同样
接近地板。

## N32 tile 实验（负面）

假设：N≤5120 形状（o_proj/gdn.out，80 CTA）换 tile N=32（160 CTA）可提
带宽。实施 `GemmM16N32`（`Shape<_16,_32,_128>`）重建 .so 对比：各轮总量
在 bench 噪声内重叠（基线中位 ~6.8 ms vs N32 ~6.8 ms），单形状读数被
方法 2 的 L2 假象污染无法分辨。**判定无收益，改动已回退**（
`git checkout -- runtime/kernels/fp8_w8a8_sm120.cu` + 原版重建，24 个
W8A8 单测通过）。真冷视角下 o_proj/gdn.out 已 57-58% 峰值，CTA 数不是
瓶颈（160 CTA 的 qkvz 也只到 70%），进一步印证。

## 生产对账：表观 10.64 ms 的构成

nsys 表观 W8A8 = 681 ms / 64 轮 = 10.64 ms/轮。按时长分布拆解：

- 188 次/轮中 50% 为 10-25 µs、47% 为 45-70 µs——与真冷孤立值吻合；
- **另有 295 次 >110 µs 的长调用（~850 µs×187 + ~800 µs×55 为主），合计
  3.69 ms/轮（35%）**。它们的**最小间隔 348 ms >> 单轮 38 ms**——不是
  每轮 decode GEMM，是 prefill 大 M GEMM 与一次性长尾混入了 decode 窗口
  统计（decode 窗口按最后一个 extend kernel 切分，切不干净 prefill 尾部）。

剔除后真实每轮 decode W8A8 ≈ **6.9 ms，与真冷孤立求和 ~6.4 ms 吻合**。
**不存在隐藏的 4 ms 缺口**——decode 路径的 GEMM 就跑在真冷带宽上。

## 结论与规划影响

1. **W8A8/W4A4 GEMM 已跑在真冷 DRAM 带宽（57-71% 峰值）**；把全部 GEMM
   优化到 100% 峰值的理论上限也只有 ~13% 轮时，且不可达。kernel 级调优
   （tile/stage/CTA）基本无空间。
2. **decode 轮已逼近该模型+精度下的 DRAM 带宽地板**：三大族（W8A8 33%、
   W4A4 21%、attention 16.5%）全部带宽受限且接近峰值。
3. 规划层面：§6.3/§7 的 kernel 微调项（FA4 迁移、tile/stage 搜索、GEMM
   调优）整体降级。剩余真实杠杆只有三类：
   - **减字节**：更激进的权重/KV 量化（超出当前范围）；
   - **提有效 M**：并发/批处理结构（架构级，round 数不变时提升总吞吐）；
   - **小 kernel 融合**：quant/norm/copy/gemvx 合计 ~17%（其中 gemvx +
     wmma bf16 ~5.4% 是 cuBLAS 小 M 路径，norm reduce ~2.9%，per-token
     quant 4%），单项小但合计是仅剩的 kernel 可动面。
4. bench 方法学：小 M GEMM 冷带宽测量必须满足（a）批量发射免 sync 放大、
   （b）轮转足迹 >> L2（≥1.5×，本卡 ≥190 MB）、（c）或每次迭代前主动冲刷
   L2。三者缺一会出假数字。

## 证据文件

- `/tmp/opencode/bench_w8a8_truecold.py`（真冷 bench，可复跑）
- `/tmp/opencode/bench_w8a8_cold2.py`（含 L2 假象的教训版）
- `/tmp/opencode/nsys_c1prod2.nsys-rep`（node-trace 生产采集）
- N32 实验：改动已回退，无代码残留
