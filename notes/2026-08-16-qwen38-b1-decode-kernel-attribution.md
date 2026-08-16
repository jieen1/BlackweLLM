# Qwen3.8 B1 128K decode：生产 kernel-family 归因与 P0-A3 判定（2026-08-16）

状态：🟢 **实测完成，P0-A3（attention tile/stage 联合搜索）判定为低优先**。

## 测量方法

生产服务（K=3、FP8 KV、elastic pool、CUDA Graph 全捕获）包在
`nsys profile --trace=cuda --cuda-graph-trace=node` 下跑 c1 128K
（131072 prompt + 256 tokens，decode 104.18 tok/s = 64 轮）。按
extend_generic kernel 的最后一次出现切出 decode 段，聚合 graph 内全部
kernel。**graph-node 追踪是必须的**：默认 graph 粒度下 graph 内 kernel
不进 kernel 表（同日第一次采集只看到 9 ms/轮 GPU，是假象）。

## Decode 轮 GPU 分解（64 轮，GPU 合计 2062 ms = 32.2 ms/轮）

与轮级 profile 的 B1 GPU≈36 ms（verify 29.5 + sync 2.7 + draft 3.8）吻合。

| family | ms/轮 | 占 GPU | kernel |
|---|---:|---:|---|
| W8A8 FP8 GEMM（attn/GDN 投影） | 10.64 | **33.0%** | qsr_fp8_w8a8 CUTLASS ×188/轮，56.6 µs/次 |
| W4A4 NVFP4 MLP（down + gate/up） | 6.68 | **20.8%** | b12x dense_gemm f4E2M1FN，37.5/44.6 µs/次 |
| **attention（b12x paged，FP8）** | **5.30** | **16.5%** | 16 层 verify ×264.4 µs + MTP ×362 µs |
| W8A8 激活量化 | 1.27 | 4.0% | per_token_e4m3_quant ×188/轮 |
| gemvx + wmma bf16（cuBLAS 小 M） | ~1.05 | 3.3% | MTP head / 杂项 |
| rmsnorm（reduce+tail） | ~0.95 | 2.9% | |
| copy/index/elementwise | ~1.5 | 4.6% | |
| GDN recurrent | 0.41 | 1.3% | fused_recurrent_gdn_multistep_indexed |

## Attention 带宽已达峰值附近

- 生产 B1 verify attention：264.4 µs/层，每层读 4×…= 0.27 GB KV
  → **≈1023 GB/s ≈ 79% DRAM 峰值**。
- 独立 graph-path bench（同生产驱动 for_contract + replay）：B1 0.405 ms
  （664 GB/s）、B4 0.926 ms（**1160 GB/s ≈ 89% 峰值**）。
- 对照：同形状 **eager 路径** B1 要 3.8 ms（70 GB/s）——NCU 显示 1 CTA/SM
  （67.58 KiB SMEM 上限）、8.33% 占用率。生产不走 eager 路径，该 stall
  画像不代表生产。

## P0-A3 判定

规划 §7.3 的前提（attention 是最大且低效的 family，值得 tile/stage/
residency/SplitKV 联合搜索）**在生产 decode 上不成立**：

1. attention 只占轮 GPU 的 16.5%（B1；c4 为 ~32%），不是最大 family；
2. 它已在 DRAM 峰值的 79–89%，tile/stage 调优的理论上限 ≤20% 的
   attention 时间 = **≤3.3% 轮时（B1）/ ≤6%（c4 且乐观）**；
3. 代价是 b12x kernel 手术 + 全形状回归——ROI 显著低于其余方向。

**判定：P0-A3 降级为"有 NCU 证据表明 attention 离开峰值带宽时再触发"的
条件项**；eager 路径的 1-CTA/SM 问题留档（只影响诊断/fallback 路径）。

## 数据指向的下一个优化面

轮时的 54% 在 W8A8+W4A4 GEMM（小 M 权重带宽受限）：
- W8A8 188 次/轮 ×56.6 µs——attn qkvz/o + GDN 投影，M=4；
- W4A4 MLP 112 次/轮——gate/up/down 三 GEMM；
- 结构性解法只有三条：更大的有效 M（并发/投机结构）、更少的权重字节
  （更激进量化，超出当前范围）、小 M GEMM kernel 本身的效率。
其余 ~17% 是 quant/norm/copy/gemvx 小 kernel 集合（融合候选，单项收益小）。

## 证据文件

- `/tmp/opencode/nsys_c1prod2.nsys-rep`（node-trace 采集）
- `/tmp/opencode/bench_graph_verify_attn.py` / `bench_graph_decode_attn.py`
  （独立 graph-path bench，可复跑）
- c1 性能：104.18 tok/s（TTFT 60.5 s，与基线一致）
