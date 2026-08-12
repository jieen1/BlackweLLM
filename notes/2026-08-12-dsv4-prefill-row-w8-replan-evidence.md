# DSV4 prefill row-W8 重规划证据

> 日期：2026-08-12
>
> runtime 基线：`main@882d209`
>
> 结论：旧 Phase 2B 已关闭；row-scaled W8 + per-token A8 是唯一进入下一轮正式
> C0 gate 的候选，但目前只完成 single-expert 数值预筛，**没有性能通过结论**。

## 问题

K32 scale-amortized IQ2 Tensor-Core kernel 已达到 quality gate，却在真实 E=256 路径上得到
gate+up 6.93 ms、down 6.92 ms，远超 2.4/1.3 ms。需要寻找一种仍只常驻 IQ2_XS、但能把
GEMM 交给标准整 K Tensor Core mainloop 的表示。

## 最新代码审计

- `runtime/backends/dsv4.py::prefill()` 仍调用 `_prefill_logits()`；
- `_prefill_superchunk_logits()` 没有接 production；
- `Dsv4MoE.forward()` 的多 row 路径仍调用 `_route_expanded_prefill()`；
- `iq2_mma16_tc` 是独立 candidate，没有 production wiring；
- `grouped_moe_prefill()` 仍有 `.item()`、动态 allocation 和 eager sort/reduce。

这些事实由 `main@882d209` 源码审计得出，不是性能推断。

## 硬件容量

复现命令：

```bash
~/.venvs/vllm/bin/python -c "import torch; p=torch.cuda.get_device_properties(0); print(p); print('total_memory', p.total_memory); print('l2_cache_size', p.L2_cache_size)"
```

输出：

```text
name='NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'
major=12, minor=0, total_memory=97886MB, multi_processor_count=188
total_memory 102641369088
l2_cache_size 134217728
```

因此本机 L2 是 128 MiB。这个值只证明 `tile_E=2/4` 的 W8 working set 有容量可能性，不证明
producer→consumer 一定命中 L2；后者必须由 Phase C0 Nsight counters 证明。

## row-W8 数值预筛

配置：真实 GGUF
`/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf`，
`blk.4` expert 0，M=24，seed 20260812。gate/up 使用 K=4096 的单个 row scale；down 使用
K=2048 的单个 row scale；activation 每 row 一个 scale。exact oracle 来自
`runtime.model.dsv4_quant.dequantize_iq2_xs`。

复现命令使用仓库已有 `tools.prescreen_iq2_kgroup_fold`，把 K group 设为整行；未创建临时脚本：

```bash
~/.venvs/vllm/bin/python -c "import torch; import tools.prescreen_iq2_kgroup_fold as p; torch.manual_seed(20260812); A=(torch.randn(24,p.HIDDEN)*0.1).cuda(); pg=p.load_expert_packed(p.GATE_NAME,0,p.INTER,p.HIDDEN); pu=p.load_expert_packed(p.UP_NAME,0,p.INTER,p.HIDDEN); pd=p.load_expert_packed(p.DOWN_NAME,0,p.HIDDEN,p.INTER); Wg=p.dequantize_iq2_xs(pg).reshape(p.INTER,p.HIDDEN); Wu=p.dequantize_iq2_xs(pu).reshape(p.INTER,p.HIDDEN); Wd=p.dequantize_iq2_xs(pd).reshape(p.HIDDEN,p.INTER); rg=A@Wg.t(); ru=A@Wu.t(); hr=torch.nn.functional.silu(torch.clamp(rg,max=10.0))*torch.clamp(ru,min=-10.0,max=10.0); rd=hr@Wd.t(); g=p.k_group_fold_gemm_iq2(A,pg,p.INTER,p.HIDDEN,4096); u=p.k_group_fold_gemm_iq2(A,pu,p.INTER,p.HIDDEN,4096); h=torch.nn.functional.silu(torch.clamp(g,max=10.0))*torch.clamp(u,min=-10.0,max=10.0); d=p.k_group_fold_gemm_iq2(h,pd,p.HIDDEN,p.INTER,2048); p.report('K4096 gate',g,rg); p.report('K4096 up',u,ru); p.report('K4096/2048 down',d,rd)"
```

输出：

```text
K4096 gate: cos=0.9999238 cos_min_row=0.9999151 rel_l2_max=12.2955
K4096 up: cos=0.9999241 cos_min_row=0.9999127 rel_l2_max=10.8460
K4096/2048 down: cos=0.9996461 cos_min_row=0.9992452 rel_l2_max=12.1587
```

`rel_l2_max` 的值被接近零的 reference 元素放大，不能单独作为质量结论。进入 C0 的依据是
gate/up per-row cosine 过 0.9999、传播后 cosine 仍高于 0.999；最终仍需真实 top-6、完整 layer
和 full-model gate。

同一会话还探索了 FP4/FP6/FP8 transient 表示，但没有保留可复现 operation/run record；
这些临时数字不用于正式关闭或通过路线，必须在 C0 重新测量。

## 流量账

一个 expert 的单个 2048×4096 W8 projection 是 8 MiB。E=256 时：

- gate+up W8 materialize 是 4 GiB，写后再读共 8 GiB/layer；
- down W8 materialize 是 2 GiB，写后再读共 4 GiB/layer；
- 1792 GB/s 峰值下，仅上述 logical W8 流量的理想下限约 4.8/2.4 ms，已超 2.4/1.3 ms；
- `tile_E=4` gate+up W8 只有 64 MiB，down 32 MiB，能够在容量上进入 128 MiB L2。

所以 full-layer/global materialize 被流量账关闭；circular tile 只获得“值得直接测”的资格。
若实际 DRAM counter 仍接近 logical W8 bytes，circular tile 同样关闭。

## 证据边界与下一步

当前没有：

- 多层、多 expert、真实 top-6 route quality；
- CUTLASS W8A8 candidate；
- L2 hit / DRAM read-write counters；
- bfdiag run record；
- end-to-end throughput。

因此下一步只能执行
[`../docs/dsv4-prefill-2k-implementation-plan.md`](../docs/dsv4-prefill-2k-implementation-plan.md)
的 Phase C0，不能接 production backend，也不能宣称 2K 已证可达。

## 补充：手写 W8A8 GEMM 原型（2026-08-12）

CUTLASS 4.6.1 的 SM120 builder 只支持 F8F6F4 MMA（无 s8），TCGEN05 s8 在
compute_120f 不可用。故手写 m16n8k16.s8.s8.s32 GEMM（指令峰值实测 198 TOPS）。

手写 kernel（/tmp/opencode/iq2_w8a8_probe.cu）已正确（cos 0.999998）：
- BLOCK_M=32、cp.async A + 预转置 n-major B staging
- M=512 shared-B 达 121 TFLOPS；batched B 73 TFLOPS @ M=256
- gate+up 估计 2.75-3.67ms（target 2.4）、down 1.34-2.7ms（target 1.3）

**关键失败点：L2 hit 只有 23%**（C0 门禁要求 >=90%）。原因：B 的 N=128
分片消费，B 在 L2 中未被充分复用；tile_E=4（B=64MB）和 tile_E=8（B=128MB）
都只有 ~23% hit。DRAM 100% 饱和。

结论：**C0 的 L2 生死门（>=90% hit）在现 kernel 结构下失败**。B 的 L2 复用
需要更大的 N-per-block（整 B 一次读）或 persistent kernel + 显式 L2 管理，
超出当前原型范围。真实 IQ2 转码的 circular scratch 可能改善局部性，但
当前证据不支撑 >=90% 目标。

## 补充 2：transcode 与 W8 缓存矛盾（2026-08-12，C0 生死门判定）

实现并验证了 IQ2→row-W8 transcode（两 pass 并行，32us/expert，数值正确：
scale/w8 与 python 逐行 1.0000 匹配）。GEMM 原型 M=512 达 121 TFLOPS。

**但组合暴露 C0 致命矛盾**：
1. W8 每 expert 8MB（gate 或 up 单矩阵）。全 256 experts × 3（gate/up/down）=
   6GB W8。超出 C0 scratch 上限 256MiB（文档 §4.2）。
2. 若每 token 重新 transcode：256 experts × 32us = 8.1ms/矩阵种类，
   gate+up+down = 24ms。远超 2.4/1.3ms。
3. 只有 transcode 一次性（启动 prewarm）才可行，但 W8 6GB 无地可放。

**C0 判定**：row-W8 表示本身可行（GEMM 121 TFLOPS、transcode 正确），但
**W8 的"生成一次 vs 常驻"矛盾无解**——circular scratch（每 tile 重新
transcode）使 transcode 成为 per-token 瓶颈（24ms），常驻 W8 超内存。
加上 L2 hit 23%（B 分片消费），C0 的 2.4/1.3ms 和 >=90% L2 门禁均未达。

这与 Phase 2B 的 int8-codebook DRAM 死结是同一结构：**放大驻留（W8）换取
零 decode，代价是内存/带宽，最终都撞墙**。row-W8 只是把 decode 成本换成
了 transcode+W8 带宽成本。

## 补充 3：C0 生死门最终判定（2026-08-12）

端到端 tile_E=4 circular 流程实测：transcode 4 experts gate → W8 scratch →
GEMM = **0.220ms/tile**，x64 tiles = **14.1ms（gate alone）/ 28.2ms（gate+up）**。
C0 kill gate 2.4ms 超 11.8x。

构成：transcode 0.163ms/tile（74%）、GEMM 0.057ms/tile（26%）。

**C0 判定：失败。** 三个致命约束在 row-W8 上同时成立：
1. **per-token transcode 是主瓶颈**（circular scratch 每 tile 重新 transcode，
   W8 写 32MB/tile 的 DRAM 成本 28ms/全量）；transcode 已接近 DRAM 理论下限
   （decode kernel 77% SM throughput），无数量级优化空间。
2. **W8 常驻 6.4GB 超内存**（gate+up+down 全部 256 experts），与 256MiB
   scratch 和 2×128K KV 无法共存。
3. **GEMM L2 hit 23%**（B 分片消费，tile_E=4/8 均无跨 GEMM 复用），DRAM-bound。

row-W8 表示本身可行（GEMM 121 TFLOPS），但与 Phase 2B int8-codebook 是同一
结构死结：**放大驻留换零 decode，代价是带宽/内存，最终撞墙**。

**§9 分支 (b)/(c) 待用户定夺**：重新核算 6.5ms/layer 预算，或如实记录
"当前 IQ2_XS + 单 SM120 + 质量/容量约束下 2K 无已知可实现路径"。
