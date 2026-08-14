# DSV4 动态 MoE routes>64 修复（2026-08-14）

## 状态与结论

**已修复。** `iq2_mma16_tc_dynamic_launch` 和 down 侧
`iq2_mma16_tc_launch_single_dynamic` 现在由每个 expert 的 CTA 在设备端按 64 routes
循环，最后一个 tile 继续使用原来的 route mask。真实权重和合成回归均确认：超过
64 的 tail 不再保留未初始化值，动态路径与旧的分批参考逐元素相等。

最初记录的“第一个 64-route tile 已经产生 NaN”不是 kernel 算术故障，而是诊断调用
违反了 launcher 的索引契约：诊断构造了全局 `expert_bounds[245:247]` 和
`eids=[245]`，却以 `E=1` 启动。kernel 的 `e=blockIdx.x` 同时索引 `eids[e]` 和
`expert_bounds[e:e+2]`，因此该调用读取的是 `expert_bounds[0:2] == 0`，没有处理任何
route；随后读取 `torch.empty` 输出，把未初始化显存误判成了 kernel 写出的 NaN。

## 实际缺陷

旧动态 kernel 固定 `M_PAD_C=64`，只从 `expert_bounds[e]` 处理一个 tile。Python
wrapper 每次只 launch 一次，也没有传入 tile offset。因此 routes>64 时：

- 前 64 行计算正确；
- 第 65 行起从未写入；
- 后续 SwiGLU/down/scatter 消费 `torch.empty` 中的未初始化值，可能表现为 NaN。

这也解释了为何问题严格跨过 64-route 边界才出现。`compact_xq`、`compact_xs`、
packed weights 和全局 `expert_bounds` 本身没有问题。

## 修复结构

gate/up 与 down kernel 均使用相同结构：

```cuda
const int route_begin = expert_bounds[e];
const int route_hi = expert_bounds[e + 1];

for (int route_lo = route_begin; route_lo < route_hi; route_lo += 64) {
    // 只保留 64-route 的 accumulator；每个 tile 独立跑完整 K loop。
    // 最后一个 tile 由 route_lo + row < route_hi 掩码。
}
```

选择 device-side loop，而不是 Python 分段 launch，是为了保留一次 launch、避免读取
route 计数造成主机同步，同时把累加器限制在已验证的 M=64 资源规模。gate/up 动态
kernel 的资源与固定 M=64 路径相同：96 registers、128-byte stack、0 local memory；
down 动态 kernel 为 64 registers、80-byte stack、0 local memory。

Python wrapper 还显式拒绝非 BF16 activation。`_preq(flat)` 的 scale dtype 跟随
`flat`，而 gather CUDA ABI 按 BF16 读取原 scale、再写入 FP32 compact scale；显式检查
避免未来传入 FP32 时发生静默 reinterpret。

## 正确性证据

### 合成回归

`tests/test_iq2_mma16_tc_kernel.py::test_dynamic_moe_tiles_expert_routes_over_64`
构造 256 experts：expert 0 先占 8 条 compact routes，expert 245 再占 81 条，覆盖
“非零 compact offset + 超过 64”的组合。完整 gate/up → SwiGLU → down → combine
动态路径与 `grouped_moe_prefill_k32(bucket=64)` 分批参考逐元素相等且全有限。

```text
6 passed in 2.60s
```

### 真实 DSV4 权重

在常驻 `bf` engine 中使用第 0 层真实 GGUF packed weights，expert 245 接收 81
routes，并保留生产复现中的非零 compact offset。前 64 行和 tail 分别对拍：

```text
ref_finite=true
dyn_finite_first64=true
dyn_finite_tail=true
first64_maxdiff=0.0
tail_maxdiff=0.0
all_equal=true
```

### 端到端同 prompt A/B

同一常驻模型、同一 prompt、greedy 单 token，原位 monkey-patch eager/dynamic MoE；
两条路径输出 token 相同。以下是 warm-engine 比较，能说明实现差异，但不是 cold-prefill
正式基线：

| prompt | eager | dynamic | 加速 |
|---:|---:|---:|---:|
| 512 tokens | 146.4 tok/s | 233.5 tok/s | 1.595x |
| 1024 tokens | 157.7 tok/s | 249.5 tok/s | 1.582x |

## 性能跟进：从 249.5 到 509.2 tok/s

后续优化没有把单个 microbenchmark 直接外推到 E2E，而是在同一 warm engine 中交错
A/B，并以冷重启后的 production profile 收口。固定配置为同一 GGUF、SM120、
`M=1024`、prefill tile 64、无 CUDA Graph、无 canary；prompt 是固定自然文本，交替修改
首 token 避免误用 prefix cache。

### 已保留的优化

1. **一次物化 GPU positions。** 原路径在 43 层 × 15 个 mid tiles 中重复
   `torch.tensor([abs_pos], device="cuda")`，profile 显示 647 次
   `cudaStreamSynchronize`。传入已知 host position 并一次创建 positions vector 后，
   sync 647→2（最终 profile 为 1）、`_local_scalar_dense` 931→0；同进程 A/B
   238.7→309.4 tok/s（+29.6%）。
2. **Q8_0 直接读常驻 SoA。** 旧 M>1 路径每次 projection 都把 `qcode/qscale`
   拼回 34-byte interleaved layout。1024-token profile 中，仅三组 code-plane copy
   就是 139.8 ms，scale copy 另 28.5 ms。新 dense/grouped Triton GEMM 直接读 SoA；
   `aten::copy_` GPU 时间 265.7→69.0 ms，整模型 logits 逐 bit 相等。同进程 10 对
   A/B：300.3→342.4 tok/s，平均省 0.420 s，8/10 对为正。
3. **tile 级 indexer compressor。** 旧 ratio-4 indexer 每 token 发 postgemv + migrate
   两个 kernel，整次 profile 各 20,160 launches。新路径用一个顺序 state/finalize
   kernel 加一个并行 Hadamard/FP4 kernel 处理整 tile：两者各 315 launches，GPU 时间
   从 82.2 ms 降到 24.3 ms。总 launch 由 `80,354 + 48,331` 降到
   `59,879 + 8,641`（-46.8%）；独立状态机和完整 1024-token logits 均逐 bit 相等。
   15 轮均值 354.6→468.4 tok/s（+32.1%）。
4. **cooperative route gather。** 4 threads/route + 16-byte vector copy 将 gather GPU
   时间 87.1→0.584 ms。它在旧的 CPU-submit-bound 路径中被提交空洞掩盖，首次 12 对
   A/B 甚至表现为 E2E 负收益；indexer 将 launch 数减半后重新测试，12 对中 10 对
   为正，483.5→502.3 tok/s，平均省 79.3 ms，因此才切为默认。serial symbol 保留供
   诊断回退。
5. **共享 dynamic MoE workspace/native handle。** 43 层复用同一套 route/output
   buffers；native library 也随 workspace 缓存，避免每层重复 manifest/hash/CDLL
   设置（实测 43 次 load 约 28.0 ms）。

### 冷启动最终结果与 llama.cpp 对照

从磁盘冷重启 daemon（不依赖热重绑定/monkeypatch），一次 JIT warmup 后运行 20 次：

| 指标 | 本 runtime | llama.cpp `llama-bench` |
|---|---:|---:|
| mean | **509.2 tok/s** | **567.8 tok/s** |
| median | **541.3 tok/s** | 567.8 tok/s（3-run mean） |
| 2-side trimmed mean | **519.0 tok/s** | — |
| min / max | 392.3 / 563.9 tok/s | 558.8 / 576.7 tok/s |

本 runtime 的 mean 仍落后 10.3%，trimmed mean 落后 8.6%；最好一轮已接近，但尾部
抖动显著，不能宣称追平。20/20 首 token 相同，所有 logits finite。最终冷启动
profile 确认 production 路径实际使用：

```text
iq2 gate/up dynamic       362.383 ms / 43
iq2 down dynamic          222.630 ms / 43
main compressor seq       279.546 ms / 615
Q8 SoA dense+grouped      328.869 ms / 4113
indexer seq state+quant    24.285 ms / 630
cooperative gather          0.584 ms / 43
cudaStreamSynchronize            1
_local_scalar_dense              0
```

### 已证伪/暂不默认的路线

- 根据真实 43 层 route histogram，`ceil(routes/16)` 的理论 padding 仅 22.7%，但把
  `n_mtiles` 改成 runtime loop bound 会破坏编译器展开，gate microbenchmark
  8.77→11.25 ms；固定四段加 runtime predicate 也无收益（8.354→8.371 ms）。tail
  优化需要真正的 template bucket dispatch/worklist，不能只加分支。
- cooperative gather 的第一次 E2E 负结果不是其 GPU profile 错误，而是 CPU 尚未提交
  完后续工作，87 ms GPU 工作落在非临界路径。只有先消除 40k indexer launches 后，
  它才成为可测的端到端收益。这是本轮最重要的“micro 快、E2E 不动”因果解释。

## 被推翻的判断与剩余工作

- **推翻：** “第一个 tile NaN，tile-loop 也无效。”第一 tile 诊断实际没有执行 kernel。
- **确认：** routes>64 的真实 bug 是 tail 未写，device tile-loop 已关闭该缺陷。
- **已更新：** 249.5 tok/s 是只完成 NaN/tile-loop 后的历史节点，不是当前结果；完成
  position sync、SoA Q8、tile-indexer 和 cooperative gather 后，冷启动 20-run mean 为
  509.2 tok/s。仍不得用最好一轮 563.9 宣称追平 llama.cpp。
- 下一阶段的硬目标是收敛 392–564 tok/s 的尾部抖动，并继续减少剩余 68,520 次
  launches；GPU 大头仍是 routed MoE（gate/up+down 585.0 ms）、main compressor
  279.5 ms 和 Q8 SoA projection 328.9 ms。完整 layer/prefill CUDA Graph 或进一步融合
  必须先证明状态/KV 写入和 graph memory 的正确性，不能只按 launch 数外推。

## 第二阶段：并行 compressor 与整层 attention graph

### cold-prefill compressor 并行化

对固定 `prefix=0, M=1024, tile=64` 路径，首 tile 仍按已验证的顺序状态机执行，
后续 compression boundary 在 GPU 上并行生成，并单独恢复与 64-token oracle 一致的
`kv_state/score_state`。固定 prompt 的 16-token greedy 序列逐项相同。20 次 prefill-only
实测如下：

| 指标 | 串行 compressor 基线 | 并行 compressor |
|---|---:|---:|
| mean | 509.2 tok/s | **560.9 tok/s** |
| median | 541.3 tok/s | **567.5 tok/s** |
| trimmed mean | 519.0 tok/s | **565.1 tok/s** |
| fastest | 563.9 tok/s | **603.9 tok/s** |

profile 中 main compressor 从 `279.5 ms / 615` 降为 `1.36 ms / 41`，但 E2E 只提升约
10%：被删掉的 compressor kernels 之前大量与 CPU submit 空洞及其他 GPU 工作重叠，
不在完整关键路径上。这个差异正是不能用 micro/profile total 直接预测 E2E 的证据。

### dynamic MoE route bucket 实验（已回退）

设备端把 experts 分为 `<=16/<=32/<=48/>48` 四组、gate/down 各发四个模板 kernel。
输出逐 token 相同，但 20 次 E2E 从并行-compressor 基线的 560.9/567.5 tok/s
降为 532.7/538.9 tok/s（mean/median）。真实 profile 只省了 gate 约 10.5 ms，down 反而
慢约 5.7 ms，每层多 6 次 launch 的代价最终导致约 5% E2E 回归。该改动已完全回退。

### 整层 attention tile-loop CUDA Graph

将每层 16 个 attention tiles（包括首 tile、并行 compressor 预计算和后续 15 tiles）
捕获为一个 graph，43 层共享 graph pool。捕获严格限定在
`num_slots=1, slot=0, prefix=0, M=1024, tile=64`，其他形状回退 eager tile-loop。

- 43 层 capture 耗时 9.76 s；reserved 增量 1.10 GiB，allocated 增量 1.02 GiB。
- 固定 prompt 的 16-token greedy 输出与基线逐项相同：
  `[5148, 16, 223, 455, 2502, 344, 260, 73615, 45750, 22891, 12275, 418, 12529, 5148, 16, 455]`。
- 20 次 graph-on prefill-only：mean 714.2、median 752.0、trimmed 730.2、fastest
  785.0 tok/s。20/20 首 token 为 5148。
- 同一进程内交替 graph off/on 的 12 对对照：off median 534.7 tok/s，on median
  **759.9 tok/s**，完整 E2E 提升 **42.1%**。因 graph pool 常驻且 prompt/输出相同，
  这个 A/B 排除了冷启、内存配置和测试口径差异。

graph-on 的真实 profile 仍显示主要 GPU 时间在 MoE/Q8，而不是 attention：

```text
iq2 gate/up dynamic       372.6 ms / 43
Q8 SoA dense              287.1 ms / 3425
iq2 down dynamic          232.2 ms / 43
Q8 SoA grouped             55.1 ms / 688
HC + dense CUTLASS       ~109.8 ms
attention kernels         ~61.0 ms
```

因此当前可对外声称的是“从稳定约 565 提到配对中位约 760 tok/s”，不是
1000 tok/s。下一步必须优先批量化 attention 中 tile-invariant 的 Q/KV 与输出投影
（当前 3425+688 次 Q8 kernel），并继续攻 routed MoE 的 604.8 ms；attention 本体即使
完全消失也不足以达到 1000 tok/s。

## 第三阶段：attention 投影批量化实验（已回退）

上一节把 attention 的 Q/KV 与输出投影批量化列为下一步，2026-08-14 实机验证后该
判断只成立一半，不能作为默认实现：

- 1024 行一次完成 `wq_a/wq_b/wkv`，MLA 仍按 64 行 tile，单层 ratio-0/4/128
  与旧路径逐元素相等；全模型固定 prompt 的 16-token greedy 序列也逐项相同。
- 如果把 grouped `wo_a` 也合成 1024 行，真实 profile 中 dense Q8 从约
  `287 ms` 降到 `231 ms`，但 grouped Q8 从约 `55 ms` 升到 `214 ms`。固定 prompt
  同进程交替 A/B 只有 eager `497.1` → graph `571.2 tok/s`（+14.9%），显著低于
  旧 attention graph 的配对中位 `759.9 tok/s`。
- 改成“Q/KV 批量、`wo_a/wo_b` 仍逐 tile”的混合结构后保持 bit-exact，但 capture
  pool 把进程推到显存边缘：PyTorch `reserved=92.44 GiB`、GPU 仅余约 `0.16 GiB`；
  开关图两侧随后都出现 5–11 秒尖峰。`empty_cache()` 后也只恢复到约 0.87 GiB，
  无法形成可信稳定 E2E，更不满足生产余量。
- 中途发现并修正了一个诊断原型错误：b12x MLA 返回复用 scratch 的视图，把 16 个
  raw tile 存入列表会被后续调用覆盖；逐 tile 物化后 ratio-0/4/128 均恢复
  `max_diff=0`。该修正只证明原型数值语义，不改变上述性能/显存否决。

结论：projection 调用数减少并不等于 E2E 关键路径缩短。1024-row grouped `wo_a` 的
tile 选择明显不适合该形状，而 Q/KV 批量又扩大 graph 中间量生命周期和 capture pool。
整条代码分支已回退，当前默认仍是 16×64 attention graph。下一步应优先把现有
attention + HC + dynamic MoE 串入完整 block graph，同时保持 tile 局部中间量生命周期；
若再碰 Q8，只做 dominant-shape 的 kernel/selector 调优，不再用全序列批量化硬换 launch
数。

## 第四阶段：完整 block CUDA Graph 实验（已回退）

为验证上一阶段的最后一个结构性假设，原型将每层完整 block 捕获进 graph：
`HC-attn pre/norm -> 16×64 attention -> HC post -> HC-FFN pre/norm -> dynamic MoE -> HC post`。
固定 `M=1024, tile=64, slot=0, prefix=0`，43 层共享 graph pool；eager 与 graph
共用同一个 dynamic MoE workspace。

- 真实模型 capture 8.69 s，43 graphs，reserved 增量 1.506 GiB、allocated 增量
  1.311 GiB，捕获后 driver free 3.93 GiB。
- 固定 prompt 的 16-token greedy 输出仍逐项等于上面的 hard sequence，所有 logits
  finite；因此本次否决是性能结论，不是 correctness 失败。
- 在同一 daemon 同时保留完整 block graph 与旧 attention-only graph，预热后做 12 对
  交替 A/B：完整 block median `1.272965 s = 804.4 tok/s`，attention-only median
  `1.274406 s = 803.5 tok/s`，完整 block 仅快 **0.11%**，属于零收益。
- profiler 复核也一致：完整 block / attention-only 的 self CUDA 总时分别
  `1.238 s / 1.240 s`。主要 kernel 几乎不变：

```text
                              block graph   attention-only
iq2 gate/up dynamic             376.2 ms       375.7 ms
Q8 SoA dense                    282.2 ms       288.8 ms
iq2 down dynamic                231.7 ms       232.0 ms
Q8 SoA grouped                   54.6 ms        55.4 ms
attention shared kernels        ~49.7 ms       ~49.3 ms
```

结论：此前把 profiler 中 host/launch 汇总空洞直接换算成 `170-230 ms` E2E 收益是错误
外推；attention graph 之后，剩余 host 提交已被 GPU 工作覆盖，不在关键路径上。完整
block graph 多占约 1.5 GiB 显存、扩大固定形状特殊分支，却没有 E2E 收益，因此代码和
测试原型均已回退。

当前 `~760 tok/s` 的干净配对基线已经超过本轮同权重 llama.cpp `567.8 tok/s` 的实测
参考；同一高度预热 daemon 可到约 `804 tok/s`，但不把它替代冷净进程基线。要达到
`1000 tok/s`，从 `1.273 s` 仍需减少约 `249 ms`，而仅 routed MoE 就占约 `608 ms`。
下一阶段若继续，必须是能让 IQ2 gate/up/down **实际少做指令或改用更高吞吐格式**的
kernel/权重格式工程；再消 launch、再扩大 graph、或只融合仅 `2.8 ms` 的 SwiGLU
epilogue 都不可能补足差额。Q8 dominant-shape 调优仍可作为几十毫秒级边际实验，但
没有证据支持它单独达到 1000 tok/s。
