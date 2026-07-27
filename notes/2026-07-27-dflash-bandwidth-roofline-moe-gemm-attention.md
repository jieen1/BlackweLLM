# DFlash 64K verify replay:三大耗时分量的显存带宽打满率(2026-07-27)

## 结论先行

用同一套方法论(真实内存带宽 ceiling 实测 + kernel 耗时反推打满率,冷缓存条件)检查了
verify replay(37.5ms/replay)里三个最大的耗时分量:

| 分量 | 占比 | 打满率 | 结论 |
|---|---:|---:|---|
| MoE | 58.5% | ~100.7% | 已经打满,没有空间 |
| dense GEMM(QKV/O proj) | 16.8% | 86-95% | 接近打满,~5-15%潜在空间 |
| attention(12 层 full-attn 为主) | 11.9% | **~37%** | **有真实空间,且根因已知** |

**attention 的 headroom 不是"kernel 实现差",是一个此前已经调研过、有明确技术方案、
但因为"当时另一个任务用别的办法已经达标所以没做"而搁置的已知项:Laguna 的 kernel 特化
要求 `page_size==128`,我们生产用 `block_size=64`,所以 decode/extend/verify 全部三条
路径都拿不到任何 Laguna 专用特化,只能走通用 kernel。**

## 方法论

沿用 MoE 那次已验证的套路(见另一篇笔记):

1. 实测这张卡(RTX PRO 6000 Blackwell **Max-Q**)真实可达显存带宽,不查文档规格:
   `torch.empty` 2GB uint8 → `copy_()`,稳态测得 **≈1300 GB/s**。
2. 对每个 kernel,用真实 shape 算出理论上必须搬运的字节数(权重为主,忽略远小于权重的
   激活/输出),除以实测耗时,得到"如果是纯显存带宽瓶颈,应该对应多少 GB/s"。
3. **关键:必须冷缓存**——这张卡 L2 有 128MiB,如果测试脚本在一个紧凑循环里重复用
   同一个权重张量,权重会常驻 L2,测出的数字会被 L2 带宽污染、明显偏高(不代表真实
   48 层各自不同权重、每层都是冷 HBM 读取的生产场景)。第一次测 dense GEMM 时就踩了
   这个坑(不 flush L2 时算出"1122GB/s"看着不错,flush 后其实是更高的表观值,说明
   之前没 flush 时数字被污染更严重——本笔记最终数字均为 `--flush-l2`/手动 L2 flush
   后的冷缓存结果)。

## MoE(58.5%,另一篇笔记已发,这里汇总）

`sparkinfer/benchmarks/benchmark_moe.py --model-profile laguna-s21-shape`,M=16,
`--no-flush-l2`(模拟真实连续多层调用的缓存状态,不是完全冷启动但更接近生产):
472.4us/call,和真实模型 profiler 测的 458.9-467.1us/call 几乎完全吻合。131个命中
expert × 4.5MiB/expert ÷ 472.4us ≈ **1308.6 GB/s,打满率 ~100.7%**。已经没有继续在
kernel 内部抠 tile/fusion 的空间(除非能减少搬运字节数——见下面"和阶段2/3的联系"）。

## dense GEMM(16.8%,QKV/O 投影,4 个真实形状)

自建脚本(`/tmp/probe_dense_gemm_kernel.py`,用 config.json 里的真实 head 数:full-attn
48-head、SWA 72-head、8 kv-head、head_dim=128),`F.linear` 直接调,不经过 vLLM 的
`QKVParallelLinear`/`RowParallelLinear`——**结果证明这不是 vLLM wrapper 类的问题,裸
`F.linear` 在这几个具体 shape 下选的也是同一个 `cutlass_80_wmma_tensorop_*`(SM80)
kernel**,推翻了 `STATUS_speed_optimization_0726.md` 里"因为走了 vLLM 特殊路径才选错
kernel"的归因。

冷缓存(手动 L2 flush,256MiB×2 filler)结果:

| 形状 | 权重大小 | 耗时 | 打满率 |
|---|---:|---:|---:|
| full_attn_qkv (M=16,K=3072,N=8192) | 48.0MiB | 44.86us | 1122 GB/s (86%) |
| swa_qkv (M=16,K=3072,N=11264) | 66.0MiB | 59.39us | 1165 GB/s (90%) |
| full_attn_o (M=16,K=6144,N=3072) | 36.0MiB | 31.91us | 1183 GB/s (91%) |
| swa_o (M=16,K=9216,N=3072) | 54.0MiB | 45.62us | 1241 GB/s (95%) |

**同一件事也纠正了 `STATUS_speed_optimization_0726.md` 的"2.4x slower"结论**:那篇
笔记的"isolated F.linear test: 19.9μs"大概率也是没 flush L2 的产物(我自己第一次不
flush L2 跑出来的 `full_attn_qkv` 恰好也是 19.39us,几乎一样),不是"vLLM 路径比裸
`F.linear` 慢 2.4 倍",而是"没 flush L2 的孤立测试比真实多层冷缓存场景快 2.4 倍,这个
差距是测量方法的假象,不是可以白拿的优化空间"。**dense GEMM 已经接近打满(86-95%），
即使换成 SM120 原生 kernel,理论收益上限也就 5-15%,不是 2.4x。**

## attention(11.9%,重点,有真实空间)

`sparkinfer/benchmarks/benchmark_paged_attention.py --mode legacy-matrix --paged-mode
verify --window-left -1 --q-seqlens 16 --cache-seqlens 65536 --page-size 64
--q-heads 48 --kv-heads 8 --head-dim 128 --kv-dtype fp8_e4m3fn --flush-l2`(full-attention
层,12层里的代表,权重占比小、以 KV 读取为主）:

- 实测:**279.5us/call**
- KV 字节:65536(kv_len)×8(kv_heads)×128(head_dim)×2(K+V)×1(fp8)=128MiB
- 打满率:128MiB÷279.5us ≈ **480 GB/s,只有实测带宽上限的 ~37%**

**根因已经在今天早些时候的另一项工作里查清楚、写进了 notes
(`notes/2026-07-27-verify-cg-mode-fix-and-block-size-eval.md` 任务B部分)——只是当时
因为验证 CG 追平 eager 的目标已经靠 mode="extend"→"verify" 单独达成,这项"page_size
迁移"被判定"可行但非必要"而搁置,没有因为这次的带宽数据重新评估优先级。**

要点摘录(详见那篇笔记):

- sparkinfer 的 Laguna 专用 kernel 特化(`traits.py` 的
  `select_paged_forward_traits_from_plan`)全部 5 条分支都要求 `page_size==128` 且
  `kv_dtype==FP8`——这就是为我们当前生产配置(FP8 KV)量身定做的特化,只是要求
  page_size=128,不是我们现在用的 64。
- 当前 `block_size=64` 下,**decode/extend/verify 三条路径全部吃不到任何 Laguna 专用
  特化,全部落回通用路径**——这很可能就是 attention 打满率只有 37% 的直接原因。
- 迁移到 128 已有可行性证据(sparkinfer generic planner 本来就同时支持 64/128,
  block_size=64 是我们自己代码加的硬限制,不是 sparkinfer 强加的;核心寻址逻辑已经
  符号化,不需要重写),但**成本和风险也已经写清楚**:76 个 benchmark/测试文件量级的
  机械改动 + KV cache 显存布局改变需要专门验证(这类改动历史上出过"CUDA Graph 地址
  失效导致接受率暴跌到0.13%"的真实事故,`STATUS_dflash_acceptance.md` 有记录)。

## 和阶段2/3(并发)的联系

MoE 那篇笔记提到的"多请求并发合并 verify 调用,提高 expert 权重摊销"的思路,同样适用于
attention:如果 2/4 并发把多个请求的 verify 批到一起,KV 读取本身不会减少(每个请求有
自己的 KV),但如果同一个 kernel launch 能服务更多 query(不同请求但共享 kernel 调用
开销),对当前"打满率只有37%"这种可能有固定开销成分的场景也可能有正面作用——不过这个
推测比 MoE 那条(有明确的字节摊销数学)弱,没有实测验证,只是记录一个后续可以验证的假设。

## 建议(留给协调者/用户决策,不在这次任务里直接做)

`page_size 64→128` 迁移是三个分量里**唯一被量化证明有实质 headroom(37%→接近100%,
attention 这部分理论上限约 2.7x,对应 12 层 full-attn 从 3.35ms 降到约 1.24ms,
round_total 44.16ms 里省下约 2.1ms,对应吞吐提升约 5%)**的一项,但也是三项里成本/
风险最高、之前被判定"非必要"而主动搁置的一项。是否现在投入,建议结合"阶段1 已确认
和 vLLM 打平"(见另一篇笔记)一起评估优先级。

## 代码/产物

- `/tmp/claude-1002/.../scratchpad/probe_dense_gemm_kernel.py`(scratchpad,未入库,
  纯诊断脚本,内容已完整记录在本笔记)。
- attention/MoE 测试直接用 sparkinfer 自带的 `benchmarks/benchmark_paged_attention.py`
  / `benchmarks/benchmark_moe.py`,没有修改 sparkinfer 代码,只是调用现成工具做只读
  profiling。
