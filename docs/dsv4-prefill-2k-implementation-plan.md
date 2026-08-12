# DSV4 单卡 prefill >=2000 tok/s：历史失败合同

> **SUPERSEDED（2026-08-12）**：用户将发布目标调整为 `>=1000 tok/s`。当前实施合同是
> [`dsv4-prefill-1k-implementation-plan.md`](dsv4-prefill-1k-implementation-plan.md)。
> 本文保留 2K 路线的失败证据和决策过程，不再用于安排新开发。

> 状态：**Phase 2B 已关闭；row-W8A8 候选 Phase C0 已实测并判定失败（circular
> transcode 28ms gate+up vs 2.4ms 门禁，W8 常驻 6.4GB 超内存，L2 hit 23%）。
> §9 分支 (b)/(c) 待用户定夺：重新核算 6.5ms/layer 预算，或如实记录 2K 在当前
> 约束下无已知可实现路径。**
>
> 本文基线：`qwen-sm120-runtime main@882d209`，审计时与 `origin/main` 一致。
>
> 目标硬件：NVIDIA RTX PRO 6000 Blackwell Max-Q，SM120，96 GB，300 W；本机实测 L2 为 128 MiB。
>
> 目标模型：当前 DeepSeek-V4-Flash GGUF，routed experts 常驻格式为 IQ2_XS。

本文替代 2026-08-11 版执行顺序。旧版把下一步写成 two-plane / K-group
microkernel；最新代码和实测已经把这两条都关闭。历史证据保留在：

- [`../notes/2026-08-11-dsv4-phase2b-scale-amortized-falsified.md`](../notes/2026-08-11-dsv4-phase2b-scale-amortized-falsified.md)
- [`../notes/2026-08-11-dsv4-2bit-tc-native-infeasible.md`](../notes/2026-08-11-dsv4-2bit-tc-native-infeasible.md)
- [`../notes/2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md`](../notes/2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md)
- [`../notes/2026-08-12-dsv4-prefill-row-w8-replan-evidence.md`](../notes/2026-08-12-dsv4-prefill-row-w8-replan-evidence.md)

这仍然是一个以 `>=2000 tok/s` 为完成条件的计划，不把 100–500 tok/s 的小修补计作达标。
但本文也不把尚未测过的候选写成“已经可达”：Phase C0 是新的生死门，失败即说明当前
硬约束下没有剩余的已知本地实现路径。

## 1. 结论先行

### 1.1 最新代码意味着什么

截至 `882d209`：

- production `prefill()` 仍调用 chunk-major `_prefill_logits()`；
  `_prefill_superchunk_logits()` 仍是 dormant prototype；
- multi-row MoE 仍进入 `Dsv4MoE._route_expanded_prefill()`，以 route-M1 dp4a 执行；
- `iq2_mma16_tc` 已完成 K32 scale-amortized candidate、wrapper、build target 和 CUDA test，
  但没有接 production；
- K-group direct folding、two-plane、N_ROWS=64/128/256 均已实测失败；
- 当前服务真实 prefill 仍约 `130 tok/s`，没有 production 加速被合入。

因此，后续不能继续按旧版“再调一次 `iq2_mma16_tc` 然后接 superchunk”的顺序实施。

### 1.2 新的核心判断

失败的 global int8-codebook 路径把整层 widened weights 当成新的 HBM 流量。仅 routed
gate+up 的 W8 写入再读取就是 8 GiB/layer，down 另有 4 GiB/layer；按 1792 GB/s 峰值计算，
它们的理想下限已经分别约 4.8 ms 和 2.4 ms，未计 IQ2 source、activation 和输出，必然超过
`2.4/1.3 ms` 门禁。

新的候选不是整层 materialize，而是：

1. 对真实 1024-token routes 做 device-side expert grouping；
2. 每次只处理 `2–4` 个 expert；
3. 从 IQ2_XS 临时转码为标准 row-scaled INT8 weights；
4. 转码结果写入反复复用的固定 circular scratch；
5. 立即用 grouped W8A8 Tensor Core GEMM 消费，再覆盖同一 scratch 地址；
6. gate/up 后立即 SwiGLU；复用 scratch 做 down；最后 stable combine。

gate+up 的 `tile_E=4` W8 主体为 64 MiB，连同 scales、A8 和输出 staging 目标小于 72 MiB；
down 小于 40 MiB。它们能够容纳在本机实测的 128 MiB L2 中。这个设计只有在 scratch
producer→consumer 命中 L2、旧 dirty line 不形成 12 GiB/layer 的 HBM 写回时才成立。

这就是 Phase C0 必须直接测 `dram bytes written` 和 L2 hit rate、不能只测 kernel wall time
的原因。

### 1.3 固定实施顺序

```text
Phase C0  transient row-W8A8 representation + L2 residency proof
  -> Phase C1  IQ2 -> W8 circular-tile + grouped GEMM candidate
  -> Phase C2  device route/group + fused routed/shared MoE
  -> Phase D1  bounded layer-major superchunk 接 production
  -> Phase D2  SGLang-style DSV4 sparse attention / captured metadata / mHC fusion
  -> Phase E   full-model、4K/16K/128K fresh-process release gates
```

任何阶段未过 kill gate，不得提前接 backend，也不得用后续 overlap 掩盖当前单算子超预算。

## 2. 不可改变的产品约束

- 只有一张 RTX PRO 6000 Blackwell Max-Q；`TP=PP=EP=1`；
- 保留当前 DeepSeek-V4-Flash GGUF，IQ2_XS 是 routed experts 的唯一常驻权重；
- 不降低 top-k、层数、上下文长度或最终质量门禁；
- 不常驻第二份 model-sized W4/W8/BF16 routed weights；
- 2 slots × 128K 和 decode CUDA Graph 必须继续装入 96 GB；
- 无 prefix hit 的真实服务 prefill，4K、16K、128K median 均 `>=2000 tok/s`；
- p10 `>=1800 tok/s`；每个长度至少 5 次；
- end-to-end worst logits cosine `>=0.99`，greedy `39/39`；
- 所有性能比较必须有 bfdiag run record，并先通过 `bf diff`。

`256 MiB` 仍是 grouped MoE 的总 scratch 上限；新方案额外要求 W8 hot working set
`<=80 MiB`，为 L2 metadata、route buffers 和并发流量留余量。

## 3. 最新证据与证据边界

### 3.1 Phase 2B 已关闭

真实 GGUF、E=256、M_PAD=32、1024-token route distribution 的最终记录为：

| 路径 | gate+up | down | 结论 |
|---|---:|---:|---|
| exact IQ2 Tensor Core | 13.4–14.8 ms | — | 只保留为 oracle |
| K32 direct folding 最终版 | 6.93 ms | 6.92 ms | 分别超过 2.4/1.3 ms kill gate |
| nodecode 理想上限 | 3.21 ms | — | 即使不解码仍超过 gate+up 门禁 |
| two-plane global codebook | 10.21–10.57 ms | — | widened global traffic 导致 DRAM 饱和 |

Nsight 显示 tensor 指令占比很低，ALU decode 与 L1TEX/shared traffic 主导；K64 以上又被
质量门禁限制。继续做 I2F、launch bounds、N_ROWS 或同结构 SoA 小修，没有达到 2K 的空间。

注意：历史 note 中 two-plane 有 `10.21 ms` 与 `10.57 ms` 两个记录，且这些性能记录没有
bfdiag run id / `bf diff`。它们足以支持“远超 2.4 ms、当前分支关闭”的数量级结论，不能作为
新候选的精确 baseline。Phase C0 必须重新产生可比较 run record。

### 3.2 2026-08-12 表示预筛

可复现的新结果是本机真实 `blk.4`、expert 0、M=24、seed 20260812 的
row-scaled W8 + per-token A8：gate/up `cos_min=0.9999151/0.9999127`，传播到
gate+up→SwiGLU→down 后 `cos_min=0.9992452`。完整命令、shape 和输出保存在
[`../notes/2026-08-12-dsv4-prefill-row-w8-replan-evidence.md`](../notes/2026-08-12-dsv4-prefill-row-w8-replan-evidence.md)。

同轮还探索了 FP4/FP6/FP8 transient 表示，但当时没有保留可复现 operation/run record；
这些临时数字不作为本计划关闭或通过任何路线的证据。C0 的多层 sweep 会把 row-W8 与
需要的对照组一起做成正式 bfdiag 记录。

row-W8 的 gate/up 单 projection 已达到 `0.9999`；误差主要在 gate→SwiGLU→down 传播后累积。
因此不直接降低最终产品质量门禁，而是把内部验收改成分层门禁：

- gate/up projection 各自 `cos_min >=0.9999`；
- 完整 routed MoE 输出 `cos >=0.9990`；
- 完整 layer 输出 `cos >=0.9990`；
- 最终仍必须 full-model worst logits cosine `>=0.99`、greedy `39/39`。

若真实多层 calibration 或 full-model 失败，row-W8 路径直接关闭，不能以吞吐换质量上线。

### 3.3 当前自动化证据的缺口

- `tests/test_iq2_mma16_tc_kernel.py` 只覆盖 E=2、单 shape、gate/up，阈值仅 `0.99`；
- `grouped_moe_prefill` 测试只检查 finite/nonzero，不是 reference parity；
- GPU candidate 不在默认 CPU CI 中执行；
- 缺少真实 routed down/SwiGLU/reduce 的自动化质量 gate；
- 缺少 Phase 2B 的 bfdiag run records。

新阶段不能用“现有 tests 通过”代替上述缺口。

## 4. 新目标架构

```text
1024-token bounded superchunk
  -> embedding
  -> each layer
       -> mHC-attention pre/norm
       -> DSV4 sparse attention over fixed tiles
       -> mHC post + FFN pre/norm
       -> device top-k + stable expert grouping
       -> for expert_tile in fixed {2, 4}:
            IQ2 metadata scan -> per-output-row W8 scale
            IQ2 decode -> same-address circular W8 scratch
            grouped s8xs8->s32 GEMM -> scaled BF16 gate/up
            immediate SwiGLU
            reuse circular scratch for down
            grouped down GEMM -> stable route combine
       -> shared expert + mHC post
  -> final-token LM head only
```

### 4.1 为什么 row-W8 与失败的 K32 folding 不同

K32 folding 每 32 个 K 都需要独立 scale，MMA partial 之后仍要执行大量 FP32 scale/accumulate；
最新 Nsight 已证明这部分和 decode/L1TEX 一起卡住 tensor pipeline。

row-W8 为每个 output row 只生成一个 weight scale，activation 为每个 route row 只生成一个
scale。整个 K 可以交给标准 grouped INT8 GEMM，最终 epilogue 一次应用
`sA[m] * sB[n]`。它用略大的量化误差换掉 K32 facc 链，但不改变最终产品质量 gate。

### 4.2 为什么必须是 circular tile

每个 expert、每个 2048×4096 projection 的 W8 是 8 MiB。以 `tile_E=4`、M_PAD=32、
FP16 scales/BF16 outputs 计：

| working set | gate+up | down |
|---|---:|---:|
| W8 weights | 64 MiB | 32 MiB |
| per-output-row W scale | 32 KiB | 32 KiB |
| route-expanded A8 + A scale | 约 0.5 MiB | 约 0.25 MiB |
| INT32/BF16 output staging 上限 | 约 2 MiB | 约 2 MiB |
| route/descriptor/对齐余量 | 目标 <5.5 MiB | 目标 <5.7 MiB |
| hot working set 门禁 | **<72 MiB** | **<40 MiB** |

`tile_E=8` gate+up 的 W8 主体已经 128 MiB，会挤出其他 L2 working set，只作负对照，
不作 production 默认。

scratch 地址必须在每个 expert tile 上重复使用。若每个 expert 分配新地址，逻辑 12 GiB/layer
会重新变成 HBM 写流量，等价于已失败的 global codebook 路径。

### 4.3 权重转码

每个 IQ2 row 分两步：

1. 只扫描 block scale/meta plane，求该 row 的最大绝对值和 W8 scale；
2. 扫 code/sign plane，直接写 Tensor-Core 需要的 packed INT8 layout。

不能先解码 BF16/FP32 再量化；不能产生 full-layer W8；不能把 scratch 隐式缓存到 Python
对象中。转码、GEMM 和 epilogue 都必须接 caller-owned buffers，并在启动期 prewarm。

### 4.4 SparkInfer 落点

`/home/bot/project/sparkinfer@583e313` 已有 planned `b12x.moe.fused_moe` 接口：
`plan_weights -> prepare_weights -> plan -> bind -> run`，支持 caller-owned scratch、device route、
FC1/activation/FC2/combine 的 operator 边界。当前 recipe 有 NVFP4、MXFP4、W4A8、W6A8、
W4A16，没有 IQ2/W8A8。

production 方案是在该 fork 增加一个窄 recipe，而不是在 runtime 再复制一套通用 MoE：

```text
source_format = gguf_iq2_xs
quant_mode    = iq2_stream_w8a8_row
execution     = expert-grouped, tile_E={2,4}, fixed M buckets
storage       = resident IQ2 source + caller-owned circular W8 scratch
```

Phase C0 的后端选择固定，不留给实现者二次选型：

- 在本仓库新增独立 candidate；IQ2 transcode 只复用 `iq2_mma16.cu` 的 exact block-unpack
  逻辑，不复用 K32 folding candidate `iq2_mma16_tc.cu` 的 mainloop；
- W8A8 GEMM 使用已固定的 `/home/bot/project/cutlass-4.6.1` headers，从现有
  `runtime/kernels/fp8_w8a8_sm120.cu` 的 SM120 mainloop/scale-broadcast epilogue 派生
  signed-INT8×signed-INT8→INT32 版本；
- C0 只支持真实 gate/up/down shapes 和固定 M_PAD=32；每个 expert tile 以
  `GemmUniversalMode::kBatched` 单次发射，gate+up 的 batch 为 `2*tile_E`、down 为
  `tile_E`，不允许退化成每 expert 一个 launch；不先做通用 API，不改 b12x；
- C0 通过后，C1 才把同一实现迁入 SparkInfer `iq2_stream_w8a8_row` planned recipe。

这样 C0 验证的是具体 kernel family，而不是抽象的“某个成熟 W8A8 GEMM”。SparkInfer
迁移必须在独立 worktree 开发，不能直接改其繁忙主工作树。

## 5. SGLang / DSV4 最新参考边界

2026-08-12 已 fetch `/home/bot/project/sglang`，以下结论审计的是远端引用
`origin/main@c7c03ec53b`；工作树仍停在 `b296e1a503`，没有被切换。该远端版本的
`default_prefill_backend()` 在 CUDA 上返回 breakable，DSV4 backend 声明
`use_captured_forward_metadata_for_breakable_cuda_graph=True`。可迁移的是：

- breakable prefill graph 的固定 bucket 和 captured metadata 生命周期；
- DSV4 sparse prefill，在 SM120 使用 `flash_mla_with_kvcache_sm120`；
- SWA + extra attention、compressor/indexer metadata 的 operator 组织；
- device-side MoE route/dispatch/combine 和 fused FC1/activation/FC2 边界；
- 固定 page/layout contract：head dim 512、page size 256、DSV4 KV 584 B/token。

不能直接迁移的是：

- SGLang 的 DSV4 expert 路径以 FP4/FP8/NVFP4 为主，不读取 GGUF IQ2_XS；
- 它的 TP/DP/EP、MegaMoE 和多卡通信假设；
- 在没有本机显存验证前直接开启 prefill graph。当前 runtime 在 2 slots×128K load 后约
  90.3 GB，decode capture 后约 95.0 GB，graph pool 与 circular scratch 必须共同计价。

因此调度参考 SGLang，核心 IQ2→W8 转码仍需在本项目的 SparkInfer fork 自建。

## 6. 每层预算

1024 tokens 在 2000 tok/s 下是 512 ms，即 11.91 ms/layer：

| 模块 | p50 目标 |
|---|---:|
| router score/top-k | <=0.3 ms |
| route/group/gather + A8 quant | <=0.5 ms |
| IQ2 transient gate+up + W8A8 GEMM | <=2.4 ms |
| SwiGLU + down input quant | <=0.4 ms |
| IQ2 transient down + W8A8 GEMM | <=1.3 ms |
| stable combine | <=0.3 ms |
| shared expert | <=1.0 ms |
| MoE 余量 | >=0.3 ms |
| router+routed+shared MoE | <=6.5 ms |
| sparse attention | <=3.5 ms |
| 两段 mHC + norm | <=1.4 ms |
| full layer | <=11.91 ms |

上述 budget 是 kill gate，不是预测值。

## 7. 实施阶段

### Phase C0：representation 与 L2 生死门 — **已实测，判定失败**

2026-08-12 实测完成（见 `notes/2026-08-12-dsv4-prefill-row-w8-replan-evidence.md`）：
- 手写 s8×s8→s32 W8A8 GEMM（CUTLASS 4.6.1 SM120 无 s8 builder，TCGEN05 s8
  在 compute_120f 不可用，故手写 m16n8k16）验证正确，M=512 达 121 TFLOPS；
- IQ2→row-W8 transcode 两 pass 并行实现，数值与 oracle 逐行 1.0000 匹配；
- **端到端 tile_E=4 circular：0.220ms/tile × 64 = 14.1ms（gate alone），
  28.2ms gate+up，超 2.4ms 门禁 11.8x**。

**判定失败的三重约束**：
1. per-token transcode 主瓶颈（74%），已近 DRAM 下限，无数量级优化空间；
2. W8 常驻 6.4GB（gate+up+down × 256 experts）超内存，与 256MiB scratch
   和 2×128K KV 冲突；
3. GEMM L2 hit 23%（B 分片消费），DRAM-bound。

row-W8 与 Phase 2B int8-codebook 同构死结：放大驻留换零 decode，代价是
带宽/内存。**§9 分支 (b)/(c) 待定夺：重新核算预算或记录未证可达。**

### Phase C1：candidate kernel

Phase C0 通过后：

- 新增独立 IQ2→row-W8 transcode artifact；exact `iq2_mma16` 保留为 oracle；
- 新增 grouped W8A8 GEMM/epilogue，固定 M buckets `{16,32,48,64}`；
- gate/up dual projection 共用 A8；
- down 复用同一 circular scratch；
- 先验证独立 candidate，再增加 SparkInfer `iq2_stream_w8a8_row` recipe；
- 禁止运行期 `.item()`、动态 sort/empty/zeros、Python per-expert loop。

### Phase C2：完整 MoE operator

- device top-k、counts、offsets、stable grouping；
- route count 超 bucket 时拆多个 tile，不能截断；
- fused SwiGLU、down quant、stable combine；
- shared expert 接入同一预算；
- 完整 MoE p50 `<=6.5 ms`、p90 `<=7.0 ms`；
- output cosine `>=0.9990`，100 replay 无 fallback。

只有 C2 通过，才允许 production scheduler 接新 operator。

### Phase D1：bounded layer-major superchunk

将 dormant `_prefill_superchunk_logits()` 改成真实有界入口：

- 默认 1024 rows；tail buckets 64/128/256/512/1024；
- outer loop 逐 superchunk，layer 内 attention 仍按因果 tile 推进；
- absolute position、compressor/indexer、KV、prefix checkpoint 语义不变；
- 补齐 1/63/64/65/255/256/1023/1024/1025 parity；
- 覆盖 prefix full/partial hit、slot reset/reuse、2 slots；
- LM head 只算最后一个有效 token；循环内零分配。

### Phase D2：attention、mHC 与 prefill graph

- 先把 sparse attention 压到 `<=3.5 ms/layer`；
- 合并相邻 mHC post/pre、norm 和中间写回，达到 `<=1.4 ms/layer`；
- 移植 SGLang 的 captured metadata 生命周期，而不是复制其服务调度器；
- eager 正确性和显存通过后，再 A/B breakable prefill graph；
- scratch 必须在 graph capture 前预分配，并验证 2 slots×128K + decode graph 仍能加载；
- graph 若导致 OOM，只能缩减 graph capture pool/bucket，不能静默降低上下文或 slots。

### Phase E：全模型和发布

依次通过：

1. real 1024-token full-model `<=512 ms`，无 fallback；
2. worst logits cosine `>=0.99`、greedy `39/39`；
3. fresh-process 4K、16K、128K median `>=2000 tok/s`、p10 `>=1800 tok/s`；
4. long-context、prefix cache、reset/reuse、decode CUDA Graph 无回归；
5. 96 GB resident memory gate；
6. torch-free CI、full pytest、ruff、CUDA artifact verification 全绿。

## 8. 文件边界与提交顺序

| 阶段 | 主要所有权 | 禁止混入 |
|---|---|---|
| C0 | bfdiag reusable operation、candidate CUDA/Python、独立 tests | backend production wiring、一次性 benchmark 脚本 |
| C1 | SparkInfer worktree 下的新 fused-MoE recipe；本项目 adapter/build provenance | 改 decode M=1 路径、覆盖 exact oracle |
| C2 | `runtime/model/dsv4_model.py`、route/group scratch、trace | host sync、动态 allocation、旧 grouped prototype |
| D1 | `runtime/backends/dsv4.py`、prefix/checkpoint/slot tests | C2 未过门时提前设默认 |
| D2 | DSV4 attention/mHC kernels、captured metadata、memory tests | 在 eager parity 前开 graph/多流 |
| E | server recipe、prewarm、bfdiag fresh-process workloads、live docs | 用 warm daemon 代替 cold-prefill 验收 |

每阶段独立提交、独立回滚。SparkInfer 和本项目不得在同一个不可拆分提交里同时大改。

## 9. 测试与诊断合同

### 正确性

- 512 grid codes、128 sign patterns、全部 scale nibbles；
- M=1/8/16/24/32/48/64；K/N=4096/2048；
- row scale 极值、zero row、NaN/Inf guard；
- gate/up dual、SwiGLU clamp、down、router weight、top-6 stable combine；
- 多层 calibration，不允许只报 expert 0；
- 1–1025 boundary、4K/16K/128K、prefix、slot reset/reuse。

### 性能记录

不得只留下终端文本：

```bash
bf show <run-id>
bf diff <baseline-run-id> <candidate-run-id>
bf trace show <candidate-run-id>
```

外部 IQ2 source 必须是冷缓存/真实大足迹；transcode→GEMM 之间不得人为 flush，因为 L2
producer-consumer locality 正是被测设计。bfdiag trace 只记录 source bytes、scratch logical bytes、
tile_E、route histogram、fallback reason 和 Nsight report linkage；实际 DRAM read/write 与 L2 hit
的 authoritative source 只认 `.ncu-rep`。

### 合并门禁

```bash
make verify-iq2-mma16
~/.venvs/vllm/bin/python -m pytest -q tests/test_iq2_mma16_kernel.py
ruff check .
/tmp/ci-sim/bin/python -m pytest -q
~/.venvs/vllm/bin/python -m pytest -q
git diff --check
```

C0/C1 必须新增自己的 build target 和严格质量 test；不得继续复用当前阈值仅 0.99 的
`test_iq2_mma16_tc_kernel.py` 作为完成证据。

## 10. 明确拒绝项

| 方案 | 决策 |
|---|---|
| 继续 route-M1 dp4a 小修 | 拒绝；只有约 1.2× 空间，离目标数量级太远 |
| 继续 K32 folding / two-plane / N_ROWS | 拒绝；最新实测均已过 kill gate 失败 |
| full-layer/global W8 materialize | 拒绝；12 GiB/layer 逻辑 W8 流量的 HBM 下限已超预算 |
| resident model-sized W8/W4/BF16 副本 | 拒绝；96 GB 无法与当前模型、KV、decode graph 共存 |
| NVFP4/MXFP4 transient | 不作主线；非正式预筛较差，若重新提案必须先补可复现 C0 级证据 |
| 直接复制 SGLang FP4 MegaMoE | 拒绝；它不能读取 IQ2_XS，硬件/并行假设也不同 |
| 直接接当前 `grouped_moe_prefill` | 拒绝；host sync、动态 allocation、弱测试 |
| 先接 superchunk 再看性能 | 拒绝；旧 MoE operator 仍会成为 route-M1 瓶颈 |
| 多 GPU / 远程 prefill | 越界；用户只有一张卡 |
| 以 100–500 tok/s 宣布完成 | 拒绝；发布门禁固定为 2000 tok/s |

## 11. 下一个可提交里程碑

**Phase C0 已实测并判定失败**（2026-08-12，见 §7）。C0 的第 8 项"二选一结论"
已落到**未通过分支**：

- 手写 s8 W8A8 GEMM 正确（cos 0.999998）且 M=512 达 121 TFLOPS；
- IQ2→row-W8 transcode 正确（逐行 1.0000 匹配）；
- 但端到端 circular tile_E=4 实测 28.2ms gate+up（超 2.4ms 门禁 11.8x），
  W8 常驻 6.4GB 超内存，GEMM L2 hit 23%。

在 C0 判定后，不再接受新的 IQ2 microkernel 或 W8/W4 transient 猜测分支。
下一步是 **§9 分支 (b)/(c) 的用户定夺**：
- (b) 重新核算 6.5ms/layer MoE 预算（接受 gate+up ~28ms 现实）；
- (c) 如实记录"当前 IQ2_XS + 单 SM120 + 质量/容量约束下 2K 无已知可实现路径"，
  把 2K 目标降级或改为服务层决策。

## 12. 外部依据

- [SGLang DeepSeek-V4 model execution](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4.py)
- [SGLang DSV4 attention backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/deepseek_v4_backend.py)
- [SGLang prefill CUDA graph runner](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py)
- [NVIDIA RTX PRO 6000 Blackwell Max-Q 规格](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/)
- [CUTLASS integer sub-byte types](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/integer_subbyte.h)
