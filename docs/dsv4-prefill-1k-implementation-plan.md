# DSV4 单卡 prefill >=1000 tok/s：设计与实施合同

> 日期：2026-08-12
>
> 状态：**Phase B 预检失败（2026-08-12）；CTA-local row-W8（§9 reopen）也已实测
> 判定失败——性能可行（eff_pad=32 拆分 core 11.56ms）但整行 scale 使 down/
> routed cos 0.99965 < 0.9990 门禁。K32（数值 0.99990 ✓ 但 core 17.06ms ✗）与
> CTA-local（性能 ✓ 但数值 ✗）各缺一半，均无完整已证路径。**
> 证据见 `../notes/2026-08-12-dsv4-prefill-1k-confirmation.md`。**
>
> 代码基线：`qwen-sm120-runtime main@6be316f`，与 `origin/main` 一致。
>
> 替代：[`dsv4-prefill-2k-implementation-plan.md`](dsv4-prefill-2k-implementation-plan.md)
> 的发布目标。旧文档保留 2K 失败过程，不再指导当前实施。

## 1. 结果定义

### 1.1 硬约束

- 一张 NVIDIA RTX PRO 6000 Blackwell Max-Q，SM120，96 GB，300 W；
- `TP=PP=EP=1`，不引入第二张卡或远程 prefill；
- 继续服务当前 DeepSeek-V4-Flash GGUF，routed experts 常驻格式仍为 IQ2_XS；
- 不降低 top-k、层数、上下文、模型语义或质量门禁；
- 不常驻 model-sized W4/W8/BF16 第二份 routed weights；
- 2 slots x 128K、prefix cache 和 decode CUDA Graph 继续可用。

### 1.2 发布性能门禁

服务器加载、JIT 和 prewarm 完成后，在无 prefix hit 的真实服务路径上：

- 4K、16K、128K prompt 的 prefill median 均 `>=1000 tok/s`；
- p10 均 `>=900 tok/s`；每个长度至少 10 次；
- 每个长度使用 fresh process 固定 load-time 配置；ready 后用 reset/no-hit 请求采样；
- 1024-token full-model prefill 的 10 次 median `<=990 ms`；
- final logits worst cosine `>=0.99`，greedy `39/39`；
- 2 slots x 128K、prefix restore、slot reset/reuse、decode graph capture/replay 无回归。

`990 ms` 是阶段门禁，不是对 `1000 tok/s` 的重新定义；它为服务调度和长度尾块保留约
3.4% 余量。最终是否完成只认 4K/16K/128K 的服务级门禁。

### 1.3 不算完成

- 130 -> 200/500/700 tok/s 的局部提升；
- llama.cpp、microbenchmark 或单层达到 1K，但本服务未达到；
- prefix full hit 的“等效 tok/s”；
- warm daemon 代替 fresh-process 冷 prefill 验收；
- 只报 gate/up 与 down kernel，不包含 route、shared expert、combine；
- 通过放松质量、上下文、slots 或 decode graph 获得速度。

## 2. 当前事实与为什么目标变化后路线会变

### 2.1 生产路径仍是旧架构

- `prefill()` 仍调用 chunk-major `_prefill_logits()`：
  [`runtime/backends/dsv4.py:587`](../runtime/backends/dsv4.py) 和
  [`runtime/backends/dsv4.py:861`](../runtime/backends/dsv4.py)；
- 每个 64/128-row chunk 都完整经过 43 层：
  [`runtime/backends/dsv4.py:614`](../runtime/backends/dsv4.py)；
- 多 row MoE 仍展开 top-6 routes，以 indexed M=1 dp4a 执行：
  [`runtime/model/dsv4_model.py:481`](../runtime/model/dsv4_model.py)；
- `_prefill_superchunk_logits()` 只是 whole-prompt prototype，包含 `outs` list 和
  `torch.cat`，没有接 production：
  [`runtime/backends/dsv4.py:682`](../runtime/backends/dsv4.py)。

当前真实服务约 `130 tok/s`。同卡同模型的 llama.cpp IQ2_XS 参考约为
`706 tok/s @1024`、`680 tok/s @4096`；它说明现 runtime 有数倍工程空间，但不是
本项目的质量或实现 oracle。

### 2.2 1K 的预算与 2K 不同

DSV4 固定为 43 层、256 routed experts、top-6：
[`runtime/model/dsv4_config.py:30`](../runtime/model/dsv4_config.py)、
[`runtime/model/dsv4_config.py:73`](../runtime/model/dsv4_config.py)、
[`runtime/model/dsv4_config.py:75`](../runtime/model/dsv4_config.py)。

```text
1024 tokens / 1000 tok/s = 1024 ms
1024 ms / 43 layers      = 23.81 ms/layer theoretical
engineering gate         = 22.50 ms/layer
22.50 ms * 43            = 967.5 ms
full-model stage gate    = 990 ms
```

旧 2K 计划每层只有 `11.91 ms`，所以 K32 candidate 的 gate+up `6.93 ms` + down
`6.92 ms` 必然失败。目标改为 1K 后，同一候选第一次进入 `22.50 ms/layer` 的
可行区。因此这不是把旧失败改名，而是重新核算后改变候选排序。

**2026-08-12 实测修正**（见 `notes/2026-08-12-dsv4-prefill-1k-confirmation.md`）：
- 文档引用的 `13.85 ms`（gate+up 6.93 + down 6.92）基于 eff_pad=32 + 复制
  权重（32 unique × 8），**低估真实成本**；
- 真实 256 unique experts + eff_pad=48：gate+up **8.81 ms**、single-output
  down **8.25 ms**、**K32 kernel core 17.06 ms**（已占满 MoE 17 ms 预算）；
- 完整 MoE layer（无 shared/attention）实测 **21.97 ms**；
- single-output down 已落地（11.45→5.73 ms @ eff_pad=32，~2x），但不足以
  弥合缺口。**K32 all-in gate 已触发失败**，见 §7 Phase B。

### 2.3 每层预算

| 模块 | p50 门禁 | p90 门禁 |
|---|---:|---:|
| router + device group/gather + A8 quant | 含在 MoE 内 | 含在 MoE 内 |
| K32 gate/up + SwiGLU + down + stable combine + shared expert | `<=17.0 ms` | `<=18.0 ms` |
| tiled sparse attention | `<=3.5 ms` | `<=3.8 ms` |
| 两段 mHC + norm + 其余 glue | `<=2.0 ms` | `<=2.2 ms` |
| 完整 layer | `<=22.5 ms` | `<=23.5 ms` |

不能拿一个模块的余量掩盖另一个模块未测；完整 layer 和 full-model 都必须独立通过。

## 3. RALPLAN-DR 决策摘要

### 原则

1. 单卡容量、质量和服务语义是硬约束，速度不能靠删功能换取。
2. 以完整 operator/full-model 计账，不再用单 kernel 数字宣称可达。
3. 优先完成已有数值依据的 IQ2 路线，不再并行发散表示实验。
4. correctness gate 在 production wiring 之前。
5. 每个阶段有数字 kill gate；失败就停止该分支并保留证据。

### 决策驱动

1. 1K 将每层预算从 11.91 ms 提高到 23.81 ms，K32 core 不再被算术排除。
2. global/circular row-W8 已被 transcode、resident memory 和 L2 三重证据关闭。
3. 当前 83% chunk 时间在 MoE，必须先完成 grouped MoE，再接 superchunk。

### 选项

| 选项 | 决策 | 原因 |
|---|---|---|
| K32 grouped IQ2 TC + bounded superchunk | **主线** | K32 数值已预筛通过；13.85 ms core 可进入 1K 预算；不增加 resident 格式 |
| CTA-local decode-to-shared row-W8 | **关闭状态下的条件研究** | 临时 6.42/6.35 ms 使用 placeholder scale，非 correctness-complete；只在 K32 all-in 失败后允许重新立项 |
| global/circular row-W8 | **永久拒绝** | gate+up 28.2 ms、W8 resident 6.4 GB、consumer L2 hit 23% |
| llama.cpp 移植 | **只作性能参照** | 约 680-706 tok/s 未达目标，且不是本项目语义 oracle |
| native FP4 / 多卡 | **当前越界** | resident capacity 或用户硬件约束不满足 |

## 4. 目标架构

```text
prompt suffix
  -> outer bounded superchunks: 1024, tail split at deepest absolute 256 boundary
  -> for each completed segment
       embed segment
       for layer in 0..42
         mHC-attn pre + norm
         causal attention in fixed 64/128-row tiles
         mHC post + mHC-ffn pre + norm
         device top-k and stable expert grouping
         grouped K32 IQ2 -> shared/register INT8 -> mma.sync
           gate/up -> fused SwiGLU -> down -> stable combine
         shared Q8 expert
         mHC post
       final-token head only when segment is final
       capture deepest retained prefix checkpoint when absolute end % 256 == 0
```

关键点：这是**有界的 1024-token outer superchunk**，不是 dormant prototype 的
whole-prompt layer-major 执行。attention 内部仍按因果 tile 推进，所以 MLA scratch 不随
完整 prompt 长度增长。

## 5. K32 complete operator 合同

### 5.1 必须包含的工作

完整 MoE gate 一次计入：

- router/top-k；
- device counts、offsets、固定 bucket grouping；
- activation gather 和 per-K32 A8 quant；
- K32 gate/up；
- clamp + SwiGLU；
- down activation quant 和 K32 down；
- router weight、stable combine；
- shared Q8 expert。

bucket 固定为 `{16, 32, 48, 64}`。expert route count 超 64 时拆分，不能截断；空 expert
不引发 host decision。运行区间禁止 `.item()`、D2H sync、动态 `sort/empty/zeros`、JIT 和
Python per-expert loop。当前不合格 helper 的具体问题在
[`runtime/kernels/iq2_mma16.py:226`](../runtime/kernels/iq2_mma16.py)、
[`runtime/kernels/iq2_mma16.py:237`](../runtime/kernels/iq2_mma16.py) 和
[`runtime/kernels/iq2_mma16.py:250`](../runtime/kernels/iq2_mma16.py)。

### 5.2 Workspace ownership

- 在 `DeepseekV4Backend.__init__` 中、shared MLA scratch 之后分配一个 backend-owned
  `Dsv4PrefillMoEWorkspace`，总量 `<=256 MiB`；
- 通过显式 `Dsv4MoE.bind_prefill_workspace(workspace_views)` 绑定给每个 block；43 层按顺序
  复用同一 arena，不为每层复制；
- binding 只供 multi-row prefill/superchunk；`forward_decode_batch()` 和 M=1 decode 永远
  不读取该 workspace；
- backend 构造完成后 server 才调用 `capture_decode_cuda_graph()`：
  [`server/engine.py:841`](../server/engine.py) 到
  [`server/engine.py:863`](../server/engine.py)；因此 workspace 在 graph capture 前固定；
- ready 后不增长；overflow 只能拆固定 bucket。

SparkInfer 已有 caller-owned scratch 的 `plan -> bind -> run` 与 cache clear API：
`/home/bot/project/sparkinfer/b12x/moe/fused_moe/api.py:43-87`。外部依赖改动只包括缺失的
`source_format=gguf_iq2_xs`、`quant_mode=iq2_k32` lowering 和对应 IQ2 kernel；不重复开发
scratch API。local workspace ownership/binding 仍在本仓库。

### 5.3 第一生死门

在保留的真实 1024-token route histogram 上：

- 10 warmups + 200 replays；all-in MoE p50 `<=17.0 ms`、p90 `<=18.0 ms`；
- routed MoE output cosine `>=0.9990`；完整 layer output cosine `>=0.9990`；
- 100 replay 内无 allocation、host sync、JIT 或 fallback；
- workspace `<=256 MiB`；
- 记录 bfdiag run、`bf diff`、trace、NCU 与 NSYS artifact SHA256。

任一项失败，不接 backend，不先做 superchunk 掩盖失败。

## 6. bounded superchunk 与 prefix 合同

### 6.1 执行边界

- 主 bucket 为 1024；尾块 bucket 为 `64/128/256/512/1024`；不足 64 使用 mask；
- 每个 outer segment 完整跑完 43 层，才进入下一个 segment；
- attention output 写 caller-owned buffer，禁止 dormant prototype 的 list + `torch.cat`；
- absolute position、compressor/indexer、KV 和 final-token LM head 语义保持不变；
- fast path 不再走 `_prefill_logits()` 的 chunk-major 43-layer loop，也不再走
  `_route_expanded_prefill()` 的 route-M1 dp4a。

### 6.2 prefix 是单一最深 checkpoint，不是 ladder

当前每个 slot 只保留一个最深的 256-aligned recurrent/window checkpoint：
[`runtime/backends/dsv4.py:339`](../runtime/backends/dsv4.py) 和
[`runtime/backends/dsv4.py:371`](../runtime/backends/dsv4.py)。新路径必须保持这一语义：

- 完成 1024 segment 后在 1024 capture，它覆盖较浅深度；
- reset 后若新请求只共享该 cached 1024 的 256/512/768，仍是 state miss；
- 最后一个 partial outer block 在最深 absolute 256 boundary 拆段：先完整跑完该段的 43 层并
  capture，再跑 `<256` tail；
- 不在 layer-major segment 半完成时 capture；
- 不在本项目中顺手增加 checkpoint ladder。

新增测试必须覆盖“prefill 1024 -> reset -> shorter prefix miss”和 partial tail 的
effective hit、anchor、recurrent/window state 与旧路径一致。

## 7. 实施阶段与 kill gates

### Phase A：证据重置

不改 production dispatch，新增可复用 bfdiag operation：

- 固化真实 1024-token ids 和逐层 route histogram；
- 重新记录当前 130 tok/s baseline、K32 kernel pair、当前 32 ms grouped prototype；
- 冷 source footprint；先 `bf diff`，再 profiler；
- 内存快照：weights settled、prefill workspace allocated、decode graph captured、首次 128K
  prefill completed。

输出必须是 run records，不是 `/tmp` 终端数字。

### Phase B：K32 complete MoE

- 强化 `iq2_mma16_tc` 数值测试；
- 完成 device grouping、fixed buckets、caller-owned views、stable combine、shared expert；
- 接 SparkInfer 的现有 planned/bind surface，只补 IQ2 lowering/kernel；
- 达到 §5.3 all-in gate。

**2026-08-12 预检失败**：真实 256 experts + eff_pad=48 下，仅 gate+up（8.81 ms）
+ single-output down（8.25 ms）已 **17.06 ms**，占满 MoE 17 ms 预算；完整 MoE
layer（无 shared/attention）21.97 ms。按评审"不能进入 17-18 ms 应立即判失败"
触发。失败处理：冻结 trace/profile，停止 K32。只有此时才允许按 §9 CTA-local
reopen contract 重新评审，不自动切换。

### Phase C：bounded superchunk correctness

- 新建 bounded production entry，不直接调用 dormant whole-prompt prototype；
- 接入 Phase B operator；
- 实现 §6 的 single-deepest prefix 语义；
- 通过 `1/63/64/65/255/256/257/511/512/768/1023/1024/1025` parity；
- 完整 layer p50 `<=22.5 ms`、p90 `<=23.5 ms`；
- 1024 full-model 10-run median `<=990 ms`。

### Phase D：非 MoE 收口

只在 Phase C profile 指向时执行：

- sparse attention `<=3.5 ms/layer`；
- mHC + norm + glue `<=2.0 ms/layer`；
- 移植 SGLang 可用的 fixed metadata 生命周期和 SM120 sparse attention 组织；
- prefill CUDA Graph 不是默认要求。只有 NSYS 证明 launch/host gap 显著，并且 graph pool 在
  2 slots x 128K memory gate 下装得下，才启用 breakable prefill graph。

### Phase E：服务发布

依次执行 fresh process 4K、16K、128K；每个长度至少 10 次无 hit/reset 请求，报告
median/p10/p90。任何长度 median `<1000 tok/s` 或 p10 `<900 tok/s` 都未完成。

同时通过质量、prefix、slot、long-context、decode graph 和 96 GB resident gate。

## 8. 测试与验证

### GPU/operator

- IQ2 512 grid codes、128 sign patterns、全部 scale nibble；
- M buckets `16/32/48/64` 和 overflow split；
- gate/up/down、clamp、SwiGLU、router weight、stable grouping/combine；
- real top-6 multi-expert、多层 calibration；
- routed cosine `>=0.9990`、layer cosine `>=0.9990`；
- `tests/test_dsv4_moe.py` 证明 prefill 使用 bound workspace，而
  `forward_decode_batch()` 与 M=1 decode 完全不访问它；
- 不再接受现有 E=2、gate/up-only、cos `>=0.99` 测试作为完成证据：
  [`tests/test_iq2_mma16_tc_kernel.py:38`](../tests/test_iq2_mma16_tc_kernel.py)。

### backend/state

- boundary lengths：`1/63/64/65/255/256/257/511/512/768/1023/1024/1025`；
- prefix full/partial hit、1024 后 shorter-prefix miss、cross-slot restore；
- 2 slots、reset/reuse、异常 rollback；
- decode B=1/2/4 eager/graph parity；
- workspace allocated before graph capture and unchanged after ready。

### release

```bash
ruff check .
/tmp/ci-sim/bin/python -m pytest -q
~/.venvs/vllm/bin/python -m pytest -q
make verify-iq2-mma16
git diff --check
```

GPU/服务结果还必须有：

```bash
bf show <run-id>
bf diff <baseline-run-id> <candidate-run-id>
bf trace show <candidate-run-id>
```

## 9. CTA-local row-W8 条件重开合同

global/circular row-W8 永久关闭。CTA-local 也不是当前 implementation phase。只有 K32 all-in
MoE 未过 §5.3 时，新的设计同时满足以下条件，才允许重新进入规划：

- single fused decode-to-shared + MMA kernel，无独立 per-token transcode pass；
- 不写 global W8，不存在 logical W8 DRAM write stream；
- resident metadata 增量 `<=64 MiB`，不产生 6.4 GB W8 cache；
- 不依赖已失败的 23% L2 W8 consumer reuse；
- real row scales、A8、epilogue 和 real layout correctness complete；
- real-route gate+up + down core `<=14 ms`；
- NCU 直接证明 DRAM write、L2、SM/instruction counters，而不是从 wall time 推断。

当前 `/tmp` probe 的 6.42/6.35 ms 只证明这种组织值得保留为研究线索，不满足重开条件。

**2026-08-12 实测判定**：CTA-local row-W8 已按此合同实测——
- 性能：eff_pad=32（route 拆分）core 11.56 ms 达标，但 eff_pad=48（真实）15.47 ms；
- **数值失败**：整行 scale 使 down cos 0.99965、routed MoE out cos 0.99965，
  <0.9990 门禁（prescreen 实测，row-scale 是表示固有属性）；
- Phase 1 整行 scale 的 smem atomicMax 归约需额外正确性工作。

**CTA-local 因数值不达标排除**。K32 因性能不达标（17.06 ms）排除。两者互补
但无完整已证路径。

## 10. ADR

### Decision

以 K32 grouped IQ2 Tensor-Core complete MoE + bounded 1024 superchunk 作为 DSV4 单卡
prefill >=1K 的唯一实施主线。

### Drivers

- 1K 的 22.5 ms engineering layer budget 能容纳已测 13.85 ms K32 core；
- K32 是已有数值依据、且不增加 resident weight format 的候选；
- global/circular row-W8 已被结构性证据关闭；
- 当前 MoE 占比最高，必须先做 complete operator gate。

### Alternatives considered

- CTA-local row-W8：保留严格条件研究门，不作为 fallback 承诺；
- global/circular W8：拒绝；
- llama.cpp 路线：只作性能参照；
- native FP4、多 GPU：违反当前容量/硬件合同。

### Why chosen

它是唯一同时满足单卡 resident IQ2、已有质量证据、且在 1K 数字预算内没有被物理或实测
排除的路线。

### Consequences

- 余量窄：13.85 ms kernel core 之外只剩约 3.15 ms 给 MoE glue/shared expert；
- 第一道 all-in gate 仍可能杀死整个主线；
- superchunk 必须等待 MoE gate，避免把已知慢 operator 接入 production；
- 不在本轮扩展 prefix checkpoint ladder。

### Follow-ups

- Phase B 失败：按 §9 重新评审 CTA-local；没有自动成功路线；
- Phase C 只差非 MoE：进入 Phase D；
- service 与 1024 microbench 差距超过 10%：先用 trace 归因，再决定 graph/metadata 工作。

## 11. 执行人员与验证路径

可用角色：`executor`（kernel/runtime 实现）、`test-engineer`（GPU/state gates）、
`performance-reviewer`（NCU/NSYS）、`verifier`（release evidence）、`architect`（阶段复核）。

- `$ralph`：一个 `executor` 顺序持有 Phase A -> B -> C；每个 kill gate 后由 `verifier`
  独立复核。适合减少跨仓库冲突。
- `$team`：仅在 Phase B 已过后启用；runtime/superchunk、state tests、profiling 三条 lane 可并行，
  SparkInfer kernel lane 必须有独立 worktree。

建议 team 启动提示：

```text
$team 4:executor implement docs/dsv4-prefill-1k-implementation-plan.md from the first open phase;
stop at every kill gate and preserve bfdiag/Nsight evidence
```

team shutdown 前必须证明当前阶段的 correctness、performance、memory 和无 fallback；最终仍由
独立 `verifier` 运行 Phase E，不以 worker 自报完成替代发布门禁。

## 12. 评审采纳记录

- 将 K32 的门禁从 kernel pair 改为包含 shared expert 的 all-in MoE；
- 删除 row-W8 “fallback”承诺，改为严格条件研究门；
- 明确 backend-owned prefill-only workspace 与 decode graph capture 顺序；
- 明确 single-deepest 256-aligned prefix checkpoint 语义；
- 将 SparkInfer 范围缩到缺失的 IQ2 lowering/kernel；
- 明确 p50/p90、warmup/replay 和 service sample 数。
