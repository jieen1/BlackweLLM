# DSV4 CUDA-Graph decode 驱动：捕获成功、状态机修复与真实权重门禁

日期：2026-08-09

## 背景

上一阶段 decode 已优化到 ~254ms（fused Q8_0 + wo_a grouped + HC fused，见
`2026-08-09-dsv4-serving-e2e-and-bf16q-prefill-bug.md`）。本轮目标是继续压掉 decode
的 CPU launch 开销，并把 CUDA Graph、compressor/indexer 状态机和服务器接入收口到同一
条可回归路径上。

## 当前事实

### 1. compressor 状态迁移已改成 `torch.where`，并补上 ratio-4 / ratio-128 回归

`Dsv4Compressor.forward_graph` 里原来的乘法 mask 已经改成选择式迁移，避免
`0 * -inf = NaN`。对应的 regression 现在覆盖两条边界：

- `tests/test_dsv4_compressor_graph.py::test_forward_graph_overlap_state_matches_eager_across_steps`
- `tests/test_dsv4_compressor_graph.py::test_forward_graph_ratio128_matches_eager_across_boundary`

这两条把 ratio-4 和 ratio-128 的连续多步状态迁移都钉住了，重点不再是“定位到
nan”，而是“修完后不再回归”。

### 2. compressed pack 写址和 indexer 状态机都改成自己的责任边界

`Dsv4AttnKernelLayer._pack_compressed` 现在在 capture 路径里写当前压缩槽，而不是旧的
“上一槽”地址；这避免了非边界步覆盖有效 compressed KV 的问题。

`Dsv4Indexer.forward_graph` 也不再借用主 attention compressor 的状态机，它会先把自己的
`compressor.kv_cache/freqs_cis` 绑定起来，然后自己推进 compressor，再做 index top-k。
这条边界现在被 `tests/test_dsv4_compressor_graph.py::test_indexer_forward_graph_advances_its_own_compressor`
锁定。

### 3. decode CUDA Graph 已完整接入 backend 和 engine

`runtime/backends/dsv4_cudagraph.py` 现在有完整的 `Dsv4DecodeGraphDriver`：

- 预分配 slot-owned 输入张量
- warmup 后 capture 整个 43 层 decode
- replay 时只改 token / position 内容
- 中间张量由 graph pool 固定地址；slot 间只允许串行复用同一 graph pool
- capture 失败则整条回落 eager

这和早期方案里“每个 binding/out 都由 driver 显式预分配”的措辞不同，是本轮实际采用的
capture contract：输入、递归状态和 KV 页是显式 persistent tensor，其余 capture 内分配由
CUDA graph pool 保持地址。backend 的 decode loop 本来就是逐槽串行 replay，因此不允许把
共享 pool 的两个 slot graph 并发 replay。首次 capture 还会硬性拒绝任何非 fresh slot，避免
后续误调用清空活跃请求；成功后的重复 capture 调用是无状态改动的幂等查询。

`runtime/backends/dsv4.py` 已把 backend capability 设为 `cuda_graph=True`，并在
`capture_decode_cuda_graph()` 成功时为每个 slot 建 driver；`server/engine.py` 也在加载
阶段直接尝试 capture，成功则记录“captured at load”，失败则回退 eager。

对应的 A/B 门禁也已经进入脚本：

```bash
python scripts/dsv4_align_eager_vs_kernel.py <model.gguf> --cuda-graph --steps 12
```

### 4. grouped Q8 prefill 和 MoE prefill 的真实模型阻断都修掉了

`runtime/kernels/dsv4_q8_gemm.py` 的 grouped Q8 kernel 现在按 row tile 扩展 grid，不再被
“rows_per_group 必须小于 BLOCK_M”卡死，补上的回归覆盖了非 2 的幂和超过 16 行的真实
prefill 形状。

`runtime/model/dsv4_model.py` 里 MoE prefill 也不再把 expert id 拉回 CPU；batch expert id
保留在 GPU tensor 上，测试锁在 `tests/test_dsv4_moe.py::test_cuda_prefill_keeps_batched_expert_ids_on_device`。

### 5. 真实权重短门禁和 P0 全仓门禁均已跑通

修复过程中的第一轮真实权重门禁是 3 个 workload、连续 12 个 decode step。结果是：

- worst logits cosine: `0.99999988`
- greedy token: `12/12`
- eager: `449.1 ms`
- graph: `272.1 ms`

P0 代码审查、格式化和全仓测试完成后，以同一命令重新冷加载并跑了最终短门禁（3 个
workload × 4 decode step，3 个 prefill anchor 也计入 token agreement）：

- worst logits cosine: `0.99999988`
- greedy token: `15/15`
- eager: `218.8 ms`
- graph: `116.0 ms`
- speedup: `1.89x`

这次重跑已经把 graph replay 拉回历史 `136 ms` 的同一量级，但仍不能把不同日期、不同热身
状态的单次均值当成长期基线；P1 会把环境和 trace 固化到 `bfdiag` 后再下稳定性能结论。

P0 的仓库门禁为：

- ruff check：全仓通过；本轮 12 个 Python 文件 format check 通过
- torch-free CI simulation：`1195 passed, 217 skipped`
- 完整 venv suite：`1826 passed, 34 skipped`
- DSV4 backend 定向测试：`24 passed`

## 剩余风险

这一轮已经不再保留“NaN 未修”的状态。当前需要持续盯住的是 graph capture 的 slot 绑定、
真实权重门禁的口径一致性，以及不要把短跑数字误当成统一基线。

## 下一步

1. 把当前 3-workload / 12-step 门禁扩成更长的真实权重回归，先补足可比性，再谈统一性能结论。
2. 补齐 `bfdiag` warm daemon 的 DSV4 provider，让后续 `bf exec` / `bf diff` 的性能数值有单一口径。
3. 继续推进 P1 诊断与统一基线：trace 事件、snapshot、eager fallback 原因、slot / position /
   ratio 可观测性。
4. 在没有完整可比基线之前，继续把 `prefix_cache` / `chunked_prefill` 留在保守状态，不要为了
   追求功能“看起来完整”而跳过门禁。
