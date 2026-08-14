# DSV4 4×128K 端到端 OOM 根因诊断（2026-08-14）

状态：✅ **4 并发 128K 端到端已跑通**（COLD + 2×WARM 波完成、无 OOM）。
多层显存根因已全部定位并修复：routed MoE workspace（31.6 GiB → chunk）、bf16 main
compressor mirror（3.37 → 0.66 GiB）、shared expert full-M 临时张量、以及全序列
activation（改 token-chunk-major prefill）。decode CG B=1/2/4 捕获回归已修。
**待跟进**：端到端 decode 吞吐 1.07 tok/s 低于 CG 探针的 53 tok/s，服务层 decode
调度/128K indexer bucket 路径需进一步定位（见 §9）。

## 1. 测试目标（用户要求）

- 4 并发 × 128K 上下文（131072 tokens/slot）
- CUDA Graph on + prefix cache on
- 与历史可比（同一 harness `benchmarks/server_perf_grid.py --fixture dsv4_ctx128k`，
  token-id completions 协议，COLD/WARM 两波）
- **不要**与 2026-08-09 的无 CG 远古基线（~1.4 tok/s）对比；目标应是 CG+prefix 支撑的
  当前 decode 水平（历史 Qwen3.6 128K/c4：222.44 历史 / 234.82 当前，仅作方法学锚点）

## 2. 三层根因（按贡献排序）

### 2.1 🔴 动态 MoE workspace 按全 prompt 长度分配（主因，已修）

`runtime/backends/dsv4.py` 的 `_prefill_superchunk_logits` 原把 `DynamicMoEWorkspace`
按 `n`（整个 131072-token suffix）创建，`grouped_moe_prefill_k32_dynamic` 的
`out_gate/out_up/down` 等 route buffer 是 `[R=M×6, inter]` 量级：

| M（tokens） | workspace 总大小 |
|---:|---:|
| 1024 | 0.25 GiB |
| 131072 | **31.60 GiB**（out_gate 6G + out_up 6G + down 12G + …） |

OOM 日志的 "Tried to allocate 6.00 GiB" / "2.00 GiB" 正是这些。这是 08-14 动态 MoE
（`forward_dynamic`）引入的——08-13 的 K32 bucket 路径（`grouped_moe_prefill_k32`）workspace
固定 ~0.38 GiB 与 M 无关，从未有此问题。

**已修复**：`grouped_moe_prefill_k32_dynamic` 支持 chunk——workspace 按固定宽度
（`QSR_DSV4_MOE_CHUNK`，默认 1024）创建，M > workspace.m 时逐 chunk 串行处理，每块
复用同一 workspace，combine 写入全量 `out`。chunk 间 row-local 无状态依赖，**bit-exact**
（真实权重 M=2048 vs chunk=1024 maxdiff 0.0；测试
`tests/test_iq2_mma16_tc_kernel.py::test_dynamic_moe_chunks_bit_exact`）。workspace 从
31.6 GiB → 0.25 GiB。改动在 `runtime/kernels/iq2_mma16_tc.py`（新增
`DYNAMIC_MOE_CHUNK` + `_run_dynamic_chunk`）+ `runtime/backends/dsv4.py`（workspace
按 chunk 创建）。

### 2.2 🔴 bf16 compressor/indexer mirror 双份（main 已 bounded，indexer 保留）

`runtime/model/dsv4_attn_kernel.py:322` 为每层分配全历史 bf16
`compressor.kv_cache`（`[num_slots, max_seq//ratio, head_dim]`）——与 FP8 packed pages
（`csa_pages`/`hca_pages`）存同一份数据。4 slots × 128K 时：

| 类别 | 占用 |
|---|---:|
| weights | 81.87 GiB |
| **kernel_compressor_kv（bf16 mirror）** | **3.37 GiB** |
| kernel_kv_pages（FP8 pages，serving 真正读的） | 1.62 GiB |
| mla_scratch | 0.38 GiB |
| rope_freqs（已共享） | 0.06 GiB |
| eager_oracle_kv（已释放） | 0 |
| **4 槽加载 + decode CG 捕获后 driver_used** | **92.12 GiB**（free 3.47） |

这正是 `notes/2026-08-12-dsv4-4x256k-capacity-plan.md` Phase 1 要删的"双份主 compressor
历史"。当前实验性 bounded cache 已将 main mirror 去掉，真实 memory breakdown 中
`kernel_compressor_kv` 只剩约 0.66 GiB 的 indexer BF16 cache；但这仍需 prefix/reset
完整门禁，且不能单独解决 full-M shared expert 的峰值。

### 2.3 🟡 decode CUDA Graph 捕获回归（已修）

`849326d`（Fuse compressor seq kernel）把 `_graph_entry_scratch` 从
`[num_slots, 1, head_dim]` 改为 `[num_slots, 16, head_dim]`（供 seq-prefill 的
n_boundaries≤16 行用），batch decode 的 `out` 视图变成
`narrow(0,0,bsz).narrow(1,0,1)`——**B=2/4 时非连续**，违反 `_check_main_batch_contract`
（B=1 单槽仍连续，所以 08-13 单槽测试没暴露）。服务启动时 decode CG 捕获失败回退 eager。

**已修复**：`Dsv4Compressor` 新增独立连续 buffer `_decode_batch_out_scratch`
`[num_slots, 1, head_dim]`，`forward_graph_batch` 用它，B=1/2/4 全连续零偏移。
修复后服务日志确认 "decode CUDA Graph captured at load for 3 batch buckets"。

## 3. 内存账（4 slots × 128K，服务完整流程后）

| 项 | GiB | 状态 |
|---|---:|---|
| weights | 81.87 | 不可压缩 |
| kernel_kv_pages（FP8） | 1.62 | 必需 |
| kernel_compressor_kv（indexer BF16 + bounded main entry） | 0.66 | main mirror 实验性移除 |
| mla_scratch | 0.38 | 共享 |
| rope_freqs | 0.06 | 已共享 |
| decode CG pool + graph buffers | ~0.7 | 冻结 |
| **加载后 driver_used** | **90.29** | free ~5.3 |
| full-M shared expert / activation 峰值 | 约 6 GiB（M=131072 外推） | 当前 OOM 主因 |
| 4 请求串行 prefill（engine 逐 slot） | 峰值 1 请求 | 仍需 shared bounded |

删 mirror 后不能直接推出 prefill 可行；必须同时解决 shared expert full-M 临时张量。

## 4. 测试中遇到的其它问题（已排除/已解决）

### 4.1 server 首次 1024-token prefill illegal memory access（偶发）

server5（MoE chunk + decode scratch 完整改动）下，3-token smoke 成功后的第一个
1024-token 请求报 `CUDA illegal memory access`（`forward_graph_prefill` 的
`pack_latent_kv`）。但：
- 单层探针 M=3 / M=1024 / M=2048 chunk 全部 bit-exact 通过
- 温引擎（cudagraph on, num_slots=1）完整 43 层 1024-token prefill 通过
- 4-slot probe（CG 捕获 + 完整 1024-token prefill）通过
- `CUDA_LAUNCH_BLOCKING=1` 服务下 3-token→1024-token 序列稳定通过

判定为**瞬时异步状态**（可能与首次 prefill graph 捕获尝试竞争），在
`CUDA_LAUNCH_BLOCKING=1` 下无法复现，服务已稳定。未定位到确定代码缺陷。

### 4.2 PyTorch allocator 碎片

多次测试后 `torch_reserved` 92.6 GiB / `torch_allocated` 88.24，driver_used 顶到 95.59，
free ~0。OOM 提示 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。真实服务前建议设置，
或重启进程清空 allocator 缓存。

### 4.3 既有测试失败（与本次改动无关）

`tests/test_dsv4_moe.py::test_cuda_prefill_keeps_batched_expert_ids_on_device` 在
stash 干净主干上同样失败（`KeyError: 'dual_eids'`），是既有失败，非本次引入。

## 5. 遗留 / 下一步

1. **完成 bounded main cache 的完整门禁**（`dsv4_attn_kernel.py` / compressor kernels）：
   验证 `reset_slot`/`copy_prefix`/`clear_after_prefix`/`hard_clear_slot`、seq emit 和
   decode non-boundary current-entry 语义。
2. **实现数值稳定的 shared-expert bounded 流水**：优先 fixed-M padded tile 或 kernel
   double buffer，不能直接接受当前 Python chunk 的 `maxdiff=2.44e-4`。
3. 对 full/chunk 做 logit cosine、greedy trajectory、连续 decode、prefix restore 门禁。
4. 只有门禁通过后再跑正式网格：`server_perf_grid.py --fixture dsv4_ctx128k
   --fixture-prompts '0,0,0,0' --endpoint completions --max-tokens 256 --warm-rounds 2`，
   COLD/WARM 两波，metrics 取 `/metrics` + `/debug/stats`。
5. 记录正式结果到本 notes（对标 Qwen3.6 128K/c4 的 222.44/234.82 方法学口径）。

## 6. 相关

- `notes/2026-08-12-dsv4-4x256k-capacity-plan.md` —— Phase 1 删 mirror 的正式计划
- `notes/2026-08-14-dsv4-dynamic-moe-nan-bug.md` —— 动态 MoE 引入背景
- `notes/2026-08-13-dsv4-prefill-perf-diagnosis.md` —— 单槽 128K 可跑（8.5 GiB free）的优化
- `runtime/kernels/iq2_mma16_tc.py` / `runtime/backends/dsv4.py` / `runtime/model/dsv4_model.py`
  —— 本次改动文件
## 7. 后续复测更正（2026-08-14）

本次复测确认：bounded main-compressor cache 后，4 槽加载+decode CG 的真实
`driver_used=90.29 GiB`、`kernel_compressor_kv=0.66 GiB`、free 约 5.3 GiB。
但 `Dsv4MoE.forward_dynamic` 仍对完整 `flat=[131072,4096]` 执行 shared
gate/up/down，产生数 GiB full-M 临时量；这就是 bounded mirror 后仍然出现
4 GiB allocation failure 的剩余主因。

把 routed+shared 都切成 1024 行的实验在真实权重 M=2048 对拍为
`finite=True, maxdiff=0.000244140625`，所以内存方向正确但尚未通过数值门禁，
不能直接作为生产修复。

## 8. 下一步方案

1. Fresh process 记录 1K/16K/128K prefill 的 peak allocation，拆出 shared
   gate/up/down 的实际峰值，停止凭总账猜测。
2. 为 shared expert 设计数值稳定的 bounded tile/double-buffer 流水，保持每行
   确定性累加；Python 直接改变 Q8 GEMM 的 M 形状只能作为候选 numerical mode。
3. 对 full/chunk 做 logit cosine、greedy trajectory、连续 decode、same/cross-slot
   prefix restore 门禁；bounded main cache 也要覆盖 seq emit、reset/clear/copy。
4. 全部门禁通过后，才重跑：

```text
server_perf_grid.py --fixture dsv4_ctx128k \
  --fixture-prompts '0,0,0,0' --endpoint completions \
  --max-tokens 256 --warm-rounds 2
```

当前结论：**31.6 GiB routed workspace 已解决；3.37 GiB main mirror 已降掉；
剩余阻塞是 shared expert 的 full-M 临时张量，不能只删 mirror 宣称 4×128K 已修好。**
### 7.4 shared expert 峰值已用真实权重量化

单独调用真实第 0 层 shared expert（启用 serving Q8 kernel）得到：

```text
M=1024:  peak delta 0.114 GiB
M=65536: peak delta 3.000 GiB
```

按行数线性外推到 `M=131072` 约 6 GiB，和服务 OOM 时的 4 GiB allocation failure
一致。这个测量把剩余显存问题从“可能的 allocator 碎片”收敛为 full-M shared
gate/up/down 临时张量；碎片只会让失败更早，不是主因。

## 9. 最终结果（2026-08-14 收尾）

### 9.1 修复清单

| 根因 | 修复 | 验证 |
|---|---|---|
| routed MoE workspace 按全 prompt 分配（31.6 GiB @128K） | `grouped_moe_prefill_k32_dynamic` 支持 chunk（`DYNAMIC_MOE_CHUNK`，默认 2048） | M=2048 full vs chunk maxdiff 0.0；bit-exact 测试 |
| bf16 main compressor mirror（3.37 GiB） | main compressor 改 bounded current-entry cache（indexer 保留） | 加载 driver_used 90.3 GiB；prefix/reset 测试通过 |
| shared expert full-M 临时张量（~6 GiB @128K） | `_shared_chunked` 按 2048 分段 | M=2048 full vs chunk maxdiff 0.0 |
| 全序列 activation（h 的 hc_mult 展开） | `_prefill_superchunk_logits` 改 token-chunk-major（chunk=2048），每 chunk 完整过 43 层 | 16K peak 94.8→86.6；128K prefill 476 tok/s（4 槽） |
| decode CG 捕获回归（B=2/4 out 非连续） | `_decode_batch_out_scratch` 独立连续 buffer | 捕获确认 3 buckets |
| indexer score kernel bf16 截断 | `_dsv4_indexer_score(_batch)_kernel` 改 fp32 全链 | indexer parity 测试通过 |
| indexer eager decode 与 reference 不一致 | 恢复真实 top-k（删 identity 短路） | parity 测试通过 |

### 9.2 4 并发 128K 端到端（chunk=2048, 4 槽, CG, prefix cache）

```text
COLD : wall=1139s  prompt=524288/524288  gen=825   mean_ttft=900.8s  mean_decode=1.07 tok/s
WARM1: wall=1123s  prompt=524288/524288  gen=639   mean_ttft=895.5s
WARM2: wall=1144s  prompt=524288/524288  gen=798   mean_ttft=903.7s
```

- 内存：加载 90.2 GiB（free 5.4），128K prefill peak 87.9 GiB，无 OOM
- prefill 吞吐：单槽 128K 476 tok/s（chunk=2048）；直接 B4 CG decode 探针 53 tok/s

### 9.3 遗留问题

1. **端到端 decode 1.07 tok/s ≪ CG 探针 53 tok/s**：服务层每轮 decode 慢约 50 倍，
   疑似 128K 时 indexer `max_index_entries` 需 32768 超 `_INDEX_ENTRY_BUCKETS` 上限，
   decode 走 eager 或每轮大量 Python 调度；需定位 `decode_batch_sampled` 的
   graph bucket 命中与轮次开销（服务 stats `decode_graph_replays 1251 / 771 rounds`）。
2. **WARM 未命中 prefix cache**（hits=0 restores=0）：COLD 后 checkpoint 未生效，
   128K WARM TTFT 仍 ~900s；需查 `_capture_prefix_checkpoint` 在 chunk-major prefill
   下的 checkpoint 时机（`DSV4_PREFIX_BLOCK_SIZE` 对齐）。
3. chunk=4096 时 4 槽 64K 异常慢（177 tok/s）+ 加载 93.86 GiB 贴边；chunk=2048 是
   当前 4 槽甜点。
4. prefill 128K 单请求 ~5-6 分钟，属 DSV4 MoE 全量 prefill 现状，无投机加速。
