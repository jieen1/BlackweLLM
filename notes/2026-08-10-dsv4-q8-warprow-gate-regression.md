# DSV4 decode 优化质量门禁回归：Q8_0 warp-per-row GEMV 引入的数值漂移

日期：2026-08-10

## 1. 背景

为突破 DSV4（DeepSeek V4 Flash）M=1 decode 的权重带宽墙，我们引入 **Q8_0 对齐 SoA 重打包 + warp-per-row GEMV**（ds4 报告验证 SM120 上 196→1386 GB/s，7.1×）。单 kernel 验证**数值正确、性能 7×**，但**未通过项目官方端到端门禁**，已回滚。本文档完整说明现象、已确证事实、尝试、未解疑点，供接手者快速进入。

## 2. 现象

门禁脚本：`scripts/dsv4_align_eager_vs_kernel.py --cuda-graph --steps 12`

| 版本 | worst logits cos | greedy | verdict |
|---|---|---|---|
| 基线（tensor-core tl.dot，bf16 累加） | **0.99999988** | 39/39 (100%) | **PASS** |
| Q8_0 warp-per-row（fp32 累加） | 0.99907 | 39/39 (100%) | REVIEW |
| Q8_0 + wo_a warp-per-row | 0.979 | 39/39 (100%) | REVIEW |
| Q8_0 warp-per-row（bf16 权重舍入） | 0.99903 | 39/39 (100%) | REVIEW |

- **greedy token 始终 100% 一致**，只有 logits cosine 漂移（<0.99 门禁）。
- 漂移只在门禁的**第 3 个 workload**（wl2）出现，从 decode step 2 开始（step 0-1 完美 cos 1.0，step 2 起 0.99987→0.99908）。wl0/wl1 完美（cos 1.0）。
- **单 workload 12 步完美**（cos 0.99999988），只有门禁的 3-workload 结构触发漂移。
- 排除累积污染：只跑 wl2（不跑 wl0/wl1）也漂移。

## 3. 门禁脚本结构（关键）

门禁对比两条路径（`scripts/dsv4_align_eager_vs_kernel.py`）：
- **eager 路径**：`backend._forward(0, ids, position)` → `_forward_decode_batch`（M=1 decode）
- **graph 路径**：`backend._decode_graphs[1].replay_host([token],[position],[1], ...)` → 捕获的 `_forward_decode_batch`

两者**共享同一权重、同一 fused 路径**（后端构造时 `_enable_serving_q8_kernels` 设 `fused_q8=True`）。**都走 warp-row**。所以门禁测的是 graph 捕获 vs 连续 eager 的**自洽性**，不是 kernel vs eager 的数值差。

门禁结构：3 个 workload（prompt 长度 [5,9,5]），每 workload reset_slot(0/1) → `_prefill_logits`（多行 forward，tensor-core M>1）→ 12 步 decode（M=1，warp-row）。prefill 走 tensor-core（M>1 分支），decode 走 warp-row（M=1 分支）。

## 4. 已确证事实（数据）

### 4.1 单 kernel 数值完全正确
- warp-row（fp32 累加）vs eager fp32 oracle：rel ~1e-7（更准）
- warp-row（bf16 权重 `(qs*d).to(bf16)`）vs tensor-core tl.dot：**maxdiff 0.0**（完全一致）
- 结论：warp-row kernel 本身数值正确，且 bf16 版本与 tensor-core 位级一致

### 4.2 性能真实（未受影响）
- wo_b 43 权重旋转（冷）：tensor-core 196 GB/s → warp-row 1386 GB/s（7.1×）
- wq_b：450 → 1377 GB/s（3.1×）
- graph 58→40ms（fp32 warp-row），wo_a 再 -5ms（35ms）

### 4.3 漂移隔离
- **Q8_0-only warp-row**（wo_a 保持 eager/tc）：单 workload 12 步 cos 0.99999988（完美）—— 但门禁 3-workload 仍 REVIEW
- 门禁 wl2 step 2 起漂移，与累积无关（只跑 wl2 也漂移）

### 4.4 d（scale）非连续 bug（已发现但非主因）
- `repack_q8_0_soa` 的 d 平面是 `wv[:, :, :2].view(fp16).squeeze(-1)` —— **非连续**（stride (544,17)），kernel 线性索引 `d_ptr + row*n_blocks + b` 会读错位
- 修复（`.contiguous()`）后仍不过门禁，说明非主因

### 4.5 soa_planes 懒构建（疑点）
- `PackedQ8_0Weight.soa_planes()` 懒构建缓存 q/d 普通属性
- graph 捕获期间 warmup 时构建，连续 eager 复用同一 planes
- 疑点：graph 捕获内的分配（planes）在 graph pool 中，捕获后地址固定，连续 eager 是否用同一内容

## 5. 尝试过的方案

| 方案 | 结果 |
|---|---|
| fp32 累加 warp-row | REVIEW（0.99907）|
| bf16 权重舍入 warp-row（匹配 tl.dot 契约） | REVIEW（0.99903）—— 单 kernel 与 tl.dot maxdiff 0 仍漂移 |
| wo_a warp-row | REVIEW 更差（0.9994 单独，0.979 合并）|
| d 平面 contiguous 修复 | 仍 REVIEW |
| 只跑 wl2（排除累积） | 仍漂移 |
| 排除 Q8_0 kernel 本身（maxdiff 0 证明） | 漂移来自别处 |

## 6. 未解疑点（接手者重点）

**核心矛盾**：Q8_0 warp-row bf16 与 tensor-core **单 kernel maxdiff 0**，但门禁（graph vs eager，两者都用 warp-row）**不自洽**。这意味着漂移**不是 Q8_0 计算的数值差**，而是**引入 warp-row 这个改动改变了 graph 捕获 vs 连续执行的某个共享状态**。

候选假设（未验证）：
1. **`soa_planes()` 懒构建时机与 graph 捕获的内存池交互**：graph 捕获期间（warmup）构建 planes，planes 分配落在 graph pool；捕获后连续 eager 复用。若 graph 重放时 planes 的**地址内容**与捕获时不一致（如 weight 加载后 planes 未重建）→ 漂移。需验证：graph 捕获前 vs 连续执行时，planes 是否同一内容/地址。
2. **`_forward_decode_batch` 的 M=1 分支在 graph 捕获与连续执行间行为不同**：graph 捕获时可能走了 M>1（预填 warmup）或不同代码路径，捕获的 body 与连续 M=1 不一致。
3. **wo_a 的 grouped 分支**（M=1 时也可能受影响）：wl2 的特定状态触发 wo_a 的某分支差异。
4. **非 Q8_0 算子**（MLA run、compressor、indexer）在 graph 捕获 vs 连续间的差异，被 warp-row 的引入**放大**（虽然 warp-row 本身数值对，但改变了时序/内存布局，暴露了既有差异）。

**关键验证建议**：
- 在门禁 wl2 漂移点，**逐层对比 eager vs graph 的 block 输出**（`_forward_decode_batch` 每层 x），定位漂移从第几层、哪个算子开始。
- 对比**同一输入下** eager 的 `_forward` 与 graph replay 的**逐 kernel 中间值**（wq_a/wq_b/wkv/wo_a/wo_b/MLA/compressor），找出第一个不一致的算子。
- 验证 soa_planes 在 graph 捕获 vs 连续执行的内容一致性（打印 data_ptr + 首值）。

## 7. 当前代码状态

- 已回滚 warp-row（`037cbb5` Revert wo_a、`3e9420f` Revert Q8_0），**门禁 PASS（0.99999988）**，decode 58ms。
- 门禁脚本已修复适配新后端（`22f746f`）：`--cuda-graph` 从崩溃恢复，能跑（之前 `_decode_graphs.pop(0)` KeyError）。
- 工作区干净。未提交的调试脚本在 /tmp/opencode/*.py。

## 8. 参考

- ds4 对齐方案：`notes/2026-08-10-ds4-cuda-deep-dive.md` §2A（warp-per-row 结构、实测 +43~66%）
- 门禁脚本：`scripts/dsv4_align_eager_vs_kernel.py`（`run_cuda_graph_gate` 128-230 行）
- 后端 `_forward_decode_batch`：`runtime/backends/dsv4.py:682`
- 门禁脚本的历史 PASS 记录：`notes/2026-08-09-dsv4-cudagraph-decode-driver.md:74-88`（0.99999988，旧 per-slot 后端）
