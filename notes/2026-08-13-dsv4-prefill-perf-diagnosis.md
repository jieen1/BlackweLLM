# DSV4 服务启动与 prefill 性能诊断（2026-08-13）

三轮排查：OOM → 乱码 → prefill 慢。正确性已全部修复，性能瓶颈已精确定位。

## 1. 已修复的 bug（正确性）

### 1.1 RoPE 共享顺序导致 decode 乱码
`server/engine.py` 原来在 `capture_decode_cuda_graph()` **之后**调用
`_share_rope_freqs()`。decode CUDA graph 捕获时烘焙了 kernel-path 层的 RoPE 表
地址；共享后旧表被 GC，graph replay 读已释放的存储 → 输出垃圾 token
（实测 graph=124208 ' buruj' vs eager=1 EOS）。

**修复**：`_share_rope_freqs()` 移到 decode graph 捕获**之前**（capture 烘焙共享表
地址），`_free_eager_oracle_caches()` 移到捕获之后（无害，只为 prefill scratch
腾空间）。顺序：share rope → capture decode → free eager KV。

### 1.2 MoE graph 缺 batch2 拆分（量化溢出）
`grouped_moe_prefill_k32_graph` 原来单 batch bucket=64，超 64 的 route 用
`clamp(max=63)` **静默截断**。DSV4 的 hash 层（layer 0/1/2）与非 hash 层在真实
数据下路由极度集中（max route 实测：64-token=46、256-token=103、1024-token=384），
截断导致输出错误（cos 0.995 → 端到端 token 漂移）。计划 §5.1 明确合同
"expert route 超 64 时拆分，不能截断"，实现遗漏了。

**修复**：graph 版改成两批固定拆分（bucket=32 + overflow 批覆盖 64），fill 用
`index_put_ accumulate=True` + `where` mask（非法 route 写 0 累加不影响合法 route），
combine 用 `where` mask。MoE 分段必须 ≤64-token 才安全（实测 64-token max route 46 < 64）。

### 1.3 combine 顺序（MoE bit-exact）
eager `grouped_moe_prefill_k32` 在 sum 前先 `argsort(indices)` gather 按 expert-id
排序 top-k 贡献；graph 版直接 sum。fp32 累加顺序不满足交换律，1e-5 差异被
sparse-attention topk 跨 43 层放大成 token 漂移。**修复**：graph 版加
`final_order = indices.argsort(dim=1, stable=True)` gather 后 sum，与 eager bit-exact。

### 1.4 尾块回退
`_forward` 里 prefill graph 分支原来无条件 `replay_layer`，但最后 chunk 不足 64 行
（或 M=1 decode fallback 复用 `_forward`）会撞 graph 的固定 (M, H) 契约。**修复**：
仅 `input_ids.shape[1] == m` 时走 graph，否则 eager `block.moe`。

### 1.5 cursor resize 破坏 graph
`device_group_counts_into` 里 `torch.arange(R, out=cursor)` 把声明为 n_experts 的
cursor 就地 resize 到 R（R=m*top_k=384 > 256），违反 CUDA graph 固定形状契约。
**修复**：cursor 声明为 R，加 `numel() < R` 守卫。

## 2. 显存优化（参考 notes/2026-08-12-dsv4-4x256k-capacity-plan.md）

| 项 | 效果 |
|---|---|
| RoPE 按 regime 共享（compressed 41 层 + window 2 层 → 2 张表） | 省 1.31 GiB |
| prefill graph 43 层共享 CUDA graph pool | 冻结 10 GiB → 1.14 GiB |
| `cg_extra=0`（DSV4 decode 是 shared batched driver，无需独立 warmup slot） | 省一个 slot |
| prefill graph 内存守卫 + 失败 `empty_cache` | 不 OOM，干净回退 eager |

128K 单 slot 服务可跑（driver_free ~8.5 GiB），prefill graph 成功捕获。

## 3. 性能瓶颈（未达标，已精确定位）

880-token prefill = 7.8s GPU（112 tok/s）：

| 组件 | 耗时 | 占比 | 本质 |
|---|---|---|---|
| attention | 5.2s | 66% | ratio-4 indexer + compressor **CPU-bound** |
| MoE | 2.6s | 34% | GPU-bound，eager 已优化 |

关键证据：
- ratio-4 层 17.8ms/tile（vs ratio-128 4.2ms、window 1.9ms）
- attention tile 的 CUDA event 18.9ms vs 纯 kernel 3.5ms → **15ms 是 GPU 空转等
  CPU 调度**（compressor 的 64 次 Python 循环 × ~10 op/token 的 dispatch gap）
- 验证：固定 position 的 attention tile CUDA graph 化 = **5.4x**（18.9→3.5ms）

## 4. 剩余方案（达成 1000 tok/s 的必要工程）

1. **attention 的 seqlen=64 批量 GPU 化**（最大 ROI）：
   - 写 compressor 的 seqlen=64 批量状态更新（保持 wkv/wgate 批量 GEMM，状态机
     用 GPU position + torch.where 向量化，消除 64 次 Python 循环）
   - indexer 的 seqlen=64 批量评分（`_MAX_ROWS` 32→64 已可改，但评分本身只 0.1ms，
     真正要消除的是 forward 内 20 个小 op 的 dispatch gap）
   - 整个 tile 用一个 GPU-position CUDA graph 服务所有 position
2. **MoE 的 graph 化放弃**：DSV4 路由极端集中（max route 384），固定 bucket 无法
   高效处理，eager 的动态 batch2 是正确方式（已 GPU-bound，graph 化无益）。

**性能参考**：llama.cpp 同权重 706 tok/s @ 1024（高度优化 C++）。1000 tok/s 目标
需要 attention kernel 工程（预期先到 ~300 tok/s，再逐步逼近）。

## 5. 验证结果

- 完整 superchunk（64-token MoE 分段 + batch2 + combine final_order）：
  cos **1.00000012**，greedy token **全部 match**（355/706/985 token）
- 单层 MoE graph vs eager：**bit-exact**（maxdiff 0）
- ruff 全绿；56 passed（1 个 main 基线既有失败 `test_cuda_prefill_keeps_batched_expert_ids_on_device`）
