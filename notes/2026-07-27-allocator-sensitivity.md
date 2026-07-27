# DFlash 接受率依赖 caching-allocator 布局(2026-07-27)

**一句话:`block_size=128` 下 DFlash 的接受率不是一个数,而是 {0.452525, 0.602564, 0.675362}
中的一个,由分配器布局决定;`block_size=64` 在同样扰动下逐位不变。**

配套工具:`bf sensitivity sweep` / `bf sensitivity cycles`(`bfdiag/sensitivity/`)。
上游根因排查见 `notes/2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md`。

---

## 1. 怎么发现的

复现同一份 A/B 时,用自己写的脚本得到 0.675362,而 `benchmarks/ab_dflash_block_size_64_vs_128.py`
稳定给出 0.452525。逐项排除后(**每一项都实测过,不是推断**):

| 排除项 | 结果 |
|---|---|
| 代码根(我的分支 vs main+未提交 WIP) | 两边都给 0.675362,**不是代码** |
| cwd(仓库根 vs 别处,`.autotune_cache` 可见性) | 都给 0.675362,**不是 autotune 缓存** |
| venv / sparkinfer(含 `wait_group` race 修复) | 同一套 |
| `blocks_per_slot`(130)、`max_model_len`(12544) | 逐值相同 |
| CUDA Graph 状态 | 两边 `verify_cg=True draft_cg=True` |
| prompt | 同一个,sha256 `0c57d020…` |
| `reset_slot` / `enable_prefix_cache` | 单独变化都不改变结果 |

最后机械 diff 两个脚本,**唯一的功能差异是一行 `gc.collect()`**。

## 2. 受控数据(每行一个全新进程,全新加载)

| block_size | 扰动 | 接受率 | 输出 token sha | allocated MiB | reserved MiB | segments | inactive_split |
|---|---|---|---|---|---|---|---|
| 64 | none | 0.718182 | `4fe0bef347b7` | — | 72900 | 470 | 83 |
| 64 | gc | **0.718182** | **`4fe0bef347b7`** | — | 72900 | 470 | 83 |
| 128 | none | 0.452525 | `e5028b36258b` | 72759.15 | 72920 | 471 | 85 |
| 128 | reset | 0.452525 | `e5028b36258b` | 72759.15 | 72920 | 471 | 85 |
| 128 | **pad16** | **0.452525** | **`e5028b36258b`** | — | **72920** | **471** | **85** |
| 128 | gc | **0.675362** | `d6e4833404d4` | **72171.15** | 72920 | 471 | 85 |
| 128 | pad256 | **0.602564** | `3cc5b31ad685` | — | **73176** | **472** | 85 |

**读法**:
- `pad16` 是干净对照 —— 分配器三项与 `none` 完全相同 → 输出**逐位相同**。
- `gc`(allocated −588.00 MiB)和 `pad256`(reserved +256 MiB、segments +1)改变了分配器状态 → 输出变。
- **「分配器状态不变则结果不变,状态变则结果变」这条对应关系是干净的。**
- `pad256` 只是 `torch.empty(256MiB, cuda)` 后 `del`,**整条路径不含 vLLM、不含 gc** ——
  所以这不是「gc 的问题」,是**分配器布局敏感性**,gc 只是最容易观察到的触发器。

## 3. 触发源:588 MiB 死 lm_head 被引用环挂着(vLLM 侧)

`gc.collect()` 恰好释放**一个** 588.00 MiB 的块,`active_blocks` 1768 → 1767。

用 `gc.set_debug(DEBUG_SAVEALL)` 抓出来:一个**全零的二维 Parameter**,存活于某个模块的
`_parameters` dict 构成的引用环里。尺寸精确匹配 `vocab_size 100352 × hidden 3072 × 2 B (bf16)
= 616,562,688 B = 588.00 MiB`,即一份**未使用的 lm_head**。`runtime/backends/laguna.py`
里没有任何 lm_head 处理代码,所以它来自 vLLM 侧的模型构造。

两个后果:
1. **588 MiB 显存泄漏**,直到某次分代 GC 才释放。
2. **释放时机不确定** —— Python 自动 GC 按分配计数触发,所以「脚本制造了多少 Python 对象」
   会决定它在 `generate()` 之前还是之后被回收。**这解释了历史上「调整代码格式之后结果就变了」
   这类现象:改动改变了对象数量 → 自动 GC 时机变 → 分配器布局变 → 数值结果变。**

## 4. 顺带发现:CUDA Graph 捕获后调 `torch.cuda.empty_cache()` 必崩

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```
它会释放已捕获的 graph 仍在引用的块。`bfdiag/sensitivity/perturbations.py` 把
`empty_cache` 列入 `FORBIDDEN` 并在报错里写明原因,避免下一个人踩。

## 5. kernel 层:未定位(重要:不要据此给 sparkinfer 定罪或免罪)

固定内容 / `cache_seqlens` / `page_table` / `window_left`,只在前面垫 0–1024 MiB 再跑
sparkinfer paged attention:两个真实层组(full_attn 48/8 wl=-1、swa 72/8 wl=511)在 6 种布局下
输出**逐位相同**。

**但这个测试不够强,结论不可当作"sparkinfer 无罪":**
- 实测地址列显示 `k` 的地址在 6 次里**完全没变**(垫的内存被分到别处),只有 `q` 挪了一次。
  它只证明了"对这几种地址变化不敏感"。
- 它是**单次 decode + 全新 workspace**,而真实分叉发生在**多轮有状态累积**的路径上。

**还没测的、机制上更可疑的**:`runtime/backends/laguna.py:185-187` 仍调用
`init_flashinfer_workspace()` → `vllm.v1.worker.workspace.init_workspace_manager`,
这是一块共享 scratch;NVFP4 dense GEMM(约 304 次调用)也仍走 FlashInfer
(见 `runtime/nvfp4_cutlass_direct_patch.py:3`),除非启用 `QSR_A2_*` 补丁。

## 6. 三层图景

| 层 | 内容 | 归属 | 状态 |
|---|---|---|---|
| 触发源 | 588 MiB 死 lm_head 引用环(以及任意其它分配扰动) | vLLM | ✅ 已确认 |
| **放大器** | **bs=128 的对齐余量把计算推进刀锋区** | **我们的代码** | ✅ 已确认(bs=64 完全免疫) |
| 敏感性本身 | kernel/workspace 输出随分配布局变化 | sparkinfer / FlashInfer | ⚠️ 现象确认,未定位 |

**触发源可以有无数个,换掉 vLLM 只是少一个。真正该修的是第 2、3 层。**
仅凭本调查**不足以**把"剥离 vLLM"的优先级提上去 —— 该结论需要先坐实第 3 层是否在 FlashInfer/vLLM workspace。

## 7. 对测量方法的直接影响

- **「bs=128 的接受率是 X」这种表述不成立**,必须连同扰动条件一起报告,或报告一个集合。
- 真实退化幅度在 **0.043 ~ 0.266** 之间(bs=64 的 0.718182 对 bs=128 的三个值),不是单一数字。
- 这**支持**根因笔记的结论:对齐余量把 bs=128 推进敏感区,而敏感区内微小扰动会级联放大
  (与实测的"浅层 0.0156 → 深层 20.0"一致)。
- 任何 A/B 都必须**固定扰动条件**;`bf sensitivity sweep` 就是用来先确认"这个配置稳不稳"的。

## 8. 需要 GPU 的后续待办

1. 测 `init_flashinfer_workspace` / FlashInfer NVFP4 GEMM 是否地址敏感(**优先**,坐实第 3 层归属)。
2. 用真正推开 K/V cache 地址的方式重做 kernel 隔离测试(本次 `k` 地址未变,测试偏弱)。
3. 定位那个死 lm_head 具体属于哪个模块(target 还是 DFlash draft),确认能否在构造后显式释放。
4. bs=128 扫更多扰动,确认值域是 3 个还是更多。
5. 修掉对齐余量后重测:bs=128 是否恢复到 bs=64 那样的免疫状态 —— **这是判断修复是否有效的判据**。
