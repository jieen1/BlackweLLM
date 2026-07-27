# 给 sparkinfer 团队的具体任务:把 Laguna kernel 特化从 num_kv_heads=4 泛化到 8

(本仓库不直接改 sparkinfer 代码,这是完整背景+具体位置+建议方案,交给用户转 sparkinfer 开发)

## 背景

`_paged_determine_cta_tile_q`(`planner.py`)本身用 `packed_qo_len = qo_len * gqa_group_size`
判断,`gqa_group_size` 在 TP=1/TP=2 下不变(6 用于全注意力,9 用于 SWA),所以这一层
调度选择**已经对 TP 度数无关**,理论上直接适用于我们的 TP=1 生产配置。

但真正决定"是否走快速特化路径"的是另外 9 处硬编码精确匹配条件,**同时**检查绝对头数
(`num_q_heads`/`num_kv_heads`),不是只看 `gqa_group_size`:

| 文件 | 行号(2026-07-27 master@8db352c) | 场景 |
|---|---|---|
| `attention/paged/traits.py` | 389 | extend, window_left=511 (SWA), q=36/kv=4/gqa=9 |
| `attention/paged/traits.py` | 417 | extend, window_left<0 (全注意力), q=24/kv=4/gqa=6 |
| `attention/paged/traits.py` | 445 | decode, split_kv, window_left<0, q=24/kv=4/gqa=6 |
| `attention/paged/traits.py` | 465 | decode, split_kv, q=36/kv=4/gqa=9 |
| `attention/paged/traits.py` | 484 | verify, split_kv, window_left<0, q=24/kv=4/gqa=6 |
| `attention/paged/_forward.py` | 446 | `_use_laguna_verify_analytic_kernel`,同上verify条件 |
| `attention/paged/_forward.py` | 485 | `_use_laguna_decode_analytic_kernel`,同上decode条件 |
| `attention/paged/_forward.py` | 1491 | (第三处,同一家族的另一个变体) |
| `attention/paged/workspace.py` | 1040 | CUDA Graph capture 条件判断的同款 gate |

**每一处都要求 `num_kv_heads == 4` 且 `num_q_heads == 24`(全注意力)或 `36`(SWA)**——
这些是 TP=2 切分后的绝对值。我们生产是 TP=1,真实形状是**全注意力 48 Q头/8 KV头
(gqa=6)、SWA 72 Q头/8 KV头(gqa=9)**——`gqa_group_size` 完全匹配,但绝对头数是
TP=2 值的整整两倍,9 处检查全部不触发,我们走的是通用回退路径,吃不到这些精调
kernel 的收益。

## 建议的泛化方案

把这 9 处的 `num_q_heads == 24/36 and num_kv_heads == 4` 替换成同时接受 TP=1
(`num_q_heads == 48/72 and num_kv_heads == 8`)和 TP=2(现状)两组值——或者更彻底地,
只保留 `gqa_group_size == 6/9` 这一个判断条件,去掉绝对头数检查(如果这些 kernel 的
资源分配公式本身是用 `gqa_group_size` 参数化的,不是写死用绝对头数算的常量)。

## 已知的风险点,不是"改一行就完事"

这些不是简单的路由开关,而是给特定 KV 头数量精确调过的资源分配:每一处都伴随
`exact_num_mma_kv`(精确 KV MMA 数)、`minimum_shared_storage_bytes`(共享内存大小)、
`compact_sync_rows` 这类具体数值,这些是基于 num_kv_heads=4 时每个 CTA 处理的
独立 KV 头分组数量算出来的。**num_kv_heads 从 4 变到 8,独立 CTA 分组数量翻倍**,
以下需要重新验证,不能假设直接复用同一组常量:

1. `minimum_shared_storage_bytes` 等资源占用公式是否还成立(共享内存/寄存器压力
   随 KV 头分组数变化)。
2. **grid 级别的 occupancy 调优**(比如日志里提到的 `resolve_decode_graph_ctas_per_sm`
   这类函数)——CTA 组数量翻倍后,每 SM 能驻留的 CTA 数量、warp 调度可能需要
   单独重新调优,不能直接套用 kv=4 时调好的参数。
3. 需要针对 kv=8 跑一遍完整的正确性回归(参照 `select_paged_forward_traits_from_plan`
   现有的测试模式)+ 性能基准,不能只改条件就直接上生产。

## 我们这边能做的验证(不需要改 sparkinfer 代码)

一旦 sparkinfer 团队落地这个泛化,我们这边可以直接用现有的 kernel 隔离测试方法论
(`isolate_kernel_test_v2.py` 那一套,今天 block_size 排查里已经验证过可靠)测出
真实收益:构造 num_kv_heads=8、真实 Laguna 形状(48/8 或 72/8)的输入,对比走
特化路径前后的耗时/带宽打满率,不需要等完整引擎集成就能拿到数字。
