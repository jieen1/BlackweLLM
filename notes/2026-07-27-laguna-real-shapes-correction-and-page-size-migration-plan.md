# Laguna 真实 attention 形状纠正 + block_size 64→128 迁移完整方案(2026-07-27)

## 背景

之前调研 sparkinfer master 的 Laguna kernel 特化时,一直沿用 `laguna_sparkinfer_attn.py`
文档注释里的"24 Q头/8 KV头"这个形状描述来评估收益(测过 kv_heads=4 形状下 2.1x/1.4x
的加速比)。这次深挖 attention 只打满 ~37% 带宽的根因时,直接翻查了
`~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4` 的真实 safetensors
权重 tensor shape(不是配置文件字段),发现这个描述是错的。

## 纠正 1:真实 attention 形状

直接读权重 tensor:

- layer 1(SWA)`q_proj.weight=[9216,3072]` = 72×128 头,`k_proj.weight=[1024,3072]` = 8×128 头
- layer 4(全注意力)`q_proj=[6144,3072]` = 48×128 头,`k_proj` 同样 8×128 头

**真实形状:全注意力层 48 Q头/8 KV头(gqa_group_size=6);SWA 层 72 Q头/8 KV头
(gqa_group_size=9)。** 不是文档里写的"24 Q头/8 KV头"。`laguna_sparkinfer_attn.py`
顶部文档注释已经改过来。

## 纠正 2:生产环境是 TP=1,不是 TP=2

`docs/roadmap.md` 明确记录"B6 多GPU可行性评估 | 仅评估不实施 | 远期/M4"——
`qwen-sm120-runtime` 现在没有实现张量并行,这台机器也只有 1 张卡。**生产环境现在
就是 TP=1**,所以上面纠正的 48/8、72/8 就是运行时真实看到的形状,不是"需要再除以
TP 的中间值"。

## 收回:之前"sparkinfer Laguna 特化能直接用"的判断

上游 sparkinfer master 那 5 个 Laguna 硬编码 kernel 特化,精确匹配条件要求
`num_kv_heads==4`(TP=2 切分后的值)。生产环境 TP=1、`num_kv_heads=8`,**一个都对
不上,一个都没在生产路径里生效**。之前报告的 2.1x/1.4x 加速比数字是真实测出来的,
但测的是 kv_heads=4 这个下游根本不会跑的形状——**收回"这些加速现在就能用"的说法**。

## page_size 64→128:为什么卡在 64、能不能改、完整迁移方案

### 为什么卡在 64

纯粹是下游(本仓库)自己的限制,不是 sparkinfer 的限制。`runtime/backends/laguna.py:125`
硬编码校验 `if block_size != 64: raise ValueError(...)`,注释写"sparkinfer paged
attention 固定 64-token page layout"。这条限制来自 commit `e66d254`(2026-07-26,
"sparkinfer paged planner requires 64-token pages")——**当时的 sparkinfer 版本确实
只支持 64,但现在 sparkinfer(master @ 8db352c)的 planner/workspace/_scratch/
_forward/traits 全模块统一支持 `page_size in (64, 128)`,128 不是实验性的,反而是
最新 Laguna 调优的落点**。

### 能不能改:已用真实形状验证正确性

用 Laguna 真实形状(48/8 全注意力、72/8 SWA,`window_left=511`)分别在 page_size=64
和 128 下跑 extend、decode 两种模式,对比 sparkinfer 自己的参考实现:

| 场景 | page_size=64 | page_size=128 |
|---|---|---|
| 全注意力 extend | cos=0.999991 | cos=0.999992 |
| 全注意力 decode | cos=0.999991 | cos=0.999992 |
| SWA extend | cos=0.999992 | cos=0.999991 |
| SWA decode | cos=0.999993 | cos=0.999993 |

**结论:page_size=128 在 Laguna 真实形状下和 64 一样正确,误差在噪声范围内,没有
精度问题。**

### 具体迁移方案(下游侧,不涉及 sparkinfer 代码改动)

1. 去掉/放宽 `LagunaBackend.__init__`(`runtime/backends/laguna.py:125`)的
   `block_size != 64` 校验。
2. `blocks_per_slot`(现在 1088,按 bs=64 算是 69,632 token 容量)要重新算成
   ~544,保持每 slot 的 token 容量不变——沿用 `e66d254` 自己定的原则
   `blocks_per_slot = ceil(ctx / block_size)`。**不改的话每个 slot 显存直接翻倍**。
3. SWA ring buffer 数学(`_ring_blocks_for_window`)和 KV cache tensor 分配本来就是
   按 `block_size` 参数化的,代码不用改,重新跑数值验证即可。
4. 这个改动牵扯到 CUDA graph capture 的地址假设,需要走一次完整的正确性(接受率
   精确不变)+ 性能回归,但不是开放式的结构重写。

### 重要预期管理:迁移到 128 不会自动拿到那 5 个特化

即使把 page_size 迁移到 128,**还是吃不到 sparkinfer 那 5 个 Laguna 硬编码特化**,
因为它们同时要求 `num_kv_heads==4`,生产环境是 8。page_size=128 本身可能仍有独立
于这 5 个特化的收益(比如更少的 page table 项、更好的 kernel 调度粒度),但不能
假设"迁移到 128 = 拿到之前测的 2.1x/1.4x"。真实收益需要用同样"实测带宽反推打满率"
的方法论在迁移后重新测量,不能靠外推。

## 给 sparkinfer 团队的具体任务建议(未来动作,本仓库不直接改 sparkinfer 代码)

价值最高、最具体可落地的 sparkinfer 侧任务:**把 `traits.py` 里那些精确匹配条件
泛化,同时覆盖 TP=1 的真实形状(48/8、72/8),判断依据用 `gqa_group_size`(6/9,
TP=1 和 TP=2 下这个比值相同)而不是绝对头数**。

关键调优逻辑核实过:`cta_tile_q` 的选择用的是 `packed_qo_len = qo_len *
gqa_group_size`(不是绝对头数),所以每个 CTA 内部的调优大概率能直接照搬到
`num_kv_heads=8` 的形状,不需要重新调。唯一没法保证的是**整网格的 occupancy 调优**
(比如 `resolve_decode_graph_ctas_per_sm` 这类)——`kv_heads` 从 4 变到 8,独立 CTA
组数量翻倍,这部分需要单独重新调优验证。

这条建议已经完整交给用户,由用户决定是否转给 sparkinfer 开发团队处理;本仓库这边
不会直接修改 sparkinfer 源码。
