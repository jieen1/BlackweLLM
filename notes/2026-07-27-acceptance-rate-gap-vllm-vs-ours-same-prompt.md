# 关键发现:相同 prompt 下,vLLM 接受率 100% vs 我们 68.7%——差距的真正来源(2026-07-27)

## 结论先行

**用我们自己引擎的确切重复短语("The quick brown fox jumps over the lazy dog. ",
和"near the river bank"版本不同)去跑 vLLM 原生 DFlash,vLLM 测出 100% 接受率
(720/720 draft token 全部接受),accepted_tok_s=383.3。我们自己引擎在完全相同的
prompt/上下文长度/K/greedy 设置下只有 68.7% 接受率。这不是 prompt 选择的问题——
是同一个 prompt,两边接受率差了 31 个百分点。**

## 数据

`benchmarks/fixtures/laguna_vllm_dflash_baseline_matched_text_20260727.json`:

```json
{
  "accepted_tok_s": 383.3,
  "acceptance_rate": 1.0,
  "spec_decode_draft_tokens_delta": 720,
  "spec_decode_accepted_tokens_delta": 720
}
```

对照我们自己(`benchmarks/fixtures/dflash_ab_verify_cg_after_fillbuffers_fix_20260727.txt`,
同一个 `BASE_TEXT = "The quick brown fox jumps over the lazy dog. "`,同样 64K、K=15、
greedy):`acceptance_rate=0.6869565217391305`,`tok_per_s=252.89/259.14`。

## 为什么这个发现推翻了"接受率匹配后我们已经打平"的结论

上一篇笔记(`notes/2026-07-27-dflash-fair-comparison-vllm-parity.md`)用"两边都在
~99% 接受率区间"这个前提,得出"每轮耗时几乎一致,我们已经打平"的结论——但那次比的是
vLLM 用**它自己的默认短语**("...near the river bank.")测出的 99.22%,和我们自己用
**我们的短语**("...lazy dog.")测出的 68.7%,当时以为"两边短语相似,接受率应该也
相似"是合理假设。**这次直接控制变量——用我们的确切短语喂给 vLLM——证明这个假设是
错的:vLLM 在这个短语上是 100%,不是介于两者之间或者向 68.7% 靠拢。**

也就是说:**vLLM 的 DFlash 在"重复短语续写"这类任务上,不管具体是哪个重复短语,都能
稳定拿到接近满分的接受率;我们的引擎在完全相同的输入下明显更容易"跟不上"重复模式,
接受率掉到 68.7%。这是一个真实的、两边行为不一致的现象,不是测量口径问题。**

## 这意味着什么(为阶段1定调)

结合之前的两个发现:
1. MoE/dense-GEMM kernel 已经打满/接近打满显存带宽,没有大空间。
2. 如果我们的接受率也能到 100%(和 vLLM 一样),按 44.16ms/round、K=15 反推:
   16 个 token/round ÷ 0.04416s ≈ **362.3 tok/s**——已经和 vLLM 的 367.3/383.3
   在同一量级,不再有"kernel 效率差"这回事。

**结论:阶段1"追上 367 tok/s"这个目标,瓶颈几乎完全在接受率,不在 kernel 速度。**
这把优化方向从"继续抠 kernel"整个转移到"为什么我们的投机解码在这个场景下接受率更低,
能不能提高"——一个和这次任务(kernel 级 profiling/带宽打满率检查)完全不同性质的问题,
需要专门的、深入到数值/模型行为层面的调研(不是我这次任务的方法论能覆盖的范围)。

## 可能的原因(未验证,记录下来供后续调研参考,不要当结论用)

1. **MoE 数值路径不同**:我们用 sparkinfer 的 `deterministic_output=True`(为 CUDA
   Graph 捕获做的确定性修复,`989723d`),vLLM 用 FlashInfer CUTLASS MoE,两条路径的
   浮点求和顺序/舍入行为不同。历史调研(`STATUS_dflash_acceptance.md`,"数值漂移"章节
   最终被推翻,但推翻的是"64K 固有解释",没有排除"两个不同 MoE kernel 数值路径不同"
   这个更窄的可能性)。
2. **Draft 模型集成方式不同**:vLLM 是官方 stock DFlash 实现,我们是自己重新集成的
   draft 模型 forward + aux hidden states 组合逻辑,细节实现差异可能影响 draft 预测
   质量本身(不是 verify 端的问题,是 draft 端预测得准不准)。
3. **纯粹是这类"复读机"式重复文本在长上下文下的混沌效应**(历史笔记提到的"TheOkay"
   现象——主模型在 65568 位置附近开始偏离重复模式,是模型的真实行为,不是 bug)——
   但如果这是真的,vLLM 跑同一个模型/同一个 prompt 不应该稳定 100%,除非 vLLM 的数值
   路径恰好没有触发这个偏离点,或者触发点位置因数值路径不同而不同。

## 建议

这个发现的重要性和方向都超出了这次"kernel 级带宽检查"任务的范围,需要协调者/用户决定
是否要单独立项调研("为什么两个引擎在相同输入下接受率不同"),以及要不要用逐层 logits
对比(类似历史上验证 BFAttention"cos=0.999999"那次的方法论)去定位具体哪一层/哪个算子
先开始分叉。这次任务到此为止,不擅自展开数值级调试。
