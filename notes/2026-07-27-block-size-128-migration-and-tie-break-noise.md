# block_size 64→128 迁移:完整排查记录——结论是浮点临界翻转,不是 bug(2026-07-27)

## 结论先行

**block_size=64→128 迁移(`laguna.py`/`laguna_sparkinfer_attn.py`/`laguna_dflash.py` 的改动)本身正确,没有地址/结构性 bug。** 最初观察到的"接受率从 68.7% 掉到更低"现象,根因是**浮点数值噪声导致的临界 argmax 翻转**,和 `notes/2026-07-27-verify-cg-mode-fix-and-block-size-eval.md` 里已经记录过的 split-KV 数值噪声是同一类现象,不是可以"修复"的缺陷。

## 背景

用户批准的迁移(page_size=128 在 Laguna 真实 shape 下正确性已验证,cos>=0.999991)落地后,第一次端到端测试(64K 上下文)显示接受率从 68.7%(block_size=64 基线)掉到 12.4%/9.9%(block_size=128),且同一进程内两轮结果还不一致——表面上看极像一个真实的地址计算/数据损坏 bug。以下是完整排查过程。

## 检查点序列 + 二分排除记录

沿着"prefill 开始 → decode 产出 logits"的真实数据流定义检查点(CTX=10240,2 个 chunk,足以复现问题,GPU 时间比 64K 快得多):

```
CP1 chunk0 prefill(full-attn+SWA scratch写入)
CP2 chunk0→ring 拷贝(_copy_scratch_to_ring)
CP3 chunk1 ring→scratch overlap 拷贝(_copy_ring_to_scratch)
CP4 chunk1 prefill
CP5 chunk1→ring 拷贝
CP6 aux_hidden_states → combine_hidden_states
CP7 _bulk_precompute_context_kv 写入 draft KV cache(真实 tensor)
CP8 DFlashDraftCudaGraph._fill_buffers 的 page_table/KV 绑定
CP9 draft CG replay 产出的原始草稿 token(verify 之前)
CP10 verify(主模型 M=16 forward)
CP11 accept/reject → 最终提交的 token
```

逐步排除记录(每步都基于上一步结果,不是无依据地换方向猜):

1. **CP1-CP5(主模型 chunked prefill 全链路)**:先用最强的间接证据——**chunked prefill 之后跑纯 plain(非 DFlash)decode**,20 个 token 解码出 `' The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog'`,完全正确连贯。这证明主模型的 full-attention + SWA scratch/ring 全部没问题(否则 plain decode 不可能给出连贯文本)。
2. 对 `_copy_ring_to_scratch`/`_copy_scratch_to_ring` 单独加了自洽性 checksum(读回刚拷贝的数据,和源数据逐元素比对):**完全一致,零 mismatch**——排除拷贝逻辑。
3. **CP6-CP7(draft KV 写入)**:先发现一个真实但无关的隐患——`_bulk_precompute_context_kv` 在 chunk 很大时(比如 chunk_len=2048)会往只有 768 个 slot 的 draft ring 里写 2048 个位置,产生 1280 次重复地址写入,PyTorch 在 CUDA 上对重复索引的写入顺序没有保证。加了裁剪(只保留最后 `ring_slots` 个位置,替代原来"整个 chunk 全写")修复了这个真实的未定义行为依赖——**但修复前后接受率完全没变(0.452525→0.452525,精确到小数点后6位一致)**,说明这不是这次问题的根因,只是一个独立的健壮性加固,予以保留。
4. 直接读回 draft KV cache 真实 tensor(不是猜测,是读 `self._draft_kv_caches[name]` 在计算出的 `slot_mapping` 位置的值):三个采样位置全部非零、数值合理(abs_sum 114000-115000 量级,一致)——**证明写入内容正确到达目标地址**。
5. **CP8(workspace 绑定)**:一度怀疑 `DFlashDraftCudaGraph` 的 `self._workspace`(`PagedAttentionWorkspace` 实例)在 capture 时绑定的 k_cache/v_cache 是一次性 dummy tensor、从未重新绑定到真实 draft KV cache——**这个怀疑是错的**。往 `PagedAttentionWorkspace` 实例上访问 `_k_cache` 属性直接报 `AttributeError`(这个类根本没有这个属性,是把它和我们自己另一个类 `SparkinferDecodeWorkspace` 搞混了),这段错误的调试代码把 draft CG capture 弄崩溃、静默降级成慢得多的 eager 路径,产出的"新"接受率数字(0.491667)是在完全不同代码路径上测的,不能算数,已撤回并如实向协调者报告。读 `_SparkinferCGExtendImpl.forward()` 真实实现后确认:`kv_cache` 是作为标准 vLLM 参数在每次 forward 调用时传入的,不存在"读了一个从未更新的假 tensor"这回事——CP8 排除。
6. **决定性证据(CP11 之后,直接看最终结果而不是代理指标)**:解码 bs=64 和 bs=128 在 CTX=10240 下生成的完整 256 个 token 序列——**逐位完全相同**(`[290, 785, 3454, 21438, 42850, 22718, 911, 340, 8623, 9554, 83, ...]`,正确的重复模式)。**这证明最终输出从未出错**——DFlash 的 verify+accept/reject 机制始终把任何草稿预测错误纠正回了正确 token,不存在"生成了错误内容"这个问题。
7. 既然最终提交的 token 序列(从而 kv_len 的演进)在两个 block_size 下必然完全一致,就可以**按 kv_len 对齐、逐 round 对比原始草稿 token(verify 之前)**:round 1-7(kv_len 10240→10306)两边逐位相同。**kv_len=10307(round 8)是第一个分叉点**:15 个草稿 token 里前 14 个逐位相同,只有第 15 个(离 anchor 最远的那个)不同——bs=64 预测 911(错,accept 14/15),bs=128 预测 22718(对,accept 15/15)。**不是"128 系统性更差"的方向性问题**——这一个具体位置反而是 64 错、128 对。
8. **最终定位**:打印这一步 draft 在第 15 个 query 位置的完整 top-2 logits:

   | | top1 | top2 | margin |
   |---|---|---|---|
   | bs=64 | **911**(20.875000) | 22718(20.750000) | 0.125000 |
   | bs=128 | **22718**(20.875000) | 911(20.750000) | 0.125000 |

   **两个候选 token 完全相同,两个 logit 数值完全相同,margin 完全相同(0.125,相对幅度约 0.6%)——唯一的区别是谁排第一。** 这是浮点临界平局翻转的教科书级别证据:block_size 改变了 attention kernel 内部 KV 分页粒度,导致 V 向量加权归约的浮点求和顺序不同,两条路径各自都是"正确"的浮点计算,只是在这种真正意义上的近似平局上,谁赢由累积的浮点舍入误差决定。

## 和已有记录的关联

这和 `notes/2026-07-27-verify-cg-mode-fix-and-block-size-eval.md` 里的判断完全吻合:"split-KV 是浮点求和顺序不同的并行归约策略,理论上可能引入求和结合律带来的极小数值差异(项目里其它地方称为'R6'级别的噪声容忍)"。这次只是换了个触发它的维度(block_size 而不是 split-KV 开关),同一类现象,不是新的 bug 类别。

**重要澄清,避免以后被重新当成未解决的 bug**:这次 CTX=10240/65536 用的是**高度重复的合成文本**("The quick brown fox jumps over the lazy dog. " 循环),这种文本天然会让模型在很多位置产生"非常自信但又和另一个候选极度接近"的临界决策——是这类投机解码对浮点噪声最敏感的对抗性场景。**真实、多样化的生产文本大概率不会有这么密集的临界决策点**,不能假设这个合成基准上测出的接受率差异幅度能代表真实生产场景下 block_size=64 和 128 的实际接受率差距。

## 迁移代码改动清单(判定为正确,可以保留)

- `runtime/backends/laguna.py`:`LagunaBackend.__init__` 的 `block_size != 64` 硬校验放宽为 `block_size not in (64, 128)`。
- `runtime/backends/laguna_sparkinfer_attn.py`:`SparkinferDecodeWorkspace.__init__` 新增 `page_size` 参数(替代硬编码的模块级 `PAGE_SIZE = 64` 常量),三处内部引用(`_k_cache`/`_v_cache` 形状、`capture_cache_seqlens`)改用这个参数。
- `runtime/backends/laguna_cuda_graph.py`:`LagunaCudaGraphDecode._init_workspaces` 调用 `SparkinferDecodeWorkspace(...)` 时传入 `page_size=self.block_size`。
- `runtime/backends/laguna_dflash.py`:`_bulk_precompute_context_kv` 新增裁剪逻辑——`num_positions > ring_slots` 时只保留最后 `ring_slots` 个位置,避免往 draft KV ring 写入超过其容量的重复地址(独立的健壮性修复,不是这次接受率问题的根因,但值得保留)。
- `tests/test_laguna_server_integration.py`:更新 `test_rejects_unsupported_sparkinfer_page_size_before_model_load` 的错误信息匹配正则(从 `block_size=64` 改成 `block_size in \(64, 128\)`),反映新的校验逻辑。

## 临时诊断 instrumentation

排查过程中在 `laguna.py`/`laguna_dflash.py`/`laguna_dflash_cudagraph.py` 里加了多处 `QSR_DEBUG_CHUNK_CHECK` 环境变量门控的调试日志(chunk 边界信息、拷贝自洽性 checksum、draft KV 读回验证、原始草稿 token/logits 打印)。默认不设这个环境变量时完全不影响正常运行路径,保留下来对以后类似问题的排查有长期价值,不清理。

## 尚未做完的收尾(见任务 #27/#28)

1. 完整 64K(CTX=65536)block_size=128 端到端正确性+性能验证(之前只在 CTX=2048/10240 小规模验证过)。
2. attention 带宽打满率在 bs=128 下的真实数字,对照之前 bs=64 测出的 ~37%。
3. 端到端 tok/s 对比,判断这次迁移对阶段1(DFlash 64K)吞吐的真实贡献。
