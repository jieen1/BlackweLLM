# DFlash Acceptance Rate Investigation (2026-07-26)

## Current Status
- Acceptance rate: **15%** (target: >50%)
- E2E correctness: ✅ "The capital of France is" → " Paris"
- Verify path consistency: 88% match with sequential decode (14/16 positions)

## Bugs Found & Fixed

### 1. CG Capture Impl Leak (CRITICAL) ✅ FIXED
- `LagunaCudaGraphVerify.capture()` and `LagunaCudaGraphDecode.capture()` patched
  all 48 main model attention layers to `_SparkinferCGExtendImpl` but never restored them
- After DFlashEngine init, the eager attention path produced garbage
- **Fix**: Call `unpatch_impls()` after capture (commit c90b009)
- **Impact**: 6.7% → 12.7% acceptance

### 2. Draft Position Offset ✅ FIXED
- `generate_verify_only` called `_draft_forward(slot, bonus, kv_len + 1)` but draft KV
  cache only had positions 0..kv_len-1 from precompute — position kv_len was skipped
- **Fix**: Initial draft uses `kv_len`, subsequent uses `new_kv_len - 1` (commit c90b009)
- **Impact**: 12.7% → 15% acceptance

## Verified Correct
- Main model prefill/decode: ✅ " Paris" with coherent 50+ token output
- BFAttention module replacement: ✅ cos=0.999999 vs SDPA reference
- FP8 KV write/descale: ✅ correct
- MoE determinism: ✅ SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1
- Aux hidden states: ✅ per-slice RMSNorm in combine_hidden_states handles scale differences
- Verify path (qo=16): 88% consistent with sequential decode (14/16 match)

## Under Investigation
- **Draft model prediction quality**: Draft produces semantically related but positionally
  wrong tokens. E.g., draft=['.', '\n\n', '###', ' capital', ' the'] vs
  expected=['.', '\n', 'In', ' a', ' discussion']
- Possible causes:
  1. Draft KV cache population via `precompute_and_store_context_kv`
  2. Draft attention metadata (ring buffer mapping for SWA window=512)
  3. Draft model's KV scatter during forward (vLLM 0.26.0 `unified_kv_cache_update`)
  4. Position encoding mismatch between draft and main model

## 2026-07-26 Code Diagnosis Update

### Verify-only state transition bug (fixed in current worktree, GPU pending)

`generate_verify_only` mixed two different quantities under `num_accepted`:

- number of matching draft tokens;
- number of verifier inputs whose KV may be committed (old anchor + matches).

The old helper also counted a target recovery token as an accepted draft. After the
first rejection this caused the target KV length, next anchor, draft-context KV, and
next draft position to disagree. The full-accept branch additionally failed to emit
the verifier's final bonus token.

The current worktree now uses the same strict acceptance contract as
`runtime/mtp_accept.py`:

- `num_accepted` counts matching draft tokens only;
- emitted tokens are matching drafts plus exactly one recovery/bonus token;
- committed verifier context is `1 + num_accepted` positions (old anchor + matches);
- the recovery/bonus becomes the next pending anchor;
- the next draft starts at the updated target `slot_kv_len`.

CPU evidence:

```text
CUDA_VISIBLE_DEVICES='' /home/bot/.venvs/vllm/bin/python -m pytest -q \
  tests/test_mtp_accept.py tests/test_dflash_engine.py
26 passed in 1.11s
```

This correction changes the acceptance metric to strict draft acceptance. The old
reported 15% included recovery tokens, so it was inflated and should not be compared
directly with the next GPU result.

### Regression boundary to isolate next

The state bug predates the historical high-acceptance runs, so it is a correctness
bug but does not by itself explain the entire 55% -> 15% regression. The highest-value
regression boundary is commit `8e5c504`, which changed three draft components together
without an acceptance-rate validation:

1. FlashInfer draft attention -> Sparkinfer paged extend;
2. backend-shaped draft KV -> self-allocated ring KV;
3. draft KV storage -> hard-coded FP8 `uint8`.

Run the next GPU diagnosis in one process and serially:

1. Disable DFlash CUDA Graph and prefix cache.
2. Inspect only the first draft/verify round at the correct initial position
   `kv_len == prompt_len`; this is before accept/reject state can corrupt later rounds.
3. Compare Sparkinfer draft attention against an SDPA/BF16 reference layer by layer,
   then compare FP8 vs BF16 draft KV while keeping positions and aux states identical.
4. Only after first-round parity, run a multi-round trace and confirm after each reject:
   target KV delta=`1 + accepted_drafts`, next anchor=`recovery`, and next draft
   position=`slot_kv_len`.

Do not use the older `/tmp/test_dflash_diag*.py` results as position evidence: those
scripts call the initial draft with `kv_len + 1`.

### Prefix-cache-specific bug

`generate_verify_only` currently zeros the entire draft KV on every request, including
prefix-cache hits, then reconstructs only the uncached suffix aux states. On a prefix
hit the main KV retains the prefix but the draft KV does not, so draft attention loses
its cached context. This is independent of the simple `enable_prefix_cache=False`
acceptance test and should be fixed after first-round eager parity.

## Architecture Notes
- Main model: 48 layers (12 full + 36 SWA-512), NVFP4, BFAttention + sparkinfer
- Draft model: 6 layers (all SWA-512), bf16, vLLM Attention + SparkinferAttentionImpl
- Aux layers: [2, 11, 20, 30, 39, 48] → target_layer_ids [1, 10, 19, 29, 38, 47]
- Draft KV: self-allocated uint8 (FP8), ring buffer for SWA window
- combine_hidden_states: per-slice RMSNorm → concat → fc → hidden_norm

## Test Scripts
- `/tmp/test_dflash_generate.py` — DFlash generate_verify_only E2E
- `/tmp/test_verify_vs_decode.py` — verify (qo=16) vs sequential decode comparison
- `/tmp/test_full_dflash_init.py` — DFlash init + single draft/verify step
- `/tmp/test_isolate_corruption.py` — isolate which DFlash init step corrupts main model

## 2026-07-26 GPU 验证结果（状态机修复后）

### A/B 对比（同 prompt、同参数、4K 上下文）

| 引擎 | 接受率 | tok/step | tok/s | 模式 |
|------|--------|----------|-------|------|
| BlackweLLM (sparkinfer, 修复后) | **86.3%** | **13.42** | 11.8 | eager |
| BlackweLLM (sparkinfer, 修复前) | 35.4% | 5.20 | 9.3 | eager |
| vLLM 0.26.0 (MARLIN, DFlash) | — | — | 190.8 | eager |
| Baseline b056100 (MARLIN) | 51.2% (64K) / 55.1% (128K) | 7.5 / 8.2 | 77.6 / 210.9 | CG |

**结论：55% 接受率不仅可以复现，修复后在 4K 即达 86.3%。**

### 64K 上下文退化（新发现）

64K 接受率仅 19%（3.81 tok/step），远低于 4K 的 86.3%。

逐轮诊断（/tmp/diag_draft_rounds.py）：
- Round 0-1: 15/15 (100%) — draft 完美预测
- Round 2: 1/15 — 首次 reject 后 verify 产生异常 token
- Round 3-7: 15/15 — 恢复
- Round 8-9: 0-2/15 — verify 开始产生完全错误的 token（"Okay" 而非 "The"）
- Round 10-19: 0-4/15 — 模型完全偏离重复文本模式

### 根因定位：decode 和 verify 路径在 64K 下不一致

验证实验（/tmp/diag_verify_consistency.py）：
- 64K prefill 后，sequential decode（M=1）产生 " decoding" × 16（卡死循环）
- verify forward（M=16 extend）产生 " offers offers... decoding..."（前 3 个不同）
- 一致性：12/15 (80%)

**sequential decode 在 64K 下完全坏了**——每次 decode 都产生相同 token，
说明 KV cache 写入或 attention 读取在长位置下有 bug。

### 待排查方向

1. **sparkinfer extend kernel 在大 seq_lens 下的正确性**
   - decode (M=1) 和 extend (M=16) 使用同一 SparkinferAttentionImpl.forward()
   - 但 metadata 不同：decode seq_lens=65537 vs extend seq_lens=65552
   - 需要确认 sparkinfer 在 seq_lens > 64K 时是否有 int32 溢出或 plan 错误

2. **SWA ring buffer 在 decode 模式下的 block table 正确性**
   - decode_ring: seq_lens=513, 9 ring blocks
   - verify_ring: seq_lens=528, 9 ring blocks
   - 需要确认 ring block table 在 pos > 64K 时的 modulo 映射

3. **full attention block table 在 decode 模式下是否溢出**
   - n_blocks = ceil(65537/64) = 1025
   - blocks_per_slot = 1152（足够）
   - 但 _decode_block_table 的预分配大小需要确认

### 测试脚本

- `/tmp/test_acceptance_repro.py` — 接受率复现（4K/64K）
- `/tmp/test_vllm_acceptance.py` — vLLM 对比（moe_backend=marlin）
- `/tmp/diag_draft_quality.py` — draft 预测质量诊断
- `/tmp/diag_draft_rounds.py` — 20 轮 draft/verify 逐轮追踪
- `/tmp/diag_verify_consistency.py` — verify vs sequential decode 一致性

### vLLM 依赖修复记录

- `torchvision` 缺失 → `pip install torchvision --no-deps`
- `quack-kernels` 0.4.1 与 `nvidia-cutlass-dsl` 4.6.x 不兼容
  （`cute.core.ThrMma`/`ThrCopy` 缺失）→ sed 替换为字符串注解
- vLLM 0.26.0 需要 `moe_backend="marlin"` 绕过 FlashInfer MoE API 不兼容

## 2026-07-26 污染测试结果（关键转折）

### 实验设计
在 Round 2 首次异常（kv_len=65568, accepted=1/15）处：
1. 记录 sustained verify argmax
2. Reset slot，重新 prefill prompt + committed tokens 到相同 kv_len
3. 用完全相同的 verify_tokens 做 fresh forward

### 结果
```
Sustained: '  TheOkay brown fox jumps over the'
Fresh:     '  TheOkay brown fox jumps over the'
Matches: 16/16
→ 无 KV 污染
```

### 结论更新

**"TheOkay" 是模型在 65568 位置的真实预测，不是 bug。**

- 主模型 KV 缓存在 DFlash 循环中完全正确
- 4K 下主模型完美跟随重复文本模式（15/15 draft 匹配）
- 64K 下主模型开始偏离模式（产生 "Okay" 等非模式 token）
- Draft 模型预测重复模式，但主模型不跟随 → 接受率下降

### 根因重新定位

55%→20% 的回归不是逻辑 bug，而是 **数值漂移**：
- Baseline 用 Marlin MoE + FlashInfer attention
- 当前用 sparkinfer MoE + BFAttention (sparkinfer attention)
- 两条路径的数值精度差异在 64K 自回归生成中累积
- 导致主模型在长上下文下产生不同的 token 序列
- Draft 模型（基于主模型 aux hidden states）无法预测这些偏差

### 下一步

1. **数值对比**：同一 prompt 同一位置，比较 sparkinfer vs Marlin/FlashInfer 的 logits
   - 如果 logits 有系统性偏差 → 修 sparkinfer 精度
   - 如果 logits 一致但 token 不同 → 是混沌效应（微小差异放大），需要提高 draft 模型鲁棒性
2. **vLLM A/B at 64K**：用 vLLM (Marlin) 跑同一 prompt 同一参数，确认 vLLM 的接受率
3. **Draft 模型质量**：检查 draft 在 64K 下的 aux hidden states 是否与 4K 一致

## 2026-07-26 最终根因与修复（以本节覆盖前述“数值漂移”结论）

前述“64K 固有约 20% 接受率”“`TheOkay` 是长期退化的模型真实行为”
以及“根因是 Marlin/FlashInfer 与 Sparkinfer 数值漂移”的判断均不成立。
这些判断混用了 eager、draft Graph 和 verify Graph 路径，并且只比较了
argmax，没有控制 CUDA Graph binding 的地址生命周期。

### 当前 eager 基线

严格状态机、同一 64K token prompt、256 个输出 token：

```text
QSR_DFLASH_CUDA_GRAPH=0 ... /home/bot/.venvs/vllm/bin/python \
  /tmp/test_acceptance_repro.py 65536 256

acceptance_rate=87.0%
tokens_per_step=13.42
tok_per_s=44.6
num_steps=19
```

逐轮 eager 诊断同样得到 `263/300 = 87.7%`。Round 2 的 token 边界差异
只造成一次局部拒绝，后续立即恢复；不存在 Round 8 后持续崩塌。

### 真正根因：CUDA Graph attention binding 指向 warmup 临时地址

`_SparkinferCGExtendImpl` 在第一次 warmup forward 时创建并永久缓存
attention binding。binding 内保存 Q/K/V/output 裸地址，但模型后续 warmup
以及正式 graph capture 会重新分配 Q/output，进入 graph-private memory pool。
因此被捕获的 kernel 仍从第一次 warmup 的废弃 Q 地址读取，并写往废弃
output 地址。

故障证据（修复前，64K、256 token，生产路径为 eager verify + draft Graph）：

```text
acceptance_rate=0.13%
tokens_per_step=1.02
num_steps=250
```

修复：binding key 记录 Q/K/V/output 的 `data_ptr()`；任一地址变化时重新
构建 binding，使正式 capture 使用 graph-private buffer。CPU 回归测试覆盖
“相同地址复用、Q/output 移动后必须重绑”。

修复后，64K、256 token、eager verify + draft CUDA Graph：

```text
acceptance_rate=86.0%
tokens_per_step=13.42
tok_per_s=46.96
num_steps=19
```

与 eager 的 87.0% 基本一致，且 Round 8 后保持正常。

### 主模型 verify Graph 暂不启用

地址修复后又分别实测了主模型 verify Graph 的两个 planner：

| verify Graph planner | 64K / 256 token 接受率 | 结论 |
|---|---:|---|
| `verify`（split KV） | 23.8% | 长序列跨 page 后退化 |
| `extend`（non-split） | 25.1% | 长序列跨 page 后退化 |

Sparkinfer 当前只为 decode Graph 提供基于运行时 `cache_seqlens` 的 schedule
更新。qo=16 的 SWA verify Graph 捕获固定 worklist；ring 的相对
`cache_seqlens` 会随 `kv_len % 64` 在 528..591 之间变化，跨 page 后捕获的
window/worklist 不再匹配。短样本（5 steps）正常不能证明长样本正确。

生产策略因此改为：

- DFlash draft forward：CUDA Graph（已验证 64K 正确）；
- main verify forward：eager Sparkinfer；
- 不再让首请求进入不安全的 main verify Graph；
- draft Graph 在引擎初始化、任何请求占用 slot 之前捕获，避免延迟捕获污染
  prefix-cache slot。

主 verify Graph 的后续正确修法是给 Sparkinfer qo>1 graph replay 增加
运行时 worklist/window metadata updater；在该能力完成前不得仅凭短样本
重新启用。

### 同期修复：chunked prefill 的 draft KV 尾窗不完整

`prefill_with_aux()` 只返回最后一个 chunk 的 aux。若 prompt 长度对 8192
取模后余数小于 DFlash SWA window=512（例如 65568 的最后 chunk 只有 32
token），清空 draft KV 后只能重建 32 个位置。chunk 划分现会从前一 chunk
借 token，保证最后 aux chunk 至少覆盖完整 512-token 窗口。

CPU 验证：

```text
CUDA_VISIBLE_DEVICES='' /home/bot/.venvs/vllm/bin/python -m pytest -q \
  tests/test_laguna_sparkinfer_attn.py tests/test_dflash_engine.py \
  tests/test_mtp_accept.py

34 passed
```
