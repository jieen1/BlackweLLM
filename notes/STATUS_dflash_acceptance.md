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
