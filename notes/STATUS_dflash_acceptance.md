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
