# Performance Optimization Results (2026-07-29)

## Summary

**Overall throughput: +10.3% (279.6 → 308.4 tok/s)**
**ITL: -10.1% (3.75ms → 3.37ms)**

## Changes Made

### 1. Vectorized slot_mapping in _bulk_precompute_context_kv
- Python for-loop (64K iterations) → torch vectorized ops
- Draft KV precompute: 173.6ms → 6ms (29x)
- Impact: TTFT -13.5% at 4K

### 2. Pre-allocated CG buffer offsets
- Eliminated 5 torch.arange() GPU allocations per round
- Verify CG: pre-allocated position, full_kv_indices, swa_block offsets
- Draft CG: pre-allocated position, ring_block offsets
- torch.as_tensor instead of torch.tensor in replay
- Impact: Verify 38.6ms → 36.0ms (-6.7%)

### 3. Hot loop invariant caching
- Cached os.environ.get() lookups outside decode loop
- Cached _physical_slot() and ring-layout constants
- Pre-allocated position buffer for context KV update
- list.extend() instead of for-loop append
- Impact: reduced Python overhead per round

## E2E Comparison (before → after)

| Prompt | tok/s | TTFT(ms) | ITL(ms) | Accept% |
|--------|-------|----------|---------|---------|
| 4K_english | 276→340 (+23%) | 1076→862 (-20%) | 3.62→2.94 (-19%) | 88.1% |
| 16K_english | 239→294 (+23%) | 3562→3378 (-5%) | 4.20→3.40 (-19%) | 85.3% |
| 64K_english | 277→288 (+4%) | 14867→14285 (-4%) | 3.61→3.48 (-4%) | 87.7% |
| 4K_code | 395→398 (+1%) | 899→876 (-3%) | 2.53→2.51 (-1%) | 99.6% |
| 4K_qa | 211→222 (+5%) | 907→865 (-5%) | 4.77→4.51 (-5%) | 55.9% |

## vs vLLM DFlash Baseline

| Metric | Ours | vLLM | Gap |
|--------|------|------|-----|
| tok/s @64K | 287.5 | 377 | -23.7% |
| tok/s @4K (code) | 398.3 | ~400 | ~0% |
| Accept @64K | 87.7% | 99.2% | -11.5% |
| ITL @64K | 3.48ms | 2.65ms | +31% |

The 64K gap is mainly acceptance rate (87.7% vs 99.2%) due to sparkinfer MoE
numerical path difference ("TheOkay" phenomenon). At high-acceptance workloads
(code), we match vLLM.

## Verified Non-Issues

- SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=0: 7.7% SLOWER, keep =1
- SPARKINFER_DYNAMIC_WORK_SOURCE=ready_queue: crashes (sparkinfer bug)
- _physical_slot bug lead: RESERVED=0 in both paths, not a bug

## Data Files

- `benchmarks/fixtures/e2e_daemon_bench_20260729_112935.json` — before
- `benchmarks/fixtures/e2e_daemon_bench_20260729_122252.json` — after
- `benchmarks/fixtures/round_profile_baseline_20260729.json` — component timing
- `benchmarks/fixtures/profile_verify_kernels_20260729.json` — kernel breakdown

## Remaining Optimization Targets

1. **64K acceptance gap** (87.7% vs 99.2%): sparkinfer MoE numerical path
2. **MoE kernel** (59% of verify): sparkinfer internal, limited tunability
3. **Chinese acceptance** (31.5%): draft model quality, not engine bug
4. **Prefill at 64K** (14.3s): dominated by 48-layer forward pass
