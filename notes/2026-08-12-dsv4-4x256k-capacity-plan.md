# DSV4 4x256K Capacity Plan

Status: approved implementation plan
Date: 2026-08-12
Scope: planning only; no production code changes in this document

## Goal

Enable DeepSeek-V4-Flash on one RTX PRO 6000 Blackwell GPU with:

- `capacity=4`
- exactly `4` physical slots
- four simultaneously live `256K` contexts (`262144` tokens each)
- unchanged packed GGUF weights (`81.857772734 GiB`)
- unchanged numerics for the first landing except where a dedicated parity gate explicitly approves a tighter storage form

Success is defined by a **fresh-process service-level run** with four concurrent 256K requests. Formula-only success is not sufficient.

## Current Diagnosis

Current production DSV4 is not failing because of allocator mystery or a few stray buffers. It is failing because the serving path keeps two persistent representations of the same main compressor history:

- packed attention pages that the attention kernels actually consume
- full-history BF16 `compressor.kv_cache` mirrors that remain resident after packing

That double persistence is visible in `runtime/model/dsv4_attn_kernel.py`. The server then still enforces a fifth physical slot for non-DFlash CUDA Graph in `server/engine.py`, even though DSV4 now uses a shared batched graph driver in `runtime/backends/dsv4.py` and `runtime/backends/dsv4_cudagraph.py`.

The plan should therefore prefer the **narrow structural fix**:

- keep the existing packed pages authoritative
- keep persistent BF16 indexer storage for the first landing
- remove only the full-history BF16 main compressor persistence
- replace it with bounded recurrent state plus bounded emit scratch
- run DSV4 with exactly four slots
- cap or lazily capture the 256K graph buckets

## Evidence Base

### Repo-local evidence

- `runtime/model/dsv4_attn_kernel.py` currently allocates packed page buffers and also allocates `self.compressor.kv_cache` sized for the whole sequence.
- `runtime/model/dsv4_model.py` allocates the indexer’s BF16 `kv_cache`; keeping this BF16 for the first landing is the lower-risk path.
- `runtime/backends/dsv4.py` already shares RoPE tables by reference across eager/kernel copies; the next tightening is to share by regime rather than per layer.
- `runtime/backends/dsv4.py` and `runtime/backends/dsv4_cudagraph.py` already use shared batched decode graphs over `B=1/2/4` and ratio-4 history buckets; DSV4 is no longer architecturally tied to a dedicated warmup slot.
- `server/engine.py` still applies the generic `capacity + cg_extra` rule and therefore overcharges DSV4 by one slot.

### Upstream evidence

- vLLM’s DeepSeek V4 write-up says the compressed K goes through RMSNorm, RoPE, and KV-cache insertion immediately after compression, and it presents production deployment with `--kv-cache-dtype fp8` plus `--attention_config.use_fp4_indexer_cache=True`. Source: [vLLM DeepSeek V4 post](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md).
- In the same post, vLLM describes the fusion as “Compressor + RMSNorm + RoPE + cache insertion”, which directly supports deleting a persistent full-history BF16 main-compressor mirror after pack/insertion.
- SGLang’s `deepseek_v4_memory_pool.py` stores indexer cache in packed `uint8` pools and builds compressor state as bounded rings, with `ring_size` driven by ratio and `online=(ratio == 128 and ONLINE_C128)`. Source: [SGLang DeepSeek V4 memory pool](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py).
- SGLang’s pool structure supports the second-stage direction:
  - main compressor state is bounded, not full-history
  - indexer storage can be packed later
  - `c128` online mode can collapse persistent state pressure further

These upstream references do not justify copying their code blindly. They do justify the planning claim that persistent full-history BF16 main-compressor history is not required by the architecture.

## Exact Persistent Budget

Exact known persistent components for the preferred first landing:

- weights: `87,894,114,204 B = 81.85777273401618 GiB`
- packed pages: `3,413,514,240 B = 3.1790828704833984 GiB`
  - CSA: `3,220,439,040 B`
  - HCA: `141,557,760 B`
  - window/prefill: `51,517,440 B`
- persistent BF16 indexer cache: `1,409,286,144 B = 1.3125 GiB`
- two shared RoPE regimes: `134,217,728 B = 0.125 GiB`
- recursive states + existing graph-entry scratch: `49,013,760 B = 0.045647621154785156 GiB`

Exact known persistent total:

`92,900,146,076 B = 86.52000322565436 GiB`

Interpretation:

- This is the **known persistent floor** after removing the persistent full-history BF16 main-compressor mirror.
- New multi-row emit/workspace memory is **not** counted into this static total. It must be bounded by the prefill chunk contract and accounted under runtime dynamic memory.
- The plan therefore separates:
  - persistent bytes that must exist even when idle
  - dynamic emit/workspace bytes that may appear during prefill/decode but must remain bounded

Dynamic-memory policy:

- multi-row emit scratch and other new workspaces must scale with the configured prefill chunk bound, not with full 256K history
- dynamic emit/workspace must never reintroduce a hidden full-history BF16 mirror in another form
- graph resident delta must stay `<= 1.0 GiB`

Fresh-process runtime target:

- peak `driver_used` must stay `<= 93.5-94.0 GiB`
- observed free memory should stay `>= 1.5-2.0 GiB`

Anything materially above that is too close to the cliff for a production 4x256K claim.

## Savings Reconciliation

The current four-slot 256K layout, including the same recurrent state and graph-entry
scratch counted above, is `101,456,526,236 B = 94.48875322565436 GiB` before MLA
scratch, CUDA Graph residency, allocator reserve, and driver context. The proposed
layout is `92,900,146,076 B = 86.52000322565436 GiB` on the same accounting basis.
The four-slot structural saving is therefore exactly:

`8,556,380,160 B = 7.96875 GiB`

That saving has two sources:

| Lever | 4-slot 256K saving | Why it is safe enough for the first landing |
|---|---:|---|
| Remove full-history BF16 main-compressor mirror | `5.40625 GiB` | Serving attention reads the already-packed pages; bounded compressor state and emit output remain. |
| Share RoPE across the two parameter regimes | `2.5625 GiB` | Ratio-4 and ratio-128 layers use the same compressed-RoPE parameters; window-only layers use the second regime. Bitwise identity is a gate. |

The current server rule would actually allocate five 256K slots. Under the current
layout, the known persistent components alone would be
`104,125,708,956 B = 96.97462334856391 GiB`, so that route is impossible before any
runtime overhead. In the optimized layout, preventing the fifth slot avoids another
approximately `1.1343 GiB` of packed pages, BF16 indexer, and recurrent state.

The following are not first-line levers:

- Weight re-quantization: the `81.85777273401618 GiB` GGUF payload is the model
  floor for this plan; changing it also changes GEMM and numerical risk.
- Allocator knobs or `empty_cache()`: they cannot eliminate active tensors and do
  not repair the persistent double representation.
- CPU/SSD offload: useful only as a separate oversubscription mode; it adds transfer
  latency and lifecycle complexity and is not required by the primary budget.
- Recursive-state micro-optimization: the whole retained state plus existing graph
  entry scratch is only `0.045647621154785156 GiB`.

The intended data flow is:

`token/chunk -> bounded compressor state -> emitted BF16 entry -> norm/RoPE/pack -> authoritative packed page`

The emitted BF16 entry may exist only for the bounded prefill/decode operation. It
must not be retained as a sequence-length-sized history. The BF16 indexer remains a
separate authoritative scoring cache in the first landing.

## RALPLAN-DR

### Principles

1. No second BF16 mirror: once packed main-attention KV has been inserted into its authoritative page storage, the serving path may not retain a full-history BF16 mirror of the same history.
2. Narrowest structural fix first: preserve the current packed-page kernel contract and BF16 indexer semantics for the first landing; defer broader arena rewrites and indexer compaction unless measurement forces them.
3. Exactly four slots means exactly four slots: DSV4 must not hide a fifth physical slot behind generic CUDA-Graph warmup policy.
4. Capacity claims require fresh-process proof: the final gate is a four-request service run with memory accounting through `driver_used`, not a paper budget.
5. Prefix correctness is non-negotiable: same-slot reuse, cross-slot copy, and clear/reset behavior must remain correct under the new ownership model.

### Top 3 Drivers

1. The verified root cause is persistent double storage of main-compressor history: full-history BF16 plus packed pages.
2. The stale `cg_extra=1` server rule blocks the correct DSV4 slot geometry even though the backend graph design no longer needs it.
3. 4x256K becomes plausible if and only if the main-compressor mirror is removed, graphs are bounded, and the indexer stays BF16 for the first landing.

### Options

#### Option A: Narrow structural fix over the current packed-page design

Shape:

- keep the existing packed pages authoritative
- retain persistent BF16 indexer cache
- delete full-history BF16 main-compressor persistence
- replace it with:
  - existing bounded recurrent state
  - bounded emit scratch only large enough to feed pack/insert
- share RoPE by two regimes
- run DSV4 with `num_slots=capacity=4`
- lazily capture or cap the `65536` bucket

Pros:

- Directly attacks the verified memory bug
- Preserves today’s packed-page serving contract
- Smaller blast radius than a full slot-arena rewrite
- BF16 indexer parity risk stays low in the first landing
- Aligns with upstream vLLM/SGLang evidence

Cons:

- Headroom is real but not huge; graph/private-pool discipline still matters
- Leaves indexer compaction for a second stage
- Requires careful prefix/cache lifecycle tests because ownership becomes stricter

Assessment:

- **Preferred.** This is the right first landing.

#### Option B: Option A plus second-stage indexer compaction

Shape:

- everything from Option A
- later compact indexer storage with one of two explicit formats only:
  - FP8 packed storage first
  - FP4-compatible packed storage only after Hadamard + block32 semantics are specified and validated

Pros:

- Additional headroom if Option A lands too close to the cliff
- Supported directionally by SGLang’s packed `DeepSeekV4IndexerPool`

Cons:

- Higher numerical risk
- Not needed to justify the first 4x256K attempt
- Must not degrade top-k routing stability silently

Assessment:

- Defer until after Option A is measured.

Contingency savings targets:

- FP8 packed indexer storage: approximate additional savings `~0.65 GiB`
- FP4-compatible packed indexer storage after Hadamard + block32 semantics: approximate additional savings `~0.98 GiB`

Mandatory gates before either contingency is allowed:

- dedicated top-k parity
- logit parity
- token-level parity

No generic “just store it as uint8” path is acceptable.

#### Option C: Full `Dsv4SlotPool` production rewrite now

Shape:

- make a new FP8-hybrid arena the owner of all persistent serving storage
- rewire attention, backend, and graph capture around it

Pros:

- Clean long-term ownership model
- Potentially the best eventual architecture

Cons:

- Much broader rewrite
- Unnecessary for the first measured attempt if Option A is sufficient
- Larger correctness and schedule risk

Assessment:

- **Rejected for first landing; defer.** Keep as a later architectural cleanup only if Option A proves insufficient or too fragile.

#### Option D: Keep current persistence and only tune graphs/slots

Assessment:

- Reject. The static memory math is already too high before shared scratch and runtime overhead.

## Preferred Architecture

Adopt **Option A**.

### Architectural decision

The current packed pages remain the authoritative main-attention serving cache. The first landing does **not** rewrite DSV4 around a new persistent slot arena. Instead, it narrows the serving path to one persistent representation for main-attention history:

- packed pages: persistent and authoritative
- BF16 indexer cache: persistent
- recurrent state: persistent and bounded
- main-compressor BF16 emit scratch: transient/bounded only

That means:

- no full-history `compressor.kv_cache` persistence after pack/insert
- no second BF16 mirror of the main-attention history
- no hidden fifth slot

### Why this is the best fit

- It fixes the proven root cause with the smallest structural change.
- It keeps the most mature serving surface intact: today’s packed-page kernels.
- It preserves BF16 indexer semantics for the first landing.
- It leaves the door open for later indexer compaction or deeper arena cleanup if measurement demands it.

## ADR

### Decision

Keep the existing packed-page serving design authoritative, remove only the full-history BF16 main-compressor persistence, retain BF16 indexer persistence for the first landing, make DSV4’s server-level graph slot extra zero, and constrain the 256K graph bucket policy through cap/lazy capture.

### Drivers

- Full-history BF16 main-compressor persistence is the dominant avoidable memory cost.
- DSV4’s current graph driver no longer justifies `cg_extra=1`.
- A smaller, more local fix has better odds of landing safely than a full storage-ownership rewrite.

### Alternatives considered

- Full `Dsv4SlotPool` production rewrite now: deferred because the first landing does not require that blast radius.
- Immediate packed indexer storage: deferred because the first landing should preserve BF16 indexer semantics.
- Graph-only tuning: rejected because it does not remove the dominant wasted memory.

### Why chosen

It is the smallest change set that can plausibly reach 4x256K while preserving the most existing behavior and test surface.

### Consequences

- `dsv4_attn_kernel.py` becomes the primary memory-contract change point.
- Prefix/cache lifecycle tests need to become stricter because the system now depends on the packed pages being the only persistent main-attention history.
- Graph policy must become explicit and backend-aware; DSV4 must stop inheriting the generic fifth-slot assumption.

### Follow-ups

- If Option A lands but peak `driver_used` remains above the target envelope, evaluate second-stage indexer compaction:
  - FP8 packed indexer first
  - FP4-compatible packed indexer second
- If Option A is correct but brittle, revisit a deeper arena unification later as an architecture cleanup, not as the first capacity fix.

## Implementation Plan

### Phase 1: Remove the persistent BF16 main-compressor mirror

Primary touchpoints:

- `runtime/model/dsv4_attn_kernel.py`
- `runtime/model/dsv4_model.py`
- `runtime/backends/dsv4.py`
- `runtime/kernels/dsv4_kv_pack.py`

Work:

- Stop treating `compressor.kv_cache` as a persistent full-history main-attention store.
- Introduce bounded emit scratch sufficient for pack/insert only.
- Preserve existing bounded recurrent state and reset discipline.
- Keep persistent BF16 indexer storage unchanged in the first landing.
- Share RoPE by two regimes instead of per-layer duplicates where safe.
- Update backend memory accounting so the plan can prove the second BF16 mirror is gone.
- Classify new multi-row emit/workspace bytes as runtime dynamic, not static persistent memory.

Acceptance:

- No production path retains full-history BF16 main-attention history after pack/insert.
- Backend memory classification shows the main-compressor persistent category collapsed from full-history scale to bounded scratch scale.
- RoPE accounting drops to the regime-shared target.
- Tests prove there is no second BF16 mirror.
- Persistent known components remain aligned with the exact `86.52000322565436 GiB` target budget, with any extra bytes explicitly attributed to runtime dynamic memory.

### Phase 2: Make DSV4 graph slot budgeting exact

Primary touchpoints:

- `server/engine.py`
- `runtime/backends/dsv4.py`
- `runtime/backends/dsv4_cudagraph.py`

Work:

- Make `cg_extra=0` backend-aware for DSV4.
- Require `capacity=4` to map to exactly `num_slots=4`.
- Ensure graph capture/replay remains slot-id driven and does not depend on a hidden warmup slot.

Acceptance:

- DSV4 with CUDA Graph enabled is valid at `capacity=4`, `num_slots=4`.
- No tests or runtime paths assume a hidden fifth slot.
- Capture remains atomic and replay remains slot-isolated.

### Phase 3: Make the 256K graph policy explicit and bounded

Primary touchpoints:

- `runtime/backends/dsv4_cudagraph.py`
- `runtime/backends/dsv4.py`

Concrete policy:

- keep current fixed bucket family for `512, 1024, 4096, 16384, 32768`
- treat `65536` as capped/lazy:
  - do not pre-capture it unconditionally at startup
  - capture it only at a quiescent point with no live requests and only if a memory guard allows it
  - otherwise fall back eagerly for that shape
- keep the shared graph pool across batch sizes
- cap total resident graph-pool growth at `<= 1.0 GiB`; if the fixed bucket family
  already exceeds the cap, reduce the pre-captured set and use the existing eager
  fallback rather than weakening the memory envelope

Work:

- Measure graph-pool delta for each bucket family in a fresh process.
- Encode lazy/capped policy for `65536`.
- Preserve eager fallback as the safety path.

Acceptance:

- The graph policy is explicit in code and tests.
- Startup capture no longer blindly installs all 18 possible captures.
- The memory delta of the chosen graph set is recorded and bounded.
- Resident graph delta stays `<= 1.0 GiB`.

### Phase 4: Prefix/cache lifecycle gates

Primary touchpoints:

- DSV4 backend and prefix tests

Work:

- Add gates for:
  - same-slot prefix reuse
  - cross-slot prefix copy/restore
  - slot clear/reset correctness
- Ensure all of them operate correctly when packed pages are the only persistent main-attention history.

Acceptance:

- Same-slot reuse remains correct.
- Cross-slot copy/restore remains correct.
- Clearing one slot does not corrupt another slot’s packed pages or retained prefix state.

### Phase 5: Fresh-process service proof

Primary touchpoints:

- repo-approved DSV4 capacity probe / integration surface

Work:

- Launch the real service in a fresh process with `capacity=4`, `num_slots=4`, 256K slot sizing, and the chosen graph policy.
- Drive four live slots to position `262143`, replay one `B=4` decode step so all
  four reach `262144` resident tokens, and separately exercise sustained `B=4`
  decode from a lower starting position.
- Capture:
  - backend-classified memory
  - `torch_allocated`
  - `torch_reserved`
  - `driver_used`

Acceptance:

- Four concurrent 256K requests are admitted and sustained without OOM.
- No fifth physical slot or full-context GPU prefix replica appears in the tensor
  inventory. A retained prefix must alias/reuse packed storage or stay outside the
  acceptance profile; it may not allocate another 256K GPU history.
- Peak `driver_used <= 93.5-94.0 GiB`.
- Free memory stays `>= 1.5-2.0 GiB`.
- Prefix semantics and decode replay remain correct under 4-way load.

## Test Strategy

Required coverage additions:

- parity tests that prove packed pages remain authoritative and numerics do not regress
- memory-accounting tests that prove the persistent BF16 main-compressor mirror is gone
- graph tests for `capacity=4`, `num_slots=4`, `cg_extra=0`
- graph-policy tests for lazy/capped `65536`
- prefix same-slot / cross-slot / clear gates
- fresh-process service-level 4x256K validation

Verification lanes:

- CPU-safe unit tests for policy/accounting/shape logic
- CUDA parity tests for kernel-path correctness
- fresh-process GPU capacity run for the final claim

## Risks

1. The target `86.52000322565436 GiB` persistent subtotal assumes the packed-page authoritative design can eliminate the full-history BF16 mirror cleanly; if hidden consumers still depend on that mirror, implementation will surface new coupling.
2. The `65536` graph bucket may still cost enough that eager fallback or lazy capture at quiescent points becomes necessary to preserve runtime headroom.
3. Prefix lifecycle bugs are more dangerous after removing the redundant BF16 history because the packed pages become the single source of truth.
4. BF16 indexer retention is lower risk than compaction, but it still needs confirmation that bounded main-compressor scratch does not accidentally change indexer timing or mutation semantics.

## Execution Staffing

Suggested agent roster for execution:

- `architect` or `planner`
  - own memory budget, policy decisions, and acceptance gates
- `executor`
  - lane 1: `dsv4_attn_kernel.py` / `dsv4_model.py` narrow memory-contract edits
  - lane 2: `dsv4.py` / `dsv4_cudagraph.py` / `server/engine.py` graph-slot policy
- `test-engineer`
  - add parity, prefix, and policy tests
- `verifier`
  - own fresh-process memory proof and final evidence table

Suggested reasoning levels:

- memory-contract lane: high
- graph-policy lane: medium-high
- test lane: medium
- verification lane: high

Concrete handoff sequence:

1. memory-contract lane lands first
2. graph-slot/policy lane rebases on it
3. test lane expands gates
4. verifier runs the fresh-process proof

Available role surfaces for that handoff are `architect`, `planner`, `executor`,
`test-engineer`, `performance-reviewer`, and `verifier`. Use `ralph` when one owner
should implement the phases sequentially and remain responsible through the GPU
proof. Use `team` only when the memory-contract edit has landed and the graph-policy
and test lanes can proceed with explicit file ownership.

Concrete launch hints:

```text
$ralph Implement notes/2026-08-12-dsv4-4x256k-capacity-plan.md phase by phase; do not pass a phase gate without its stated evidence.
```

```text
$team 4:executor Execute the approved DSV4 4x256K plan. Keep the memory-contract lane sequentially first; assign graph policy, tests, and independent verification only after that gate passes.
```

Team verification path:

1. `executor` records the exact post-load persistent tensor inventory.
2. `test-engineer` independently runs compressor/page parity, prefix, four-slot,
   and graph-fallback gates.
3. `performance-reviewer` checks that the graph cap/eager fallback does not create
   an unacceptable decode regression.
4. `verifier` starts a new service process and owns the final 4x256K memory proof;
   implementation-lane measurements alone cannot close the plan.

## Recommended Execution Order

1. Remove the persistent BF16 main-compressor mirror while preserving packed pages as authoritative.
2. Make DSV4 `cg_extra=0` and enforce exactly four slots.
3. Cap/lazily capture `65536`.
4. Add prefix lifecycle gates.
5. Prove the target in a fresh process.

## Changelog From Previous Draft

- Preferred path changed from full `Dsv4SlotPool` rewrite to the narrower packed-page-authoritative fix.
- ADR updated accordingly.
- Exact persistent budget corrected to `86.52000322565436 GiB`, with dynamic emit/workspace separated explicitly.
- Added fresh-process `driver_used` and free-memory targets.
- Added concrete graph policy: backend-aware `cg_extra=0`, exactly four slots, quiescent-only lazy `65536`, resident graph delta cap `<= 1.0 GiB`.
- Added upstream vLLM/SGLang evidence.
- Added prefix same-slot / cross-slot / clear gates.
- Added execution staffing / agent roster.

## Exit Criteria

This plan is complete only when all of the following are true:

- DSV4 runs at `capacity=4`, `num_slots=4` with no hidden fifth slot.
- There is no persistent full-history BF16 main-attention mirror.
- Packed pages are the only persistent main-attention history.
- Four concurrent 256K requests pass a fresh-process service run.
- Peak `driver_used` stays within the target envelope with non-trivial free headroom.
