# A3 Cache Coordinator — Implementable Design

> Status: **design only, not implemented**. Track A step 7 (`docs/implementation-plan.md`
> §6 row 7; `docs/architecture.md` §3.5.5 row 7). Branch `work/a3-design-20260802`,
> worktree `/home/bot/project/qsr-w-a3` — **not merged into `main`**.
>
> Zero GPU used to produce this document. Zero production code touched: `runtime/`,
> `server/`, `oracle/` are read-only inputs here.

## 0. How to read this

Every claim below is tagged as one of:

- **[FACT — path:line]** — read directly from source in this repo, or from the local
  vLLM/SGLang checkouts, this round. Where a fact corrects something an earlier document
  claimed, the correction is called out explicitly (`docs/roadmap.md` §1.5-S4's
  characterization of `block_pool.py`'s GDN hooks turned out to be stale — see §2.5).
- **[JUDGMENT]** — a design recommendation. Not a fact; argued for, but a call someone
  could reasonably make differently. §6 collects the ones big enough that they should not
  be decided unilaterally here.
- **[OPEN]** — genuinely unresolved, needs a GPU or a human decision this document cannot
  supply.

Primary sources consulted directly (not by re-reading `hybrid-cache-prior-art.md`'s
transcription of them — that note's quotes were independently re-verified against the
files below, and two corrections came out of doing so; see §3):

| Source | What | Where |
|---|---|---|
| vLLM 0.25.1 | `v1/core/{single_type_kv_cache_manager,kv_cache_coordinator,block_pool,kv_cache_utils}.py` | `~/.venvs/vllm025/lib/python3.12/site-packages/vllm/` |
| SGLang | `srt/mem_cache/{allocator/mamba,hi_mamba_radix_cache,base_prefix_cache,hybrid_cache/hybrid_cache_controller}.py`, `srt/managers/{schedule_batch,schedule_policy}.py` | `/home/bot/project/sglang` @ `b296e1a5035b` (2026-07-16) |
| This repo's own retired prior art | `oracle/qwen36_vllm/{gdn_state,prefix_cache,direct_model_runner}.py`, `notes/prefix-cache-design.md`, `benchmarks/prefix_cache_eviction_check.py` | this repo, see §2.2 for why it exists and is retired |
| This repo's live substrate | `runtime/block_pool.py`, `runtime/architecture.py`, `runtime/backends/{protocol,laguna}.py` | this repo |

No newer vLLM checkout than 0.25.1 exists on this machine (confirmed by search); vLLM
0.26.0's advertised "partial prefix-cache hit support for hybrid models" remains
unverified here — flagged again in §3.1.

---

## 1. Executive summary

1. **The framing in `docs/implementation-plan.md` row 7 needs two corrections before
   anyone writes code against it** (§2.5, §3): the "S4 GDN remnants" are not remnants —
   `docs/qwen36-rebuild-spec.md` §1.5/§1.10 already found this and my own read of
   `runtime/block_pool.py` confirms it — and "vLLM/SGLang return `(kv_hit, state_hit)`"
   as a pair the *scheduler* chooses between is true of **neither**: vLLM's default
   scheduling path converges to one number before the scheduler ever sees two (§3.1),
   and tracing SGLang's `match_prefix` to its actual truncation point shows its
   scheduler-facing hit is collapsed to the state-constrained number too (§3.2) — the
   second number SGLang carries answers a different question entirely. This matters
   because it changes what "faithful to upstream precedent" means for A3's own return
   type (§4.2): no surveyed system is precedent for branching scheduling logic on two
   competing hit lengths, only for keeping a second, differently-purposed number around.
2. **Laguna's production prefix cache does not use `runtime/block_pool.py` at all.**
   `LagunaBackend.reconcile_prefix_hit` (`runtime/backends/laguna.py:2179-2220`) is a
   same-slot linear token comparison over private per-slot arrays
   (`_prefix_cache_tokens`/`_prefix_cache_kv_len`), not a content-hash lookup into
   `BlockPool.hash_to_block`. `BlockPool` is exercised today only by `benchmarks/`,
   `tests/test_block_pool.py`, and the retired `oracle/qwen36_vllm/`. This is the single
   most important fact for the migration story in §5: "zero behavior change for Laguna"
   is not a hard constraint A3 must engineer around, it is close to automatic, because
   A3's new machinery and Laguna's live machinery do not currently share a call path.
3. **This project already solved this exact problem once**, for a different tenant, at
   GPU-validated quality, and then retired that tenant. `oracle/qwen36_vllm/gdn_state.py`
   + `notes/prefix-cache-design.md` (INV1–INV9, R1–R10) + `benchmarks/prefix_cache_eviction_check.py`
   is a complete, tested design for exactly "two co-indexed cache groups, reconciled,
   evicted in lockstep, budgeted independently." A3's job is to re-derive that design
   against the *new* `ModelBackend` protocol substrate, not invent it from scratch and not
   revive the old code (`docs/implementation-plan.md` §10 explicitly forbids reviving
   `oracle/qwen36_vllm/`). §4 and §7 are built on top of it, with citations.
4. **Recommendation for the `(kv_hit, state_hit)` question (§4.2)**: expose both numbers
   in a frozen `PrefixHit` dataclass — a diagnostic pairing loosely in the spirit of
   SGLang keeping a second field (`mamba_branching_seqlen`) alongside its primary hit
   result, though that field answers a different question than this design's `kv_hit`
   does (§3.2's traced-through correction: SGLang's own scheduler-facing hit is already
   collapsed to `state_hit`, same as everyone else's) — but have the scheduler
   act only on `PrefixHit.effective = state_hit` (matches vLLM's converged single number
   and our own retired `L = G ≤ A` rule). The two raw numbers exist for observability
   (`/metrics`, bfdiag, and the A6 "prefix hit rate must not regress" gate) — no real
   system branches *scheduling* logic on the gap between them, and this project's own
   retired prior art already proved out why the invariant `state_hit ≤ kv_hit` makes that
   safe.
5. **Recommended sub-step breakdown (§7)**: 5 sub-steps inside step 7, each independently
   gate-able, all but the last GPU-free, mirroring the phase discipline
   `notes/prefix-cache-design.md` §5 already used once (P0–P4) and Track A's own
   "zero-behavior-change steps before the risky one" discipline (steps 1–4 of the 8-step
   plan).
6. **Six decisions in §6 need a human call**, not a default from this document — most
   consequentially (Decision 0), whether A3 should also rewire Laguna's own prefix cache
   onto the new content-addressed allocator while the machinery is being built anyway, or
   leave that structurally separate and out of scope for this step (recommended). Second
   most consequential (Decision 1): whether `runtime/block_pool.py`'s dormant
   `_on_evict_block` hook is the mechanism the coordinator uses, or whether the
   coordinator supersedes it per the two-independent-allocators framing.

---

## 2. Ground truth: what exists today

### 2.1 Laguna's real prefix cache is not content-addressed

**[FACT]** `runtime/backends/laguna.py:2179-2220`, `reconcile_prefix_hit`:

```python
def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
    ...
    for s in range(len(self._prefix_cache_tokens)):
        cached = self._prefix_cache_tokens[s]
        ...
        for i in range(limit):
            if token_ids[i] != cached[i]:
                break
            match_len += 1
        aligned = (match_len // self.block_size) * self.block_size
        ...
```

This walks every slot's own saved token history (`_prefix_cache_tokens[s]`,
`_prefix_cache_kv_len[s]`, populated by `reset_slot` at `laguna.py:2157-2178`) and
returns the longest block-aligned match — **same-slot reuse**, not cross-request
content-addressed sharing. `find_prefix_match` (`laguna.py:2531-2548`) is the
single-slot version of the same idea, called from `laguna_dflash.py:1379` and from
`bfdiag/workloads.py:2362`. Neither touches `BlockPool`, `hash_to_block`, or any chained
hash. `runtime/backends/protocol.py:192`'s contract for this member is `(self,
token_ids: list[int]) -> int` — a single integer, matching what the implementation
actually returns.

**[FACT]** `docs/qwen36-rebuild-spec.md:144` confirms this independently: *"`LagunaBackend`
从不构造 `BlockPool`，从不调用 `cache_block`/`touch`/`hash_to_block`"*.

Why this matters: any worry about A3 regressing Laguna's *cross-request* prefix cache
hit rate is moot — there isn't one to regress. Laguna's only prefix-cache behavior is
per-slot warm reuse across a slot's own successive requests, and A3 does not need to
touch that code path to add a second resource type; it needs to not break it, and the
two are already structurally disjoint (§5).

### 2.2 `BlockPool` is real, tested, and used only by non-production code today

**[FACT]** `runtime/block_pool.py` (270 lines) implements a full content-addressed,
LRU-evicted, reference-counted block pool: chained hashing (`hash_block_tokens`,
:96-…), `FreeBlockQueue` (intrusive O(1) deque), `BlockPool.allocate`/`free`/`touch`/
`_evict_one` (:343-…), and a lockstep eviction hook:

```python
# block_pool.py:322-329
self._on_evict_block: Callable[[int], None] | None = None
```

with the docstring at `block_pool.py:284-289` explaining exactly what it is for: *"a
popped block that still carries a published hash is EVICTED first ... and, in lockstep,
the co-keyed GDN checkpoint is dropped via the `_on_evict_block` callback the runner
wires after construction (INV2/INV3/R5)"*. It is currently `None` in every live code
path — **[FACT]** grep confirms the only place anything is ever assigned to
`_on_evict_block` in this repo is `oracle/qwen36_vllm/direct_model_runner.py:590`
(retired code).

`_ssm_spec_row` (`block_pool.py:45-79`) and `_physical_slot` (`:23-24`) are the addressing
primitives a recurrent-state allocator would need (column-0 row = ordinary state,
columns 1..K = per-slot speculative scratch rows) — already written, already unit-tested
today (`tests/test_block_pool.py:39-52`, CPU-only, no torch), and already used by the
retired `oracle/qwen36_vllm/metadata_builders.py` and `cuda_graphs.py`.

**Where this machinery came from**: `git log` shows `BlockPool` was extracted from a
larger `direct_model_runner.py` in `8ec9cd3` (2026-07-22, *"B5 模块化 Domain 1: block_pool
提取"*) into today's `runtime/block_pool.py`, while that runner (which used to serve
Qwen3.6 via vLLM) was later moved wholesale to `oracle/qwen36_vllm/` in `a9cb932`
(2026-07-30, *"Isolate retired Qwen runtime from Laguna distribution"*). The extraction
happened first; the retirement happened second and did not touch `block_pool.py` because
`block_pool.py` had already been generalized out. That is why the hooks are still there,
unused, structurally sound, and dated after the model they were built for was retired.

### 2.3 `roadmap.md` §1.5-S4's "残迹" framing is stale — corrected in `qwen36-rebuild-spec.md`, confirmed here

**[FACT — correction]** `docs/roadmap.md:158` (S4 row): *"`block_pool.py` 只管 paged KV；
GDN/SSM 递归状态的挂钩（`evict_gdn_checkpoint` 等）是 Qwen3.6 时代留下的**残迹**，当前
Laguna 无 GDN 层，这条路径没有活代码"* — this reads as "dead code, safe to delete."

**[FACT — the correction]** `docs/qwen36-rebuild-spec.md:562-567` already re-investigated
this on 2026-08-02 and reached the opposite conclusion: *"`runtime/block_pool.py` 里没有
需要清理的 GDN 专属残留代码 —— `_on_evict_block` 是通用回调，`_ssm_spec_row`/
`_physical_slot` 是通用寻址原语，两者都是干净、可直接复用的挂钩，不是"死代码待删"...
它们是**休眠但设计良好的挂钩**"*. My own read in §2.2 above agrees: there is nothing
GDN-*specific* sitting in `block_pool.py` — `_on_evict_block` is a generic `Callable[[int],
None]`, and its only GDN-flavored *name* (`evict_gdn_checkpoint`) lives entirely in
retired code, referenced only in a comment as an example of what a caller might name the
callback.

**[JUDGMENT]** `docs/implementation-plan.md` row 7's instruction "清掉 S4 的 GDN 残迹"
should be read as *"resolve the stale S4 characterization,"* not *"delete code from
`block_pool.py."* The only concrete cleanup action this implies is: (a) fix
`roadmap.md:158`'s S4 row text once A3 actually lands (small doc edit, not scoped to this
design step); (b) decide explicitly whether `_on_evict_block`/`_ssm_spec_row` are the
mechanism A3's coordinator builds on or whether the coordinator supersedes them — this is
Decision 1 in §6, and `qwen36-rebuild-spec.md:433-436` already flags it as unresolved
rather than prejudging it, which this document agrees with.

### 2.4 The protocol today: 13 members, capability-gated, single-int prefix contracts

**[FACT]** `runtime/backends/protocol.py:59-79` — `BackendCapabilities` is a frozen
dataclass with five booleans (`speculative_decode`, `prefix_cache`, `cuda_graph`,
`chunked_prefill`, `warm_continue`); **no field exists yet for "has a second cache
resource type."** `runtime/backends/protocol.py:233-243` — `CAPABILITY_MEMBERS` maps
`"prefix_cache"` to exactly `("reconcile_prefix_hit", "find_best_slot_for_prompt")`, both
declared `-> int` / `-> tuple[int, int]` where the second element of that tuple is
`(slot, hit_depth)` — a *slot id* paired with a hit length, not `(kv_hit, state_hit)`.

**[FACT]** `runtime/architecture.py:56-64,96-142` already carries the signal A3 needs on
the input side, landed in step 3 (2026-08-01, shadow mode): `LayerSpec.cache` is
`CACHE_PAGED_KV` or `CACHE_RECURRENT`, and `ArchitectureSpec.needs_two_cache_families`
is `True` exactly when a checkpoint's layers span both — with the docstring at
`architecture.py:60-64` naming this explicitly: *"This is the field `SlotResourceManager`
(step 7) is built around."* This is tested today against real checkpoints
(`tests/test_architecture_spec.py:80,85,168,203`) and returns `False` for Laguna,
`True` for Qwen3.6. **A3 can gate its entire second-allocator code path on this one
property being true**, with zero risk to Laguna, whose spec already reports `False`.

### 2.5 Two documents disagree with each other; a third resolves it

To be explicit about the discrepancy this document is correcting, since the project has
a known failure mode of documents inheriting stale numbers from each other
(`notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md` is the most recent instance):

| Document | Date | Claim | Status |
|---|---|---|---|
| `docs/roadmap.md` §1.5-S4 | pre-2026-08-01 | `block_pool.py` GDN hooks are dead-code remnants | **stale** |
| `docs/qwen36-rebuild-spec.md` §1.5/§1.10 | 2026-08-02 | Same hooks are clean, reusable, dormant infrastructure | **current, confirmed independently in §2.2/§2.3 above** |
| `notes/2026-08-01-hybrid-cache-prior-art.md` | 2026-08-01 | vLLM and SGLang both keep allocators separate; A3 should return `(kv_hit, state_hit)` | **partly stale — see §3.1/§3.2**: the "keep allocators separate" half is confirmed for both projects; the "both hand the scheduler a two-number pair to choose between" half holds for **neither** — vLLM converges to one number before scheduling, and SGLang's scheduler-facing `device_indices` is traced to the same collapsed-to-`state_hit` value (`hi_mamba_radix_cache.py:1083`); SGLang's second field (`mamba_branching_seqlen`) answers a cache-population question, not a competing hit length |

---

## 3. What vLLM and SGLang actually do (re-verified, not re-quoted)

### 3.1 vLLM 0.25.1 — separate managers, but the scheduler sees **one** number

**[FACT]** Confirmed verbatim, unchanged from the prior-art note: `MambaManager`
(`v1/core/single_type_kv_cache_manager.py`) searches right-to-left with early stop and
pads with null blocks (lines 1065-1085), subtracts the full speculative window before
freeing (lines 1151-1156, the `tdoublep` comment), and reports a fake shortage
(`num_gpu_blocks + 1`) to defer admission by one step when it cannot safely allocate
(lines 1196-1204, guarded by a `cached_blocks_this_step` check the note did not mention).

**[FACT — correction to the note's implication]** `HybridKVCacheCoordinator.
find_longest_cache_hit` (`kv_cache_coordinator.py:630-740`) does **not** hand two numbers
to the scheduler. It runs an iterative fixed-point: each per-type manager either accepts
the current candidate hit length or shrinks it; any shrink triggers re-checking all
groups; this converges because the length only ever decreases (docstring at
`kv_cache_coordinator.py:636-641`). Full-attention blocks past the converged length are
then truncated (lines 727-733) before the result reaches the scheduler. **A single
number is what schedules the request.**

A different method, `find_longest_cache_hit_per_group` (line 742), does return per-group
hit lengths independently — but its only caller is `scheduler.py:697-698`, gated behind
`self.connector is not None and self.has_mamba_layers`, i.e. **disaggregated-prefill
(KV-connector) bookkeeping only**, where it feeds a `max()` over per-group hits into an
external-transfer-savings estimate (`scheduler.py:715`: Mamba state transfers
unconditionally regardless of the KV-side hit). This is very likely the seed of vLLM
0.26.0's advertised "partial prefix-cache hit support for hybrid models" — already present
in 0.25.1, but scoped to disaggregation, not general local scheduling. **Not independently
verified against 0.26.0** — no newer checkout exists on this machine.

**[FACT — second correction]** `block_pool.py`'s `BlockPool` is **one shared, untyped
pool** — a single `free_block_queue`, a single `num_gpu_blocks` id space, constructed once
and handed to every `SingleTypeKVCacheManager` subclass including `MambaManager`. Eviction
order is one global LRU across KV and Mamba blocks alike; there is no
resource-type-aware budget in vLLM's coordinator. This works because
`kv_cache_utils.unify_hybrid_kv_cache_specs` pads every group's per-block byte size to a
common `page_size_bytes`, so one integer block id validly indexes into every group's
tensor. **The "separate per-resource eviction budget" idea belongs to SGLang, not
vLLM** — the note's phrasing ("both upstreams... separate budgets") over-generalizes from
SGLang to vLLM; corrected here.

**Alignment**: `HybridKVCacheCoordinator.__init__` asserts `block_size %
hash_block_size == 0` for every group (lines 552-555); there is no direct assertion
tying Mamba's block size to full-attention's beyond that shared divisor.

### 3.2 SGLang — two fields on `MatchResult`, but not two competing hit lengths

**[FACT]** Re-verified directly against `/home/bot/project/sglang` (commit
`b296e1a5035b`, 2026-07-16; no newer checkout found; `UnifiedRadixTree`/v0.5.16's
"smarter state resets" not present in this checkout).

`MambaSlotAllocator` (`srt/mem_cache/allocator/mamba.py:29-34`), verbatim: *"Unlike
`BaseTokenToKVPoolAllocator` which is designed for per-token KV pages, Mamba slots are
request-level (typically 1 slot per request). We keep the interface minimal and do NOT
inherit the KV base class."* Confirmed directly, exact lines.

**The two-field result** — this looked at first like the fact that resolves the vLLM/SGLang
tension in §3.1 in SGLang's favor (two numbers, not one); tracing where each field is
actually computed (below, and the correction later in this subsection) shows it does not —
lives in `srt/mem_cache/base_prefix_cache.py:155-190`:

```python
class MatchResult(NamedTuple):
    device_indices: torch.Tensor            # defines the KV hit length
    ...
    mamba_branching_seqlen: Optional[int]   # "the longest page-aligned position
                                             #  that could've been cache hit if
                                             #  there exists a mamba state"
```

`device_indices` (its length) is the KV hit; `mamba_branching_seqlen` is the state-side
candidate boundary — **two independent fields on the same returned object**, not a
converged single number. **[FACT, traced to the actual computation site]**
`HiRadixCache._match_post_processor` (`hi_mamba_radix_cache.py:1016-1053`) computes it —
and, critically, computes it from a *different, longer* candidate than the one that
becomes `device_indices`:

```python
# hi_mamba_radix_cache.py:1040-1053
if len(value) > best_value_len:
    mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size
    mamba_cache_chunk_aligned_seqlen = (
        sum(len(v) for v in value) // mamba_cache_chunk_size
    ) * mamba_cache_chunk_size
    mamba_branching_seqlen = (
        mamba_cache_chunk_aligned_seqlen if mamba_cache_chunk_aligned_seqlen > 0 else None
    )
else:
    mamba_branching_seqlen = None
...
value = value[:best_value_len]           # line 1083 — THIS becomes device_indices
```

`value` here is the *un-truncated* KV-hash walk (the `kv_hit` in this design's vocabulary);
`best_value_len` is the deepest node that actually carries a live/backed-up Mamba
checkpoint (the `state_hit`). So `device_indices` is deliberately truncated **to
`state_hit`** before it ever leaves this function — SGLang's scheduler-facing KV hit is
*already* the collapsed number, matching vLLM's and our own retired design's pattern, not
diverging from it. `mamba_branching_seqlen` is a *third*, different quantity: the deepest
**chunk-aligned** position within the *un-truncated* KV walk (`value`, i.e. `kv_hit`) — a
forward-looking hint of "where would it be worth materializing a *new* Mamba checkpoint,"
not a backward-looking hit-resolution number at all. `schedule_policy.py:140-141` copies it
onto the request unchanged; deciding whether/how to actually use it (branch a checkpoint
there, or not) happens downstream in Mamba-specific code. **Net correction to the framing
above**: SGLang does not hand the scheduler two competing hit lengths to reconcile —
it hands back one collapsed hit length (`device_indices`, already `≤ state_hit`) plus one
unrelated cache-population hint (`mamba_branching_seqlen`). This sharpens, and slightly
revises, the "genuinely two numbers" framing at the top of this subsection: two numbers
reach the caller, but they answer different questions, and only one of them is a hit
length in this design's sense.

**[FACT]** Per-resource eviction budgets, confirmed directly at
`srt/mem_cache/base_prefix_cache.py:83-98`:

```python
@dataclasses.dataclass
class EvictParams:
    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0

@dataclasses.dataclass
class EvictResult:
    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0
```

and the hit-test that validates both resources independently, `hi_mamba_radix_cache.py:357`:
`if last_node.evicted or (last_node.mamba_evicted and last_node.mamba_backuped):` —
confirmed verbatim at the cited line. `mamba_evictable_size_` (separate accounting from
the KV side) confirmed at lines 341, 540, 919, 947.

**Not found in this checkout, despite looking**: an explicit "assume all draft tokens
rejected before freeing Mamba state" rule analogous to vLLM's (grepped the mamba/hybrid
cache files and `schedule_batch.py`/`schedule_policy.py` for `speculative`/`draft`/`EAGLE`
near mamba-eviction code; no hit). This may live in a part of the speculative-decode
worker this pass did not cover — **[OPEN]**, not claimed as absent, only as not found.

### 3.3 Our own retired prior art — the closest analog, and it chose a third answer

**[FACT]** `notes/prefix-cache-design.md` §3.4 (lines 310-340) — written for the
vLLM-based Qwen3.6 tenant before it was retired — derives the two-group case as a
*specialization* of vLLM's general solver, not an alternative to it:

> "This is the fixed-point of vLLM's `HybridKVCacheCoordinator`... specialized to two
> groups where the attention group is downward-closed... and the GDN group is
> snapshot-constrained: the converged hit length is the min, block-aligned,
> snapshot-constrained boundary. We don't need the general iterative solver — two groups
> with `G ≤ A` gives `L = G` directly."

The actual returned value from `reconcile_prefix_hit`
(`oracle/qwen36_vllm/prefix_cache.py:131-166`) is a **single int**, `L`, computed
internally from two intermediate quantities — `A` (left-to-right attention-hash walk,
downward-closed, stop at first miss) and `G` (right-to-left search among the matched
blocks for the deepest boundary with a matching GDN checkpoint) — with the invariant
`L = G ≤ A` always holding by construction (checkpoints are only ever created at
boundaries already inside the hashed attention prefix). This is independently confirmed
by the eviction-direction asymmetry in `oracle/qwen36_vllm/gdn_state.py:183-209`
(`evict_gdn_checkpoint`, full text and reasoning in §4.3 below): evicting a KV block
always evicts its co-keyed checkpoint, but evicting a checkpoint under GDN-side memory
pressure only clears the KV block's *hash index entry* (never its live memory) and only
when that block is already unreferenced — so `G` can never end up greater than `A`.

This means **this project has already tried the "converge to one number" approach
once, for the same architecture, and it worked well enough to GPU-validate**
(`benchmarks/prefix_cache_eviction_check.py`'s `chunk_boundary_partial_share` case
exercises exactly this) — and, per §3.2's correction, this is also what every system
surveyed here actually does at the scheduling boundary, not a choice unique to vLLM and
our own retired code. §4.2 gives the recommendation for what (if anything) A3 should keep
a second, unconverged number *around for*, since "hand the scheduler two competing hit
lengths" turned out not to be a real precedent anywhere.

---

## 4. Answers

### 4.1 Q1 — Invariants and their observable failure symptoms

**[JUDGMENT, built on FACT]** These are adapted from `notes/prefix-cache-design.md` §4
(INV1-INV9), which were written for a two-cache-group system and already have a working
test methodology (`benchmarks/prefix_cache_eviction_check.py`) — not invented fresh. Names
below are renumbered for A3 (`INV-A3-n`) since the original INV numbers were scoped to a
retired module; the mapping to the original is given for traceability.

| # | Invariant | Maps to | Symptom when broken |
|---|---|---|---|
| INV-A3-1 | A resource is never referenced by two live slots that don't agree it's shared (ref-counted correctly; no phantom double-free) | INV2, INV9 | Silent: one request's output changes because another request's write landed in "its" memory. No crash. Only visible via a signal-probe test (marker tokens per slot) or a bit-exact regression — **this is the "many tokens later" bug class the roadmap already names as the reason A3 exists** |
| INV-A3-2 | `state_hit ≤ kv_hit` always, for every prefix, at every point in time | INV3/R5 (`L = G ≤ A`) | If violated: the coordinator tells the scheduler it can skip prefilling more tokens than it has valid *recurrent* state for → the recomputed suffix starts from wrong GDN state → silently wrong logits from that token on, no exception, no crash. Symptom is a quality/acceptance-rate regression with no stack trace, discovered (if at all) by a `bf diff` several thousand tokens later |
| INV-A3-3 | Evicting a KV block always evicts its co-keyed recurrent-state checkpoint (forward lockstep); evicting a checkpoint under state-side budget pressure never reclaims a still-referenced (`ref_cnt>0`) KV block, only clears the hash-index entry of an already-free one (reverse lockstep, asymmetric) | INV3/R5, `gdn_state.py:183-209`'s two-direction comment | If forward lockstep breaks: a future hit thinks it has valid state for a KV region that's actually been reallocated to someone else — same silent-corruption symptom as INV-A3-2. If the reverse direction is made *symmetric* by mistake (state eviction force-evicts live KV): a live request's KV cache is yanked out from under it — this manifests as a crash (illegal memory access / wrong shape) or, worse, silently wrong attention output, and it is the difference between "safe compute-miss" and "corruption" this repo's own comment (`gdn_state.py:196`, `L = G <= A still holds`) exists to prevent |
| INV-A3-4 | A resource actively held by any in-flight or mid-admission slot is never evicted, by either allocator | INV9/R4 | Crash (use-after-free-shaped: wrong tensor read) under concurrent admission + eviction pressure; only reproducible under load, which is exactly why `notes/prefix-cache-design.md`'s own test for this (`admission_under_pressure`, ported into `benchmarks/prefix_cache_eviction_check.py:501-575`) deliberately forces eviction while other slots stay active rather than testing eviction in isolation |
| INV-A3-5 | Speculative/draft tokens never populate either cache (only committed positions do); a rejected draft cannot poison a future hit | INV4 | A future request "hits" a prefix that includes tokens that were never actually accepted — output diverges from a cold reference by exactly the rejected draft content, at exactly the hit boundary. Caught by comparing hit-path output to a cold-path reference for the identical prompt, never by a shape check |
| INV-A3-6 | A backend with no recurrent layers reports `state_hit == kv_hit` unconditionally (there is no second resource to diverge from the first) | new — no analog needed under one cache family | If violated: Laguna (`needs_two_cache_families == False`) starts skipping less prefill than it used to, i.e. a pure performance regression with no correctness signature — caught by the A6 gate's throughput/bit-exact checks, but only after the fact; a dedicated unit test asserting this equivalence for every KV-only backend is cheap enough to write first (§7) |
| INV-A3-7 | CUDA-graph replay is oblivious to which blocks/rows came from a hit vs. fresh compute | INV5 | Wrong-shaped or stale addresses baked into a captured graph; manifests as an illegal memory access or, worse, correct-looking-but-wrong output on replay. Already partially covered for Laguna's decode graph (no recurrent state touched there today); genuinely new territory once a recurrent-state row is graph-replayed (§4.3 below, Decision 4 in §6) |
| INV-A3-8 | Reserved-address conventions (`RESERVED_PHYSICAL_SLOTS`) are consistent between whatever allocator(s) A3 wires up | INV7 | Deterministic wrong output from slot 0 onward — this exact bug already happened once in this project's history (`block_pool.py:17-19`'s own comment: *"something about index 0... makes the model read/write the wrong state"*) and is the reason the convention exists at all. **Flagged as currently inconsistent**: `block_pool.py` reserves 1 physical slot; Laguna's own local constant is `RESERVED_PHYSICAL_SLOTS=0` (`laguna.py:53`, `laguna_cuda_graph.py:41`) — i.e. Laguna does not even use `block_pool.py`'s convention today (consistent with §2.1: it doesn't use `BlockPool` at all). `qwen36-rebuild-spec.md:131` already flags this divergence as "应废弃（待核实）" rather than assuming block_pool's `=1` is required — A3 should not silently inherit `=1` for a new allocator without re-deriving whether it's still needed, since the original justification was vLLM-scheduler-specific, not hardware-specific |

### 4.2 Q2 — `(kv_hit, state_hit)`: what to do when they disagree, and why

**The concrete example**: KV hit is 900 tokens, state hit is 400. **[JUDGMENT]** Take
400 — i.e. `effective = state_hit`, never `kv_hit`, and never anything computed by
averaging or otherwise mixing the two. This is not a close call; it falls straight out of
INV-A3-2. Serving from a state that is 500 tokens behind the KV means the recurrent
layers' forward pass for tokens `[400, 900)` would run with the *wrong* accumulated
gate/conv/ssm state — those layers have no positional index to "look up" 500 tokens of
history from KV the way full attention does. The only correct move is to treat `[400,
900)` as a compute-miss for the recurrent layers specifically. Whether the *attention*
layers still get to reuse the KV in `[400, 900)` is a genuine design fork (does A3 support
per-layer-family partial reuse, recomputing only the recurrent layers for that span?) —
today's retired prior art (`prefix_cache.py:131-166`) does **not** do this; it treats
`A > 0, G = 0`-shaped mismatches as a full compute-miss for simplicity (`prefix_cache.py:135-139`'s
comment: *"attention cached but GDN never checkpointed... treated as a compute miss in
v1... write-time dedup still reclaims the memory"*). **[OPEN/Decision 2 in §6]**: whether
A3 should do better than that v1 simplification is a real scoping question, not something
this document should decide for the same reason architecture.md deferred A1→A6 sequencing
decisions to a human — it trades implementation complexity against a real performance
win whose size is currently unmeasured.

**How vLLM/SGLang actually handle "the two numbers disagree" (§3.1/§3.2 facts, restated
as the answer to this question specifically)**:

- vLLM's *default* path never lets them disagree observably: the coordinator's
  fixed-point loop keeps shrinking the candidate until every group agrees, so by the time
  scheduling happens there is only one number, already reconciled to the more restrictive
  side. The two-number shape exists but is scoped to disaggregated-prefill bookkeeping,
  not local admission.
- **[Revised per §3.2's traced-through correction]** SGLang's scheduler-facing hit
  (`device_indices`) is **also already collapsed to `state_hit`** by the time it leaves
  `match_prefix` (`value[:best_value_len]`, `hi_mamba_radix_cache.py:1083`) — it does not
  hand the scheduler a raw, uncollapsed `kv_hit` either. The second field that does ride
  along, `mamba_branching_seqlen`, is not an alternative, larger hit length for the
  scheduler to choose between — it answers a different question (where to materialize a
  *future* checkpoint), computed from the pre-truncation `kv_hit` for that purpose only.
  So all three systems agree on the scheduling-facing number; SGLang's distinctive move is
  keeping a second, differently-purposed number around at all, not keeping two competing
  hit lengths.
- Our own retired code picked the same one-number shape but derived it explicitly rather
  than letting an iterative solver find it, because two groups with a known
  monotonicity (`G ≤ A` always) don't need the general solver.

**[JUDGMENT] Recommendation**: keep both raw numbers around at the *type* level (satisfies
the literal design mandate in `implementation-plan.md` row 7 and gives `/metrics`/bfdiag/
the A6 "prefix hit rate must not regress" gate something concrete to observe, e.g. "how
often does `state_hit < kv_hit` fire, and by how much" — a real signal for whether GDN
checkpoint granularity, §4.3, is coarse enough — this is this design's own diagnostic use
for the pair, not a copy of SGLang's `mamba_branching_seqlen`, which serves a different,
cache-population purpose this design does not need at the coordinator level), but do what
every system actually surveyed does at the *scheduling* level (one effective number,
`state_hit`, is what actually gates how much prefill gets skipped — not "SGLang does
something different here," §3.2's correction). Concretely:

```python
@dataclass(frozen=True)
class PrefixHit:
    """kv_hit: longest block-aligned prefix with paged KV present and reference-able.
    state_hit: longest block-aligned boundary at or before kv_hit with a matching
    recurrent-state checkpoint. Invariant (INV-A3-2): 0 <= state_hit <= kv_hit always.
    For a backend with no recurrent layers, state_hit == kv_hit unconditionally
    (INV-A3-6)."""
    kv_hit: int
    state_hit: int

    @property
    def effective(self) -> int:
        return self.state_hit
```

No system surveyed here — not vLLM, not SGLang, not our own prior art — branches
scheduling logic on the *difference* between the two numbers; they only ever use the more
restrictive one to decide "how much can I skip." Recommending a dataclass over a bare
`tuple[int, int]` for the same reason `BackendCapabilities` is a dataclass and not a
string (`protocol.py:63-68`): `.kv_hit`/`.state_hit` cannot be transposed by accident the
way `result[0]`/`result[1]` can, and the invariant lives next to the data as a docstring
a reviewer will actually read.

### 4.3 Q3 — Eviction budgets: who gets evicted first

**[FACT, from `oracle/qwen36_vllm/gdn_state.py`, re-derived for A3 rather than copied]**
The retired implementation already answers this with an asymmetric rule, and the
asymmetry is the load-bearing part:

- **KV-side pressure evicts forward**: when `BlockPool._evict_one` reclaims a still-hashed
  attention block (`block_pool.py:343-362`), it calls `_on_evict_block`, which — in the
  retired wiring — drops the co-keyed recurrent-state checkpoint unconditionally
  (`gdn_state.py:183-209`, "LOCKSTEP, reverse direction" comment block, first half).
- **State-side pressure evicts checkpoints against their own byte budget, independent of
  KV pressure**: `_evict_gdn_checkpoints_for_budget` (`gdn_state.py:211-226`) evicts
  LRU-oldest checkpoints purely to keep `gdn_ckpt_meta`'s total bytes under
  `gdn_checkpoint_byte_budget` — a **separate accounting** from the KV pool's own
  free-block count, matching SGLang's `EvictParams(num_tokens, mamba_num)` split (§3.2)
  and *not* matching vLLM's single untyped pool (§3.1).
- **The asymmetry that makes this safe**: when state-side pressure evicts a checkpoint, it
  only clears the co-keyed KV block's *hash index entry* — never the block's live
  memory — and only when that block is already `ref_cnt == 0` (`gdn_state.py:183-209`,
  full comment: *"If the block is ref_cnt > 0 (an active slot still references it), its
  hash stays — losing only the checkpoint, which merely turns a future would-be hit into
  a safe compute miss (`L = G <= A` still holds)."*). State-side pressure can never evict
  live KV memory. KV-side pressure, symmetrically, only ever evicts a checkpoint that is
  already dead weight (its keyed block is gone).

**[JUDGMENT] Recommendation for A3**: keep this asymmetry, restated as a coordinator-level
rule rather than a cross-allocator callback (see Decision 1, §6, for why the *mechanism*
— callback vs. coordinator-owned — is still open even though the *rule* is not):

1. Each resource type gets its own budget and its own accounting — paged KV by block
   count (as today), recurrent state by byte budget (as the retired code already did).
   Neither allocator needs to know the other's numbers to enforce its own budget.
2. **KV eviction never asks permission from the state allocator** — it evicts whatever its
   own LRU says to evict, then *notifies* the coordinator, which drops the now-orphaned
   checkpoint. This preserves KV's existing eviction order unchanged (zero behavior
   change for Laguna, §5) and is a natural fit for `_on_evict_block`'s existing signature
   *if* Decision 1 keeps that hook.
3. **State eviction never reclaims live KV** — it may only demote a KV block's
   cache-hash-index entry (turn a future hit into a future miss) when that block's
   `ref_cnt == 0`, and it must never touch a `ref_cnt > 0` block's memory or hash. This is
   the one rule that must be enforced regardless of which mechanism carries it, because
   it is the rule that keeps "the state allocator ran out of budget" from ever becoming a
   correctness incident on the KV side.
4. Order between the two budgets, when both are under pressure simultaneously: **KV first,
   then state** — **[FACT, independently re-verified this round]**, upgraded from the prior
   note's unverified carry-forward. `HybridCacheController._page_transfer`
   (`hybrid_cache/hybrid_cache_controller.py:651-666`) is explicit and structural, not just
   an ordering convention:

   ```python
   def _page_transfer(self, operation):
       # KV pools first — determines actual completed page count
       super()._page_transfer(operation)

       # Extra pools only after KV fully completes. If KV terminated early
       # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
       # data misalignment.
       kv_completed_pages = operation.completed_tokens // self.page_size
       if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
           ...
   ```

   The extra-pool (Mamba/SWA) transfer is gated on `kv_completed_pages == len(operation.
   hash_value)` — i.e. it does not merely go *second*, it is skipped **entirely** if KV
   did not fully complete, specifically to avoid the two halves disagreeing about how much
   of the prefix is actually valid (the same INV-A3-2 hazard, at the IO layer instead of
   the eviction layer). `_resolve_pool_transfers_allocation`
   (`hybrid_cache/hybrid_cache_controller.py:725-806`) additionally wraps extra-pool
   allocation in an explicit `rollback_allocated()` (defined `:739`, invoked on failure at
   `:777,802`) that frees every extra-pool allocation made so far if any one of them fails
   — atomic all-or-nothing for the *state* side specifically, layered on top of (not
   instead of) the KV-first ordering. Reasoning for keeping KV-first even before this
   verification held: KV is the larger, more contended resource under this project's actual
   capacity constraints (Track B's F2 note: KV and speculative scratch already compete for a
   tight 96 GB budget), so it should be evicted/transferred on its own terms first; state
   eviction is cheap relative to it and mostly a safety valve. The verified SGLang mechanism
   gives a second, independent reason: correctness under partial failure, not just resource
   priority.

### 4.4 Q4 — Migration path: zero behavior change for Laguna

**[FACT, established in §2.1]** Laguna's live prefix-cache code path
(`reconcile_prefix_hit`/`find_best_slot_for_prompt`/`find_prefix_match` in
`laguna.py`) does not call into `BlockPool` at all today. **This is the reason zero
behavior change is achievable by construction, not by careful engineering**: A3's new
machinery and Laguna's existing machinery are already two different call graphs. The
migration risk is not "will A3 change what Laguna returns" — it structurally cannot,
unless someone chooses to rewire Laguna's own methods on top of the new coordinator, which
is explicitly out of scope for this step (that rewiring, if ever done, is its own future
step with its own bit-exact gate).

**The actual thing that needs proving** is narrower: *the protocol surface change itself
must be additive*. Concretely:

1. `BackendCapabilities` gains a new field (name TBD, e.g. `recurrent_state: bool`,
   Decision 3 in §6) that is `False` for `LagunaBackend` — a one-line addition, covered by
   the existing shadow-conformance test pattern (`test_backend_protocol.py:32-38` already
   asserts an exact member count; that count goes up by one field, not one *required*
   member, so `LagunaBackend`'s conformance is unaffected).
2. `reconcile_prefix_hit`'s **return type changing from `int` to `PrefixHit`** is the one
   real compatibility question, since it is a `CAPABILITY_MEMBERS["prefix_cache"]` member
   today (`protocol.py:235`) with a concrete `-> int` signature every caller (`engine.py`,
   `bfdiag`) currently expects. **[JUDGMENT]**: do not change this signature in place.
   Either (a) add a new protocol member (e.g. `reconcile_prefix_hit_v2` /
   a name-neutral rename per the naming-debt note already on file at
   `protocol.py:29-46`) that returns `PrefixHit` and is only required when
   `capabilities.recurrent_state` is `True`, leaving the existing `int`-returning member
   untouched for every backend that doesn't need the second number; or (b) keep
   `reconcile_prefix_hit -> int` as `PrefixHit.effective` and add `state_hit`/`kv_hit` as
   a separate, optional query. Both keep Laguna's call sites bit-for-bit unchanged. Which
   one is cleaner for `Qwen36Backend`'s eventual implementation is Decision 3 in §6 — not
   decided here because it is exactly the kind of call `architecture.md`'s own migration
   discipline (steps 1-4: define the shape in shadow mode, prove equivalence, *then*
   switch) says should be proven in shadow mode against a real (if not-yet-implemented)
   second backend's needs, not guessed at now.
3. **INV-A3-6 as the actual proof obligation**: whatever shape is chosen, the test that
   matters is "for every backend where `capabilities.recurrent_state` is `False` (today:
   every shipping backend), the new code path is provably equivalent to the old one" —
   this is a shadow-mode unit test in the same style as steps 1/3/4 of the existing 8-step
   plan (`architecture.md:430,432,433`: define, assert equal to today's hardcoded value,
   do not drive anything from it yet), and it is the concrete, automatable form of "zero
   behavior change," not a promise taken on faith.

### 4.5 Q5 — The six pitfalls, checked one at a time

| # | Pitfall (`hybrid-cache-prior-art.md` §6) | This design's answer | Verified this round? |
|---|---|---|---|
| 1 | Don't unify the allocators | Two allocators (existing `BlockPool` for paged KV; a new, not-yet-written recurrent-state allocator modeled on SGLang's `MambaSlotAllocator` shape — fixed slot per request, no paging) + a coordinator that owns cross-allocator invariants, not a merged data structure. **[FACT]** confirmed independently for both vLLM (type-dispatch via `SingleTypeKVCacheManager` subclasses) and SGLang (`MambaSlotAllocator` explicitly "do NOT inherit the KV base class") in §3.1/§3.2 this round | Yes, both projects, this round |
| 2 | Prefix search runs in the opposite direction; block sizes must be aligned | Confirmed exactly in our own retired code (`prefix_cache.py:157-165`: right-to-left search for the deepest usable checkpoint boundary `≤ A`) as well as vLLM's `MambaManager` (§3.1, unchanged from the note). Alignment: vLLM asserts `block_size % hash_block_size == 0` per group (§3.1); our retired code sidesteps a general alignment problem by using the *same* `block_size` for both groups and deriving `G` as a multiple of it by construction. **[JUDGMENT]**: A3 should keep one shared `block_size` between paged KV and recurrent-checkpoint boundaries rather than reintroducing vLLM's general multi-block-size alignment machinery — Qwen3.6 gives no reason yet to need different granularities, and "explicit single knob" beats "general solver nobody has a second use case for" | Yes (vLLM + our own code); SGLang's alignment mechanism not separately re-checked this round |
| 3 | Speculative decoding: subtract the full window before freeing | Confirmed in vLLM (§3.1, `tdoublep` comment, re-verified exact lines). **Not found** in SGLang's mem_cache layer this round (§3.2, explicitly reported as a negative result, not an absence claim) — may live in speculative-worker code not covered this pass. Our own retired code takes a *different*, arguably stronger approach for the *state* half specifically: rather than a budget-subtraction heuristic, `_ssm_spec_row` (`block_pool.py:45-79`) gives every speculative candidate its own dedicated, never-shared state row, so a rejected candidate's state is simply never read again — no snapshot/restore, no conservative-freeing budget needed for state at all. **[JUDGMENT]**: for A3, prefer the dedicated-row approach for recurrent state (it already exists, is unit-tested, and eliminates this whole failure class rather than bounding it) and reserve vLLM's "subtract the speculative window" rule only for whatever *block-count* accounting the KV side needs under speculative decoding — which is a Track B question, not an A3-coordinator question, since A3 itself does not do speculative decoding | Partially — vLLM yes, SGLang not found, our own code yes (different mechanism) |
| 4 | Recurrent state can't be borrowed across requests in the same step | **[OPEN]** Confirmed in vLLM only this round (`get_num_blocks_to_allocate`'s fake-shortage return, unchanged from the note, not re-quoted here since §3.1 already re-verified the adjacent methods in the same file). Not independently re-checked in SGLang this round. This is a **scheduler-level** constraint (`ServerEngine` decides admission), not a coordinator-level one — A3 (the coordinator) should *expose* enough information (e.g. "this prefix's recurrent-state boundary requires a fresh, not-yet-taken slot") for the scheduler to enforce it, but should not itself own the admission decision. Concretely: this is a property `ServerEngine.admit_batch` (wherever it is today) would need to check, not a new method on the coordinator beyond what `PrefixHit`/allocator availability already exposes. **No code change proposed here** — flagged for whoever writes `Qwen36Backend`'s scheduling integration (Track B) to re-derive against the real scheduler, since `ServerEngine` today has no concept of this at all (confirmed: it manages exactly one resource type) | Vetted for vLLM only |
| 5 | Answer "does the in-checkpoint MTP layer carry GDN" before scoping B3 | Out of scope for A3 itself — this is `docs/roadmap.md` B0-8, explicitly assigned to a separate parallel investigation and explicitly not to be predicted here (per this project's "别问太多次"/"不代模型" norms already in the user's own working style). Mentioned here only because if the answer is "no," the speculative/recurrent-state interaction in row 3 above simplifies further (no candidate rows ever get written by a draft step at all) | N/A — deliberately not investigated here |
| 6 | Per-resource eviction budgets; hit test validates both independently | Answered in full in §4.3. **[FACT]** re-confirmed directly against SGLang source this round (`EvictParams`/`EvictResult`, `base_prefix_cache.py:83-98`; hit-test condition, `hi_mamba_radix_cache.py:357`) and against our own retired code (`gdn_state.py`'s asymmetric lockstep). **[FACT — correction]** vLLM does **not** do this (§3.1) — its pool is unified and untyped, so this pitfall's "both upstreams" framing in the original note is only half right; corrected here | Yes, SGLang + our own code, this round; vLLM explicitly does the opposite |

### 4.6 Q6 — Sub-steps inside step 7

**[JUDGMENT]**, modeled directly on two precedents already in this repo: Track A's own
"zero-behavior-change steps before the risky one" discipline (`architecture.md:428-437`,
steps 1-4 shadow-mode / step 5 the switch), and `notes/prefix-cache-design.md`'s own P0-P4
phasing for the *same* underlying problem the first time it was built. Five sub-steps,
ordered by blast radius, matching the pattern that already worked once:

| Sub-step | What | Behavior change | Gate | GPU |
|---|---|---|---|---|
| 7.1 | **Type layer**: `PrefixHit`, extended `BackendCapabilities` (+1 field), protocol member(s) per Decision 3 (§6) — types and signatures only, `runtime/backends/protocol.py` stays torch-free. `INV-A3-6` unit test (every `capabilities.recurrent_state=False` backend behaves identically under the new types) | None | Existing shadow-conformance test pattern extended (`test_backend_protocol.py`-style); `ruff` + both pytest jobs | No |
| 7.2 | **Recurrent-state allocator skeleton**: new module (not `block_pool.py` — per pitfall 1/Decision 1), fixed-slot, no paging, addressing via existing `_ssm_spec_row`/`_physical_slot`. No model wired to it yet — this is pure bookkeeping, testable the same way `_check_byte_budget`/`_check_lockstep_eviction` in `benchmarks/prefix_cache_eviction_check.py:96-317` already proved out for the retired version, minus any GPU/tensor dependency (those checks were already pure-Python; port the *pattern*, not the import from `oracle/`) | None (nothing calls it) | New unit tests, CPU-only, no torch | No |
| 7.3 | **Coordinator**: owns INV-A3-1 through INV-A3-5 as explicit, testable rules over the two allocators from 7.1/7.2. Reconciliation (`kv_hit`/`state_hit` computation), eviction ordering (§4.3 rule 4), same-step admission signal (row 4, §4.5) exposed as a queryable fact, not enforced by the coordinator itself | None (`ArchitectureSpec.needs_two_cache_families` gates it off for every checkpoint that exists in production today) | Full invariant test suite, adapted from `notes/prefix-cache-design.md` §4/§5-P3's methodology (signal-probe, admission-under-pressure, byte-budget, lockstep-both-directions) — all CPU-only except where a real recurrent-state tensor is unavoidable | Mostly no; a small GPU-optional subset if any check needs a real tensor shape |
| 7.4 | **Decision 1 resolution, wired**: whichever of "reuse `_on_evict_block`" vs. "coordinator supersedes it" was chosen in §6 gets implemented for real against `BlockPool`. This is the step that actually changes `runtime/block_pool.py` or actually wires a callback — everything before this point is new, additive code with nothing calling it | Only in the sense that `BlockPool._on_evict_block` goes from always-`None` to sometimes-set — still no observable effect for Laguna, since nothing constructs a coordinator for a `needs_two_cache_families=False` checkpoint | Bit-exact regression on Laguna (must be unaffected — this is the step most likely to accidentally touch a shared code path) + the full invariant suite from 7.3 run against a synthetic two-resource fixture (no real model needed yet) | Bit-exact gate needs GPU; the synthetic fixture does not |
| 7.5 | **A6-adjacent closeout**: prefix-hit-rate metric wired into `/metrics`/bfdiag per Decision 5 (§6), documentation of the final chosen shapes back into `architecture.md` §3.2-C (replacing the "更正" note with the settled design), `roadmap.md` §1.5-S4 text corrected per §2.3 | None | Doc-link check (this document itself, see §8); C-LIVE (metrics surface changed) | Only for C-LIVE |

Sub-steps 7.1-7.3 have **no real consumer** until Track B exists (same argument
`architecture.md:424-426` already made for why A3 as a whole was moved to position 7, not
position 3) — they are pure infrastructure, buildable and fully testable today, zero GPU,
zero risk to Laguna. 7.4 is the step that first touches shared production code
(`block_pool.py`) and is where the bit-exact gate earns its keep. 7.5 is bookkeeping.

---

## 5. Why this is safe for Laguna specifically (summary of §2.1/§4.4, stated as a proof)

1. Laguna's `capabilities.prefix_cache` methods never call `BlockPool` (§2.1, direct
   source read).
2. `ArchitectureSpec.needs_two_cache_families` is `False` for Laguna today and is already
   tested against the real checkpoint (§2.4).
3. Every new type/allocator/coordinator method proposed in §7 is either (a) additive to
   the protocol (new capability field, defaulted `False`; new optional member, only
   required when the new capability is `True`) or (b) inert until something sets
   `_on_evict_block` or constructs a coordinator for a two-family checkpoint, neither of
   which happens for Laguna.
4. Therefore the only sub-step with any theoretical exposure to Laguna's behavior is 7.4
   (`_on_evict_block` goes from always-`None` to conditionally-set) — and even there,
   exposure requires someone to construct a coordinator for a checkpoint reporting
   `needs_two_cache_families=True`, which no checkpoint in production does.

This is a stronger position than "we will test carefully and not regress" — it is "the
new code and Laguna's live code do not share a call graph," which is why §4.4 could point
to a specific, small, provable set of proof obligations instead of a general regression
sweep.

---

## 6. Decisions needing a human call

**[OPEN]** — options given, no default silently assumed elsewhere in this document.

### Decision 0 — Should A3 also move Laguna onto the new content-addressed allocator?

The most consequential fork in this whole document, surfaced explicitly rather than left
as a background assumption inside §4.4/§5: §2.1 established that Laguna's real prefix
cache (same-slot linear token scan) is structurally separate from `BlockPool`'s
content-addressed, cross-request sharing. That separation is *why* zero behavior change
is close to automatic — but it is also a live, real limitation of what Laguna can do
today (no cross-slot/cross-request sharing of a repeated system prompt, for instance),
and A3 is precisely the step that builds the machinery that would fix it.

- **(a) Leave Laguna's mechanism untouched; A3 builds only the new infrastructure for a
  second (not-yet-existing) cache resource type.** This is the assumption every other
  section of this document is written against. Cost: a real, valuable improvement to
  Laguna (cross-request KV sharing) sits unbuilt for another cycle, even though the
  allocator that would provide it already exists and is already tested.
- **(b) While A3 is being built anyway, also rewire Laguna's `reconcile_prefix_hit`/
  `find_best_slot_for_prompt` onto `BlockPool`.** This would be a genuine behavior
  change for Laguna (better cache hit rates, possibly different latency
  characteristics under load) layered onto **already the highest-blast-radius step in
  the entire 8-step plan** (`architecture.md:436`'s own characterization). It reintroduces
  exactly the risk the plan's "zero-behavior-change steps before the risky one" ordering
  (`architecture.md:428-437`) was built to avoid, and would need its own bit-exact-under-
  the-new-mechanism gate distinct from A3's.
- **[JUDGMENT] Recommendation: (a).** Bundling a real Laguna behavior change into the one
  step this project's own risk register already flags as the largest-blast-radius item is
  the specific failure mode Track A's whole sequencing discipline exists to prevent — "we
  were already in there, so we also changed X" is how scope creep turns a contained step
  into an uncontained one. If (b) is worth doing, it is worth doing as its own
  small, independently-gated step *after* A3 lands (using exactly the "does the KV-only
  path get slower or better" bit-exact + throughput comparison A6 already runs), not
  folded into it. This recommendation is why §4.4/§5 write "out of scope for this step"
  as if settled — it is this document's judgment, not yet anyone else's decision.

### Decision 1 — Does the coordinator reuse `BlockPool._on_evict_block`, or supersede it?

- **(a) Reuse it.** `_on_evict_block` already exists, is exactly the right shape
  (`Callable[[int], None]`), and is exactly what the retired code used successfully. Cost:
  it is a callback *from* the KV allocator *into* something else — which is the coupling
  shape pitfall 1 warns against in spirit (an allocator calling out, rather than a
  coordinator observing both). In practice the direction here is benign (KV→coordinator
  notification, not KV→state-allocator control), but it is still the KV allocator knowing
  a second thing exists.
- **(b) Supersede it.** The coordinator polls or wraps both allocators' `evict`/`allocate`
  entry points itself, and `_on_evict_block` stays `None` forever, eventually removed.
  Cost: more new code; the existing dormant hook (already tested for the retired case)
  goes unused, which is a small waste but not a risk.
- **[JUDGMENT] Recommendation: (a) for the forward direction (KV pressure → drop
  checkpoint), because that direction is a pure notification and the hook is already the
  right shape and already load-bearing in tested retired code; but the reverse direction
  (state pressure → touch KV hash) should be a coordinator-owned method the state
  allocator calls, not a second symmetric callback wired the same way — the asymmetry in
  §4.3 rule 3 is exactly the thing a shared bidirectional-callback design would be easiest
  to accidentally make symmetric, which is the failure mode INV-A3-3 exists to name.** This
  is a recommendation, not a decision made here, because it trades a small amount of
  coupling against a small amount of new code and reasonable engineers could land either
  way.

### Decision 2 — Should A3 do better than "any state miss is a full compute-miss"?

Per §4.2: today's (retired) design treats `A > 0, G = 0` as a full recompute, not a
partial one. Options: **(a)** keep that simplification for A3 v1 (matches precedent,
least new code, `write-time dedup still reclaims the memory` per the existing comment so
it's not even wasteful, just not maximally fast); **(b)** support partial reuse (recompute
only the recurrent layers for the mismatched span, reuse attention KV for all of it) —
more implementation surface, unmeasured performance upside, and it is the kind of
correctness-adjacent complexity this project's own risk register (RK1) already flags GDN
work as prone to. **[JUDGMENT] Recommendation: (a) for A3 itself**, revisit only once
Track B has real profiling showing how often `A > G` actually fires in practice (an
empirical question §4.2 already noted as unmeasured) — do not build (b) speculatively.

### Decision 3 — Protocol shape: new member name, or overload the existing one behind a capability check?

Per §4.4 point 2: add a differently-named member that returns `PrefixHit` (required only
under `capabilities.recurrent_state`), or keep `reconcile_prefix_hit -> int` as the stable
contract and add a second, always-optional query for the two-number breakdown. **[OPEN]**
— this document intentionally does not pick, because whichever a real `Qwen36Backend`
implementation finds easier to satisfy is evidence this document does not yet have (no
`Qwen36Backend` exists — `IMPLEMENTED_BACKENDS` is `frozenset({"laguna"})` today,
`model_registry.py:76`), and the naming-debt note already on file at `protocol.py:29-46`
suggests this project prefers to defer renames until a real second call site forces the
question rather than guess ahead of it.

### Decision 4 — CUDA-graph state-neutral capture: new problem, not reusable from Laguna

Flagged for completeness, not decided here (it is Track B/B2 scope, not A3 scope): per
`docs/qwen36-rebuild-spec.md:135`, Laguna's decode graph never touches recurrent state, so
its warmup-reuse safety argument does not transfer. The retired code's answer (permanently
reserved `2 × batch_size` warmup slots, `cuda_graphs.py:87-130`) is available as a
starting point but was flagged in that same document as "全新问题，无可抄" — genuinely
needs re-deriving against whichever kernel Track B ends up using (B0-4's three options),
not against A3's coordinator design. Listed here only so it is not lost between two
documents that each assume the other owns it.

### Decision 5 — What exactly does "前缀命中率不回归" (A6 gate) measure once there are two hit numbers?

Today `/metrics`/bfdiag presumably reports one hit-rate number (not verified in this
document — out of scope for a design pass with zero GPU and no metrics code read this
round). Once `PrefixHit` exists, does the A6 non-regression gate track `kv_hit`-based
rate, `state_hit`(`effective`)-based rate, or both separately? **[OPEN]** — recommend
tracking both once 7.5 lands (§4.6), specifically *because* the gap between them
(§4.2's "how often does state_hit < kv_hit fire") is the one number this whole design
doesn't otherwise surface anywhere, and it is the number that would tell Track B whether
Decision 2 is worth revisiting.

---

## 7. Gates for this document

Per the task's instructions, this design must pass `ruff` + both pytest environments +
the doc-link check, even though it adds no production code. No source under `runtime/`,
`server/`, `oracle/`, `tests/`, `bfdiag/` was modified to produce it — this section
records what was actually run, in this worktree, against this new file.

Actually run, in this worktree (`work/a3-design-20260802`), against the final state of this
file:

- `ruff check .` (repo-wide) — **`All checks passed!`**, both under `~/.venvs/vllm`
  (torch/vllm/sparkinfer present, the environment `[[test-venv-is-venvs-vllm]]` specifies
  for this repo's real test runs) and under a freshly-built clean venv with only `.[dev]`
  installed (simulating the CI `lint-and-test` job, which has neither torch nor vllm).
- `ruff format --check runtime server loader model oracle tests tools` — **`191 files
  already formatted`** (clean venv); production packages untouched by this change, as
  expected.
- `python -m pytest -q`, `~/.venvs/vllm` (broad-coverage environment; real torch, real
  vllm/sparkinfer present) — **1179 passed, 3 warnings**, unchanged from the pre-existing
  baseline measured on this same worktree before this file was added.
- `python -m pytest -q`, clean `.[dev]`-only venv (CI `lint-and-test` job simulation, no
  torch, no vllm) — **794 passed, 127 skipped**, unchanged from the pre-existing baseline.
  (The CI `test-cpu-torch` job's own extra surface — `.[dev,serving,cpu-test]` + CPU-only
  torch wheel + triton, no sparkinfer/cutlass — was not separately provisioned this round;
  its unique exposure over the two environments actually run here is torch-dependent tests
  that skip under `~/.venvs/vllm`'s guards for a *different* reason than under the clean
  venv. Since this change touches zero Python files, this is not expected to matter, but is
  recorded as a gap rather than silently assumed equivalent.)
- Manual doc-link check: no automated doc-link checker exists in this repo today (searched
  for `doc.link`/`doc_link`/markdown-link-walking patterns under `tools/`, `scripts/`,
  `tests/`; none found — confirmed absent, not just unfound). This document contains no
  `[text](path)`-style markdown links at all (`grep -n '](' ` on the file returns nothing —
  it cites paths as backtick-quoted strings, matching `notes/prefix-cache-design.md`'s and
  `hybrid-cache-prior-art.md`'s own convention); every backtick-quoted `docs/*.md` and
  `notes/*.md` path referenced was checked by hand to exist in this worktree.

---

## 8. Appendix — citation index

For quick re-verification, every file:line cited above, grouped by file:

- `runtime/backends/laguna.py:2157-2178` (`reset_slot`), `:2179-2220` (`reconcile_prefix_hit`),
  `:2222-2260` (`find_best_slot_for_prompt`), `:2531-2548` (`find_prefix_match`), `:53`
  (`RESERVED_PHYSICAL_SLOTS=0`)
- `runtime/backends/protocol.py:59-79` (`BackendCapabilities`), `:82-127`
  (`SlotSnapshot`/`PrefixSnapshot`/`BackendSnapshot`), `:146-228` (`ModelBackend`),
  `:233-252` (`CAPABILITY_MEMBERS`/`REQUIRED_MEMBERS`)
- `runtime/block_pool.py:17-24` (`RESERVED_PHYSICAL_SLOTS`/`_physical_slot`, the index-0
  incident comment), `:45-79` (`_ssm_spec_row`), `:270-303` (`BlockPool` class docstring),
  `:305-338` (`__init__`, `_on_evict_block`), `:343-362` (`_evict_one`)
- `runtime/architecture.py:42-49` (cache-kind constants), `:56-64` (`LayerSpec.cache`),
  `:96-142` (`ArchitectureSpec`, `needs_two_cache_families`)
- `runtime/model_registry.py:63-76` (`REGISTRY`, `IMPLEMENTED_BACKENDS`)
- `tests/test_block_pool.py:39-52`, `tests/test_architecture_spec.py:80,85,168,203`,
  `tests/test_backend_protocol.py:32-38`
- `oracle/qwen36_vllm/gdn_state.py:33-92` (`_allocate_gdn_checkpoint_pool`),
  `:106-161` (`materialize_gdn_checkpoint`), `:183-209` (`evict_gdn_checkpoint`),
  `:211-226` (`_evict_gdn_checkpoints_for_budget`)
- `oracle/qwen36_vllm/prefix_cache.py:112-129` (`_compute_prompt_block_hashes`),
  `:131-166` (`reconcile_prefix_hit`, `L = G ≤ A`), `:168-224` (`restore_cached_prefix`)
- `benchmarks/prefix_cache_eviction_check.py:1-66` (module docstring, INV/R mapping),
  `:96-317` (pure-Python checks), `:458-782` (GPU checks)
- `notes/prefix-cache-design.md:310-340` (§3.4 reconciliation), `:408-422` (§3.9 eviction),
  `:426-558` (§4 INV1-INV9), `:750-797` (§8 decision forks), `:801-` (§9 appendix)
- `docs/qwen36-rebuild-spec.md:68-90` (§1.1 `gdn_state.py` verdict), `:140-146` (§1.5
  `prefix_cache.py` verdict), `:229-249` (§1.10 summary table), `:410-436` (§3.3, A3
  specifically)
- `docs/roadmap.md:158` (S4, stale), `:287` (Track A's A3 line)
- `docs/architecture.md:253-274` (§3.2-C, the 2026-08-01 correction), `:419-441` (§3.5.5,
  the 8-step plan)
- `notes/2026-08-01-hybrid-cache-prior-art.md` (full file — the note this document
  re-verifies and partially corrects)
- vLLM 0.25.1: `v1/core/single_type_kv_cache_manager.py:1065-1085,1151-1156,1196-1204`;
  `v1/core/kv_cache_coordinator.py:552-555,630-742`; `v1/core/kv_cache_utils.py` (~1067,
  ~1293, page-size unification, not line-cited precisely this round)
- SGLang `b296e1a5035b`: `srt/mem_cache/allocator/mamba.py:14-34`;
  `srt/mem_cache/base_prefix_cache.py:83-98,155-190`;
  `srt/mem_cache/hi_mamba_radix_cache.py:341,357,540,713,800-842,919,947,1016-1053,1083`
  (`_match_post_processor`, the `device_indices = value[:best_value_len]` truncation);
  `srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:651-666` (`_page_transfer`,
  KV-first + skip-on-partial-failure), `:725-806` (`_resolve_pool_transfers_allocation`,
  `rollback_allocated` defined `:739`, invoked `:777,802`)
- `git log`: `8ec9cd3` (2026-07-22, block_pool extraction), `a9cb932` (2026-07-30, oracle
  isolation)
