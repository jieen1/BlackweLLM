# Prefix Cache — Architecture & Phased Implementation Plan

> ⚠️ **数字更正（2026-08-02）：本文档中的 ~151 MB 是按 FP32 递归状态估的，实际是 BF16，
> 真值约 76 MiB，恰好一半。** B0-4 已确认 Qwen3.6 的 GDN state dtype 为 BF16
> （机制："单步 FP32 计算 + 跨步 BF16 舍入"）；B0-7 的容量表里 FP32 行正是
> 144.0 + 7.5 = **151.5 MiB**，BF16 行是 72.0 + 3.75 = **75.75 MiB**。
> B2 从真实 config 独立重算得到 ~77 MiB，与 BF16 行一致。
>
> 因此本文档所有由 151 MB 派生的数字都应对折：R8 的"256K 全 checkpoint ≈ 4.8 GB/前缀"
> 实际约 **2.4 GB**。风险方向不变，量级减半。
> 见 `notes/2026-08-02-trackB-b0-facts.md` §B0-7 与 `docs/implementation-plan.md` §7.1。


Status: **DESIGN, not yet built.** This is the durable reference the
implementation phases follow. It supersedes the "do not build it now"
recommendation in `notes/2026-07-18-session-review-and-next-steps.md`
§25.9 — the user has seen that trade-off and decided prefix caching is
CORE to the real workload (multi-agent coding) and must be built to
production quality. This document is *how* to build it well, not *whether*.

Author's stance: I am decisive here. Where reasonable engineers would
choose differently in a way that changes the plan, the fork is called out
explicitly in **§8 Decision forks** with a recommendation and reasoning —
not left open.

Reading order: §1 (why), §2 (the one finding that makes this tractable),
§3 (the design), §4 (why it's correct), §5 (how it's built, phase by
phase), §6 (what will bite us), §7 (what we deliberately leave out), §8
(the forks I resolved). §9 is a code-anchor appendix for implementers.

---

## 0. TL;DR

Build **vLLM-v1-style automatic prefix caching, adapted to this runtime's
fixed 4-slot scale**, in four rollback-safe phases. The model is hybrid
(16 full-attention + 48 GDN linear-attention layers + 1 MTP draft
attention layer), so the cache has **two co-indexed halves**:

1. **Attention KV (17 layers: 16 target + 1 draft)** — a real
   logical→physical **block table** over a shared physical block pool,
   with **chained per-block content hashing**, **reference counting**, and
   an **LRU free queue**. Block granularity `block_size=16`. This is a
   near-direct adaptation of vLLM v1's `BlockPool`/`FreeKVCacheBlockQueue`.
2. **GDN recurrent state (48 layers)** — a single **coarse-checkpoint
   snapshot** per cached prefix boundary (reusing this runtime's already
   built, already tested `snapshot_gdn_state`/`restore_gdn_state`
   primitives). Not block-composable; reusable only *at* an exact
   snapshot boundary. This mirrors vLLM's Mamba `"align"` mode.

The reconciliation rule (identical in spirit to vLLM's hybrid coordinator
fixed-point): **the effective reusable prefix length `L` = the deepest GDN
checkpoint boundary that is ≤ the attention full-block match and shares
the same chained hash, block-aligned.** A cache hit then **restores** the
GDN snapshot + **references** the shared attention blocks + **restores**
the draft KV, sets `kv_len = L`, and **continue-prefills only the suffix
`[L, prompt_len)`** through the *already-validated chunked-prefill
continuation machinery*. Copy-on-write is **designed out** (only full
blocks are ever published, and full blocks are immutable — vLLM's own
v1 rule). The two real sharing patterns (§1) fall out as two lookups into
this one cache, not two separate mechanisms.

---

## 1. Problem framing

### 1.1 The workload and the two sharing patterns

Target: Qwen3.6-27B-NVFP4, single RTX PRO 6000 (sm120, Max-Q), `capacity=4`
concurrent requests, MTP K=3, real multi-agent coding-agent traffic. Two
structurally different prefix-sharing patterns exist (established in
`notes/2026-07-18-session-review-and-next-steps.md` §25.9), and a
production-quality cache must capture *both*:

- **Pattern A — simultaneous fan-out.** N sub-agents launched together
  share an identical system prompt + initial repo/file-context bundle,
  then diverge. The shared prefix is *new* (not yet computed anywhere);
  the win is computing it once and sharing, plus not storing N copies.
  Arrives in the *same* admission window.

- **Pattern B — sequential per-conversation growth.** A single agent
  re-sends its full, growing conversation history every turn. Turn N's
  prompt = turn N−1's prompt + (assistant reply + new user message). The
  shared prefix is *already computed* (it's last turn's content); the win
  is skipping re-prefill of everything up to the last turn. Arrives
  *over time*, across rounds. This is the canonical vLLM APC use case and
  is likely the larger, more universal value.

There is a third, incidental variant of B worth naming because it changes
the design's required generality: **cross-request incidental sharing** —
two *unrelated* agents (different conversations, arriving at different
times) that nonetheless share a large common prefix (the same system
prompt, the same pinned codebase context). Only a **content-addressed**
cache captures this; neither "same-round dedupe" nor "same-session
continuation" alone does. The user explicitly named "system prompts,
shared codebase/file context" as prefixes to capture — so content
addressing (not just session affinity) is in scope for the final product.

### 1.2 The quantitative justification (real, first-party)

`notes/...` §25.4: native vLLM's own `--enable-prefix-caching`, hit by an
accidental byte-identical re-run of a 256K/c=4 prompt on *this exact
model/GPU*, produced **775.9 s cold → 49.6 s warm = ~15.4×**, with
near-identical acceptance-rate/mean-acceptance-length between the two runs
(only prefill cost differed). This is not a hypothetical or a
cross-hardware extrapolation — it is direct evidence of the ceiling this
design targets for Pattern B / exact-repeat, on our own hardware.

§25 also measured our runtime's *cold* long-context throughput and the
gap to native (ours ahead 1.24–2.68× at 64K/128K/256K, near-tie at 200K)
— but on the `-S` synthetic fixture which is *non-overlapping by design*,
i.e. the worst case for prefix caching on *both* sides. On real multi-turn
traffic, native's already-active APC pulls ahead on turn-2+ effective TTFT
in a way those cold numbers do not capture. **This cache is how we close
that specific gap**, not the raw-compute gap §14–19 already narrowed.

### 1.3 What "done" means

- Captures **both** patterns (A and B) and the incidental cross-request
  variant, via one content-addressed mechanism.
- **Partial** prefix matches (block-granular attention; checkpoint-granular
  effective reuse), not just byte-identical whole-prompt hits.
- Correct under the GDN recurrent-state constraint (the hard part).
- Correct interaction with every existing mechanism: chunked prefill,
  ragged admission, mid-flight admission, MTP draft/verify, CUDA graphs.
- Eviction under the real `capacity=4`.
- **Correctness above all** — this project's hard-won discipline is that
  GDN/quantization/state bugs are silent. Every phase ships behind a
  dedicated test that would catch such a bug *before* the next phase.

---

## 2. The finding that makes this tractable (read this before §3)

§25.7 framed prefix caching here as structurally huge because "there is no
logical→physical block-table indirection at all." That is true of the
*Python metadata construction*, but a full read of the runtime shows the
**substrate already supports indirection at the kernel level**. This
materially de-risks the whole effort:

1. **The attention kernel already consumes a paged block table.**
   `SM120GQAMetadata.kv_page_indices` / `kv_page_indptr`
   (`build_attention_metadata*`) is exactly vLLM's dense-block-table→CSR
   form — the kernel reads *"for logical page i of this request, attend
   over physical block `kv_page_indices[i]`."* The fixed-slot runtime just
   happens to always fill it with a contiguous `torch.arange(first_block,
   first_block + num_pages)` (direct_model_runner.py:199-201, 455-463).
   **Replacing that `arange` with an arbitrary per-slot list of physical
   block ids is a Python-side change; the kernel needs nothing.**

2. **The physical KV tensor is already a flat, block-granular pool.**
   `allocate_fixed_slot_kv_caches` allocates
   `(num_blocks, 2, block_size, num_kv_heads, head_size)` where
   `num_blocks = (num_slots + 1) * blocks_per_slot`
   (direct_model_runner.py:136-142). Blocks are individually addressable
   today; only `first_block = _physical_slot(slot) * blocks_per_slot` +
   `arange` ties a slot to a *contiguous* range. Nothing in the tensor
   layout requires contiguity.

3. **The write path is the same shape.** `_slot_mapping` /
   `_slot_mapping_batch` compute `block_id = first_block + pos //
   block_size` (direct_model_runner.py:1264-1277, 1348-1376) — the only
   other place the contiguous-offset assumption lives.

4. **The CUDA-graph decode/verify path already refills block indices
   per replay.** `CapturedBatchDecodeGraph._fill_buffers` writes
   `static_kv_page_indices` (worst-case sized `num_reqs * blocks_per_slot`,
   zeroed then refilled) and `static_slot_mapping` **every replay** from
   `range(first_block, ...)` (direct_model_runner.py:3413-3444). The
   captured kernel launches **never bake in physical block ids** — they
   read them from the fixed-address buffer at replay time. So a *variable
   set of reused vs. fresh blocks per round is already structurally
   supported*; only the fill formula changes.

**Consequence:** the "build the indirection layer" work reduces to four
Python touch-points (metadata read, `_slot_mapping` write, the graph's
`_fill_buffers`, and the allocator), plus the hash/refcount/eviction/GDN
layers on top. GDN, `_physical_slot`, and the reserved-slot convention are
untouched by the attention indirection. This is why Phase 0 (§5) can be a
*behavior-identical* refactor.

**The genuinely hard part remains GDN** (48 of 64 layers), which has no
per-block structure and must be handled by snapshotting — §3.3.

---

## 3. Architecture

### 3.1 Two co-indexed cache groups (mirroring vLLM's hybrid KV cache)

The model has three layer families, which map to **two** cache groups:

| Group | Layers | Mechanism | Sharing granularity |
|---|---|---|---|
| **Attention** | 16 target + 1 MTP-draft = 17 | block table + chained hash + refcount + LRU | `block_size = 16` tokens (block boundary) |
| **GDN** | 48 | single recurrent-state snapshot per boundary | coarse checkpoint (chunk boundary), all-or-nothing |

The 17 attention layers share **one** block-id namespace: physical block
`b` holds each layer's own KV for the same token range in that layer's own
`(num_blocks, …)` tensor. One block table per slot therefore addresses all
17 attention tensors consistently (target and draft KV for the same tokens
live at the same block id in their respective tensors). This is exactly
vLLM's "one KV-cache group, many same-shape layers" model. The draft layer
is *not* special — it is the 17th member of the attention group. Only GDN
is a separate group.

This two-group split is the load-bearing structural decision. Everything
below is "attention group behaves like vLLM v1 APC" + "GDN group behaves
like vLLM Mamba `align` mode" + "a reconciliation rule that ties them."

### 3.2 Attention group — block table, hashing, refcount, LRU

Adapted near-directly from vLLM v1 (`vllm/v1/core/kv_cache_utils.py`,
`block_pool.py`, `single_type_kv_cache_manager.py`; see §9 for anchors).

**Physical block pool.** Keep the existing flat KV tensors. Carve the
`num_blocks` physical blocks into:
- A small fixed **reserved region** (physical block 0 stays reserved, per
  `RESERVED_PHYSICAL_SLOTS`; see §4-INV7 for why this is preserved).
- A **shared pool** of the remainder, managed by a free list.

Blocks are allocated on demand as a slot's `kv_len` grows, not statically
partitioned. A `BlockPool` owns:
- `blocks: list[Block]`, each `Block{ block_id, ref_cnt, block_hash|None,
  prev_free, next_free }`.
- `free_queue: FreeQueue` — an intrusive doubly-linked list (front =
  evict-next, tail = most-recently-freed) supporting O(1) middle removal
  (needed to revive an evictable block on a late hit). Copy vLLM's
  sentinel-head/tail structure.
- `hash_to_block: dict[BlockHashWithGroupId, Block]` — the content index.

**Block table.** Per logical slot, `block_table[slot]: list[int]` where
index = logical block position, value = physical `block_id`. Append-only
for a slot's lifetime (a logical position's physical block never changes
mid-life — vLLM's own invariant, `block_pool.py:52`). The attention
read/write paths consult `block_table[slot]` instead of the `arange`:
- read: `kv_page_indices = block_table[slot][:num_pages]`
- write: `_slot_mapping` block = `block_table[slot][pos // block_size]`

**Chained block hash** (whole-prefix identity, the entire correctness
argument, and cheap): for full block *i*,
```
h_i = H(parent = h_{i-1} (or NONE_HASH seed), token_ids[i*16:(i+1)*16], extra_keys)
```
`extra_keys` carries the `kv_cache_dtype` and any future disambiguator
(fp8 vs nvfp4 KV must never collide); a process-global `NONE_HASH` seed
(or `PYTHONHASHSEED` for repro) prevents cross-run collisions. Block *i*'s
hash depends on *all* tokens `0..(i+1)*16`, so two prompts diverging at any
earlier token get different hashes for every block from divergence on —
which is exactly why prefix lookup can stop at the first miss. Store the
per-slot growing `block_hashes` list; only ever hash **completed full
blocks** (partial tail never gets a hash — §3.8).

**Reference counting.**
- fresh alloc → `ref_cnt = 1`.
- cache hit (`touch`) → `ref_cnt += 1`; if the block was parked in the
  free queue (`ref_cnt` was 0), yank it out first.
- release → `ref_cnt -= 1`; at 0, append to free-queue tail (retaining its
  hash so it stays hit-able until actually evicted).

**LRU eviction.** `get_new_blocks` pops from the free-queue front; if the
popped block still carries a hash, drop that hash from `hash_to_block`
(`_maybe_evict`) — only *then* is the cached content gone. Free a slot's
blocks in **reverse logical order** so deep-prefix (tail) blocks are
enqueued closest to the front and die first (keeps shallow, more-shared
prefixes longer). Blocks with no hash are prepended (evicted first).

### 3.3 GDN group — coarse-checkpoint snapshot (the hard part, precisely)

GDN's `conv_state` + `ssm_state` (per layer) is **one accumulated
recurrent value per slot**, the result of a sequential scan over *all*
tokens so far — not a per-position/per-block cache. Two consequences:

1. **It is deterministic in the prefix content** (same tokens ⇒ same
   state), so it *is* cacheable in principle.
2. **It is only reusable *at* an exact boundary.** You cannot reconstruct
   the state at position L from a state at position L' ≠ L; you need *the*
   snapshot at L, or you recompute the scan from the nearest earlier
   snapshot forward. There is no "compose earlier blocks" like attention.

This is exactly vLLM's Mamba constraint: its manager searches
right-to-left and keeps **exactly one** block — the deepest matching state
snapshot — padding earlier logical positions with `null_block`
(`single_type_kv_cache_manager.py:1044-1090`). We adopt the same shape.

**What a GDN checkpoint is.** A full-layer-stack snapshot: for all 48 GDN
layers, `(conv_state, ssm_state)` at an exact prefix length `Lc`. This
runtime *already has the primitive*: `snapshot_gdn_state(slot)` /
`restore_gdn_state(slot, snap)` (direct_model_runner.py:1760-1906), which
`torch._foreach_copy_` the 48-layer state to/from fixed-address GPU
buffers with generation/slot/consumed guards. Cost measured at **~151 MB
per snapshot** (all 48 layers conv+ssm; from the ~604 MB/4-slot figure in
`_allocate_gdn_snapshot_buffers`). This size — and that it is *independent
of prefix length* (it's the recurrent state, not the history) — is the
central budgeting fact for GDN caching.

**When checkpoints are materialized (near-zero extra cost).** A GDN
forward *already* writes its final state to the state tensor at the end of
every forward call. **Chunked prefill already ends a GDN forward at every
`chunk_size`-token boundary** (default 8192; direct_model_runner.py:2896-
2965) and correctly continues via `has_initial_state` on the next chunk.
So chunk boundaries are *natural, already-materialized* checkpoint points.
The GDN checkpoint policy is therefore:

- **Always** checkpoint at each cached prefix's completion boundary (turn
  end for Pattern B, shared-prefix end for Pattern A) — cheap, highest
  value.
- **Additionally**, checkpoint at each `chunk_size` chunk boundary during
  long prefills, so *intermediate* boundaries exist for cross-request
  partial sharing (the incidental variant). At 8192 stride this is a
  bounded number of 151 MB snapshots (32 for a fully-checkpointed 256K
  prefix ≈ 4.8 GB — evictable, and rare; most real shared contexts are
  8–32 K = 1–4 snapshots).

A GDN checkpoint is stored in a dedicated checkpoint pool (separate from
the live per-slot snapshot buffers, which stay reserved for their existing
role) and **keyed by the same chained attention block hash at boundary
`Lc`** — so an attention hit and a GDN snapshot for the *same* content are
guaranteed to describe the same prefix (§4-INV3).

### 3.4 Reconciliation — the one rule that ties the two groups

This is §25.8 point 5's "two mechanisms kept mutually consistent," made
concrete. On admission of a request with token ids `T`:

1. Compute chained block hashes of `T` (full blocks only), capped at
   `len(T) − 1` (the last token must always be recomputed for logits —
   vLLM's rule, `kv_cache_manager.py:225-231`).
2. **Attention match** `A` = number of leading blocks whose hash hits the
   index × 16 (left-to-right, stop at first miss).
3. **GDN boundary** `G` = the largest checkpoint boundary `Lc ≤ A` for
   which a GDN checkpoint exists **under the same chained hash at `Lc`**.
   If none, `G = 0`.
4. **Effective reuse `L = G`** (always ≤ A, always block-aligned because
   checkpoints are taken only at block-aligned boundaries).

This is the fixed-point of vLLM's `HybridKVCacheCoordinator`
(`kv_cache_coordinator.py:631-742`) specialized to two groups where the
attention group is downward-closed (any prefix of a hit is a hit) and the
GDN group is snapshot-constrained: the converged hit length is the min,
block-aligned, snapshot-constrained boundary. We don't need the general
iterative solver — two groups with `G ≤ A` gives `L = G` directly.

Because `G` is derived from checkpoints taken *during the same prefills
that populated the attention index*, in the common case any `A > 0` comes
with a `G > 0` at some chunk boundary ≤ A. The `A > 0, G = 0` case
(attention cached but GDN never checkpointed for this prefix) is treated
as a **compute miss** in v1 (prefill `[0, prompt)` fresh); attention
write-time dedup (§3.8) still reclaims the memory. This keeps v1 simple
and never wrong.

### 3.5 The reuse execution path (reuses validated machinery)

Given `L > 0` for a fresh slot `s`:
1. **Attention:** set `block_table[s]` = the `L/16` shared physical block
   ids (from the index), `touch` each (`ref_cnt += 1`) for all 17
   attention layers' shared namespace.
2. **GDN:** `restore_gdn_state(s, checkpoint_at_L)` → the 48-layer state;
   set `slot_gdn_initialized[s] = True`.
3. **Draft KV:** the draft layer is in the attention group, so its `[0,L)`
   blocks are already referenced in step 1. Set
   `slot_draft_sync_len[s] = L`.
4. **Bookkeeping:** `slot_kv_len[s] = L`, `slot_num_accepted_tokens[s] = 1`.
5. **Continue-prefill the suffix `[L, prompt_len)`** exactly as chunked
   prefill's chunk-2+ does: `_forward_batch(..., kv_lengths=[L], qo_len=
   suffix_len, commit=True, is_decode=False)` for the target, GDN with
   `has_initial_state=True`, plus the draft step-0 sync over the suffix.
   The suffix's fresh KV writes go to freshly-allocated **private** blocks
   (`ref_cnt = 1`) appended to `block_table[s]`.

Steps 1–4 reproduce *exactly* the state that computing `[0, L)` fresh in
slot `s` would have produced (same tokens ⇒ same fp8 KV bytes, same
deterministic GDN state). Step 5 is byte-for-byte the existing,
already-validated chunked-prefill continuation — the *only* new thing is
that "chunk 1" came from the cache instead of from a forward. This is the
correctness reduction that Phase 3's test must pin (§4-INV1, §5-P3).

### 3.6 The two patterns as two lookups into one cache

- **Pattern A (fan-out):** among same-round waiting requests, group by
  shared prefix. The first request of a group prefills fully, forcing a
  chunk/checkpoint boundary at the detected shared-prefix length (so a GDN
  checkpoint + full attention blocks exist there); the others immediately
  **hit** that just-populated entry via §3.5. (Phase 2 implements this
  directly by token comparison, without the persistent index, as a
  self-contained early win; Phase 3 subsumes it into "populate + hit.")
- **Pattern B (sequential) and incidental cross-request:** ordinary cache
  **hits** on entries populated by earlier rounds/turns/other requests,
  via §3.4–3.5. Session affinity (§5-P4) is an *optimization* on top —
  a caller-supplied `session_id` lets us skip the hash walk and, if the
  slot is still warm, continue in place with zero restore — but the
  content hash is the correctness-bearing fallback that also catches
  incidental sharing the session id can't.

### 3.7 Copy-on-write: designed out

vLLM v1 has **no COW** — grep-confirmed by the study. The rule that
eliminates it: **a block is published to the shared index only when it is
completely full, and a full+published block is immutable.** New tokens
always land in the slot's own private, `ref_cnt = 1` tail block; when that
tail fills, it is hashed and published. Shared blocks are strictly the
`[0, L)` full blocks, which are position-addressed and never written again
(attention KV is append-only; positions past the committed length are
never re-read). Therefore no sharer ever needs to write into a shared
block, and no copy path is needed. **We adopt this rule verbatim.** It is
also what makes the partial-final-block question (§3.8) trivial.

### 3.8 Partial blocks / alignment

Only whole 16-token blocks are hashed, published, and reused; the trailing
partial block is always private and recomputed. On a hit of a `100`-token
prefix, blocks 0–5 (96 tokens) may be shared; tokens 96–99 are recomputed.
Because GDN checkpoints are block-aligned and `L = G ≤ A` is block-aligned,
the shared boundary is always a block boundary — no partial-block sharing,
no COW, no alignment fixups beyond "round `L` down to a block multiple"
(already true since `L = G`). Additionally cap at `len(T) − 1` so the last
token is always recomputed for logits.

### 3.9 Eviction under capacity = 4

Unlike vLLM's hundreds of concurrent sequences over a huge pool, we have 4
live slots but a pool sized for 4 × (64K–256K). That is *substantial* spare
capacity for cached-but-unreferenced prefixes — the design *helps* here:
few live requests means most of the pool can hold cached prefixes.
Eviction is the LRU free queue (§3.2): a cached prefix's blocks are
`ref_cnt = 0` once no live slot references them but remain hit-able until
the pool pressure forces `get_new_blocks` to reclaim them (dropping their
hash). GDN checkpoints are evicted in lockstep with their keyed attention
blocks (when the attention entry's tail block is evicted, drop the
co-keyed GDN checkpoint) so the two halves never disagree about what is
cached (§4-INV3). Budget knobs: a cap on total GDN-checkpoint bytes
(default e.g. 8–16 GB) and the attention pool's own free-block count as
the admission signal.

---

## 4. Correctness invariants

These MUST always hold. Each is followed by why the interacting mechanisms
(MTP, CUDA graphs, chunked/ragged/mid-flight admission, eviction) do not
violate it. This project's discipline is that violations here are *silent*
— so every invariant maps to a dedicated test in §5.

**INV1 — Cache-hit equivalence.** A request served via a cache hit
(restore + continue-prefill the suffix) produces the *same* committed
tokens as the same request served by a full cold prefill, within this
project's established `NEAR_TIE_LOGIT_MARGIN = 2.0` tolerance (fp8/batch
non-associativity is the only permitted source of difference; a real
addressing/state bug is not).
- *Why it holds:* shared attention blocks hold identical fp8 KV bytes for
  identical tokens (deterministic quant); the restored GDN snapshot is the
  identical recurrent-state bytes; the suffix forward is the validated
  chunked-prefill continuation. The reduction is "restore reproduces
  fresh-compute state" (bytes) + "chunked continuation is correct"
  (already validated §19). *Test:* §5-P3 exact/near-tie diff of hit vs.
  cold for the same prompt, plus the signal-probe crosstalk check.

**INV2 — No cross-request / stale-block reads.** A request never attends
over a block it does not reference, nor a block whose content is not its
own true prefix, nor an evicted block.
- *Why:* a block is shareable only while its chained hash is in the index;
  the hash encodes the entire prefix, so a referenced shared block is
  provably this request's true prefix. `ref_cnt > 0` blocks are never in
  the free queue and never handed to `get_new_blocks`; eviction drops the
  hash *before* the block can be re-handed out. `touch` yanks a revived
  block out of the free queue before use. *Test:* §5-P2/P3 signal-probe
  (marker tokens per slot; zero leakage), plus an allocator unit test
  asserting `ref_cnt`/free-queue invariants.

**INV3 — Attention/GDN checkpoint agreement.** For any cached prefix, the
attention blocks and the GDN checkpoint describe *exactly the same* prefix
length and content; neither can be reused without the other agreeing.
- *Why:* the GDN checkpoint is keyed by the *same chained attention block
  hash at its boundary `Lc`*; `L = G ≤ A` is computed from that agreement;
  eviction removes the GDN checkpoint in lockstep with its keyed attention
  tail block. There is no code path that references attention blocks past
  the GDN boundary on a hit. *Test:* §5-P3 mixed-length/mismatched-prefix
  cases; an eviction test that evicts an entry and asserts both halves go.

**INV4 — MTP draft/verify safety across a partially-cached prefix.**
Speculative draft tokens are never cached; only committed tokens are; the
draft model's own KV and `slot_draft_sync_len` are consistent with the
target after a hit.
- *Why:* (a) the cache is populated only from *prefill* and from
  *committed* decode positions — draft/verify tokens beyond the commit
  point are never hashed/published (mirrors vLLM `kv_cache_manager.py:456-
  465`; rejected drafts can't poison the cache). (b) The draft layer is in
  the attention group, so a hit restores its `[0,L)` KV alongside the
  target's; §3.5 sets `slot_draft_sync_len = L`, and the suffix forward
  runs the draft step-0 sync over `[L, prompt)`, producing the anchor from
  the recompute — the draft never needs target hidden states for `[0,L)`
  (analogous to vLLM EAGLE's "recompute the last matched block," here
  generalized to "recompute the suffix"). (c) The K+1 dedicated SSM spec
  rows per slot (`ssm_rows_per_slot = 1 + num_speculative_tokens`,
  direct_model_runner.py:158) are per-slot scratch, *not* cached and *not*
  shared — matching vLLM's `MambaSpec.supports_eagle_cache_peek = False`
  rule that recurrent snapshots must not be rewound. `slot_num_accepted_
  tokens` bootstraps to 1 after a hit exactly as after a cold prefill.
  *Test:* §5-P3 multi-round MTP after a hit; `mtp_*_check` regression;
  reuse `mtp_gdn_rollback_check` semantics for the spec-row bootstrap.

**INV5 — CUDA-graph replay is oblivious to which blocks are reused.** A
captured decode/verify graph replays correctly regardless of how many of a
slot's blocks came from the cache vs. fresh compute, and regardless of
non-contiguous physical block ids.
- *Why:* the captured launches read `static_kv_page_indices` /
  `static_slot_mapping` / `static_state_indices` from fixed-address
  buffers refilled *every* replay (direct_model_runner.py:3413-3444); no
  physical block id is baked into the capture. The only change is that
  `_fill_buffers` sources page ids from `block_table[slot]` instead of
  `range(first_block, …)` — same buffer shape (worst-case
  `num_reqs * blocks_per_slot`, zeroed then filled), same `kv_page_indptr`
  per-request page counts. Dispatch branch still depends only on
  `qo_len`/dtype/config, never live kv_len or block layout (class docstring
  :3184-3193). Prefill (where hits happen) is eager, never graph-captured,
  so the cache machinery and the graph path are disjoint. *Test:* §5-P1/P3
  `cudagraph_decode_regression` / `mtp_verify_cudagraph_check` re-run with
  block-table addressing and with cache-hit-populated (non-contiguous)
  block tables; assert eager-vs-graph parity (`cudagraph_eager_parity_
  check`).

**INV6 — Append-only, immutable-published, private-tail.** A logical
position's physical block never changes mid-life; a published (full) block
is never mutated; a slot's in-progress tail block is private (`ref_cnt=1`)
until it fills.
- *Why:* the block table is append-only (vLLM invariant we adopt); publish
  happens only on fill; §3.7. This is what makes INV2 and "no COW" hold.
  *Test:* allocator unit test; a "publish only full blocks" assertion.

**INV7 — Reserved physical slot 0 preserved.** Physical block index 0 (and
GDN state row 0) stays reserved and unused for real data.
- *Why:* the 2026-07-16 root cause (direct_model_runner.py:41-51) —
  vLLM's convention never assigns index 0 to real request data, and doing
  so produced 100% deterministic wrong output. The new free-list allocator
  must **exclude physical block 0** from the shared pool (start the pool at
  1), and GDN state rows keep the `_physical_slot` offset. *Test:* Phase 0
  regression (20/20 deterministic-correct), and an allocator assertion
  that block 0 is never returned.

**INV8 — Ragged / mid-flight admission composes with hits.** Admitting a
batch where some requests hit the cache (different `L`) and others are cold
(`L = 0`), possibly alongside long-running active slots, is correct.
- *Why:* a per-slot `L` is just a per-slot `kv_lengths[i]` at the start of
  the suffix continue-prefill — exactly the ragged `qo_len`/`kv_lengths`
  list mechanism `_forward_batch` / `build_attention_metadata_batch` /
  `build_gdn_metadata_batch` already generalize to (direct_model_runner.py
  :397-425, 565-583). A mid-flight-admitted hit slot joins the next verify
  round like any freshly-prefilled slot. *Caveat:* `mtp_prefill_batch`
  currently raises `NotImplementedError` for ragged + `chunk_size` together
  (direct_model_runner.py:2773-2781). Suffix continue-prefill of *ragged*
  hit lengths that individually exceed `chunk_size` inherits this limit —
  §5-P3 must either lift it or restrict cache-hit suffix chunking to the
  uniform/short-suffix case (real hit suffixes are typically short, so the
  latter is an acceptable v1 scope, matching the existing rationale). *Test:*
  §5-P3 mixed hit/cold ragged admission + `mtp_ragged_prefill_check` /
  `mtp_async_arrival_check` extended.

**INV9 — Eviction never races a live reference.** A block referenced by
any active or mid-admission slot is never evicted; admission that needs
new blocks either finds free capacity or evicts only `ref_cnt = 0` blocks.
- *Why:* single-event-loop engine (server/engine.py:28-40) — admission and
  decode rounds run on one thread, so there is no concurrent allocator
  mutation; `touch` before use raises `ref_cnt` before any round reads the
  block; free happens only in `_finish_request`/`reset_slot`. The
  "reserve-before-forward" order (allocate + `touch` all needed blocks,
  *then* forward) is mandatory. *Test:* §5-P3 admission-under-pressure test
  that forces eviction while other slots are active; assert no active
  slot's blocks were reclaimed.

---

## 5. Phased implementation plan

Five phases (P0–P4). Each: **builds** a slice, ships behind a **dedicated
test that would catch a silent bug before the next phase**, and leaves a
**rollback-safe boundary** — if the next phase is not done, the system is
in a previously-validated working state. The gating headline is always the
4K/c=4 run (`mtp_w1s_our_runtime_perf --batched --cudagraph`) confirmed
*bit-identical* on `total_committed_tokens` / `draft_acceptance_rate_pct`,
plus the full `benchmarks/mtp_*_check.py` battery green — this project's
established zero-regression bar.

Every phase that touches addressing carries a **feature flag** (e.g.
`DirectModelRunner(..., enable_block_table=False)` default off, then
`enable_prefix_cache=False`) so `main` behavior is byte-identical until a
phase is proven and flipped on.

### P0 — Block-table indirection substrate (behavior-identical refactor)

**Build:** introduce `block_table[slot]: list[int]`, initialized to the
current contiguous range `[first_block, first_block + blocks_per_slot)`.
Route the four touch-points through it: `build_attention_metadata*`
(read), `_slot_mapping*` (write), `CapturedBatchDecodeGraph._fill_buffers`
(graph), and a thin allocator that hands out the *same* contiguous ids.
GDN, `_physical_slot`, reserved-slot-0 all untouched. Also add
prompt-prefix-overlap logging to `server/engine.py`'s `self.stats`
(§25.9's cheap recommendation) to gather real hit-rate/pattern data during
the build.
**Test (safety before P1):** full `mtp_*_check` battery + 4K/c=4 headline
**bit-identical** (same logits hash where applicable) — the refactor must
be a no-op. Add a unit test asserting `block_table` equals the old `arange`
for every slot. Re-run `cudagraph_eager_parity_check`.
**Rollback-safe:** identical behavior; the table just re-expresses today's
addressing.

### P1 — Dynamic free-list allocator + reference counting (no sharing yet)

**Build:** replace the static per-slot partition with a `BlockPool` (free
queue, `ref_cnt`, exclude block 0). Each slot allocates blocks on demand as
`kv_len` grows, frees on `reset_slot`/`_finish_request`. Still **no
cross-slot sharing** — every block has exactly one referencer, so behavior
is identical, but blocks are now dynamically placed (a slot's blocks may be
non-contiguous). This is the real proof the block table + graph path
tolerate non-contiguous ids.
**Test (safety before P2):** dedicated `prefix_cache_allocator_check.py` —
alloc/free/`ref_cnt`/free-queue-order invariants, block-0-never-returned
(INV7), append-only (INV6). Full battery + headline green. `cudagraph_
decode_regression` / `mtp_verify_cudagraph_check` with deliberately
**fragmented** (non-contiguous) allocations to exercise INV5.
**Rollback-safe:** dynamic allocation with 1 referencer/block is
behaviorally identical to P0; no cache semantics yet.

### P2 — Fan-out fork (Pattern A, same-round sharing; self-contained)

**Build:** at admission, detect a common token prefix among the same-round
`admit_now` batch by direct comparison (cheap for ≤4 requests). Prefill the
group leader fully, forcing a checkpoint boundary at the common-prefix
length `Lc` (block-aligned); `snapshot_gdn_state` there; for each sibling,
reference the leader's `[0,Lc)` attention blocks (`ref_cnt += 1`, all 17
layers), `restore_gdn_state` the snapshot, set `slot_draft_sync_len = Lc`,
and continue-prefill each sibling's suffix. No persistent hash index, no
eviction — the shared entry lives only for this admission. Reuses only the
P1 block table + the existing GDN snapshot primitive + chunked continuation.
**Test (safety before P3):** `prefix_cache_fanout_check.py` — N=2..4
siblings sharing a large prefix, distinct suffixes: (a) **INV1** each
sibling's committed tokens match an independent cold single-slot reference
(near-tie); (b) **INV2** signal-probe with per-slot marker tokens in the
*suffix*, zero crosstalk; (c) multi-round MTP decode after the fork
(INV4). Full battery + headline.
**Rollback-safe:** triggers only when ≥2 same-round requests share ≥
threshold tokens; otherwise byte-identical to P1. Feature-flagged.

### P3 — Persistent content-addressed prefix cache (Patterns B + incidental)

**Build:** the full mechanism — chained block hashing on prefill/commit,
`hash_to_block` index, `touch`/`free`/LRU eviction (§3.2); GDN
chunk-boundary checkpoints keyed by chained hash + a checkpoint pool with
byte-budget eviction (§3.3); the §3.4 reconciliation (`L = G ≤ A`,
block-aligned, capped at `len(T)−1`); the §3.5 restore-and-continue path;
write-time attention dedup (§3.8). Populate from cold prefills *and*
committed decode positions. Fan-out (P2) is re-expressed as "populate +
hit" (or kept as a fast path — the persistent cache subsumes it either
way).
**Test (the load-bearing round; safety before P4):**
`prefix_cache_hit_check.py` —
  - **INV1** cold-vs-hit exact/near-tie diff for the same prompt at
    several `L` (partial and full-prefix hits); repeat across 20+ decode
    rounds (INV4) and a natural-language + a code prompt.
  - **INV3** mismatched/mixed-length prefixes: a request sharing only the
    first `Lc` tokens must reuse exactly `Lc` (not more); a request whose
    attention matches deeper than the deepest GDN checkpoint must reuse
    only up to that checkpoint.
  - **INV2** signal-probe crosstalk with cache hits interleaved across
    slots; zero leakage.
  - **INV5** eager-vs-graph parity with cache-hit-populated non-contiguous
    block tables (`cudagraph_eager_parity_check` extended).
  - **INV8** mixed hit/cold ragged admission + mid-flight admission of a
    hit slot alongside long-running slots.
  - **INV9** admission-under-pool-pressure forcing eviction while slots are
    active; assert no live block reclaimed and no correctness failure.
  - eviction correctness: evict an entry, re-request it → clean cold
    recompute (INV3, both halves gone together).
  Plus full `mtp_*_check` battery + 4K/c=4 headline bit-identical, and a
  real long-context (≥64K) Pattern-B re-run demonstrating the actual
  speedup vs. the §25.3 cold number (target: approach the 15.4×
  exact-repeat ceiling on turn-2+ TTFT).
**Rollback-safe:** cache lookup returning `L = 0` reproduces P1 behavior
exactly; the whole cache is behind `enable_prefix_cache` (default off until
this round passes). Populating the index has no effect on a request that
never hits.

### P4 — Server integration, session affinity, and productionization

**Build:** plumb an optional `session_id` through `server/app.py`
(`ChatCompletionRequest`/`CompletionRequest`) and `GenerationRequest`; on
`_finish_request`, instead of unconditional `reset_slot`, *optionally
retain* the slot warm for a short TTL if a `session_id` is set (session-
affinity fast path — the next turn continues in place with zero restore,
subject to the real capacity=4 vs. tail-latency trade-off, made an explicit
policy knob). Session affinity is a pure *optimization* over P3's content
hash (which remains the correctness-bearing fallback and catches incidental
sharing). Wire the §25.9 hit-rate instrumentation into `/debug/stats`.
Raise the server's default `blocks_per_slot` so real long-context requests
are admissible (the §25.10 follow-up), now that a shared pool makes the
memory affordable.
**Test:** `server_e2e_check.py` extended — turn-1 then turn-2 of the same
session over real HTTP; assert turn-2 is a hit (measured TTFT drop +
`debug` fields prove reuse) *and* its committed tokens match an independent
cold reference (INV1 end-to-end). Confirm defensive rejections and health
still hold.
**Rollback-safe:** session affinity off by default; content-hash cache
(P3) already validated; server behavior without `session_id` unchanged.

**Sequencing note.** P0–P1 are pure infrastructure (no user-visible
change) and must land first — they are where the "no logical→physical
indirection" gap (§25.7) is actually closed, behind a no-op refactor. P2
is an optional early ship (real fan-out value on validated ground). P3 is
the big one but sits entirely on P0–P2's proven substrate. P4 makes it a
product. Each is a *dedicated round* in this project's sense (multiple
checks), consistent with §25.8's estimate — plausibly P3 is two rounds
(machinery, then long-context perf validation).

---

## 6. Risk register

| # | Risk | Why it's plausible here | Mitigation / catching test |
|---|---|---|---|
| R1 | **GDN state corruption on restore** — a restored snapshot silently doesn't equal fresh-compute state (wrong bytes, wrong layer, stale generation). | 48 layers, silent failure class this project repeatedly warns about; the snapshot primitive's generation/slot/consumed guards were added precisely because of past near-misses. | INV1 cold-vs-hit exact diff at GDN layer 0 first (proves addressing), then through the 48-layer stack (§5-P3), reusing the `mtp_chunked_prefill_check` methodology that already characterized the benign bf16-roundtrip signature. Keep the snapshot guards; add a checkpoint-hash tag so a wrong-prefix restore is rejected, not used. |
| R2 | **MTP draft desync after a hit** — draft KV or `slot_draft_sync_len` inconsistent with the restored target prefix, producing subtly-wrong drafts (the §_mtp_forward bug class: right shape, wrong content). | The draft is a separate KV + separate counter; the 2026-07-17 `prior_kv_len` bug shows this exact desync is easy to introduce and shape-checks miss it. | INV4 test: multi-round MTP after a hit with per-step oracle-aligned logits comparison (not shape checks), plus `draft_acceptance_rate` sanity vs. the cold baseline. Draft layer restored as part of the attention group (§3.5) so it can't drift from the target. |
| R3 | **CUDA-graph capture assumptions break under variable block reuse** — a captured launch reads a stale/contiguous block layout. | Historically the source of an illegal-memory-access crash on this backend. | INV5: the graph already refills all block indices per replay (§2/§4); P1 exercises non-contiguous allocations *before* any sharing; P3 re-runs `cudagraph_eager_parity_check` with cache-hit block tables. If parity ever fails, fall back to eager for hit slots (the graph path already has a documented eager fallback). |
| R4 | **Eviction race under concurrent admission** — a block evicted while another slot still needs it. | Classic prefix-cache failure mode. | INV9: single-event-loop serializes admission/rounds (no true concurrency); mandatory reserve-and-`touch`-before-forward order; §5-P3 admission-under-pressure test. |
| R5 | **Attention/GDN disagree on cached prefix length** — attention reused deeper than the GDN snapshot, so GDN state is for the wrong length. | The two-mechanism seam §25.8 explicitly flagged. | INV3: `L = G ≤ A` by construction; GDN checkpoint keyed by the same chained hash; lockstep eviction; §5-P3 mismatched-length cases. |
| R6 | **fp8 non-determinism defeats the equivalence test** — cold and hit legitimately differ by near-tie noise, masking or mimicking a real bug. | This project already established bytewise identity is not achievable even in native vLLM (batch non-associativity). | Use `NEAR_TIE_LOGIT_MARGIN=2.0` + signal-probe (the established methodology), *not* bytewise, as the INV1 bar; but require GDN-layer-0 exact match as the addressing proof (R1) so noise can't hide a wrong-block read. |
| R7 | **Hash collision → wrong prefix served.** | Content addressing's inherent risk. | Full-width hash (vLLM's `hash_function` over `(parent, tokens, extra_keys)`); `extra_keys` carries kv dtype; optionally verify token-ids on hit for the first block (cheap) in a paranoid mode. Collision probability with a 64-bit+ hash over ≤4-slot traffic is negligible; documented. |
| R8 | **Memory blow-up from GDN checkpoints** (~151 MB each) exceeds headroom at 256K. | 256K fully checkpointed at 8192 stride ≈ 4.8 GB per prefix. | Byte-budget cap on the checkpoint pool with LRU eviction; checkpoint only at chunk boundaries + completion boundaries, not densely; §3.9 knobs. §5-P3 long-context test monitors peak memory with the established climbing-trend watchdog. |
| R9 | **Cache hurts throughput** at low hit rate (hashing/lookup overhead, retained warm slots stealing capacity=4). | Small concurrency; hashing every prefill isn't free. | P0 instrumentation gives real hit-rate before P3 commits; lookup is O(blocks) dict probes on hashes computed once; session-affinity retention is a policy knob with a TTL, defaulting conservative; end-to-end re-check (not just microbench) per this project's rule. |
| R10 | **`reset_slot` no longer sufficient** — it resets counters but a slot's block table + refs must also be released, or blocks leak. | `reset_slot` today only touches counters (direct_model_runner.py:1733-1758). | Extend `reset_slot`/`_finish_request` to `free` the slot's blocks (decrement `ref_cnt`, return to free queue) and clear `block_table[slot]`; allocator unit test asserts no leak across many admit/finish cycles (reuse the D3 memory-flatness watchdog). |

---

## 7. Explicitly out of scope for v1

- **Cross-process / on-disk / persistent-across-restart cache.** vLLM's own
  APC is in-memory only; a 4-slot single-GPU workstation gains little from
  disk spill relative to its complexity and correctness surface. Revisit
  only if profiling shows repeated cold system-prompt prefills across
  server restarts dominate.
- **Distributed / multi-GPU shared prefix cache.** Single GPU; no.
- **Sampling-aware caching (temperature/top-p).** The server is greedy-only
  (server/app.py rejects non-greedy); caching is exact-prefix, which is
  sound for greedy/MTP-verify. Sampling is separate, larger work.
- **Dense per-block GDN caching (vLLM "all" mode).** Requires
  `supports_mamba_prefix_caching` and per-block state snapshots; at ~151 MB
  per full-stack snapshot it is not affordable at 16-token granularity here.
  We deliberately take the coarse-checkpoint ("align"-like) path.
- **General N-way radix *tree*** (branch-sharing beyond linear prefixes).
  Linear prefix chains (a list of block hashes per request) capture the two
  real patterns; a tree adds machinery for a sharing shape (mid-sequence
  branch reuse) the coding-agent workload does not exhibit. The chained-hash
  index already gives arbitrary N-way *prefix* sharing.
- **Copy-on-write.** Designed out (§3.7), not deferred — it is unnecessary
  given the full-block-immutable rule, and adding it would be pure risk.
- **Automatic checkpoint placement tuning.** v1 uses fixed policy
  (completion boundary + chunk boundaries). Adaptive placement keyed to
  observed reuse is a post-v1 optimization once P0's instrumentation has
  real data.

---

## 8. Decision forks (resolved, with reasoning)

**Fork 1 — v1 scope: two targeted mechanisms vs. a real content-addressed
cache.** One could ship only (A) same-round fan-out dedupe + (B)
session-affinity continuation, with *no* persistent content-hash index —
much less code, lower risk (this is close to §25.9's own recommendation).
**Resolved: build the content-addressed cache, but reach it in phases where
the early phases (P0–P2) deliver the targeted wins first.** Reasoning: the
user explicitly named "system prompts, shared codebase/file context"
shared across requests as a target — that is *incidental cross-request*
sharing (unrelated agents, different rounds), which *only* content
addressing captures; session affinity and same-round dedupe both miss it.
So the end state must be content-addressed. But we don't gamble the whole
thing at once: P2 ships fan-out value on the validated block-table
substrate, and session affinity (P4) rides on top of P3 as an optimization,
not a replacement. This gives the user the full capability they asked for
while preserving rollback-safe increments.

**Fork 2 — GDN checkpoint granularity.** Dense (every block, vLLM "all")
vs. coarse (chunk/completion boundaries, "align"-like) vs. session-only
(one snapshot per warm session). **Resolved: coarse — checkpoint at
`chunk_size` chunk boundaries + each cached prefix's completion boundary.**
Reasoning: dense is unaffordable (~151 MB × 16K checkpoints at 256K); the
chunk boundary is *free* (the GDN forward already materializes state there
during chunked prefill), block-aligned, and gives 8192-token-granular
cross-request partial sharing — enough for real shared contexts (8–32K =
1–4 snapshots) without the dense cost. `chunk_size` is a tunable knob;
default `_DEFAULT_PREFILL_CHUNK_SIZE = 8192`.

**Fork 3 — should the fan-out (P2) path survive after P3?** P3's persistent
cache subsumes fan-out (populate + hit). **Resolved: keep P2 as an explicit
same-round fast path** (detect shared prefix among `admit_now` by direct
comparison, prefill once, fork) because it avoids a race between "leader
populates the index" and "siblings look up" within one admission tick, and
avoids hashing the (possibly very long) shared prefix twice. It is a thin
wrapper over the same restore/continue primitives, so the maintenance cost
is low and the correctness surface is shared with P3.

**Fork 4 — session affinity: retain warm slot vs. always release + rely on
content hash.** Retaining a slot warm for a possible next turn competes
with the genuinely scarce capacity=4. **Resolved: default to
release-and-rely-on-content-hash; make warm retention an opt-in policy knob
with a short TTL.** Reasoning: content-hash reuse already delivers the
Pattern-B win (the next turn hits the cached blocks) *without* holding a
slot hostage; warm retention only saves the restore cost (small vs. the
prefill it already avoids) at the cost of tail latency for other requests.
Expose it, default it off, let real traffic (P0 instrumentation) justify
turning it on.

---

## 9. Appendix — code anchors

**This runtime (`runtime/direct_model_runner.py`)**
- `RESERVED_PHYSICAL_SLOTS` / `_physical_slot` — :41-55 (INV7)
- `allocate_fixed_slot_kv_caches` — :101-166 (flat pool, per-attn-layer
  tensors, GDN `ssm_rows_per_slot = 1 + num_speculative_tokens`)
- `build_attention_metadata` / `_batch` — :169-217 / :298-502 (the
  `arange` → block-table touch-point; `kv_page_indices`/`kv_page_indptr`)
- `build_gdn_metadata` / `_batch` / `_spec_batch` — :219-274 / :505-639 /
  :642+ (`state_indices = _physical_slot(slot)`, `has_initial_state`)
- `_slot_mapping` / `_slot_mapping_batch` — :1264-1277 / :1348-1376 (write
  touch-point)
- `_forward` — :1279-1333 (`slot_kv_len += n`, `slot_gdn_initialized=True`)
- `snapshot_gdn_state` / `restore_gdn_state` — :1760-1906 (the GDN
  checkpoint primitive; foreach_copy, gen/slot/consumed guards)
- `_allocate_gdn_snapshot_buffers` — :1169-1238 (~604 MB/4-slot ⇒ ~151
  MB/snapshot)
- `reset_slot` — :1733-1758 (must also free blocks — R10)
- `mtp_prefill_batch` — :2635-2981 (fresh-slot requirement; chunked loop;
  ragged+chunk `NotImplementedError` at :2773-2781)
- `_mtp_run_continuation_steps` (shared tail) / `_forward_batch` — :1378+
  (`kv_lengths`/`commit`/`qo_len` ragged — the suffix-continuation path)
- `mtp_verify_and_commit_batch` — :2983-3126 (spec-decode GDN, verify
  graph)
- `CapturedBatchDecodeGraph` — :3129+; `_fill_buffers` — :3360-3461
  (per-replay refill of `static_kv_page_indices`/`static_slot_mapping`,
  worst-case sizing — INV5)

**Server**
- `server/engine.py` — `_step` admission :299-433 (where lookup/fork hooks
  in), `_finish_request` :263-278 (calls `reset_slot` — R10 / P4 session
  retention point), `capacity_ok` :193-217, `GenerationRequest` :67-73 (no
  `session_id` yet — P4), `self.stats` :172 (hit-rate instrumentation — P0)
- `server/app.py` — request schemas :103-125, chat handler :205-249 (add
  `session_id` — P4), `_validate_capacity` :159-167

**vLLM v1 reference (adapt, do not copy — Apache-2.0, study only)**
`/home/bot/vllm` @ `e12b91b03`
- hash + chain: `kv_cache_utils.py:591-618`, `:701-742`; LRU queue
  `:179-422`
- `BlockPool`: lookup `block_pool.py:199-224`; alloc/`touch`/free/evict
  `:542-635`
- full-attn hit (left-to-right, stop at first miss)
  `single_type_kv_cache_manager.py:585-613`
- mamba hit (right-to-left, single snapshot, null-pad) `:1044-1090`
- hybrid reconciliation fixed-point `kv_cache_coordinator.py:631-742`
- mamba cache modes / constraints `model_executor/models/config.py:558-602`,
  `kv_cache_interface.py:683-718`; `supports_eagle_cache_peek = False`
  `:692-696`
- only-verified-tokens-cached `kv_cache_manager.py:456-465`; hit cap at
  `num_tokens-1` `:225-231`

(FlashInfer contains no block-management logic — it is a kernel library;
all prefix-cache design lives in vLLM.)
