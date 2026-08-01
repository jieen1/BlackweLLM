# SGLang RadixCache Prefix Cache — Technical Study (2026-07-31)

Source: `/home/bot/project/sglang/` (line numbers from this checkout).
Core file: `python/sglang/srt/mem_cache/radix_cache.py` (831 lines).

## 1. Data structures

### RadixKey (radix_cache.py:60)
- Wraps `array("q")` of token ids + optional `extra_key` (LoRA id / cache_salt namespace) + `is_bigram` (EAGLE) + `limit` (O(1) raw-token cap, avoids O(n) slice copy per prefill-batch build).
- `match(other, page_size)` (:162): exponential/gallop search + binary search over C-level `array` slice compares — no per-token Python loop on long shared prefixes; result floored to `page_size`.
- `child_key(page_size)` (:198): hashable dict key = first `page_size` logical units, namespaced by `extra_key` → `(extra_key, tokens)` tuple when extra_key set. Different extra_key ⇒ disjoint subtrees, never share.
- `page_aligned` (:136): truncates to multiple of page_size.
- `maybe_to_bigram_view` (:146): EAGLE mode flips bigram flag O(1); keys become overlapping (t_i, t_{i+1}) pairs, value truncated to len-1.

### TreeNode (radix_cache.py:217)
- `children: defaultdict(TreeNode)` keyed by `child_key`; `parent`; `key: RadixKey` (edge label); `value: torch.Tensor` (int64 KV-pool indices, one per token in key).
- `lock_ref` — eviction lock count (NOT a block refcount); `last_access_time`, `creation_time`, `hit_count`, `priority` — eviction metadata.
- `host_value` / `host_ref_counter` / `hash_value` — HiCache L2 extension.
- `evicted` property ⇔ `value is None`.

### RadixCache (radix_cache.py:280)
- Mixins: `SessionRadixCacheMixin` (session tagging), `KVCacheEventMixin` (event bus); base `BasePrefixCache` (base_prefix_cache.py, ABC + MatchResult/params dataclasses).
- State: `root_node` (lock_ref=1, never evicted), `evictable_size_`, `protected_size_`, `evictable_leaves: set` (incrementally maintained), `eviction_strategy`.

## 2. KV block sharing model — ownership transfer, no CoW, no per-block refcount

The allocator (`allocator/token.py:29 TokenToKVPoolAllocator`) is a plain free-list (`free_pages`/`release_pages` torch tensors; `alloc` pops, `free` cats back). It has zero reference counting.

Every physical KV slot is owned by exactly ONE of:
1. the allocator free-list,
2. exactly one tree node's `value` tensor,
3. a live request's uncached tail (in its `req_to_token` row).

Sharing = multiple requests' `req_to_token` rows pointing at the tree's canonical indices for the cached prefix. No KV data is ever copied on the standard path.

Ownership lifecycle:
- Admission: `match_prefix` returns the tree's indices as `req.prefix_indices`; `alloc_for_extend` (allocation.py:303) → `write_cache_indices` (allocation.py:55) writes `[tree indices | freshly allocated extend slots]` into `req_to_token[req_pool_idx]` (Triton kernel or Python fallback). Only the extend part allocates new slots.
- `cache_unfinished_req` (radix_cache.py:489, chunked prefill): inserts fill_ids so far; tree takes ownership of the NEW tail (`value.clone()` into new node, _insert_helper:705); request's duplicate indices for the already-cached range `[cache_protected_len : prefix_len]` are freed; row is rewritten to the tree's canonical indices via a second `match_prefix` + `req_to_token_pool.write`; lock moved old→new last_node.
- `cache_finished_req` (radix_cache.py:437): inserts `(origin_input_ids + output_ids)` page-aligned; frees duplicates `[cache_protected_len : prefix_len]`; frees unaligned tail `[key_len:]`; `dec_lock_ref(req.last_node)`.

## 3. Eviction (radix_cache.py:564)

- Trigger: lazy, from allocation paths only — `alloc_token_slots`/`alloc_paged_token_slots_*` (allocation.py:146/188/397) → `evict_from_tree_cache` (common.py:105) → `tree_cache.evict(EvictParams)` iff `allocator.available_size() < num_tokens`. No background eviction.
- Mechanism: min-heap over `evictable_leaves` keyed by `eviction_strategy.get_priority(node)`; pop leaf → `allocator.free(x.value)` → `_delete_leaf` (:778) removes from parent; if parent now childless AND `lock_ref==0`, push parent. Over-evicts to leaf granularity.
- Strategies (evict_policy.py): LRU (default, `last_access_time`), LFU `(hit_count, time)`, FIFO, MRU, FILO, Priority `(priority, time)`, SLRU (probationary/protected by hit_count threshold).
- Leaf set maintained incrementally by `_update_leaf_status` (:789): evictable ⇔ not evicted ∧ lock_ref==0 ∧ no non-evicted child. Updated on insert/split/lock/unlock/delete.

## 4. How prefill skips cached prefix

1. `Req.init_next_round_input(tree_cache)` (schedule_batch.py:1143): builds `RadixKey(full_untruncated_fill_ids, extra_key, limit=_compute_max_prefix_len(input_len))`; calls `tree_cache.match_prefix`; stores `prefix_indices`, `last_node`, `cache_protected_len`.
2. `_compute_max_prefix_len` (schedule_batch.py:1263): caps match at `input_len - 1` (always ≥1 token prefilled — needed for logprob/last-token logits) and at `logprob_start_len` when logprobs requested.
3. `scheduler._get_new_batch_prefill_raw` (scheduler.py:2782) → `PrefillAdder.add_one_req` (schedule_policy.py:976): budget = `fill_len - len(prefix_indices)` (only uncached tokens count against prefill budget); under `_lock_node(req.last_node)` (temporary inc/dec_lock_ref around admission, :846) then `_req_inc_lock_ref(req)` (:763) for accepted reqs; host load-back (`init_load_back`) if HiCache host hit.
4. `ScheduleBatch.prepare_for_extend` (schedule_batch.py:2111): `input_ids = fill_ids[len(prefix_indices):]`; `alloc_for_extend` allocates only extend slots and writes prefix (borrowed tree indices) + extend (new) into the req row. Attention reads KV through the row — cached prefix is never recomputed.

## 5. Granularity

Token-level tree (node.value has one KV index per token); dict branching key = first `page_size` tokens. With `page_size==1` any token-boundary prefix can be shared and splits expose arbitrary boundaries (finer than vLLM's per-block hashing). With `page_size>1`: match/insert lengths floored to page size; unaligned tail stays request-owned and is freed at finish.

## 6. Correctness invariants (what prevents stale/corrupt KV)

I1. Single ownership — each KV slot in exactly one of {free-list, one tree node, one request tail}; duplicates freed immediately at tree takeover (`free(kv_indices[cache_protected_len:prefix_len])` in cache_unfinished/finished_req). Double-free prevented by `cache_protected_len` watermark (page-aligned cached length; partial page freed exactly once).

I2. Borrow safety via lock_ref — a request borrowing tree indices holds `lock_ref>0` on the ENTIRE matched chain (inc_lock_ref walks node→root, :593) from admission until finish/unchunked-cache swap. Eviction only touches `lock_ref==0` nodes ⇒ borrowed indices can never be freed/reallocated under a live request. evictable_size_/protected_size_ moved in lockstep.

I3. Leaf-only eviction — internal nodes with live children are never evicted; tree shrinks leaves-inward; parent becomes evictable only after all children gone and unlocked. A request's prefix chain cannot lose an internal segment.

I4. Content-addressed keys — exact token equality (RadixKey.match); same tokens ⇒ same KV (causal attention determinism) makes sharing semantically valid. `extra_key` namespaces LoRA/cache_salt so different adapters never share nodes. Embed overrides (`positional_embed_overrides`) force an empty match key (schedule_batch.py:~1190) — same tokens with different embeddings must not share.

I5. ≥1 token always prefilled — match capped at input_len-1 (schedule_batch.py:1263) so the extend pass always produces the final token's KV + logits.

I6. Split preserves data & locks — `_split_node` (:675) clones BOTH halves of `value` (no parent/child aliasing); new parent inherits child's `lock_ref`, so locked chains stay fully locked across splits. hash_value split lazily.

I7. SWA re-prefill — hybrid SWA layouts cap match by `swa_reprefill_tail_tokens()` (schedule_batch.py:~1180): SWA KV lives in a per-request ring, not content-stable, so the trailing window is recomputed into this request's ring.

I8. Retraction — memory pressure: running reqs retracted with `is_insert=False` (schedule_batch.py:1698), KV freed, re-queued with `retracted_stain`; prefix re-matched on reschedule.

I9. Deterministic mode — `disable_finished_insert` skips tree insert on finish.

## 7. Multi-turn conversation data flow

Turn 1: P1 → match (miss) → prefill all → decode O1 → finish: insert(P1+O1) as new chain, dec_lock_ref → chain evictable (LRU).
Turn 2: P2 = P1+O1+U2 → match_prefix hits the P1+O1 chain → prefix_indices = tree's KV slots → inc_lock_ref → extend computes only U2 → decode O2 → finish: insert extends chain (new tail node for U2+O2), duplicates freed, dec_lock_ref.
The tree is a conversation trie; shared system prompts become shared internal nodes automatically; reuse is opportunistic LRU unless `--enable-session-radix-cache` (session_radix_cache.py) tags leaves by session_id so `release_session` frees a session's chains on close (nodes shared by several sessions freed only when last holder closes; closed-id tombstone LRU of 8192 guards late finishes).

## 8. Extensions (noted, not detailed)

- HiCache (hiradix_cache.py): `host_value` CPU backup, `backuped`, MatchResult.host_hit_length, `init_load_back` under temporary `_lock_node` in add_one_req; write-through pending ids.
- SWA/hybrid (swa_radix_cache.py, pure_swa_radix_cache.py), Mamba (mamba_radix_cache.py, hi_mamba_radix_cache.py, COW mamba), unified cache (unified_radix_cache.py), C++ radix tree (cpp_radix_tree/, radix_cache_cpp.py), chunk_cache.py (no-tree baseline).
- free_group batching (allocator/base.py:69): `free_group_begin/end` coalesces frees per scheduling step.
- `SGLANG_RADIX_FORCE_MISS` env zeroes matches for debugging.

## 9. Key file map

| File | Landmarks |
|---|---|
| mem_cache/radix_cache.py | RadixKey:60, TreeNode:217, RadixCache:280, match_prefix:355, insert:415, cache_finished_req:437, cache_unfinished_req:489, evict:564, inc/dec_lock_ref:593/608, _match_prefix_helper:649, _split_node:675, _insert_helper:705, _delete_leaf:778, _update_leaf_status:789 |
| mem_cache/base_prefix_cache.py | BasePrefixCache ABC, MatchResult/Params dataclasses |
| mem_cache/allocator/token.py:29 | TokenToKVPoolAllocator — free-list, no refcount |
| mem_cache/allocation.py | write_cache_indices:55, alloc_token_slots:146, alloc_req_slots:252, alloc_for_extend:303 |
| mem_cache/common.py | maybe_cache_unfinished_req:98, evict_from_tree_cache:105, release_kv_cache:131 |
| mem_cache/evict_policy.py | LRU/LFU/FIFO/MRU/FILO/Priority/SLRU |
| mem_cache/session_radix_cache.py | session tagging mixin |
| managers/schedule_batch.py | init_next_round_input:1143, _compute_max_prefix_len:1263, prepare_for_extend:2111; memory_pool.py ReqToTokenPool:238 (int32 [max_reqs, max_ctx] table) |
| managers/schedule_policy.py | PrefillAdder.add_one_req:976, _req_inc_lock_ref:763, _lock_node:846, add_chunked_req:805 |
| managers/scheduler.py | _get_new_batch_prefill_raw:2782, stash_chunked_request:2532 |
| managers/scheduler_components/batch_result_processor.py | finish→release_kv_cache:96/241, unfinished→maybe_cache_unfinished_req:244 |
