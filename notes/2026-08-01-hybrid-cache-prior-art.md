# Hybrid KV + recurrent-state caching: what vLLM and SGLang already built

> Design input for **Track A3** (`docs/architecture.md` §3.2-C, §3.5.5 step 7).
> Read 2026-08-01, before A3 was designed. Both projects solved this problem in public.
>
> **Evidence**: read from local checkouts, not from web summaries.
> vLLM `0.25.1` at `~/.venvs/vllm025/lib/python3.12/site-packages/vllm/`;
> SGLang at `/home/bot/project/sglang` (`b296e1a503`, 2026-07-16). Quoted comments are
> verbatim. Note vLLM 0.26.0 is out and adds "partial prefix-cache hit for hybrid models"
> — **not** in the 0.25.1 read here, so that specific feature is unverified.

## Why this matters

`docs/roadmap.md` calls the coupling "the kind of bug that only shows up many tokens later,
the hardest class to find": continuing prefix B's KV with prefix A's recurrent state. Both
upstreams hit it. Their defences are more specific than "evict them together", and three of
them are non-obvious enough that we would probably have shipped the bug first.

## Files

| Project | Path |
|---|---|
| vLLM | `v1/core/single_type_kv_cache_manager.py` (`MambaManager`), `v1/core/kv_cache_coordinator.py`, `v1/core/block_pool.py` |
| SGLang | `srt/mem_cache/allocator/mamba.py`, `srt/mem_cache/hi_mamba_radix_cache.py`, `srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` |

---

## 1. Do not unify the allocators — A3's current framing needs changing

`architecture.md` §3.2-C frames A3 as a `SlotResourceManager` giving *"unified management
of two resource classes"*. **Both upstreams deliberately keep the allocators separate and
coordinate above them.** SGLang says so in the class docstring:

> "Unlike `BaseTokenToKVPoolAllocator` which is designed for per-token KV pages, Mamba
> slots are request-level (typically 1 slot per request). We keep the interface minimal and
> **do NOT inherit the KV base class**." — `allocator/mamba.py`

and in the module header:

> "Mamba caches one whole state tensor per request, so the allocator hands out fixed-size
> slots (1 per request) rather than paged token KV indices."

vLLM splits by type into `SingleTypeKVCacheManager` subclasses — `FullAttentionManager`,
`SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager`,
`SinkFullAttentionManager`, `RSWAManager`, `CrossAttentionManager` — coordinated by
`kv_cache_coordinator.py`.

They do not share an interface because they do not share a shape:

| | Paged KV | Recurrent state |
|---|---|---|
| Size | Grows with sequence length | **Fixed, one per request** |
| Granularity | Block / page | Whole-slot tensor |
| Shared across requests | Yes — content hash + refcount | **No** (§4) |
| Reuse | Reference the block | **Copy** the state |

**Change to make**: keep `block_pool` as the paged-KV allocator, add a separate
recurrent-state allocator, and make A3 the **coordinator that owns the invariant between
them**. Drop "unified" from the name — it is the word that will mislead the implementer.

## 2. Prefix search runs in the opposite direction, and block sizes must be aligned

`MambaManager.find_longest_cache_hit`, verbatim:

```python
# Search from right to left and early stop when a match is found.
for i in range(max_num_blocks - 1, -1, -1):
    if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
        ...
        break  # we just need the last match - early stopping
```

Full attention accumulates a longest common prefix left-to-right. Recurrent state has no
prefix — only the terminal state exists — so the search is right-to-left for the *latest*
usable checkpoint.

The second half is the part I would not have predicted:

```python
# When enable Mamba prefix caching, `block_size` will be aligned
# across full attention layers and Mamba layers to ensure the
# prefix hit length aligned at block
if (block_size != alignment_tokens  # Faster for common case.
        and (i + 1) * block_size % alignment_tokens != 0):
    continue
```

**They force block-size alignment between the two layer families** so the two hit lengths
can be expressed in the same units. And to keep downstream length arithmetic uniform, the
Mamba hit is padded with null blocks:

```python
# the hit length logic later assumes:
#  hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size
# so we insert dummy blocks at the beginning:
computed.extend([block_pool.null_block] * i)
computed.append(cached)
```

**Change to make**: A3's prefix match returns **two numbers**, `(kv_hit, state_hit)`, and
the scheduler may only skip prefill up to `state_hit`. Tokens between `state_hit` and
`kv_hit` must be recomputed for the recurrent layers even though their KV is present. Pick
the block-size alignment constraint deliberately rather than discovering it.

## 3. Speculative decoding: the conservative rule, stated by someone who got bitten

This lands directly on B3's hardest item. `MambaManager.remove_skipped_blocks`, verbatim:

```python
# NOTE (tdoublep) with async scheduling, the num_computed_tokens can contain
# draft tokens from the previous step that may or may not be rejected later.
# This can make us think we are further ahead in the sequence than we actually
# are, so let's assume that all tokens are rejected so we don't free blocks
# that we might actually need.
num_computed_tokens = max(0, num_computed_tokens - self.num_speculative_blocks)
```

**The rule: before freeing, subtract the entire speculative window — assume every draft
token is rejected.** Cheap, and it converts a correctness hazard into a bounded memory
cost.

Allocation is mode-dependent:

```python
# Allocate extra `num_speculative_blocks` blocks for
# speculative decoding (MTP/EAGLE) with linear attention.
if self.num_speculative_blocks > 0:
    num_tokens += self.block_size * self.num_speculative_blocks
```

…in the default mode, while `"align"` mode does not, because *"if x * block_size tokens are
scheduled, num_tokens is x * block_size + num_lookahead_tokens and breaks the alignment"*.

And the observation that may **delete** work rather than guide it —
`reachable_block_mask`:

> "A Mamba hit needs exactly the single state block ending on the boundary (no window, and
> **draft models have no mamba layers, so no eagle shift**)."

**If Qwen3.6's in-checkpoint MTP layer carries no GDN, recurrent speculative rollback does
not need solving at all.** That is a B0 question, answerable from `config.json` and the
`mtp.*` tensor names without a GPU, and it should be answered before B3 is scheduled.

## 4. Recurrent state cannot be borrowed across requests in the same step

`MambaManager.get_num_blocks_to_allocate`, verbatim:

```python
# Mamba can't rely on blocks generated by other requests in the current step
# To put it in the next step, we return num_gpu_blocks + 1 so
# that kv_cache_manager will think there is no enough blocks to allocate now
# and don't schedule it in the current step.
return self.block_pool.num_gpu_blocks + 1
```

They report a fake shortage to defer the request by one step. This is a **scheduler-level**
constraint, not a cache-level one, and it lands on our fixed-slot continuous batching:
admitting two requests on the same fresh prefix in one round is safe for KV and unsafe for
recurrent state. A3 must expose enough for the scheduler to see that; today `ServerEngine`
has no notion of it.

## 5. Eviction: two budgets, two accountings, and hits must test both

SGLang's `hi_mamba_radix_cache.py` is the more developed design. Its `evict` takes
**separate budgets per resource**:

```python
def evict(self, params: EvictParams) -> EvictResult:
    full_num_tokens = params.num_tokens
    ...
    evicted_full, evicted_mamba = self._evict_device_leaf(x)
    ...
    if params.mamba_num > 0:
        mamba_num_evicted += self.evict_mamba(params.mamba_num)
    return EvictResult(num_tokens_evicted=..., mamba_num_evicted=...)
```

Tree nodes carry both resources with independent state — `mamba_value`, and flags
`evicted`, `mamba_evicted`, `mamba_backuped` — plus separate accounting
(`mamba_evictable_size_`). Evicting a KV leaf frees both; there is also a mamba-only
eviction path.

The consequence for hit testing, which is where the silent-corruption bug would live:

```python
if last_node.evicted or (last_node.mamba_evicted and last_node.mamba_backuped):
```

**A node with live KV but evicted recurrent state is only usable if the state was backed
up.** So the two resources are allowed to diverge, and the hit test validates each
independently — which is stronger than "evict them together", and is the shape A3 should
copy. "Evict both when one goes" would be simpler and would throw away usable KV.

SGLang's `HybridCacheController` adds the partial-failure half, for the tiering path:
atomic allocation with `rollback_allocated()`, and a fixed ordering — KV pools first, then
extra pools only after KV fully completes, skipping extra IO entirely if KV terminated
early "to avoid data misalignment".

---

## 6. Proposed changes to A3, in order

1. **Split the abstraction** (§1): two allocators + one coordinator. Reword
   `architecture.md` §3.2-C; drop "unified".
2. **Two-number prefix match** (§2): `find_prefix_match → (kv_hit, state_hit)`; prefill
   skipping bounded by `state_hit`; decide the block-size alignment constraint explicitly.
3. **Conservative speculative freeing** (§3): subtract the full speculative window before
   freeing recurrent slots. Reserve `num_speculative_blocks` at allocation.
4. **Answer the MTP-carries-GDN question in B0** (§3) — it may delete B3's hardest item.
5. **Scheduler-visible same-step constraint** (§4): two requests on the same fresh prefix
   cannot both take the recurrent hit in one round.
6. **Per-resource eviction budgets and accounting; hit test validates both** (§5). Not
   "evict both together".

None of this needs a GPU, and all of it is cheaper than rediscovering the same constraints
from a NaN two thousand tokens into a run.

## 7. Also worth knowing

- vLLM 0.26.0 (newer than the local 0.25.1) adds *"partial prefix-cache hit support for
  hybrid models"* and *"selective hybrid cache retention"* per its release notes. If A3
  goes ahead, upgrade the local checkout and re-read `MambaManager` first — the design
  above may already have moved.
- SGLang v0.5.16 makes `UnifiedRadixTree` the default for SWA, Mamba and DSA models "with
  smarter state resets". The local checkout (2026-07-16) predates that release; the
  `hi_mamba_radix_cache.py` read here is the older path.
