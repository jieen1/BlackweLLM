"""A3 step 7-c (docs/a3-cache-coordinator-design.md §7): skeleton for the
second (recurrent/GDN) resource allocator.

Deliberately a **new module**, not an addition to ``runtime/block_pool.py``
(design doc's pitfall #1): ``BlockPool``/``hash_block_tokens`` are
well-tested but have zero production call sites (``runtime/`` and
``server/`` mention them exactly once, in a comment at ``server/app.py:1249``
explaining that ``LagunaBackend`` does NOT use them) -- tests passing there
proves the original author's assumptions were internally consistent, not
that the module works against a real caller. Building the state allocator as
its own module keeps this one's tests honest about the same risk: they cover
what is written here, not what a future real integration will actually need.

**Nothing constructs this today.** ``ArchitectureSpec.needs_two_cache_families``
is ``False`` for every backend this runtime currently ships (Laguna has no
recurrent layers), so step 7-d's coordinator forwards straight to the
backend's own ``reconcile_prefix_hit`` without ever instantiating a
:class:`RecurrentStatePool`. This module exists so Track B has a tested,
addressed, budget-aware skeleton to wire a real GDN checkpoint tensor pool
into -- not because anything needs one yet.

Ported methodology, not imported code (hard constraint: never import
``oracle/``). Two independent things get ported from two different existing
implementations, reimplemented fresh:

* Fixed-slot, non-paged addressing (:func:`physical_slot`, :func:`spec_row`)
  -- the shape of ``runtime/block_pool.py``'s ``_physical_slot``/
  ``_ssm_spec_row`` (``:23-24``, ``:45-79``), which address the ALWAYS-
  resident per-slot recurrent-state row (one per active decode slot, plus a
  dedicated row per MTP speculative column) that a real GDN kernel reads/
  writes every round. This is never evicted -- it is not a cache, it is the
  live state's home address.
* The evictable, byte-budgeted checkpoint side-pool (:class:`RecurrentStatePool`)
  -- the shape of ``oracle/qwen36_vllm/gdn_state.py``'s ``gdn_ckpt_meta``/
  ``_gdn_ckpt_lru``/``evict_gdn_checkpoint``/``_evict_gdn_checkpoints_for_budget``
  (``:183-226``), which is a SEPARATE, much smaller pool of persistent-cache
  checkpoints keyed by content hash, LRU-evicted under an independent byte
  budget, in bidirectional-asymmetric lockstep with the KV side (INV-A3-3).
  Test methodology (not code) is also ported from
  ``benchmarks/prefix_cache_eviction_check.py``'s pure-Python checks
  (``lockstep_eviction``/``refcnt_never_evicted``/``byte_budget``, ``:96-317``)
  -- see ``tests/test_recurrent_state_pool.py``.

RESERVED_PHYSICAL_SLOTS: explicitly decided, not inherited
------------------------------------------------------------
See docs/a3-cache-coordinator-design.md §1.8/INV-A3-8: ``block_pool.py``
reserves 1 physical slot; ``runtime/backends/laguna.py`` and
``laguna_cuda_graph.py`` both use 0. The design doc records why they
disagree: block_pool.py's ``=1`` is a **vLLM-scheduler-specific** fact
("vLLM's scheduler never produces physical index 0" -- a scheduler quirk,
not a hardware constraint; see ``docs/qwen36-rebuild-spec.md:131,603,677``),
and there has never been a vLLM scheduler behind Laguna's own addressing.
This module has never been driven by a vLLM scheduler either -- Track B
wires it to a LagunaBackend-family runner, addressed the same way that
runner already addresses its own KV slots. Inheriting block_pool.py's
``=1`` here would be exactly the failure this project has already had once:
block_pool.py's own comment records "physical index 0 ... makes the model
read/write the wrong state ... a 100%-deterministic wrong-output incident."
Explicitly choosing 0 (Laguna's convention) rather than defaulting to
block_pool.py's 1 is the point of this docstring existing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

#: See "RESERVED_PHYSICAL_SLOTS: explicitly decided, not inherited" above.
#: Matches runtime/backends/laguna.py's convention (0), NOT
#: runtime/block_pool.py's (1) -- the two are allowed to differ (§1.8), and
#: this module's choice is Laguna's, made on purpose.
RESERVED_PHYSICAL_SLOTS = 0


def physical_slot(logical_slot: int, *, reserved: int = RESERVED_PHYSICAL_SLOTS) -> int:
    """Fixed mapping from a logical decode slot to its physical state row.

    Reimplementation of ``runtime/block_pool.py``'s ``_physical_slot``
    shape, parameterized by ``reserved`` instead of silently assuming the
    module-global default -- callers that must match a specific runtime's
    convention (as opposed to this module's own default) pass it explicitly.
    """
    if logical_slot < 0:
        raise ValueError(f"logical_slot must be non-negative, got {logical_slot}")
    return logical_slot + reserved


def spec_row(
    logical_slot: int,
    col: int,
    total_physical_slots: int,
    num_spec: int,
    *,
    reserved: int = RESERVED_PHYSICAL_SLOTS,
) -> int:
    """Fixed physical row for MTP verify's K+1 candidate positions.

    Reimplementation of ``runtime/block_pool.py``'s ``_ssm_spec_row`` shape:
    column 0 always resolves to :func:`physical_slot` (the same row the
    ordinary, non-speculative path writes), so the bootstrap case (the first
    spec-verify round right after a real prefill) is correct by
    construction. Columns ``1..num_spec`` each get their own dedicated row
    in a separate address range past every slot's column-0 row, written
    unconditionally every verify round -- a rejected candidate's row is
    simply never selected by a future round's read, which is what removes
    any need for an explicit snapshot/restore/recompute repair on rejection
    (see ``block_pool.py``'s own docstring for the full derivation against
    the real kernel; unchanged here, only re-parameterized by ``reserved``).
    """
    if col < 0 or col > num_spec:
        raise ValueError(f"col={col} out of range for num_spec={num_spec}")
    phys = physical_slot(logical_slot, reserved=reserved)
    if col == 0:
        return phys
    return total_physical_slots + phys * num_spec + (col - 1)


@dataclass(frozen=True)
class CheckpointMeta:
    """Bookkeeping for one persistent-cache recurrent-state checkpoint.

    ``key`` is the caller's identity for "the co-keyed KV resource" (in the
    real integration, an attention tail block id or its hash -- opaque here
    on purpose, since this module has no KV pool to co-key against yet).
    """

    hash_value: Hashable
    num_tokens: int
    nbytes: int


class RecurrentStatePool:
    """Byte-budgeted LRU pool of recurrent-state checkpoints.

    Owns none of INV-A3-1/4/5/7 by itself (those belong to the step 7-d
    coordinator once a real second resource exists) but implements the two
    invariants that are purely about ITS OWN bookkeeping:

    * INV-A3-3 (bidirectional asymmetric lockstep): :meth:`evict` is the
      single method both directions call. The KV side calls it after
      evicting its own co-keyed block (forward -- unconditional cascade,
      the KV side already dropped its own hash). This pool's own
      byte-budget pressure (:meth:`evict_for_budget`) also calls it
      (reverse) -- in that direction, whether the co-keyed KV hash should
      also be dropped is decided by the injected ``should_drop_kv_hash``
      predicate, never by this pool guessing. With no KV pool wired at all
      (today's reality -- see module docstring), both callbacks default to
      ``None`` and eviction never touches anything KV-side, which is the
      only safe default when there is nothing to coordinate with yet.
    * INV-A3-4 generalized to this resource: a *pinned* checkpoint (backing
      a live, not-yet-completed generation's in-flight state) is never
      chosen by :meth:`evict_for_budget`, and :meth:`evict` raises if asked
      to drop one anyway -- a pinned key reaching forward-direction eviction
      means the KV side evicted a block whose co-keyed state this pool
      still considers live, which is a caller contract violation worth
      surfacing loudly rather than silently corrupting bookkeeping (matching
      ``runtime/block_pool.py``'s own style of hard ``RuntimeError``s for
      invariant violations, e.g. its INV7/INV9 checks).

    Byte budget is a soft cap, unlike ``BlockPool.allocate``'s hard
    exhaustion ``RuntimeError``: going over budget because every evictable
    checkpoint is pinned is tolerated (the KV pool is the contested, hard-
    limited resource; checkpoints are the cheaper "safety valve" --
    docs/a3-cache-coordinator-design.md §4). A caller can close the valve
    for its own family: :meth:`evict_for_budget` with ``include_pinned``
    makes the matching pinned keys evictable too, which is how the
    persistent prefix family keeps its own residency hard-bounded (plan
    §4.7 P1-M) without weakening the protection rolling-checkpoint
    pressure gets against it.
    """

    def __init__(
        self,
        byte_budget: int,
        *,
        should_drop_kv_hash: Callable[[Any], bool] | None = None,
        drop_kv_hash: Callable[[Any], None] | None = None,
    ) -> None:
        if byte_budget < 0:
            raise ValueError(f"byte_budget must be non-negative, got {byte_budget}")
        self.byte_budget = byte_budget
        self._should_drop_kv_hash = should_drop_kv_hash
        self._drop_kv_hash = drop_kv_hash
        self._meta: dict[Any, CheckpointMeta] = {}
        self._by_hash: dict[Hashable, Any] = {}
        # OrderedDict as LRU: front = oldest (evict-next), back = MRU.
        self._lru: OrderedDict[Any, None] = OrderedDict()
        self._pinned: set[Any] = set()

    def __len__(self) -> int:
        return len(self._meta)

    def __contains__(self, key: Any) -> bool:
        return key in self._meta

    @property
    def total_bytes(self) -> int:
        return sum(meta.nbytes for meta in self._meta.values())

    def register(self, key: Any, *, hash_value: Hashable, num_tokens: int, nbytes: int) -> None:
        """Materialize a new checkpoint at ``key``, MRU.

        Raises if ``key`` is already registered -- re-materializing a live
        key must go through an explicit :meth:`evict` first (mirrors
        ``BlockPool.cache_block``'s idempotent-skip-on-duplicate-VALUE
        guard, but keyed here on the checkpoint identity instead: a second
        ``register`` at the same key without an intervening evict is a
        caller bug, not a benign race, since nothing else in this module
        recomputes checkpoints)."""
        if key in self._meta:
            raise RuntimeError(f"checkpoint key {key!r} is already registered")
        self._meta[key] = CheckpointMeta(
            hash_value=hash_value, num_tokens=num_tokens, nbytes=nbytes
        )
        self._by_hash[hash_value] = key
        self._lru[key] = None

    def touch(self, key: Any) -> None:
        """Revive ``key``'s LRU recency on a hit, without changing pin state."""
        if key not in self._meta:
            raise RuntimeError(f"cannot touch unregistered checkpoint key {key!r}")
        self._lru.move_to_end(key)

    def get_by_hash(self, hash_value: Hashable) -> Any | None:
        return self._by_hash.get(hash_value)

    def pin(self, key: Any) -> None:
        """Mark ``key`` as backing live, in-flight generation -- never
        chosen by :meth:`evict_for_budget`; :meth:`evict` refuses it."""
        if key not in self._meta:
            raise RuntimeError(f"cannot pin unregistered checkpoint key {key!r}")
        self._pinned.add(key)

    def unpin(self, key: Any) -> None:
        self._pinned.discard(key)

    def is_pinned(self, key: Any) -> bool:
        return key in self._pinned

    def evict(self, key: Any) -> None:
        """Drop the checkpoint at ``key``. Idempotent no-op if absent.

        The single lockstep entry point for BOTH directions of INV-A3-3 --
        see the class docstring. Raises if ``key`` is pinned and still
        registered (contract violation: see class docstring)."""
        if key in self._pinned and key in self._meta:
            raise RuntimeError(
                f"refusing to evict pinned checkpoint key {key!r} -- INV-A3-4: "
                "a live resource must never be evicted by either allocator"
            )
        meta = self._meta.pop(key, None)
        if meta is None:
            return
        self._by_hash.pop(meta.hash_value, None)
        self._lru.pop(key, None)
        if self._should_drop_kv_hash is not None and self._drop_kv_hash is not None:
            if self._should_drop_kv_hash(key):
                self._drop_kv_hash(key)

    def evict_for_budget(
        self,
        incoming_bytes: int,
        *,
        include_pinned: Callable[[Any], bool] | None = None,
    ) -> None:
        """Evict LRU-oldest-first, skipping pinned keys, until adding
        ``incoming_bytes`` fits within ``byte_budget`` -- or until no more
        evictable (unpinned) checkpoints remain, in which case the budget
        is exceeded and this returns anyway (soft cap; see class
        docstring).

        ``include_pinned`` (plan §4.7 P1-M), when given, additionally makes
        pinned keys for which it returns true evictable: they are unpinned
        and evicted in LRU order exactly like unpinned keys. The caller
        chooses the direction: rolling-checkpoint pressure passes ``None``
        (persistent entries stay protected, the 2026-08-05 fix), while a
        new persistent store passes a predicate matching the persistent
        family so that family LRU-evicts within its own budget instead of
        growing residency without bound. Eviction still runs through
        :meth:`evict`, so the lockstep KV-hash callbacks fire for every
        key dropped."""
        total = self.total_bytes
        for key in list(self._lru.keys()):  # oldest-first; snapshot, evict mutates
            if total + incoming_bytes <= self.byte_budget:
                break
            if key in self._pinned:
                if include_pinned is None or not include_pinned(key):
                    continue
                self.unpin(key)
            nbytes = self._meta[key].nbytes
            self.evict(key)
            total -= nbytes
