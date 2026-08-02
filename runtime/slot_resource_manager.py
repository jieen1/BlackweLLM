"""A3 step 7-d (docs/a3-cache-coordinator-design.md §7): the cache
coordinator skeleton.

Named to match the forward reference already planted in shipped code:
``runtime/architecture.py``'s ``LayerSpec.cache`` field docstring says "This
is the field ``SlotResourceManager`` (step 7) is built around: a checkpoint
mixing both is what makes two cache families necessary rather than
hypothetical." (``architecture.py:59-62``).

Scope, precisely
-----------------
Reads :attr:`ArchitectureSpec.needs_two_cache_families` (§1.5 -- an
existing, already-checkpoint-verified signal parsed from ``config.json``;
deliberately NOT a new ``BackendCapabilities`` field, per §8 decision 3,
which would duplicate a signal that already exists and must then be kept
in sync with it forever) to decide how to answer ``reconcile_prefix_hit``/
``find_best_slot_for_prompt``:

* ``needs_two_cache_families`` is ``False`` (every backend this runtime
  ships today -- Laguna has no recurrent layers, ``tests/
  test_architecture_spec.py::test_laguna_has_no_recurrent_layers``): pure
  forward to the backend's own implementation. No second allocator is
  instantiated. §5's zero-behavior-change argument for this whole migration
  step rests entirely on this branch being a literal passthrough -- see
  ``tests/test_slot_resource_manager.py``'s shadow-consistency tests for the
  claim made concrete (byte-for-byte equal to calling the backend directly).
* ``needs_two_cache_families`` is ``True``: §7 row 7-h, landed by Track B /
  B2. This used to raise ``NotImplementedError`` pointing here; what
  replaced it is deliberately **small**, because most of the two-resource
  work belongs to the backend that owns both allocators
  (``runtime/backends/qwen36.py``). Two things do belong here, and only
  here:

  1. **The min rule, applied once.** §3's decision -- "取 ``state_hit``
     （恒 ``<= kv_hit``，即取 min），且必须 block-aligned" -- is stated in
     one place and enforced in one place, instead of being a rule every
     future backend is trusted to have reimplemented correctly. A backend
     that answers with a ``state_hit`` above its own ``kv_hit`` cannot even
     construct the ``PrefixHit``; a backend that answers with an
     unaligned one is clamped here.
  2. **Ranking free slots by ``.effective``, not by KV depth.** A backend's
     ``find_best_slot_for_prompt`` returns one number, and for a
     single-family backend that number is unambiguous. With two families it
     is not: a slot whose KV matches 900 tokens but whose recurrent
     checkpoint reaches 0 is worth exactly as much as a cold slot. Picking
     by KV depth there does not produce a wrong answer -- it produces a
     *slower* one, silently, which is the failure mode §2's INV-A3-6 row
     describes ("纯性能回归，没有正确性信号"). This is the only place with
     both the per-slot numbers and the authority to choose.

Wired into ``server/engine.py`` as of step 7-g
(docs/a3-cache-coordinator-design.md §7 row 7-g): ``ServerEngine.
slot_resources`` constructs one of these (bound to ``self.runner``/
``self.architecture_spec``) on every access, and the admission block's two
``capabilities.prefix_cache`` call sites (``find_best_slot_for_prompt``,
``reconcile_prefix_hit``) read it instead of calling ``self.runner``
directly. Every production checkpoint's ``needs_two_cache_families`` is
still ``False`` today, so this remains a pure forward in practice -- 7-g
made the call path real without changing what any call answers (its own
gate is the shadow-consistency claim this module's tests already made,
plus the same claim re-checked at the wiring boundary in
``tests/test_engine_prefix_cache_admission.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from runtime.backends.protocol import PrefixHit

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module torch-free
    from runtime.architecture import ArchitectureSpec
    from runtime.backends.protocol import ModelBackend

_MISSING_PER_SLOT = (
    "backend {cls} declares capabilities.prefix_cache with a checkpoint whose "
    "ArchitectureSpec.needs_two_cache_families is True, but does not implement "
    "prefix_hit_for_slot(token_ids, slot) -> PrefixHit. The coordinator needs "
    "per-slot (kv_hit, state_hit) to rank free slots by the recurrent-bounded "
    "depth (docs/a3-cache-coordinator-design.md §3); ranking by whatever a "
    "backend's own find_best_slot_for_prompt returns would silently reuse "
    "slots whose recurrent state cannot actually be resumed."
)


class SlotResourceManager:
    """Coordinates cache-family resources across one backend's slots.

    Holds a bound ``(backend, architecture_spec)`` pair for the lifetime of
    one ``ServerEngine``, mirroring how ``ServerEngine`` itself reaches the
    execution layer through exactly one ``self.runner`` attribute
    (``runtime/backends/protocol.py``'s own module docstring).

    Deliberately does NOT re-check ``backend.capabilities.prefix_cache`` --
    that gate belongs to the caller (``server/engine.py``, since step 7-b),
    the same way it always has. This class answers "how do the resources
    reconcile", not "is this backend allowed to be asked at all".
    """

    def __init__(
        self,
        backend: ModelBackend,
        architecture_spec: ArchitectureSpec,
        *,
        block_size: int = 1,
    ) -> None:
        self._backend = backend
        self._spec = architecture_spec
        #: Only consulted on the two-family branch, where §3 requires the
        #: chosen boundary to be block-aligned. Defaults to 1 (no-op) so the
        #: single-family branch stays a literal passthrough even in the
        #: shadow-consistency tests that construct this without a block size.
        self._block_size = max(1, int(block_size))

    @property
    def needs_two_cache_families(self) -> bool:
        return self._spec.needs_two_cache_families

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        """INV-A3-2 (``state_hit <= kv_hit``) is enforced by
        :class:`PrefixHit` itself (``runtime/backends/protocol.py``'s
        ``__post_init__``) regardless of which branch below answers -- this
        method does not re-derive it, only forwards or clamps.

        The two-family branch applies §3's rule on top: ``state_hit`` is
        floored to a ``block_size`` boundary. A backend cannot hand back
        something above ``kv_hit`` (``PrefixHit`` refuses to exist), but it
        *can* hand back an unaligned boundary, and an unaligned resume
        point is not a crash -- it is a wrong answer several hundred tokens
        later. Flooring is always safe: a shorter reusable prefix is a
        performance loss, never a correctness one.
        """
        hit = self._backend.reconcile_prefix_hit(token_ids)
        if not self._spec.needs_two_cache_families:
            return hit
        return self._aligned(hit)

    def find_best_slot_for_prompt(
        self, token_ids: list[int], free_slots: list[int]
    ) -> tuple[int, int]:
        if not self._spec.needs_two_cache_families:
            return self._backend.find_best_slot_for_prompt(token_ids, free_slots)
        per_slot = getattr(self._backend, "prefix_hit_for_slot", None)
        if per_slot is None:
            raise NotImplementedError(
                _MISSING_PER_SLOT.format(cls=type(self._backend).__name__)
            )
        if not free_slots:
            raise ValueError("find_best_slot_for_prompt requires at least one free slot")
        best_slot = free_slots[0]
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in free_slots:
            hit = self._aligned(per_slot(token_ids, slot))
            # Ordered by effective depth first, KV depth only as a
            # tie-break: among slots that can be resumed equally deep, the
            # one holding more matching KV is the better bet for the *next*
            # checkpoint, and preferring it costs nothing.
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best = hit
                best_slot = slot
        return best_slot, best.effective

    def _aligned(self, hit: PrefixHit) -> PrefixHit:
        block = self._block_size
        if block <= 1:
            return hit
        state = (hit.state_hit // block) * block
        if state == hit.state_hit:
            return hit
        return PrefixHit(kv_hit=hit.kv_hit, state_hit=state)
