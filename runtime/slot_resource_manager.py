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
* ``needs_two_cache_families`` is ``True``: NOT implemented here, on
  purpose. Track B lands the real ``(kv_hit, state_hit)`` merge logic and
  the point at which a ``runtime.recurrent_state_pool.RecurrentStatePool``
  actually gets instantiated (§7 row 7-h, §5 point 2: "是否要改变现有方法的
  返回类型...是本设计里我判断不出唯一正确答案的一个具体分叉"). Raising
  :class:`NotImplementedError` with a pointer to where the real logic lands
  is the honest thing to do here -- pretending to support a merge whose
  shape nothing has decided yet would be worse than refusing outright.

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

_NOT_IMPLEMENTED = (
    "SlotResourceManager does not yet merge a second (recurrent-state) resource -- "
    "Track B lands this (docs/a3-cache-coordinator-design.md §7 row 7-h). "
    "needs_two_cache_families is True for this checkpoint, which no backend "
    "this runtime ships today declares."
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

    def __init__(self, backend: ModelBackend, architecture_spec: ArchitectureSpec) -> None:
        self._backend = backend
        self._spec = architecture_spec

    @property
    def needs_two_cache_families(self) -> bool:
        return self._spec.needs_two_cache_families

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        """INV-A3-2 (``state_hit <= kv_hit``) is enforced by
        :class:`PrefixHit` itself (``runtime/backends/protocol.py``'s
        ``__post_init__``) regardless of which branch below answers -- this
        method does not re-derive it, only forwards or refuses."""
        if not self._spec.needs_two_cache_families:
            return self._backend.reconcile_prefix_hit(token_ids)
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def find_best_slot_for_prompt(
        self, token_ids: list[int], free_slots: list[int]
    ) -> tuple[int, int]:
        if not self._spec.needs_two_cache_families:
            return self._backend.find_best_slot_for_prompt(token_ids, free_slots)
        raise NotImplementedError(_NOT_IMPLEMENTED)
