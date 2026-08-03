"""Step 1 of the Track A migration: the shadow contract must already hold.

Two layers, deliberately split so the CPU-only job still gets coverage:

* ``runtime.backends.protocol`` is torch-free, so the contract's own shape is
  tested unconditionally;
* the conformance check needs the real ``LagunaBackend``, which imports torch
  eagerly, so it sits behind ``importorskip``.

Shadow mode means these tests must be green **before** any call site changes.
If one goes red, the protocol and the shipping backend have diverged, and
that is the whole signal this file exists to give. See
``docs/architecture.md`` §3.5.5.
"""

from __future__ import annotations

import pytest

from runtime.backends.protocol import (
    CAPABILITY_MEMBERS,
    REQUIRED_MEMBERS,
    BackendCapabilities,
    ModelBackend,
    PrefixHit,
    check_conformance,
)


class TestContractShape:
    """Torch-free: the contract's own invariants."""

    def test_covers_exactly_the_members_the_scheduler_uses(self) -> None:
        # 12 members are reached through ``self.runner`` in server/engine.py,
        # plus ``capabilities`` (consulted instead of ``hasattr``/``try-except``)
        # and ``snapshot`` (the observability contract). Locking the count here
        # makes silent contract growth visible in review: a 15th member means
        # someone widened what every future backend must implement.
        #
        # B3 (2026-08-03): was 15 -- ``enable_dflash`` (a Laguna-specific
        # LOAD-TIME wiring method, never reached through ``self.runner`` at
        # all) dropped out of ``speculative_decode``'s governed set; see
        # ``CAPABILITY_MEMBERS``'s own updated docstring in
        # ``runtime/backends/protocol.py``.
        governed = [m for members in CAPABILITY_MEMBERS.values() for m in members]
        assert len(set(REQUIRED_MEMBERS) | set(governed)) == 14

    def test_no_member_is_governed_by_two_capabilities(self) -> None:
        seen: set[str] = set()
        for members in CAPABILITY_MEMBERS.values():
            for member in members:
                assert member not in seen, f"{member} is governed twice"
                seen.add(member)

    def test_required_and_governed_sets_are_disjoint(self) -> None:
        governed = {m for members in CAPABILITY_MEMBERS.values() for m in members}
        assert not (set(REQUIRED_MEMBERS) & governed)

    def test_every_declared_member_exists_on_the_protocol(self) -> None:
        governed = [m for members in CAPABILITY_MEMBERS.values() for m in members]
        for name in (*REQUIRED_MEMBERS, *governed):
            assert hasattr(ModelBackend, name), f"{name} is declared but not defined"

    def test_capabilities_is_frozen(self) -> None:
        # It gets serialized into run records; a mutable one could be edited
        # after the fact and would stop describing the run it came from.
        caps = BackendCapabilities(True, True, True, True, False)
        with pytest.raises(Exception):
            caps.speculative_decode = False  # type: ignore[misc]


class TestPrefixHit:
    """A3 step 7-a (docs/a3-cache-coordinator-design.md §3, decision 2):
    ``reconcile_prefix_hit`` was changed in place to return this instead of
    a bare ``int``. These tests cover the type's own guarantees -- the
    conformance checker (below) cannot, since it deliberately excludes
    return-type annotations from its comparison (see this module's
    ``_comparable`` docstring)."""

    def test_kv_hit_equals_state_hit_is_allowed(self) -> None:
        hit = PrefixHit(kv_hit=64, state_hit=64)
        assert hit.kv_hit == 64
        assert hit.state_hit == 64

    def test_state_hit_below_kv_hit_is_allowed(self) -> None:
        hit = PrefixHit(kv_hit=900, state_hit=400)
        assert hit.state_hit < hit.kv_hit

    def test_effective_is_always_state_hit_not_kv_hit(self) -> None:
        # §3's worked example: kv_hit=900, state_hit=400 -> effective=400,
        # never 900 (the [400, 900) region has KV but no recurrent-state
        # checkpoint to resume it from) and never 0 (the [0, 400) region's
        # hit is real and safe to use).
        hit = PrefixHit(kv_hit=900, state_hit=400)
        assert hit.effective == 400

    def test_state_hit_above_kv_hit_raises_inv_a3_2(self) -> None:
        # Deliberately constructed violation (INV-A3-2: state_hit <= kv_hit
        # must always hold) -- this must actually raise, not just document
        # the rule in a docstring.
        with pytest.raises(ValueError, match="INV-A3-2"):
            PrefixHit(kv_hit=100, state_hit=101)

    def test_negative_hit_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            PrefixHit(kv_hit=-1, state_hit=-1)

    def test_is_frozen(self) -> None:
        hit = PrefixHit(kv_hit=64, state_hit=64)
        with pytest.raises(Exception):
            hit.kv_hit = 128  # type: ignore[misc]


class TestConformanceChecker:
    """The checker must actually catch drift, or the gate is decorative."""

    def _caps(self, **overrides: bool) -> BackendCapabilities:
        base = dict(
            speculative_decode=True,
            prefix_cache=True,
            cuda_graph=True,
            chunked_prefill=True,
            warm_continue=True,
        )
        base.update(overrides)
        return BackendCapabilities(**base)  # type: ignore[arg-type]

    def test_empty_class_fails_on_required_members(self) -> None:
        class Empty:
            pass

        problems = check_conformance(Empty, self._caps())
        assert any("reset_slot" in p for p in problems)
        assert any("capabilities" in p for p in problems)

    def test_missing_optional_member_is_fine_when_capability_is_false(self) -> None:
        class NoWarmContinue:
            pass

        with_cap = check_conformance(NoWarmContinue, self._caps())
        without_cap = check_conformance(NoWarmContinue, self._caps(warm_continue=False))
        assert any("mtp_prefill_warm_continue" in p for p in with_cap)
        assert not any("mtp_prefill_warm_continue" in p for p in without_cap)

    def test_wrong_signature_is_caught(self) -> None:
        class WrongArity:
            def reset_slot(self, slot: int, extra: int) -> None: ...

        problems = check_conformance(WrongArity, self._caps())
        assert any("reset_slot" in p and "signature differs" in p for p in problems)

    def test_method_where_protocol_wants_a_property_is_caught(self) -> None:
        # The exact drift that would have shipped: has_speculative_decode is a
        # property, and engine.py passes it by value. A backend defining it as
        # a method would hand the scheduler a bound method -- always truthy --
        # and silently route every slot down the speculative path.
        class MethodNotProperty:
            def has_speculative_decode(self) -> bool: ...

        problems = check_conformance(MethodNotProperty, self._caps())
        assert any("has_speculative_decode" in p and "property" in p for p in problems)


class TestLagunaConformance:
    """The shipping backend against the contract. Needs torch, not a GPU."""

    @staticmethod
    def _backend_cls() -> type:
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend

        return LagunaBackend

    def test_laguna_conforms_to_the_protocol(self) -> None:
        cls = self._backend_cls()
        problems = check_conformance(cls, cls.capabilities.fget(None))  # type: ignore[attr-defined]
        assert problems == [], "LagunaBackend no longer matches the contract:\n" + "\n".join(
            problems
        )

    def test_capabilities_needs_no_instance(self) -> None:
        # capabilities must be readable without constructing a backend, which
        # would allocate GPU memory. The registry has to pick a backend from a
        # checkpoint config before anything is loaded.
        cls = self._backend_cls()
        caps = cls.capabilities.fget(None)  # type: ignore[attr-defined]
        assert isinstance(caps, BackendCapabilities)

    def test_warm_continue_is_advertised_as_unsupported(self) -> None:
        # N8. server/engine.py calls mtp_prefill_warm_continue anyway, inside
        # `except Exception`, so --session-affinity silently degrades to cold
        # prefill on every turn. This asserts the honest half of that: the
        # backend says it cannot do it. When the flag's fate is decided --
        # implement, delete, or reject at startup -- this test changes with it.
        cls = self._backend_cls()
        caps = cls.capabilities.fget(None)  # type: ignore[attr-defined]
        assert caps.warm_continue is False
        assert not hasattr(cls, "mtp_prefill_warm_continue")

    def test_prefix_cache_capability_agrees_with_the_hasattr_probe_it_replaced(
        self,
    ) -> None:
        """A3 step 7-b (docs/a3-cache-coordinator-design.md §1.1): server/
        engine.py's admission block used to gate cache-aware slot assignment
        on ``hasattr(self.runner, "find_best_slot_for_prompt")`` and now
        reads ``capabilities.prefix_cache`` instead. This is the
        shadow-consistency claim for the one real shipping backend: the new
        mechanism must agree with what the old one would have concluded.
        ``tests/test_engine_prefix_cache_admission.py`` covers the same
        claim behaviorally, with fakes standing in for both possible
        answers."""
        cls = self._backend_cls()
        caps = cls.capabilities.fget(None)  # type: ignore[attr-defined]
        assert caps.prefix_cache is hasattr(cls, "find_best_slot_for_prompt")
        assert caps.prefix_cache is hasattr(cls, "reconcile_prefix_hit")
