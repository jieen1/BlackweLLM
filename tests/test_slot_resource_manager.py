"""A3 step 7-d (docs/a3-cache-coordinator-design.md §7): tests for the
coordinator skeleton, ``runtime.slot_resource_manager.SlotResourceManager``.

Torch-free by construction (the module only imports ``runtime.architecture``
and ``runtime.backends.protocol``, both torch-free), so the bulk of this
file runs against fakes with no ``pytest.importorskip``. One class
(``TestRealLagunaShadowConsistency``) additionally proves the same claim
against the real ``ArchitectureSpec``/``LagunaBackend`` pairing and is
guarded accordingly.

Four things are covered, matching §7's gate for this step:

* Shadow consistency -- the coordinator's forwarded result equals calling
  the backend directly, byte-for-byte (§5's zero-behavior-change claim,
  made concrete).
* The explicit refusal for the not-yet-implemented ``needs_two_cache_
  families=True`` branch (Track B's job, not this step's) -- a deliberately
  constructed case that must actually raise, not a hand-wave.
* A "signal probe" for INV-A3-1 (docs/a3-cache-coordinator-design.md §2):
  two backends, each tagged with a distinct marker, must never see each
  other's answers through the coordinator -- the failure mode INV-A3-1
  describes ("no crash, output changes because of another request's
  write") would show up here as a marker mismatch, not an exception.
* "Admission under pressure": many interleaved calls across multiple
  coordinators stay correct, ruling out an accidental cache/memoization bug
  that only shows up under volume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.architecture import parse_architecture
from runtime.backends.protocol import PrefixHit
from runtime.slot_resource_manager import SlotResourceManager

_HUB = Path.home() / ".cache" / "huggingface" / "hub"
_LAGUNA_REPO = "models--poolside--Laguna-S-2.1-NVFP4"


def _load_laguna_config() -> dict:
    matches = sorted((_HUB / _LAGUNA_REPO).glob("snapshots/*/config.json"))
    if not matches:
        pytest.skip(f"{_LAGUNA_REPO} not present in the local HF cache")
    return json.loads(matches[0].read_text())


def _spec(*, needs_two_cache_families: bool):
    """A minimal, hand-built ArchitectureSpec -- same helper shape as
    tests/test_architecture_spec.py's minimal_config, reimplemented locally
    to keep this file self-contained."""
    if needs_two_cache_families:
        layer_types = ["full_attention", "linear_attention"]
    else:
        layer_types = ["full_attention", "sliding_attention"]
    config = {
        "architectures": ["TestForCausalLM"],
        "model_type": "test",
        "num_hidden_layers": len(layer_types),
        "layer_types": layer_types,
        "hidden_size": 8,
        "vocab_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    spec = parse_architecture(config)
    assert spec.needs_two_cache_families is needs_two_cache_families  # sanity on the helper itself
    return spec


class _FakeBackend:
    """Deterministic, explicitly-tagged fake: every (marker, prompt) pair
    maps to its own distinct PrefixHit / slot choice, so any cross-talk
    between two backends or two call orders shows up as a value mismatch,
    not as an exception."""

    def __init__(self, marker: str):
        self.marker = marker
        self.reconcile_calls: list[list[int]] = []
        self.find_best_slot_calls: list[tuple[list[int], list[int]]] = []

    def _tag(self, token_ids: list[int]) -> int:
        # Deterministic, marker-specific value -- NOT hash()-based (hash()
        # randomization would make this flaky across interpreter runs);
        # just enough arithmetic that backend A and backend B disagree on
        # every prompt they are both asked about. sum(ord(c) for c in
        # marker), not len(marker): single-character markers ("A" vs "B")
        # have equal length but distinct ordinal sums.
        marker_value = sum(ord(c) for c in self.marker)
        return (sum(token_ids) + marker_value * 1000) % 512

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        self.reconcile_calls.append(list(token_ids))
        depth = (self._tag(token_ids) // 64) * 64
        return PrefixHit(kv_hit=depth, state_hit=depth)

    def find_best_slot_for_prompt(
        self, token_ids: list[int], free_slots: list[int]
    ) -> tuple[int, int]:
        self.find_best_slot_calls.append((list(token_ids), list(free_slots)))
        chosen = free_slots[self._tag(token_ids) % len(free_slots)]
        return (chosen, 64)


class TestForwardingWhenNoSecondCacheFamily:
    """§5 point 3: pure forward, no second allocator instantiated."""

    def test_needs_two_cache_families_property_reflects_spec(self) -> None:
        mgr_false = SlotResourceManager(_FakeBackend("x"), _spec(needs_two_cache_families=False))
        mgr_true = SlotResourceManager(_FakeBackend("x"), _spec(needs_two_cache_families=True))
        assert mgr_false.needs_two_cache_families is False
        assert mgr_true.needs_two_cache_families is True

    def test_reconcile_prefix_hit_forwards_byte_for_byte_to_backend(self) -> None:
        backend = _FakeBackend("laguna")
        mgr = SlotResourceManager(backend, _spec(needs_two_cache_families=False))
        prompt = [1, 2, 3, 4, 5]

        direct = backend.reconcile_prefix_hit(prompt)
        via_coordinator = mgr.reconcile_prefix_hit(prompt)

        assert via_coordinator == direct
        assert via_coordinator.kv_hit == direct.kv_hit
        assert via_coordinator.state_hit == direct.state_hit

    def test_find_best_slot_for_prompt_forwards_byte_for_byte_to_backend(self) -> None:
        backend = _FakeBackend("laguna")
        mgr = SlotResourceManager(backend, _spec(needs_two_cache_families=False))
        prompt, free_slots = [7, 8, 9], [0, 1, 2]

        direct = backend.find_best_slot_for_prompt(prompt, free_slots)
        via_coordinator = mgr.find_best_slot_for_prompt(prompt, free_slots)

        assert via_coordinator == direct


class TestSecondCacheFamilyNotImplemented:
    """Deliberately constructed violation, not just the positive path:
    needs_two_cache_families=True must actually raise, since Track B has
    not landed the merge logic (§7 row 7-h)."""

    def test_reconcile_prefix_hit_raises(self) -> None:
        mgr = SlotResourceManager(_FakeBackend("x"), _spec(needs_two_cache_families=True))
        with pytest.raises(NotImplementedError, match="Track B"):
            mgr.reconcile_prefix_hit([1, 2, 3])

    def test_find_best_slot_for_prompt_raises(self) -> None:
        mgr = SlotResourceManager(_FakeBackend("x"), _spec(needs_two_cache_families=True))
        with pytest.raises(NotImplementedError, match="Track B"):
            mgr.find_best_slot_for_prompt([1, 2, 3], [0, 1])

    def test_backend_is_never_called_on_the_not_implemented_path(self) -> None:
        # The refusal must happen BEFORE touching the backend at all -- a
        # backend that does not implement the True-branch shape yet (none
        # does) must not be invoked speculatively.
        backend = _FakeBackend("x")
        mgr = SlotResourceManager(backend, _spec(needs_two_cache_families=True))
        with pytest.raises(NotImplementedError):
            mgr.reconcile_prefix_hit([1, 2, 3])
        assert backend.reconcile_calls == []


class TestRealLagunaShadowConsistency:
    """The same forwarding claim, against the real ArchitectureSpec/
    LagunaBackend pairing rather than fakes -- the evidentiary-weight half,
    mirroring tests/test_architecture_spec.py's own inline-dict-vs-real-
    checkpoint split (skips when the checkpoint is not on this machine
    rather than being deleted)."""

    def test_real_laguna_checkpoint_needs_no_second_cache_family(self) -> None:
        spec = parse_architecture(_load_laguna_config())
        assert spec.needs_two_cache_families is False

    def test_coordinator_forwards_byte_for_byte_to_real_laguna_backend(self) -> None:
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend

        spec = parse_architecture(_load_laguna_config())
        backend = LagunaBackend.__new__(LagunaBackend)
        backend._prefix_cache_tokens = [[10, 20, 30, 40, 50] * 20, None]  # 100 tokens
        backend._prefix_cache_kv_len = [100, 0]
        backend._pending_prefix_hits = {}
        backend.block_size = 64
        mgr = SlotResourceManager(backend, spec)
        prompt = [10, 20, 30, 40, 50] * 20 + [99, 98, 97]

        # Two bare backends with identical warm-cache setup: one answers
        # directly, the other only through the coordinator. Same setup, so
        # "equal" here is the shadow-consistency claim, not a tautology.
        direct_backend = LagunaBackend.__new__(LagunaBackend)
        direct_backend._prefix_cache_tokens = [[10, 20, 30, 40, 50] * 20, None]
        direct_backend._prefix_cache_kv_len = [100, 0]
        direct_backend._pending_prefix_hits = {}
        direct_backend.block_size = 64

        direct = direct_backend.reconcile_prefix_hit(prompt)
        via_coordinator = mgr.reconcile_prefix_hit(prompt)
        assert via_coordinator == direct == PrefixHit(kv_hit=64, state_hit=64)


class TestStatelessnessSignalProbe:
    """INV-A3-1 (docs/a3-cache-coordinator-design.md §2): "no crash -- a
    request's output changes because of another request's write." Two
    backends, two coordinators, one shared prompt: if the coordinator ever
    leaked one backend's answer into the other's, this would show a wrong
    marker's value, not an exception."""

    def test_two_coordinators_never_cross_talk_on_a_shared_prompt(self) -> None:
        backend_a = _FakeBackend("A")
        backend_b = _FakeBackend("B")
        mgr_a = SlotResourceManager(backend_a, _spec(needs_two_cache_families=False))
        mgr_b = SlotResourceManager(backend_b, _spec(needs_two_cache_families=False))
        shared_prompt = [42, 43, 44, 45]

        # Interleave on purpose: A, B, A, B -- a shared-state bug (e.g. a
        # module-level cache keyed only on the prompt) would show up as
        # mgr_a and mgr_b agreeing when their backends' tags say they must
        # not.
        r1 = mgr_a.reconcile_prefix_hit(shared_prompt)
        r2 = mgr_b.reconcile_prefix_hit(shared_prompt)
        r3 = mgr_a.reconcile_prefix_hit(shared_prompt)
        r4 = mgr_b.reconcile_prefix_hit(shared_prompt)

        assert r1 == r3 == backend_a.reconcile_prefix_hit(shared_prompt)
        assert r2 == r4 == backend_b.reconcile_prefix_hit(shared_prompt)
        assert backend_a._tag(shared_prompt) != backend_b._tag(shared_prompt), (
            "test setup bug: markers must disagree for this probe to mean anything"
        )
        assert r1 != r2


class TestAdmissionUnderPressure:
    """Many interleaved calls across multiple coordinators stay correct --
    rules out a cache/memoization bug that only shows up under volume
    (docs/a3-cache-coordinator-design.md §2 INV-A3-4's methodology note:
    "故意在其它槎活跃时强制驱逐,而不是孤立测试驱逐", applied here to call
    volume/interleaving rather than eviction since this skeleton owns no
    resources to evict yet)."""

    def test_many_interleaved_calls_stay_correct(self) -> None:
        backends = [_FakeBackend(f"backend-{i}") for i in range(4)]
        managers = [
            SlotResourceManager(b, _spec(needs_two_cache_families=False)) for b in backends
        ]

        for round_num in range(200):
            idx = round_num % len(managers)
            prompt = [round_num, round_num + 1, round_num + 2]
            expected = backends[idx].reconcile_prefix_hit(prompt)
            got = managers[idx].reconcile_prefix_hit(prompt)
            assert got == expected, f"round {round_num}, backend {idx}: mismatch"
