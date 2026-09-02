"""A3 step 7-b (docs/a3-cache-coordinator-design.md §1.1, §7): server/
engine.py's admission block used to gate cache-aware slot assignment on
``hasattr(self.runner, "find_best_slot_for_prompt")`` -- exactly the
``try/except AttributeError``-shaped anti-pattern ``runtime/backends/
protocol.py``'s own module docstring names as the one the scheduler must
never use. It now reads ``self.runner.capabilities.prefix_cache`` instead.

These are the shadow-consistency tests §7's gate calls for: for a backend
that HAS the capability, admission must still call
``find_best_slot_for_prompt`` and honor its slot choice, exactly as the old
``hasattr`` probe (True for it) would have allowed; for a backend that does
NOT have it, admission must fall back to popping the first free slot,
exactly as the old probe (False for it) did. Both fakes below are the direct
counterparts of ``test_engine_session_affinity.py``'s ``_FakeWarmRunner``
(which is the ``prefix_cache=False`` case, exercised there incidentally by
the warm-continue-fallback tests; this file exercises both cases directly
and is the one place that names the mechanism itself as the thing under
test).

``TestLagunaConformance`` in ``tests/test_backend_protocol.py`` covers the
other half of the shadow-consistency claim: for the one real shipping
backend (``LagunaBackend``), the new capability bit and the old ``hasattr``
probe agree.

``TestCoordinatorWiring`` (A3 step 7-g, docs/a3-cache-coordinator-design.md
§7 row 7-g) covers the piece this docstring's own claims above do NOT: that
admission actually calls through ``ServerEngine.slot_resources`` (a
``SlotResourceManager``) rather than ``self.runner`` directly. Every test
above this point in the file was written for 7-b, when the call site really
was ``self.runner.find_best_slot_for_prompt``/``.reconcile_prefix_hit``, and
still passes unmodified after 7-g's wiring -- strong evidence the observable
outcome did not change, but not, on its own, proof that the coordinator is
actually in the call path (a `self.slot_resources` property that silently
fell back to `self.runner` under the hood would pass those same assertions).
``TestCoordinatorWiring`` spies on ``SlotResourceManager`` itself to close
that gap.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from runtime.backends.protocol import BackendCapabilities, PrefixHit
from runtime.slot_resource_manager import SlotResourceManager
from server.engine import GenerationRequest, ServerEngine


class _FakeIdTok:
    def __init__(self):
        self._mapping: dict[int, str] = {}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._mapping.get(i, "") for i in ids)


class _FakeCacheAwareRunner:
    """prefix_cache=True and a real find_best_slot_for_prompt implementation
    -- the counterpart _FakeWarmRunner (prefix_cache=False) does not have."""

    def __init__(self, best_slot_by_prompt: dict[tuple[int, ...], int]):
        self.capabilities = BackendCapabilities(
            speculative_decode=False,
            prefix_cache=True,
            cuda_graph=False,
            chunked_prefill=True,
            warm_continue=False,
        )
        self.has_speculative_decode = False
        self._best_slot_by_prompt = best_slot_by_prompt
        self.find_best_slot_calls: list[tuple[list[int], list[int]]] = []
        self.reset_calls: list[int] = []

    def reset_slot(self, slot: int) -> None:
        self.reset_calls.append(slot)

    def slot_state(self, slot: int):
        return SimpleNamespace(kv_len=0, is_fresh=True)

    def find_best_slot_for_prompt(
        self,
        token_ids: list[int],
        free_slots: list[int],
        *,
        prefix_cache_key: object | None = None,
    ) -> tuple[int, int]:
        del prefix_cache_key
        self.find_best_slot_calls.append((list(token_ids), list(free_slots)))
        chosen = self._best_slot_by_prompt[tuple(token_ids)]
        assert chosen in free_slots
        return (chosen, 64)  # hit depth here is discarded by engine.py (§1.1)

    def reconcile_prefix_hit(
        self,
        token_ids: list[int],
        *,
        prefix_cache_key: object | None = None,
    ) -> PrefixHit:
        del token_ids, prefix_cache_key
        return PrefixHit(kv_hit=0, state_hit=0)

    def prefill_chunked_begin(
        self,
        slots,
        prompts_per_slot,
        chunk_size: int = 512,
        *,
        params_per_slot=None,
        vision_inputs_per_slot=None,
    ):
        # `params_per_slot` (E2-b) is accepted and ignored: this fake exercises
        # slot *assignment*, not sampling. Omitting it made every admission raise
        # TypeError inside ServerEngine's admission try/except, which fails the
        # futures and leaves the engine with nothing to do -- so `_step_sync`
        # reached its idle blocking read and the test hung instead of failing.
        del vision_inputs_per_slot
        result = {s: {"anchor": 900 + s, "draft_tokens": []} for s in slots}
        return SimpleNamespace(done=True, result=result)

    def decode_batch_sampled(
        self, slot_ids, token_ids, kv_lengths, params_list, *, return_logprobs=False, top_logprobs=0
    ):
        return [901 for _ in slot_ids]

    def mtp_verify_and_commit_batch(self, *args, **kwargs):
        raise NotImplementedError("has_speculative_decode=False; must not be called")


class _FakeNoCacheRunner:
    """prefix_cache=False and NO find_best_slot_for_prompt at all -- must
    never be called; admission must fall back to the plain free-slot pop."""

    def __init__(self):
        self.capabilities = BackendCapabilities(
            speculative_decode=False,
            prefix_cache=False,
            cuda_graph=False,
            chunked_prefill=True,
            warm_continue=False,
        )
        self.has_speculative_decode = False
        self.reset_calls: list[int] = []

    def reset_slot(self, slot: int) -> None:
        self.reset_calls.append(slot)

    def slot_state(self, slot: int):
        return SimpleNamespace(kv_len=0, is_fresh=True)

    def reconcile_prefix_hit(
        self,
        token_ids: list[int],
        *,
        prefix_cache_key: object | None = None,
    ) -> PrefixHit:
        del token_ids, prefix_cache_key
        return PrefixHit(kv_hit=0, state_hit=0)

    def prefill_chunked_begin(
        self,
        slots,
        prompts_per_slot,
        chunk_size: int = 512,
        *,
        params_per_slot=None,
        vision_inputs_per_slot=None,
    ):
        # `params_per_slot` (E2-b) is accepted and ignored: this fake exercises
        # slot *assignment*, not sampling. Omitting it made every admission raise
        # TypeError inside ServerEngine's admission try/except, which fails the
        # futures and leaves the engine with nothing to do -- so `_step_sync`
        # reached its idle blocking read and the test hung instead of failing.
        del vision_inputs_per_slot
        result = {s: {"anchor": 700 + s, "draft_tokens": []} for s in slots}
        return SimpleNamespace(done=True, result=result)

    def decode_batch_sampled(
        self, slot_ids, token_ids, kv_lengths, params_list, *, return_logprobs=False, top_logprobs=0
    ):
        return [701 for _ in slot_ids]

    def mtp_verify_and_commit_batch(self, *args, **kwargs):
        raise NotImplementedError("has_speculative_decode=False; must not be called")


class _FakeInBatchCacheRunner(_FakeCacheAwareRunner):
    """Cache-aware fake that opts into exact-prefix wave deduplication."""

    def __init__(self):
        super().__init__({})
        self.capabilities = replace(self.capabilities, prefix_cache_dedup=True)


class _FakeReservedRunner(_FakeNoCacheRunner):
    def __init__(self, reservable_slots: set[int]):
        super().__init__()
        self.capabilities = BackendCapabilities(
            speculative_decode=False,
            prefix_cache=False,
            cuda_graph=False,
            chunked_prefill=True,
            warm_continue=False,
            kv_reservation=True,
        )
        self.reservable_slots = set(reservable_slots)
        self.reserve_calls: list[tuple[int, int]] = []
        self.release_calls: list[int] = []

    def reserve_kv_capacity(self, slot: int, total_tokens: int) -> bool:
        self.reserve_calls.append((slot, total_tokens))
        return slot in self.reservable_slots

    def release_kv_reservation(self, slot: int) -> None:
        self.release_calls.append(slot)


def _bare_admission_engine(runner, capacity: int = 2) -> ServerEngine:
    engine = ServerEngine(
        backend="laguna", capacity=capacity, num_slots=capacity, enable_cudagraph=False
    )
    engine.tok = _FakeIdTok()
    engine.eos_token_ids = frozenset()
    engine.runner = runner
    engine._asyncio_loop = asyncio.new_event_loop()
    return engine


def _req(engine: ServerEngine, prompt_ids: list[int], request_id: str) -> GenerationRequest:
    from runtime.sampling import SamplingParams

    return GenerationRequest(
        request_id=request_id,
        prompt_ids=prompt_ids,
        max_tokens=10,
        future=engine._asyncio_loop.create_future(),
        sampling_params=SamplingParams(temperature=0.0),
    )


def _vision_req(
    engine: ServerEngine,
    prompt_ids: list[int],
    request_id: str,
    *,
    vision_inputs: object,
) -> GenerationRequest:
    req = _req(engine, prompt_ids, request_id)
    req.vision_inputs = vision_inputs
    return req


class TestCapabilityGatedSlotAssignment:
    def test_capability_true_uses_find_best_slot_for_prompt_and_honors_choice(self) -> None:
        # Two prompts, two free slots (0, 1); the fake deliberately picks the
        # OPPOSITE slot from pop-order for each, so a passing assertion can
        # only mean find_best_slot_for_prompt's return value actually drove
        # the assignment (not a coincidence of iteration order).
        prompt_a, prompt_b = (11, 12, 13), (21, 22, 23)
        runner = _FakeCacheAwareRunner(best_slot_by_prompt={prompt_a: 1, prompt_b: 0})
        engine = _bare_admission_engine(runner)
        engine.waiting = [
            _req(engine, list(prompt_a), "req-a"),
            _req(engine, list(prompt_b), "req-b"),
        ]

        engine._step_sync()

        assert len(runner.find_best_slot_calls) == 2
        # req-a went to slot 1 (anchor 900+1), req-b went to slot 0 (900+0).
        assert engine.active[1]["anchor"] == 901
        assert engine.active[0]["anchor"] == 900
        assert engine.free_slots == []

    def test_capability_false_never_calls_find_best_slot_and_pops_first_free(self) -> None:
        runner = _FakeNoCacheRunner()
        assert not hasattr(runner, "find_best_slot_for_prompt")
        engine = _bare_admission_engine(runner)
        engine.waiting = [_req(engine, [1, 2, 3], "req-c")]

        engine._step_sync()  # must not raise (no find_best_slot_for_prompt exists)

        # Fell back to the first free slot (0), exactly as
        # hasattr(runner, "find_best_slot_for_prompt") == False did before.
        assert engine.active[0]["anchor"] == 700
        assert 1 in engine.free_slots

    def test_vision_requests_forward_prefix_cache_keys_through_coordinator(self) -> None:
        class _FakeVisionCacheRunner(_FakeCacheAwareRunner):
            def __init__(self):
                super().__init__({})
                self.keyed_find_calls: list[tuple[list[int], list[int], object]] = []
                self.keyed_reconcile_calls: list[tuple[list[int], object]] = []

            def prefix_cache_key_for_vision_inputs(self, vision_inputs):
                return tuple(vision_inputs.image_cache_keys)

            def find_best_slot_for_prompt_with_key(self, token_ids, free_slots, prefix_cache_key):
                self.keyed_find_calls.append((list(token_ids), list(free_slots), prefix_cache_key))
                return free_slots[0], 64

            def reconcile_prefix_hit_with_key(self, token_ids, prefix_cache_key):
                self.keyed_reconcile_calls.append((list(token_ids), prefix_cache_key))
                return PrefixHit(kv_hit=0, state_hit=0)

            def find_best_slot_for_prompt(
                self, token_ids, free_slots, *, prefix_cache_key=None
            ):
                del token_ids, free_slots, prefix_cache_key
                raise AssertionError("plain slot matcher must not run for keyed vision requests")

            def reconcile_prefix_hit(self, token_ids, *, prefix_cache_key=None):
                del token_ids, prefix_cache_key
                raise AssertionError(
                    "plain prefix reconciliation must not run for keyed vision requests"
                )

        runner = _FakeVisionCacheRunner()
        engine = _bare_admission_engine(runner, capacity=1)
        engine.waiting = [
            _vision_req(
                engine,
                [1, 2, 3],
                "req-vision",
                vision_inputs=SimpleNamespace(image_cache_keys=("img-a",)),
            )
        ]

        engine._step_sync()

        assert runner.keyed_find_calls == [([1, 2, 3], [0], ("img-a",))]
        assert runner.keyed_reconcile_calls == [([1, 2, 3], ("img-a",))]


class TestInBatchPrefixDedup:
    def test_exact_aligned_duplicates_wait_for_first_prefix_publish(self) -> None:
        runner = _FakeInBatchCacheRunner()
        engine = _bare_admission_engine(runner, capacity=2)
        engine.block_size = 4
        first = _req(engine, [1, 2, 3, 4], "req-a1")
        duplicate = _req(engine, [1, 2, 3, 4], "req-a2")
        distinct = _req(engine, [5, 6, 7, 8], "req-b")
        engine.waiting = [first, duplicate, distinct]

        selected = engine._select_admission_requests(2)

        assert selected == [first, distinct]
        assert engine.waiting == [duplicate]
        assert engine.stats["prefix_cache_dedup_deferrals"] == 1

        engine._release_prefix_dedup_keys(selected, published=True)
        assert engine._select_admission_requests(2) == [duplicate]

    def test_unaligned_duplicates_stay_batchable(self) -> None:
        runner = _FakeInBatchCacheRunner()
        engine = _bare_admission_engine(runner, capacity=2)
        engine.block_size = 4
        first = _req(engine, [1, 2, 3], "req-a1")
        duplicate = _req(engine, [1, 2, 3], "req-a2")
        engine.waiting = [first, duplicate]

        assert engine._select_admission_requests(2) == [first, duplicate]
        assert engine.stats["prefix_cache_dedup_deferrals"] == 0


class TestKVReservationAdmission:
    def test_capacity_miss_stays_waiting_before_prefill(self) -> None:
        runner = _FakeReservedRunner(reservable_slots={0})
        engine = _bare_admission_engine(runner)
        req_a = _req(engine, [1, 2, 3], "req-reserved")
        req_b = _req(engine, [4, 5, 6], "req-wait")
        engine.waiting = [req_a, req_b]

        engine._step_sync()

        assert runner.reserve_calls == [(0, 13), (1, 13)]
        assert 0 in engine.active
        assert engine.waiting == [req_b]
        assert not req_b.future.done()
        assert engine.free_slots == [1]

    def test_finish_releases_unmaterialized_tail_before_slot_retention(self) -> None:
        runner = _FakeReservedRunner(reservable_slots={0})
        engine = _bare_admission_engine(runner, capacity=1)
        req = _req(engine, [1, 2, 3], "req-finish")

        engine._finish_request(0, req, [9], "length")

        assert runner.release_calls == [0]


class TestCoordinatorWiring:
    """A3 step 7-g: prove the coordinator is actually in the call path, not
    just that the observable outcome is unchanged (the other tests in this
    file, all inherited from 7-b, already prove that half)."""

    def test_slot_resources_is_a_slot_resource_manager_bound_to_runner(self) -> None:
        runner = _FakeNoCacheRunner()
        engine = _bare_admission_engine(runner)

        resources = engine.slot_resources

        assert isinstance(resources, SlotResourceManager)
        assert resources._backend is runner
        # Default architecture_spec (no real checkpoint passed to this bare
        # test engine) still forces the pure-forward branch -- see
        # server/engine.py's _DEFAULT_ARCHITECTURE_SPEC comment.
        assert resources.needs_two_cache_families is False

    def test_admission_calls_through_slot_resource_manager_not_runner_directly(
        self,
    ) -> None:
        """Spies on SlotResourceManager's own methods (not the fake runner's)
        -- a property that quietly bypassed the coordinator and called
        ``self.runner`` directly would still pass every test above this
        class, but would leave these spies uncalled."""
        prompt_a, prompt_b = (11, 12, 13), (21, 22, 23)
        runner = _FakeCacheAwareRunner(best_slot_by_prompt={prompt_a: 1, prompt_b: 0})
        engine = _bare_admission_engine(runner)
        engine.waiting = [
            _req(engine, list(prompt_a), "req-a"),
            _req(engine, list(prompt_b), "req-b"),
        ]

        with (
            patch.object(
                SlotResourceManager,
                "find_best_slot_for_prompt",
                autospec=True,
                side_effect=SlotResourceManager.find_best_slot_for_prompt,
            ) as spy_find_slot,
            patch.object(
                SlotResourceManager,
                "reconcile_prefix_hit",
                autospec=True,
                side_effect=SlotResourceManager.reconcile_prefix_hit,
            ) as spy_reconcile,
        ):
            engine._step_sync()

        assert spy_find_slot.call_count == 2
        assert spy_reconcile.call_count == 2
        # The underlying fake runner's own calls (already asserted by the
        # 7-b tests above) must still have happened -- the coordinator forwarded,
        # it did not intercept and answer on its own.
        assert len(runner.find_best_slot_calls) == 2
        assert engine.active[1]["anchor"] == 901
        assert engine.active[0]["anchor"] == 900
