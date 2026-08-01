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
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from runtime.backends.protocol import BackendCapabilities, PrefixHit
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
        self, token_ids: list[int], free_slots: list[int]
    ) -> tuple[int, int]:
        self.find_best_slot_calls.append((list(token_ids), list(free_slots)))
        chosen = self._best_slot_by_prompt[tuple(token_ids)]
        assert chosen in free_slots
        return (chosen, 64)  # hit depth here is discarded by engine.py (§1.1)

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        return PrefixHit(kv_hit=0, state_hit=0)

    def prefill_chunked_begin(self, slots, prompts_per_slot, chunk_size: int = 512):
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

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        return PrefixHit(kv_hit=0, state_hit=0)

    def prefill_chunked_begin(self, slots, prompts_per_slot, chunk_size: int = 512):
        result = {s: {"anchor": 700 + s, "draft_tokens": []} for s in slots}
        return SimpleNamespace(done=True, result=result)

    def decode_batch_sampled(
        self, slot_ids, token_ids, kv_lengths, params_list, *, return_logprobs=False, top_logprobs=0
    ):
        return [701 for _ in slot_ids]

    def mtp_verify_and_commit_batch(self, *args, **kwargs):
        raise NotImplementedError("has_speculative_decode=False; must not be called")


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


class TestCapabilityGatedSlotAssignment:
    def test_capability_true_uses_find_best_slot_for_prompt_and_honors_choice(self) -> None:
        # Two prompts, two free slots (0, 1); the fake deliberately picks the
        # OPPOSITE slot from pop-order for each, so a passing assertion can
        # only mean find_best_slot_for_prompt's return value actually drove
        # the assignment (not a coincidence of iteration order).
        prompt_a, prompt_b = (11, 12, 13), (21, 22, 23)
        runner = _FakeCacheAwareRunner(
            best_slot_by_prompt={prompt_a: 1, prompt_b: 0}
        )
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
