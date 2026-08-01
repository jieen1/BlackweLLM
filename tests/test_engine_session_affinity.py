"""N8 (docs/architecture.md §3.5.6): coverage for ``--session-affinity`` /
warm-continue, which had zero test coverage before this file existed.

Two things are covered, deliberately kept apart:

* ``TestSessionAffinityRejectedAtStartup`` -- the fix. Before it landed,
  ``ServerEngine(enable_session_affinity=True, enable_prefix_cache=True)``
  built successfully against the real ``LagunaBackend`` and only failed
  later, silently, inside ``except Exception`` in ``_step_sync`` (every
  warm-continue attempt raised ``AttributeError``, was swallowed, and fell
  back to a cold prefill -- outputs stayed correct,
  ``session_warm_continuations`` stayed at zero, and nothing ever told the
  operator their flag was a no-op). Option (c) from §3.5.6: reject the flag
  at construction time, before any GPU work, so the failure is loud.

* ``TestWarmContinueMechanism`` -- the pre-existing ``_step_sync`` P4b
  admission block (session retention in ``_finish_request``, warm-continue
  admission, prefix-mismatch fallback, and the exact
  ``except Exception -> cold fallback`` path that is the production bug)
  never had a test exercising it at all. These use a bare engine with the
  real ``enable_session_affinity`` guard bypassed by direct attribute
  assignment (the same technique ``test_engine_stop_sequences.py`` already
  uses), plus a fake runner that *can* warm-continue -- unlike any shipping
  backend today -- so the mechanism itself is verified independent of
  whether any real backend currently implements it.

No GPU/model required: ``ServerEngine.__init__`` only loads the tokenizer
(offline-cached) and pure Python state.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from runtime.backends.protocol import BackendCapabilities, PrefixHit
from server.engine import GenerationRequest, ServerEngine


class _FakeIdTok:
    """Fake tokenizer: token id -> fixed string. Not exercised by these
    tests (no stop-sequence / streaming behavior under test here), but
    ServerEngine methods touched along the way may call ``.decode``."""

    def __init__(self, mapping: dict[int, str]):
        self._mapping = mapping

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._mapping.get(i, "") for i in ids)


class _FakeWarmRunner:
    """Fake LagunaBackend-shaped runner that *can* warm-continue -- no
    shipping backend does this today (that is N8), but the ``_step_sync``
    mechanism that would drive it is real and untested, so this fake
    exists to test that mechanism in isolation.

    ``warm_continue_error`` reproduces the exact production path: today,
    ``LagunaBackend`` has no ``mtp_prefill_warm_continue`` at all, so the
    real call raises ``AttributeError`` and ``_step_sync`` catches it via a
    bare ``except Exception``.
    """

    def __init__(
        self,
        *,
        warm_continue_result: dict | None = None,
        warm_continue_error: Exception | None = None,
        cold_anchor: int = 900,
        decode_token: int = 901,
    ):
        self.has_speculative_decode = False
        # A3 step 7-b (docs/a3-cache-coordinator-design.md §1.1): engine.py's
        # admission path now queries capabilities.prefix_cache instead of
        # hasattr(runner, "find_best_slot_for_prompt"). This fake never
        # defined find_best_slot_for_prompt, so prefix_cache=False here
        # reproduces the exact same "no cache-aware slot assignment" branch
        # the old hasattr probe took (which was also False for this fake).
        self.capabilities = BackendCapabilities(
            speculative_decode=False,
            prefix_cache=False,
            cuda_graph=False,
            chunked_prefill=False,
            warm_continue=True,
        )
        self.warm_continue_result = warm_continue_result
        self.warm_continue_error = warm_continue_error
        self.cold_anchor = cold_anchor
        self.decode_token = decode_token
        self.reset_calls: list[int] = []
        self.warm_continue_calls: list[tuple[int, list[int], int]] = []

    def reset_slot(self, slot: int) -> None:
        self.reset_calls.append(slot)

    def slot_state(self, slot: int):
        return SimpleNamespace(kv_len=0, is_fresh=True)

    def mtp_prefill_warm_continue(self, slot: int, prompt: list[int], prior_len: int) -> dict:
        self.warm_continue_calls.append((slot, list(prompt), prior_len))
        if self.warm_continue_error is not None:
            raise self.warm_continue_error
        assert self.warm_continue_result is not None
        return self.warm_continue_result

    # -- cold admission path (fallback lands here) --
    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        # PrefixHit(0, 0): a fake standing in for a real backend must return
        # the same shape a real ModelBackend does (runtime/backends/
        # protocol.py) -- engine.py now reads .effective off this value.
        return PrefixHit(kv_hit=0, state_hit=0)

    def prefill_chunked_begin(self, slots, prompts_per_slot, chunk_size: int = 512):
        result = {s: {"anchor": self.cold_anchor, "draft_tokens": []} for s in slots}
        return SimpleNamespace(done=True, result=result)

    # -- decode round tail (reached once a slot is active) --
    def decode_batch_sampled(
        self, slot_ids, token_ids, kv_lengths, params_list, *, return_logprobs=False, top_logprobs=0
    ):
        return [self.decode_token for _ in slot_ids]

    def mtp_verify_and_commit_batch(self, *args, **kwargs):
        raise NotImplementedError("has_speculative_decode=False; must not be called")


def _make_bare_session_engine(runner: _FakeWarmRunner) -> ServerEngine:
    """A real ``ServerEngine`` (offline tokenizer load, no GPU) with the
    session-affinity guard bypassed the same way
    ``tests/test_engine_stop_sequences.py::_bare_step_sync_engine`` bypasses
    it in the other direction, and its runner swapped for a fake that can
    actually warm-continue."""
    engine = ServerEngine(backend="laguna", capacity=1, num_slots=1, enable_cudagraph=False)
    engine.tok = _FakeIdTok({})
    engine.eos_token_ids = frozenset()
    engine.enable_session_affinity = True
    engine.enable_prefix_cache = True
    engine.runner = runner
    engine._asyncio_loop = asyncio.new_event_loop()
    return engine


def _make_req(engine: ServerEngine, prompt_ids: list[int], session_id: str) -> GenerationRequest:
    from runtime.sampling import SamplingParams

    fut = engine._asyncio_loop.create_future()
    return GenerationRequest(
        request_id="req-2",
        prompt_ids=prompt_ids,
        max_tokens=100,
        future=fut,
        session_id=session_id,
        sampling_params=SamplingParams(temperature=0.0),
    )


def _seed_retained_session(
    engine: ServerEngine, session_id: str, slot: int, committed_full: list[int]
) -> None:
    """Simulate turn 1 having already finished and retained ``slot`` --
    exactly the state ``_finish_request`` leaves behind (see its
    session-retention branch), constructed directly so these tests do not
    also have to drive a full first round.

    ``_finish_request``'s retention branch returns before ever appending
    the slot to ``free_slots`` -- a retained slot is not free -- so this
    removes it too, rather than leaving ``__init__``'s
    ``free_slots = list(range(capacity))`` untouched and silently wrong.
    """
    if slot in engine.free_slots:
        engine.free_slots.remove(slot)
    engine.retained[session_id] = {
        "slot": slot,
        "expire_t": time.perf_counter() + engine.session_ttl_s,
        "prior_len": len(committed_full),
        "committed_full": list(committed_full),
    }


class TestSessionAffinityRejectedAtStartup:
    """The N8 fix: reject at construction time instead of degrading
    silently at request time."""

    def test_rejects_session_affinity_because_backend_lacks_warm_continue(self) -> None:
        with pytest.raises(ValueError, match="warm_continue"):
            ServerEngine(
                backend="laguna",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                enable_prefix_cache=True,
                enable_session_affinity=True,
            )

    def test_session_affinity_off_is_unaffected(self) -> None:
        # The default (and only shipped) configuration must still construct
        # without needing warm_continue at all.
        engine = ServerEngine(backend="laguna", capacity=1, num_slots=1, enable_cudagraph=False)
        assert engine.enable_session_affinity is False

    def test_prefix_cache_requirement_is_checked_first(self) -> None:
        # Pre-existing check; must still fire before the new one is reached,
        # so the error message stays about the actual first problem.
        with pytest.raises(ValueError, match="enable_prefix_cache"):
            ServerEngine(
                backend="laguna",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                enable_prefix_cache=False,
                enable_session_affinity=True,
            )


class TestWarmContinueMechanism:
    """The pre-existing ``_step_sync`` P4b block, previously untested."""

    def test_matching_prefix_warm_continues_and_activates_slot(self) -> None:
        runner = _FakeWarmRunner(warm_continue_result={"anchor": 42, "draft_tokens": [43, 44]})
        engine = _make_bare_session_engine(runner)
        _seed_retained_session(engine, "sess-1", slot=0, committed_full=[1, 2, 3, 4, 5])
        req = _make_req(engine, prompt_ids=[1, 2, 3, 4, 5, 6, 7], session_id="sess-1")
        engine.waiting = [req]

        engine._step_sync()

        assert "sess-1" not in engine.retained
        assert runner.warm_continue_calls == [(0, [1, 2, 3, 4, 5, 6, 7], 5)]
        assert engine.stats["session_warm_continuations"] == 1
        assert engine.stats["session_warm_fallbacks"] == 0
        sample = engine.stats["session_warm_continuation_samples"][0]
        assert sample["session_id"] == "sess-1"
        assert sample["slot"] == 0
        assert sample["prior_len"] == 5
        assert sample["suffix_len"] == 2
        # _activate_slot ran with the warm-continue anchor, not a cold one.
        assert engine.active[0]["anchor"] == 42
        assert engine.active[0]["drafts"] == [43, 44]
        assert engine.active[0]["committed_tokens"][0] == 42
        # The warm path never touched free_slots -- the slot was reused
        # directly, exactly as the cold path's alternative would not.
        assert 0 not in engine.free_slots

    def test_prefix_mismatch_falls_back_to_cold_admission(self) -> None:
        runner = _FakeWarmRunner(cold_anchor=77)
        engine = _make_bare_session_engine(runner)
        # committed_full's prefix does NOT match the new prompt's prefix.
        _seed_retained_session(engine, "sess-1", slot=0, committed_full=[1, 2, 3, 4, 5])
        req = _make_req(engine, prompt_ids=[1, 2, 3, 9, 9, 6, 7], session_id="sess-1")
        engine.waiting = [req]

        engine._step_sync()

        assert "sess-1" not in engine.retained
        assert runner.warm_continue_calls == []  # never attempted -- caught before the call
        assert engine.stats["session_warm_fallbacks"] == 1
        assert engine.stats["session_warm_continuations"] == 0
        assert 0 in runner.reset_calls
        # Fallback re-queues, and the same round's cold-admission path
        # picks it back up (reconcile_prefix_hit / prefill_chunked_begin).
        assert engine.active[0]["anchor"] == 77

    def test_warm_continue_exception_falls_back_to_cold_admission(self) -> None:
        # This is the actual production path today: no LagunaBackend
        # defines mtp_prefill_warm_continue, so the real call always raises
        # AttributeError here. _step_sync must swallow it and fall back --
        # and the fallback itself must actually work, which nothing checked
        # before this test.
        runner = _FakeWarmRunner(
            warm_continue_error=AttributeError(
                "'LagunaBackend' object has no attribute 'mtp_prefill_warm_continue'"
            ),
            cold_anchor=88,
        )
        engine = _make_bare_session_engine(runner)
        _seed_retained_session(engine, "sess-1", slot=0, committed_full=[1, 2, 3, 4, 5])
        req = _make_req(engine, prompt_ids=[1, 2, 3, 4, 5, 6, 7], session_id="sess-1")
        engine.waiting = [req]

        engine._step_sync()

        assert runner.warm_continue_calls == [(0, [1, 2, 3, 4, 5, 6, 7], 5)]
        assert "sess-1" not in engine.retained
        assert engine.stats["session_warm_fallbacks"] == 1
        assert engine.stats["session_warm_continuations"] == 0
        assert 0 in runner.reset_calls
        assert engine.active[0]["anchor"] == 88

    def test_expired_retained_slot_is_reclaimed_without_a_new_request(self) -> None:
        runner = _FakeWarmRunner()
        engine = _make_bare_session_engine(runner)
        _seed_retained_session(engine, "sess-1", slot=0, committed_full=[1, 2, 3])
        engine.retained["sess-1"]["expire_t"] = time.perf_counter() - 1  # already expired
        engine.waiting = []

        engine._expire_retained_slots()

        assert "sess-1" not in engine.retained
        assert 0 in runner.reset_calls
        assert 0 in engine.free_slots
        assert engine.stats["session_expirations"] == 1
