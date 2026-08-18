"""The engine must never block on the wakeup pipe while a request is already
sitting in ``_req_deque``.

Live failure, 2026-08-01. ``_step_sync`` drains the deque into ``self.waiting``
at the top of a round, then immediately calls ``_drain_pipe`` to clear every
wakeup byte pending at that instant. A request appended by the asyncio thread
*between* those two lines therefore ends up in ``_req_deque`` with its wakeup
byte already consumed -- invisible to the ``not self.waiting`` test that guards
the idle branch at the bottom of the same round. The engine then blocked on a
read that nothing would ever satisfy.

The window is small in wall-clock terms and enormous in practice: a
conversational client sends its next turn milliseconds after the previous
response completes, which is exactly when the engine is winding down into that
branch. The observed case was an agent whose follow-up arrived 152 ms later;
`rounds` froze, the GPU went idle, `active` and `waiting` were both empty, and
the client eventually reported the stream stalled and timed out.

Blocking forever is not something an assertion can catch directly, so the
transition into blocking mode is intercepted instead of being allowed to
happen -- entering it is the failure signal, and the test never hangs.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("transformers")

import server.engine as engine_mod  # noqa: E402
from server.engine import GenerationRequest, ServerEngine  # noqa: E402


class _EnteredIdleBlock(Exception):
    """Raised in place of the blocking read so tests never actually hang."""


def _idle_engine() -> ServerEngine:
    engine = ServerEngine(
        backend="laguna", capacity=2, num_slots=2, enable_cudagraph=False, production=True
    )
    engine.runner = None
    engine._asyncio_loop = asyncio.new_event_loop()
    return engine


def _queued_request(engine: ServerEngine) -> GenerationRequest:
    return GenerationRequest(
        request_id="queued-while-winding-down",
        prompt_ids=[1, 2, 3],
        max_tokens=8,
        future=engine._asyncio_loop.create_future(),
    )


def _step_blocks(engine: ServerEngine, monkeypatch) -> bool:
    """Run one round; report whether the engine entered the idle blocking read.

    Rather than letting the step actually block (untestable, and it would hang
    the suite), the transition into blocking mode is intercepted: the branch
    switches the pipe to blocking right before reading, so that call is the
    signal. Everything the step does after that point is irrelevant here.
    """
    real_set_blocking = os.set_blocking

    def _spy(fd, blocking):
        if fd == engine._req_pipe_r and blocking:
            raise _EnteredIdleBlock
        return real_set_blocking(fd, blocking)

    monkeypatch.setattr(engine_mod.os, "set_blocking", _spy)
    try:
        engine._step_sync()
    except _EnteredIdleBlock:
        return True
    except Exception:
        # The step got past the idle branch and failed later (no real runner
        # is wired up here). Past the branch is all this test cares about.
        return False
    return False


class TestIdleWakeup:
    def test_coalesces_idle_wave_before_admission(self, monkeypatch):
        """A second request arriving on the wakeup pipe joins the first.

        The fake select result models the asyncio producer writing the pipe
        while the engine is in its bounded coalescing window.  This keeps the
        test CPU-only and proves the request is collected before the normal
        admission branch can start a long prefill.
        """
        engine = _idle_engine()
        try:
            engine._admission_coalesce_s = 0.01
            engine.waiting = [_queued_request(engine)]
            calls = []

            def _select(readable, _writable, _exceptional, _timeout):
                calls.append(True)
                if len(calls) == 1:
                    req = _queued_request(engine)
                    req.request_id = "coalesced-request"
                    engine._req_deque.append(req)
                return ([engine._req_pipe_r], [], [])

            monkeypatch.setattr(engine_mod.select, "select", _select)
            engine._coalesce_admission_wave()

            assert calls
            assert [req.request_id for req in engine.waiting] == [
                "queued-while-winding-down",
                "coalesced-request",
            ]
            assert engine.stats["admission_coalesce_waits"] == 1
        finally:
            engine._asyncio_loop.close()

    def test_does_not_block_when_request_arrives_mid_round(self, monkeypatch):
        """Reproduce the live race at its exact instant.

        The request must land *after* the top-of-round `_drain_requests` and
        be swallowed by the `_drain_pipe` immediately following it -- putting
        it in the deque before the round starts would just be drained
        normally and prove nothing. `_drain_pipe` is the seam: it runs at
        precisely the moment the asyncio thread's append-then-write is unsafe,
        so appending from inside it, and writing no wakeup byte, is the race.
        """
        engine = _idle_engine()
        try:
            real_drain_pipe = engine_mod._drain_pipe
            arrived = []

            def _drain_pipe_and_race(fd):
                real_drain_pipe(fd)
                if not arrived:
                    # asyncio thread appends here; its wakeup byte is written
                    # before this drain and therefore already consumed.
                    engine._req_deque.append(_queued_request(engine))
                    arrived.append(True)

            monkeypatch.setattr(engine_mod, "_drain_pipe", _drain_pipe_and_race)

            assert not _step_blocks(engine, monkeypatch), (
                "engine entered the idle blocking read while a request sat in "
                "_req_deque with its wakeup byte already consumed -- this is "
                "the 2026-08-01 stall"
            )
            assert [r.request_id for r in engine.waiting] == ["queued-while-winding-down"]
        finally:
            engine._asyncio_loop.close()

    def test_still_blocks_when_genuinely_idle(self, monkeypatch):
        """The fix must not turn the idle wait into a busy loop.

        With nothing queued the step is expected to block; that is what makes
        the idle path cost zero CPU. Asserting it here keeps a future "just
        return early" simplification from silently reintroducing a spin.
        """
        engine = _idle_engine()
        try:
            assert _step_blocks(engine, monkeypatch), (
                "engine skipped the idle wait with nothing to do -- the loop would spin"
            )
        finally:
            engine._asyncio_loop.close()
