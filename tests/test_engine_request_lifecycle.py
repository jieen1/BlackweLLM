"""Regression coverage for bounded request ownership and stream cleanup."""

from __future__ import annotations

import asyncio
import collections
import os
import threading

import pytest

from server.app import _submit_stream_with_thinking_budget
from server.engine import (
    _PREFIX_DEDUP_PUBLISHED_KEYS_KEPT,
    GenerationRequest,
    RequestQueueFull,
    ServerEngine,
)
from server.formats.stream import StreamProcessor


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _bare_engine(loop: asyncio.AbstractEventLoop, limit: int = 2) -> ServerEngine:
    engine = ServerEngine.__new__(ServerEngine)
    engine._req_deque = collections.deque()
    engine.max_pending_requests = limit
    engine._request_count_lock = threading.Lock()
    engine._request_count = 0
    engine._req_pipe_r, engine._req_pipe_w = os.pipe()
    engine._asyncio_loop = loop
    return engine


def _request(loop: asyncio.AbstractEventLoop, request_id: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_ids=[1, 2, 3],
        max_tokens=8,
        future=loop.create_future(),
    )


def test_request_budget_rejects_and_reuses_capacity():
    async def run():
        loop = asyncio.get_running_loop()
        engine = _bare_engine(loop, limit=2)
        try:
            first = _request(loop, "first")
            second = _request(loop, "second")
            engine._enqueue_request(first)
            engine._enqueue_request(second)
            with pytest.raises(RequestQueueFull):
                engine._enqueue_request(_request(loop, "rejected"))
            assert engine.pending_request_count() == 2

            assert engine._req_deque.popleft() is first
            engine._release_request(first)
            replacement = _request(loop, "replacement")
            engine._enqueue_request(replacement)
            assert engine.pending_request_count() == 2
            assert [req.request_id for req in engine._req_deque] == ["second", "replacement"]
        finally:
            os.close(engine._req_pipe_r)
            os.close(engine._req_pipe_w)

    asyncio.run(run())


class _ResetRunner:
    def __init__(self):
        self.reset_slots = []

    def reset_slot(self, slot):
        self.reset_slots.append(slot)


def test_engine_error_releases_every_non_active_request_container():
    async def run():
        loop = asyncio.get_running_loop()
        engine = _bare_engine(loop, limit=4)
        engine.block_size = 128
        engine.waiting = []
        engine.free_slots = [0, 1]
        engine.runner = _ResetRunner()
        engine._pending_prefill = object()
        engine._pending_prefill_reqs = []
        engine._admission_inflight_reqs = []
        engine._prefix_dedup_inflight = set()
        engine._prefix_dedup_published = set()
        requests = [_request(loop, f"request-{index}") for index in range(4)]
        for req in requests:
            req.prompt_ids = list(range(128))
            engine._enqueue_request(req)

        pending_a = engine._req_deque.popleft()
        pending_b = engine._req_deque.popleft()
        waiting = engine._req_deque.popleft()
        engine.waiting.append(waiting)
        engine._pending_prefill_reqs = [(0, pending_a)]
        engine._admission_inflight_reqs = [(1, pending_b)]

        engine._fail_pending_requests(RuntimeError("synthetic engine failure"))
        results = await asyncio.gather(
            *(req.future for req in requests), return_exceptions=True
        )
        assert engine.pending_request_count() == 0
        assert not engine._req_deque
        assert not engine.waiting
        assert engine._pending_prefill is None
        assert engine._pending_prefill_reqs == []
        assert engine._admission_inflight_reqs == []
        assert engine.runner.reset_slots == [0, 1]
        assert all(isinstance(result, RuntimeError) for result in results)

        os.close(engine._req_pipe_r)
        os.close(engine._req_pipe_w)

    asyncio.run(run())


def test_prefix_dedup_history_is_bounded_and_does_not_retain_prompt_tuples():
    loop = asyncio.new_event_loop()
    engine = _bare_engine(loop, limit=2)
    engine.block_size = 4
    engine._prefix_dedup_inflight = set()
    engine._prefix_dedup_published = set()
    try:
        for index in range(_PREFIX_DEDUP_PUBLISHED_KEYS_KEPT + 32):
            req = _request(loop, f"published-{index}")
            req.prompt_ids = [index, index + 1, index + 2, index + 3]
            engine._release_prefix_dedup_keys([req], published=True)
        assert len(engine._prefix_dedup_published) <= _PREFIX_DEDUP_PUBLISHED_KEYS_KEPT
        assert all(isinstance(key, int) for key in engine._prefix_dedup_published)
    finally:
        loop.close()
        os.close(engine._req_pipe_r)
        os.close(engine._req_pipe_w)


class _ClosableStreamEngine:
    def __init__(self):
        self.tok = _Tokenizer()
        self.closed = False

    def submit_stream(self, *_args, **_kwargs):
        async def stream():
            try:
                yield [ord("o"), ord("k")]
                yield {
                    "committed_token_ids": [ord("o"), ord("k")],
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "finish_reason": "stop",
                }
            finally:
                self.closed = True

        return stream()


def test_stream_adapter_closes_engine_iterator_after_terminal_result():
    async def run():
        engine = _ClosableStreamEngine()
        processor = StreamProcessor(engine.tok, thinking_capable=False)
        items = []
        async for item in _submit_stream_with_thinking_budget(
            engine,
            [ord("p")],
            8,
            thinking_budget=None,
            processor=processor,
        ):
            items.append(item)
        assert isinstance(items[-1], dict)
        assert engine.closed is True

    asyncio.run(run())
