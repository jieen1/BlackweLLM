"""N2: stop-sequence tests for server/engine.py's decode-round bookkeeping.

Covers:
- Unit tests of ServerEngine._stop_check_token / _flush_stop_pending /
  _drop_stop_pending_from_committed against a bare engine instance
  (no admission loop involved) with a fake per-token-id tokenizer, so a
  stop sequence split exactly across a token boundary can be constructed
  deterministically.
- Full-round integration tests via _activate_slot + _step_sync with a
  fake LagunaBackend-shaped runner, covering:
  * the plain sampled-decode path (one token committed per round),
  * the MTP verify/commit path committing SEVERAL tokens in one round,
    with the stop sequence landing mid-batch (the speculative-decode
    interaction called out in the task),
  * a stop sequence matched entirely within the anchor (first) token,
  * natural EOS/max_tokens finishing while a safe (never-matching) tail
    is still held back -- it must be flushed, not silently dropped.

No GPU/model required: ServerEngine's real __init__ only loads the
tokenizer (offline-cached) and pure Python state; self.runner stays None
until the engine thread's model load, which these tests never invoke.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from runtime.backends.protocol import BackendCapabilities
from server.engine import GenerationRequest, ServerEngine, StreamChannel


class _FakeIdTok:
    """Fake tokenizer: each token id maps to a FIXED string; decode()
    concatenates the mapped strings for the given ids, in order. Gives
    full, deterministic control over exactly which characters a token
    boundary falls on -- the real tokenizer's BPE vocabulary would make
    constructing an exact cross-boundary split fragile to depend on."""

    def __init__(self, mapping: dict[int, str]):
        self._mapping = mapping

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._mapping[i] for i in ids)


def _make_bare_engine(id_to_str: dict[int, str], eos_ids: frozenset[int] = frozenset()):
    """A real ServerEngine (offline tokenizer load, no GPU) with its
    tokenizer swapped for a deterministic fake and runner left for the
    caller to fill in.

    Carries its own (never-run) event loop as ``_asyncio_loop``, matching
    production wiring (asyncio thread's loop) closely enough for
    ``_resolve_future``'s ``call_soon_threadsafe`` to work: tests pump it
    with ``_pump(engine)`` after any call that might finish a request.
    """
    engine = ServerEngine(
        backend="laguna", capacity=2, num_slots=2, enable_cudagraph=False, production=True
    )
    engine.tok = _FakeIdTok(id_to_str)
    engine.eos_token_ids = eos_ids
    engine.runner = None
    engine._asyncio_loop = asyncio.new_event_loop()
    return engine


def _pump(engine: ServerEngine) -> None:
    """Let the engine's event loop process any ``call_soon_threadsafe``
    callbacks queued by ``_resolve_future``/``StreamChannel.put`` so far."""
    engine._asyncio_loop.run_until_complete(asyncio.sleep(0))


def _drain_channel_buf(channel: StreamChannel) -> list[int]:
    """Flatten every token-id batch pushed to a StreamChannel's internal
    buffer (never having awaited .get()) into one flat list, in order."""
    out: list[int] = []
    for item in channel._buf:
        if item is None:
            continue
        out.extend(item)
    return out


def _make_req(
    engine: ServerEngine,
    stop_sequences,
    stream: bool = False,
    request_id: str = "req-1",
    stop_on_tool_call: bool = False,
) -> tuple[GenerationRequest, StreamChannel | None]:
    from runtime.sampling import SamplingParams

    fut = engine._asyncio_loop.create_future()
    channel = StreamChannel() if stream else None
    req = GenerationRequest(
        request_id=request_id,
        prompt_ids=[1, 2, 3],
        max_tokens=100,
        future=fut,
        stream_channel=channel,
        sampling_params=SamplingParams(temperature=0.0),
        stop_sequences=stop_sequences,
        stop_on_tool_call=stop_on_tool_call,
    )
    return req, channel


class TestStopCheckTokenUnit:
    """Direct tests of the per-token helper methods, bypassing admission
    and the decode loop entirely."""

    def _setup(self, id_to_str, stop_sequences):
        from server.formats.stream import StreamProcessor

        engine = _make_bare_engine(id_to_str)
        req, channel = _make_req(engine, stop_sequences, stream=True)
        st = {
            "req": req,
            "committed_tokens": [],
            "stop_sequences": stop_sequences,
            "stop_tracker": StreamProcessor(engine.tok, thinking_capable=False),
            "stop_pending_ids": [],
            "stop_pending_text": "",
        }
        return engine, st, channel

    def test_cross_token_boundary_match_never_streamed(self):
        """stop='STOP' arrives split as two tokens: 'ST' then 'OP tail'.
        Neither token may ever reach the stream channel."""
        id_to_str = {1: "ST", 2: "OP tail"}
        engine, st, channel = self._setup(id_to_str, ["STOP"])

        st["committed_tokens"].append(1)
        assert engine._stop_check_token(st, 1) is None
        assert _drain_channel_buf(channel) == []  # ambiguous, held back

        st["committed_tokens"].append(2)
        matched = engine._stop_check_token(st, 2)
        assert matched == "STOP"

        engine._drop_stop_pending_from_committed(st)
        assert st["committed_tokens"] == []
        assert _drain_channel_buf(channel) == []  # never leaked

    def test_ambiguous_tail_resolves_safe_and_flushes_both_tokens(self):
        """'ST' looks like it could become 'STOP', but the next token
        ('range') rules that out -- both tokens must flush together, in
        order, once resolved."""
        id_to_str = {1: "ST", 2: "range"}
        engine, st, channel = self._setup(id_to_str, ["STOP"])

        st["committed_tokens"].append(1)
        assert engine._stop_check_token(st, 1) is None
        assert _drain_channel_buf(channel) == []

        st["committed_tokens"].append(2)
        assert engine._stop_check_token(st, 2) is None
        assert _drain_channel_buf(channel) == [1, 2]

    def test_immediately_safe_token_flushes_alone(self):
        id_to_str = {1: "hello world"}
        engine, st, channel = self._setup(id_to_str, ["STOP"])
        st["committed_tokens"].append(1)
        assert engine._stop_check_token(st, 1) is None
        assert _drain_channel_buf(channel) == [1]

    def test_match_within_a_single_token(self):
        """A whole stop sequence can arrive inside ONE token's decoded
        text (e.g. a multi-char BPE token)."""
        id_to_str = {1: "say STOP now"}
        engine, st, channel = self._setup(id_to_str, ["STOP"])
        st["committed_tokens"].append(1)
        matched = engine._stop_check_token(st, 1)
        assert matched == "STOP"
        engine._drop_stop_pending_from_committed(st)
        assert st["committed_tokens"] == []
        assert _drain_channel_buf(channel) == []

    def test_stop_sequence_inside_reasoning_is_not_matched(self):
        """A stop sequence occurring only inside a <think>...</think> span
        must not trigger -- stop applies to content, not reasoning (see
        server/engine.py module docstring for the OpenAI reasoning_content
        rationale). Reasoning tokens still stream immediately (same
        latency as a request with no stop_sequences at all) -- only
        CONTENT is ever held back for stop-matching purposes."""
        id_to_str = {1: "<think>STOP", 2: "</think>answer"}
        engine, st, channel = self._setup(id_to_str, ["STOP"])

        st["committed_tokens"].append(1)
        assert engine._stop_check_token(st, 1) is None
        # Still inside reasoning -- nothing confirmed as content yet, and
        # the token streams immediately regardless (no latency penalty).
        assert st["stop_pending_text"] == ""
        assert _drain_channel_buf(channel) == [1]

        st["committed_tokens"].append(2)
        assert engine._stop_check_token(st, 2) is None
        # Reasoning closed; "answer" is the only content revealed, and it
        # does not contain "STOP", so it flushes too.
        assert "STOP" not in st["stop_pending_text"]
        assert _drain_channel_buf(channel) == [1, 2]

    def test_non_streaming_slot_never_touches_channel(self):
        """stream_channel is None for non-streaming requests -- flush must
        be a safe no-op, not a crash."""
        id_to_str = {1: "hello"}
        engine, st, _channel = self._setup(id_to_str, ["STOP"])
        st["req"] = _make_req(engine, ["STOP"], stream=False)[0]
        st["committed_tokens"].append(1)
        assert engine._stop_check_token(st, 1) is None  # no exception


class _FakeRunner:
    """Fake LagunaBackend-shaped runner. Scripts decode_batch_sampled /
    mtp_verify_and_commit_batch responses per call, one dict of
    {slot: [tokens...]} per round, consumed in order."""

    def __init__(self, has_speculative_decode: bool, rounds: list[dict[int, list[int]]]):
        self.capabilities = BackendCapabilities(
            speculative_decode=has_speculative_decode,
            prefix_cache=False,
            cuda_graph=False,
            chunked_prefill=False,
            warm_continue=False,
        )
        self.has_speculative_decode = has_speculative_decode
        self._rounds = list(rounds)
        self._kv_len: dict[int, int] = {}

    def slot_state(self, slot: int):
        return SimpleNamespace(kv_len=self._kv_len.get(slot, 0))

    def reset_slot(self, slot: int) -> None:
        pass

    def decode_batch_sampled(
        self, slot_ids, token_ids, kv_lengths, params_list, *, return_logprobs=False, top_logprobs=0
    ):
        round_tokens = self._rounds.pop(0)
        for s in slot_ids:
            self._kv_len[s] = self._kv_len.get(s, 0) + 1
        return [round_tokens[s][0] for s in slot_ids]

    def mtp_verify_and_commit_batch(
        self,
        slot_ids,
        anchors,
        drafts,
        *,
        params_per_slot=None,
        return_logprobs=False,
        top_logprobs=0,
    ):
        round_tokens = self._rounds.pop(0)
        decisions = {}
        for s in slot_ids:
            committed = round_tokens[s]
            decisions[s] = {
                "committed": committed,
                "num_accepted": len(committed),
                "next_anchor": committed[-1] if committed else anchors[s],
                "next_draft_tokens": [],
            }
        return decisions


def _run_step_sync_until_finished(engine: ServerEngine, slot: int, max_rounds: int = 20) -> None:
    for _ in range(max_rounds):
        if slot not in engine.active:
            return
        engine._step_sync()
    raise AssertionError(f"slot {slot} did not finish within {max_rounds} rounds")


def _bare_step_sync_engine(id_to_str, runner: _FakeRunner, eos_ids=frozenset()) -> ServerEngine:
    engine = _make_bare_engine(id_to_str, eos_ids=eos_ids)
    engine.runner = runner
    r, w = os.pipe()
    os.set_blocking(r, False)
    os.set_blocking(w, False)
    engine._req_pipe_r = r
    engine._req_pipe_w = w
    engine.request_timeout_s = 0
    engine.watchdog_max_stale_rounds = 0
    engine.enable_session_affinity = False
    engine.retained = {}
    engine.waiting = []
    engine._pending_prefill = None
    return engine


class TestStepSyncSampledPathStopSequences:
    """has_speculative_decode=False -- every slot routes through the plain
    per-token decode_batch_sampled path (classify_decode_slots)."""

    def test_cross_token_boundary_match_ends_request_and_never_leaks(self):
        # anchor=10 "Hello " (safe) -> round1 tok=11 "ST" (ambiguous, held)
        # -> round2 tok=12 "OP world!" (completes "STOP", must never stream
        # tokens 11/12, and must not commit them).
        id_to_str = {10: "Hello ", 11: "ST", 12: "OP world!"}
        runner = _FakeRunner(
            has_speculative_decode=False,
            rounds=[{0: [11]}, {0: [12]}],
        )
        engine = _bare_step_sync_engine(id_to_str, runner)
        req, channel = _make_req(engine, ["STOP"], stream=True)
        engine._activate_slot(0, req, anchor=10, drafts=[])
        assert 0 in engine.active  # not finished by the anchor alone

        _run_step_sync_until_finished(engine, 0, max_rounds=5)
        _pump(engine)

        result = req.future.result()
        assert result["finish_reason"] == "stop"
        assert result["matched_stop_sequence"] == "STOP"
        assert result["committed_token_ids"] == [10]
        assert _drain_channel_buf(channel) == [10]

    def test_no_stop_configured_behaves_exactly_as_before(self):
        """Regression guard: a request with no stop_sequences must keep
        streaming every token immediately, unaffected by any of this."""
        id_to_str = {10: "a", 11: "b", 12: "c"}
        runner = _FakeRunner(has_speculative_decode=False, rounds=[{0: [11]}, {0: [12]}])
        engine = _bare_step_sync_engine(id_to_str, runner, eos_ids=frozenset({99}))
        req, channel = _make_req(engine, None, stream=True)
        engine._activate_slot(0, req, anchor=10, drafts=[])
        engine._step_sync()
        engine._step_sync()
        # tok 12 (from round 2) is EOS-free and not at max_tokens, so the
        # slot stays active -- force finish via a synthetic EOS round.
        assert _drain_channel_buf(channel) == [10, 11, 12]
        assert engine.active[0]["committed_tokens"] == [10, 11, 12]


class TestStepSyncMtpPathStopSequences:
    """has_speculative_decode=True -- greedy slots route through
    mtp_verify_and_commit_batch, which can commit several tokens in ONE
    round. The stop sequence lands in the middle of that batch."""

    def test_multi_token_round_truncates_mid_batch(self):
        # One MTP round commits 3 draft tokens at once: "Hello ", "ST",
        # "OP", "!!!" -- "STOP" completes at the 3rd token; "!!!" (a 4th,
        # never even reached) must not appear anywhere.
        id_to_str = {10: "Hello ", 20: "ST", 21: "OP", 22: "!!!"}
        runner = _FakeRunner(
            has_speculative_decode=True,
            rounds=[{0: [20, 21, 22]}],
        )
        engine = _bare_step_sync_engine(id_to_str, runner)
        req, channel = _make_req(engine, ["STOP"], stream=True)
        engine._activate_slot(0, req, anchor=10, drafts=[])
        assert _drain_channel_buf(channel) == [10]  # anchor: safe, flushed

        _run_step_sync_until_finished(engine, 0, max_rounds=3)
        _pump(engine)

        result = req.future.result()
        assert result["finish_reason"] == "stop"
        assert result["matched_stop_sequence"] == "STOP"
        assert result["committed_token_ids"] == [10]
        assert _drain_channel_buf(channel) == [10]

    def test_anchor_alone_completes_a_stop_sequence(self):
        """A single-token stop sequence (or one whose whole text lands in
        the anchor) must stop generation before any decode round runs."""
        id_to_str = {10: "STOP right away"}
        runner = _FakeRunner(has_speculative_decode=True, rounds=[])
        engine = _bare_step_sync_engine(id_to_str, runner)
        req, channel = _make_req(engine, ["STOP"], stream=True)

        engine._activate_slot(0, req, anchor=10, drafts=[])
        _pump(engine)

        assert 0 not in engine.active  # finished immediately, never decoded
        result = req.future.result()
        assert result["finish_reason"] == "stop"
        assert result["matched_stop_sequence"] == "STOP"
        assert result["committed_token_ids"] == []
        assert _drain_channel_buf(channel) == []


class TestStopPendingFlushedOnNaturalFinish:
    """A held-back-but-never-matching tail must be flushed when
    generation ends for an unrelated reason (EOS/max_tokens), not
    silently dropped."""

    def test_eos_flushes_pending_safe_tail(self):
        # tok=11 "ST" stays ambiguous (ends the round); tok=12 is EOS.
        # "ST" was never confirmed *unsafe* -- it must reach the client
        # once generation is known to be over.
        id_to_str = {10: "Hello ", 11: "ST"}
        runner = _FakeRunner(has_speculative_decode=False, rounds=[{0: [11]}])
        engine = _bare_step_sync_engine(id_to_str, runner, eos_ids=frozenset({12}))
        req, channel = _make_req(engine, ["STOP"], stream=True)
        engine._activate_slot(0, req, anchor=10, drafts=[])
        engine._step_sync()  # commits tok 11, held back (ambiguous)
        assert _drain_channel_buf(channel) == [10]
        assert 0 in engine.active

        runner._rounds.append({0: [12]})  # EOS next round
        engine._step_sync()
        _pump(engine)

        result = req.future.result()
        assert result["finish_reason"] == "stop"
        assert result["matched_stop_sequence"] is None
        assert result["committed_token_ids"] == [10, 11]
        assert _drain_channel_buf(channel) == [10, 11]


class TestToolCallTerminalCondition:
    """A parsed tool block ends the scheduler request before model chatter
    after ``</tool_call>`` can turn into a visible repetition loop."""

    @staticmethod
    def _set_qwen_parser():
        from server.formats.tool_parsers import get_active_parser, set_active_parser

        previous = get_active_parser().name
        set_active_parser("qwen3_coder")
        return previous

    def test_plain_decode_stops_after_complete_tool_block(self):
        previous = self._set_qwen_parser()
        try:
            id_to_str = {
                10: "prefix ",
                11: (
                    "<tool_call><function=read><parameter=path>x</parameter>"
                    "</function></tool_call>"
                ),
                12: "</function>",
            }
            runner = _FakeRunner(has_speculative_decode=False, rounds=[{0: [11]}, {0: [12]}])
            engine = _bare_step_sync_engine(id_to_str, runner)
            req, channel = _make_req(engine, None, stream=True, stop_on_tool_call=True)
            engine._activate_slot(0, req, anchor=10, drafts=[])

            _run_step_sync_until_finished(engine, 0, max_rounds=3)
            _pump(engine)

            result = req.future.result()
            assert result["finish_reason"] == "tool_calls"
            assert result["committed_token_ids"] == [10, 11]
            assert _drain_channel_buf(channel) == [10, 11]
        finally:
            from server.formats.tool_parsers import set_active_parser

            set_active_parser(previous)

    def test_mtp_round_discards_tokens_after_complete_tool_block(self):
        previous = self._set_qwen_parser()
        try:
            id_to_str = {
                10: "prefix ",
                11: (
                    "<tool_call><function=read><parameter=path>x</parameter>"
                    "</function></tool_call>"
                ),
                12: "</function>",
            }
            runner = _FakeRunner(has_speculative_decode=True, rounds=[{0: [11, 12]}])
            engine = _bare_step_sync_engine(id_to_str, runner)
            req, channel = _make_req(engine, None, stream=True, stop_on_tool_call=True)
            engine._activate_slot(0, req, anchor=10, drafts=[])

            _run_step_sync_until_finished(engine, 0, max_rounds=3)
            _pump(engine)

            result = req.future.result()
            assert result["finish_reason"] == "tool_calls"
            assert result["committed_token_ids"] == [10, 11]
            assert _drain_channel_buf(channel) == [10, 11]
        finally:
            from server.formats.tool_parsers import set_active_parser

            set_active_parser(previous)
