"""Reasoning/thinking contract coverage (docs/roadmap.md T0-3/T0-4/R3/R4).

The contract (see docs/api-layer-design.md for the full writeup):
  - `content` (OpenAI) / `text` block (Anthropic) NEVER contains reasoning.
  - Reasoning is exposed via OpenAI's `reasoning_content` (delta/message)
    and a non-standard Anthropic `reasoning_content_delta` SSE event /
    top-level `reasoning_content` field -- NOT a spec Anthropic `thinking`
    content block (see server/formats/anthropic.py's build_response
    docstring: that broke Claude Desktop before, commit f13fd4a).
  - A `<think>`/`</think>` that is not the very first thing generated is
    ordinary visible content and must reach the client byte-for-byte.

CPU-only: no GPU, no tokenizer, no running server.
"""

from __future__ import annotations

import pytest

from server.formats import anthropic as anthropic_format
from server.formats import openai as openai_format
from server.formats.stream import StreamProcessor
from server.formats.thinking import find_reasoning_span, strip_thinking, strip_usage_artifacts

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class _FakeTok:
    """Printable-ASCII-only fake tokenizer: token id == ord(char)."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    @staticmethod
    def ids(text: str) -> list[int]:
        return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# server/formats/thinking.py: find_reasoning_span (pure, no tokenizer)
# ---------------------------------------------------------------------------


class TestFindReasoningSpan:
    def test_no_think_at_all(self):
        assert find_reasoning_span("Just an answer.") is None

    def test_think_not_at_position_zero_is_not_a_span(self):
        """The core R4 fix: <think> anywhere but position 0 is content."""
        assert find_reasoning_span("Answer first. <think>not reasoning</think>") is None

    def test_paired_at_start(self):
        span = find_reasoning_span(THINK_OPEN + "reason" + THINK_CLOSE + "Ans")
        assert span == (len(THINK_OPEN), len(THINK_OPEN) + len("reason"), True)

    def test_leading_newline_after_open_tag_skipped(self):
        text = THINK_OPEN + "\nreason" + THINK_CLOSE
        start, end, closed = find_reasoning_span(text)
        assert text[start:end] == "reason"
        assert closed

    def test_unclosed_at_start(self):
        text = THINK_OPEN + "incomplete"
        start, end, closed = find_reasoning_span(text)
        assert not closed
        assert end == len(text)
        assert text[start:end] == "incomplete"

    def test_orphan_close_with_no_open_is_not_a_span(self):
        assert find_reasoning_span("preamble</think>answer") is None


class TestStripThinking:
    """Text-only helper (QSR_REASONING_MODE=strip path)."""

    def test_paired_stripped(self):
        r = strip_thinking(THINK_OPEN + "reason" + THINK_CLOSE + "Ans")
        assert r == "Ans"

    def test_unclosed_at_start_yields_empty(self):
        assert strip_thinking(THINK_OPEN + "incomplete") == ""

    def test_mid_body_literal_think_untouched(self):
        text = "The <think> tag marks reasoning; </think> closes it."
        assert strip_thinking(text) == text

    def test_fffd_cleanup_independent_of_think(self):
        assert strip_thinking("Hello" + chr(0xFFFD)) == "Hello"

    def test_usage_block_still_removed_anywhere(self):
        text = "Answer.<usage>tokens: 5</usage>"
        assert strip_usage_artifacts(text).strip() == "Answer."


# ---------------------------------------------------------------------------
# StreamProcessor: thinking_capable=False (Laguna's mode) -- position-0
# literal detection, streaming buffering, and boundary cases.
# ---------------------------------------------------------------------------


class TestStreamProcessorNonInjecting:
    def test_plain_answer_no_think_at_all(self):
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        p.add_tokens(_FakeTok.ids("Just an answer."))
        assert p.drain_thinking() == []
        assert p.drain_content() == ["Just an answer."]
        assert p.reasoning_content() is None

    def test_self_opened_think_block_split_from_content(self):
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        p.add_tokens(_FakeTok.ids(THINK_OPEN + "reasoning" + THINK_CLOSE + "answer"))
        assert p.drain_thinking() == ["reasoning"]
        assert p.drain_content() == ["answer"]
        assert p.reasoning_content() == "reasoning"
        assert p.finalize() == ("answer", [])

    def test_unclosed_think_at_max_tokens_yields_no_content(self):
        """Generation hits max_tokens mid-thought: the whole thing was
        reasoning, there is no visible content to show."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        p.add_tokens(_FakeTok.ids(THINK_OPEN + "still thinking when cut off"))
        assert p.drain_content() == []  # never resolves to visible content
        assert p.finalize() == ("", [])
        assert p.reasoning_content() == "still thinking when cut off"

    def test_orphan_close_tag_is_ordinary_content(self):
        """No matching <think> anywhere -- an orphan </think> is content,
        not a signal to delete the preceding text (the R4 bug)."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        text = "Some preamble" + THINK_CLOSE + "more text"
        p.add_tokens(_FakeTok.ids(text))
        assert p.drain_content() == [text]
        assert p.finalize() == (text, [])
        assert p.reasoning_content() is None

    def test_legitimate_mid_body_think_literal_not_truncated(self):
        """THE required regression: <think> discussed mid-answer must
        reach the client byte-for-byte, streamed incrementally too."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        text = (
            "Sure! The <think> tag opens a reasoning block and "
            "<think>like this</think> closes it with </think>."
        )
        emitted = []
        # Feed it in small chunks to also exercise the streaming diff path.
        for i in range(0, len(text), 5):
            p.add_tokens(_FakeTok.ids(text[i : i + 5]))
            emitted.extend(p.drain_content())
        assert "".join(emitted) == text
        assert p.reasoning_content() is None
        assert p.finalize() == (text, [])

    def test_streaming_ambiguous_prefix_resolves_when_not_a_think_tag(self):
        """Mid-stream, a partial "<th" is ambiguous (could still become
        "<think>") -- drain_content() must withhold it, not guess. Once
        enough tokens arrive to rule out "<think>", it must flush."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        p.add_tokens(_FakeTok.ids("<th"))
        assert p.drain_content() == []  # still ambiguous, withheld
        p.add_tokens(_FakeTok.ids("ose one"))  # "<those one" != <think> prefix
        assert p.drain_content() == ["<those one"]

    def test_streaming_ambiguous_prefix_resolves_when_it_is_a_think_tag(self):
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        p.add_tokens(_FakeTok.ids("<th"))
        assert p.drain_content() == []
        p.add_tokens(_FakeTok.ids("ink>reasoning</think>answer"))
        assert p.drain_content() == ["answer"]
        assert p.reasoning_content() == "reasoning"

    def test_finalize_resolves_ambiguous_prefix_that_never_completed(self):
        """Generation ends while still ambiguous (e.g. hit max_tokens=2
        right after "<t"): it never became a real <think> tag, so at
        finalize time it must be treated as ordinary (short) content."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        p.add_tokens(_FakeTok.ids("<t"))
        assert p.finalize() == ("<t", [])
        assert p.reasoning_content() is None


class TestStreamProcessorInjecting:
    """thinking_capable=True: template injects <think> into the PROMPT, so
    the model's own tokens start directly with the reasoning body (the
    opening tag is synthesized here, not present in generated text). No
    backend uses this today, but the mechanism must keep working since a
    future thinking-capable backend can opt in (docs/roadmap.md Track B)."""

    def test_open_tag_not_in_output_still_detected(self):
        p = StreamProcessor(_FakeTok(), thinking_capable=True)
        close = THINK_CLOSE
        p.add_tokens(_FakeTok.ids("reasoning body" + close + "the answer"))
        assert p.drain_thinking() == ["reasoning body"]
        assert p.finalize() == ("the answer", [])

    def test_content_text_matches_finalize(self):
        p = StreamProcessor(_FakeTok(), thinking_capable=True)
        p.add_tokens(_FakeTok.ids("r" + THINK_CLOSE + "answer"))
        visible_text, tools = p.finalize()
        assert p.content_text() == visible_text == "answer"
        assert tools == []


# ---------------------------------------------------------------------------
# Non-streaming re-use: content_text()/reasoning_content() give the same
# split as the incremental drain_thinking()/drain_content() would, when fed
# all tokens in one shot -- this is what server/app.py's non-streaming
# handlers rely on instead of re-implementing the split.
# ---------------------------------------------------------------------------


class TestNonStreamingReusesSameStateMachine:
    def test_all_at_once_matches_incremental(self):
        text = THINK_OPEN + "because X" + THINK_CLOSE + "the result is Y"

        incremental = StreamProcessor(_FakeTok(), thinking_capable=False)
        thinking_parts, content_parts = [], []
        for ch in text:
            incremental.add_tokens(_FakeTok.ids(ch))
            thinking_parts.extend(incremental.drain_thinking())
            content_parts.extend(incremental.drain_content())

        one_shot = StreamProcessor(_FakeTok(), thinking_capable=False)
        one_shot.add_tokens(_FakeTok.ids(text))

        assert "".join(thinking_parts) == one_shot.reasoning_content()
        assert "".join(content_parts) == one_shot.content_text()


# ---------------------------------------------------------------------------
# Response-shape wiring: reasoning_content placement per protocol.
# ---------------------------------------------------------------------------


class TestOpenAIReasoningContentWiring:
    def test_reasoning_content_attached_to_message(self):
        resp = openai_format.build_response(
            model="t",
            text="answer",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_content="because X",
        )
        assert resp["choices"][0]["message"]["reasoning_content"] == "because X"
        assert resp["choices"][0]["message"]["content"] == "answer"

    def test_no_reasoning_content_key_when_none(self):
        resp = openai_format.build_response(
            model="t", text="answer", finish_reason="stop", prompt_tokens=1, completion_tokens=1
        )
        assert "reasoning_content" not in resp["choices"][0]["message"]


class TestAnthropicReasoningContentWiring:
    def test_reasoning_content_is_a_top_level_field_not_a_content_block(self):
        """Must NOT show up as a `thinking` block in `content` -- see the
        build_response docstring for why (Claude Desktop compat, f13fd4a)."""
        resp = anthropic_format.build_response(
            model="t",
            text="answer",
            finish_reason="stop",
            input_tokens=1,
            output_tokens=1,
            reasoning_content="because X",
        )
        assert resp["reasoning_content"] == "because X"
        assert all(block.get("type") != "thinking" for block in resp["content"])
        assert resp["content"] == [{"type": "text", "text": "answer"}]

    def test_no_reasoning_content_key_when_none(self):
        resp = anthropic_format.build_response(
            model="t", text="answer", finish_reason="stop", input_tokens=1, output_tokens=1
        )
        assert "reasoning_content" not in resp


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestLagunaTemplateInjectsThink:
    """Laguna's chat template ends the prompt with ``<assistant><think>``.

    Captured from a live server on 2026-08-01: the decoded prompt ends
    ``...</user>\\n<assistant><think>`` and every completion begins with a
    bare ``</think>``. Generation therefore starts *inside* a think block,
    which is what ``thinking_capable=True`` means.

    With the flag off, no ``<think>`` sits at position 0 of the generated
    text, the anchored span rule finds no reasoning segment, and the orphan
    closing tag is served as the first characters of ``message.content`` --
    observed live as ``'</think>快速排序是...'``.
    """

    def test_bare_close_tag_is_reasoning_boundary_when_template_injects(self):
        from server.formats.stream import StreamProcessor

        proc = StreamProcessor(_FakeTok(), thinking_capable=True)
        proc.add_tokens(_FakeTok.ids("</think>Hello! How can I help you today?"))
        assert proc.content_text() == "Hello! How can I help you today?"
        assert "</think>" not in proc.content_text()

    def test_reasoning_text_before_close_tag_is_captured(self):
        from server.formats.stream import StreamProcessor

        proc = StreamProcessor(_FakeTok(), thinking_capable=True)
        proc.add_tokens(_FakeTok.ids("weighing options</think>the answer"))
        assert proc.content_text() == "the answer"
        assert proc.reasoning_content() == "weighing options"

    def test_server_default_matches_the_live_template(self):
        # server.app pulls in FastAPI (a `serving` extra); this module must
        # still import under dev extras alone -- see the CPU-only CI job.
        pytest.importorskip("fastapi")

        import server.app as app

        assert app.SERVER_THINKING_CAPABLE is True, (
            "Laguna's template injects <think>; with this off the orphan "
            "</think> leaks into message.content (observed live 2026-08-01)"
        )
