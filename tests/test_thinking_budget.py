"""HTTP adapter coverage for the low-level thinking-budget contract."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from runtime.thinking_budget import ThinkingBudgetConfig
from server.app import (
    ChatCompletionRequest,
    _resolve_thinking_token_budget,
    _submit_stream_with_thinking_budget,
    _submit_with_thinking_budget,
)
from server.formats.stream import StreamProcessor


class _CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    @staticmethod
    def ids(text: str) -> list[int]:
        return [ord(char) for char in text]


def _budget(tok: _CharTokenizer) -> ThinkingBudgetConfig:
    return ThinkingBudgetConfig(
        budget=8,
        start_token_ids=tuple(tok.ids("<think>")),
        end_token_ids=tuple(tok.ids("</think>")),
    )


class _SingleStageEngine:
    def __init__(self, tokenizer: _CharTokenizer):
        self.tok = tokenizer
        self.calls: list[tuple[list[int], int, dict]] = []

    async def submit(self, prompt_ids, max_tokens, **kwargs):
        self.calls.append((list(prompt_ids), max_tokens, kwargs))
        ids = self.tok.ids("unfinished reasoning</think>answer")
        return {
            "committed_token_ids": ids,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(ids),
            "finish_reason": "stop",
            "logprobs": [{"token_id": token_id} for token_id in ids],
        }

    async def submit_stream(self, prompt_ids, max_tokens, **kwargs):
        self.calls.append((list(prompt_ids), max_tokens, kwargs))
        ids = self.tok.ids("unfinished reasoning</think>answer")
        yield ids[:8]
        yield ids[8:]
        yield {
            "committed_token_ids": ids,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(ids),
            "finish_reason": "stop",
        }


def test_non_streaming_budget_is_one_submit_and_merges_usage():
    async def run():
        tok = _CharTokenizer()
        engine = _SingleStageEngine(tok)
        proc = StreamProcessor(tok, thinking_capable=True)
        prompt_ids = tok.ids("prompt")
        budget = _budget(tok)

        result = await _submit_with_thinking_budget(
            engine,
            prompt_ids,
            32,
            thinking_budget=budget,
            processor=proc,
            logprobs=True,
        )

        assert len(engine.calls) == 1
        assert engine.calls[0][0] == prompt_ids
        assert engine.calls[0][1] == 32
        assert engine.calls[0][2]["thinking_budget"] == budget
        assert result["committed_token_ids"] == tok.ids("unfinished reasoning</think>answer")
        assert result["completion_tokens"] == len(result["committed_token_ids"])
        assert len(result["logprobs"]) == result["completion_tokens"]
        assert proc.reasoning_content() == "unfinished reasoning"
        assert proc.content_text() == "answer"

    asyncio.run(run())


def test_streaming_budget_keeps_one_client_stream_and_forwards_config():
    async def run():
        tok = _CharTokenizer()
        engine = _SingleStageEngine(tok)
        proc = StreamProcessor(tok, thinking_capable=True)
        items = []

        async for item in _submit_stream_with_thinking_budget(
            engine,
            tok.ids("prompt"),
            32,
            thinking_budget=_budget(tok),
            processor=proc,
        ):
            items.append(item)

        token_items = [item for item in items if not isinstance(item, dict)]
        final = items[-1]
        assert b"".join(bytes(item) for item in token_items).decode() == (
            "unfinished reasoning</think>answer"
        )
        assert final["finish_reason"] == "stop"
        assert final["completion_tokens"] == len("unfinished reasoning</think>answer")
        assert engine.calls[0][2]["thinking_budget"] == _budget(tok)
        assert proc.reasoning_content() == "unfinished reasoning"
        assert proc.content_text() == "answer"

    asyncio.run(run())


def test_budget_validation_and_request_field():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "2+2"}],
        thinking_token_budget=256,
    )
    assert request.thinking_token_budget == 256
    assert _resolve_thinking_token_budget(256, {"enable_thinking": True}) == 256

    with pytest.raises(HTTPException, match="positive integer"):
        _resolve_thinking_token_budget(0, None)
    with pytest.raises(HTTPException, match="requires thinking mode"):
        _resolve_thinking_token_budget(256, {"enable_thinking": False})


def test_internal_budget_tokens_are_not_public_reasoning():
    tok = _CharTokenizer()
    proc = StreamProcessor(tok, thinking_capable=True)
    proc.add_tokens(tok.ids("thought"))
    assert proc.has_unclosed_thinking()
    proc.add_internal_tokens(tok.ids(" injected</think>\n"))
    proc.drain_content()
    proc.add_tokens(tok.ids("answer"))

    assert proc.reasoning_content() == "thought"
    assert proc.content_text() == "answer"
