"""Request-level reasoning controls for Qwen3.8-compatible chat templates."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server.app import ChatCompletionRequest, _resolve_chat_template_kwargs


def test_openai_request_preserves_reasoning_effort() -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        reasoning_effort="low",
    )

    assert request.reasoning_effort == "low"
    assert _resolve_chat_template_kwargs(
        request.chat_template_kwargs,
        reasoning_effort=request.reasoning_effort,
    ) == {"reasoning_effort": "low", "enable_thinking": True}


def test_none_maps_to_qwen_hard_thinking_switch() -> None:
    assert _resolve_chat_template_kwargs(None, reasoning_effort="none") == {
        "enable_thinking": False
    }


def test_high_aliases_to_qwen_xhigh_template_value() -> None:
    assert _resolve_chat_template_kwargs(None, reasoning_effort="high") == {
        "reasoning_effort": "xhigh",
        "enable_thinking": True,
    }


def test_explicit_chat_template_kwargs_win() -> None:
    assert _resolve_chat_template_kwargs(
        {"enable_thinking": False},
        reasoning_effort="xhigh",
    ) == {"enable_thinking": False}
    assert _resolve_chat_template_kwargs(
        {"reasoning_effort": "low"},
        reasoning_effort="xhigh",
    ) == {"reasoning_effort": "low"}


@pytest.mark.parametrize("value", ["minimal", "max", "bogus", 2])
def test_invalid_reasoning_effort_is_rejected(value: object) -> None:
    with pytest.raises(HTTPException, match="reasoning_effort must be one of"):
        _resolve_chat_template_kwargs(None, reasoning_effort=value)  # type: ignore[arg-type]
