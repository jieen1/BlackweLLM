"""Request-level reasoning controls for Qwen3.8-compatible chat templates."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server.app import (
    ChatCompletionRequest,
    _resolve_chat_template_kwargs,
    _resolve_thinking_token_budget,
)


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


def test_nested_reasoning_and_root_enable_switch_are_not_dropped() -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        reasoning={"effort": "low"},
        enable_thinking=True,
    )

    assert _resolve_chat_template_kwargs(
        request.chat_template_kwargs,
        reasoning_effort=request.reasoning_effort,
        enable_thinking=request.enable_thinking,
        reasoning=request.reasoning,
    ) == {"enable_thinking": True, "reasoning_effort": "low"}


def test_non_native_template_gets_a_real_effort_budget() -> None:
    assert _resolve_thinking_token_budget(
        None,
        {"enable_thinking": True, "reasoning_effort": "low"},
        native_reasoning_effort=False,
        enable_effort_budget=True,
    ) == 4096


def test_nested_budget_overrides_the_effort_default() -> None:
    assert _resolve_thinking_token_budget(
        None,
        {"enable_thinking": True},
        reasoning={"effort": "low", "budget_tokens": 37},
        native_reasoning_effort=False,
    ) == 37


def test_service_default_budget_is_skipped_when_thinking_is_disabled() -> None:
    assert _resolve_thinking_token_budget(
        None,
        {"enable_thinking": False},
        default_budget=131072,
    ) is None


def test_implicit_budget_leaves_visible_output_headroom() -> None:
    assert _resolve_thinking_token_budget(
        None,
        {"enable_thinking": True},
        default_budget=8192,
        max_tokens=8192,
    ) == 4096


def test_explicit_budget_is_not_rewritten_by_completion_window() -> None:
    assert _resolve_thinking_token_budget(
        8192,
        {"enable_thinking": True},
        default_budget=4096,
        max_tokens=1024,
    ) == 8192


@pytest.mark.parametrize("value", ["minimal", "max", "bogus", 2])
def test_invalid_reasoning_effort_is_rejected(value: object) -> None:
    with pytest.raises(HTTPException, match="reasoning_effort must be one of"):
        _resolve_chat_template_kwargs(None, reasoning_effort=value)  # type: ignore[arg-type]
