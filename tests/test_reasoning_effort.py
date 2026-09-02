"""Request-level reasoning controls for Qwen3.8-compatible chat templates."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server.app import (
    ChatCompletionRequest,
    _build_sampling_params,
    _is_opencode_title_request,
    _resolve_chat_template_kwargs,
    _resolve_engine_chat_template_kwargs,
    _resolve_thinking_token_budget,
    _sampling_defaults_for_request,
)
from server.formats.thinking import apply_qwen_default_reasoning_effort


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


def test_openai_request_accepts_camel_case_effort_alias() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "reasoningEffort": "low",
        }
    )

    assert request.reasoning_effort == "low"


def test_opencode_title_request_is_detected_without_changing_completion_window() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a title generator. You output ONLY a thread title."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Generate a title for this conversation:\n"
                        "\"Check the service\""
                    ),
                },
            ],
            "max_tokens": 8192,
            "reasoning_effort": "medium",
        }
    )

    assert _is_opencode_title_request(request)
    assert request.max_tokens == 8192


def test_opencode_title_detection_does_not_match_tool_requests() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a title generator. Output only a title.",
                },
                {
                    "role": "user",
                    "content": "Generate a title for this conversation:",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {}},
                }
            ],
        }
    )

    assert not _is_opencode_title_request(request)


def test_none_maps_to_qwen_hard_thinking_switch() -> None:
    assert _resolve_chat_template_kwargs(None, reasoning_effort="none") == {
        "enable_thinking": False
    }


def test_high_is_not_a_runtime_effort_level() -> None:
    with pytest.raises(HTTPException, match="reasoning_effort must be one of"):
        _resolve_chat_template_kwargs(None, reasoning_effort="high")


@pytest.mark.parametrize("value", ["high", "xhigh", "max"])
def test_qwen_downgrades_unsupported_high_effort_to_medium(value: str) -> None:
    engine = type("Engine", (), {"backend_name": "qwen36"})()

    assert _resolve_engine_chat_template_kwargs(
        engine,
        None,
        reasoning_effort=value,
    ) == {"reasoning_effort": "medium", "enable_thinking": True}


def test_qwen_downgrades_explicit_template_high_effort_to_medium() -> None:
    engine = type("Engine", (), {"backend_name": "qwen36"})()

    assert _resolve_engine_chat_template_kwargs(
        engine,
        {"reasoning_effort": "xhigh"},
    ) == {"reasoning_effort": "medium"}


@pytest.mark.parametrize(
    ("requested", "effective"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "xhigh"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
    ],
)
def test_flashnext_accepts_opencode_effort_aliases(
    requested: str, effective: str
) -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()

    assert _resolve_engine_chat_template_kwargs(
        engine,
        None,
        reasoning_effort=requested,
    ) == {
        "preserve_thinking": False,
        "reasoning_effort": effective,
        "enable_thinking": True,
    }


def test_flashnext_normalizes_nested_reasoning_effort() -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()

    assert _resolve_engine_chat_template_kwargs(
        engine,
        None,
        reasoning={"effort": "high"},
        thinking={"level": "minimal"},
    ) == {
        "preserve_thinking": False,
        "reasoning_effort": "xhigh",
        "enable_thinking": True,
    }


def test_flashnext_explicit_template_alias_is_normalized() -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()

    assert _resolve_engine_chat_template_kwargs(
        engine,
        {"reasoning_effort": "high"},
    ) == {"reasoning_effort": "xhigh", "preserve_thinking": False}


def test_flashnext_drops_historical_reasoning_by_default() -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()

    assert _resolve_engine_chat_template_kwargs(engine, None) == {
        "preserve_thinking": False
    }


def test_flashnext_preserve_thinking_is_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()
    monkeypatch.setenv("QSR_FLASHNEXT_PRESERVE_THINKING", "1")

    assert _resolve_engine_chat_template_kwargs(engine, None) == {
        "preserve_thinking": True
    }


def test_flashnext_sampler_defaults_follow_template_thinking_mode() -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()

    # Flash-Next operates in thinking mode by default.
    assert _sampling_defaults_for_request(engine, None) == (1.0, 0.95, 20)
    assert _sampling_defaults_for_request(engine, {"enable_thinking": True}) == (
        1.0,
        0.95,
        20,
    )
    # ``reasoning_effort=none`` resolves to this explicit template switch.
    assert _sampling_defaults_for_request(engine, {"enable_thinking": False}) == (
        0.7,
        0.80,
        20,
    )


def test_flashnext_sampler_defaults_are_applied_only_when_fields_are_omitted() -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()
    defaults = _sampling_defaults_for_request(engine, None)

    params = _build_sampling_params(defaults=defaults)
    assert (params.temperature, params.top_p, params.top_k) == (1.0, 0.95, 20)

    # Explicit values remain independent request-level overrides, including
    # the deliberate temperature=0 greedy escape hatch.
    override = _build_sampling_params(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        defaults=defaults,
    )
    assert (override.temperature, override.top_p, override.top_k) == (0.0, 1.0, 0)

    partial = _build_sampling_params(temperature=0.35, defaults=defaults)
    assert (partial.temperature, partial.top_p, partial.top_k) == (0.35, 0.95, 20)


def test_non_flashnext_sampler_defaults_remain_legacy_greedy() -> None:
    engine = type("Engine", (), {"backend_name": "qwen36"})()

    assert _sampling_defaults_for_request(engine, None) == (0.0, 1.0, 0)
    params = _build_sampling_params(defaults=_sampling_defaults_for_request(engine, None))
    assert params.is_greedy


def test_flashnext_explicit_preserve_thinking_wins() -> None:
    engine = type("Engine", (), {"backend_name": "flashnext"})()

    assert _resolve_engine_chat_template_kwargs(
        engine,
        {"preserve_thinking": True},
        reasoning_effort="medium",
    ) == {
        "preserve_thinking": True,
        "reasoning_effort": "medium",
        "enable_thinking": True,
    }


def test_unspecified_effort_leaves_request_kwargs_unchanged() -> None:
    assert _resolve_chat_template_kwargs(None) is None


def test_service_default_changes_template_only(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = type("Tokenizer", (), {})()
    tokenizer.chat_template = "{{ reasoning_effort|default('xhigh') }}"
    monkeypatch.setenv("QSR_DEFAULT_REASONING_EFFORT", "medium")

    assert apply_qwen_default_reasoning_effort(tokenizer) == "medium"
    assert "reasoning_effort|default('medium')" in tokenizer.chat_template


def test_explicit_effort_remains_request_override() -> None:
    assert _resolve_chat_template_kwargs(None, reasoning_effort="low") == {
        "reasoning_effort": "low",
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


def test_reasoning_effort_never_synthesizes_a_thinking_budget() -> None:
    assert _resolve_thinking_token_budget(
        None,
        {"enable_thinking": True, "reasoning_effort": "low"},
    ) is None


def test_nested_budget_overrides_the_effort_default() -> None:
    assert _resolve_thinking_token_budget(
        None,
        {"enable_thinking": True},
        reasoning={"effort": "low", "budget_tokens": 37},
    ) == 37


def test_explicit_budget_is_not_rewritten_by_reasoning_effort() -> None:
    assert _resolve_thinking_token_budget(
        8192,
        {"enable_thinking": True},
    ) == 8192


@pytest.mark.parametrize("value", ["minimal", "high", "max", "bogus", 2])
def test_invalid_reasoning_effort_is_rejected(value: object) -> None:
    with pytest.raises(HTTPException, match="reasoning_effort must be one of"):
        _resolve_chat_template_kwargs(None, reasoning_effort=value)  # type: ignore[arg-type]
