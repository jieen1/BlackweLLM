"""Scheduler integration tests for token-level thinking-budget forcing."""

from __future__ import annotations

from types import SimpleNamespace

from runtime.thinking_budget import ThinkingBudgetConfig, ThinkingBudgetState
from server.engine import ServerEngine


def _config(*, budget: int = 2) -> ThinkingBudgetConfig:
    return ThinkingBudgetConfig(budget=budget, start_token_ids=(10,), end_token_ids=(99,))


def test_plain_decode_builds_a_sparse_force_vector_for_mixed_batch():
    state = ThinkingBudgetState([10], _config())
    state.add_output([20, 21])

    engine = ServerEngine.__new__(ServerEngine)
    engine.active = {
        0: {"thinking_state": state},
        1: {"thinking_state": None},
    }

    assert engine._thinking_decode_kwargs([0, 1]) == {"force_token_ids": [99, None]}


def test_mtp_force_position_is_relative_to_the_next_verify_block():
    state = ThinkingBudgetState([10], _config(budget=3))
    state.add_output([20])

    engine = ServerEngine.__new__(ServerEngine)
    engine.K = 3
    engine.active = {0: {"thinking_state": state}}

    assert engine._thinking_mtp_kwargs([0]) == {
        "thinking_force_positions": {0: 2},
        "thinking_force_token_ids": {0: 99},
    }


def test_prefill_forces_an_exhausted_prompt_before_draft_generation():
    req = SimpleNamespace(prompt_ids=[10, 20], thinking_budget=_config(budget=1))

    assert ServerEngine._thinking_prefill_kwargs([(4, req)]) == {
        "force_token_ids": {4: 99}
    }


def test_prefill_leaves_unbudgeted_requests_on_the_legacy_path():
    req = SimpleNamespace(prompt_ids=[10, 20], thinking_budget=None)

    assert ServerEngine._thinking_prefill_kwargs([(4, req)]) == {}
