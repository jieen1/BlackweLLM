"""Pure token-state tests for sampler-level thinking-budget forcing."""

from __future__ import annotations

from runtime.thinking_budget import ThinkingBudgetConfig, ThinkingBudgetState


def _config(*, budget: int = 3, end: str = "</think>") -> ThinkingBudgetConfig:
    return ThinkingBudgetConfig(
        budget=budget,
        start_token_ids=tuple(ord(char) for char in "<think>"),
        end_token_ids=tuple(ord(char) for char in end),
    )


def test_budget_force_is_relative_to_next_output_block():
    state = ThinkingBudgetState([ord(char) for char in "<think>"], _config(budget=2))

    assert state.force_for(1) is None
    state.add_output([ord("a")])
    assert state.force_for(1) is None
    state.add_output([ord("b")])
    assert state.force_for(1) == (0, ord("<"))


def test_mtp_force_position_accounts_for_tokens_before_the_boundary():
    state = ThinkingBudgetState([ord(char) for char in "<think>"], _config(budget=3))

    assert state.force_for(4) == (3, ord("<"))
    state.add_output([ord("a")])
    assert state.force_for(4) == (2, ord("<"))
    state.add_output([ord("b"), ord("c")])
    assert state.force_for(4) == (0, ord("<"))


def test_natural_close_disables_forcing_and_a_later_span_restarts_it():
    state = ThinkingBudgetState([], _config(budget=1))
    state.add_output([*map(ord, "<think>a</think>")])

    assert state.force_for(1) is None
    state.add_output([*map(ord, "<think>")])
    assert state.force_for(1) is None
    state.add_output([ord("x")])
    assert state.force_for(1) == (0, ord("<"))


def test_partial_multi_token_end_marker_is_completed_at_position_zero():
    state = ThinkingBudgetState([*map(ord, "<think>")], _config(budget=100))
    state.add_output([*map(ord, "</thi")])

    assert state.force_for(4) == (0, ord("n"))
    state.add_output([*map(ord, "nk>")])
    assert state.force_for(4) is None


def test_without_an_open_start_marker_no_token_is_forced():
    state = ThinkingBudgetState([*map(ord, "ordinary answer")], _config(budget=1))

    assert state.force_for(1) is None
