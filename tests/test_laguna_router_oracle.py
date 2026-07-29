from __future__ import annotations

import argparse

import pytest

from tools.laguna_router_oracle import (
    DEFAULT_FAMILIES,
    DEFAULT_ROWS,
    parse_families,
    parse_rows,
)


def test_default_router_oracle_matrix_covers_required_row_sizes() -> None:
    assert DEFAULT_ROWS == (0, 1, 2, 3, 4, 8, 16, 64, 8192)


def test_default_router_oracle_matrix_covers_nonfinite_and_zero_sum_inputs() -> None:
    assert {"nonfinite", "zero_sum", "near_tie", "signed_zero"} <= set(DEFAULT_FAMILIES)


def test_parse_rows_preserves_a_valid_explicit_matrix() -> None:
    assert parse_rows("0,1,16,8192") == (0, 1, 16, 8192)


@pytest.mark.parametrize("value", ("", "1,1", "-1", "one"))
def test_parse_rows_rejects_invalid_matrices(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_rows(value)


def test_parse_families_accepts_a_subset() -> None:
    assert parse_families("normal,zero_sum") == ("normal", "zero_sum")


@pytest.mark.parametrize("value", ("", "normal,normal", "normal,unknown"))
def test_parse_families_rejects_invalid_names(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_families(value)
