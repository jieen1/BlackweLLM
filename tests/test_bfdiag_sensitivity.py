"""CPU-only tests for the allocator-sensitivity probes.

The GPU-side measurement path is deliberately untested here (it needs
model weights); what is tested is the part that decides what a set of
measurements *means*, plus the perturbation grammar.
"""

from __future__ import annotations

import pytest

from bfdiag.sensitivity.measure import derive_shape
from bfdiag.sensitivity.perturbations import FORBIDDEN, build, parse
from bfdiag.sensitivity.verdict import Measurement, format_table, judge


def _m(pert: str, metric: float, out: str, **alloc) -> Measurement:
    base = {"reserved_mib": 72920.0, "segments": 471, "inactive_split_blocks": 85}
    base.update(alloc)
    return Measurement(perturbation=pert, metric=metric, output_hash=out, allocator=base)


# --- perturbation grammar --------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("none", ("none", None)),
        ("gc", ("gc", None)),
        ("gc+reset", ("gc+reset", None)),
        ("pad16", ("pad", 16)),
        ("holdpad256", ("holdpad", 256)),
    ],
)
def test_parse_known_names(name: str, expected: tuple) -> None:
    assert parse(name) == expected


def test_empty_cache_is_refused_with_the_reason() -> None:
    """It frees blocks captured CUDA Graphs still point at; the next
    replay dies with an illegal memory access (observed 2026-07-27)."""
    assert "empty_cache" in FORBIDDEN
    with pytest.raises(ValueError, match="illegal memory access"):
        parse("empty_cache")


def test_unknown_perturbation_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown perturbation"):
        parse("defragment")


def test_reset_requires_a_backend() -> None:
    with pytest.raises(ValueError, match="needs a backend"):
        build("reset")


def test_none_is_callable_without_a_backend() -> None:
    build("none")()  # must not raise


# --- load-time shape derivation -------------------------------------------


def test_derive_shape_matches_the_benchmark_formula() -> None:
    """Recomputed independently, not imported: this formula existing in
    two places is how a warm daemon came to compare blocks_per_slot=4096
    against a script's 130 and call the acceptance rates comparable."""
    ctx, bs, max_tokens, margin = 10240, 128, 256, 4096
    shape = derive_shape(ctx, bs, max_tokens=max_tokens)
    expected_mml = ctx + max_tokens + 2048
    expected_bps = -(-expected_mml // bs) + -(-margin // bs)
    assert shape.max_model_len == expected_mml == 12544
    assert shape.blocks_per_slot == expected_bps == 130


def test_derive_shape_at_block_size_64() -> None:
    shape = derive_shape(10240, 64)
    assert shape.blocks_per_slot == 196 + 64


# --- verdict ---------------------------------------------------------------


def test_identical_outputs_are_stable() -> None:
    """block_size=64's real behaviour: gc changed nothing, bit-identical."""
    v = judge([_m("none", 0.718182, "4fe0bef347b7"), _m("gc", 0.718182, "4fe0bef347b7")])
    assert v.stable
    assert v.distinct_outputs == ("4fe0bef347b7",)
    assert v.metric_spread == 0.0


def test_the_2026_07_27_block_size_128_sweep_is_sensitive() -> None:
    """The real measurements: three outputs on one prompt and one commit."""
    v = judge(
        [
            _m("none", 0.452525, "e5028b36258b"),
            _m("pad16", 0.452525, "e5028b36258b"),
            _m("gc", 0.675362, "d6e4833404d4", alloc_mib=72171.15),
            _m("pad256", 0.602564, "3cc5b31ad685", reserved_mib=73176.0, segments=472),
        ]
    )
    assert v.stable is False
    assert len(v.distinct_outputs) == 3
    assert v.metric_spread == pytest.approx(0.675362 - 0.452525)
    assert "SENSITIVE" in v.summary


def test_same_metric_different_tokens_is_still_a_failure() -> None:
    """Two token sequences that happen to score alike are not a match."""
    v = judge([_m("none", 0.5, "aaaa"), _m("gc", 0.5, "bbbb")])
    assert v.stable is False
    assert v.metric_spread == 0.0


def test_unexplained_flags_changes_with_identical_allocator_state() -> None:
    """A result that moves while allocator state is byte-identical cannot
    be blamed on layout -- that is genuine nondeterminism, and the
    verdict must say so rather than lumping it in."""
    v = judge([_m("none", 0.45, "aaaa"), _m("gc", 0.67, "bbbb")])
    assert v.unexplained == ("gc",)
    assert "not explainable by layout" in v.summary


def test_layout_change_is_not_flagged_as_unexplained() -> None:
    v = judge([_m("none", 0.45, "aaaa"), _m("pad256", 0.67, "bbbb", segments=472)])
    assert v.unexplained == ()


def test_format_table_includes_every_row_and_the_verdict() -> None:
    ms = [_m("none", 0.452525, "e5028b36258b"), _m("gc", 0.675362, "d6e4833404d4")]
    text = format_table(ms, judge(ms))
    assert "none" in text and "gc" in text
    assert "SENSITIVE" in text


def test_judge_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no measurements"):
        judge([])
