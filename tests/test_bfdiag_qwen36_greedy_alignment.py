"""CPU-only contracts for ``bfdiag.divergence.qwen36_greedy_alignment``.

Plain Python ints throughout -- no torch, no GPU, no model. This is the
decision logic the actual GPU run (never executed in this pass, see
``scripts/b1_verify_greedy_alignment.py``) will produce a
``GreedyAlignmentReport`` for; these tests lock in what "aligned" and
"passes the B1 gate" mean before that run ever happens.
"""

from __future__ import annotations

import json

from bfdiag.divergence.qwen36_greedy_alignment import (
    GreedyAlignmentReport,
    compare_greedy_token_ids,
    passes_b1_gate,
)


class TestCompareGreedyTokenIds:
    def test_identical_sequences_are_fully_aligned(self) -> None:
        result = compare_greedy_token_ids("w", [1, 2], [10, 20, 30], [10, 20, 30])
        assert result.fully_aligned
        assert result.first_divergence_index is None
        assert result.match_rate == 1.0
        assert result.num_tokens_compared == 3

    def test_divergence_at_first_mismatch(self) -> None:
        result = compare_greedy_token_ids("w", [1], [10, 20, 99, 40], [10, 20, 30, 40])
        assert not result.fully_aligned
        assert result.first_divergence_index == 2
        # Position 3 happens to match again after the divergence -- still
        # counts toward num_matched (this is a token-match-rate metric,
        # not "stop counting after the first miss").
        assert result.num_matched == 3
        assert result.num_tokens_compared == 4

    def test_unequal_lengths_compare_only_the_shared_prefix(self) -> None:
        result = compare_greedy_token_ids("w", [], [1, 2, 3], [1, 2])
        assert result.num_tokens_compared == 2
        assert result.fully_aligned

    def test_empty_generation_on_both_sides_is_trivially_aligned(self) -> None:
        result = compare_greedy_token_ids("w", [5], [], [])
        assert result.num_tokens_compared == 0
        assert result.fully_aligned
        assert result.match_rate == 0.0  # 0/0 defined as 0.0, not NaN/ZeroDivisionError

    def test_prompt_tokens_never_enter_the_comparison(self) -> None:
        # A prompt full of "divergent-looking" ids must not affect the
        # comparison at all -- only generated tokens are compared.
        result = compare_greedy_token_ids("w", [999, 888], [1, 2], [1, 2])
        assert result.fully_aligned
        assert result.prompt_token_ids == (999, 888)


class TestGreedyAlignmentReport:
    def test_overall_match_rate_is_token_weighted_not_workload_averaged(self) -> None:
        # Workload A: 1/2 matched. Workload B: 100/100 matched. A plain
        # mean of rates would give (0.5 + 1.0) / 2 = 0.75; token-weighted
        # gives (1 + 100) / (2 + 100) ~= 0.9902 -- the long workload should
        # dominate, not be diluted by the short one.
        a = compare_greedy_token_ids("a", [], [1, 2], [1, 99])
        b = compare_greedy_token_ids("b", [], list(range(100)), list(range(100)))
        report = GreedyAlignmentReport(workloads=(a, b))
        assert abs(report.overall_match_rate - (101 / 102)) < 1e-9

    def test_all_fully_aligned_requires_every_workload(self) -> None:
        good = compare_greedy_token_ids("good", [], [1, 2], [1, 2])
        bad = compare_greedy_token_ids("bad", [], [1, 2], [1, 9])
        assert GreedyAlignmentReport(workloads=(good,)).all_fully_aligned
        assert not GreedyAlignmentReport(workloads=(good, bad)).all_fully_aligned

    def test_round_trips_through_json(self, tmp_path) -> None:
        original = GreedyAlignmentReport(
            workloads=(compare_greedy_token_ids("w", [1], [2, 3], [2, 3]),)
        )
        path = tmp_path / "report.json"
        original.save(path)

        # File contents are plain, human-readable JSON -- not a pickle.
        raw = json.loads(path.read_text())
        assert raw["all_fully_aligned"] is True

        restored = GreedyAlignmentReport.load(path)
        assert restored.overall_match_rate == original.overall_match_rate
        assert restored.workloads[0].workload_name == "w"


class TestPassesB1Gate:
    def test_fails_with_fewer_than_the_required_workload_count(self) -> None:
        report = GreedyAlignmentReport(
            workloads=(compare_greedy_token_ids("only-one", [], [1] * 512, [1] * 512),)
        )
        passed, reason = passes_b1_gate(report, min_workloads=3, min_tokens_per_workload=512)
        assert not passed
        assert "only 1 workload" in reason

    def test_fails_if_any_workload_compared_fewer_than_512_tokens(self) -> None:
        long_ok = compare_greedy_token_ids("long", [], [1] * 512, [1] * 512)
        too_short = compare_greedy_token_ids("short", [], [1] * 100, [1] * 100)
        third = compare_greedy_token_ids("third", [], [1] * 512, [1] * 512)
        report = GreedyAlignmentReport(workloads=(long_ok, too_short, third))
        passed, reason = passes_b1_gate(report, min_workloads=3, min_tokens_per_workload=512)
        assert not passed
        assert "short (100)" in reason

    def test_fails_on_any_divergence_even_if_match_rate_is_high(self) -> None:
        # 511/512 matched (99.8%) still fails -- the gate is literal
        # token-for-token alignment, not a fuzzy threshold.
        near_perfect = [1] * 512
        oracle = [1] * 512
        oracle[500] = 2
        w1 = compare_greedy_token_ids("w1", [], near_perfect, oracle)
        w2 = compare_greedy_token_ids("w2", [], [1] * 512, [1] * 512)
        w3 = compare_greedy_token_ids("w3", [], [1] * 512, [1] * 512)
        report = GreedyAlignmentReport(workloads=(w1, w2, w3))
        passed, reason = passes_b1_gate(report)
        assert not passed
        assert "w1@token#500" in reason

    def test_passes_when_all_workloads_are_long_enough_and_fully_aligned(self) -> None:
        workloads = tuple(
            compare_greedy_token_ids(f"w{i}", [], [i] * 512, [i] * 512) for i in range(3)
        )
        report = GreedyAlignmentReport(workloads=workloads)
        passed, reason = passes_b1_gate(report)
        assert passed
        assert "all 3 workloads fully aligned" in reason
