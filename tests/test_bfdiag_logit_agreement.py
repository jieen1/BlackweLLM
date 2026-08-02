"""CPU-only tests for the B1-R step-locked logit-agreement metrics.

No torch, no GPU, no model -- the module under test is deliberately pure
(see its docstring), so every threshold rule, the ULP conversion, and the
``tie_slack <= gap_error`` identity the whole criterion rests on can be
checked exhaustively here.
"""

from __future__ import annotations

import math
import random

import pytest

from bfdiag.divergence.logit_agreement import (
    UNCALIBRATED_THRESHOLDS,
    AgreementReport,
    AgreementThresholds,
    StepAgreement,
    WorkloadAgreement,
    bf16_ulp,
    compare_step,
    missing_from_intersection,
    passes_b1r_gate,
)

# --------------------------------------------------------------------------
# bf16_ulp
# --------------------------------------------------------------------------


def test_bf16_ulp_matches_the_measured_b1_value() -> None:
    """The B1 note recorded ULP = 0.0625 for logits in [8, 16). That is the
    one value in this repository measured against real divergences, so it
    is the calibration point for this function."""
    assert bf16_ulp(8.0) == pytest.approx(0.0625)
    assert bf16_ulp(13.7) == pytest.approx(0.0625)
    assert bf16_ulp(15.99) == pytest.approx(0.0625)


@pytest.mark.parametrize(
    ("magnitude", "expected"),
    [
        (1.0, 2.0**-7),
        (2.0, 2.0**-6),
        (4.0, 2.0**-5),
        (16.0, 0.125),
        (32.0, 0.25),
        (0.5, 2.0**-8),
    ],
)
def test_bf16_ulp_doubles_every_binade(magnitude: float, expected: float) -> None:
    assert bf16_ulp(magnitude) == pytest.approx(expected)


def test_bf16_ulp_is_sign_insensitive_and_finite_at_zero() -> None:
    assert bf16_ulp(-13.7) == bf16_ulp(13.7)
    assert bf16_ulp(0.0) > 0.0
    assert math.isfinite(bf16_ulp(0.0))
    assert math.isfinite(bf16_ulp(float("inf")))


# --------------------------------------------------------------------------
# compare_step
# --------------------------------------------------------------------------


def test_identical_logits_give_zero_everything() -> None:
    logits = {5: 12.0, 7: 11.5, 9: 3.0}
    step = compare_step(0, 5, logits, dict(logits))
    assert step.agrees
    assert step.gap_error == pytest.approx(0.0)
    assert step.tie_slack == pytest.approx(0.0)
    assert step.kl_topk == pytest.approx(0.0)
    assert step.mine_top1 == step.oracle_top1 == 5


def test_constant_offset_is_invisible_by_construction() -> None:
    """A constant added to every logit changes neither argmax nor softmax,
    so the metric must read it as zero disagreement -- this is the
    documented, intentional blind spot, pinned here so nobody "fixes" it."""
    oracle = {1: 10.0, 2: 9.0, 3: 8.0}
    mine = {t: v + 4.25 for t, v in oracle.items()}
    step = compare_step(0, 1, mine, oracle)
    assert step.gap_error == pytest.approx(0.0)
    assert step.kl_topk == pytest.approx(0.0)


def test_multiplicative_scale_is_visible() -> None:
    """Unlike a constant offset, a scale change does alter relative gaps
    (and therefore softmax), so the metric must NOT be blind to it."""
    oracle = {1: 10.0, 2: 9.0, 3: 8.0}
    mine = {t: v * 1.1 for t, v in oracle.items()}
    step = compare_step(0, 1, mine, oracle)
    assert step.gap_error > 0.19
    assert step.agrees  # ranking preserved, yet the disagreement is measured


def test_near_tie_flip_reports_small_tie_slack_in_ulps() -> None:
    """Reproduces the shape of the real B1 finding: two candidates one bf16
    ULP apart, the two sides ranking them oppositely."""
    oracle = {1007: 13.0625, 13901: 13.0}
    mine = {1007: 13.0, 13901: 13.0625}
    step = compare_step(0, 1007, mine, oracle)
    assert not step.agrees
    assert step.mine_top1 == 13901
    assert step.oracle_top1 == 1007
    assert step.tie_slack == pytest.approx(0.125)
    assert step.tie_slack_ulps == pytest.approx(2.0)


def test_confident_disagreement_reports_large_tie_slack() -> None:
    """The signal a real bug produces: the oracle is confident and we chose
    something else entirely."""
    oracle = {1007: 20.0, 13901: 5.0}
    mine = {1007: 5.0, 13901: 20.0}
    step = compare_step(0, 1007, mine, oracle)
    assert not step.agrees
    assert step.tie_slack == pytest.approx(30.0)
    assert step.tie_slack_ulps > 100.0


def test_tie_slack_never_exceeds_gap_error_on_random_data() -> None:
    """The identity the criterion is built on: ``0 <= tie_slack <=
    gap_error`` at every step. Proven algebraically in the module
    docstring; checked here against randomised inputs so a future change
    to either formula cannot quietly break it."""
    rng = random.Random(20260802)
    for _ in range(2000):
        tokens = rng.sample(range(1000), 12)
        oracle = {t: rng.uniform(-20.0, 20.0) for t in tokens}
        mine = {t: oracle[t] + rng.gauss(0.0, 0.4) for t in tokens}
        step = compare_step(0, tokens[0], mine, oracle)
        assert step.tie_slack >= -1e-9
        assert step.tie_slack <= step.gap_error + 1e-9


def test_gap_error_is_zero_iff_relative_gaps_match() -> None:
    rng = random.Random(7)
    tokens = list(range(10))
    oracle = {t: rng.uniform(-5.0, 5.0) for t in tokens}
    mine = {t: oracle[t] - 3.5 for t in tokens}
    assert compare_step(0, 0, mine, oracle).gap_error == pytest.approx(0.0)
    mine[tokens[3]] += 0.75
    assert compare_step(0, 0, mine, oracle).gap_error == pytest.approx(0.75)


def test_gap_top_k_limits_which_tokens_can_contribute() -> None:
    """A large error on a token far outside both sides' plausible set must
    not dominate the metric -- it cannot flip an argmax."""
    oracle = {1: 10.0, 2: 9.0, 3: 8.0, 4: 7.0, 5: -50.0}
    mine = dict(oracle)
    mine[5] = -10.0  # 40.0 of error, but token 5 is nowhere near the top
    assert compare_step(0, 1, mine, oracle, gap_top_k=4).gap_error == pytest.approx(0.0)
    assert compare_step(0, 1, mine, oracle, gap_top_k=5).gap_error == pytest.approx(40.0)


def test_our_argmax_outside_the_oracle_slice_is_unbounded_not_passing() -> None:
    """If we pick a token the oracle did not even rank, no finite margin
    exists; the metric must not silently report a small number."""
    oracle = {1: 10.0, 2: 9.0}
    mine = {1: 5.0, 2: 4.0, 99: 30.0}
    step = compare_step(0, 1, mine, oracle)
    assert step.mine_top1 == 99
    assert math.isinf(step.tie_slack)
    assert math.isinf(step.tie_slack_ulps)


def test_oracle_argmax_absent_from_our_capture_is_an_error() -> None:
    with pytest.raises(ValueError, match="oracle's own argmax is absent"):
        compare_step(0, 1, {5: 1.0}, {1: 10.0, 5: 1.0})


def test_empty_side_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compare_step(0, 1, {}, {1: 1.0})


def test_kl_is_asymmetric_and_charges_the_candidate() -> None:
    oracle = {1: 10.0, 2: 0.0}
    mine = {1: 0.0, 2: 10.0}
    assert compare_step(0, 1, mine, oracle).kl_topk > 5.0


def test_missing_from_intersection_flags_capture_width_problems() -> None:
    mine = {1: 10.0, 2: 9.0, 42: 8.0}
    oracle = {1: 10.0, 2: 9.0, 77: 8.5}
    assert missing_from_intersection(mine, oracle, top_k=3) == (42, 77)
    assert missing_from_intersection(mine, oracle, top_k=2) == ()


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def _step(
    index: int,
    *,
    gap: float,
    agrees: bool = True,
    slack: float = 0.0,
    kl: float = 0.0,
    scale: float = 8.0,
    logprob_error: float = 0.0,
) -> StepAgreement:
    return StepAgreement(
        step_index=index,
        forced_token_id=1,
        mine_top1=1 if agrees else 2,
        oracle_top1=1,
        gap_error=gap,
        mine_margin=slack / 2,
        oracle_margin=slack / 2,
        tie_slack=slack,
        logit_scale=scale,
        kl_topk=kl,
        logprob_error=logprob_error,
    )


def _workload(name: str, steps: list[StepAgreement]) -> WorkloadAgreement:
    return WorkloadAgreement(workload_name=name, prompt_token_ids=(1, 2), steps=tuple(steps))


def test_workload_summary_statistics() -> None:
    steps = [_step(i, gap=0.1) for i in range(99)] + [_step(99, gap=5.0, agrees=False, slack=1.0)]
    w = _workload("w", steps)
    assert w.num_steps == 100
    assert w.max_gap_error == pytest.approx(5.0)
    assert w.p99_gap_error == pytest.approx(0.1)
    assert w.num_disagreements == 1
    assert w.disagreement_rate == pytest.approx(0.01)
    assert w.max_tie_slack_ulps == pytest.approx(1.0 / 0.0625)


def test_drift_ratio_is_one_without_enough_steps() -> None:
    assert _workload("w", [_step(i, gap=1.0) for i in range(10)]).drift_ratio(window=128) == 1.0


def test_drift_ratio_detects_a_growing_error() -> None:
    early = [_step(i, gap=0.1) for i in range(128)]
    late = [_step(128 + i, gap=0.8) for i in range(128)]
    assert _workload("w", early + late).drift_ratio(window=128) == pytest.approx(8.0)


def test_drift_ratio_is_flat_for_stationary_noise() -> None:
    rng = random.Random(3)
    steps = [_step(i, gap=abs(rng.gauss(0.0, 0.1))) for i in range(512)]
    assert 0.5 < _workload("w", steps).drift_ratio(window=128) < 2.0


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def _report(*workloads: WorkloadAgreement) -> AgreementReport:
    return AgreementReport(workloads=tuple(workloads))


def _clean_workload(name: str, n: int = 512, gap: float = 0.05) -> WorkloadAgreement:
    return _workload(name, [_step(i, gap=gap) for i in range(n)])


def test_gate_passes_a_clean_run() -> None:
    report = _report(*(_clean_workload(f"w{i}") for i in range(3)))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert passed, reasons
    assert reasons == ()


def test_gate_requires_three_workloads() -> None:
    passed, reasons = passes_b1r_gate(_report(_clean_workload("w0")), UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert any("workload" in r for r in reasons)


def test_gate_requires_the_full_step_count() -> None:
    report = _report(_clean_workload("w0", n=100), _clean_workload("w1"), _clean_workload("w2"))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert any("fewer than 512 steps" in r for r in reasons)


def test_gate_fails_on_a_single_confident_disagreement() -> None:
    """The whole point: one flip is not a failure, but one flip whose
    combined margin is far above the bf16 noise floor is."""
    steps = [_step(i, gap=0.05) for i in range(511)]
    steps.append(_step(511, gap=30.0, agrees=False, slack=30.0))
    report = _report(_workload("w0", steps), _clean_workload("w1"), _clean_workload("w2"))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert any("max_tie_slack_ulps" in r for r in reasons)
    assert any("max_gap_error" in r for r in reasons)


def test_gate_tolerates_a_one_ulp_flip() -> None:
    """The exact situation that made the literal B1 gate unreachable must
    now pass."""
    steps = [_step(i, gap=0.05) for i in range(511)]
    steps.append(_step(511, gap=0.0625, agrees=False, slack=0.0625))
    report = _report(_workload("w0", steps), _clean_workload("w1"), _clean_workload("w2"))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert passed, reasons


def test_gate_fails_on_too_many_near_tie_flips() -> None:
    """A systematically biased implementation can keep every individual
    flip inside the tie window while flipping far more often than noise
    would -- the disagreement-rate bar exists for exactly that."""
    steps = [
        _step(i, gap=0.06, agrees=(i % 5 != 0), slack=0.0625 if i % 5 == 0 else 0.0)
        for i in range(512)
    ]
    report = _report(_workload("w0", steps), _clean_workload("w1"), _clean_workload("w2"))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert any("disagreement_rate" in r for r in reasons)


def test_gate_fails_on_distribution_shift_without_any_flip() -> None:
    """A bug that shifts probability mass but never flips an argmax is
    invisible to token comparison and must be caught by KL."""
    steps = [_step(i, gap=0.4, kl=0.2) for i in range(512)]
    report = _report(_workload("w0", steps), _clean_workload("w1"), _clean_workload("w2"))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert any("mean_kl_topk" in r for r in reasons)


def test_gate_fails_on_accumulating_drift() -> None:
    early = [_step(i, gap=0.02) for i in range(256)]
    late = [_step(256 + i, gap=0.4) for i in range(256)]
    report = _report(_workload("w0", early + late), _clean_workload("w1"), _clean_workload("w2"))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert any("max_drift_ratio" in r for r in reasons)


def test_gate_reports_every_violated_bar_not_just_the_first() -> None:
    steps = [_step(i, gap=9.0, agrees=False, slack=9.0, kl=1.0) for i in range(512)]
    report = _report(_workload("w0", steps), _workload("w1", steps), _workload("w2", steps))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert not passed
    assert len(reasons) >= 4


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def test_report_round_trips_through_json(tmp_path) -> None:
    report = AgreementReport(
        workloads=(_clean_workload("w0", n=4),),
        metadata={"commit": "abc123", "injection": "none"},
    )
    path = tmp_path / "runs" / "agreement.json"
    report.save(path)
    loaded = AgreementReport.load(path)
    assert loaded.metadata == {"commit": "abc123", "injection": "none"}
    assert loaded.workloads[0].workload_name == "w0"
    assert loaded.workloads[0].num_steps == 4
    assert loaded.max_gap_error == pytest.approx(report.max_gap_error)


def test_logprob_error_needs_both_logsumexps() -> None:
    logits = {5: 12.0, 7: 11.5}
    assert math.isnan(compare_step(0, 5, logits, dict(logits)).logprob_error)
    assert math.isnan(
        compare_step(0, 5, logits, dict(logits), mine_logsumexp=12.5).logprob_error
    )


def test_logprob_error_is_shift_invariant_like_log_softmax() -> None:
    """log_softmax subtracts each side's own logsumexp, so a constant
    offset on one side must cancel exactly."""
    oracle = {1: 10.0, 2: 9.0}
    mine = {t: v + 3.0 for t, v in oracle.items()}
    step = compare_step(
        0, 1, mine, oracle, mine_logsumexp=13.0, oracle_logsumexp=10.0
    )
    assert step.logprob_error == pytest.approx(0.0)


def test_logprob_error_measures_relative_movement() -> None:
    oracle = {1: 10.0, 2: 9.0}
    mine = {1: 10.0, 2: 9.05}
    step = compare_step(0, 1, mine, oracle, mine_logsumexp=0.0, oracle_logsumexp=0.0)
    assert step.logprob_error == pytest.approx(0.05)


def test_gate_ignores_logprob_error_when_ungated() -> None:
    steps = [_step(i, gap=0.05, logprob_error=float("nan")) for i in range(512)]
    report = _report(*(_workload(f"w{i}", steps) for i in range(3)))
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    assert passed, reasons


def test_gate_fails_when_logprob_error_is_gated_but_unmeasured() -> None:
    """A bar that silently passes because nobody measured it is exactly the
    kind of dead gate e2e-and-quality-plan.md's C8 audit exists to find."""
    thresholds = AgreementThresholds.from_dict(
        {**UNCALIBRATED_THRESHOLDS.to_dict(), "max_logprob_error": 0.06}
    )
    steps = [_step(i, gap=0.05, logprob_error=float("nan")) for i in range(512)]
    report = _report(*(_workload(f"w{i}", steps) for i in range(3)))
    passed, reasons = passes_b1r_gate(report, thresholds)
    assert not passed
    assert any("not measured" in r for r in reasons)


def test_gate_checks_logprob_error_against_the_sglang_comparable_bar() -> None:
    thresholds = AgreementThresholds.from_dict(
        {**UNCALIBRATED_THRESHOLDS.to_dict(), "max_logprob_error": 0.06}
    )
    ok = [_step(i, gap=0.05, logprob_error=0.02) for i in range(512)]
    bad = ok[:-1] + [_step(511, gap=0.05, logprob_error=0.5)]
    assert passes_b1r_gate(_report(*(_workload(f"w{i}", ok) for i in range(3))), thresholds)[0]
    passed, reasons = passes_b1r_gate(
        _report(_workload("w0", bad), _workload("w1", ok), _workload("w2", ok)), thresholds
    )
    assert not passed
    assert any("max_logprob_error" in r for r in reasons)


def test_thresholds_round_trip() -> None:
    restored = AgreementThresholds.from_dict(UNCALIBRATED_THRESHOLDS.to_dict())
    assert restored == UNCALIBRATED_THRESHOLDS
