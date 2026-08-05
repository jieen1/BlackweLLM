"""Does the B1-R gate actually go red? Replayed from a recorded sweep.

``docs/e2e-and-quality-plan.md`` §4 makes it a standing rule that every
new gate ships with a method for proving it can fail, and lists three
acceptable ones. B1-R uses the third: known-bad-input replay. On
2026-08-02 seventeen configurations -- one control plus sixteen
deliberately injected bugs (``bfdiag/divergence/qwen36_bug_injection.py``)
-- were measured against the real 27B checkpoint and the HF reference.
Their summary metrics are checked in as
``tests/fixtures/b1_injection_sweep_2026-08-02.json`` and replayed here
through ``evaluate_summary`` -- the same function a live run calls.

That makes the calibration falsifiable on a CPU: widen any bar past the
point where it separates the control from the weakest detected bug, and
this test fails without a GPU, a 27B checkpoint or a 40-minute sweep.

The three configurations in ``EXPECTED_PASS`` that are *not* the control
are there on purpose and are documented individually below. Two of them
are provably no-ops rather than misses; the third measures the criterion's
detection floor. Deleting them would make this file look tidier and would
throw away the only evidence about where the floor actually is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bfdiag.divergence.logit_agreement import (
    CALIBRATED_THRESHOLDS,
    AgreementThresholds,
    evaluate_summary,
)

FIXTURE = Path(__file__).parent / "fixtures" / "b1_injection_sweep_2026-08-02.json"

#: Must pass. The control, plus configurations measured to be at or below
#: the noise floor -- each with its reason, because "the gate did not fire"
#: is only acceptable when it is explained.
EXPECTED_PASS: dict[str, str] = {
    "none": "control -- an unmodified run must not trip its own gate",
    # RoPE scores depend only on the *relative* offset i-j, so adding the
    # same constant to every position leaves every q_i . k_j unchanged.
    # This is a specificity check, not a miss: a criterion that reddened
    # here would be reporting a mathematical no-op as a defect.
    "rope-positions-offset:1": "exact no-op: RoPE is relative to i-j",
    # rope_theta only enters through inv_freq = theta**(-2k/rotary_dim);
    # at <=300 positions a 1% change moves the phase by well under a
    # radian in every rotated dimension. Genuinely below the floor at B1's
    # context length -- and it would grow with context, so B2/B3 at 128K
    # should not assume this stays true.
    "rope-theta-rel:1e-05": "below the noise floor at <=300 positions",
    "rope-theta-rel:0.0001": "below the noise floor at <=300 positions",
    "rope-theta-rel:0.001": "below the noise floor at <=300 positions",
    "rope-theta-rel:0.01": "below the noise floor at <=300 positions",
    # The GDN recurrent state is bf16, whose ULP near 1.0 is 2**-8 =
    # 0.0039. Multiplying by 0.999 or 0.9999 rounds straight back to 1.0,
    # so these two configurations are bit-identical to the control -- the
    # format cannot represent the bug, let alone the criterion detect it.
    "gdn-state-decay:0.0001": "bf16 rounds the decay factor to exactly 1.0",
    "gdn-state-decay:0.001": "bf16 rounds the decay factor to exactly 1.0",
    # Dropping one recurrent-state writeback in 128 (or 256) decode steps
    # is under the floor at a 256-step horizon; 1-in-64 is over it. This
    # pair is the measured detection floor for this bug class.
    "gdn-state-stale-every:128": "under the detection floor at a 256-step horizon",
    "gdn-state-stale-every:256": "under the detection floor at a 256-step horizon",
    # Same three workloads run to their natural end (1164 steps instead of
    # 768). The bars were fitted at 256 steps per workload; this checks
    # they are not an artefact of that horizon.
    "none@full": "control at the full 1164-step horizon",
}

#: Must fail, and must fail on at least these bars. Listing the specific
#: bars is what stops the suite from passing for the wrong reason -- e.g.
#: a GDN state bug tripping only the NLL leg, which is structurally
#: incapable of seeing it (that leg is one prefill forward; the state bug
#: only exists across decode steps).
EXPECTED_FAIL: dict[str, tuple[str, ...]] = {
    "drop-q-norm": (
        "p90_gap_error",
        "mean_kl_topk",
        "max_tie_slack_ulps",
        "nll_relative_excess",
        "min_logits_cosine",
    ),
    "drop-k-norm": (
        "p90_gap_error",
        "mean_kl_topk",
        "max_tie_slack_ulps",
        "nll_relative_excess",
        "min_logits_cosine",
    ),
    "gdn-state-stale-every:1": ("p90_gap_error", "mean_kl_topk", "max_tie_slack_ulps"),
    "gdn-state-stale-every:8": ("p90_gap_error", "mean_kl_topk", "max_tie_slack_ulps"),
    "gdn-state-stale-every:64": ("p90_gap_error", "mean_kl_topk", "max_tie_slack_ulps"),
    "gdn-state-decay:0.01": ("p90_gap_error", "mean_kl_topk", "max_drift_ratio"),
    "gdn-state-decay:0.05": ("p90_gap_error", "mean_kl_topk", "max_drift_ratio"),
    "drop-q-norm@full": ("p90_gap_error", "mean_kl_topk", "nll_relative_excess"),
    "gdn-state-decay:0.01@full": ("p90_gap_error", "mean_kl_topk", "max_drift_ratio"),
}


@pytest.fixture(scope="module")
def sweep() -> dict:
    return json.loads(FIXTURE.read_text())


def _metrics(sweep: dict, name: str) -> dict[str, float]:
    return sweep["configs"][name]


def test_fixture_covers_every_expectation(sweep: dict) -> None:
    recorded = set(sweep["configs"])
    assert set(EXPECTED_PASS) | set(EXPECTED_FAIL) == recorded


@pytest.mark.parametrize("name", sorted(EXPECTED_PASS))
def test_configurations_at_or_below_the_noise_floor_pass(sweep: dict, name: str) -> None:
    passed, reasons = evaluate_summary(_metrics(sweep, name), CALIBRATED_THRESHOLDS)
    assert passed, f"{name} ({EXPECTED_PASS[name]}) unexpectedly failed: {reasons}"


@pytest.mark.parametrize("name", sorted(EXPECTED_FAIL))
def test_injected_bugs_are_caught(sweep: dict, name: str) -> None:
    passed, reasons = evaluate_summary(_metrics(sweep, name), CALIBRATED_THRESHOLDS)
    assert not passed, f"{name} passed the gate -- the criterion does not catch it"
    fired = {r.split("=")[0].split(" ")[0] for r in reasons}
    missing = set(EXPECTED_FAIL[name]) - fired
    assert not missing, f"{name}: expected these bars to fire too: {sorted(missing)}"


def test_the_nll_leg_is_blind_to_every_recurrent_state_bug(sweep: dict) -> None:
    """Pinned as a KNOWN blind spot, not an accident.

    R4 is one full-length prefill forward. A GDN recurrent-state bug only
    manifests across decode steps, so the state writeback the injection
    corrupts is never read back within a single forward. Measured: all
    five gdn-* configurations, including the one that freezes the state
    entirely, report an NLL identical to the control's to five decimal
    places. Anyone tempted to promote NLL to the primary bar has to delete
    this test first.
    """
    control = _metrics(sweep, "none")["nll_relative_excess"]
    for name in sweep["configs"]:
        if not name.startswith("gdn-"):
            continue
        assert _metrics(sweep, name)["nll_relative_excess"] == pytest.approx(control, abs=1e-5), (
            f"{name} moved the NLL -- update this test's premise"
        )


def test_the_prefill_layer_scan_is_blind_to_every_recurrent_state_bug(sweep: dict) -> None:
    """Same structural blindness as the NLL leg, same reason."""
    control = _metrics(sweep, "none")["min_logits_cosine"]
    for name in sweep["configs"]:
        if not name.startswith("gdn-"):
            continue
        assert _metrics(sweep, name)["min_logits_cosine"] == pytest.approx(control, abs=1e-7)


def test_only_the_step_locked_legs_catch_recurrent_state_bugs(sweep: dict) -> None:
    """The load-bearing justification for R1 existing at all.

    Both cheap legs (R3 prefill scan, R4 NLL) are blind to GDN state bugs,
    so if R1 were dropped the criterion would have no coverage whatsoever
    of the single highest-risk subsystem in this model (48 of 64 layers,
    RK1 in implementation-plan.md §7.1).
    """
    for name in ("gdn-state-stale-every:64", "gdn-state-decay:0.01"):
        _passed, reasons = evaluate_summary(_metrics(sweep, name), CALIBRATED_THRESHOLDS)
        fired = {r.split("=")[0].split(" ")[0] for r in reasons}
        assert "nll_relative_excess" not in fired
        assert "min_logits_cosine" not in fired
        assert fired, f"{name} was caught by nothing at all"


def test_max_gap_error_alone_would_not_separate(sweep: dict) -> None:
    """Why the primary bar is p90 and not max.

    Measured: the control's max gap error (10.56) is *higher* than that of
    two injected configurations, so a criterion built on max() would rank
    the buggy runs as cleaner than the correct one.
    """
    control = _metrics(sweep, "none")["max_gap_error"]
    lower = [
        name
        for name, m in sweep["configs"].items()
        if name != "none" and m["max_gap_error"] < control
    ]
    assert lower, "premise changed: no injected run has a lower max than the control"


def test_calibration_has_real_margin_on_both_sides(sweep: dict) -> None:
    """A bar wedged between control and bug with no room is a bar that
    will flake. Requires >= 1.5x headroom above the control AND >= 1.5x
    below the weakest bug that bar catches, for each primary bar."""
    control = _metrics(sweep, "none")
    for metric, attribute in (
        ("p90_gap_error", "p90_gap_error"),
        ("mean_kl_topk", "max_mean_kl_topk"),
        ("max_tie_slack_ulps", "max_tie_slack_ulps"),
    ):
        bar = getattr(CALIBRATED_THRESHOLDS, attribute)
        assert bar >= 1.5 * control[metric], f"{metric}: too little headroom over control"
        caught = [
            sweep["configs"][name][metric]
            for name in EXPECTED_FAIL
            if sweep["configs"][name][metric] > bar
        ]
        assert min(caught) >= 1.5 * bar, f"{metric}: weakest catch is too close to the bar"


def test_widening_the_primary_bar_breaks_the_weakest_catch(sweep: dict) -> None:
    """The calibration is load-bearing, not decorative: relaxing p90 to
    swallow the weakest detected bug lets that bug through on that bar."""
    weakest = _metrics(sweep, "gdn-state-stale-every:64")
    widened = AgreementThresholds.from_dict(
        {**CALIBRATED_THRESHOLDS.to_dict(), "p90_gap_error": weakest["p90_gap_error"] + 0.1}
    )
    _passed, reasons = evaluate_summary(weakest, widened)
    assert not any(r.startswith("p90_gap_error") for r in reasons)


def test_the_calibration_is_not_an_artefact_of_the_fitting_horizon(sweep: dict) -> None:
    """The bars were fitted on 256 steps per workload. Run to the natural
    end (1164 steps total) the control gets *quieter*, not noisier -- p99
    3.375 -> 2.125, mean KL 1.58e-3 -> 1.06e-3, drift 1.5 -> 1.0 -- so the
    calibration is conservative at the horizon the gate actually runs at,
    not tuned to a short one."""
    short = _metrics(sweep, "none")
    full = _metrics(sweep, "none@full")
    for metric in ("p99_gap_error", "mean_kl_topk", "disagreement_rate", "max_drift_ratio"):
        assert full[metric] <= short[metric] * 1.05, f"{metric} got worse at the full horizon"


def test_every_recorded_metric_is_finite_where_it_is_gated(sweep: dict) -> None:
    for name, m in sweep["configs"].items():
        for metric, value in m.items():
            assert not math.isnan(value), f"{name}.{metric} was never measured"
