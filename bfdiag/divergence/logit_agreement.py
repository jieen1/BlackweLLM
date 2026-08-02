"""Step-locked logit-agreement metrics -- the replacement for B1's
"token-for-token greedy alignment" gate.

Why the literal gate cannot work (measured, not argued):
``notes/2026-08-02-b1-greedy-alignment-fails.md`` recorded three real
first-divergence points whose two candidate tokens were **1-2 bf16 ULP**
apart (0.0625 / 0.125 / 0.125 at a magnitude whose ULP is 0.0625). Two
independently-written bf16 implementations of the same math cannot be
required to agree on a comparison whose operands are one representable
step apart -- so "512 tokens, zero argmax flips" is not a correctness
bar, it is a lottery. The same repository already recorded the same
phenomenon once before between two of its *own* paths
(``notes/2026-08-02-eager-verify-cg-verify-divergence.md``).

What this module measures instead, and why it is still a real bar.

Both sides are driven through the **same** token sequence (see
``scripts/b1_forced_decode_agreement.py``) so that neither side's own
argmax choice can send it down a different trajectory. At every step the
caller hands this module the two sides' top-K ``{token_id: logit}``
slices, and the primary metric is the **gap error**

    delta(i) = max_t | (mine[t] - mine[a]) - (oracle[t] - oracle[a]) |

over the union top-K set, anchored at ``a`` = the oracle's own argmax.
Reading: *"by how much do the two implementations disagree about the
relative scores of the plausible next tokens?"* It is invariant to a
constant offset of either side's logits (which is a true no-op for both
argmax and softmax), and it is defined at every step whether or not the
argmax happens to flip -- so 512 steps yield 512 measurements instead of
the single "first divergence index" that the literal gate reduced an
entire run to.

The near-tie property comes out of this for free, as an identity rather
than a second, separately-tuned rule. With ``a`` = the oracle's argmax
``o1`` and ``m1`` = our argmax, define the two one-sided margins that a
human would actually look at after a flip::

    mine_margin   = mine[m1]   - mine[o1]      >= 0
    oracle_margin = oracle[o1] - oracle[m1]    >= 0
    tie_slack     = mine_margin + oracle_margin

Substituting, ``tie_slack = (mine[m1] - mine[a]) - (oracle[m1] -
oracle[a])``, which is exactly the (signed) ``delta`` term at ``t = m1``.
Hence

    **0 <= tie_slack <= delta**   at every step, always.

So a bound on ``delta`` *implies* a bound on how confidently the two
sides can disagree: if ``delta <= tau`` then no flip whose combined
margin exceeds ``tau`` is possible. Gating on ``delta`` is strictly
stronger than gating on "every flip must be a near tie", and unlike it,
does not go blind on a run that happens to produce no flips at all.

``tie_slack`` is also the right thing to express in ULP: it is the amount
by which our logits would have to move to reproduce the oracle's ranking
of that one token pair, and bf16 cannot represent a move smaller than one
ULP at that magnitude. A flip at ``tie_slack`` of 1-2 ULP is what the
measurements above found and is, by construction, the smallest
disagreement the format admits.

A second, deliberately redundant statistic rides along: ``logprob_error``,
the max ``|delta log_softmax|`` over the same comparison set. It carries
no information the gap error does not (the two differ only by which
per-side constant is subtracted -- each side's own ``logsumexp`` instead
of its logit at the anchor token), and it exists purely so this
repository's number is directly comparable to an **upstream, independently
calibrated** one: SGLang gates exactly this quantity, across its entire
model zoo, at ``prefill_tolerance=5e-2`` / ``decode_tolerance=6e-2``
(``sglang/test/registered/models/test_generation_models.py``'s
``ModelCase`` defaults, with the decode bar carrying the comment
"Increased to fix numerical error in issue #8614"). A threshold argued
only from our own measurements is a threshold argued from one sample.

Everything here is torch-free and operates on plain mappings, following
the same split as ``bfdiag/divergence/scan.py`` vs ``capture.py``: the
GPU-side script does the top-K extraction with torch and this module does
all comparison and pass/fail logic, so the logic is exhaustively testable
on CPU (``tests/test_bfdiag_logit_agreement.py``) without a GPU.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How many of each side's top tokens participate in ``gap_error``. Only
#: tokens that could plausibly become the argmax matter for the flip
#: question, and a wider window just imports tail noise into a max().
DEFAULT_GAP_TOP_K = 8

#: bf16 has 1 sign + 8 exponent + 7 explicit mantissa bits.
_BF16_MANTISSA_BITS = 7

#: Below this magnitude, treat the ULP as the subnormal-free floor of the
#: smallest normal bf16 exponent rather than letting ``log2`` run off to
#: -inf. Only reached for logits that round to (near) zero, where the ULP
#: conversion is meaningless anyway.
_MIN_ULP = 2.0 ** (-126 - _BF16_MANTISSA_BITS)


def bf16_ulp(magnitude: float) -> float:
    """Unit in the last place of bf16 at ``|magnitude|``.

    bf16 keeps 7 explicit mantissa bits, so within the binade
    ``[2**e, 2**(e+1))`` consecutive representable values are
    ``2**(e - 7)`` apart. At the magnitude the B1 divergences were found
    at (logits in ``[8, 16)``, i.e. ``e = 3``) this is
    ``2**(3-7) = 0.0625`` -- the exact value that note recorded, which is
    what this function is calibrated against.
    """
    magnitude = abs(float(magnitude))
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return _MIN_ULP
    exponent = math.floor(math.log2(magnitude))
    return max(_MIN_ULP, 2.0 ** (exponent - _BF16_MANTISSA_BITS))


@dataclass(frozen=True)
class StepAgreement:
    """One step's comparison between two implementations' logits.

    ``mine_top1``/``oracle_top1`` are each side's argmax **restricted to
    the captured top-K slice**; the caller is responsible for capturing
    enough of each side that the true argmax is inside its own slice
    (trivially true for K >= 1, since a side's top-K always contains its
    own argmax).
    """

    step_index: int
    forced_token_id: int
    mine_top1: int
    oracle_top1: int
    gap_error: float
    mine_margin: float
    oracle_margin: float
    tie_slack: float
    logit_scale: float
    kl_topk: float
    #: max |delta log_softmax| over the comparison set, or ``nan`` when the
    #: caller did not supply both sides' full-vocabulary logsumexp. Exists
    #: to be comparable with SGLang's published 5e-2/6e-2 bars; see the
    #: module docstring.
    logprob_error: float = float("nan")

    @property
    def agrees(self) -> bool:
        return self.mine_top1 == self.oracle_top1

    @property
    def tie_slack_ulps(self) -> float:
        return self.tie_slack / bf16_ulp(self.logit_scale)

    @property
    def gap_error_ulps(self) -> float:
        return self.gap_error / bf16_ulp(self.logit_scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "forced_token_id": self.forced_token_id,
            "mine_top1": self.mine_top1,
            "oracle_top1": self.oracle_top1,
            "gap_error": self.gap_error,
            "mine_margin": self.mine_margin,
            "oracle_margin": self.oracle_margin,
            "tie_slack": self.tie_slack,
            "logit_scale": self.logit_scale,
            "kl_topk": self.kl_topk,
            "logprob_error": self.logprob_error,
            "agrees": self.agrees,
            "tie_slack_ulps": self.tie_slack_ulps,
            "gap_error_ulps": self.gap_error_ulps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StepAgreement:
        return cls(
            step_index=int(data["step_index"]),
            forced_token_id=int(data["forced_token_id"]),
            mine_top1=int(data["mine_top1"]),
            oracle_top1=int(data["oracle_top1"]),
            gap_error=float(data["gap_error"]),
            mine_margin=float(data["mine_margin"]),
            oracle_margin=float(data["oracle_margin"]),
            tie_slack=float(data["tie_slack"]),
            logit_scale=float(data["logit_scale"]),
            kl_topk=float(data["kl_topk"]),
            logprob_error=float(data.get("logprob_error", float("nan"))),
        )


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _kl_over_union(
    mine: Mapping[int, float], oracle: Mapping[int, float], union: Sequence[int]
) -> float:
    """KL(oracle_topk || mine_topk), both renormalised over ``union``.

    Truncated deliberately: tail mass below the captured top-K cannot
    change a greedy or top-p decision, and including it would require the
    caller to ship a full 250k-entry vocabulary distribution per step. The
    oracle is the first argument because KL's asymmetry should charge the
    candidate for putting low probability where the reference puts high
    probability, not the other way round.
    """
    oracle_probs = _softmax([oracle[t] for t in union])
    mine_probs = _softmax([mine[t] for t in union])
    total = 0.0
    for p, q in zip(oracle_probs, mine_probs, strict=True):
        if p <= 0.0:
            continue
        if q <= 0.0:
            return math.inf
        total += p * math.log(p / q)
    return total


def compare_step(
    step_index: int,
    forced_token_id: int,
    mine_logits: Mapping[int, float],
    oracle_logits: Mapping[int, float],
    *,
    gap_top_k: int = DEFAULT_GAP_TOP_K,
    mine_logsumexp: float | None = None,
    oracle_logsumexp: float | None = None,
) -> StepAgreement:
    """Compare one step's two top-K logit slices.

    ``mine_logits``/``oracle_logits`` map token id -> logit for that
    side's captured top-K. Only tokens present on **both** sides can
    participate in a difference, so the comparison set is the
    intersection of the two slices restricted to the union of each side's
    own top ``gap_top_k`` tokens. Anything one side ranked highly but the
    other did not capture at all is a *stronger* disagreement than
    anything this function can measure, and is reported by inflating
    nothing -- the caller must capture K large enough that this does not
    silently happen (see :func:`missing_from_intersection`).
    """
    if not mine_logits or not oracle_logits:
        raise ValueError("both sides must provide at least one (token, logit) pair")

    mine_top = sorted(mine_logits, key=lambda t: (-mine_logits[t], t))
    oracle_top = sorted(oracle_logits, key=lambda t: (-oracle_logits[t], t))
    mine_top1 = mine_top[0]
    oracle_top1 = oracle_top[0]

    shared = set(mine_logits) & set(oracle_logits)
    wanted = set(mine_top[:gap_top_k]) | set(oracle_top[:gap_top_k])
    comparison_set = sorted(wanted & shared)
    if oracle_top1 not in comparison_set:
        raise ValueError(
            "the oracle's own argmax is absent from the candidate's captured "
            "top-K: capture a larger K, this comparison is not meaningful"
        )

    anchor = oracle_top1
    mine_anchor = mine_logits[anchor]
    oracle_anchor = oracle_logits[anchor]
    gap_error = max(
        abs((mine_logits[t] - mine_anchor) - (oracle_logits[t] - oracle_anchor))
        for t in comparison_set
    )

    if mine_top1 in shared and oracle_top1 in mine_logits:
        mine_margin = mine_logits[mine_top1] - mine_logits[oracle_top1]
        oracle_margin = oracle_logits[oracle_top1] - oracle_logits[mine_top1]
        tie_slack = mine_margin + oracle_margin
    else:
        # Our argmax is a token the oracle did not even rank in its top-K.
        # No finite margin can be computed; report it as unbounded so no
        # threshold can accidentally pass it.
        mine_margin = math.inf
        oracle_margin = math.inf
        tie_slack = math.inf

    union_for_kl = sorted(shared)
    kl = _kl_over_union(mine_logits, oracle_logits, union_for_kl)

    logit_scale = max(abs(oracle_logits[t]) for t in comparison_set)

    if mine_logsumexp is None or oracle_logsumexp is None:
        logprob_error = float("nan")
    else:
        logprob_error = max(
            abs((mine_logits[t] - mine_logsumexp) - (oracle_logits[t] - oracle_logsumexp))
            for t in comparison_set
        )

    return StepAgreement(
        step_index=step_index,
        forced_token_id=forced_token_id,
        mine_top1=mine_top1,
        oracle_top1=oracle_top1,
        gap_error=gap_error,
        mine_margin=mine_margin,
        oracle_margin=oracle_margin,
        tie_slack=tie_slack,
        logit_scale=logit_scale,
        kl_topk=kl,
        logprob_error=logprob_error,
    )


def missing_from_intersection(
    mine_logits: Mapping[int, float], oracle_logits: Mapping[int, float], *, top_k: int
) -> tuple[int, ...]:
    """Tokens in either side's top ``top_k`` that the other side did not
    capture at all -- a capture-width problem, reported separately from
    the numeric verdict so it can never be mistaken for a pass."""
    mine_top = sorted(mine_logits, key=lambda t: (-mine_logits[t], t))[:top_k]
    oracle_top = sorted(oracle_logits, key=lambda t: (-oracle_logits[t], t))[:top_k]
    missing = [t for t in mine_top if t not in oracle_logits]
    missing += [t for t in oracle_top if t not in mine_logits]
    return tuple(sorted(set(missing)))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


@dataclass(frozen=True)
class WorkloadAgreement:
    """All steps of one workload, plus the summary statistics gated on."""

    workload_name: str
    prompt_token_ids: tuple[int, ...]
    steps: tuple[StepAgreement, ...]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def max_gap_error(self) -> float:
        return max((s.gap_error for s in self.steps), default=0.0)

    @property
    def p99_gap_error(self) -> float:
        return _quantile(sorted(s.gap_error for s in self.steps), 0.99)

    @property
    def median_gap_error(self) -> float:
        return _quantile(sorted(s.gap_error for s in self.steps), 0.5)

    @property
    def max_tie_slack_ulps(self) -> float:
        flips = [s for s in self.steps if not s.agrees]
        return max((s.tie_slack_ulps for s in flips), default=0.0)

    @property
    def num_disagreements(self) -> int:
        return sum(1 for s in self.steps if not s.agrees)

    @property
    def disagreement_rate(self) -> float:
        if not self.steps:
            return 0.0
        return self.num_disagreements / len(self.steps)

    @property
    def mean_kl_topk(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.kl_topk for s in self.steps) / len(self.steps)

    @property
    def max_kl_topk(self) -> float:
        return max((s.kl_topk for s in self.steps), default=0.0)

    @property
    def max_logprob_error(self) -> float:
        """``nan`` propagates deliberately: a caller that did not supply
        logsumexp values must not read a small number here."""
        values = [s.logprob_error for s in self.steps]
        if not values:
            return 0.0
        if any(math.isnan(v) for v in values):
            return float("nan")
        return max(values)

    def drift_ratio(self, *, window: int = 128) -> float:
        """``median(gap_error)`` over the last ``window`` steps divided by
        the same over the first ``window`` steps.

        Targets the one failure mode that step-locking otherwise hides: a
        recurrent-state bug (GDN keeps a state across every decode step)
        whose per-step effect is under the noise floor but accumulates.
        Forcing both sides onto one trajectory removes the autoregressive
        amplification that would eventually make such a bug visible as
        garbage text, so the trend has to be gated explicitly instead.
        Returns 1.0 when there are too few steps to form two windows.
        """
        if len(self.steps) < 2 * window:
            return 1.0
        early = sorted(s.gap_error for s in self.steps[:window])
        late = sorted(s.gap_error for s in self.steps[-window:])
        early_median = _quantile(early, 0.5)
        late_median = _quantile(late, 0.5)
        if early_median <= 0.0:
            return 1.0 if late_median <= 0.0 else math.inf
        return late_median / early_median

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_name": self.workload_name,
            "prompt_token_ids": list(self.prompt_token_ids),
            "steps": [s.to_dict() for s in self.steps],
            "summary": {
                "num_steps": self.num_steps,
                "max_gap_error": self.max_gap_error,
                "p99_gap_error": self.p99_gap_error,
                "median_gap_error": self.median_gap_error,
                "max_tie_slack_ulps": self.max_tie_slack_ulps,
                "num_disagreements": self.num_disagreements,
                "disagreement_rate": self.disagreement_rate,
                "mean_kl_topk": self.mean_kl_topk,
                "max_kl_topk": self.max_kl_topk,
                "max_logprob_error": self.max_logprob_error,
                "drift_ratio": self.drift_ratio(),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkloadAgreement:
        return cls(
            workload_name=data["workload_name"],
            prompt_token_ids=tuple(data["prompt_token_ids"]),
            steps=tuple(StepAgreement.from_dict(s) for s in data["steps"]),
        )


@dataclass(frozen=True)
class AgreementThresholds:
    """Every pass bar of the B1-R gate, in one overridable place.

    Defaults are the *calibrated* values -- see
    ``docs/b1-correctness-criterion.md`` for how each was derived from a
    measured control run plus the injected-bug separation, and why a
    number that has not been calibrated on real hardware must be treated
    as provisional.
    """

    max_gap_error: float
    p99_gap_error: float
    max_tie_slack_ulps: float
    max_disagreement_rate: float
    max_mean_kl_topk: float
    max_drift_ratio: float
    #: ``None`` leaves the SGLang-comparable bar ungated (for reports whose
    #: caller did not supply logsumexp values). A float gates it, and an
    #: unmeasured (``nan``) value then fails rather than silently passing.
    max_logprob_error: float | None = None
    min_steps_per_workload: int = 512
    min_workloads: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_gap_error": self.max_gap_error,
            "p99_gap_error": self.p99_gap_error,
            "max_tie_slack_ulps": self.max_tie_slack_ulps,
            "max_disagreement_rate": self.max_disagreement_rate,
            "max_mean_kl_topk": self.max_mean_kl_topk,
            "max_drift_ratio": self.max_drift_ratio,
            "max_logprob_error": self.max_logprob_error,
            "min_steps_per_workload": self.min_steps_per_workload,
            "min_workloads": self.min_workloads,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgreementThresholds:
        return cls(**{k: v for k, v in data.items()})


#: Placeholder bars, deliberately NOT a calibrated gate. Kept so the pure
#: logic and its tests have something to run against before a control run
#: exists; ``docs/b1-correctness-criterion.md`` records the measured
#: numbers that replace these.
UNCALIBRATED_THRESHOLDS = AgreementThresholds(
    max_gap_error=1.0,
    p99_gap_error=0.5,
    max_tie_slack_ulps=8.0,
    max_disagreement_rate=0.05,
    max_mean_kl_topk=0.01,
    max_drift_ratio=4.0,
)


@dataclass(frozen=True)
class AgreementReport:
    workloads: tuple[WorkloadAgreement, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_gap_error(self) -> float:
        return max((w.max_gap_error for w in self.workloads), default=0.0)

    @property
    def p99_gap_error(self) -> float:
        combined = sorted(s.gap_error for w in self.workloads for s in w.steps)
        return _quantile(combined, 0.99)

    @property
    def max_tie_slack_ulps(self) -> float:
        return max((w.max_tie_slack_ulps for w in self.workloads), default=0.0)

    @property
    def disagreement_rate(self) -> float:
        total = sum(w.num_steps for w in self.workloads)
        if total == 0:
            return 0.0
        return sum(w.num_disagreements for w in self.workloads) / total

    @property
    def mean_kl_topk(self) -> float:
        total = sum(w.num_steps for w in self.workloads)
        if total == 0:
            return 0.0
        return sum(s.kl_topk for w in self.workloads for s in w.steps) / total

    @property
    def max_drift_ratio(self) -> float:
        return max((w.drift_ratio() for w in self.workloads), default=1.0)

    @property
    def max_logprob_error(self) -> float:
        values = [w.max_logprob_error for w in self.workloads]
        if not values:
            return 0.0
        if any(math.isnan(v) for v in values):
            return float("nan")
        return max(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "workloads": [w.to_dict() for w in self.workloads],
            "summary": {
                "max_gap_error": self.max_gap_error,
                "p99_gap_error": self.p99_gap_error,
                "max_tie_slack_ulps": self.max_tie_slack_ulps,
                "disagreement_rate": self.disagreement_rate,
                "mean_kl_topk": self.mean_kl_topk,
                "max_logprob_error": self.max_logprob_error,
                "max_drift_ratio": self.max_drift_ratio,
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> AgreementReport:
        data = json.loads(Path(path).read_text())
        return cls(
            workloads=tuple(WorkloadAgreement.from_dict(w) for w in data["workloads"]),
            metadata=dict(data.get("metadata", {})),
        )


def passes_b1r_gate(
    report: AgreementReport, thresholds: AgreementThresholds
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate the B1-R gate. Returns ``(passed, failure_reasons)``.

    Every bar is checked (no short-circuit) so one run tells the reader
    *everything* that is out of band, not just the first thing -- the
    injected-bug experiment in ``docs/b1-correctness-criterion.md`` reads
    the whole list to show which legs of the criterion each bug trips.
    """
    reasons: list[str] = []

    if len(report.workloads) < thresholds.min_workloads:
        reasons.append(
            f"only {len(report.workloads)} workload(s), gate requires "
            f">= {thresholds.min_workloads}"
        )
    short = [
        w for w in report.workloads if w.num_steps < thresholds.min_steps_per_workload
    ]
    if short:
        names = ", ".join(f"{w.workload_name}({w.num_steps})" for w in short)
        return False, tuple(
            reasons
            + [
                f"{len(short)} workload(s) ran fewer than "
                f"{thresholds.min_steps_per_workload} steps: {names}"
            ]
        )

    if report.max_gap_error > thresholds.max_gap_error:
        reasons.append(
            f"max_gap_error={report.max_gap_error:.6g} > {thresholds.max_gap_error:.6g}"
        )
    if report.p99_gap_error > thresholds.p99_gap_error:
        reasons.append(
            f"p99_gap_error={report.p99_gap_error:.6g} > {thresholds.p99_gap_error:.6g}"
        )
    if report.max_tie_slack_ulps > thresholds.max_tie_slack_ulps:
        reasons.append(
            f"max_tie_slack_ulps={report.max_tie_slack_ulps:.4g} > "
            f"{thresholds.max_tie_slack_ulps:.4g}"
        )
    if report.disagreement_rate > thresholds.max_disagreement_rate:
        reasons.append(
            f"disagreement_rate={report.disagreement_rate:.4g} > "
            f"{thresholds.max_disagreement_rate:.4g}"
        )
    if report.mean_kl_topk > thresholds.max_mean_kl_topk:
        reasons.append(
            f"mean_kl_topk={report.mean_kl_topk:.6g} > {thresholds.max_mean_kl_topk:.6g}"
        )
    if thresholds.max_logprob_error is not None:
        measured = report.max_logprob_error
        if math.isnan(measured):
            reasons.append(
                "max_logprob_error is gated but was not measured (no logsumexp supplied)"
            )
        elif measured > thresholds.max_logprob_error:
            reasons.append(
                f"max_logprob_error={measured:.6g} > {thresholds.max_logprob_error:.6g}"
            )
    if report.max_drift_ratio > thresholds.max_drift_ratio:
        reasons.append(
            f"max_drift_ratio={report.max_drift_ratio:.4g} > "
            f"{thresholds.max_drift_ratio:.4g}"
        )

    return (not reasons), tuple(reasons)
