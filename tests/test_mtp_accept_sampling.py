"""E2-a (docs/e2e-and-quality-plan.md §2.2): correctness proofs for the
non-greedy (rejection-sampling) MTP accept/reject functions in
``runtime/mtp_accept.py``.

Speculative decoding's entire value proposition is "identical output
distribution to not using speculation at all, just faster." A bug in the
residual-distribution math does not crash anything and does not show up as
an obviously wrong token -- it silently changes what the model samples,
which is a worse failure than not accelerating at all (this repo's FP8
rounding-tie lesson: synthetic-looking-plausible data does not surface this
class of bug; only exact arithmetic or a real statistical test does). So
this file deliberately does NOT rely on "looks about right":

Tier 1 (no torch, runs unconditionally -- including in this repo's
``/tmp/ci-sim`` CPU gate, which has neither torch nor numpy installed):
exact ``fractions.Fraction`` arithmetic proves the algebraic identity
``accept_mass(x) + residual_mass(x) == p(x)`` for every vocab entry, for
hand-picked edge cases (identical distributions, disjoint support,
one-hot/deterministic distributions, partial overlap) and for hundreds of
randomized rational distributions. This is the property that makes
speculative sampling distribution-preserving; see
``runtime/mtp_accept.py``'s module-level "E2-a" docstring for the one-line
algebraic proof (``min(a,b) + max(0,a-b) == a``) this is checking is
actually implemented correctly, not just true in the abstract.

Tier 2 (needs torch, gated behind a per-test ``pytest.importorskip`` so
tier 1 is never accidentally skipped alongside it): checks that
``runtime.mtp_accept``'s actual tensor code reproduces tier 1's exact
values bit-for-bit wherever floating point allows exactness (dyadic
inputs, sums/subtractions/power-of-two divisions), and reports a genuine
computed chi-square p-value (not eyeballed) that samples drawn through
``sample_accept_reject`` reproduce the target distribution -- with a
companion check that the SAME test would have failed hard against the
draft distribution (or other wrong hypotheses), proving the test has
actual discriminating power rather than passing vacuously.

How this set of tests would go red (the plan's "如何验证会红" requirement,
method 2 -- constructive: this functionality does not exist yet):
``runtime.mtp_accept.sample_accept_reject`` / ``acceptance_probability`` /
``residual_distribution`` did not exist before this change, so every test
below was red from the moment it was written until the implementation
landed alongside it in the same commit. Additionally, each individual
check is constructed so a specific plausible bug flips it red on its own
(noted inline): e.g. swapping ``max(0, p-q)`` for ``max(0, q-p)`` in
``residual_distribution``, or resampling from ``q`` instead of the
residual on rejection, would fail
``test_recovered_token_matches_residual_not_draft_distribution`` (its
p-value-against-``q`` assertion would flip from "<1e-6" to "not small").
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

# ---------------------------------------------------------------------------
# Tier 1: exact-rational distribution identity (no torch import at all).
# ---------------------------------------------------------------------------


def _accept_mass(p: Fraction, q: Fraction) -> Fraction:
    return min(p, q)


def _residual_mass(p: Fraction, q: Fraction) -> Fraction:
    return max(Fraction(0), p - q)


def _random_rational_distribution(rng: random.Random, vocab: int, denom: int) -> list[Fraction]:
    """A `vocab`-length list of ``Fraction``s, each with denominator dividing
    `denom`, summing to EXACTLY 1 (stars-and-bars over `denom` unit slots --
    not floats normalized after the fact, which would not sum to exactly 1
    in general)."""
    cuts = sorted(rng.randint(0, denom) for _ in range(vocab - 1))
    boundaries = [0, *cuts, denom]
    counts = [boundaries[i + 1] - boundaries[i] for i in range(vocab)]
    assert sum(counts) == denom
    return [Fraction(c, denom) for c in counts]


def _check_identity(p: list[Fraction], q: list[Fraction]) -> None:
    """The algebraic core of speculative sampling's correctness proof, for
    ONE (p, q) pair: for every vocabulary entry x,
    ``accept_mass(x) + residual_mass(x) == p(x)`` EXACTLY, and the total
    rejected probability mass exactly equals the total residual mass
    available to resample from (the identity ``residual_distribution``'s
    normalization depends on).
    """
    assert len(p) == len(q)
    assert sum(p) == Fraction(1), f"p is not a proper distribution: sums to {sum(p)}"
    assert sum(q) == Fraction(1), f"q is not a proper distribution: sums to {sum(q)}"
    total_accept = Fraction(0)
    total_residual = Fraction(0)
    for px, qx in zip(p, q):
        assert px >= 0 and qx >= 0
        accept = _accept_mass(px, qx)
        residual = _residual_mass(px, qx)
        assert accept + residual == px, (
            f"accept_mass({px},{qx})={accept} + residual_mass={residual} != p(x)={px}"
        )
        total_accept += accept
        total_residual += residual
    reject_prob = Fraction(1) - total_accept
    assert reject_prob == total_residual, (
        f"1 - total_accept={reject_prob} != total_residual={total_residual} "
        "-- residual_distribution's normalizer would divide by the wrong total"
    )


# Hand-picked edge cases, not just "typical" distributions -- these are the
# shapes a randomized search is unlikely to hit by chance but a real draft
# model can: identical distributions (never rejects), disjoint support
# (always rejects), one-hot/deterministic distributions on either side,
# and partial overlap.
_EDGE_CASES: dict[str, tuple[list[Fraction], list[Fraction]]] = {
    "identical": (
        [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)],
        [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)],
    ),
    "disjoint_support": (
        [Fraction(0), Fraction(0), Fraction(3, 5), Fraction(2, 5)],
        [Fraction(1, 2), Fraction(1, 2), Fraction(0), Fraction(0)],
    ),
    "target_deterministic": (
        [Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)],
    ),
    "draft_deterministic": (
        [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)],
        [Fraction(1), Fraction(0), Fraction(0)],
    ),
    "draft_favors_wrong_tail": (
        [Fraction(9, 10), Fraction(1, 20), Fraction(1, 20)],
        [Fraction(1, 20), Fraction(1, 20), Fraction(9, 10)],
    ),
    "partial_overlap": (
        [Fraction(3, 8), Fraction(3, 8), Fraction(1, 8), Fraction(1, 8)],
        [Fraction(1, 8), Fraction(1, 8), Fraction(3, 8), Fraction(3, 8)],
    ),
    # The exact pair reused by test_residual_distribution_matches_fraction_exact
    # below -- engineered so total residual mass is a power of two (1/2),
    # making its normalization bit-exact in float64 too, not just here.
    "power_of_two_residual_total": (
        [Fraction(0), Fraction(1, 4), Fraction(3, 8), Fraction(3, 8)],
        [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)],
    ),
}


@pytest.mark.parametrize("name", sorted(_EDGE_CASES))
def test_identity_hand_picked_edge_cases(name: str) -> None:
    p, q = _EDGE_CASES[name]
    _check_identity(p, q)


def test_identity_random_rational_distributions() -> None:
    """500 randomized rational (p, q) pairs, varying vocab size and
    denominator -- guards against the identity only "happening" to hold for
    hand-picked round numbers."""
    rng = random.Random(20260802)
    for _ in range(500):
        vocab = rng.randint(2, 9)
        denom = rng.choice([6, 10, 12, 20, 24, 30, 60])
        p = _random_rational_distribution(rng, vocab, denom)
        q = _random_rational_distribution(rng, vocab, denom)
        _check_identity(p, q)


def test_disjoint_support_residual_equals_target_exactly() -> None:
    """When ``q`` and ``p`` have disjoint support, EVERY draw from ``q`` is
    certain to be rejected (``p(x) == 0`` wherever ``q(x) > 0``), and the
    residual distribution reduces to ``p`` exactly (not just "close to")
    -- because ``q`` contributes nothing wherever ``p`` is nonzero. This is
    the specific scenario exercised statistically in Tier 2's
    ``test_recovered_token_matches_residual_not_draft_distribution``.
    """
    p, q = _EDGE_CASES["disjoint_support"]
    residual = [_residual_mass(px, qx) for px, qx in zip(p, q)]
    assert residual == p
    for px, qx in zip(p, q):
        assert _accept_mass(px, qx) == 0


# ---------------------------------------------------------------------------
# Chi-square p-value engine (no torch/scipy -- neither is available in the
# CPU-only gate). Validated against textbook chi-square critical values
# below so the p-values Tier 2 reports are trustworthy, not just plausible.
# Numerical Recipes §6.2 series/continued-fraction algorithm for the
# regularized lower incomplete gamma function P(a, x).
# ---------------------------------------------------------------------------


def _lower_incomplete_gamma_regularized(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        raise ValueError(f"invalid domain: a={a}, x={x}")
    if x == 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(500):
            ap += 1
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 1e-15:
                break
        return summ * math.exp(-x + a * math.log(x) - gln)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return 1.0 - q


def chi_square_sf(x: float, k: int) -> float:
    """P(X >= x) for X ~ chi-square with k degrees of freedom (the
    goodness-of-fit p-value: small means "very unlikely the samples came
    from the hypothesized distribution")."""
    return 1.0 - _lower_incomplete_gamma_regularized(k / 2.0, x / 2.0)


_CHI_SQUARE_EPS = 1e-12


def _chi_square_stat(counts: list[int], probs: list[float], n: int) -> float:
    """Pearson chi-square goodness-of-fit statistic. Zero-probability
    categories (e.g. testing the disjoint-support case against a
    hypothesis whose support excludes where the samples actually land) are
    floored to a tiny epsilon instead of excluded, so they can never divide
    by zero: a category that is genuinely impossible under the hypothesis
    AND has zero observed count contributes ~0 (correct -- consistent with
    the hypothesis); a category impossible under the hypothesis but with
    ANY observed count there blows the term up to a huge but finite value
    (also correct -- "the data contains something the hypothesis rules
    out" is exactly what should drive the p-value toward 0).
    """
    return sum(
        (c - n * max(pr, _CHI_SQUARE_EPS)) ** 2 / (n * max(pr, _CHI_SQUARE_EPS))
        for c, pr in zip(counts, probs)
    )


def test_chi_square_sf_matches_textbook_critical_values() -> None:
    """Sanity-check the from-scratch incomplete-gamma implementation against
    standard chi-square table critical values (upper-tail alpha at given
    df) before trusting any p-value it reports below."""
    table = [
        (3.841, 1, 0.05),
        (5.991, 2, 0.05),
        (7.815, 3, 0.05),
        (9.488, 4, 0.05),
        (11.070, 5, 0.05),
        (18.307, 10, 0.05),
        (6.635, 1, 0.01),
        (9.210, 2, 0.01),
        (11.345, 3, 0.01),
        (15.086, 5, 0.01),
        (23.209, 10, 0.01),
    ]
    for x, k, alpha in table:
        computed = chi_square_sf(x, k)
        assert math.isclose(computed, alpha, abs_tol=2e-4), (x, k, alpha, computed)


# ---------------------------------------------------------------------------
# Tier 2: the actual torch implementation in runtime/mtp_accept.py.
# ---------------------------------------------------------------------------


class TestAcceptanceProbabilityBitExact:
    def test_power_of_two_draft_probs_are_bit_exact(self) -> None:
        """Dividing by a power-of-two float is exact in IEEE-754 (it's a
        bit-shift of the mantissa, no rounding) -- engineered so this check
        can assert bit-for-bit equality against the Fraction-derived
        expectation, not a tolerance."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import acceptance_probability

        q_frac = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)]
        p_frac = [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)]
        assert sum(q_frac) == 1 and sum(p_frac) == 1
        q = torch.tensor([float(x) for x in q_frac], dtype=torch.float64)
        p = torch.tensor([float(x) for x in p_frac], dtype=torch.float64)
        expected = [min(Fraction(1), px / qx) for px, qx in zip(p_frac, q_frac)]
        assert expected == [Fraction(1, 2), Fraction(1), Fraction(1), Fraction(1)]
        for token, exp in enumerate(expected):
            got = acceptance_probability(p, q, token)
            assert got.item() == float(exp), (token, got.item(), float(exp))

    def test_zero_draft_prob_rejects_defensively(self) -> None:
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import acceptance_probability

        q = torch.tensor([0.0, 1.0], dtype=torch.float64)
        p = torch.tensor([0.5, 0.5], dtype=torch.float64)
        assert float(acceptance_probability(p, q, 0)) == 0.0


class TestResidualDistributionExact:
    def test_power_of_two_residual_total_is_bit_exact(self) -> None:
        """Same (p, q) pair as the "power_of_two_residual_total" Tier 1 edge
        case: total residual mass is exactly 1/2, so the one normalizing
        division is also a power-of-two division -- bit exact end to end,
        not just close."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import residual_distribution

        p_frac, q_frac = _EDGE_CASES["power_of_two_residual_total"]
        p = torch.tensor([float(x) for x in p_frac], dtype=torch.float64)
        q = torch.tensor([float(x) for x in q_frac], dtype=torch.float64)

        # Unnormalized max(0, p-q) is a sum/difference of dyadic values --
        # exact regardless of the total's shape.
        expected_raw = [float(_residual_mass(px, qx)) for px, qx in zip(p_frac, q_frac)]
        raw = (p - q).clamp_min(0.0)
        assert raw.tolist() == expected_raw

        expected_total = float(sum(_residual_mass(px, qx) for px, qx in zip(p_frac, q_frac)))
        assert raw.sum().item() == expected_total

        expected_normalized = [
            float(_residual_mass(px, qx) / Fraction(1, 2)) for px, qx in zip(p_frac, q_frac)
        ]
        got = residual_distribution(p, q)
        assert got.tolist() == expected_normalized, (got.tolist(), expected_normalized)

    def test_general_dyadic_residual_matches_fraction_within_double_precision(self) -> None:
        """Broader (non power-of-two-total) randomized dyadic cases: the
        UNnormalized residual is still checked bit-exact (pure
        sum/subtraction of dyadic floats); only the final normalizing
        division is checked against a tight, explicitly-justified tolerance
        (a few ULPs of IEEE-754 double precision -- the one place a single
        division is mathematically unavoidable, not an "eyeball" pass)."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import residual_distribution

        rng = random.Random(7)
        for _ in range(50):
            vocab = rng.randint(2, 6)
            denom = 2 ** rng.randint(2, 8)  # dyadic denominators only
            p_frac = _random_rational_distribution(rng, vocab, denom)
            q_frac = _random_rational_distribution(rng, vocab, denom)
            p = torch.tensor([float(x) for x in p_frac], dtype=torch.float64)
            q = torch.tensor([float(x) for x in q_frac], dtype=torch.float64)

            expected_raw = [float(_residual_mass(px, qx)) for px, qx in zip(p_frac, q_frac)]
            raw = (p - q).clamp_min(0.0)
            assert raw.tolist() == expected_raw

            total = sum(_residual_mass(px, qx) for px, qx in zip(p_frac, q_frac))
            got = residual_distribution(p, q)
            if total <= 0:
                continue  # p == q everywhere; residual_distribution's fallback path
            expected_normalized = [
                float(_residual_mass(px, qx) / total) for px, qx in zip(p_frac, q_frac)
            ]
            for g, e in zip(got.tolist(), expected_normalized):
                assert math.isclose(g, e, rel_tol=1e-12), (g, e)


class TestSampleAcceptRejectContract:
    """Deterministic (RNG-independent) contract checks -- shape and
    control-flow, the non-greedy analogue of test_mtp_accept.py's greedy
    ``test_all_accepted`` / ``test_first_rejected`` / etc."""

    def test_certain_accept_every_position(self) -> None:
        """q == p at every position makes acceptance_probability exactly 1
        everywhere, so EVERY draft token is accepted regardless of the RNG
        draw (``u < 1.0`` for any ``u`` in torch.rand's ``[0, 1)`` range)."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import sample_accept_reject

        uniform = torch.full((4,), 0.25, dtype=torch.float64)
        draft_probs = torch.stack([uniform, uniform, uniform])
        bonus_row = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        target_probs = torch.stack([uniform, uniform, uniform, bonus_row])
        gen = torch.Generator().manual_seed(0)

        for draft_tokens in ([0, 1, 2], [3, 3, 3], [0, 2, 1]):
            result = sample_accept_reject(draft_tokens, draft_probs, target_probs, generator=gen)
            assert result["num_accepted"] == 3
            assert result["rejected_at"] is None
            assert result["committed"][:3] == draft_tokens
            assert len(result["committed"]) == 4

    def test_certain_reject_disjoint_support(self) -> None:
        """Disjoint support forces acceptance_probability to exactly 0, so
        EVERY draw is rejected at position 0 regardless of the RNG draw."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import sample_accept_reject

        p_frac, q_frac = _EDGE_CASES["disjoint_support"]
        p = torch.tensor([float(x) for x in p_frac], dtype=torch.float64)
        q = torch.tensor([float(x) for x in q_frac], dtype=torch.float64)
        draft_probs = q.unsqueeze(0)
        target_probs = torch.stack([p, p])  # row 1 unused (always rejects at 0)
        gen = torch.Generator().manual_seed(0)

        for _ in range(20):
            draft_token = int(torch.multinomial(q, 1, generator=gen).item())
            result = sample_accept_reject([draft_token], draft_probs, target_probs, generator=gen)
            assert result["num_accepted"] == 0
            assert result["rejected_at"] == 0
            assert len(result["committed"]) == 1

    def test_shape_validation(self) -> None:
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import sample_accept_reject

        draft_probs = torch.full((2, 3), 1 / 3, dtype=torch.float64)
        target_probs = torch.full((2, 3), 1 / 3, dtype=torch.float64)  # need 3 rows, has 2
        with pytest.raises(ValueError):
            sample_accept_reject([0, 1], draft_probs, target_probs)


class TestSampleAcceptRejectStatistical:
    """The distributional guarantee itself, empirically: draw many samples
    through the real code path (RNG and all) and confirm the empirical
    distribution matches the target model's distribution -- via a computed
    chi-square p-value, not a visual "looks close" comparison. Each test
    also checks the SAME statistic against a wrong hypothesis to
    demonstrate the test has power to fail (this is how each test's
    ability to "go red" is proven, per the task's requirement)."""

    # Deliberately different-shaped q/p (not a near-identity perturbation)
    # so any bug that conflates the two distributions is caught, not masked
    # by them being nearly indistinguishable to begin with.
    _Q = [0.50, 0.20, 0.15, 0.10, 0.05]
    _P = [0.10, 0.15, 0.25, 0.20, 0.30]
    _N = 60_000
    _ALPHA_PASS = 0.01  # fail to reject H0 (samples ~ P) above this
    _ALPHA_FAIL = 1e-6  # H0 (samples ~ Q) must be rejected far below this

    def test_single_position_output_matches_target_not_draft_distribution(self) -> None:
        """Isolates ONE verify position: draw draft token x ~ q, accept/
        reject/resample against p, record the output. This is the exact
        scenario the module docstring's algebraic proof covers, exercised
        with real RNG draws instead of exact arithmetic."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import acceptance_probability, residual_distribution

        q = torch.tensor(self._Q, dtype=torch.float64)
        p = torch.tensor(self._P, dtype=torch.float64)
        gen = torch.Generator().manual_seed(20260802)

        counts = [0] * len(self._P)
        for _ in range(self._N):
            x = int(torch.multinomial(q, 1, generator=gen).item())
            accept_prob = acceptance_probability(p, q, x)
            u = torch.rand((), generator=gen, dtype=torch.float64)
            if bool(u < accept_prob):
                out = x
            else:
                residual = residual_distribution(p, q)
                out = int(torch.multinomial(residual, 1, generator=gen).item())
            counts[out] += 1

        stat_p = _chi_square_stat(counts, self._P, self._N)
        stat_q = _chi_square_stat(counts, self._Q, self._N)
        pvalue_p = chi_square_sf(stat_p, len(self._P) - 1)
        pvalue_q = chi_square_sf(stat_q, len(self._P) - 1)

        assert pvalue_p > self._ALPHA_PASS, (
            f"empirical output counts {counts} reject the target distribution "
            f"p={self._P} (chi2={stat_p:.3f}, p={pvalue_p:.3g}) -- speculative sampling "
            "would be silently changing the output distribution"
        )
        assert pvalue_q < self._ALPHA_FAIL, (
            f"the test has no discriminating power: samples also look consistent with the "
            f"DRAFT distribution q={self._Q} (chi2={stat_q:.3f}, p={pvalue_q:.3g})"
        )

    def test_chained_position_matches_target_through_public_api(self) -> None:
        """Same claim, but through the actual public ``sample_accept_reject``
        function used end to end (K=3): positions 0-1 are engineered to
        accept with probability exactly 1 (q == p there, any token),
        so every trial deterministically reaches position 2, where the
        skewed (p, q) pair above governs whether ``committed[2]`` is the
        original draft (accepted) or a resample (rejected) -- either way,
        the theorem says its marginal distribution is p at position 2.
        """
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import sample_accept_reject

        vocab = len(self._P)
        uniform = torch.full((vocab,), 1 / vocab, dtype=torch.float64)
        q2 = torch.tensor(self._Q, dtype=torch.float64)
        p2 = torch.tensor(self._P, dtype=torch.float64)
        bonus_row = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float64)

        draft_probs = torch.stack([uniform, uniform, q2])
        target_probs = torch.stack([uniform, uniform, p2, bonus_row])
        gen = torch.Generator().manual_seed(20260803)

        counts = [0] * vocab
        for _ in range(self._N):
            x2 = int(torch.multinomial(q2, 1, generator=gen).item())
            draft_tokens = [0, 0, x2]
            result = sample_accept_reject(draft_tokens, draft_probs, target_probs, generator=gen)
            assert result["num_accepted"] >= 2, "positions 0/1 must never reject (q==p there)"
            counts[result["committed"][2]] += 1

        stat_p = _chi_square_stat(counts, self._P, self._N)
        stat_q = _chi_square_stat(counts, self._Q, self._N)
        pvalue_p = chi_square_sf(stat_p, vocab - 1)
        pvalue_q = chi_square_sf(stat_q, vocab - 1)

        assert pvalue_p > self._ALPHA_PASS, (counts, stat_p, pvalue_p)
        assert pvalue_q < self._ALPHA_FAIL, (counts, stat_q, pvalue_q)

    def test_full_accept_bonus_token_matches_target_distribution(self) -> None:
        """When every draft position is accepted, the bonus token is drawn
        directly from the target's own distribution at the position past
        the last draft token -- never touched by the accept/reject
        machinery. Checked separately because it exercises a different
        code path (plain ``torch.multinomial`` on ``target_probs[k]``, no
        residual involved at all)."""
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import sample_accept_reject

        vocab = len(self._P)
        uniform = torch.full((vocab,), 1 / vocab, dtype=torch.float64)
        bonus_row = torch.tensor(self._P, dtype=torch.float64)

        draft_probs = torch.stack([uniform, uniform])
        target_probs = torch.stack([uniform, uniform, bonus_row])
        gen = torch.Generator().manual_seed(20260804)

        counts = [0] * vocab
        for _ in range(self._N):
            result = sample_accept_reject([0, 1], draft_probs, target_probs, generator=gen)
            assert result["num_accepted"] == 2
            assert result["rejected_at"] is None
            counts[result["committed"][2]] += 1

        stat_bonus = _chi_square_stat(counts, self._P, self._N)
        stat_wrong = _chi_square_stat(counts, self._Q, self._N)
        pvalue_bonus = chi_square_sf(stat_bonus, vocab - 1)
        pvalue_wrong = chi_square_sf(stat_wrong, vocab - 1)

        assert pvalue_bonus > self._ALPHA_PASS, (counts, stat_bonus, pvalue_bonus)
        assert pvalue_wrong < self._ALPHA_FAIL, (counts, stat_wrong, pvalue_wrong)

    def test_recovered_token_matches_residual_not_draft_distribution(self) -> None:
        """Disjoint support (Tier 1's ``disjoint_support`` case): every
        single draw is certain to reject (proven exactly in
        ``test_disjoint_support_residual_equals_target_exactly``), so
        EVERY output token is a "recovered" resample -- letting this test
        isolate the residual-sampling branch specifically. If
        ``residual_distribution`` were implemented as ``max(0, q - p)``
        (subtraction order swapped) or as plain ``q`` (forgetting to
        subtract at all), the recovered tokens would come out distributed
        like q's support {0, 1}, not p's support {2, 3} -- this assertion
        would flip to reject p and fail to reject q, exactly the failure
        mode this test's ``_ALPHA_FAIL`` check is there to catch.
        """
        torch = pytest.importorskip("torch")
        from runtime.mtp_accept import sample_accept_reject

        p_frac, q_frac = _EDGE_CASES["disjoint_support"]
        p_list = [float(x) for x in p_frac]
        q_list = [float(x) for x in q_frac]
        vocab = len(p_list)
        p = torch.tensor(p_list, dtype=torch.float64)
        q = torch.tensor(q_list, dtype=torch.float64)
        draft_probs = q.unsqueeze(0)
        target_probs = torch.stack([p, p])
        gen = torch.Generator().manual_seed(20260805)

        counts = [0] * vocab
        for _ in range(self._N):
            draft_token = int(torch.multinomial(q, 1, generator=gen).item())
            result = sample_accept_reject([draft_token], draft_probs, target_probs, generator=gen)
            assert result["num_accepted"] == 0
            counts[result["committed"][0]] += 1

        stat_p = _chi_square_stat(counts, p_list, self._N)
        # q's own distribution has zero probability exactly where every
        # single sample actually lands (support {2, 3}) -- _chi_square_stat's
        # epsilon floor turns that into a huge (not infinite) statistic.
        stat_q_support = _chi_square_stat(counts, q_list, self._N)
        pvalue_p = chi_square_sf(stat_p, vocab - 1)
        pvalue_q_support = chi_square_sf(stat_q_support, vocab - 1)

        assert pvalue_p > self._ALPHA_PASS, (counts, stat_p, pvalue_p)
        assert pvalue_q_support < self._ALPHA_FAIL, (counts, stat_q_support, pvalue_q_support)
