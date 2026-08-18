"""Small, dependency-free pieces of the SGLang DSpark verify planner.

The production scheduler in SGLang keeps the expensive draft forward on the
device and uses the draft confidence head to decide how much of that block is
worth verifying.  This module contains the mathematical part of that contract
without depending on a scheduler implementation.  The CUDA runtime may use
the helpers with a tiny host-side confidence snapshot; unit tests can exercise
the exact prefix/survival rules on CPU.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpsCostTable:
    """Piecewise-constant SPS table used by SGLang's DSpark planner."""

    sample_batch_tokens: tuple[int, ...]
    sample_steps_per_sec: tuple[float, ...]
    max_batch_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_batch_tokens", tuple(self.sample_batch_tokens))
        object.__setattr__(self, "sample_steps_per_sec", tuple(self.sample_steps_per_sec))
        if not self.sample_batch_tokens:
            raise ValueError("SpsCostTable requires at least one probe")
        if tuple(sorted(set(self.sample_batch_tokens))) != self.sample_batch_tokens:
            raise ValueError("sample_batch_tokens must be strictly increasing")
        if len(self.sample_batch_tokens) != len(self.sample_steps_per_sec):
            raise ValueError("sample_batch_tokens and sample_steps_per_sec must have equal length")
        if self.max_batch_tokens < self.sample_batch_tokens[-1]:
            raise ValueError("max_batch_tokens must be >= the largest SPS probe")
        if any(token < 1 for token in self.sample_batch_tokens):
            raise ValueError("SPS probe batch-token counts must be positive")
        if any(rate <= 0.0 for rate in self.sample_steps_per_sec):
            raise ValueError("SPS probe rates must be positive")

    def lookup(self, batch_tokens: int) -> float:
        index = max(
            0,
            min(
                bisect_right(self.sample_batch_tokens, int(batch_tokens)) - 1,
                len(self.sample_steps_per_sec) - 1,
            ),
        )
        return self.sample_steps_per_sec[index]


@dataclass(frozen=True)
class SpsAdditiveCostTable:
    """Additive step-time model accepted by SGLang's SPS planner."""

    bias_seconds: float
    bs_probes: tuple[int, ...]
    alpha_seconds: tuple[float, ...]
    m_probes: tuple[int, ...]
    theta_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bs_probes", tuple(self.bs_probes))
        object.__setattr__(self, "alpha_seconds", tuple(self.alpha_seconds))
        object.__setattr__(self, "m_probes", tuple(self.m_probes))
        object.__setattr__(self, "theta_seconds", tuple(self.theta_seconds))
        for probes, values, name in (
            (self.bs_probes, self.alpha_seconds, "bs"),
            (self.m_probes, self.theta_seconds, "m"),
        ):
            if not probes or tuple(sorted(set(probes))) != probes:
                raise ValueError(f"{name}_probes must be strictly increasing")
            if len(probes) != len(values):
                raise ValueError(f"{name}_probes and values must have equal length")
        if self.bias_seconds <= 0.0:
            raise ValueError("bias_seconds must be positive")

    def step_time(self, *, num_requests: int, budget: int) -> float:
        return (
            self.bias_seconds
            + _interp_clamped(self.bs_probes, self.alpha_seconds, float(num_requests))
            + _interp_clamped(self.m_probes, self.theta_seconds, float(num_requests + budget))
        )


@dataclass(frozen=True)
class VerifyBudgetDecision:
    budget: int
    predicted_step_seconds: float | None = None
    predicted_theta: float | None = None


def _interp_clamped(probes: Sequence[int], values: Sequence[float], point: float) -> float:
    if point <= probes[0]:
        return float(values[0])
    if point >= probes[-1]:
        return float(values[-1])
    high = bisect_right(probes, point)
    low = high - 1
    fraction = (point - probes[low]) / (probes[high] - probes[low])
    return float(values[low]) + fraction * (float(values[high]) - float(values[low]))


def _as_rows(history_survival_probs: Sequence[Sequence[float]]) -> list[list[float]]:
    # ``torch.Tensor.tolist`` is deliberately the only supported tensor bridge;
    # the planner itself runs on the tiny confidence snapshot, never logits.
    tolist = getattr(history_survival_probs, "tolist", None)
    rows = tolist() if callable(tolist) else history_survival_probs
    return [[float(value) for value in row] for row in rows]


def _table_from_mapping(data: Mapping[str, Any]) -> SpsCostTable | SpsAdditiveCostTable:
    if "bias_seconds" in data:
        return SpsAdditiveCostTable(
            bias_seconds=float(data["bias_seconds"]),
            bs_probes=tuple(int(value) for value in data["bs_probes"]),
            alpha_seconds=tuple(float(value) for value in data["alpha_seconds"]),
            m_probes=tuple(int(value) for value in data["m_probes"]),
            theta_seconds=tuple(float(value) for value in data["theta_seconds"]),
        )
    return SpsCostTable(
        sample_batch_tokens=tuple(int(value) for value in data["sample_batch_tokens"]),
        sample_steps_per_sec=tuple(float(value) for value in data["sample_steps_per_sec"]),
        max_batch_tokens=int(data["max_batch_tokens"]),
    )


def load_sps_table(path: str) -> SpsCostTable | SpsAdditiveCostTable:
    """Load the JSON format used by SGLang's ``--speculative-dspark-sps-table``."""

    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError("DSpark SPS table JSON must contain an object")
    return _table_from_mapping(data)


def build_uninitialized_sps_table(*, max_batch_tokens: int) -> SpsCostTable:
    """Match SGLang's no-profile fallback (which intentionally verifies all)."""

    return SpsCostTable((1,), (1.0,), max(1, int(max_batch_tokens)))


def compute_verify_token_budget(
    history_survival_probs: Sequence[Sequence[float]],
    *,
    sps_table: SpsCostTable | SpsAdditiveCostTable,
    max_verify_len: int | None = None,
    min_verify_len: int = 1,
    survival_eps: float = 1e-6,
) -> VerifyBudgetDecision:
    """Port SGLang's SPS objective ``argmax(tau_star * SPS)`` exactly."""

    rows = _as_rows(history_survival_probs)
    num_requests = len(rows)
    if num_requests == 0:
        return VerifyBudgetDecision(0, None, 0.0)
    width = max((len(row) for row in rows), default=0)
    max_len = width + 1 if max_verify_len is None else int(max_verify_len)
    if max_len < 1:
        raise ValueError(f"max_verify_len must be positive, got {max_len}")
    if min_verify_len < 0 or min_verify_len > max_len:
        raise ValueError(f"min_verify_len must be in [0,{max_len}], got {min_verify_len}")
    if survival_eps < 0.0:
        raise ValueError(f"survival_eps must be non-negative, got {survival_eps}")

    candidates = [
        min(max(float(value), 0.0), 1.0)
        for row in rows
        for value in row[:max_len]
        if float(value) >= survival_eps
    ]
    candidates.sort(reverse=True)
    prefix_sum = [0.0]
    for value in candidates:
        prefix_sum.append(prefix_sum[-1] + value)
    tau_star = [float(num_requests) + value for value in prefix_sum]
    if isinstance(sps_table, SpsAdditiveCostTable):
        step_times = [
            sps_table.step_time(num_requests=num_requests, budget=index)
            for index in range(len(tau_star))
        ]
        theta = [tau / step for tau, step in zip(tau_star, step_times, strict=True)]
        best = max(range(len(theta)), key=theta.__getitem__)
        return VerifyBudgetDecision(
            best,
            step_times[best],
            theta[best],
        )

    theta = [tau * sps_table.lookup(num_requests + index) for index, tau in enumerate(tau_star)]
    best = max(range(len(theta)), key=theta.__getitem__)
    rate = sps_table.lookup(num_requests + best)
    return VerifyBudgetDecision(best, 1.0 / rate, theta[best])


def schedule_verify_lens_topk(
    confidences: Sequence[Sequence[float]],
    *,
    budget: int,
    min_verify_len: int = 1,
    max_verify_len: int | None = None,
    survival_eps: float = 1e-6,
) -> list[int]:
    """Port SGLang's stable global top-k verify-lens scheduler."""

    rows = _as_rows(confidences)
    num_requests = len(rows)
    width = max((len(row) for row in rows), default=0)
    max_len = width + 1 if max_verify_len is None else int(max_verify_len)
    if min_verify_len < 0 or not min_verify_len <= max_len <= width + 1:
        raise ValueError(
            "verify-len bounds must satisfy "
            f"0 <= min <= max <= gamma+1, got min={min_verify_len}, max={max_len}"
        )
    if budget < 0:
        raise ValueError(f"verify budget must be non-negative, got {budget}")

    candidates: list[tuple[float, int, int]] = []
    for request, row in enumerate(rows):
        survival = survival_prefix(row, epsilon=0.0)
        for position, value in enumerate(survival[:max_len]):
            if value >= survival_eps:
                # Sorting by (-value, position, request) is the stable order
                # used by SGLang's torch and Triton implementations.
                candidates.append((value, position, request))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [0] * num_requests
    for _value, _position, request in candidates[: int(budget)]:
        selected[request] += 1
    lower_bound = max(min_verify_len, 1)
    return [min(max(min_verify_len + extra, lower_bound), max_len) for extra in selected]


def schedule_verify_widths_topk(
    confidences: Sequence[Sequence[float]],
    *,
    budget: int,
    min_verify_len: int = 1,
    max_verify_len: int | None = None,
    survival_eps: float = 1e-6,
) -> list[int]:
    """Return local draft widths for SGLang verify lenses (lens includes anchor)."""

    return [
        verify_len - 1
        for verify_len in schedule_verify_lens_topk(
            confidences,
            budget=budget,
            min_verify_len=min_verify_len,
            max_verify_len=max_verify_len,
            survival_eps=survival_eps,
        )
    ]


def survival_prefix(confidence: Sequence[float], *, epsilon: float = 1e-6) -> list[float]:
    """Return ``P(first n drafts survive)`` for each draft position.

    SGLang's confidence head predicts a per-position acceptance probability.
    Verify budgeting operates on the product of the prefix, not on an
    independent threshold for every position.  Clamping keeps malformed
    checkpoint output from producing negative or NaN budgets at the scheduler
    boundary while leaving normal probabilities untouched.
    """

    survival: list[float] = []
    product = 1.0
    for value in confidence:
        probability = min(max(float(value), 0.0), 1.0)
        product *= probability
        survival.append(0.0 if product < epsilon else product)
    return survival


def choose_verify_width(
    confidence: Sequence[float],
    *,
    min_width: int = 1,
    max_width: int | None = None,
    survival_threshold: float | None = None,
) -> int:
    """Choose a compact verify width for one request.

    ``min_width`` is the number of draft tokens always verified.  When a
    threshold is configured, the width is the longest confidence-surviving
    prefix.  With no threshold this returns the full draft block, which is the
    SGLang ``static``/verify-all behavior and the safe default for a new
    deployment.

    This is intentionally a pure fallback policy.  A profiled multi-request
    SPS table can replace it later without changing the draft or target state
    contracts.
    """

    available = len(confidence)
    if available <= 0:
        raise ValueError("DSpark confidence must contain at least one position")
    if max_width is None:
        max_width = available
    if not 1 <= min_width <= max_width <= available:
        raise ValueError(
            "DSpark verify width bounds must satisfy "
            f"1 <= min_width <= max_width <= {available}, got "
            f"{min_width}, {max_width}"
        )
    if survival_threshold is None:
        return max_width
    threshold = float(survival_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"DSpark survival threshold must be in [0,1], got {threshold}")

    width = min_width
    for position, probability in enumerate(survival_prefix(confidence), start=1):
        if position > max_width:
            break
        if probability >= threshold:
            width = position
        else:
            break
    return width


def choose_verify_widths(
    confidences: Sequence[Sequence[float]],
    *,
    min_width: int = 1,
    max_width: int | None = None,
    survival_threshold: float | None = None,
) -> list[int]:
    """Apply :func:`choose_verify_width` to a request batch."""

    return [
        choose_verify_width(
            row,
            min_width=min_width,
            max_width=max_width,
            survival_threshold=survival_threshold,
        )
        for row in confidences
    ]
