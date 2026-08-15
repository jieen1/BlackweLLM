"""Router kernel parity: Triton fused selection vs the eager Gate semantics.

The eager Dsv4Gate parity against the official reference is already proven
(test_dsv4_reference_parts.py); this pins the kernel to the same math.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from runtime.kernels.dsv4_router import (  # noqa: E402
    dsv4_route_hashed_scores,
    dsv4_route_scores,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

EXPERTS = 256
TOP_K = 6
ROUTE_SCALE = 1.5


def eager_route(logits: torch.Tensor, bias: torch.Tensor):
    """Dsv4Gate semantics (tests/test_dsv4_moe.py reference transcription)."""
    scores = torch.nn.functional.softplus(logits).sqrt()
    selection = scores + bias
    indices = selection.topk(TOP_K, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTE_SCALE
    return weights, indices


def eager_hashed_route(logits: torch.Tensor, supplied_ids: torch.Tensor):
    """Dsv4Gate hash semantics: skip selection, not score computation."""
    scores = torch.nn.functional.softplus(logits).sqrt()
    indices = supplied_ids.to(torch.int64)
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTE_SCALE
    return weights, indices


@pytest.mark.parametrize("tokens", [1, 6, 37, 512])
def test_kernel_matches_eager(tokens: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(20260807 + tokens)
    logits = torch.randn(tokens, EXPERTS, generator=gen, device="cuda") * 3
    bias = torch.randn(EXPERTS, generator=gen, device="cuda") * 0.1

    ew, ei = eager_route(logits, bias)
    kw, ki = dsv4_route_scores(logits, bias, TOP_K, ROUTE_SCALE)

    assert torch.equal(ki, ei), (
        f"index mismatch at rows {(ki != ei).any(dim=1).nonzero()[:3].tolist()}"
    )
    assert torch.allclose(kw, ew, rtol=1e-5, atol=1e-6)
    assert torch.allclose(
        kw.sum(dim=-1), torch.full((tokens,), ROUTE_SCALE, device="cuda"), rtol=1e-4
    )


def test_selection_bias_does_not_leak_into_weights() -> None:
    """Bias must move selection only: huge bias flips the pick, weights still
    renormalize over the unbiased scores."""
    logits = torch.zeros(1, EXPERTS, device="cuda")
    logits[0, 7] = 5.0  # clear score winner
    bias = torch.full((EXPERTS,), 0.0, device="cuda")
    bias[7] = -1000.0  # but forbid it via bias
    bias[11] = 100.0  # force expert 11 first
    weights, indices = dsv4_route_scores(logits, bias, TOP_K, ROUTE_SCALE)
    assert indices[0, 0] == 11
    assert 7 not in indices[0].tolist()
    assert abs(weights.sum().item() - ROUTE_SCALE) < 1e-4


def test_softplus_threshold_branch() -> None:
    """Logits above softplus' threshold (20) take the identity branch."""
    logits = torch.tensor([[30.0] + [-30.0] * 255], device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    weights, indices = dsv4_route_scores(logits, bias, TOP_K, ROUTE_SCALE)
    assert indices[0, 0] == 0
    # softplus(30)=30 -> sqrt(30); the five -30 scores are ~0, so expert 0
    # carries essentially the whole renormalized mass
    assert weights[0, 0] > ROUTE_SCALE * 0.999


@pytest.mark.parametrize("tokens", [1, 6, 37, 512])
def test_hashed_kernel_strictly_matches_eager(tokens: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(20260810 + tokens)
    logits = torch.randn(tokens, EXPERTS, generator=gen, device="cuda") * 3
    supplied_ids = torch.stack(
        [torch.randperm(EXPERTS, generator=gen, device="cuda")[:TOP_K] for _ in range(tokens)]
    ).to(torch.int32)

    expected_weights, expected_indices = eager_hashed_route(logits, supplied_ids)
    actual_weights, actual_indices = dsv4_route_hashed_scores(logits, supplied_ids, ROUTE_SCALE)

    torch.testing.assert_close(actual_indices, expected_indices, atol=0, rtol=0)
    # Both paths use fp32 throughout.  The small tolerance covers libdevice
    # softplus/log lowering while remaining far below a BF16 routing ulp.
    torch.testing.assert_close(actual_weights, expected_weights, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual_weights.sum(dim=-1),
        torch.full((tokens,), ROUTE_SCALE, device="cuda"),
        atol=2e-6,
        rtol=2e-6,
    )


def test_hashed_kernel_preserves_id_order_and_duplicates_exactly() -> None:
    # softplus(36)=36 through the threshold branch and sqrt(36)=6 exactly;
    # 6 / (6*6) * 1.5 = 0.25 is exactly representable.  This pins output bits
    # as well as supplied-id order and duplicate preservation.
    logits = torch.full((2, EXPERTS), 36.0, dtype=torch.float32, device="cuda")
    supplied_ids = torch.tensor(
        [[17, 3, 17, 255, 0, 3], [8, 7, 6, 5, 4, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    weights, indices = dsv4_route_hashed_scores(logits, supplied_ids, ROUTE_SCALE)

    assert torch.equal(indices, supplied_ids.to(torch.int64))
    assert torch.equal(weights, torch.full_like(weights, 0.25))


def test_hashed_kernel_softplus_threshold_and_device_ids() -> None:
    logits = torch.full((1, EXPERTS), -30.0, dtype=torch.float32, device="cuda")
    logits[0, 9] = 30.0
    supplied_ids = torch.tensor([[9, 10, 11, 12, 13, 14]], dtype=torch.int32, device="cuda")
    expected_weights, expected_indices = eager_hashed_route(logits, supplied_ids)
    actual_weights, actual_indices = dsv4_route_hashed_scores(logits, supplied_ids, ROUTE_SCALE)

    assert torch.equal(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, atol=2e-7, rtol=2e-6)
    assert actual_weights[0, 0] > ROUTE_SCALE * 0.999
