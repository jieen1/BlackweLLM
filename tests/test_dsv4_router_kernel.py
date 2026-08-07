"""Router kernel parity: Triton fused selection vs the eager Gate semantics.

The eager Dsv4Gate parity against the official reference is already proven
(test_dsv4_reference_parts.py); this pins the kernel to the same math.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from runtime.kernels.dsv4_router import dsv4_route_scores  # noqa: E402

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
