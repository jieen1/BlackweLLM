"""Device-side MoE grouping: deterministic counts / within, graph-safe.

``device_group_counts_into`` is pure torch (stable sort + scatter-add +
cumsum + arange); it needs no CUDA device, only torch.  It is the CUDA-graph
replacement for the atomic-cursor variant: the ``within`` index must be a
pure function of the route order so graph replays reproduce eager output.
"""
import pytest

torch = pytest.importorskip("torch")

from runtime.kernels.dsv4_grouping import (  # noqa: E402
    device_group_counts,
    device_group_counts_into,
)


def _reference_within(routes: torch.Tensor) -> torch.Tensor:
    """Stable within = number of prior routes to the same expert."""
    ref = torch.zeros(routes.numel(), dtype=torch.int32)
    seen = {}
    for r in range(routes.numel()):
        e = int(routes[r])
        ref[r] = seen.get(e, 0)
        seen[e] = seen.get(e, 0) + 1
    return ref


def test_counts_and_within_match_reference() -> None:
    R, E = 384, 256
    routes = torch.randint(0, E, (R,), dtype=torch.int32)
    counts = torch.empty(E, dtype=torch.int32)
    within = torch.empty(R, dtype=torch.int32)
    offsets = torch.empty(E, dtype=torch.int32)
    cursor = torch.empty(R, dtype=torch.int32)

    counts, within, offsets = device_group_counts_into(
        routes, counts, within, offsets, cursor
    )

    assert counts.sum().item() == R
    # every expert's routes occupy [offsets[e], offsets[e]+counts[e])
    assert (offsets + counts <= R).all()
    assert (offsets >= 0).all()
    # within[r] is a pure function of route order
    assert torch.equal(within, _reference_within(routes))


def test_permutation_invariance_and_determinism() -> None:
    """Same routes must produce identical results across repeated calls, and
    the same multiset in a different order must permute within accordingly."""
    R, E = 384, 256
    routes = torch.randint(0, E, (R,), dtype=torch.int32)
    scratch = [torch.empty(n, dtype=torch.int32) for n in (E, R, E, R)]

    a_counts, a_within, _ = device_group_counts_into(routes, *scratch)
    b_counts, b_within, _ = device_group_counts_into(routes, *scratch)

    assert torch.equal(a_counts, b_counts)
    assert torch.equal(a_within, b_within)

    # shuffling the routes permutes within the same way
    perm = torch.randperm(R)
    c_counts, c_within, _ = device_group_counts_into(routes[perm], *scratch)
    assert torch.equal(c_counts, a_counts)
    # within is per-expert stable ordinality, NOT position-invariant: the
    # permutation reorders same-expert elements, so within is its own
    # permutation of the same multiset (each expert keeps {0..count-1}).
    assert torch.equal(torch.sort(c_within).values, torch.sort(a_within).values)
    assert torch.equal(c_within, _reference_within(routes[perm]))


def test_single_expert_and_empty_experts() -> None:
    routes = torch.full((64,), 7, dtype=torch.int32)
    counts = torch.empty(256, dtype=torch.int32)
    within = torch.empty(64, dtype=torch.int32)
    offsets = torch.empty(256, dtype=torch.int32)
    cursor = torch.empty(64, dtype=torch.int32)

    counts, within, offsets = device_group_counts_into(
        routes, counts, within, offsets, cursor
    )

    assert counts[7].item() == 64
    assert counts.sum().item() == 64
    assert torch.equal(within, torch.arange(64, dtype=torch.int32))
    assert offsets[7].item() == 0


def test_cursor_must_be_arange_capacity() -> None:
    """cursor is the arange(R) output carrier, NOT a per-expert buffer.  A
    too-small cursor (e.g. sized to n_experts when R > n_experts) must fail
    loudly rather than silently resize inside a CUDA-graph capture."""
    R, E = 384, 256
    routes = torch.randint(0, E, (R,), dtype=torch.int32)
    scratch = [torch.empty(n, dtype=torch.int32) for n in (E, R, E, E)]
    with pytest.raises(ValueError, match="cursor"):
        device_group_counts_into(routes, *scratch)


def test_eager_wrapper_uses_full_capacity() -> None:
    R, E = 384, 256
    routes = torch.randint(0, E, (R,), dtype=torch.int32)
    counts, within, offsets = device_group_counts(routes)
    assert torch.equal(within, _reference_within(routes))
    assert counts.sum().item() == R
    assert tuple(offsets.shape) == (256,)
