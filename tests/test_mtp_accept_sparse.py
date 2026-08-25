import pytest

torch = pytest.importorskip("torch")

from runtime.mtp_accept import sample_accept_reject_sparse  # noqa: E402


def test_sparse_accept_reject_accepts_matching_path_and_samples_bonus():
    draft_tokens = [2, 3]
    draft_indices = torch.tensor([[2, 4], [3, 5]])
    draft_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target_probs = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    decision = sample_accept_reject_sparse(
        draft_tokens,
        draft_indices,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(7),
    )
    assert decision == {
        "num_accepted": 2,
        "committed": [2, 3, 4],
        "rejected_at": None,
    }


def test_sparse_accept_reject_residual_subtracts_candidate_mass():
    draft_tokens = [2]
    draft_indices = torch.tensor([[2, 4]])
    draft_probs = torch.tensor([[0.8, 0.2]])
    target_probs = torch.tensor(
        [[0.4, 0.2, 0.0, 0.2, 0.2], [0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    # The target probability at token 2 is zero, so rejection is forced.  The
    # residual has mass at token 0, 1, and 3 only; token 4's q mass is removed.
    decision = sample_accept_reject_sparse(
        draft_tokens,
        draft_indices,
        draft_probs,
        target_probs,
        generator=torch.Generator().manual_seed(1),
    )
    assert decision["num_accepted"] == 0
    assert decision["rejected_at"] == 0
    assert decision["committed"][0] in {0, 1, 3}
