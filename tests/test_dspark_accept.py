"""Bit-exact tests for the device-side DSpark greedy epilogue."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.kernels.dspark_accept import (  # noqa: E402
    greedy_accept_device,
    greedy_accept_ragged,
)


def _logits(predictions: list[list[int]], vocab: int = 16) -> torch.Tensor:
    out = torch.full((len(predictions), len(predictions[0]), vocab), -100.0, dtype=torch.float32)
    for row, values in enumerate(predictions):
        for position, token in enumerate(values):
            out[row, position, token] = 100.0
    return out


def test_greedy_accept_device_matches_prefix_rule() -> None:
    # Row 0 accepts two drafts and then commits the target recovery token;
    # row 1 rejects the first draft and commits its position-0 prediction.
    candidates = torch.tensor([[2, 3, 4, 5], [7, 8, 9, 10]], dtype=torch.long)
    logits = _logits([[3, 4, 12, 12], [6, 11, 12, 13]])

    accepted, committed = greedy_accept_device(candidates, logits, gamma=3)

    assert accepted.tolist() == [2, 0]
    assert committed.tolist() == [[3, 4, 12, 0], [6, 0, 0, 0]]


def test_greedy_accept_device_supports_flat_logits_and_reused_outputs() -> None:
    candidates = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits = _logits([[2, 3, 4]]).view(3, 16)
    predicted = torch.empty((1, 3), dtype=torch.long)
    accepted = torch.empty((1,), dtype=torch.int32)
    committed = torch.empty((1, 3), dtype=torch.long)

    result = greedy_accept_device(
        candidates,
        logits,
        gamma=2,
        predicted=predicted,
        accepted_out=accepted,
        committed_out=committed,
    )

    assert result[0] is accepted
    assert result[1] is committed
    assert accepted.tolist() == [2]
    assert committed.tolist() == [[2, 3, 4]]


def test_greedy_accept_device_supports_anchor_only_verify() -> None:
    candidates = torch.tensor([[7]], dtype=torch.long)
    logits = _logits([[11]])

    accepted, committed = greedy_accept_device(candidates, logits, gamma=0)

    assert accepted.tolist() == [0]
    assert committed.tolist() == [[11]]


def test_greedy_accept_ragged_keeps_request_local_prefixes() -> None:
    # Request 0 verifies anchor+3 drafts; request 1 verifies anchor+1 draft.
    # The compact rows are [request0 rows, request1 rows], with no padded
    # request-local tail between them.
    candidates = torch.tensor([2, 3, 4, 5, 7, 8], dtype=torch.long)
    logits = _logits(
        [[3], [4], [12], [12], [8], [13]],
    ).reshape(6, 16)

    accepted, committed = greedy_accept_ragged(
        candidates,
        logits,
        verify_lens=[4, 2],
        max_gamma=3,
        q_indptr=torch.tensor([0, 4, 6], dtype=torch.int32),
    )

    assert accepted.tolist() == [2, 1]
    assert committed.tolist() == [[3, 4, 12, 0], [8, 13, 0, 0]]
