"""Unit tests for runtime/mtp_accept.py — CPU-only."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.mtp_accept import (  # noqa: E402
    determine_accept_reject,
    determine_accept_reject_batch,
    determine_accept_reject_from_predictions,
)


class TestDetermineAcceptReject:
    def _make_logits(self, predictions: list[int], vocab_size: int = 100):
        """Create logits tensor where argmax gives the specified predictions."""
        logits = torch.zeros(len(predictions), vocab_size)
        for i, pred in enumerate(predictions):
            logits[i, pred] = 10.0
        return logits

    def test_all_accepted(self):
        # K=3 drafts, all match verify predictions
        draft_tokens = [10, 20, 30, 40]  # anchor + 3 drafts
        # verify_logits[p] should predict draft_tokens[p+1]
        logits = self._make_logits([20, 30, 40, 50])  # last is bonus
        result = determine_accept_reject(draft_tokens, logits)
        assert result["num_accepted"] == 3
        assert result["committed"] == [20, 30, 40, 50]
        assert result["rejected_at"] is None

    def test_first_rejected(self):
        draft_tokens = [10, 20, 30, 40]
        # First verify predicts 99 (doesn't match draft 20)
        logits = self._make_logits([99, 30, 40, 50])
        result = determine_accept_reject(draft_tokens, logits)
        assert result["num_accepted"] == 0
        assert result["committed"] == [99]
        assert result["rejected_at"] == 0

    def test_middle_rejected(self):
        draft_tokens = [10, 20, 30, 40]
        # First matches (20), second doesn't (predicts 99 instead of 30)
        logits = self._make_logits([20, 99, 40, 50])
        result = determine_accept_reject(draft_tokens, logits)
        assert result["num_accepted"] == 1
        assert result["committed"] == [20, 99]
        assert result["rejected_at"] == 1

    def test_last_rejected(self):
        draft_tokens = [10, 20, 30, 40]
        # First two match, third doesn't
        logits = self._make_logits([20, 30, 99, 50])
        result = determine_accept_reject(draft_tokens, logits)
        assert result["num_accepted"] == 2
        assert result["committed"] == [20, 30, 99]
        assert result["rejected_at"] == 2

    def test_k1_all_accepted(self):
        draft_tokens = [10, 20]  # anchor + 1 draft
        logits = self._make_logits([20, 50])
        result = determine_accept_reject(draft_tokens, logits)
        assert result["num_accepted"] == 1
        assert result["committed"] == [20, 50]

    def test_k1_rejected(self):
        draft_tokens = [10, 20]
        logits = self._make_logits([99, 50])
        result = determine_accept_reject(draft_tokens, logits)
        assert result["num_accepted"] == 0
        assert result["committed"] == [99]


class TestDetermineAcceptRejectFromPredictions:
    """The single-sync DFlash path already has target argmax ids on the CPU."""

    def test_first_reject_keeps_recovery_out_of_accepted_count(self):
        result = determine_accept_reject_from_predictions(
            [10, 20, 30, 40],
            [99, 30, 40, 50],
        )
        assert result == {
            "num_accepted": 0,
            "committed": [99],
            "rejected_at": 0,
        }

    def test_middle_reject_returns_matches_plus_recovery(self):
        result = determine_accept_reject_from_predictions(
            [10, 20, 30, 40],
            [20, 99, 40, 50],
        )
        assert result == {
            "num_accepted": 1,
            "committed": [20, 99],
            "rejected_at": 1,
        }

    def test_full_accept_appends_target_bonus(self):
        result = determine_accept_reject_from_predictions(
            [10, 20, 30, 40],
            [20, 30, 40, 50],
        )
        assert result == {
            "num_accepted": 3,
            "committed": [20, 30, 40, 50],
            "rejected_at": None,
        }


class TestDetermineAcceptRejectBatch:
    def test_batch_matches_individual(self):
        """Batch result should match per-slot individual results."""
        k = 3
        vocab = 100
        slots = [0, 1]
        drafts = {
            0: [10, 20, 30, 40],
            1: [10, 50, 60, 70],
        }
        # Slot 0: all accepted (verify predicts 20,30,40,bonus=80)
        # Slot 1: first rejected (verify predicts 99)
        verify_logits = torch.zeros(2, k + 1, vocab)
        verify_logits[0, 0, 20] = 10.0
        verify_logits[0, 1, 30] = 10.0
        verify_logits[0, 2, 40] = 10.0
        verify_logits[0, 3, 80] = 10.0  # bonus
        verify_logits[1, 0, 99] = 10.0  # reject first
        verify_logits[1, 1, 60] = 10.0
        verify_logits[1, 2, 70] = 10.0
        verify_logits[1, 3, 85] = 10.0

        result = determine_accept_reject_batch(slots, drafts, verify_logits, k)
        assert result[0]["num_accepted"] == 3
        assert result[0]["committed"] == [20, 30, 40, 80]
        assert result[1]["num_accepted"] == 0
        assert result[1]["committed"] == [99]
