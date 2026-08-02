"""B3: unit tests for the O(1) half of GDN speculative-verify rollback.

``commit_spec_snapshot`` is pure tensor bookkeeping (no FLA kernel, no CUDA)
-- runs on CPU. The expensive, kernel-calling half
(``Qwen36GatedDeltaNet.spec_forward``, which materializes the snapshots this
function selects from) needs a real GDN layer and is proven correct on GPU
in ``scripts/b3_probe_gdn_spec_rollback.py`` instead, matching this repo's
existing convention (``b0_probe_*``/``b1_verify_*``) of putting
real-kernel/real-checkpoint correctness proofs in standalone scripts, not
in the CPU-only pytest suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
# runtime.model.qwen36_model imports fla/sparkinfer at module level (real GDN
# kernels + paged attention), even though this file's own tests never call
# either -- same guard tests/test_qwen36_slot_pool.py uses for the same reason.
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")

from runtime.model.qwen36_model import GdnLayerState, commit_spec_snapshot  # noqa: E402


def _state(seed: float) -> GdnLayerState:
    return GdnLayerState(
        conv_state=torch.full((1, 4, 4), seed),
        recurrent_state=torch.full((1, 2, 3, 3), seed),
        has_previous_state=True,
    )


def _snapshots(count: int) -> list[GdnLayerState]:
    """``count`` snapshots with distinguishable values: snapshot ``j``'s
    tensors are filled with ``float(j)``, so "did we copy snapshot 2, not
    1 or 3" is checkable by value, not just by not-raising."""
    return [_state(float(j)) for j in range(count)]


class TestCommitSpecSnapshotSelection:
    def test_selects_the_requested_index_not_the_last(self) -> None:
        live = _state(-1.0)
        snapshots = _snapshots(5)  # values 0..4

        commit_spec_snapshot(live, snapshots, accepted_count=2)

        assert torch.equal(live.conv_state, torch.full((1, 4, 4), 2.0))
        assert torch.equal(live.recurrent_state, torch.full((1, 2, 3, 3), 2.0))

    def test_accepted_count_zero_restores_the_untouched_anchor(self) -> None:
        """snapshots[0] is spec_forward's pre-loop clone of the anchor --
        this is the "every candidate rejected" outcome, and it must be
        indistinguishable from "spec_forward was never called" (the
        crash-safety property the docstring describes)."""
        live = _state(-1.0)
        snapshots = _snapshots(5)

        commit_spec_snapshot(live, snapshots, accepted_count=0)

        assert torch.equal(live.conv_state, torch.full((1, 4, 4), 0.0))

    def test_accepted_count_equal_to_k_restores_full_acceptance(self) -> None:
        live = _state(-1.0)
        snapshots = _snapshots(5)  # indices 0..4, so K=4 candidates

        commit_spec_snapshot(live, snapshots, accepted_count=4)

        assert torch.equal(live.conv_state, torch.full((1, 4, 4), 4.0))

    def test_sets_has_previous_state_true(self) -> None:
        live = _state(-1.0)
        live.has_previous_state = False  # pretend it was somehow cleared
        snapshots = _snapshots(3)

        commit_spec_snapshot(live, snapshots, accepted_count=1)

        assert live.has_previous_state is True


class TestCommitSpecSnapshotCopyDiscipline:
    def test_writes_through_copy_not_rebind(self) -> None:
        """B0-5/B2 discipline: the live tensors' Python identity must
        survive a commit -- a rebind would silently detach a slot-pool
        view from the pool it is a view into (the exact bug class B2's
        module docstring names for `.recurrent_state = ...` assignment)."""
        live = _state(-1.0)
        conv_id, rec_id = id(live.conv_state), id(live.recurrent_state)
        snapshots = _snapshots(3)

        commit_spec_snapshot(live, snapshots, accepted_count=1)

        assert id(live.conv_state) == conv_id
        assert id(live.recurrent_state) == rec_id

    def test_does_not_mutate_the_snapshot_it_copied_from(self) -> None:
        live = _state(-1.0)
        snapshots = _snapshots(3)
        chosen_conv_before = snapshots[1].conv_state.clone()

        commit_spec_snapshot(live, snapshots, accepted_count=1)
        live.conv_state.fill_(99.0)  # mutate the live buffer afterward

        assert torch.equal(snapshots[1].conv_state, chosen_conv_before)


class TestCommitSpecSnapshotBoundsChecking:
    def test_rejects_negative_accepted_count(self) -> None:
        live = _state(0.0)
        snapshots = _snapshots(3)
        with pytest.raises(ValueError, match="out of range"):
            commit_spec_snapshot(live, snapshots, accepted_count=-1)

    def test_rejects_accepted_count_at_len(self) -> None:
        """len(snapshots) == K+1 valid indices are 0..K; index K+1 (here,
        len(snapshots)) does not exist -- off-by-one is the exact class of
        bug a K+1-vs-K mixup would produce."""
        live = _state(0.0)
        snapshots = _snapshots(3)
        with pytest.raises(ValueError, match="out of range"):
            commit_spec_snapshot(live, snapshots, accepted_count=3)

    def test_rejects_accepted_count_past_len(self) -> None:
        live = _state(0.0)
        snapshots = _snapshots(3)
        with pytest.raises(ValueError, match="out of range"):
            commit_spec_snapshot(live, snapshots, accepted_count=10)
