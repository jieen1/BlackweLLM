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
pytest.importorskip("b12x")

from runtime.model import qwen36_model as qwen36_model_module  # noqa: E402
from runtime.model.qwen36_model import (  # noqa: E402
    GdnLayerState,
    Qwen36GatedDeltaNet,
    commit_spec_snapshot,
)


class TestGdnFusedGateSwitch:
    def test_fused_gate_is_the_default_with_explicit_opt_out(self, monkeypatch) -> None:
        monkeypatch.delenv(qwen36_model_module.QSR_QWEN36_GDN_FUSED_GATES_ENV, raising=False)
        assert qwen36_model_module._qwen36_gdn_fused_gates_enabled() is True

        monkeypatch.setenv(qwen36_model_module.QSR_QWEN36_GDN_FUSED_GATES_ENV, "0")
        assert qwen36_model_module._qwen36_gdn_fused_gates_enabled() is False

        monkeypatch.setenv(qwen36_model_module.QSR_QWEN36_GDN_FUSED_GATES_ENV, "1")
        assert qwen36_model_module._qwen36_gdn_fused_gates_enabled() is True


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


class TestSpecRowAddressing:
    def test_rows_match_snapshot_oracle_without_mutating_source(self, monkeypatch) -> None:
        """K+1 row addressing must preserve the old state oracle exactly.

        A row path can pass shape checks while accidentally reading the live
        state after the first candidate, or while writing candidate ``j`` to
        row ``j-1``. That failure is invisible until a partial accept, when
        the next round resumes from the wrong recurrent/conv state. A tiny
        deterministic recurrent stub lets this CPU test compare every row
        and output against the legacy clone-based path without requiring a
        CUDA driver; the real FLA kernel remains covered by the GPU probe.
        """
        config = {
            "hidden_size": 8,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 3,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }
        layer = Qwen36GatedDeltaNet(config, layer_idx=0, quantized={})
        layer = layer.float()
        # Seeding BEFORE construction does not make this layer deterministic.
        # Qwen36GatedDeltaNet allocates its parameters uninitialized, expecting
        # a checkpoint to overwrite them -- measured: 8 of its 9 parameters
        # differ between two constructions under the same seed. So the values
        # come from whatever the allocator hands back, which depends on what
        # ran before. That is why this test passed alone and failed in the
        # suite: the comparison below is between two `spec_forward` paths over
        # the SAME weights, and with garbage weights the two paths can diverge
        # for reasons that have nothing to do with either one being wrong.
        #
        # Fill every parameter explicitly instead, so the fixture is hermetic.
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for param in layer.parameters():
                param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5, generator=gen))
        source = layer.new_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
        source.conv_state.copy_(
            torch.arange(source.conv_state.numel()).reshape_as(source.conv_state)
        )
        source.recurrent_state.copy_(
            torch.arange(source.recurrent_state.numel()).reshape_as(source.recurrent_state)
        )
        source.has_previous_state = True
        hidden = torch.randn(1, 4, config["hidden_size"])

        def fake_recurrent(query, key, value, *, initial_state, **kwargs):
            del query, key, kwargs
            last_state = initial_state + value[:, 0].unsqueeze(1)
            return value, last_state

        monkeypatch.setattr(
            qwen36_model_module,
            "fused_recurrent_gated_delta_rule",
            fake_recurrent,
        )
        snapshot_output, snapshots = layer.spec_forward(hidden, source)
        assert snapshots is not None

        rows = [
            GdnLayerState(
                conv_state=torch.zeros_like(source.conv_state),
                recurrent_state=torch.zeros_like(source.recurrent_state),
                has_previous_state=False,
            )
            for _ in range(hidden.shape[1])
        ]
        row_output, row_snapshots = layer.spec_forward(hidden, source, spec_state_rows=rows)

        assert row_snapshots is None
        assert torch.equal(row_output, snapshot_output)
        for row, expected in zip(rows, snapshots[1:], strict=True):
            assert torch.equal(row.conv_state, expected.conv_state)
            assert torch.equal(row.recurrent_state, expected.recurrent_state)
            assert row.has_previous_state == expected.has_previous_state

        batched_snapshot_output, batched_snapshots = layer.spec_forward(
            hidden, source, batch_large_projections=True
        )
        assert batched_snapshots is not None
        batched_rows = [
            GdnLayerState(
                conv_state=torch.zeros_like(source.conv_state),
                recurrent_state=torch.zeros_like(source.recurrent_state),
                has_previous_state=False,
            )
            for _ in range(hidden.shape[1])
        ]
        batched_row_output, batched_row_snapshots = layer.spec_forward(
            hidden,
            source,
            spec_state_rows=batched_rows,
            batch_large_projections=True,
        )

        assert batched_row_snapshots is None
        assert torch.equal(batched_row_output, batched_snapshot_output)
        for row, expected in zip(batched_rows, batched_snapshots[1:], strict=True):
            assert torch.equal(row.conv_state, expected.conv_state)
            assert torch.equal(row.recurrent_state, expected.recurrent_state)
            assert row.has_previous_state == expected.has_previous_state

    def test_batch_rows_keep_verify_requests_disjoint(self, monkeypatch) -> None:
        """M-2 writes each request's candidate states to its own rows."""
        config = {
            "hidden_size": 4,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 3,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }
        layer = Qwen36GatedDeltaNet(config, layer_idx=0, quantized={}).float()
        gen = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for param in layer.parameters():
                param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5, generator=gen))

        source = layer.new_state(batch=2, device=torch.device("cpu"), dtype=torch.float32)
        source.conv_state.copy_(
            torch.arange(source.conv_state.numel()).reshape_as(source.conv_state)
        )
        source.recurrent_state.copy_(
            torch.arange(source.recurrent_state.numel()).reshape_as(source.recurrent_state)
        )
        source.has_previous_state = True
        hidden = torch.randn(2, 3, config["hidden_size"])

        def fake_recurrent(query, key, value, *, initial_state, **kwargs):
            del query, key, kwargs
            last_state = initial_state + value[:, 0].unsqueeze(1)
            return value, last_state

        monkeypatch.setattr(qwen36_model_module, "fused_recurrent_gated_delta_rule", fake_recurrent)
        expected_output = torch.cat(
            [
                layer.spec_forward(
                    hidden[request : request + 1],
                    GdnLayerState(
                        conv_state=source.conv_state[request : request + 1],
                        recurrent_state=source.recurrent_state[request : request + 1],
                        has_previous_state=True,
                    ),
                )[0]
                for request in range(hidden.shape[0])
            ],
            dim=0,
        )
        rows = [
            [
                GdnLayerState(
                    conv_state=torch.zeros_like(source.conv_state[:1]),
                    recurrent_state=torch.zeros_like(source.recurrent_state[:1]),
                    has_previous_state=False,
                )
                for _ in range(hidden.shape[1])
            ]
            for _ in range(hidden.shape[0])
        ]

        output, snapshots = layer.spec_forward(hidden, source, spec_state_rows=rows)

        assert snapshots is None
        assert torch.equal(output, expected_output)
        assert not torch.equal(rows[0][-1].recurrent_state, rows[1][-1].recurrent_state)

    def test_indexed_rows_match_snapshot_oracle_with_permuted_destinations(
        self, monkeypatch
    ) -> None:
        """The graph-safe indexed write path must not depend on row order."""
        config = {
            "hidden_size": 4,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 3,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
        }
        layer = Qwen36GatedDeltaNet(config, layer_idx=0, quantized={}).float()
        gen = torch.Generator().manual_seed(19)
        with torch.no_grad():
            for param in layer.parameters():
                param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5, generator=gen))
        source = layer.new_state(batch=1, device=torch.device("cpu"), dtype=torch.float32)
        source.conv_state.fill_(0.25)
        source.recurrent_state.fill_(0.5)
        source.has_previous_state = True
        hidden = torch.randn(1, 3, config["hidden_size"])

        def fake_recurrent(query, key, value, *, initial_state, **kwargs):
            del query, key, kwargs
            last_state = initial_state + value[:, 0].unsqueeze(1)
            return value, last_state

        monkeypatch.setattr(qwen36_model_module, "fused_recurrent_gated_delta_rule", fake_recurrent)
        expected_output, snapshots = layer.spec_forward(hidden, source)
        assert snapshots is not None

        destination = torch.tensor([[4, 1, 3]], dtype=torch.long)
        conv_pool = torch.full((5, *source.conv_state.shape[1:]), -7.0)
        recurrent_pool = torch.full((5, *source.recurrent_state.shape[1:]), -11.0)
        output, returned_snapshots = layer.spec_forward(
            hidden,
            source,
            spec_destination_index=destination,
            spec_conv_pool=conv_pool,
            spec_recurrent_pool=recurrent_pool,
        )

        assert returned_snapshots is None
        assert torch.equal(output, expected_output)
        for step, destination_row in enumerate(destination[0].tolist(), start=1):
            assert torch.equal(
                conv_pool[destination_row : destination_row + 1], snapshots[step].conv_state
            )
            assert torch.equal(
                recurrent_pool[destination_row : destination_row + 1],
                snapshots[step].recurrent_state,
            )
        assert torch.all(conv_pool[0] == -7.0)
        assert torch.all(conv_pool[2] == -7.0)
        assert torch.all(recurrent_pool[0] == -11.0)
        assert torch.all(recurrent_pool[2] == -11.0)
