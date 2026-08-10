from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.backends import dsv4_cudagraph  # noqa: E402


def test_batched_decode_graph_driver_forwards_persistent_buffers() -> None:
    seen: dict[str, object] = {}

    class FakeBackend:
        num_slots = 4
        max_seq_len = 256
        slot_layers = []

        def reset_slot(self, slot: int) -> None:
            seen.setdefault("resets", []).append(slot)

        def _forward_decode_batch(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            slot_ids: torch.Tensor,
            *,
            max_index_entries: int | None,
        ) -> torch.Tensor:
            seen["input_ids"] = input_ids
            seen["positions"] = positions
            seen["slot_ids"] = slot_ids
            seen["cap"] = max_index_entries
            return torch.zeros((2, 1, 8), dtype=torch.float32)

    driver = dsv4_cudagraph.Dsv4BatchedDecodeGraphDriver(
        backend=FakeBackend(),
        batch_size=2,
        max_index_entries=512,
        device="cpu",
    )
    input_ids = torch.tensor([[3], [7]], dtype=torch.long)
    positions = torch.tensor([11, 29], dtype=torch.long)
    slot_ids = torch.tensor([2, 0], dtype=torch.long)

    driver._copy_inputs(input_ids, positions, slot_ids)
    logits = driver._forward()

    assert seen["input_ids"] is driver._input_ids
    assert seen["positions"] is driver._positions
    assert seen["slot_ids"] is driver._slot_ids
    assert torch.equal(driver._input_ids, input_ids)
    assert torch.equal(driver._positions, positions)
    assert torch.equal(driver._slot_ids, slot_ids)
    assert seen["cap"] == 512
    assert logits.shape == (2, 1, 8)


def test_batched_decode_graph_driver_packs_host_inputs_in_one_surface() -> None:
    class FakeBackend:
        num_slots = 4
        max_seq_len = 256
        slot_layers = []

        def reset_slot(self, slot: int) -> None:
            return None

    driver = dsv4_cudagraph.Dsv4BatchedDecodeGraphDriver(
        backend=FakeBackend(),
        batch_size=2,
        device="cpu",
    )

    driver._copy_host_inputs([3, 7], [11, 29], [2, 0])

    assert torch.equal(
        driver._packed_inputs,
        torch.tensor([[3, 7], [11, 29], [2, 0]], dtype=torch.long),
    )
    assert (
        driver._input_ids.untyped_storage().data_ptr()
        == driver._packed_inputs.untyped_storage().data_ptr()
    )
    assert (
        driver._positions.untyped_storage().data_ptr()
        == driver._packed_inputs.untyped_storage().data_ptr()
    )
    assert (
        driver._slot_ids.untyped_storage().data_ptr()
        == driver._packed_inputs.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    ("input_shape", "positions_shape", "slot_shape", "message"),
    [
        ((2, 2), (2,), (2,), "[2, 1]"),
        ((2, 1), (1,), (2,), r"positions must be \[2\]"),
        ((2, 1), (2,), (1,), r"slot_ids must be \[2\]"),
    ],
)
def test_batched_decode_graph_driver_rejects_bad_shapes(
    input_shape: tuple[int, ...],
    positions_shape: tuple[int, ...],
    slot_shape: tuple[int, ...],
    message: str,
) -> None:
    class FakeBackend:
        num_slots = 4
        max_seq_len = 256
        slot_layers = []

        def reset_slot(self, slot: int) -> None:
            return None

    driver = dsv4_cudagraph.Dsv4BatchedDecodeGraphDriver(
        backend=FakeBackend(),
        batch_size=2,
        device="cpu",
    )
    input_ids = torch.zeros(input_shape, dtype=torch.long)
    positions = torch.zeros(positions_shape, dtype=torch.long)
    slot_ids = torch.zeros(slot_shape, dtype=torch.long)

    with pytest.raises(ValueError, match=message):
        driver._copy_inputs(input_ids, positions, slot_ids)


def test_batched_decode_graph_driver_rejects_bad_dtypes() -> None:
    class FakeBackend:
        num_slots = 4
        max_seq_len = 256
        slot_layers = []

        def reset_slot(self, slot: int) -> None:
            return None

    driver = dsv4_cudagraph.Dsv4BatchedDecodeGraphDriver(
        backend=FakeBackend(),
        batch_size=2,
        device="cpu",
    )
    input_ids = torch.zeros((2, 1), dtype=torch.int32)
    positions = torch.zeros((2,), dtype=torch.long)
    slot_ids = torch.zeros((2,), dtype=torch.long)

    with pytest.raises(ValueError, match="input_ids must be torch.long"):
        driver._copy_inputs(input_ids, positions, slot_ids)


def test_batched_decode_graph_driver_capture_inputs_use_batch_boundary() -> None:
    class FakeBackend:
        num_slots = 4
        max_seq_len = 2048
        slot_layers = []

        def reset_slot(self, slot: int) -> None:
            return None

    driver = dsv4_cudagraph.Dsv4BatchedDecodeGraphDriver(
        backend=FakeBackend(),
        batch_size=4,
        max_index_entries=512,
        device="cpu",
    )

    input_ids, positions, slot_ids = driver._capture_inputs()

    assert input_ids.shape == (4, 1)
    assert torch.equal(slot_ids, torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(positions, torch.tensor([0, 1, 2, 2047], dtype=torch.long))


def test_bucketed_batched_decode_graph_driver_picks_batch_and_cap(monkeypatch) -> None:
    class FakeIndexer:
        def __init__(self) -> None:
            self.kv_cache = torch.zeros(1, 16384, 4)

    class FakeLayer:
        def __init__(self) -> None:
            self.indexer = FakeIndexer()

    class FakeBackend:
        num_slots = 4
        max_seq_len = 65536
        slot_layers = [FakeLayer()]

        def reset_slot(self, slot: int) -> None:
            return None

    seen: list[tuple[int, int | None, tuple[int, int]]] = []

    class FakeDriver:
        def __init__(
            self,
            *,
            backend,
            batch_size,
            max_index_entries=None,
            graph_pool=None,
            **kwargs,
        ):
            self.backend = backend
            self.batch_size = batch_size
            self.max_index_entries = max_index_entries
            self.graph_pool = graph_pool or object()

        def capture(self) -> None:
            return None

        def replay(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            slot_ids: torch.Tensor,
        ) -> torch.Tensor:
            seen.append((self.batch_size, self.max_index_entries, tuple(positions.shape)))
            return torch.zeros((self.batch_size, 1, 8), dtype=torch.float32)

    monkeypatch.setattr(dsv4_cudagraph, "Dsv4BatchedDecodeGraphDriver", FakeDriver)

    graph = dsv4_cudagraph.build_batched_decode_graph_driver(
        backend=FakeBackend(),
        device="cpu",
    )
    graph.capture()

    assert tuple(graph._drivers) == (1, 2, 4)
    assert [cap for cap, _driver in graph._drivers[2]] == [512, 1024, 4096, 16384]

    graph.replay(
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros((2,), dtype=torch.long),
        torch.tensor([3, 1], dtype=torch.long),
        max_index_entries=900,
    )
    graph.replay(
        torch.zeros((4, 1), dtype=torch.long),
        torch.zeros((4,), dtype=torch.long),
        torch.tensor([3, 2, 1, 0], dtype=torch.long),
        max_index_entries=16384,
    )

    assert seen == [
        (2, 1024, (2,)),
        (4, 16384, (4,)),
    ]


def test_bucketed_batched_decode_graph_driver_rejects_uncaptured_batch() -> None:
    class FakeBackend:
        num_slots = 2
        max_seq_len = 1024
        slot_layers = []

        def reset_slot(self, slot: int) -> None:
            return None

    graph = dsv4_cudagraph.Dsv4BucketedBatchedDecodeGraphDriver(
        backend=FakeBackend(),
        device="cpu",
    )
    graph._drivers = {1: [(None, object())], 2: [(None, object())]}

    with pytest.raises(ValueError, match="batch_size 4"):
        graph._pick_driver(4, None)
