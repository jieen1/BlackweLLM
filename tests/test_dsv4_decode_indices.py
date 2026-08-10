from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from runtime.kernels import dsv4_decode_indices  # noqa: E402


def _expected_swa(
    positions: list[int],
    *,
    window: int,
    slot_ids: list[int] | None = None,
    pages_per_slot: int | None = None,
    page_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    lengths: list[int] = []
    for index, position in enumerate(positions):
        row: list[int] = []
        for col in range(window):
            if position >= window - 1:
                value = (position - window + 1 + col) % window
            elif col <= position:
                value = col
            else:
                value = -1
            if value >= 0 and slot_ids is not None:
                assert pages_per_slot is not None
                assert page_size is not None
                value += slot_ids[index] * pages_per_slot * page_size
            row.append(value)
        rows.append(row)
        lengths.append(min(position + 1, window))
    return torch.tensor(rows, dtype=torch.int32), torch.tensor(lengths, dtype=torch.int32)


def _expected_comp(
    positions: list[int],
    *,
    ratio: int,
    max_comp: int,
    slot_ids: list[int] | None = None,
    pages_per_slot: int | None = None,
    page_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    lengths: list[int] = []
    for index, position in enumerate(positions):
        count = min((position + 1) // ratio, max_comp)
        row: list[int] = []
        for col in range(max_comp):
            value = col if col < count else -1
            if value >= 0 and slot_ids is not None:
                assert pages_per_slot is not None
                assert page_size is not None
                value += slot_ids[index] * pages_per_slot * page_size
            row.append(value)
        rows.append(row)
        lengths.append(count)
    return torch.tensor(rows, dtype=torch.int32), torch.tensor(lengths, dtype=torch.int32)


def test_decode_swa_indices_b1_contract_is_unchanged() -> None:
    indices = dsv4_decode_indices.decode_swa_indices(
        torch.tensor([5], dtype=torch.int64),
        8,
        device="cpu",
    )
    expected, lengths = _expected_swa([5], window=8)
    assert indices.shape == (1, 8)
    assert torch.equal(indices, expected)
    assert torch.equal((indices >= 0).sum(dim=-1).to(torch.int32), lengths)


def test_decode_swa_indices_cpu_batched_rows_and_lengths() -> None:
    positions = torch.tensor([0, 3, 7, 9], dtype=torch.int64)
    indices, lengths = dsv4_decode_indices.decode_swa_indices(
        positions,
        8,
        device="cpu",
        return_lengths=True,
    )
    expected_indices, expected_lengths = _expected_swa([0, 3, 7, 9], window=8)
    assert torch.equal(indices, expected_indices)
    assert torch.equal(lengths, expected_lengths)


def test_decode_swa_indices_cpu_slot_offsets_emit_raw_ids() -> None:
    positions = torch.tensor([1, 7, 8], dtype=torch.int64)
    slot_ids = torch.tensor([0, 2, 5], dtype=torch.int64)
    indices, lengths = dsv4_decode_indices.decode_swa_indices(
        positions,
        8,
        device="cpu",
        slot_ids=slot_ids,
        pages_per_slot=1,
        page_size=256,
        return_lengths=True,
    )
    expected_indices, expected_lengths = _expected_swa(
        [1, 7, 8],
        window=8,
        slot_ids=[0, 2, 5],
        pages_per_slot=1,
        page_size=256,
    )
    assert torch.equal(indices, expected_indices)
    assert torch.equal(lengths, expected_lengths)
    assert int(indices[2, 0]) == 5 * 256 + 1
    assert int(indices[0, 2]) == -1


def test_decode_swa_indices_rejects_incomplete_slot_metadata() -> None:
    with pytest.raises(ValueError, match="slot_ids require both pages_per_slot and page_size"):
        dsv4_decode_indices.decode_swa_indices(
            torch.tensor([0, 1], dtype=torch.int64),
            8,
            device="cpu",
            slot_ids=torch.tensor([0, 1], dtype=torch.int64),
            pages_per_slot=1,
        )


def test_decode_comp_indices_b1_contract_is_unchanged() -> None:
    comp, lengths = dsv4_decode_indices.decode_comp_indices(
        torch.tensor([9], dtype=torch.int64),
        4,
        8,
        device="cpu",
    )
    expected_comp, expected_lengths = _expected_comp([9], ratio=4, max_comp=8)
    assert comp.shape == (1, 8)
    assert torch.equal(comp, expected_comp)
    assert torch.equal(lengths, expected_lengths)


def test_decode_comp_indices_cpu_ratio4_slot_offsets_emit_raw_ids() -> None:
    comp, lengths = dsv4_decode_indices.decode_comp_indices(
        torch.tensor([0, 3, 4, 19], dtype=torch.int64),
        4,
        6,
        device="cpu",
        slot_ids=torch.tensor([0, 1, 1, 3], dtype=torch.int64),
        pages_per_slot=2,
        page_size=64,
    )
    expected_comp, expected_lengths = _expected_comp(
        [0, 3, 4, 19],
        ratio=4,
        max_comp=6,
        slot_ids=[0, 1, 1, 3],
        pages_per_slot=2,
        page_size=64,
    )
    assert torch.equal(comp, expected_comp)
    assert torch.equal(lengths, expected_lengths)
    assert int(comp[0, 0]) == -1
    assert int(comp[3, 4]) == 3 * 2 * 64 + 4


def test_decode_comp_indices_cpu_ratio128_batched_rows() -> None:
    comp, lengths = dsv4_decode_indices.decode_comp_indices(
        torch.tensor([0, 127, 128, 255], dtype=torch.int64),
        128,
        3,
        device="cpu",
        slot_ids=torch.tensor([0, 2, 2, 4], dtype=torch.int64),
        pages_per_slot=4,
        page_size=2,
    )
    expected_comp, expected_lengths = _expected_comp(
        [0, 127, 128, 255],
        ratio=128,
        max_comp=3,
        slot_ids=[0, 2, 2, 4],
        pages_per_slot=4,
        page_size=2,
    )
    assert torch.equal(comp, expected_comp)
    assert torch.equal(lengths, expected_lengths)


def test_compile_decode_swa_indices_sm120_uses_sm120_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        dsv4_decode_indices._decode_swa_indices_raw_kernel,  # noqa: SLF001
        "create_binder",
        lambda: None,
    )

    def fake_compile(src, *, target, options):
        seen["target"] = target
        seen["options"] = options
        seen["signature"] = src.signature
        seen["constexprs"] = src.constants
        return "compiled-swa"

    monkeypatch.setattr(dsv4_decode_indices.triton, "compile", fake_compile)

    result = dsv4_decode_indices.compile_decode_swa_indices_sm120(
        window=128,
        with_slot_offsets=True,
    )

    assert result == "compiled-swa"
    assert seen["target"].backend == "cuda"
    assert seen["target"].arch == 120
    assert seen["signature"]["page_offset_scale"] == "i64"
    assert 128 in seen["constexprs"].values()
    assert seen["options"]["num_warps"] == 1


def test_compile_decode_comp_indices_sm120_uses_sm120_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        dsv4_decode_indices._decode_comp_indices_raw_kernel,  # noqa: SLF001
        "create_binder",
        lambda: None,
    )

    def fake_compile(src, *, target, options):
        seen["target"] = target
        seen["options"] = options
        seen["signature"] = src.signature
        seen["constexprs"] = src.constants
        return "compiled-comp"

    monkeypatch.setattr(dsv4_decode_indices.triton, "compile", fake_compile)

    result = dsv4_decode_indices.compile_decode_comp_indices_sm120(
        ratio=128,
        max_comp=32,
        with_slot_offsets=True,
    )

    assert result == "compiled-comp"
    assert seen["target"].backend == "cuda"
    assert seen["target"].arch == 120
    assert seen["signature"]["page_offset_scale"] == "i64"
    assert set(seen["constexprs"].values()) == {128, 32}
    assert seen["options"]["num_stages"] == 1


_GPU_TESTS_ENABLED = os.environ.get("DSV4_RUN_GPU_TESTS") == "1"


@pytest.mark.skipif(
    not _GPU_TESTS_ENABLED or not torch.cuda.is_available(),
    reason="set DSV4_RUN_GPU_TESTS=1 and provide CUDA to run GPU coverage",
)
def test_decode_swa_indices_cuda_matches_cpu_reference() -> None:
    positions = torch.tensor([2, 7, 10], dtype=torch.int64, device="cuda")
    slot_ids = torch.tensor([0, 1, 3], dtype=torch.int64, device="cuda")
    indices, lengths = dsv4_decode_indices.decode_swa_indices(
        positions,
        8,
        device="cuda",
        slot_ids=slot_ids,
        pages_per_slot=1,
        page_size=256,
        return_lengths=True,
    )
    expected_indices, expected_lengths = _expected_swa(
        [2, 7, 10],
        window=8,
        slot_ids=[0, 1, 3],
        pages_per_slot=1,
        page_size=256,
    )
    assert torch.equal(indices.cpu(), expected_indices)
    assert torch.equal(lengths.cpu(), expected_lengths)


@pytest.mark.skipif(
    not _GPU_TESTS_ENABLED or not torch.cuda.is_available(),
    reason="set DSV4_RUN_GPU_TESTS=1 and provide CUDA to run GPU coverage",
)
@pytest.mark.parametrize(
    ("ratio", "max_comp", "positions", "slot_ids", "pages_per_slot", "page_size"),
    [
        (4, 8, [0, 3, 7, 15], [0, 1, 2, 3], 2, 64),
        (128, 3, [0, 127, 255], [0, 2, 4], 4, 2),
    ],
    ids=["ratio4", "ratio128"],
)
def test_decode_comp_indices_cuda_matches_cpu_reference(
    ratio: int,
    max_comp: int,
    positions: list[int],
    slot_ids: list[int],
    pages_per_slot: int,
    page_size: int,
) -> None:
    comp, lengths = dsv4_decode_indices.decode_comp_indices(
        torch.tensor(positions, dtype=torch.int64, device="cuda"),
        ratio,
        max_comp,
        device="cuda",
        slot_ids=torch.tensor(slot_ids, dtype=torch.int64, device="cuda"),
        pages_per_slot=pages_per_slot,
        page_size=page_size,
    )
    expected_comp, expected_lengths = _expected_comp(
        positions,
        ratio=ratio,
        max_comp=max_comp,
        slot_ids=slot_ids,
        pages_per_slot=pages_per_slot,
        page_size=page_size,
    )
    assert torch.equal(comp.cpu(), expected_comp)
    assert torch.equal(lengths.cpu(), expected_lengths)
