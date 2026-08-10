from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.dsv4_attn_kernel import (  # noqa: E402
    _forced_dsv4_h16,
    _pad_prefill_index_width,
)


@pytest.mark.parametrize(
    ("ratio", "seqlen", "expected"),
    [
        (0, 1, False),
        (4, 1, True),
        (128, 1, False),
        (4, 2, None),
        (128, 32, None),
    ],
)
def test_forced_dsv4_h16_only_applies_to_ratio4_decode(
    ratio: int,
    seqlen: int,
    expected: bool | None,
) -> None:
    assert _forced_dsv4_h16(ratio, seqlen) is expected


@pytest.mark.parametrize(
    ("width", "capacity", "expected_width"),
    [
        (0, 512, 0),
        (1, 512, 64),
        (64, 512, 64),
        (65, 512, 128),
        (511, 512, 512),
        (512, 512, 512),
    ],
)
def test_prefill_index_width_uses_native_chunk_buckets(
    width: int,
    capacity: int,
    expected_width: int,
) -> None:
    indices = torch.arange(2 * width, dtype=torch.int32).reshape(2, width)
    padded = _pad_prefill_index_width(indices, capacity)

    assert padded.shape == (2, expected_width)
    assert torch.equal(padded[:, :width], indices)
    assert torch.all(padded[:, width:] == -1)


def test_prefill_index_width_rejects_capacity_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds capacity"):
        _pad_prefill_index_width(torch.zeros(2, 65, dtype=torch.int32), 64)
