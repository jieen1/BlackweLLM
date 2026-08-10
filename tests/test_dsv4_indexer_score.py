"""Static/CPU coverage for the standalone DSV4 indexer long-context scorer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.kernels.dsv4_indexer_score import (  # noqa: E402
    compile_dsv4_indexer_score_batch_sm120,
    compile_dsv4_indexer_score_sm120,
    dsv4_indexer_score,
    dsv4_indexer_score_batch,
)
from runtime.model.dsv4_model import Dsv4Indexer  # noqa: E402


def _torch_oracle(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    dots = torch.einsum("bshd,nd->bshn", q, kv)
    return (dots.relu() * weights.unsqueeze(-1)).sum(dim=2)


def _torch_batch_slot_oracle(
    q: torch.Tensor,
    kv_slots: torch.Tensor,
    weights: torch.Tensor,
    slot_ids: torch.Tensor,
) -> torch.Tensor:
    kv_rows = kv_slots.index_select(0, slot_ids)
    dots = torch.einsum("bshd,bnd->bshn", q, kv_rows)
    return (dots.relu() * weights.unsqueeze(-1)).sum(dim=2)


def test_dsv4_indexer_score_rejects_cpu_contract() -> None:
    with pytest.raises(ValueError, match="requires CUDA q/kv/weights tensors"):
        dsv4_indexer_score(
            torch.zeros(1, 1, 64, 128, dtype=torch.bfloat16),
            torch.zeros(7, 128, dtype=torch.bfloat16),
            torch.zeros(1, 1, 64, dtype=torch.bfloat16),
        )


def test_dsv4_indexer_score_batch_rejects_cpu_contract() -> None:
    with pytest.raises(ValueError, match="requires CUDA q/kv/weights tensors"):
        dsv4_indexer_score_batch(
            torch.zeros(2, 1, 64, 128, dtype=torch.bfloat16),
            torch.zeros(3, 7, 128, dtype=torch.bfloat16),
            torch.zeros(2, 1, 64, dtype=torch.bfloat16),
            torch.tensor([1, 0], dtype=torch.int64),
        )


def test_indexer_score_entries_keeps_cpu_oracle_semantics() -> None:
    indexer = object.__new__(Dsv4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_heads = 2
    indexer.head_dim = 4
    indexer.num_slots = 1
    generator = torch.Generator().manual_seed(20260810)
    indexer.register_buffer(
        "kv_cache",
        torch.randn(1, 9, 4, generator=generator, dtype=torch.bfloat16),
    )
    q = torch.randn(1, 1, 2, 4, generator=generator, dtype=torch.bfloat16)
    weights = torch.randn(1, 1, 2, generator=generator, dtype=torch.bfloat16)

    actual = indexer._score_entries(q, weights, 7)
    dots = torch.einsum("bshd,btd->bsht", q, indexer.kv_cache[:, :7])
    expected = (dots.relu() * weights.unsqueeze(-1)).sum(dim=2)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rows", [1, 7, 32])
def test_dsv4_indexer_score_prefill_cpu_oracle_matches_serial_rows(rows: int) -> None:
    generator = torch.Generator().manual_seed(20260809 + rows)
    q = torch.randn(1, rows, 64, 128, generator=generator, dtype=torch.bfloat16)
    kv = torch.randn(17, 128, generator=generator, dtype=torch.bfloat16)
    weights = torch.randn(1, rows, 64, generator=generator, dtype=torch.bfloat16)

    actual = _torch_oracle(q, kv, weights)
    expected = torch.cat(
        [
            _torch_oracle(q[:, row : row + 1], kv, weights[:, row : row + 1])
            for row in range(rows)
        ],
        dim=1,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch_size, slot_ids_list", [(1, [0]), (2, [1, 0]), (4, [2, 0, 2, 1])])
def test_dsv4_indexer_score_batch_cpu_oracle_matches_serial_rows(
    batch_size: int,
    slot_ids_list: list[int],
) -> None:
    generator = torch.Generator().manual_seed(20260810 + batch_size)
    q = torch.randn(batch_size, 1, 64, 128, generator=generator, dtype=torch.bfloat16)
    kv_slots = torch.randn(3, 17, 128, generator=generator, dtype=torch.bfloat16)
    weights = torch.randn(batch_size, 1, 64, generator=generator, dtype=torch.bfloat16)
    slot_ids = torch.tensor(slot_ids_list, dtype=torch.int64)

    actual = _torch_batch_slot_oracle(q, kv_slots, weights, slot_ids)
    expected = torch.cat(
        [
            _torch_oracle(q[row : row + 1], kv_slots[slot], weights[row : row + 1])
            for row, slot in enumerate(slot_ids_list)
        ],
        dim=0,
    )

    assert torch.equal(actual, expected)


def test_dsv4_indexer_score_rejects_wrong_dtype_and_shape() -> None:
    if not torch.cuda.is_available():
        pytest.skip("dtype/shape gate needs CUDA tensors")
    device = torch.device("cuda")
    q = torch.zeros(1, 1, 64, 128, device=device, dtype=torch.float16)
    kv = torch.zeros(7, 128, device=device, dtype=torch.bfloat16)
    weights = torch.zeros(1, 1, 64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires q bf16"):
        dsv4_indexer_score(q, kv, weights)
    with pytest.raises(ValueError, match=r"requires kv shaped \[N, 128\]"):
        dsv4_indexer_score(
            torch.zeros(1, 1, 64, 128, device=device, dtype=torch.bfloat16),
            torch.zeros(7, 64, device=device, dtype=torch.bfloat16),
            weights,
        )
    with pytest.raises(ValueError, match=r"requires q rows R in \[1, 32\]"):
        dsv4_indexer_score(
            torch.zeros(1, 33, 64, 128, device=device, dtype=torch.bfloat16),
            kv,
            torch.zeros(1, 33, 64, device=device, dtype=torch.bfloat16),
        )


def test_dsv4_indexer_score_batch_rejects_bad_batch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("dtype/shape gate needs CUDA tensors")
    device = torch.device("cuda")
    kv = torch.zeros(3, 7, 128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"requires q batch B in \(1, 2, 4\)"):
        dsv4_indexer_score_batch(
            torch.zeros(3, 1, 64, 128, device=device, dtype=torch.bfloat16),
            kv,
            torch.zeros(3, 1, 64, device=device, dtype=torch.bfloat16),
            torch.tensor([0, 1, 2], device=device, dtype=torch.int64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU parity test is opt-in")
@pytest.mark.parametrize("rows", [1, 7, 32])
def test_dsv4_indexer_score_matches_torch_oracle(rows: int) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260809 + rows)
    q = torch.randn(1, rows, 64, 128, generator=generator, device=device, dtype=torch.bfloat16)
    kv = torch.randn(257, 128, generator=generator, device=device, dtype=torch.bfloat16)
    weights = torch.randn(1, rows, 64, generator=generator, device=device, dtype=torch.bfloat16)

    actual = dsv4_indexer_score(q, kv, weights)
    serial = torch.cat(
        [
            dsv4_indexer_score(
                q[:, row : row + 1].clone(),
                kv.clone(),
                weights[:, row : row + 1].clone(),
            )
            for row in range(rows)
        ],
        dim=1,
    )
    expected = _torch_oracle(q, kv, weights)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (1, rows, 257)
    torch.testing.assert_close(actual, serial, atol=0.0, rtol=0.0)
    # The existing production B1 scorer can differ from eager Torch by one BF16
    # rounding step on a small number of rows; prefill must stay exact to the
    # serial production path.
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU parity test is opt-in")
@pytest.mark.parametrize("batch_size, slot_ids_list", [(1, [0]), (2, [1, 0]), (4, [2, 0, 2, 1])])
def test_dsv4_indexer_score_batch_matches_serial_b1_torch_oracle(
    batch_size: int,
    slot_ids_list: list[int],
) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260809 + batch_size)
    q = torch.randn(
        batch_size,
        1,
        64,
        128,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_slots = torch.randn(
        3,
        257,
        128,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        batch_size,
        1,
        64,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    slot_ids = torch.tensor(slot_ids_list, device=device, dtype=torch.int64)

    actual = dsv4_indexer_score_batch(q, kv_slots, weights, slot_ids)
    expected = torch.cat(
        [
            _torch_oracle(q[row : row + 1], kv_slots[slot], weights[row : row + 1])
            for row, slot in enumerate(slot_ids_list)
        ],
        dim=0,
    )
    serial = torch.cat(
        [
            dsv4_indexer_score(
                q[row : row + 1].clone(),
                kv_slots[slot].clone(),
                weights[row : row + 1].clone(),
            )
            for row, slot in enumerate(slot_ids_list)
        ],
        dim=0,
    )

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (batch_size, 1, 257)
    torch.testing.assert_close(actual, serial, atol=0.0, rtol=0.0)
    # The exact production contract here is parity with the existing B1 scorer.
    # Eager Torch remains a useful cross-check, but on GPU it can differ by one
    # BF16 rounding step on some rows even when the serial kernel matches.
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.0)


def test_compile_dsv4_indexer_score_sm120_offline() -> None:
    try:
        compiled = compile_dsv4_indexer_score_sm120()
    except Exception as exc:  # pragma: no cover - environment/toolchain dependent
        pytest.skip(f"offline Triton compile unavailable: {exc}")
    assert compiled is not None


def test_compile_dsv4_indexer_score_batch_sm120_offline() -> None:
    try:
        compiled = compile_dsv4_indexer_score_batch_sm120()
    except Exception as exc:  # pragma: no cover - environment/toolchain dependent
        pytest.skip(f"offline Triton compile unavailable: {exc}")
    assert compiled is not None
