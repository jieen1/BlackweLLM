"""CPU regression tests for Sparkinfer MoE routed-output ownership."""
# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sparkinfer")

from runtime.backends import laguna_sparkinfer_moe as moe


def test_output_arena_grows_without_shrinking() -> None:
    arena = moe.SparkinferMoEOutputArena()

    initial = arena.acquire(torch.empty(16, moe.HIDDEN_SIZE, dtype=torch.bfloat16))
    initial_ptr = initial.data_ptr()
    grown = arena.acquire(torch.empty(64, moe.HIDDEN_SIZE, dtype=torch.bfloat16))
    reused = arena.acquire(torch.empty(8, moe.HIDDEN_SIZE, dtype=torch.bfloat16))

    assert initial.shape == (16, moe.HIDDEN_SIZE)
    assert grown.shape == (64, moe.HIDDEN_SIZE)
    assert arena.buffer is not None
    assert arena.buffer.shape == (64, moe.HIDDEN_SIZE)
    assert grown.data_ptr() != initial_ptr
    assert reused.data_ptr() == grown.data_ptr()


def test_layers_can_share_one_routed_output_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    arena = moe.SparkinferMoEOutputArena()
    first = moe.SparkinferMoELayer(object(), object(), device="cpu", output_arena=arena)
    second = moe.SparkinferMoELayer(object(), object(), device="cpu", output_arena=arena)

    monkeypatch.setattr(moe, "build_tp_moe_fp4_binding", lambda **kwargs: kwargs)
    monkeypatch.setattr(moe, "sparkinfer_moe_fp4", lambda *, binding: binding["output"])

    hidden = torch.empty(16, moe.HIDDEN_SIZE, dtype=torch.bfloat16)
    topk_ids = torch.zeros(16, moe.TOP_K, dtype=torch.int64)
    topk_weights = torch.ones(16, moe.TOP_K, dtype=torch.float32)

    first_output = first.forward(hidden, topk_ids, topk_weights)
    second_output = second.forward(hidden, topk_ids, topk_weights)

    assert first_output.data_ptr() == second_output.data_ptr()
    assert first._output_arena is second._output_arena is arena


def test_layers_keep_private_output_arenas_by_default() -> None:
    first = moe.SparkinferMoELayer(object(), object(), device="cpu")
    second = moe.SparkinferMoELayer(object(), object(), device="cpu")

    assert first._output_arena is not second._output_arena
