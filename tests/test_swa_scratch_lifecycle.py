from __future__ import annotations

# runtime.backends.laguna imports torch eagerly, so it must be imported only
# after the importorskip below has confirmed torch is present -- ruff's
# import-order/position checks are relaxed accordingly for this file.
# ruff: noqa: E402, I001

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.laguna import _LayerForwardHooks


class _ScratchWriter(torch.nn.Module):
    def __init__(self, scratch: torch.Tensor, value: int) -> None:
        super().__init__()
        self.scratch = scratch
        self.value = value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.scratch.fill_(self.value)
        return value


def test_shared_swa_scratch_is_transferred_before_and_after_each_layer() -> None:
    """Each layer retains its own result although the physical scratch is shared."""
    scratch = torch.zeros(4, dtype=torch.int64)
    ring = {
        "layer_a": torch.full((4,), 101, dtype=torch.int64),
        "layer_b": torch.full((4,), 202, dtype=torch.int64),
    }
    calls: list[tuple[str, str, int]] = []

    def before(name: str) -> None:
        scratch.copy_(ring[name])
        calls.append(("before", name, int(scratch[0])))

    def after(name: str) -> None:
        ring[name].copy_(scratch)
        calls.append(("after", name, int(scratch[0])))

    layers = {
        "layer_a": _ScratchWriter(scratch, 11),
        "layer_b": _ScratchWriter(scratch, 22),
    }
    hooks = _LayerForwardHooks(layers, before, after)

    with hooks.active():
        layers["layer_a"](torch.tensor(0))
        layers["layer_b"](torch.tensor(0))

    assert torch.equal(ring["layer_a"], torch.full((4,), 11, dtype=torch.int64))
    assert torch.equal(ring["layer_b"], torch.full((4,), 22, dtype=torch.int64))
    assert calls == [
        ("before", "layer_a", 101),
        ("after", "layer_a", 11),
        ("before", "layer_b", 202),
        ("after", "layer_b", 22),
    ]


def test_shared_swa_hooks_are_removed_after_the_scratch_forward() -> None:
    scratch = torch.zeros(1)
    layer = _ScratchWriter(scratch, 7)
    calls: list[str] = []
    hooks = _LayerForwardHooks(
        {"layer": layer},
        before=lambda name: calls.append(f"before:{name}"),
        after=lambda name: calls.append(f"after:{name}"),
    )

    with hooks.active():
        layer(torch.tensor(0))
    layer(torch.tensor(0))

    assert calls == ["before:layer", "after:layer"]
