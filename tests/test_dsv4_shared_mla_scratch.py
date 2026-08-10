from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import _share_mla_scratch_across_layers  # noqa: E402
from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer  # noqa: E402


@dataclass(frozen=True)
class _FakeScratchSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class _FakePlan:
    def __init__(self, spec: _FakeScratchSpec) -> None:
        self._spec = spec

    def scratch_specs(self):
        return (self._spec,)


class _FakeLayer:
    def __init__(self, nbytes: int, *, dtype: torch.dtype = torch.uint8) -> None:
        self._spec = _FakeScratchSpec(
            name="compressed_mla.scratch",
            shape=(nbytes,),
            dtype=dtype,
            device=torch.device("cpu"),
        )
        self.bound = None

    def mla_scratch_spec(self):
        return self._spec

    def set_mla_scratch(self, scratch: torch.Tensor) -> None:
        self.bound = scratch.reshape(-1).narrow(0, 0, int(self._spec.shape[0]))


def test_layer_set_mla_scratch_binds_prefix_view() -> None:
    layer = object.__new__(Dsv4AttnKernelLayer)
    layer._mla_plan = _FakePlan(  # noqa: SLF001 - unit test inspects binder contract
        _FakeScratchSpec(
            name="compressed_mla.scratch",
            shape=(64,),
            dtype=torch.uint8,
            device=torch.device("cpu"),
        )
    )
    shared = torch.empty(256, dtype=torch.uint8)

    layer.set_mla_scratch(shared)

    assert layer._mla_scratch.numel() == 64  # noqa: SLF001 - bound view is the behavior
    assert layer._mla_scratch.untyped_storage().data_ptr() == shared.untyped_storage().data_ptr()
    assert layer._mla_scratch.storage_offset() == 0


def test_share_mla_scratch_uses_one_max_arena_for_all_layers() -> None:
    layers = [_FakeLayer(64), _FakeLayer(128), _FakeLayer(96)]

    scratch = _share_mla_scratch_across_layers(layers)

    assert scratch is not None
    assert scratch.numel() == 128
    base_ptr = scratch.untyped_storage().data_ptr()
    for layer in layers:
        assert layer.bound is not None
        assert layer.bound.untyped_storage().data_ptr() == base_ptr
        assert layer.bound.storage_offset() == 0


def test_share_mla_scratch_rejects_mixed_dtypes() -> None:
    layers = [_FakeLayer(64), _FakeLayer(32, dtype=torch.float32)]

    with pytest.raises(RuntimeError, match="one dtype across layers"):
        _share_mla_scratch_across_layers(layers)
