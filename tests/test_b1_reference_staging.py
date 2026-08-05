"""CPU coverage for the disk-staged HF reference-weight bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
# The staged script imports runtime.model_loading -> qwen36_model, which
# imports fla and sparkinfer at module scope; the cpu-torch CI job has
# neither, so self-skip there like the sibling qwen36 tests.
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")
nn = torch.nn


def _load_staging_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "b1_verify_greedy_alignment.py"
    spec = importlib.util.spec_from_file_location("b1_reference_staging", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PackedLinear(nn.Module):
    """Small stand-in for compressed-tensors NVFP4's staging contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight_packed = nn.Parameter(torch.arange(3, dtype=torch.uint8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.weight_global_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.input_global_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.k_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.v_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
        self._weight_bf16: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            self._weight_bf16 = torch.tensor(
                [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.bfloat16
            )


class _Source(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packed = _PackedLinear()
        self.plain = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.plain.weight.copy_(torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))


class _Reference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packed = nn.Linear(3, 2, bias=True, dtype=torch.bfloat16)
        self.plain = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)


def test_staging_maps_packed_weight_to_hf_weight_and_skips_metadata(tmp_path: Path) -> None:
    staging = _load_staging_module()
    source = _Source()
    expected_packed = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.bfloat16)

    names = staging.stage_dequantized_weights_to_disk(source, tmp_path)

    assert names == ["packed.weight", "packed.bias", "plain.weight"]
    assert source.packed._weight_bf16 is None
    assert all("scale" not in name for name in names)
    assert "packed.weight_packed" not in names

    reference = _Reference()
    loaded, missing = staging.load_staged_weights_into_hf(reference, tmp_path, names)

    assert loaded == names
    assert missing == []
    torch.testing.assert_close(reference.packed.weight, expected_packed)
    torch.testing.assert_close(source.packed.bias, reference.packed.bias)
    torch.testing.assert_close(source.plain.weight, reference.plain.weight)
    assert not list(tmp_path.iterdir())
