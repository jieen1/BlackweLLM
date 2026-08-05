"""Regression boundary for raw-FP8 GDN projections.

The historical W8A8 verifier projects its whole ``B * (K + 1)`` query
matrix at once.  The no-vLLM path must not revive the BF16 cache merely
because GDN's small ``in_proj_a``/``in_proj_b`` projections formerly used a
``bmm`` rounding workaround.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="torch-free CI job")

from runtime.model.compressed_tensors_linear import CompressedTensorsFP8ChannelLinear  # noqa: E402
from runtime.model.qwen36_model import _bmm_project  # noqa: E402


def test_cuda_raw_fp8_gdn_projection_never_materializes_bf16(monkeypatch):
    """Raw all-layer execution dispatches before the legacy ``_ensure_ready`` call."""

    linear = CompressedTensorsFP8ChannelLinear(16, 8, bias=False)
    sentinel = object()
    seen: list[object] = []

    def raw_forward(x):
        seen.append(x)
        return sentinel

    monkeypatch.setattr(linear, "forward", raw_forward)
    monkeypatch.setattr(
        "runtime.model.qwen36_model.fp8_channel_raw_execution_uses_all_layers",
        lambda: True,
    )

    class FakeCudaTensor:
        device = SimpleNamespace(type="cuda")

    value = _bmm_project(linear, FakeCudaTensor())

    assert value is sentinel
    assert seen and seen[0].device.type == "cuda"
    assert linear._weight_bf16 is None
