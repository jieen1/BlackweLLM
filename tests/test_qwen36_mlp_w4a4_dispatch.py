"""CPU tests: Qwen36MLP W4A4-only dispatch (plan §4.4 P0-M1).

The all-W4A4 default must prepare the W4A4 operands directly and never
build the independent W4A16 repack, while the W4A4-unavailable fallback
and the ``QSR_QWEN36_MLP_W4A4_ALL=0`` comparison mode keep the
historical W4A16-first behavior. The dispatch is tested with the two
forward helpers monkeypatched to record which path ran -- no GPU, no
sparkinfer kernels.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.qwen36_model import Qwen36MLP  # noqa: E402
from tests.test_qwen36_mtp_head import _tiny_config  # noqa: E402


def _nvfp4_mlp() -> Qwen36MLP:
    """An MLP whose three projections are all NVFP4 (fused path active)."""
    config = _tiny_config()
    quantized = {
        "model.language_model.layers.1.mlp.gate_proj": "W4A16_NVFP4",
        "model.language_model.layers.1.mlp.up_proj": "W4A16_NVFP4",
        "model.language_model.layers.1.mlp.down_proj": "W4A16_NVFP4",
    }
    return Qwen36MLP(config, 1, quantized)


def _dispatch_recording(monkeypatch, mlp: Qwen36MLP, *, w4a4_available: bool):
    calls: list[str] = []

    def w4a4(self: Qwen36MLP, x: torch.Tensor) -> torch.Tensor:
        calls.append("w4a4")
        return x

    def w4a16(self: Qwen36MLP, x: torch.Tensor) -> torch.Tensor:
        calls.append("w4a16")
        return x

    def ensure_w4a4(self: Qwen36MLP) -> None:
        if w4a4_available:
            self._w4a4_prepared = {"prepared": True}
        else:
            self._w4a4_prepared = None
            self._w4a4_unavailable = True

    monkeypatch.setattr(Qwen36MLP, "_forward_w4a4_blockscaled", w4a4)
    monkeypatch.setattr(Qwen36MLP, "_forward_w4a16_fused", w4a16)
    monkeypatch.setattr(Qwen36MLP, "_ensure_w4a4_ready", ensure_w4a4)
    mlp._nvfp4_fused = True
    return calls


class TestW4a4OnlyDispatch:
    def test_all_w4a4_routes_every_row_to_w4a4(self, monkeypatch) -> None:
        mlp = _nvfp4_mlp()
        calls = _dispatch_recording(monkeypatch, mlp, w4a4_available=True)
        mlp.forward(torch.zeros(2, 16))
        mlp.forward(torch.zeros(1, 16))
        assert calls == ["w4a4", "w4a4"]

    def test_all_w4a4_falls_back_to_w4a16_when_unavailable(self, monkeypatch) -> None:
        mlp = _nvfp4_mlp()
        calls = _dispatch_recording(monkeypatch, mlp, w4a4_available=False)
        mlp.forward(torch.zeros(2, 16))
        assert calls == ["w4a16"]

    def test_w4a4_all_0_keeps_prefill_w4a4_and_decode_w4a16(self, monkeypatch) -> None:
        monkeypatch.setenv("QSR_QWEN36_MLP_W4A4_ALL", "0")
        mlp = _nvfp4_mlp()
        calls = _dispatch_recording(monkeypatch, mlp, w4a4_available=True)
        mlp._w4a4_prepared = {"prepared": True}
        mlp.forward(torch.zeros(2, 16))  # decode-sized: W4A16
        mlp.forward(torch.zeros(128, 16))  # prefill-sized: W4A4
        assert calls == ["w4a16", "w4a4"]

    def test_w4a4_all_0_and_w4a4_off_keeps_w4a16_only(self, monkeypatch) -> None:
        monkeypatch.setenv("QSR_QWEN36_MLP_W4A4_ALL", "0")
        monkeypatch.setenv("QSR_QWEN36_MLP_W4A4", "0")
        mlp = _nvfp4_mlp()
        calls = _dispatch_recording(monkeypatch, mlp, w4a4_available=True)
        mlp._w4a4_prepared = {"prepared": True}
        mlp.forward(torch.zeros(4096, 16))
        assert calls == ["w4a16"]

    def test_non_nvfp4_mlp_never_touches_fused_path(self, monkeypatch) -> None:
        config = _tiny_config()
        mlp = Qwen36MLP(config, 1, {})
        assert not mlp._nvfp4_fused
        calls: list[str] = []

        def w4a4(self: Qwen36MLP, x: torch.Tensor) -> torch.Tensor:
            calls.append("w4a4")
            return x

        monkeypatch.setattr(Qwen36MLP, "_forward_w4a4_blockscaled", w4a4)
        monkeypatch.setattr(Qwen36MLP, "_forward_w4a16_fused", w4a4)
        out = mlp.forward(torch.zeros(2, 32, dtype=torch.float32))
        assert calls == []
        assert out.shape == (2, 32)
