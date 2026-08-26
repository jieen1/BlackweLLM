"""Synthetic-checkpoint tests for the parameterized Qwen MoE expert loader.

Writes a tiny family-named safetensors shard and verifies the loader's
stacking, dtype handling, and the activation-gscale min formula -- the parts
of the load path that do not need a GPU or b12x.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from runtime.backends.qwen38_sparkinfer_moe import (  # noqa: E402
    load_qwen_moe_layer_activation_gscales,
    load_qwen_moe_layer_weights,
)
from runtime.model.qwen38_moe import QwenMoeGeometry  # noqa: E402

GEOM = QwenMoeGeometry(
    num_experts=4,
    top_k=2,
    hidden_size=64,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
)
LAYER = 3


def _write_checkpoint(tmp_path):
    tensors: dict[str, torch.Tensor] = {}
    values: dict[str, float] = {}
    for e in range(GEOM.num_experts):
        for proj, rows, packed_cols in (
            ("gate_proj", 32, 32),
            ("up_proj", 32, 32),
            ("down_proj", 64, 16),
        ):
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{proj}.weight_packed"] = torch.full(
                (rows, packed_cols), e * 10 + 1, dtype=torch.uint8
            )
            # Block scales are [N, K/16]: gate/up reduce hidden=64, down
            # reduces intermediate=32.
            scale_cols = (2 * packed_cols) // 16
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{proj}.weight_scale"] = torch.full(
                (rows, scale_cols), 0.5, dtype=torch.float32
            ).to(torch.float8_e4m3fn)
            gscale = float(e + 1) * (2.0 if proj == "down_proj" else 1.0)
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{proj}.weight_global_scale"] = (
                torch.tensor(gscale, dtype=torch.float32)
            )
            igs = float(e + 1) * (3.0 if proj == "down_proj" else 1.0)
            tensors[f"model.layers.{LAYER}.mlp.experts.{e}.{proj}.input_global_scale"] = (
                torch.tensor(igs, dtype=torch.float32)
            )
            values[f"{proj}.{e}"] = igs
    safetensors_torch.save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    weight_map = {k: "model-00001-of-00001.safetensors" for k in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return values


def test_loader_stacks_per_expert_tensors(tmp_path):
    _write_checkpoint(tmp_path)
    raw = load_qwen_moe_layer_weights(tmp_path, LAYER, GEOM, device="cpu")

    assert tuple(raw["gate_w"].shape) == (4, 32, 32)
    assert tuple(raw["up_w"].shape) == (4, 32, 32)
    assert tuple(raw["down_w"].shape) == (4, 64, 16)
    assert raw["gate_w"].dtype == torch.uint8
    assert tuple(raw["gate_sf"].shape) == (4, 32, 4)
    assert raw["gate_sf"].dtype == torch.float8_e4m3fn
    assert tuple(raw["gate_gs"].shape) == (4,)
    assert raw["gate_gs"].dtype == torch.float32
    # Expert e's packed payload is the constant e*10+1 everywhere.
    for e in range(4):
        assert int(raw["gate_w"][e].flatten()[0]) == e * 10 + 1
        assert int(raw["down_w"][e].flatten()[0]) == e * 10 + 1
    # gate_gs holds the per-expert weight_global_scale (gate/up share the e+1
    # pattern here; the loader reads gate_proj's).
    assert raw["gate_gs"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert raw["down_gs"].tolist() == [2.0, 4.0, 6.0, 8.0]


def test_activation_gscales_take_per_projection_min(tmp_path):
    values = _write_checkpoint(tmp_path)
    a1, a2 = load_qwen_moe_layer_activation_gscales(tmp_path, LAYER, GEOM)
    gate_up = [
        values[f"{proj}.{e}"] for e in range(GEOM.num_experts) for proj in ("gate_proj", "up_proj")
    ]
    down = [values[f"down_proj.{e}"] for e in range(GEOM.num_experts)]
    assert a1 == min(gate_up)
    assert a2 == min(down)
    assert a1 == 1.0
    assert a2 == 3.0
