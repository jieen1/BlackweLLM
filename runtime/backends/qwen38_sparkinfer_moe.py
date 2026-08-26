"""Geometry-parameterized b12x fused-MoE adapter for the Qwen MoE family.

Day-0 prep for Qwen3.8-Flash-Next (see
notes/2026-08-26-qwen38-flash-next-day0-survey.md). This mirrors the proven
scale conventions of :mod:`runtime.backends.laguna_sparkinfer_moe` exactly --
the only difference is that expert count / hidden / intermediate come from a
:class:`~runtime.model.qwen38_moe.QwenMoeGeometry` instead of Laguna's module
constants, and the checkpoint naming follows the family layout
(``model.layers.N.mlp.experts.E.{gate,up,down}_proj.*``).

Verified 2026-08-26 by tests/test_qwen38_moe.py: at Laguna's own geometry
(256 experts / hidden 3072 / intermediate 1024) this path reproduces the
production ``prepare_sparkinfer_layer`` outputs bit-for-bit, and at the
Flash-Next family geometry (512 / 8192 / 2048) the FN0 probe passes
(``scripts/fn0_probe_b12x_moe_e512.py``).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any

import torch

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

logger = logging.getLogger("qwen_sm120_runtime.qwen38_sparkinfer_moe")

# Mirror laguna_sparkinfer_moe.py's import-time contract: both switches are
# kernel selections, not runtime options, and must precede the b12x import.
os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
os.environ.setdefault("SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE", "1")
ensure_sparkinfer_path()

try:
    from b12x._lib.intrinsics import swizzle_block_scale
    from b12x.moe.fused_moe._impl import (
        allocate_tp_moe_workspace_pool,
        plan_b12x_fp4_moe_weights,
        prepare_b12x_fp4_moe_weights,
    )
except ImportError as exc:
    raise ImportError(
        "sparkinfer not found. Install via: pip install -e /path/to/sparkinfer "
        "or set BF_SPARKINFER_PATH=/path/to/sparkinfer"
    ) from exc

from runtime.backends.laguna_sparkinfer_moe import (  # noqa: E402
    SparkinferMoELayer,
    SparkinferMoEOutputArena,
)
from runtime.model.qwen38_moe import QwenMoeGeometry  # noqa: E402


def load_qwen_moe_layer_weights(
    ckpt: pathlib.Path,
    layer_idx: int,
    geometry: QwenMoeGeometry,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Load one layer's per-expert NVFP4 weights from family-named safetensors.

    Returns the same raw layout ``load_moe_layer_weights`` produces for
    Laguna: gate_w/up_w/down_w stacked ``[E, ...]`` packed tensors plus
    gate_sf/up_sf/down_sf block scales and gate_gs/down_gs global scales.
    """
    from safetensors import safe_open

    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_idx}.mlp.experts"
    needed_shards: set[str] = set()
    for eid in range(geometry.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for sfx in ("weight_packed", "weight_scale", "weight_global_scale"):
                needed_shards.add(weight_map[f"{prefix}.{eid}.{proj}.{sfx}"])

    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(needed_shards):
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    tensors[k] = f.get_tensor(k)

    result = {}
    for name, proj in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
        result[f"{name}_w"] = torch.stack(
            [tensors[f"{prefix}.{e}.{proj}.weight_packed"] for e in range(geometry.num_experts)]
        ).to(device)
        result[f"{name}_sf"] = torch.stack(
            [tensors[f"{prefix}.{e}.{proj}.weight_scale"] for e in range(geometry.num_experts)]
        ).to(device)
        result[f"{name}_gs"] = (
            torch.stack(
                [
                    tensors[f"{prefix}.{e}.{proj}.weight_global_scale"]
                    for e in range(geometry.num_experts)
                ]
            )
            .to(device)
            .float()
        )
    return result


def load_qwen_moe_layer_activation_gscales(
    ckpt: pathlib.Path,
    layer_idx: int,
    geometry: QwenMoeGeometry,
) -> tuple[float, float]:
    """Compute ``(a1_gscale, a2_gscale)`` from per-expert input_global_scale.

    Same double-reciprocal composition as
    :func:`runtime.backends.laguna_sparkinfer_moe.load_moe_layer_activation_gscales`
    (see its docstring for the verified formula): min of the raw
    gate/up input_global_scale across experts, and min of the raw down ones.
    """
    from safetensors import safe_open

    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_idx}.mlp.experts"
    needed_keys = {
        f"{prefix}.{eid}.{proj}.input_global_scale"
        for eid in range(geometry.num_experts)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    needed_shards = {weight_map[k] for k in needed_keys}

    values: dict[str, torch.Tensor] = {}
    for shard in sorted(needed_shards):
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed_keys:
                    values[k] = f.get_tensor(k)

    gate_up = [
        values[f"{prefix}.{e}.{proj}.input_global_scale"].item()
        for e in range(geometry.num_experts)
        for proj in ("gate_proj", "up_proj")
    ]
    down = [
        values[f"{prefix}.{e}.down_proj.input_global_scale"].item()
        for e in range(geometry.num_experts)
    ]
    return min(gate_up), min(down)


def prepare_qwen_moe_experts(
    raw: dict[str, torch.Tensor],
    geometry: QwenMoeGeometry,
    device: str | torch.device = "cuda",
    *,
    a1_gscale: float,
    a2_gscale: float,
) -> Any:
    """Prepare b12x expert weights from raw per-expert tensors.

    Bit-identical convention to ``prepare_sparkinfer_layer``: w13 data order
    ``[up, gate]`` with ``w13_layout="w13"``, checkpoint block scales
    swizzled unfused, ``w*_global_scale = 1/checkpoint_gs`` as runtime
    alphas, activation gscales as reciprocal scales.
    """
    num_experts = raw["gate_w"].shape[0]
    if num_experts != geometry.num_experts:
        raise ValueError(f"raw experts {num_experts} do not match geometry {geometry.num_experts}")

    gate_sf_sw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
    up_sf_sw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
    down_sf_sw = swizzle_block_scale(raw["down_sf"].clone().contiguous())

    w13_fp4 = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).contiguous()
    w13_sf = torch.cat([up_sf_sw, gate_sf_sw], dim=1).contiguous()

    w1_alpha = (1.0 / raw["gate_gs"]).float().contiguous()
    w2_alpha = (1.0 / raw["down_gs"]).float().contiguous()

    if isinstance(a1_gscale, (int, float)):
        a1_gscale_t = torch.tensor(a1_gscale, dtype=torch.float32, device=device)
    else:
        a1_gscale_t = a1_gscale
    if isinstance(a2_gscale, (int, float)):
        a2_gscale_t = torch.tensor(a2_gscale, dtype=torch.float32, device=device)
    else:
        a2_gscale_t = a2_gscale

    wplan = plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.moe_intermediate_size,
        w13_layout="w13",
    )
    return prepare_b12x_fp4_moe_weights(
        plan=wplan,
        w1_global_scale=w1_alpha,
        w2_global_scale=w2_alpha,
        w1_fp4=w13_fp4,
        w1_blockscale=w13_sf,
        w2_fp4=raw["down_w"].clone().contiguous(),
        w2_blockscale=down_sf_sw,
        a1_gscale=a1_gscale_t,
        a2_gscale=a2_gscale_t,
        params_dtype=torch.bfloat16,
    )


def make_qwen_moe_expert_layer(
    raw: dict[str, torch.Tensor],
    geometry: QwenMoeGeometry,
    workspace: Any,
    device: str | torch.device = "cuda",
    *,
    a1_gscale: float,
    a2_gscale: float,
    output_arena: SparkinferMoEOutputArena | None = None,
) -> SparkinferMoELayer:
    """One-call convenience: prepare experts and wrap them in a layer."""
    experts = prepare_qwen_moe_experts(
        raw, geometry, device, a1_gscale=a1_gscale, a2_gscale=a2_gscale
    )
    return SparkinferMoELayer(
        experts,
        workspace,
        device,
        output_arena=output_arena or SparkinferMoEOutputArena(geometry.hidden_size),
    )


__all__ = [
    "SparkinferMoELayer",
    "SparkinferMoEOutputArena",
    "allocate_tp_moe_workspace_pool",
    "load_qwen_moe_layer_activation_gscales",
    "load_qwen_moe_layer_weights",
    "make_qwen_moe_expert_layer",
    "prepare_qwen_moe_experts",
]
