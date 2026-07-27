"""bfdiag.shapes.moe -- MoE / router / NVFP4-packed expert weight shapes.

Physical NVFP4-packed tensor shapes below were read directly from the real
checkpoint's safetensors header (shape/dtype metadata only, no tensor data
materialized) on this machine -- cross-referencing
``runtime/backends/laguna_sparkinfer_moe.py`` (``load_moe_layer_weights``/
``prepare_sparkinfer_layer``), the code that actually loads these::

    model.layers.{1..47}.mlp.experts.{0..255}.gate_proj.weight_packed  [1024, 1536]  uint8
    model.layers.{1..47}.mlp.experts.{0..255}.gate_proj.weight_scale   [1024,  192]  f8_e4m3
    model.layers.{1..47}.mlp.experts.{0..255}.gate_proj.weight_global_scale  []      f32
    model.layers.{1..47}.mlp.experts.{0..255}.up_proj.*     (same shapes as gate_proj)
    model.layers.{1..47}.mlp.experts.{0..255}.down_proj.weight_packed  [3072,  512]  uint8
    model.layers.{1..47}.mlp.experts.{0..255}.down_proj.weight_scale   [3072,   64]  f8_e4m3

Packing rule (NVFP4, 2 values/byte, ``group_size=16`` from
``quantization_config.config_groups.*.weights.group_size``): for a logical
weight ``[out_features, in_features]``::

    weight_packed = [out_features, in_features // 2]          uint8
    weight_scale  = [out_features, in_features // group_size]  float8_e4m3fn
    weight_global_scale = scalar float32

``gate_proj``/``up_proj``: in=hidden_size(3072), out=moe_intermediate_size(1024).
``down_proj``:             in=moe_intermediate_size(1024), out=hidden_size(3072).

After ``prepare_sparkinfer_layer`` concatenates gate+up into sparkinfer's
fused "w13" layout (``torch.cat([up_w, gate_w], dim=1)``), the per-expert w13
doubles its "out" dim: stacked across all ``E`` experts,
``w13_fp4 = [E, 2*moe_intermediate_size, hidden_size // 2]``. ``down`` is not
concatenated: ``w2_fp4 = [E, hidden_size, moe_intermediate_size // 2]``.

Scale tensors are additionally run through ``swizzle_block_scale`` before use
-- an opaque physical-layout transform (same element count, rearranged for
the kernel's tile access pattern), not a simple reshape; this module reports
the pre-swizzle logical shape since that is what is meaningful to a test
author (the swizzle itself is sparkinfer-internal and not reproduced here).
"""

from __future__ import annotations

from dataclasses import dataclass

from bfdiag.shapes.gemm import GemmShape, dense_mlp_gemms
from bfdiag.shapes.model import LagunaModelConfig


@dataclass(frozen=True)
class Nvfp4PackedGemm:
    """One NVFP4-packed expert projection (gate_proj, up_proj, or down_proj),
    per-expert (not yet stacked across experts -- see
    :func:`stacked_expert_shapes` for that)."""

    name: str
    out_features: int
    in_features: int
    group_size: int

    @property
    def weight_packed_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features // 2)

    @property
    def weight_scale_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features // self.group_size)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "weight_packed": self.weight_packed_shape,
            "weight_scale": self.weight_scale_shape,
            "weight_global_scale": (),
            "input_global_scale": (),
        }


def expert_projection_shapes(config: LagunaModelConfig) -> dict[str, Nvfp4PackedGemm]:
    """Per-expert logical NVFP4 shapes for gate_proj/up_proj/down_proj."""
    if config.nvfp4_group_size is None:
        raise ValueError(
            "config has no NVFP4 group_size "
            "(quantization_config.config_groups.*.weights.group_size); cannot "
            "derive packed expert weight shapes without it."
        )
    if not config.num_experts:
        raise ValueError(
            f"{config.model_id} has num_experts=0 -- it has no MoE experts to derive shapes for "
            "(e.g. the DFlash draft model; use bfdiag.shapes.gemm.draft_dense_gemms instead)."
        )
    gs = config.nvfp4_group_size
    return {
        "gate_proj": Nvfp4PackedGemm(
            "expert.gate_proj",
            out_features=config.moe_intermediate_size,
            in_features=config.hidden_size,
            group_size=gs,
        ),
        "up_proj": Nvfp4PackedGemm(
            "expert.up_proj",
            out_features=config.moe_intermediate_size,
            in_features=config.hidden_size,
            group_size=gs,
        ),
        "down_proj": Nvfp4PackedGemm(
            "expert.down_proj",
            out_features=config.hidden_size,
            in_features=config.moe_intermediate_size,
            group_size=gs,
        ),
    }


def stacked_expert_shapes(config: LagunaModelConfig) -> dict[str, tuple[int, ...]]:
    """Per-checkpoint-layer weights stacked across all ``num_experts`` (as
    ``load_moe_layer_weights`` does with ``torch.stack``), before sparkinfer's
    w13 fusion."""
    projs = expert_projection_shapes(config)
    e = config.num_experts
    out: dict[str, tuple[int, ...]] = {}
    for name, proj in projs.items():
        out[f"{name}.weight_packed"] = (e, *proj.weight_packed_shape)
        out[f"{name}.weight_scale"] = (e, *proj.weight_scale_shape)
    return out


def sparkinfer_w13_shapes(config: LagunaModelConfig) -> dict[str, tuple[int, ...]]:
    """Sparkinfer's fused w13 (gate+up concatenated along the output dim) and
    w2 (down, unfused) layout, stacked across experts -- what
    ``prepare_sparkinfer_layer``/``build_tp_moe_fp4_binding`` actually
    consume."""
    projs = expert_projection_shapes(config)
    e = config.num_experts
    gate_packed = projs["gate_proj"].weight_packed_shape
    down_packed = projs["down_proj"].weight_packed_shape
    return {
        "w13_fp4": (e, 2 * gate_packed[0], gate_packed[1]),
        "w2_fp4": (e, *down_packed),
    }


def router_shapes(config: LagunaModelConfig, *, num_tokens: int) -> dict[str, tuple[int, ...]]:
    """Router logits + top-k routing tensors for ``num_tokens`` tokens."""
    return {
        "router_logits": (num_tokens, config.num_experts),
        "topk_ids": (num_tokens, config.num_experts_per_tok),
        "topk_weights": (num_tokens, config.num_experts_per_tok),
    }


def shared_expert_gemms(config: LagunaModelConfig, *, num_tokens: int) -> list[GemmShape]:
    """The dense shared_expert MLP that runs alongside the routed experts on
    every MoE layer (same shape as :func:`bfdiag.shapes.gemm.dense_mlp_gemms`,
    parameterized by ``shared_expert_intermediate_size``)."""
    return dense_mlp_gemms(
        hidden_size=config.hidden_size,
        intermediate_size=config.shared_expert_intermediate_size,
        num_tokens=num_tokens,
        prefix="moe_layer.shared_expert",
    )
