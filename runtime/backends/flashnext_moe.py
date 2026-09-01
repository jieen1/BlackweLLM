"""Flash-Next (qwen4_exp) routed-expert adapter for b12x fused MoE.

Checkpoint convention (RadixArk NVFP4, nvidia-modelopt native -- pinned
2026-08-27 against real layer-0 weights, see
notes/2026-08-27-flashnext-runtime-support-plan.md):

* tensor names: ``model.language_model.layers.N.mlp.experts.E.{gate,up,
  down}_proj.{weight, weight_scale, weight_scale_2, input_scale}``;
* ``weight`` is NVFP4-packed ``[out, in/2]`` u8 with e4m3 block scales
  ``[out, in/16]``; dequant = ``code_deq * sf * weight_scale_2`` (multiply
  by the small global scale directly -- the compressed-tensors reciprocal
  convention is NOT used);
* b12x's modelopt path (``_prepare_modelopt_nvfp4_runtime_alphas``) wants
  ``w*_global_scale = weight_scale_2`` raw and ``a*_gscale =
  1/input_scale`` (the reciprocal activation scale), composing the kernel's
  ``weight_global_scale * input_scale`` internally.

Laguna's legacy adapter (``runtime/backends/laguna_sparkinfer_moe.py``) is
retired with that model; this module is the forward path.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import torch

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

logger = logging.getLogger("qwen_sm120_runtime.flashnext_moe")

_FLASHINFER_OPS: tuple[Any, Any] | None = None
_FLASHINFER_IMPORT_ERROR: BaseException | None = None
_FLASHINFER_IMPORT_REPORTED = False

# b12x renamed its environment namespace together with the Python package.
# Keep the legacy names for older editable installs and enable deterministic
# route combine through the current name.  Dynamic FC2 scaling is deliberately
# disabled for Flash-Next: the real full-chain CUDA-graph gate produced NaNs
# with it enabled even though isolated one-layer probes stayed finite.  The
# static ModelOpt activation scales are the qualified production path.
os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
os.environ.setdefault("B12X_ENABLE_DYNAMIC_DOWN_SCALE", "0")
os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
os.environ.setdefault("SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE", "0")
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

_EXPERT_PREFIX = "model.language_model.layers.{layer}.mlp.experts"


@dataclass(frozen=True)
class FlashInferCutlassExperts:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_scales: list[torch.Tensor]


def load_flashinfer_cutlass_moe_ops() -> tuple[Any, Any] | None:
    """Return ``(cutlass_fused_moe, ActivationType)`` or ``None`` when unavailable."""

    global _FLASHINFER_OPS, _FLASHINFER_IMPORT_ERROR, _FLASHINFER_IMPORT_REPORTED
    if _FLASHINFER_OPS is not None:
        return _FLASHINFER_OPS
    if _FLASHINFER_IMPORT_ERROR is not None:
        return None

    venv_ninja = os.path.join(os.path.dirname(sys.executable), "ninja")
    if os.path.isfile(venv_ninja) and shutil.which("ninja") is None:
        os.environ["PATH"] = os.path.dirname(venv_ninja) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
    except BaseException as exc:
        _FLASHINFER_IMPORT_ERROR = exc
        if not _FLASHINFER_IMPORT_REPORTED:
            logger.warning("FlashInfer CUTLASS MoE unavailable: %s", exc)
            _FLASHINFER_IMPORT_REPORTED = True
        return None
    _FLASHINFER_OPS = (cutlass_fused_moe, ActivationType)
    return _FLASHINFER_OPS


def flashnext_flashinfer_moe_available() -> bool:
    return load_flashinfer_cutlass_moe_ops() is not None


def load_flashnext_experts(
    ckpt: pathlib.Path | str,
    layer_idx: int,
    geometry: QwenMoeGeometry,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Load one layer's routed experts + the two activation input scales.

    Returns ``raw`` with ``gate_w/up_w/down_w`` ``[E, ...]`` packed,
    ``gate_sf/up_sf/down_sf`` linear block scales, ``gate_gs/down_gs``
    HOLDING ``1/weight_scale_2`` so :func:`prepare_flashnext_experts`'s
    single reciprocal restores the modelopt-native raw value, and
    ``a1_input_scale``/``a2_input_scale`` scalars (checkpoint values).
    """
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt)
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = _EXPERT_PREFIX.format(layer=layer_idx)
    needed_shards: set[str] = set()
    for eid in range(geometry.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for sfx in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                needed_shards.add(weight_map[f"{prefix}.{eid}.{proj}.{sfx}"])

    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(needed_shards):
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    tensors[k] = f.get_tensor(k)

    e = geometry.num_experts
    raw: dict[str, torch.Tensor] = {}
    for tag, proj in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        raw[f"{tag}_w"] = torch.stack(
            [tensors[f"{prefix}.{i}.{proj}.weight"] for i in range(e)]
        ).to(device)
        raw[f"{tag}_sf"] = torch.stack(
            [tensors[f"{prefix}.{i}.{proj}.weight_scale"] for i in range(e)]
        ).to(device)
        scale2 = (
            torch.stack([tensors[f"{prefix}.{i}.{proj}.weight_scale_2"] for i in range(e)])
            .to(device)
            .float()
        )
        raw[f"{tag}_gs"] = 1.0 / scale2
        if tag == "gate":
            igs = torch.stack(
                [tensors[f"{prefix}.{i}.{proj}.input_scale"] for i in range(e)]
            ).float()
            raw["a1_input_scale"] = igs.min()
        elif tag == "down":
            igs = torch.stack(
                [tensors[f"{prefix}.{i}.{proj}.input_scale"] for i in range(e)]
            ).float()
            raw["a2_input_scale"] = igs.min()
    return raw


def prepare_flashnext_experts(
    raw: dict[str, torch.Tensor],
    geometry: QwenMoeGeometry,
    device: str | torch.device = "cuda",
) -> object:
    """b12x expert preparation with the modelopt-native scale convention.

    ``raw`` is the :func:`load_flashnext_experts` layout; activation scales
    are the checkpoint ``input_scale`` minima carried in the dict, converted
    to the reciprocal form b12x expects.
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
    a1 = float(raw["a1_input_scale"])
    a2 = float(raw["a2_input_scale"])
    if a1 <= 0 or a2 <= 0:
        raise ValueError(f"input scales must be positive, got a1={a1} a2={a2}")
    a1_gscale = torch.tensor(1.0 / a1, dtype=torch.float32, device=device)
    a2_gscale = torch.tensor(1.0 / a2, dtype=torch.float32, device=device)

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
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        params_dtype=torch.bfloat16,
    )


def prepare_flashnext_cutlass_experts(
    raw: dict[str, torch.Tensor],
    geometry: QwenMoeGeometry,
    device: str | torch.device = "cuda",
) -> FlashInferCutlassExperts:
    """Prepare FlashInfer CUTLASS W4A4 payload from checkpoint-native tensors."""

    num_experts = raw["gate_w"].shape[0]
    if num_experts != geometry.num_experts:
        raise ValueError(f"raw experts {num_experts} do not match geometry {geometry.num_experts}")

    a1 = float(raw["a1_input_scale"])
    a2 = float(raw["a2_input_scale"])
    if a1 <= 0.0 or a2 <= 0.0:
        raise ValueError(f"input scales must be positive, got a1={a1} a2={a2}")

    gate_sf_sw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
    up_sf_sw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
    down_sf_sw = swizzle_block_scale(raw["down_sf"].clone().contiguous())
    w13_weight = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).contiguous().view(torch.long)
    w13_blockscale = torch.cat([up_sf_sw, gate_sf_sw], dim=1).contiguous().view(torch.int32)
    gate_scale2 = (1.0 / raw["gate_gs"]).reshape(num_experts).float().contiguous()
    down_scale2 = (1.0 / raw["down_gs"]).reshape(num_experts).float().contiguous()
    quant_scales = [
        torch.tensor(1.0 / a1, dtype=torch.float32, device=device),
        w13_blockscale,
        (a1 * gate_scale2).to(torch.float32),
        torch.tensor(1.0 / a2, dtype=torch.float32, device=device),
        down_sf_sw.contiguous().view(torch.int32),
        (a2 * down_scale2).to(torch.float32),
    ]
    return FlashInferCutlassExperts(
        w13_weight=w13_weight,
        w2_weight=raw["down_w"].contiguous().view(torch.long),
        quant_scales=quant_scales,
    )


class FlashInferMoELayer:
    """One MoE layer backed by FlashInfer CUTLASS fused MoE."""

    def __init__(
        self,
        experts: FlashInferCutlassExperts,
        device: str | torch.device = "cuda",
        output_arena: SparkinferMoEOutputArena | None = None,
    ) -> None:
        self.experts = experts
        self.device = torch.device(device)
        self._output_arena = output_arena or SparkinferMoEOutputArena()

    def forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        ops = load_flashinfer_cutlass_moe_ops()
        if ops is None:
            raise RuntimeError(
                "FlashInfer CUTLASS MoE is unavailable"
            ) from _FLASHINFER_IMPORT_ERROR
        cutlass_fused_moe, activation_type_enum = ops
        out = self._output_arena.acquire(hidden)
        return cutlass_fused_moe(
            output=out,
            input=hidden,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=self.experts.w13_weight,
            fc2_expert_weights=self.experts.w2_weight,
            output_dtype=hidden.dtype,
            input_sf=None,
            quant_scales=self.experts.quant_scales,
            ep_size=1,
            ep_rank=0,
            tp_size=1,
            tp_rank=0,
            enable_alltoall=False,
            use_fused_finalize=True,
            tune_max_num_tokens=max(1, 1 << (hidden.shape[0] - 1).bit_length()),
            activation_type=activation_type_enum.Swiglu,
        )[0]


def make_flashnext_expert_layer(
    ckpt: pathlib.Path | str,
    layer_idx: int,
    geometry: QwenMoeGeometry,
    workspace: object,
    device: str | torch.device = "cuda",
    output_arena: SparkinferMoEOutputArena | None = None,
) -> SparkinferMoELayer:
    """Load + prepare + wrap one layer (per-layer free discipline applies:
    callers should drop the raw dict after this returns)."""
    raw = load_flashnext_experts(ckpt, layer_idx, geometry, device)
    experts = prepare_flashnext_experts(raw, geometry, device)
    return SparkinferMoELayer(
        experts,
        workspace,
        device,
        output_arena=output_arena or SparkinferMoEOutputArena(geometry.hidden_size),
    )


__all__ = [
    "FlashInferCutlassExperts",
    "FlashInferMoELayer",
    "SparkinferMoELayer",
    "SparkinferMoEOutputArena",
    "allocate_tp_moe_workspace_pool",
    "flashnext_flashinfer_moe_available",
    "load_flashinfer_cutlass_moe_ops",
    "load_flashnext_experts",
    "make_flashnext_expert_layer",
    "prepare_flashnext_cutlass_experts",
    "prepare_flashnext_experts",
]
