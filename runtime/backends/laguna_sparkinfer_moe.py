"""sparkinfer NVFP4 MoE kernel — standalone, zero vLLM dependency.

Loads Laguna NVFP4 checkpoint weights directly from safetensors,
prepares them for sparkinfer, and provides a clean forward() API.

Dependency: sparkinfer (editable install from jieen1/sparkinfer fork,
branch master -- blackforge-main was merged into origin/master 2026-07-27
and is no longer the canonical branch -- or BF_SPARKINFER_PATH env
fallback).

Scale convention (verified cosine≥0.993 vs reference, all M=1..128):
  - w13 data order: [up, gate] with w13_layout="w13"
  - Block scales: swizzle checkpoint originals (no folding)
  - w1_global_scale = 1/checkpoint_gs (fp32 runtime alpha)
  - a1_gscale = 1/input_scale (reciprocal activation scale, ~2016)
  - SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE=1 (prevents FC2 fp8 underflow)

Performance (SM120, E=256, K=3072, I=1024, top_k=10):
  - CUDA graph: ~38μs/layer → 1.8ms for 47 layers
  - vs CUTLASS eager: ~186μs/layer → 8.73ms for 47 layers
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Sequence

import torch

logger = logging.getLogger("qwen_sm120_runtime.sparkinfer_moe")

# Enable deterministic MoE output (ROUTE_BUFFER_TOPK_SUM instead of ATOMIC_SCATTER)
# Required for DFlash speculative decoding acceptance (greedy argmax must be stable).
os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")

# ---------------------------------------------------------------------------
# sparkinfer import: editable install preferred, env fallback
# ---------------------------------------------------------------------------
# FC2 intermediate quantization scale underflows fp8-e4m3 when w1_alpha is
# small (~1e-4 for Laguna).  dynamic_down_scale computes a tile-level adaptive
# scale that prevents the underflow.  Must be set before sparkinfer import.
os.environ.setdefault("SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE", "1")

_BF_SPARKINFER_PATH = os.environ.get("BF_SPARKINFER_PATH", "")
if _BF_SPARKINFER_PATH and _BF_SPARKINFER_PATH not in sys.path:
    sys.path.insert(0, _BF_SPARKINFER_PATH)

try:
    from sparkinfer._lib.intrinsics import swizzle_block_scale

    # Imported for its side effect only: it is the availability probe that
    # makes a sparkinfer build without the public fused_moe surface fail here,
    # with the actionable message below, instead of much deeper in a forward.
    from sparkinfer.moe.fused_moe import is_supported as sparkinfer_is_supported  # noqa: F401
    from sparkinfer.moe.fused_moe._impl import (
        allocate_tp_moe_workspace_pool,
        build_tp_moe_fp4_binding,
        plan_sparkinfer_fp4_moe_weights,
        prepare_sparkinfer_fp4_moe_weights,
        sparkinfer_moe_fp4,
    )
except ImportError as exc:
    raise ImportError(
        "sparkinfer not found. Install via: pip install -e /path/to/sparkinfer "
        "or set BF_SPARKINFER_PATH=/path/to/sparkinfer"
    ) from exc

# Remove cutlass-dsl base_dsl from sys.path — it contains a torch.py
# that shadows the real torch module in spawned subprocesses.
sys.path[:] = [p for p in sys.path if "nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl" not in p]


def sparkinfer_version() -> str:
    """Return sparkinfer git commit sha for version stamping."""
    try:
        import sparkinfer

        pkg_dir = pathlib.Path(sparkinfer.__file__).parent
        git_dir = pkg_dir.parent / ".git"
        if git_dir.exists():
            result = subprocess.run(
                ["git", "-C", str(git_dir.parent), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EXPERTS = 256
TOP_K = 10
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 1024
MOE_LAYER_IDS = list(range(1, 48))  # layers 1-47


def _find_checkpoint(model_id: str = "poolside/Laguna-S-2.1-NVFP4") -> pathlib.Path:
    """Resolve HF cache snapshot path for the model."""
    cache_root = pathlib.Path.home() / ".cache/huggingface/hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        snaps = sorted(snapshots.iterdir())
        if snaps:
            return snaps[0]
    raise FileNotFoundError(f"Cannot find checkpoint for {model_id} in {cache_root}")


def load_moe_layer_weights(
    ckpt: pathlib.Path,
    layer_idx: int,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Load one MoE layer's per-expert weights directly from safetensors.

    Returns dict: gate_w, up_w, down_w, gate_sf, up_sf, down_sf,
    gate_gs, up_gs, down_gs — each [E, ...] on device.
    """
    from safetensors import safe_open

    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_idx}.mlp.experts"
    needed_shards: set[str] = set()
    for eid in range(NUM_EXPERTS):
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
            [tensors[f"{prefix}.{e}.{proj}.weight_packed"] for e in range(NUM_EXPERTS)]
        ).to(device)
        result[f"{name}_sf"] = torch.stack(
            [tensors[f"{prefix}.{e}.{proj}.weight_scale"] for e in range(NUM_EXPERTS)]
        ).to(device)
        result[f"{name}_gs"] = (
            torch.stack(
                [tensors[f"{prefix}.{e}.{proj}.weight_global_scale"] for e in range(NUM_EXPERTS)]
            )
            .to(device)
            .float()
        )
    return result


def prepare_sparkinfer_layer(
    raw: dict[str, torch.Tensor],
    device: str | torch.device = "cuda",
    a1_gscale: float | None = None,
    a2_gscale: float | None = None,
):
    """Prepare sparkinfer expert weights from raw checkpoint tensors.

    Scale convention (sparkinfer benchmark pipeline):
      - w13 data order: [up, gate] with w13_layout="w13"
      - Block scales: swizzle checkpoint originals (no folding)
      - w1_global_scale = 1/checkpoint_gs (fp32 runtime alpha)
      - a1_gscale = 1/input_scale (reciprocal activation scale, ~2016)
    """
    num_experts = raw["gate_w"].shape[0]

    gate_sf_sw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
    up_sf_sw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
    down_sf_sw = swizzle_block_scale(raw["down_sf"].clone().contiguous())

    # sparkinfer "w13" layout = [up, gate] data order (alias "up_gate")
    w13_fp4 = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).contiguous()
    w13_sf = torch.cat([up_sf_sw, gate_sf_sw], dim=1).contiguous()

    w1_alpha = (1.0 / raw["gate_gs"]).float().contiguous()
    w2_alpha = (1.0 / raw["down_gs"]).float().contiguous()

    if a1_gscale is None:
        a1_gscale_t = torch.ones((), dtype=torch.float32, device=device)
    elif isinstance(a1_gscale, (int, float)):
        a1_gscale_t = torch.tensor(a1_gscale, dtype=torch.float32, device=device)
    else:
        a1_gscale_t = a1_gscale
    if a2_gscale is None:
        a2_gscale_t = torch.ones((), dtype=torch.float32, device=device)
    elif isinstance(a2_gscale, (int, float)):
        a2_gscale_t = torch.tensor(a2_gscale, dtype=torch.float32, device=device)
    else:
        a2_gscale_t = a2_gscale

    wplan = plan_sparkinfer_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        w13_layout="w13",
    )
    return prepare_sparkinfer_fp4_moe_weights(
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


class SparkinferMoELayer:
    """One MoE layer backed by sparkinfer kernel."""

    def __init__(self, experts, workspace, device="cuda"):
        self.experts = experts
        self.workspace = workspace
        self.device = torch.device(device)
        self._output_buf: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        M = hidden.shape[0]
        if self._output_buf is None or self._output_buf.shape[0] < M:
            self._output_buf = torch.empty(
                M,
                HIDDEN_SIZE,
                dtype=hidden.dtype,
                device=self.device,
            )
        out = self._output_buf[:M]
        binding = build_tp_moe_fp4_binding(
            scratch=self.workspace,
            a=hidden,
            experts=self.experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            quant_mode="nvfp4",
            input_scales_static=True,
            output=out,
        )
        return sparkinfer_moe_fp4(binding=binding)


class SparkinferMoEModel:
    """All 47 MoE layers loaded from checkpoint, ready for inference."""

    def __init__(
        self,
        ckpt: pathlib.Path | str | None = None,
        layer_ids: Sequence[int] = MOE_LAYER_IDS,
        device: str = "cuda",
    ):
        if ckpt is None:
            ckpt = _find_checkpoint()
        self.ckpt = pathlib.Path(ckpt)
        self.layer_ids = list(layer_ids)
        self.device = device
        self.layers: dict[int, SparkinferMoELayer] = {}
        self._workspace = allocate_tp_moe_workspace_pool()
        self.version = sparkinfer_version()
        logger.info(
            "SparkinferMoEModel init: %d layers, sparkinfer@%s",
            len(self.layer_ids),
            self.version,
        )

    def load_layer(self, layer_idx: int) -> SparkinferMoELayer:
        t0 = time.time()
        raw = load_moe_layer_weights(self.ckpt, layer_idx, self.device)
        experts = prepare_sparkinfer_layer(raw, self.device)
        layer = SparkinferMoELayer(experts, self._workspace, self.device)
        self.layers[layer_idx] = layer
        del raw
        torch.cuda.empty_cache()
        logger.info("Layer %d prepared in %.1fs", layer_idx, time.time() - t0)
        return layer

    def load_all(self) -> None:
        t0 = time.time()
        for lid in self.layer_ids:
            self.load_layer(lid)
        total = time.time() - t0
        logger.info(
            "All %d MoE layers loaded in %.1fs (%.1fs/layer)",
            len(self.layer_ids),
            total,
            total / len(self.layer_ids),
        )

    def forward_layer(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if layer_idx not in self.layers:
            self.load_layer(layer_idx)
        return self.layers[layer_idx].forward(hidden, topk_ids, topk_weights)


def prepare_sparkinfer_layer_from_vllm(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    device: str | torch.device = "cuda",
):
    """Prepare sparkinfer weights from vLLM's already-loaded expert tensors.

    vLLM's CUTLASS reformat stores:
      w13_weight_scale_2 = (1/checkpoint_global_scale) * input_scale
    sparkinfer computes runtime_alpha = w_global_scale / a_gscale,
    so passing ws2 as w_global_scale and input_scale as a_gscale
    yields runtime_alpha = 1/checkpoint_global_scale (correct dequant).

    Block scales are used as-is (no folding).
    """
    num_experts = w13_weight.shape[0]

    # vLLM's block scales are already in sparkinfer-compatible format;
    # applying swizzle_block_scale would double-swizzle and corrupt them.
    w13_sf = w13_weight_scale.clone().contiguous()
    w2_sf = w2_weight_scale.clone().contiguous()

    wplan = plan_sparkinfer_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=num_experts,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        w13_layout="w13",
    )
    return prepare_sparkinfer_fp4_moe_weights(
        plan=wplan,
        w1_global_scale=w13_weight_scale_2.float().clone().contiguous(),
        w2_global_scale=w2_weight_scale_2.float().clone().contiguous(),
        w1_fp4=w13_weight.clone().contiguous(),
        w1_blockscale=w13_sf,
        w2_fp4=w2_weight.clone().contiguous(),
        w2_blockscale=w2_sf,
        a1_gscale=w13_input_scale.float().clone().contiguous(),
        a2_gscale=w2_input_scale.float().clone().contiguous(),
        params_dtype=torch.bfloat16,
    )
