"""Qwen MoE family block for the hybrid qwen38 graph (Flash-Next day-0 prep).

See notes/2026-08-26-qwen38-flash-next-day0-survey.md for the pinned
sources. The routing/expert contract is the Qwen3-Next / Qwen3.5-MoE family
reference (transformers ``Qwen3NextTopKRouter`` + ``Qwen3NextSparseMoeBlock``,
verified against the installed transformers 5.x and vLLM's qwen3_next on
2026-08-26):

* router: ``softmax(FP32(logits)) -> top-k -> renormalize`` (NOT Laguna's
  sigmoid + correction bias -- :mod:`runtime.laguna_router` cannot serve
  this path);
* routed experts: b12x NVFP4 fused MoE, attached post-construction exactly
  like ``LagunaMoESelfBuilt`` + ``LagunaBackend._patch_moe_sparkinfer``;
* one shared dense SwiGLU expert gated by ``sigmoid(shared_expert_gate(x))``;
* routed output + shared output, no routed scaling factor in the family
  reference.

The module stays import-safe without b12x: everything b12x-specific is
attached via :meth:`QwenMoeLayer.attach_experts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from runtime.kernels.qwen_moe_router import qwen_moe_softmax_topk
from runtime.model.plain_linear import PlainLinear

if TYPE_CHECKING:
    from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer


@dataclass(frozen=True)
class QwenMoeGeometry:
    """Fixed MoE geometry for one Qwen-family checkpoint."""

    num_experts: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    renormalize: bool = True


class QwenSharedExpert(nn.Module):
    """Dense SwiGLU shared expert with the family's sigmoid gate."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = PlainLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = PlainLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = PlainLinear(intermediate_size, hidden_size, bias=False)
        self.shared_gate = PlainLinear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.shared_gate(hidden_states))
        return gate * self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class QwenMoeLayer(nn.Module):
    """One family MoE block: router -> fused routed experts + shared expert.

    CUDA-Graph safe by construction: the router outputs live in
    caller-visible pre-allocated arenas sized for ``max_rows`` (the decode
    graph's batch), and the attached expert layer keeps address-stable
    weights/workspace exactly like the Laguna patched path.
    """

    def __init__(
        self,
        geometry: QwenMoeGeometry,
        *,
        max_rows: int,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        self.geometry = geometry
        self.gate = PlainLinear(geometry.hidden_size, geometry.num_experts, bias=False)
        self.shared_expert = QwenSharedExpert(
            geometry.hidden_size, geometry.shared_expert_intermediate_size
        )
        self.to(device=device, dtype=dtype)
        self._router_weights = torch.empty((max_rows, geometry.top_k), dtype=dtype, device=device)
        self._router_ids = torch.empty((max_rows, geometry.top_k), dtype=torch.int32, device=device)
        self._expert_layer: SparkinferMoELayer | None = None

    @property
    def expert_layer(self) -> SparkinferMoELayer | None:
        return self._expert_layer

    def attach_experts(self, expert_layer: SparkinferMoELayer) -> None:
        self._expert_layer = expert_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._expert_layer is None:
            raise RuntimeError("QwenMoeLayer experts are not attached")
        orig_shape = hidden_states.shape
        hs = hidden_states.view(-1, hidden_states.shape[-1])
        rows = hs.shape[0]
        if rows > self._router_weights.shape[0]:
            raise ValueError(
                f"QwenMoeLayer router arena holds {self._router_weights.shape[0]} rows, got {rows}"
            )
        router_logits = self.gate(hs)
        topk_weights, topk_ids = qwen_moe_softmax_topk(
            router_logits,
            self.geometry.top_k,
            renormalize=self.geometry.renormalize,
            weights_out=self._router_weights[:rows],
            ids_out=self._router_ids[:rows],
        )
        routed = self._expert_layer.forward(hs, topk_ids, topk_weights)
        shared = self.shared_expert(hs)
        return (routed + shared).view(orig_shape)
