"""Self-built Linear layers for NVIDIA ModelOpt-quantized checkpoints
(Track B / B1). Sibling of ``runtime/model/plain_linear.py`` and
``runtime/model/fp8_linear.py`` (Laguna's weight-only FP8 Linear) -- same
per-Parameter ``weight_loader`` closure idiom, different checkpoint format.

**B1 scope decision, stated once here rather than repeated per class**:
both classes below dequantize their weight to BF16 once (lazily, on first
forward -- same "materialize on first use" idiom as
``runtime/model/fp8_linear.py::FP8Linear._ensure_ready``) and then run a
plain BF16 x BF16 ``F.linear``. Neither reproduces this checkpoint's
*intended* execution path (real FP8xFP8 / block-scaled-FP4xFP4 GEMM
kernels) -- that is a deliberate simplification for B1
("eager, batch=1, no CUDA graph, no speculation, no prefix cache -- only
ship it correct"; ``docs/implementation-plan.md`` §7.1), not an
oversight. ``docs/qwen36-rebuild-spec.md`` §3.4/§5.2 already treats GEMM
kernel selection (sparkinfer's ``moe.fused_moe(quant_mode="w4a16")`` for
NVFP4, an FP8 path for the attention/GDN projections) as a
[待验证 GPU] question separate from correctness -- revisit these classes
when that question is actually answered, not before. Every ``.input_scale``
checkpoint tensor these modules' quantized siblings carry is deliberately
never read for the same reason: it is an activation-side scale for a real
quantized GEMM this B1 implementation does not perform.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from runtime.loading.modelopt import (
    NVFP4_GROUP_SIZE,
    dequantize_fp8,
    dequantize_nvfp4,
)
from runtime.model._weight_loading import default_weight_loader


class ModelOptFP8Linear(nn.Module):
    """Per-tensor FP8 (E4M3) weight-quantized Linear, dequantized to BF16.

    Checkpoint shape (verified against real safetensors headers, B0-2):
    ``weight`` is ``[out, in]`` ``float8_e4m3fn`` (unpacked, one byte per
    element); ``weight_scale`` is a single ``float32`` scalar.
    """

    def __init__(self, input_size: int, output_size: int, *, bias: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(torch.empty((), dtype=torch.float32), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        if self.bias is not None:
            self.bias.weight_loader = default_weight_loader

        self._weight_bf16: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            self._weight_bf16 = dequantize_fp8(self.weight.data, self.weight_scale.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        return F.linear(x, self._weight_bf16, self.bias)


class ModelOptNVFP4Linear(nn.Module):
    """Block-scaled NVFP4 (E2M1) weight-only-quantized Linear ("W4A16"),
    dequantized to BF16.

    Checkpoint shape (verified against real safetensors headers, B0-2):
    ``weight`` is ``[out, in // 2]`` ``uint8`` (two 4-bit codes/byte);
    ``weight_scale`` is ``[out, in // group_size]`` ``float8_e4m3fn`` (one
    scale per 16-element input-dim block); ``weight_scale_2`` is a single
    ``float32`` scalar (global second-level scale). See
    ``runtime/loading/modelopt.py`` for the exact dequantization formula
    and what has/has not been independently verified about it.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        group_size: int = NVFP4_GROUP_SIZE,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(f"NVFP4 packs 2 elements/byte; input_size={input_size} must be even")
        if input_size % group_size != 0:
            raise ValueError(
                f"input_size={input_size} must be a multiple of group_size={group_size}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // group_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale_2 = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        self.weight_scale_2.weight_loader = default_weight_loader
        if self.bias is not None:
            self.bias.weight_loader = default_weight_loader

        self._weight_bf16: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            self._weight_bf16 = dequantize_nvfp4(
                self.weight.data,
                self.weight_scale.data,
                self.weight_scale_2.data,
                group_size=self.group_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        return F.linear(x, self._weight_bf16, self.bias)
