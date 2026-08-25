"""Native FP8 execution for an eligible plain Qwen lm_head.

The ModelOpt W4A4 Qwen3.8 export intentionally leaves ``lm_head`` in BF16.
That is the largest single read-only matrix in the decode path, so its
BF16 GEMM is visible in every DFlash2 and DSpark round.  This module
quantizes that already-loaded BF16 matrix once to per-output-channel E4M3
and uses SparkInfer's native SM120 FP8 GEMM with dynamic per-token
activation scales.  It is enabled by default only for the ModelOpt
checkpoint family that was measured here; ``QSR_NATIVE_QWEN38_LM_HEAD_FP8=0``
provides an immediate BF16 rollback, while ``=1`` explicitly enables the
eligible plain-head path.

The class is constructed only after checkpoint loading and is deliberately
not a checkpoint-loading module.  Keeping the conversion behind a loader
hook means compressed-tensors and GGUF checkpoints remain untouched, while
the quantized object can be shared by target and DFlash2 draft models before
CUDA Graph capture.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from runtime.model.compressed_tensors_linear import quantize_fp8_activation_per_token
from runtime.model.plain_linear import PlainLinear

QSR_NATIVE_QWEN38_LM_HEAD_FP8_ENV = "QSR_NATIVE_QWEN38_LM_HEAD_FP8"

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_SCALE_FLOOR = 1.0 / (_FP8_MAX * 512.0)


def native_fp8_lm_head_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether the eligible ModelOpt head should use native FP8.

    The explicit environment setting always wins. Without one, only the
    ModelOpt format opts in: the compressed-tensors/Unsloth route has its own
    checkpoint-native head policy and must not be changed by this optimization.
    """

    explicit = os.environ.get(QSR_NATIVE_QWEN38_LM_HEAD_FP8_ENV)
    if explicit is not None:
        return explicit == "1"
    quantization_config = config.get("quantization_config")
    return (
        isinstance(quantization_config, Mapping)
        and quantization_config.get("quant_method") == "modelopt"
    )


class NativeFP8LMHead(nn.Module):
    """Native SM120 FP8 replacement for a loaded :class:`PlainLinear` head.

    The weight is quantized per output row.  Activations use the same
    per-token E4M3 contract as the existing compressed-tensors FP8 path.  The
    b12x operation returns BF16; the two scale vectors are applied in FP32
    before the final BF16 cast.  ``expected_m`` is fixed by each caller's
    CUDA-Graph shape, so no graph replay allocates or recompiles a new
    geometry.
    """

    def __init__(
        self,
        *,
        packed_weight: Any,
        weight_scale: torch.Tensor,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.register_buffer("weight_scale", weight_scale.contiguous(), persistent=False)
        # TensorFP8LinearWeight is an immutable dataclass owned by b12x.  It
        # is intentionally kept as a private execution cache rather than a
        # state_dict entry: this object is produced after checkpoint loading
        # and is rebuilt from the BF16 checkpoint when the path is enabled.
        self._packed_weight = packed_weight

    @classmethod
    def from_plain_linear(cls, linear: PlainLinear) -> NativeFP8LMHead:
        """Quantize ``linear.weight`` and build the native b12x weight cache."""

        if linear.weight.device.type != "cuda":
            raise RuntimeError("native FP8 lm_head requires CUDA-resident weights")
        if linear.bias is not None:
            raise ValueError("native FP8 lm_head only supports bias-free projections")

        from b12x.gemm import tensor_fp8_linear

        with torch.inference_mode():
            weight_f32 = linear.weight.detach().float()
            weight_scale = (weight_f32.abs().amax(dim=1) / _FP8_MAX).clamp_min(
                _FP8_SCALE_FLOOR
            )
            weight_fp8 = (weight_f32 / weight_scale[:, None]).clamp(
                -_FP8_MAX, _FP8_MAX
            ).to(torch.float8_e4m3fn)
            packed_weight = tensor_fp8_linear.pack_weight(
                weight_fp8,
                torch.ones(1, dtype=torch.float32, device=weight_fp8.device),
            )
            result = cls(
                packed_weight=packed_weight,
                weight_scale=weight_scale,
                input_size=linear.input_size,
                output_size=linear.output_size,
            )
            del weight_fp8, weight_f32
            torch.cuda.empty_cache()
            return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            raise RuntimeError("native FP8 lm_head requires CUDA hidden states")
        if hidden_states.shape[-1] != self.input_size:
            raise ValueError(
                f"lm_head hidden size {hidden_states.shape[-1]} does not match "
                f"input size {self.input_size}"
            )

        from b12x.gemm import tensor_fp8_linear

        shape = hidden_states.shape
        source = hidden_states.reshape(-1, self.input_size).contiguous()
        source_fp8, activation_scale = quantize_fp8_activation_per_token(source)
        raw = tensor_fp8_linear.mm(
            source_fp8,
            self._packed_weight,
            out_dtype=torch.bfloat16,
            expected_m=int(source.shape[0]),
        )
        output = (
            raw.float() * activation_scale * self.weight_scale.view(1, -1)
        ).to(torch.bfloat16)
        return output.view(*shape[:-1], self.output_size)


__all__ = [
    "NativeFP8LMHead",
    "QSR_NATIVE_QWEN38_LM_HEAD_FP8_ENV",
    "native_fp8_lm_head_enabled",
]
