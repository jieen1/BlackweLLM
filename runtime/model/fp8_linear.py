"""FP8 weight-only quantized Linear for SM120 native tensor core dispatch.

SM120 (RTX PRO 6000 Blackwell) has NO BF16 tensor core. F.linear with BF16
weights dispatches to SM80-compatible mma.sync kernels. SM120 native MMA
is FP8 (f8f6f4) and NVFP4 only.

This module quantizes weights to FP8 e4m3 at load time and uses
torch._scaled_mm for M>=4 (verify path), falling back to F.linear for M<4
(draft decode, where cuBLAS FP8 is unsupported on SM120).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_SHARD_ID_TO_IDX = {"q": 0, "k": 1, "v": 2}


class FP8Linear(nn.Module):
    """Drop-in PlainLinear replacement with FP8 weight-only quantization.

    Stores weights as FP8 e4m3 + per-tensor FP32 scale.
    M >= 4: torch._scaled_mm (SM120 native FP8 MMA, ~1.77x vs BF16)
    M <  4: F.linear with BF16 weights (SM80 fallback, cuBLAS FP8 unsupported)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        shard_sizes: list[int] | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.shard_sizes = list(shard_sizes) if shard_sizes else [output_size]
        assert sum(self.shard_sizes) == self.output_size
        offsets = []
        running = 0
        for s in self.shard_sizes:
            offsets.append(running)
            running += s
        self.shard_offsets = offsets

        # BF16 parameter for checkpoint loading (also used as M<4 fallback)
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight.weight_loader = self._make_weight_loader("weight")
        if self.bias is not None:
            self.bias.weight_loader = self._make_weight_loader("bias")

        # FP8 quantized weight + scale (created lazily on first forward)
        self._weight_fp8: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None
        self._quantized = False

    def _make_weight_loader(self, param_name: str):
        def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id=None):
            if (
                shard_id is None
                and len(self.shard_sizes) > 1
                and loaded_weight.shape[0] == param.data.shape[0]
            ):
                param.data.copy_(loaded_weight)
                return
            if shard_id is None:
                shard_idx = 0
            elif isinstance(shard_id, str):
                shard_idx = _SHARD_ID_TO_IDX[shard_id]
            else:
                shard_idx = shard_id
            offset = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]
            dst = param.data.narrow(0, offset, size)
            dst.copy_(loaded_weight)
        return weight_loader

    def _ensure_quantized(self) -> None:
        """Lazily quantize BF16 weight to FP8 on first forward call."""
        if self._quantized:
            return
        w = self.weight.data.float()
        scale = w.abs().amax() / _FP8_MAX
        scale = scale.clamp(min=1e-12)
        self._weight_scale = scale.reshape(1).to(self.weight.device)
        w_scaled = (w / scale).clamp(-_FP8_MAX, _FP8_MAX)
        self._weight_fp8 = w_scaled.to(torch.float8_e4m3fn)
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_quantized()

        M = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]

        if M >= 4 and self._weight_fp8 is not None:
            # SM120-native FP8 path via torch._scaled_mm
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            # Dynamic per-tensor activation quantization
            x_amax = x_2d.abs().amax()
            x_scale = (x_amax.float() / _FP8_MAX).clamp(min=1e-12)
            x_fp8 = (x_2d.float() / x_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            # TN layout: weight_fp8 is [N,K], .t() gives [K,N] col-major view
            out = torch._scaled_mm(
                x_fp8,
                self._weight_fp8.t(),
                scale_a=x_scale.reshape(1),
                scale_b=self._weight_scale,
                out_dtype=torch.bfloat16,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*orig_shape[:-1], -1) if x.dim() != 2 else out
        else:
            # M < 4 fallback: BF16 F.linear (SM80 mma.sync)
            return F.linear(x, self.weight, self.bias)
