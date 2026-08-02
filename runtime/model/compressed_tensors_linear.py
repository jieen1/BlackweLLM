"""Self-built Linear layers for compressed-tensors "mixed-precision"
checkpoints (Track B, unsloth's ``unsloth/Qwen3.6-27B-NVFP4``). Sibling of
``runtime/model/modelopt_linear.py`` -- same per-Parameter ``weight_loader``
closure idiom, same "dequantize once to BF16, cache, run BF16xBF16
``F.linear``" B1 scope decision (see that module's docstring for why: this
runtime does not reproduce either checkpoint's *intended* quantized GEMM
execution path, correctness-first), different checkpoint naming and,
for FP8, a genuinely different physical scale layout.

Both classes' Parameter attribute names are the checkpoint's own tensor
suffixes verbatim (``weight``/``weight_scale`` for FP8;
``weight_packed``/``weight_scale``/``weight_global_scale`` for NVFP4) --
``runtime/model/qwen36_model.py``'s ``load_weights`` does no per-tensor name
remapping for the backbone, only a fixed top-level prefix strip, so this has
to line up 1:1 exactly like ``ModelOptFP8Linear``/``ModelOptNVFP4Linear``
already do for the other format.

See ``runtime/loading/compressed_tensors.py``'s module docstring for the
measured evidence (real safetensors headers, 2026-08-02) that:

- unsloth's FP8 scale is per-output-*channel* (``[out, 1]``, ``bfloat16``),
  not modelopt's per-*tensor* scalar (``float32``) -- a different function,
  :func:`~runtime.loading.compressed_tensors.dequantize_fp8_channel`, not a
  shape-tolerant variant of ``runtime.loading.modelopt.dequantize_fp8``.
- unsloth's NVFP4 sub-format (``config_groups`` ``format:
  "nvfp4-pack-quantized"``) is the same physical layout as modelopt's
  ``W4A16_NVFP4`` (block size 16, E2M1 codes, two-level scaling) -- so
  :class:`CompressedTensorsNVFP4Linear` reuses
  ``runtime.loading.modelopt.dequantize_nvfp4`` unchanged rather than
  reimplementing the unpack/dequant math, and only ``.input_global_scale``
  (this checkpoint's activation-side scale, never read here -- same "B1
  dequantizes weights only" reasoning as ``.input_scale`` for modelopt) is
  new relative to that format.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from runtime.loading.compressed_tensors import dequantize_fp8_channel
from runtime.loading.modelopt import NVFP4_GROUP_SIZE, dequantize_nvfp4
from runtime.model._weight_loading import default_weight_loader


class CompressedTensorsFP8ChannelLinear(nn.Module):
    """Per-output-channel FP8 (E4M3) weight-quantized Linear, dequantized to
    BF16 on first use.

    Checkpoint shape (verified against real safetensors headers,
    2026-08-02): ``weight`` is ``[out, in]`` ``float8_e4m3fn`` (unpacked, one
    byte per element -- same physical layout as modelopt's FP8 weight
    tensor); ``weight_scale`` is ``[out, 1]`` ``bfloat16`` -- one scale per
    output row, unlike modelopt's single ``float32`` scalar. ``.input_scale``
    does not exist for this group (``input_activations.dynamic: true`` in
    the checkpoint's own config -- a real per-token dynamic activation
    scale, computed at runtime for the intended FP8xFP8 GEMM this B1
    implementation does not perform), so there is nothing to ignore here the
    way modelopt's ``.input_scale`` is.
    """

    def __init__(self, input_size: int, output_size: int, *, bias: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, 1, dtype=torch.bfloat16), requires_grad=False
        )
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
            self._weight_bf16 = dequantize_fp8_channel(self.weight.data, self.weight_scale.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        return F.linear(x, self._weight_bf16, self.bias)


class CompressedTensorsNVFP4Linear(nn.Module):
    """Block-scaled NVFP4 (E2M1) weight-only-quantized Linear, dequantized to
    BF16 on first use -- unsloth's ``config_groups`` ``format:
    "nvfp4-pack-quantized"`` group (same format string, same physical layout,
    as Laguna's own NVFP4 -- ``runtime/backends/laguna_sparkinfer_moe.py``
    reads the identical three suffixes for its MoE experts, a separate
    pipeline; this class is the dense/Linear-path consumer for this format).

    Checkpoint shape (verified against real safetensors headers,
    2026-08-02): ``weight_packed`` is ``[out, in // 2]`` ``uint8`` (two
    4-bit E2M1 codes/byte); ``weight_scale`` is ``[out, in // group_size]``
    ``float8_e4m3fn`` (one scale per 16-element input-dim block, identical
    shape/dtype to modelopt's own NVFP4 block scale); ``weight_global_scale``
    is shape ``[1]`` ``float32`` -- the same second-level global scale
    modelopt calls ``weight_scale_2`` (shape ``()`` there; ``default_weight_
    loader``'s scalar-numel special case handles the shape difference
    transparently). A fourth tensor, ``input_global_scale``, also exists per
    module and is never read here -- see this module's own docstring.
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

        self.weight_packed = nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // group_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_global_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight_packed.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        self.weight_global_scale.weight_loader = default_weight_loader
        if self.bias is not None:
            self.bias.weight_loader = default_weight_loader

        self._weight_bf16: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            self._weight_bf16 = dequantize_nvfp4(
                self.weight_packed.data,
                self.weight_scale.data,
                self.weight_global_scale.data,
                group_size=self.group_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        return F.linear(x, self._weight_bf16, self.bias)
