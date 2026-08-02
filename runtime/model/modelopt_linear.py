"""Self-built Linear layers for NVIDIA ModelOpt-quantized checkpoints
(Track B / B1, plus the NVFP4-GEMM follow-up in ``work/nvfp4-gemm-20260802``).
Sibling of ``runtime/model/plain_linear.py`` and ``runtime/model/fp8_linear.py``
(Laguna's weight-only FP8 Linear) -- same per-Parameter ``weight_loader``
closure idiom, different checkpoint format.

**B1 scope decision (FP8 only now -- see below for NVFP4)**:
:class:`ModelOptFP8Linear` still dequantizes its weight to BF16 once
(lazily, on first explicit ``_ensure_ready()`` call -- same "materialize
on first use" idiom as ``runtime/model/fp8_linear.py::FP8Linear._ensure_ready``)
and runs a plain BF16 x BF16 ``F.linear``. This does not reproduce the
checkpoint's *intended* W8A8 execution path -- that remains a deliberate
B1 simplification (``docs/implementation-plan.md`` §7.1), not an
oversight. ``.input_scale`` is still never read for FP8, for the same
reason.

**NVFP4's dense-MLP hot path no longer dequantizes to BF16 -- but the fix
lives one level up, not in this class.** GPU-measured 2026-08-03
(``notes/2026-08-03-nvfp4-gemm-memory-audit.md``): a CUDA-graph warmup
forward through all 64 layers grew resident memory from 27.3 GiB to
76.1 GiB -- 18.8 GiB of genuinely-quantized weight plus a **49.7 GiB BF16
dequant cache** that bought nothing (still a BF16xBF16 GEMM, no memory
saved since both copies stayed resident).

Two things were tried, in order, on ``work/nvfp4-gemm-20260802``:

1. Route ``ModelOptNVFP4Linear.forward()`` itself through
   ``sparkinfer.gemm.blockscaled.mm`` (a real NVFP4xNVFP4 block-scaled
   GEMM), with the BF16 activation dynamically quantized to NVFP4 per
   call. This dropped memory as intended but **failed B1-R's calibrated
   gap-error bars** (worse than the weakest injected bug B1-R was
   calibrated against): ``blockscaled.mm`` requires *both* operands
   quantized, but this checkpoint's declared scheme is W4A16 (weight-only
   -- no activation scale in ``config_groups.group_1.input_activations``),
   so the dynamic activation quantization was pure unintended error with
   no checkpoint-side counterpart, compounding badly over 64 layers.
2. **What's actually in this class now**: ``forward()`` is the plain
   legacy dequant-to-BF16 path (same as :class:`ModelOptFP8Linear` below),
   used directly by whatever constructs a bare ``ModelOptNVFP4Linear``
   (currently only ``lm_head`` -- see ``runtime/model/qwen36_model.py``).
   The real fix for the 49.7 GiB problem is in
   :class:`~runtime.model.qwen36_model.Qwen36MLP`, which is where NVFP4
   actually lives for every MLP layer (``mlp.{gate,up,down}_proj``, 64
   layers, the overwhelming majority of NVFP4 parameter count -- B0-2):
   that class fuses its three ``ModelOptNVFP4Linear`` submodules into ONE
   call to ``sparkinfer.moe._shared.kernels.w4a16.kernel.run_w4a16_moe``,
   the genuine weight-only W4A16 kernel (dequantizes NVFP4 *inside* the
   kernel against the real BF16 activation -- no activation quantization
   error at all, unlike attempt 1 above) -- see that class's docstring for
   why the fusion has to happen at the MLP level and can't be expressed as
   this class's own ``forward()``. This class's ``.weight``/
   ``.weight_scale``/``.weight_scale_2`` Parameters (unpacked, unmodified)
   are what ``Qwen36MLP`` reads to build that fused representation --
   ``forward()`` below is simply never called for those three submodules
   in ordinary inference.

``lm_head`` (the one NVFP4 module ``Qwen36MLP`` doesn't own) stays on this
class's plain dequant-to-BF16 ``forward()`` deliberately: it is a single
non-gated projection (no SwiGLU partner to fuse with), the W4A16 MoE
kernel's ABI has no bare single-GEMM entry point (see ``Qwen36MLP``
docstring), and its BF16 dequant cache is cheap in isolation (~1.2 GiB for
one ``[248320, 2560]`` tensor, not 49.7 GiB across 192 MLP Linears) --
paying that instead of routing final logits through an approximate kernel
is the correct trade given logits are what every downstream correctness
metric reads directly.
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
    """Block-scaled NVFP4 (E2M1) weight-only-quantized Linear, dequantized to
    BF16 once and computed as a plain ``F.linear`` -- see module docstring
    for why this class does NOT itself run a real NVFP4 GEMM (that lives in
    :class:`~runtime.model.qwen36_model.Qwen36MLP`, fused across this
    class's three MLP-projection instances) and for the tried-and-reverted
    ``sparkinfer.gemm.blockscaled.mm`` attempt.

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
