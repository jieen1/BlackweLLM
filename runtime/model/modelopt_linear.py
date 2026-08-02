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
oversight, still true as of the 2026-08-03 FP8 investigation below.

**FP8 kernel investigation (2026-08-03, follow-up to the NVFP4-GEMM
round -- see ``notes/2026-08-03-nvfp4-raw-param-free-and-fp8-w8a8-probe.md``
for the full writeup)**: checked whether this class's ~14 GiB BF16 dequant
cache (self_attn/GDN's FP8 projections across all 64 layers) could get the
same treatment NVFP4's MLP got. The checkpoint's own scheme was verified
first (learned from the NVFP4 round's ``blockscaled.mm`` mistake): every
FP8-quantized projection genuinely IS static per-tensor W8A8 --
``config_groups.group_0`` declares both ``weights`` and
``input_activations`` as 8-bit float with ``dynamic: false``, and every
real layer ships an actual scalar ``input_scale`` tensor (confirmed off
real safetensors headers, not assumed from the config alone) -- unlike
NVFP4's W4A16 (weight-only) checkpoint, which declares no
``input_activations`` scheme at all. ``self.input_scale`` below (added
this round) loads that real checkpoint data. ``sparkinfer.gemm.
tensor_fp8_linear`` ("static per-tensor FP8 linear for SM12x") matches
this scheme exactly -- a real kernel entry point exists, unlike the
scheme mismatch that sank the first NVFP4 attempt.

**Not wired into this class's `forward()` despite the scheme matching**:
``scripts/verify_fp8_tensor_gemm_single_layer.py`` measured the kernel's
actual precision against real checkpoint weights (self_attn.q_proj,
linear_attn.in_proj_qkv) and found cosine ~0.9996 -- genuinely working,
but roughly 30-40x further from 1.0 than NVFP4's fused kernel measured
(~0.99998) on the equivalent single-layer check. That gap is not a bug in
either kernel; it is what genuinely quantizing BOTH operands to
per-tensor FP8 costs, on top of a checkpoint that (unlike NVFP4's
weight-only scheme) was never going to let this be "free" the way the
NVFP4 fusion was (that kernel dequantizes weight against an
UN-quantized BF16 activation -- no new error source at all). Applied
across every layer's attention/GDN projections (a larger footprint than
NVFP4's MLP-only fusion) on top of B1-R's full-model gap error already
sitting at its calibration bar's own noise floor after the NVFP4 fusion
(``notes/2026-08-03-nvfp4-gemm-memory-audit.md``), wiring this into
:meth:`Qwen36Attention.forward`/``decode_batch``/
:meth:`Qwen36GatedDeltaNet.forward`/``spec_forward`` (CUDA-graph-safety-
critical real serving code, not a diagnostic) without first re-running
the full B1-R gap-error harness was judged too large a correctness risk
for this round's scope -- reported as a finding, not implemented. See the
note above for the exact numbers and what a follow-up round would need to
do differently.

Separately, ``sparkinfer.gemm.tensor_fp8_linear.is_supported()`` reports
``False`` on this machine's installed ``nvidia-cutlass-dsl`` (4.5.2 <
sparkinfer's own ``MIN_CUTLASS_DSL = "4.6.0"`` pin) even though the
kernel call itself works correctly once called directly (verified, see
the note) -- an environment/packaging gap in ``sparkinfer`` (a separate
project this worktree only consumes), not a scheme mismatch and not
something to patch from here.

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

    ``input_scale`` (also a single ``float32`` scalar, 2026-08-03 follow-up
    to the NVFP4-GEMM round -- see ``runtime/model/qwen36_model.py``'s
    ``Qwen36Attention``/``Qwen36GatedDeltaNet`` docstrings for the caller
    that actually reads it) is loaded here purely as checkpoint data: this
    class's own ``forward()``/``_ensure_ready()`` below are UNCHANGED and
    never read it, exactly the same "leave the submodule's own legacy path
    alone, fuse/route one level up" split NVFP4 used
    (``runtime/model/qwen36_model.py::Qwen36MLP``) -- kept that way on
    purpose so ``_bmm_project`` (real GDN spec-decode code, not just a
    diagnostic) and ``scripts/b3_probe_batching_bar.py`` keep getting the
    same BF16-dequant-and-cache ``_weight_bf16`` they always have, whether
    or not the owning module's forward routes through a real FP8xFP8 kernel
    instead.
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
        self.input_scale = nn.Parameter(torch.empty((), dtype=torch.float32), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        self.input_scale.weight_loader = default_weight_loader
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

    def nvfp4_components_for_fuse(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(packed_weight, block_scale, global_scale)`` for
        ``Qwen36MLP``'s fused w13/w2 W4A16 path (``runtime/model/
        qwen36_model.py``) -- packed weight ``[out, in // 2]`` uint8, block
        scale ``[out, in // group_size]`` float8_e4m3fn, global scale a
        scalar float32 already in ``prepare_w4a16_modelopt_nvfp4_weights``'s
        expected convention (that function's own docstring: "raw ModelOpt
        weight global scales" -- this class's own ``weight_scale_2`` IS that
        convention, used as-is by :func:`~runtime.loading.modelopt.
        dequantize_nvfp4`'s direct multiply, ``per_block = weight_scale *
        global_scale``). See :meth:`~runtime.model.compressed_tensors_linear.
        CompressedTensorsNVFP4Linear.nvfp4_components_for_fuse` for the
        sibling checkpoint format's, which is the RECIPROCAL of this
        convention and must invert before returning -- so ``Qwen36MLP``
        never needs an isinstance branch to know which format it holds.
        """
        return (
            self.weight.data,
            self.weight_scale.data,
            self.weight_scale_2.data.reshape(()).to(torch.float32),
        )

    def free_nvfp4_raw_params(self) -> None:
        """Zero out this Linear's raw NVFP4 Parameter storage (``.weight``/
        ``.weight_scale``/``.weight_scale_2``) in place -- called by
        ``Qwen36MLP._free_raw_nvfp4_weights`` once the fused w13/w2
        representation built from :meth:`nvfp4_components_for_fuse` no
        longer needs them. Reassigns each Parameter's ``.data`` to a
        0-element tensor (same discipline as the pre-existing per-format
        loop this replaces) rather than deleting/``None``-ing the
        ``nn.Parameter``, so ``named_parameters()``/direct attribute access
        still finds a real tensor of the right dtype/device at the expected
        name.
        """
        for name in ("weight", "weight_scale", "weight_scale_2"):
            param = getattr(self, name)
            param.data = param.data.new_empty(0)
