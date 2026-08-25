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

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from runtime.loading.modelopt import (
    NVFP4_GROUP_SIZE,
    dequantize_fp8,
    dequantize_nvfp4,
)
from runtime.model._weight_loading import default_weight_loader

logger = logging.getLogger("qwen_sm120_runtime.modelopt_linear")

# The Qwen3.8 Gittensor export is the only production format that enters the
# native ModelOpt W4A4 classes below.  Keep its activation-quantizer A/B
# isolated from the legacy W4A16 ModelOpt and compressed-tensors/Unsloth
# routes.  ``flashinfer`` is the SM120 default, matching the CuTe-DSL
# quantizer selected by SGLang; ``local`` remains the explicit rollback.
QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV = "QSR_QWEN36_MODEL_OPT_FP4_QUANT"


def _modelopt_flashinfer_fp4_quant_enabled() -> bool:
    """Return whether ModelOpt W4A4 uses SGLang's SM120 quantizer."""

    value = os.environ.get(QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV, "flashinfer").strip().lower()
    if value not in {"local", "flashinfer"}:
        logger.warning(
            "ignoring invalid %s=%r; expected local or flashinfer",
            QSR_QWEN36_MODEL_OPT_FP4_QUANT_ENV,
            value,
        )
        return False
    return value == "flashinfer"


def _quantize_modelopt_w4a4_activation(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ModelOpt W4A4 activation into grouped b12x scale storage.

    The local Triton quantizer returns linear E4M3 scales, while FlashInfer
    returns the final 128x4-swizzled storage.  Normalize both here so the two
    W4A4 callers below differ only in the quantization kernel, not in the
    ``blockscaled.mm`` ABI.  The production default is FlashInfer's SM120
    CuTe-DSL implementation; ``QSR_QWEN36_MODEL_OPT_FP4_QUANT=local`` is the
    rollback for A/B and debugging.
    """

    if _modelopt_flashinfer_fp4_quant_enabled():
        from runtime.model.flashinfer_nvfp4 import quantize_nvfp4_activation

        packed, swizzled = quantize_nvfp4_activation(x, global_scale, backend="cute-dsl")
        return packed, swizzled.unsqueeze(0).view(torch.uint8)

    from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

    ensure_sparkinfer_path()
    from b12x._lib.intrinsics import swizzle_block_scale

    from runtime.kernels.nvfp4_quant import quantize_nvfp4_activation

    packed, linear = quantize_nvfp4_activation(x, global_scale)
    storage = swizzle_block_scale(linear.view(torch.float8_e4m3fn).unsqueeze(0)).view(torch.uint8)
    return packed, storage


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


class ModelOptNVFP4W4A4Linear(ModelOptNVFP4Linear):
    """Static ModelOpt NVFP4 W4A4 Linear for the Qwen3.8 export.

    This is intentionally a separate class from :class:`ModelOptNVFP4Linear`:
    the older Qwen3.6 ModelOpt checkpoint is W4A16 and must retain its
    weight-only semantics.  Qwen3.8 stores a real activation-side
    ``input_scale`` and declares both operands as static block-16 NVFP4.
    On CUDA this class builds the same SM120 ``blockscaled.mm`` operand ABI
    used by ``Qwen36MLP``; on CPU/non-CUDA it falls back to the exact
    BF16-dequant reference so loader and shape tests remain torch-only.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        group_size: int = NVFP4_GROUP_SIZE,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, group_size=group_size, bias=bias)
        self.input_scale = nn.Parameter(torch.empty((), dtype=torch.float32), requires_grad=False)
        self.input_scale.weight_loader = default_weight_loader
        self._w4a4_prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def nvfp4_w4a4_components_for_fuse(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return W4A4 operands in b12x's global-scale convention.

        ModelOpt writes ``weight_scale_2`` and ``input_scale`` as reciprocal
        export scales.  The b12x quantizer/GEMM ABI expects the corresponding
        quantizer-side global scales, so both are inverted here exactly once;
        callers can then use ``alpha = 1 / (activation_gs * weight_gs)`` for
        both ModelOpt and compressed-tensors NVFP4.
        """
        weight_global_scale = 1.0 / self.weight_scale_2.data.reshape(()).to(torch.float32)
        activation_global_scale = 1.0 / self.input_scale.data.reshape(()).to(torch.float32)
        return (
            self.weight.data,
            self.weight_scale.data,
            weight_global_scale,
            activation_global_scale,
        )

    def _ensure_w4a4_ready(self) -> None:
        if self._w4a4_prepared is not None:
            return
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from b12x._lib.intrinsics import as_grouped_scale_view, swizzle_block_scale

        weight, weight_scale, weight_global_scale, activation_global_scale = (
            self.nvfp4_w4a4_components_for_fuse()
        )
        if weight_global_scale.numel() != 1 or activation_global_scale.numel() != 1:
            raise ValueError("ModelOpt W4A4 global scales must be scalar tensors")
        if (
            not torch.isfinite(weight_global_scale).all()
            or not torch.isfinite(activation_global_scale).all()
        ):
            raise ValueError("ModelOpt W4A4 global scales must be finite")
        if weight_global_scale.item() == 0 or activation_global_scale.item() == 0:
            raise ValueError("ModelOpt W4A4 global scales must be non-zero")
        out_dim, in_dim = weight.shape[0], weight.shape[1] * 2
        scale_storage = swizzle_block_scale(weight_scale.unsqueeze(0).contiguous()).view(
            torch.uint8
        )
        weight_scale_view = as_grouped_scale_view(scale_storage, out_dim, in_dim)
        alpha = (1.0 / (weight_global_scale * activation_global_scale)).reshape(1).contiguous()
        self._w4a4_prepared = (weight.unsqueeze(-1), weight_scale_view, alpha)

    def _forward_w4a4(self, x: torch.Tensor) -> torch.Tensor:
        from b12x._lib.intrinsics import as_grouped_scale_view
        from b12x.gemm import blockscaled

        orig_shape = x.shape
        x2d = x.reshape(-1, self.input_size)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        m = x2d.shape[0]
        a_packed, a_scale_storage = _quantize_modelopt_w4a4_activation(
            x2d, 1.0 / self.input_scale.data
        )
        a_scale_view = as_grouped_scale_view(a_scale_storage, m, self.input_size)
        assert self._w4a4_prepared is not None
        b_packed, b_scale_view, alpha = self._w4a4_prepared
        output = blockscaled.mm(
            (a_packed.unsqueeze(-1), a_scale_view),
            (b_packed, b_scale_view),
            alpha=alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            expected_m=m,
        )[:, :, 0]
        if self.bias is not None:
            output = output + self.bias
        output = output.reshape(*orig_shape[:-1], self.output_size)
        return output.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            return super().forward(x)
        self._ensure_w4a4_ready()
        return self._forward_w4a4(x)

    def free_nvfp4_raw_params(self) -> None:
        super().free_nvfp4_raw_params()
        self.input_scale.data = self.input_scale.data.new_empty(0)


class FusedModelOptNVFP4W4A4QKV:
    """Fused full-attention Q/K/V projection for ModelOpt W4A4 weights.

    The Qwen3.8 ModelOpt export uses the same input calibration for the three
    full-attention projections.  When their second-level weight scales also
    match, one activation quantization and one block-scaled GEMM are
    mathematically equivalent to the three independent projections.  Keeping
    the scale checks here is important: one shared ``alpha`` cannot represent
    three different calibration pairs without changing the model.

    This object deliberately owns no ``nn.Parameter``.  The checkpoint loader
    and ``state_dict`` therefore continue to see the original three linears.
    The merged packed weights/scales are built lazily after loading, following
    ``FusedFP8ChannelQKV``'s ownership and CUDA-graph warmup pattern.
    """

    def __init__(
        self,
        q_proj: ModelOptNVFP4W4A4Linear,
        k_proj: ModelOptNVFP4W4A4Linear,
        v_proj: ModelOptNVFP4W4A4Linear,
    ) -> None:
        projections = (q_proj, k_proj, v_proj)
        if any(proj.bias is not None for proj in projections):
            raise ValueError("fused ModelOpt W4A4 QKV requires bias-less projections")
        if any(proj.input_size != q_proj.input_size for proj in projections):
            raise ValueError("fused ModelOpt W4A4 QKV requires a shared input size")
        if any(proj.group_size != q_proj.group_size for proj in projections):
            raise ValueError("fused ModelOpt W4A4 QKV requires a shared group size")
        self._q = q_proj
        self._k = k_proj
        self._v = v_proj
        self._weight: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._activation_global_scale: torch.Tensor | None = None
        self._out_split: tuple[int, int, int] | None = None

    @property
    def ready(self) -> bool:
        """Whether the merged CUDA operands have been prepared."""
        return self._weight is not None

    @staticmethod
    def _same_scalar(left: torch.Tensor, right: torch.Tensor) -> bool:
        return (
            left.numel() == 1
            and right.numel() == 1
            and torch.equal(left.reshape(()), right.reshape(()))
        )

    def _validate_raw_parameters(self) -> tuple[torch.Tensor, ...]:
        weights = tuple(proj.weight.data for proj in (self._q, self._k, self._v))
        scales = tuple(proj.weight_scale.data for proj in (self._q, self._k, self._v))
        if any(weight.numel() == 0 for weight in weights):
            raise RuntimeError("fused ModelOpt W4A4 QKV needs raw weights before they are released")
        if any(scale.numel() == 0 for scale in scales):
            raise RuntimeError(
                "fused ModelOpt W4A4 QKV needs raw block scales before they are released"
            )
        if any(weight.ndim != 2 or scale.ndim != 2 for weight, scale in zip(weights, scales)):
            raise ValueError("fused ModelOpt W4A4 QKV requires rank-2 raw tensors")
        if any(weight.shape[1] != self._q.input_size // 2 for weight in weights):
            raise ValueError("fused ModelOpt W4A4 QKV has an unexpected packed input width")
        if any(scale.shape[1] != self._q.input_size // self._q.group_size for scale in scales):
            raise ValueError("fused ModelOpt W4A4 QKV has an unexpected block-scale width")
        input_scales = tuple(proj.input_scale.data for proj in (self._q, self._k, self._v))
        weight_scales_2 = tuple(proj.weight_scale_2.data for proj in (self._q, self._k, self._v))
        if not all(self._same_scalar(input_scales[0], scale) for scale in input_scales[1:]):
            raise ValueError("fused ModelOpt W4A4 QKV requires matching input_scale values")
        if not all(self._same_scalar(weight_scales_2[0], scale) for scale in weight_scales_2[1:]):
            raise ValueError("fused ModelOpt W4A4 QKV requires matching weight_scale_2 values")
        return weights + scales + (input_scales[0], weight_scales_2[0])

    def prepare(self) -> None:
        """Prepare the merged block-scaled operand once, outside graph replay."""
        if self._weight is not None:
            return
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from b12x._lib.intrinsics import as_grouped_scale_view, swizzle_block_scale

        raw = self._validate_raw_parameters()
        q_weight, k_weight, v_weight = raw[:3]
        q_scale, k_scale, v_scale = raw[3:6]
        input_scale, weight_scale_2 = raw[6:]
        merged_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()
        merged_scale = torch.cat((q_scale, k_scale, v_scale), dim=0).contiguous()
        output_size = merged_weight.shape[0]
        input_size = self._q.input_size
        scale_view = as_grouped_scale_view(
            swizzle_block_scale(merged_scale.unsqueeze(0)).view(torch.uint8),
            output_size,
            input_size,
        )
        weight_global_scale = 1.0 / weight_scale_2.reshape(()).to(torch.float32)
        activation_global_scale = 1.0 / input_scale.reshape(()).to(torch.float32)
        if (
            not torch.isfinite(weight_global_scale).all()
            or not torch.isfinite(activation_global_scale).all()
        ):
            raise ValueError("fused ModelOpt W4A4 QKV scales must be finite")
        if weight_global_scale.item() == 0 or activation_global_scale.item() == 0:
            raise ValueError("fused ModelOpt W4A4 QKV scales must be non-zero")
        self._weight = merged_weight
        self._weight_scale = scale_view
        self._alpha = (1.0 / (weight_global_scale * activation_global_scale)).reshape(1)
        self._activation_global_scale = activation_global_scale
        self._out_split = (q_weight.shape[0], k_weight.shape[0], v_weight.shape[0])

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.prepare()
        assert self._weight is not None
        assert self._weight_scale is not None
        assert self._alpha is not None
        assert self._activation_global_scale is not None
        assert self._out_split is not None
        from b12x._lib.intrinsics import as_grouped_scale_view
        from b12x.gemm import blockscaled

        original_shape = x.shape
        x2d = x.reshape(-1, self._q.input_size)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        m = x2d.shape[0]
        a_packed, a_scale_storage = _quantize_modelopt_w4a4_activation(
            x2d, self._activation_global_scale
        )
        a_scale_view = as_grouped_scale_view(a_scale_storage, m, self._q.input_size)
        output = blockscaled.mm(
            (a_packed.unsqueeze(-1), a_scale_view),
            (self._weight.unsqueeze(-1), self._weight_scale),
            alpha=self._alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            expected_m=m,
        )[:, :, 0]
        q_size, k_size, v_size = self._out_split
        lead = original_shape[:-1]
        q_end = q_size
        k_end = q_end + k_size
        return (
            output[:, :q_end].view(*lead, q_size).to(x.dtype),
            output[:, q_end:k_end].view(*lead, k_size).to(x.dtype),
            output[:, k_end : k_end + v_size].view(*lead, v_size).to(x.dtype),
        )
