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

import os

import torch
import torch.nn.functional as F
from torch import nn

from runtime.loading.compressed_tensors import dequantize_fp8_channel
from runtime.loading.modelopt import NVFP4_GROUP_SIZE, dequantize_nvfp4
from runtime.model._weight_loading import default_weight_loader

#: Pre-flight-only diagnostic switch (2026-08-03, FP8 W8A8 B1-R pre-flight --
#: see ``scripts/verify_fp8_w8a8_activation_emulation_single_layer.py`` and
#: ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py``, the
#: two consumers this flag exists for). Default OFF: production must never
#: pay this, and must never even risk paying it
#: through a stale env var left set in a shell -- ``forward()`` below reads
#: this at call time (not cached at import time) specifically so a test can
#: toggle it with ``monkeypatch.setenv``/``delenv`` and see the effect
#: immediately, matching this codebase's existing ``QSR_*`` flag idiom (e.g.
#: ``runtime/backends/laguna.py``'s ``QSR_PROFILE_MOE_PHASES``).
QSR_EMULATE_FP8_ACTIVATION_ENV = "QSR_EMULATE_FP8_ACTIVATION"


def _fp8_activation_emulation_enabled() -> bool:
    return os.environ.get(QSR_EMULATE_FP8_ACTIVATION_ENV) == "1"


def emulate_fp8_activation_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Per-token dynamic FP8 (E4M3) activation quantize/dequantize
    round-trip -- emulates the DOMINANT new error term a genuine W8A8 GEMM
    would add over today's dequantize-weight-to-BF16-and-``F.linear`` path,
    without building an FP8xFP8 kernel.

    Matches this checkpoint's own declared scheme for every FP8-channel
    target (verified directly against the standard checkpoint's
    ``config.json``, 2026-08-03: ``quantization_config.config_groups.
    group_0.input_activations`` = ``{num_bits: 8, type: float, strategy:
    "token", dynamic: true, symmetric: true}``) -- one scale per row (per
    token), derived from that row's own max-abs value, never loaded from
    the checkpoint (there is no ``.input_scale`` Parameter for this group;
    see :class:`CompressedTensorsFP8ChannelLinear`'s docstring for why).
    This is the compressed-tensors library's own standard per-token E4M3
    recipe (``scale = amax / fp8_max``, symmetric, round-to-nearest via the
    ``float8_e4m3fn`` cast, no zero-point), the same convention
    ``scripts/verify_fp8_tensor_gemm_single_layer.py::quantize_activation_fp8``
    already used for modelopt's *static per-tensor* FP8 scheme -- here
    applied per-row instead of once for the whole tensor, because this
    checkpoint's activation scale is dynamic and per-token, not static and
    per-tensor (a different checkpoint, a different declared scheme -- see
    that script's own docstring for why the two are not interchangeable).

    This is a **lower bound** on real W8A8's error, not an equivalent of
    it: a genuine ``sparkinfer``-style FP8xFP8 GEMM would also change
    accumulation order relative to today's BF16xBF16 ``F.linear``, which
    this round-trip does not touch (only the activation values change; the
    GEMM itself still runs in BF16). See this module's docstring and
    ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py`` for
    what this is being used to decide (a B1-R gap-error gate) and why a
    lower bound is sufficient for a negative verdict there.
    """
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
    x32 = x.to(torch.float32)
    amax = x32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / fp8_max
    x_fp8 = (x32 / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (x_fp8.to(torch.float32) * scale).to(x.dtype)


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

    **FP8 W8A8 pre-flight (2026-08-03,** ``QSR_EMULATE_FP8_ACTIVATION``
    **env flag, default OFF)**: :meth:`forward` optionally round-trips the
    activation through :func:`emulate_fp8_activation_round_trip` before
    ``F.linear`` -- a cheap way to measure a genuine W8A8 GEMM's *dominant*
    new error source (activation quantization) without building an FP8xFP8
    kernel, since the weight side is already dequantized exactly from the
    checkpoint's real FP8 values either way. See
    :func:`emulate_fp8_activation_round_trip`'s docstring for the scheme
    and why it is a lower, not exact, bound, and
    ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py`` for
    the full-model verdict this flag was built to produce. Default OFF
    means this can never affect production; nothing sets this env var
    except a diagnostic script or a test.
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
        if _fp8_activation_emulation_enabled():
            x = emulate_fp8_activation_round_trip(x)
        return F.linear(x, self._weight_bf16, self.bias)


class CompressedTensorsNVFP4Linear(nn.Module):
    """Block-scaled NVFP4 (E2M1) weight-only-quantized Linear, dequantized to
    BF16 on first use -- unsloth's ``config_groups`` ``format:
    "nvfp4-pack-quantized"`` group (same format string, same physical byte
    layout, as Laguna's own NVFP4 -- ``runtime/backends/laguna_sparkinfer_moe.py``
    reads the identical three suffixes for its MoE experts, a separate
    pipeline; this class is the dense/Linear-path consumer for this format).

    Checkpoint shape (verified against real safetensors headers,
    2026-08-02): ``weight_packed`` is ``[out, in // 2]`` ``uint8`` (two
    4-bit E2M1 codes/byte); ``weight_scale`` is ``[out, in // group_size]``
    ``float8_e4m3fn`` (one scale per 16-element input-dim block, identical
    shape/dtype/value-range to modelopt's own NVFP4 block scale -- both
    checkpoints' real ``layers.0.mlp.gate_proj.weight_scale`` measure
    min=4.5/max=448/mean~22-26). ``weight_global_scale`` is shape ``[1]``
    ``float32``. A fourth tensor, ``input_global_scale``, also exists per
    module and is never read here -- see this module's own docstring.

    **``weight_global_scale`` is the RECIPROCAL of modelopt's
    ``weight_scale_2``, not the same value under a different name --
    measured, not assumed (2026-08-03, after a real GPU run of this class
    produced degenerate ``"!!!!!!!!!!!!"`` output; per-layer hidden-state
    hooks showed the very first MLP layer already saturating to ~4e22).**
    Real checkpoint values for the same module (``layers.0.mlp.gate_proj``):
    unsloth's ``weight_global_scale`` = ``6624.0``; nvidia's
    ``weight_scale_2`` = ``0.0002`` -- and ``1 / 6624 ≈ 0.000151``, the same
    order of magnitude, not a coincidence. This matches, and is explained
    by, the convention ``runtime/backends/laguna_sparkinfer_moe.py``'s own
    module docstring already documents for this exact checkpoint-side
    tensor name in Laguna's MoE pipeline: ``w1_global_scale = 1 /
    checkpoint_gs`` (its ``prepare_sparkinfer_layer`` literally computes
    ``w1_alpha = (1.0 / raw["gate_gs"]).float()`` before handing it to
    sparkinfer's kernel as the multiplicative alpha) -- this class's
    ``_ensure_ready`` does the same reciprocal before calling
    :func:`~runtime.loading.modelopt.dequantize_nvfp4`, which itself
    performs a direct multiply (``per_block = weight_scale * global_scale``,
    proven correct for modelopt's checkpoint, whose ``weight_scale_2`` is
    already stored as the direct multiplier). Passing
    ``weight_global_scale`` to that function unreciprocated -- reusing its
    math but not its checkpoint-side calling convention -- is exactly the
    bug this docstring is recording so it cannot regress silently:
    dequantizing unsloth's real ``layers.0.mlp.gate_proj`` weight the wrong
    way measured mean=247.8/std=426744 (nonsense for a neural net weight);
    with the reciprocal applied it lands at std≈0.0097, the same order of
    magnitude as a normal weight. ``default_weight_loader``'s scalar-numel
    special case handles ``weight_global_scale``'s ``[1]``-vs-modelopt's
    ``()`` shape difference transparently -- that part was never the issue.
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
        # Activation-side static global scale for this checkpoint's genuine
        # W4A4 scheme (`config_groups.group_1.input_activations`: num_bits=4,
        # strategy=tensor_group, group_size=16, dynamic="local" -- a
        # calibrated per-tensor scale with the per-block e4m3 scale computed
        # at runtime, exactly `quantize_grouped_nvfp4_torch`'s two-level
        # design). Loaded but not read by this class's own `forward()` --
        # same "checkpoint data present, not this class's job" split as
        # `weight_global_scale` before this Parameter existed (see class
        # docstring). Consumer is `nvfp4_w4a4_components_for_fuse` below,
        # used only by the diagnostic scripts
        # `scripts/verify_nvfp4_w4a4_gemm_single_layer.py` /
        # `scripts/verify_nvfp4_w4a4_gemm_full_model_gap.py`
        # (2026-08-03, ``work/w4a4-20260803``): routing this checkpoint's
        # MLP through a genuine W4A4 ``sparkinfer.gemm.blockscaled.mm`` GEMM
        # measured cosine ~0.988 at the single-layer level (vs ~0.99999 for
        # the production W4A16 path) and **failed B1-R's calibrated
        # gap-error bars** at the full-model level (median/p90/p90-logprob
        # all over their bars; one of three workloads diverged badly enough
        # to overflow the diagnostic's top-1024 capture window). Not wired
        # into `Qwen36MLP.forward` for that reason -- this Parameter and the
        # method below are kept because they are correct, real checkpoint
        # data, and load-bearing evidence for that negative result, not
        # because anything reads them in production.
        self.input_global_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight_packed.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        self.weight_global_scale.weight_loader = default_weight_loader
        self.input_global_scale.weight_loader = default_weight_loader
        if self.bias is not None:
            self.bias.weight_loader = default_weight_loader

        self._weight_bf16: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            # Reciprocal, not a direct pass-through -- see class docstring
            # for the measured evidence (unsloth weight_global_scale=6624.0
            # vs modelopt weight_scale_2=0.0002 for the same real module;
            # 1/6624 ≈ 0.000151, same order of magnitude) and for why this
            # is a real checkpoint-convention difference, not a shape quirk
            # dequantize_nvfp4 already handles.
            reciprocal_global_scale = 1.0 / self.weight_global_scale.data.to(torch.float32)
            self._weight_bf16 = dequantize_nvfp4(
                self.weight_packed.data,
                self.weight_scale.data,
                reciprocal_global_scale,
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
        scalar float32 in :class:`~runtime.model.modelopt_linear.
        ModelOptNVFP4Linear`'s convention (a direct multiplier -- see
        :meth:`~runtime.model.modelopt_linear.ModelOptNVFP4Linear.
        nvfp4_components_for_fuse`'s docstring), NOT this checkpoint's own
        ``weight_global_scale`` convention. The reciprocal happens here,
        once, so every caller downstream of this method (in particular
        ``sparkinfer.moe._shared.kernels.w4a16.prepare.
        prepare_w4a16_modelopt_nvfp4_weights``, whose docstring literally
        says "raw ModelOpt weight global scales") can stay written in the
        single convention it already expects, exactly like this class's own
        :meth:`_ensure_ready` already does for its dequant-to-BF16 path --
        see this class's docstring for the measured evidence that skipping
        this reciprocal produces degenerate output (``"!!!!!!!!!!!!"``,
        2026-08-03).
        """
        reciprocal_global_scale = 1.0 / self.weight_global_scale.data.reshape(()).to(torch.float32)
        return self.weight_packed.data, self.weight_scale.data, reciprocal_global_scale

    def nvfp4_w4a4_components_for_fuse(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(packed_weight, block_scale, weight_global_scale,
        input_global_scale)`` **verbatim, unreciprocated**, for a genuine
        W4A4 ``sparkinfer.gemm.blockscaled.mm`` GEMM (both operands
        pre-quantized) -- a different convention from
        :meth:`nvfp4_components_for_fuse` above, which reciprocates
        ``weight_global_scale`` to match ``dequantize_nvfp4``'s
        direct-multiplier convention for the W4A16 dequant-inside-kernel
        path.

        **Not called from ``Qwen36MLP.forward`` or any other production
        path** -- only from the two diagnostic scripts named in this
        Parameter's own declaration comment above. The W4A4 investigation
        this method was built for (2026-08-03, ``work/w4a4-20260803``)
        concluded negatively: correct as far as it goes (this convention IS
        the one that produces sane output, see below), but genuinely
        quantizing the activation to 4-bit NVFP4 costs enough precision,
        compounded over 56 MLP layers x 3 GEMMs each, to fail B1-R's
        calibrated gap-error bars at the full-model level. Kept for the
        record and in case a future, more precise activation-quantization
        recipe revisits this -- not because production reads it.

        ``blockscaled.mm``'s own operand-building convention (matching
        ``sparkinfer._lib.intrinsics.quantize_grouped_nvfp4_torch`` and its
        oracle test, ``tests/gemm/test_blockscaled.py::_make_quantized_operand``)
        is: ``global_scale`` is the value that maps a block's real amax onto
        the e4m3 grid (``scale = global_scale * block_max / 6``, clipped),
        and the GEMM's ``alpha = 1 / (weight_gs * activation_gs)`` undoes
        *both* operands' global scales in one shot after the block-scaled
        MMA has already applied each operand's per-block e4m3 scale
        in-kernel. That is ``weight_global_scale`` used directly, not its
        reciprocal -- confirmed by working the checkpoint's own arithmetic
        backward from :meth:`nvfp4_components_for_fuse` (whose reciprocal is
        independently verified correct for the BF16-dequant path): if
        ``dequantize_nvfp4``'s ``per_block = weight_scale * (1 /
        weight_global_scale)`` is the right per-element multiplier, and
        ``quantize_grouped_nvfp4_torch`` defines that same multiplier as
        ``scale / global_scale``, then ``weight_global_scale ==
        global_scale`` (the un-reciprocated quantizer-side value) --
        checked against a real layer's numbers in
        ``scripts/verify_nvfp4_w4a4_gemm_single_layer.py`` (GPU, real
        checkpoint weights), not just derived on paper.

        ``input_global_scale`` (this checkpoint's genuine W4A4 activation
        scale, ``config_groups.group_1.input_activations``) is assumed to
        follow the **same** un-reciprocated convention as
        ``weight_global_scale`` (both are this checkpoint's own calibration
        output, produced by the same quantization tool) -- also verified
        numerically by the same script rather than only by analogy, per
        this class's own docstring note that guessing this backward
        produces plausible-looking garbage (the ``"!!!!!!!!!!!!"`` bug).
        """
        global_scale = self.weight_global_scale.data.reshape(()).to(torch.float32)
        input_global_scale = self.input_global_scale.data.reshape(()).to(torch.float32)
        return self.weight_packed.data, self.weight_scale.data, global_scale, input_global_scale

    def free_nvfp4_raw_params(self) -> None:
        """Zero out this Linear's raw NVFP4 Parameter storage
        (``.weight_packed``/``.weight_scale``/``.weight_global_scale``/
        ``.input_global_scale``) in place -- called by
        ``Qwen36MLP._free_raw_nvfp4_weights`` once the fused w13/w2
        representation built from :meth:`nvfp4_components_for_fuse` (or, for
        the W4A4 path, :meth:`nvfp4_w4a4_components_for_fuse`) no longer
        needs them. Same discipline as :meth:`~runtime.model.modelopt_linear.
        ModelOptNVFP4Linear.free_nvfp4_raw_params` -- see that method's
        docstring; the only difference is this format's own Parameter
        names (``weight_packed``/``weight_global_scale`` rather than
        ``weight``/``weight_scale_2``, plus this format's extra
        ``input_global_scale``, which modelopt's weight-only checkpoint
        does not have).
        """
        for name in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
            param = getattr(self, name)
            param.data = param.data.new_empty(0)
