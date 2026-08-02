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

**NVFP4 is no longer dequantized to BF16 in the hot path.** GPU-measured
2026-08-03 (``notes/2026-08-03-nvfp4-gemm-memory-audit.md``): a CUDA-graph
warmup forward through all 64 layers grew resident memory from 27.3 GiB to
76.1 GiB -- 18.8 GiB of genuinely-quantized weight plus a **49.7 GiB BF16
dequant cache** that bought nothing (still a BF16xBF16 GEMM, no memory
saved since both copies stayed resident). :class:`ModelOptNVFP4Linear` now
runs a real block-scaled NVFP4xNVFP4 GEMM via ``sparkinfer.gemm.blockscaled.mm``
(``runtime/backends/laguna_sparkinfer_moe.py`` is the reference this was
modeled on -- Laguna's MoE experts never dequantized either): the packed
weight and its checkpoint block scale are prepared once (swizzled into the
kernel's MMA scale layout; no value materialization), and the BF16
activation is dynamically quantized to NVFP4 *per forward call* (cheap --
one amax reduction plus a block-scale pack, not a persistent cache) via
``sparkinfer._lib.intrinsics.quantize_grouped_nvfp4_torch``. This is a
genuine precision-path change from B1, not just a reordering: the
checkpoint's declared scheme is W4A16 (weight-only quant, no activation
scale in ``config_groups.group_1.input_activations``), and running a
block-scaled ``mm`` requires *both* operands quantized, so activations now
carry their own dynamic NVFP4 quantization error on top of the weight's.
Verify accordingly -- do not assume bit-exactness with the old BF16 path;
see ``docs/b1-correctness-criterion.md`` for the tolerance this is judged
against.

The legacy ``_ensure_ready()`` / ``_weight_bf16`` dequant-to-BF16 path on
:class:`ModelOptNVFP4Linear` is kept, but **only as an explicit opt-in**
for diagnostics that want a real BF16 reference weight
(``scripts/b1_verify_greedy_alignment.py``, ``scripts/b3_probe_batching_bar.py``)
-- ``forward()`` below never calls it, so it never becomes a resident
cache during ordinary inference.
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
    """Block-scaled NVFP4 (E2M1) weight-only-quantized Linear, computed as a
    genuine NVFP4xNVFP4 block-scaled GEMM (``sparkinfer.gemm.blockscaled.mm``)
    -- see module docstring for why (49.7 GiB dead BF16 dequant cache, no
    speed or memory benefit) and for what changed in the precision path.

    Checkpoint shape (verified against real safetensors headers, B0-2):
    ``weight`` is ``[out, in // 2]`` ``uint8`` (two 4-bit codes/byte);
    ``weight_scale`` is ``[out, in // group_size]`` ``float8_e4m3fn`` (one
    scale per 16-element input-dim block); ``weight_scale_2`` is a single
    ``float32`` scalar (global second-level scale). See
    ``runtime/loading/modelopt.py`` for the exact dequantization formula
    and what has/has not been independently verified about it.

    Scale convention this class's GEMM path relies on, worked out by
    matching ``runtime/loading/modelopt.py::dequantize_nvfp4``'s formula
    (``value = code * weight_scale[block] * weight_scale_2``) against
    sparkinfer's block-scaled-GEMM convention (``value = code * scale[block]
    / global_scale``, from ``quantize_grouped_nvfp4_torch``): the two are
    the same form with ``global_scale = 1 / weight_scale_2`` -- exactly
    ``runtime/backends/laguna_sparkinfer_moe.py``'s ``w1_alpha = 1 /
    checkpoint_gs`` for Laguna's MoE experts, same checkpoint convention,
    same reciprocal. Empirically confirmed (not just derived), see the
    single-layer cosine/max_abs_err check this class's tests exercise.
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

        # Legacy dequant-to-BF16 cache -- opt-in only, see module docstring.
        # ``forward()`` never touches this.
        self._weight_bf16: torch.Tensor | None = None

        # Real NVFP4xNVFP4 GEMM state (``forward()``'s actual path). Prepared
        # lazily on first forward -- see ``_ensure_gemm_ready``. None of this
        # is BF16-sized: ``_weight_b`` is the same packed uint8 bytes as
        # ``self.weight`` (a view, not a copy) and ``_sfb`` is the checkpoint's
        # own fp8 block scale, just swizzled into the kernel's MMA layout.
        self._gemm_ready = False
        self._weight_b: torch.Tensor | None = None
        self._sfb: torch.Tensor | None = None
        self._w_global_scale: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        """Materialize the legacy BF16-dequantized weight. Opt-in only --
        see module docstring for why ``forward()`` does not call this."""
        if self._weight_bf16 is None:
            self._weight_bf16 = dequantize_nvfp4(
                self.weight.data,
                self.weight_scale.data,
                self.weight_scale_2.data,
                group_size=self.group_size,
            )

    def _ensure_gemm_ready(self) -> None:
        """Prepare the packed weight and swizzled block scale for
        ``sparkinfer.gemm.blockscaled.mm`` -- lazy, once, no BF16
        materialization. See module/class docstrings for the scale
        convention this relies on."""
        if self._gemm_ready:
            return
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from sparkinfer._lib.intrinsics import as_grouped_scale_view, swizzle_block_scale

        # [out, in // 2] -> [out, in // 2, 1] (packed-K, batch/group-of-1
        # last, matching gemm.blockscaled.mm's (n, k, l) convention) -- a
        # view, not a copy (last dim has size 1, so no data movement).
        self._weight_b = self.weight.data.unsqueeze(-1).contiguous()

        # Swizzle the checkpoint's own fp8 block scale into the kernel's MMA
        # scale layout -- same two calls Laguna's MoE prep makes
        # (swizzle_block_scale then as_grouped_scale_view), with a batch/
        # group dim of 1 standing in for Laguna's per-expert dim.
        sfb_swizzled = swizzle_block_scale(self.weight_scale.data.unsqueeze(0).contiguous())
        self._sfb = as_grouped_scale_view(
            sfb_swizzled.view(torch.uint8), self.output_size, self.input_size
        )

        w_scale_2 = self.weight_scale_2.data.reshape(()).to(torch.float32)
        if w_scale_2.item() == 0.0:
            raise ValueError(
                f"{self.__class__.__name__}: weight_scale_2 is 0 -- cannot invert "
                "into a global_scale for the block-scaled GEMM"
            )
        self._w_global_scale = 1.0 / w_scale_2
        self._gemm_ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(
                f"{self.__class__.__name__}.forward expects bf16 activations "
                f"(dynamically quantized to NVFP4 per call), got {x.dtype}"
            )
        self._ensure_gemm_ready()
        from sparkinfer._lib.intrinsics import (
            FLOAT4_E2M1_MAX,
            FLOAT8_E4M3_MAX,
            quantize_grouped_nvfp4_torch,
        )
        from sparkinfer.gemm import blockscaled

        orig_shape = x.shape
        x2d = x.reshape(-1, self.input_size).contiguous()
        m = x2d.shape[0]

        # Dynamic per-forward activation scale (no checkpoint activation
        # scale exists for W4A16 -- see module docstring): map this call's
        # amax to the top of the FP4 representable range, the same
        # convention ``tests/gemm/test_blockscaled.py``'s NVFP4 fixtures use.
        amax = x2d.detach().abs().amax().to(torch.float32)
        amax = torch.where(amax > 0, amax, amax.new_ones(()))
        a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax

        row_counts = torch.tensor([m], dtype=torch.int32, device=x2d.device)
        a_packed, a_sf = quantize_grouped_nvfp4_torch(
            x2d.unsqueeze(0), row_counts, a_global_scale.reshape(1)
        )

        # alpha = 1 / (a_global_scale * w_global_scale) -- see class
        # docstring for the scale-convention derivation.
        alpha = (1.0 / (a_global_scale * self._w_global_scale)).reshape(1)

        out = blockscaled.mm(
            (a_packed, a_sf),
            (self._weight_b, self._sfb),
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            alpha=alpha,
            expected_m=m,
        )
        out = out[:, :, 0]
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.output_size)
