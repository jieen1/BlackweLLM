"""Self-built NVFP4-quantized Linear layer — Phase 2 of the vLLM removal plan.

See notes/2026-07-27-vllm-complete-removal-implementation-plan.md, 阶段2.
Replaces vLLM's ``ColumnParallelLinear``/``QKVParallelLinear``/
``RowParallelLinear`` + ``CompressedTensorsW4A4Fp4`` scheme +
``CutlassNvFp4LinearKernel`` stack for the one case this runtime actually
needs: TP=1, NVFP4 W4A4 (weight and activation both quantized), no LoRA.

Traced against vLLM's real implementation (not guessed):
- Parameter shapes/dtypes/names: vllm/model_executor/layers/quantization/
  compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
  (``CompressedTensorsW4A4Fp4.create_weights``/``process_weights_after_loading``).
- Shard placement semantics (QKV-style fused output-dim stacking):
  vllm/model_executor/parameter.py (``_ColumnvLLMParameter.load_qkv_weight``,
  ``PerTensorScaleParameter._load_into_shard_id``) -- reimplemented directly
  here rather than through vLLM's generic Parameter-subclass dispatch
  machinery, since TP=1 removes the need for tp_rank/tp_size narrowing and
  this runtime only ever has one caller (LagunaModelSelfBuilt.load_weights).
- Weight-side NVFP4 preprocessing (``swizzle_blockscale``,
  ``pad_nvfp4_weight_for_cutlass``): vllm/model_executor/layers/
  quantization/utils/nvfp4_utils.py -- copied verbatim below (pure tensor
  ops, no vLLM object dependency, zero risk per the implementation plan).
- GEMM: runtime.nvfp4_custom_gemm.custom_scaled_fp4_mm (already self-built,
  called directly here instead of through vLLM's monkey-patched
  cutlass_scaled_fp4_mm reference).
- Activation-side NVFP4 quantization: NOT YET self-built. Currently calls
  vLLM's compiled ``torch.ops._C.scaled_fp4_quant`` extension directly (the
  compiled CUDA op, not vLLM Python code) as an interim measure -- porting
  this ~430-line kernel (csrc/libtorch_stable/quantization/fp4/
  nvfp4_quant_kernels.cu) is explicitly scoped as its own follow-up step,
  not done in this pass. See _quantize_activation() below for the isolated
  seam where that swap happens.

Known, deliberate limitation carried over unchanged from vLLM (documented
in the plan, do not "fix" this): when a checkpoint stores per-logical-shard
global scales that differ (e.g. across q_proj/k_proj/v_proj if it happens
to have per-Linear-original scales), this class collapses them via .max()
before use, same as vLLM's CompressedTensorsW4A4Fp4.process_weights_after_
loading -- a real, checkpoint-format-inherent precision trade-off, not a
bug introduced here. Laguna's gate_proj/up_proj stay unstacked specifically
to avoid ever hitting this path for those two projections; QKV is still
fused (matches vLLM's own current behavior) and can still hit it if the
checkpoint's per-head scales genuinely differ.
"""

from __future__ import annotations

import torch
from torch import nn

GROUP_SIZE = 16  # NVFP4 block-scale group size (compressed-tensors W4A4 default).


def _round_up(x: int, to: int) -> int:
    return ((x + to - 1) // to) * to


# ---------------------------------------------------------------------------
# Weight-side NVFP4 preprocessing — copied verbatim from vLLM's
# vllm/model_executor/layers/quantization/utils/nvfp4_utils.py. Pure tensor
# ops (padding/reshape/permute), zero vLLM object dependency. Per the
# implementation plan this is "directly portable, zero risk" -- do not
# modify the math, only the import path.
# ---------------------------------------------------------------------------


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """Pad and block-interleave FP4 block-scales into the CUTLASS kernel layout."""
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects torch.float8_e4m3fn input."
    )
    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape
    M_padded = _round_up(M, 128)
    K_padded = _round_up(K, 4)

    padded = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype, device=scale.device)
    padded[:B, :M, :K] = scale

    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def pad_nvfp4_weight_for_cutlass(
    weight: torch.Tensor, alignment: int = 32
) -> tuple[torch.Tensor, int]:
    """Pad packed NVFP4 weights so N and K satisfy CUTLASS/FlashInfer alignment."""
    weight_current_rows = weight.shape[0]

    if weight_current_rows % alignment != 0:
        total_rows = _round_up(weight_current_rows, alignment)
        pad_rows = total_rows - weight_current_rows
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()

    weight_current_col_bytes = weight.shape[1]
    weight_current_col_elements = weight_current_col_bytes * 2

    weights_padding_bytes = 0
    if weight_current_col_elements % alignment != 0:
        total_cols = _round_up(weight_current_col_elements, alignment)
        pad_cols = total_cols - weight_current_col_elements
        pad_bytes = pad_cols // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
        weights_padding_bytes = pad_bytes

    return weight, weights_padding_bytes


def slice_nvfp4_output(out: torch.Tensor, output_size: int) -> torch.Tensor:
    """Slice off N-dimension padding added by pad_nvfp4_weight_for_cutlass."""
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


def _quantize_activation(
    x: torch.Tensor, input_global_scale_inv: torch.Tensor, *, padded_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Activation-side NVFP4 quantization.

    INTERIM: calls vLLM's compiled torch.ops._C.scaled_fp4_quant extension
    directly (the CUDA op, not vLLM Python code). This is the one remaining
    seam that still touches a vLLM-shipped compiled artifact -- porting it
    to a self-built kernel is explicitly out of scope for this pass, see
    module docstring. Isolated in its own function so that swap is a
    one-function change, not a refactor.
    """
    return torch.ops._C.scaled_fp4_quant(
        x,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="cutlass",
        padded_n=padded_n,
    )


class NvFp4Linear(nn.Module):
    """NVFP4 W4A4 linear layer, TP=1, with optional output-dim shard fusion.

    ``shard_sizes=None`` (default): one logical weight matrix (o_proj,
    down_proj, gate_proj, up_proj -- Laguna keeps the latter two unstacked
    specifically to avoid the global-scale merge precision loss described
    in the module docstring).

    ``shard_sizes=[q_size, k_size, v_size]``: fused QKV-style layer. Load
    each shard via ``load_shard(..., shard_idx=0|1|2)`` -- caller maps
    checkpoint "q"/"k"/"v" shard ids to indices before calling in
    (LagunaModelSelfBuilt.load_weights owns that mapping, same place
    vLLM's stacked_params_mapping lived).
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
        assert sum(self.shard_sizes) == output_size, (
            f"shard_sizes {self.shard_sizes} must sum to output_size {output_size}"
        )
        self.num_shards = len(self.shard_sizes)
        offsets = []
        running = 0
        for s in self.shard_sizes:
            offsets.append(running)
            running += s
        self.shard_offsets = offsets

        assert input_size % 2 == 0, "NVFP4 packs 2 elements/byte; input_size must be even."
        assert input_size % GROUP_SIZE == 0, (
            f"input_size {input_size} must be divisible by NVFP4 group_size {GROUP_SIZE}"
        )

        self.weight_packed = nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_global_scale = nn.Parameter(
            torch.empty(self.num_shards, dtype=torch.float32), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // GROUP_SIZE, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.input_global_scale = nn.Parameter(
            torch.empty(self.num_shards, dtype=torch.float32), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

        self._processed = False

    # -- Weight loading -----------------------------------------------------

    def load_shard(self, param_name: str, loaded_weight: torch.Tensor, shard_idx: int = 0) -> None:
        """Copy one checkpoint tensor into the right slice of a physical param.

        ``param_name`` in {"weight_packed", "weight_scale",
        "weight_global_scale", "input_global_scale"}. Row-stacked params
        (weight_packed/weight_scale) get an output-dim narrow+copy at this
        shard's offset; per-shard scalar params
        (weight_global_scale/input_global_scale) get a single-index write.
        Mirrors vLLM's _ColumnvLLMParameter.load_qkv_weight /
        PerTensorScaleParameter._load_into_shard_id semantics minus TP
        rank/size narrowing (this runtime is TP=1 -- there is no other
        rank's slice to skip).
        """
        assert not self._processed, (
            "load_shard called after process_weights_after_loading -- "
            "weight layout has already been swizzled/padded for the GEMM "
            "kernel and is no longer in checkpoint layout."
        )
        if param_name in ("weight_packed", "weight_scale"):
            param = getattr(self, param_name)
            offset = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]
            dst = param.data.narrow(0, offset, size)
            assert dst.shape == loaded_weight.shape, (
                f"{param_name} shard {shard_idx}: dst {tuple(dst.shape)} vs "
                f"loaded {tuple(loaded_weight.shape)}"
            )
            dst.copy_(loaded_weight)
        elif param_name in ("weight_global_scale", "input_global_scale"):
            param = getattr(self, param_name)
            value = loaded_weight
            if value.dim() > 0:
                assert value.numel() == 1, (
                    f"{param_name} shard {shard_idx}: expected a scalar, got shape "
                    f"{tuple(value.shape)}"
                )
                value = value.reshape(())
            param.data[shard_idx].copy_(value)
        elif param_name == "bias":
            assert self.bias is not None
            offset = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]
            self.bias.data.narrow(0, offset, size).copy_(loaded_weight)
        else:
            raise ValueError(f"NvFp4Linear.load_shard: unknown param_name {param_name!r}")

    # -- Post-load processing -------------------------------------------------

    def process_weights_after_loading(self) -> None:
        """NVFP4 scale/layout finalization. Ports vLLM's two-stage pipeline:

        1. CompressedTensorsW4A4Fp4.process_weights_after_loading (global
           scale merge-and-invert, alpha precompute).
        2. CutlassNvFp4LinearKernel.process_weights_after_loading (block
           scale swizzle, weight padding for the CUTLASS-layout GEMM).

        Both stages ported line-for-line from the vLLM source cited in the
        module docstring (not re-derived independently) -- see there for
        the precision-loss caveat around step 1's .max() merge.
        """
        assert not self._processed, "process_weights_after_loading called twice"

        if self.num_shards > 1 and torch.unique(self.weight_global_scale.data).numel() != 1:
            import logging

            logging.getLogger(__name__).warning(
                "NvFp4Linear: weight_global_scale differs across %d fused shards "
                "(e.g. q/k/v) -- checkpoint-inherent precision loss on merge, "
                "not a bug in this code. See module docstring.",
                self.num_shards,
            )
        weight_global_scale = self.weight_global_scale.data.max().to(torch.float32)
        self.weight_global_scale = nn.Parameter(1.0 / weight_global_scale, requires_grad=False)

        if self.num_shards > 1 and torch.unique(self.input_global_scale.data).numel() != 1:
            import logging

            logging.getLogger(__name__).warning(
                "NvFp4Linear: input_global_scale differs across %d fused shards.",
                self.num_shards,
            )
        input_global_scale_inv = self.input_global_scale.data.max().to(torch.float32)
        self.input_global_scale = nn.Parameter(
            (1.0 / input_global_scale_inv).to(torch.float32), requires_grad=False
        )
        self.input_global_scale_inv = nn.Parameter(input_global_scale_inv, requires_grad=False)
        self.alpha = nn.Parameter(
            self.input_global_scale * self.weight_global_scale, requires_grad=False
        )

        self.weight_scale = nn.Parameter(
            swizzle_blockscale(self.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_bytes = pad_nvfp4_weight_for_cutlass(self.weight_packed.data)
        # Rename to match the GEMM call site below (no functional
        # difference from keeping "weight_packed" -- named "weight" only
        # because that's what forward() reads; vLLM's rename step existed
        # to bridge two abstraction layers we've collapsed into one here).
        self.weight = nn.Parameter(padded_weight, requires_grad=False)
        del self.weight_packed
        self.weights_padding_bytes = weights_padding_bytes

        self._processed = True

    # -- Forward --------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._processed, "NvFp4Linear.forward called before process_weights_after_loading"
        from runtime.nvfp4_custom_gemm import custom_scaled_fp4_mm

        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], self.output_size]

        x_fp4, x_blockscale = _quantize_activation(
            x,
            self.input_global_scale_inv,
            padded_n=x.shape[-1] + self.weights_padding_bytes * 2,
        )

        out = custom_scaled_fp4_mm(
            x_fp4, self.weight, x_blockscale, self.weight_scale, self.alpha, output_dtype
        )
        out = slice_nvfp4_output(out, self.output_size)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*output_shape)
