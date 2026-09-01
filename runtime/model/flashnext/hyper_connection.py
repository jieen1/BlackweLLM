"""Gated-residual hyper-connection for Flash-Next (qwen4_exp).

Formulas pinned from the reference implementation (sglang
``layers/hyperconnection.py`` ``GatedResidual``, read 2026-08-27):

* stream layout: ``[T, hc*hs]`` -- ``hc`` residual branches of ``hs``;
* ``GroupedGemmaRMSNorm``: per-group (group = one branch, ``hs``) RMSNorm
  with Gemma-style ``(1 + weight)`` scaling, fp32 math, cast back;
* ``mix(x)``: norm x per-branch; gate = sigmoid(up(silu(down(normed)/hc)));
  output = mean over branches of gate*normed; residual carries BOTH the raw
  and normed inputs;
* ``combine(block_out, (raw, normed))``: inject = 2*sigmoid(linear(normed)
  /hc) per branch; out = raw branches + block_out*inject, flattened.

The Triton small-M kernels remain experimental; production uses the canonical
PyTorch path until their reduction order is proven equivalent on this model.
"""

from __future__ import annotations

import torch
from torch import nn

from runtime.model.flashnext.hc_kernels import (
    grouped_gemma_rmsnorm_apply_fused,
    grouped_gemma_rmsnorm_fused,
    hc_combine_fused,
    hc_fusion_supported,
    hc_mix_fused,
    hc_norm_apply_fusion_supported,
    hc_norm_fusion_supported,
    hc_pointwise_fusion_supported,
    mix_finish_exact_fused,
    scale_sigmoid_inplace_fused,
    silu_scale_inplace_fused,
)


class GroupedGemmaRMSNorm(nn.Module):
    """RMSNorm over trailing groups with Gemma ``(1 + weight)`` scaling."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, group_size: int | None = None) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.group_size = group_size if group_size is not None else hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hc_norm_fusion_supported(x):
            return grouped_gemma_rmsnorm_fused(
                x,
                self.weight,
                self.group_size,
                self.eps,
            )
        if hc_fusion_supported(x):
            return grouped_gemma_rmsnorm_fused(x, self.weight, self.group_size, self.eps)
        dtype = x.dtype
        xf = x.float().reshape(*x.shape[:-1], x.shape[-1] // self.group_size, self.group_size)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(var + self.eps)
        if hc_norm_apply_fusion_supported(x):
            return grouped_gemma_rmsnorm_apply_fused(
                x,
                scale,
                self.weight,
                self.group_size,
            )
        normed = xf * scale
        normed = normed.flatten(-2) * (1.0 + self.weight.float())
        return normed.to(dtype)


class GatedResidual(nn.Module):
    """Hyper-connection mix/combine block (per-branch norm variant)."""

    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        eps: float = 1e-6,
        *,
        use_mix: bool = True,
        use_combine: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.params_dtype = dtype
        self.hc_norm = GroupedGemmaRMSNorm(hc_count * hidden_size, eps=eps, group_size=hidden_size)
        self.use_mix = use_mix
        self.use_combine = use_combine
        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                hc_count * hidden_size, lowrank, bias=False, dtype=dtype
            )
            self.input_mix_weight_up = nn.Linear(
                lowrank, hc_count * hidden_size, bias=False, dtype=dtype
            )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                hc_count * hidden_size, hc_count, bias=False, dtype=dtype
            )

    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hc, hs = self.hc_count, self.hidden_size
        hyper_input = hyper_input.to(self.params_dtype)
        normed = self.hc_norm(hyper_input)
        if hc_fusion_supported(normed):
            mixed = hc_mix_fused(
                normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                hc,
            )
            return mixed, (hyper_input, normed)
        gate = torch.nn.functional.linear(normed, self.input_mix_weight_down.weight)
        if hc_pointwise_fusion_supported(gate):
            silu_scale_inplace_fused(gate, 1.0 / hc)
        else:
            gate = torch.nn.functional.silu(gate / hc)
        gate = torch.nn.functional.linear(gate, self.input_mix_weight_up.weight)
        if hc_pointwise_fusion_supported(gate):
            mixed = mix_finish_exact_fused(gate, normed, hc)
        else:
            gate = torch.sigmoid(gate.float()).unflatten(-1, (hc, hs)).to(hyper_input.dtype)
            mixed = (gate * normed.unflatten(-1, (hc, hs))).mean(dim=-2)
        return mixed, (hyper_input, normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hyper_input, normed = residuals
        hc, hs = self.hc_count, self.hidden_size
        block_output = block_output.to(self.params_dtype)
        if hc_fusion_supported(block_output):
            return hc_combine_fused(
                block_output,
                hyper_input,
                normed,
                self.block_inject_weight.weight,
                hc,
            )
        inject = torch.nn.functional.linear(normed, self.block_inject_weight.weight)
        if hc_pointwise_fusion_supported(inject):
            scale_sigmoid_inplace_fused(inject, 1.0 / hc, 2.0)
        else:
            inject = 2.0 * torch.sigmoid((inject / hc).float()).to(hyper_input.dtype)
        r = hyper_input.unflatten(-1, (hc, hs))
        return (r + block_output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
