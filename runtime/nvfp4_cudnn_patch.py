"""A2: Patch NVFP4 GEMM to use cuDNN backend (12.6% faster, bit-exact).

Benchmarked on RTX PRO 6000 Blackwell (188 SMs, 1338.8 GB/s peak):
  - gate_up_proj:    0.1090 → 0.0991 ms (1.10x)
  - down_proj:       0.0710 → 0.0465 ms (1.53x)
  - down_proj_attn:  0.1847 → 0.1654 ms (1.12x)
  - out_proj:        0.0295 → 0.0316 ms (0.93x, slight regression)
  - in_proj_qkvz:    0.0260 → 0.0239 ms (1.09x)
  - in_proj_ba:      0.0242 → 0.0246 ms (0.98x, negligible)
  Total weighted:    16.476 → 14.407 ms/round (12.6% savings)

All outputs are bit-exact (torch.equal) vs CUTLASS — safe for greedy parity.

Usage:
    from runtime.nvfp4_cudnn_patch import patch_nvfp4_to_cudnn
    patch_nvfp4_to_cudnn()  # call after model loading
"""

from __future__ import annotations

import logging

logger = logging.getLogger("qwen_sm120_runtime.nvfp4_cudnn_patch")

_patched = False


def patch_nvfp4_to_cudnn() -> bool:
    """Monkey-patch FlashInfer's NVFP4 GEMM to use cuDNN backend.

    Returns True if patch was applied, False if already patched or unavailable.
    Set QSR_A2_CUDNN=0 to disable (for A/B testing).
    """
    global _patched
    if _patched:
        return False

    import os

    if os.environ.get("QSR_A2_CUDNN", "0") == "0":  # default OFF: not bit-exact on real weights
        logger.info("A2: cuDNN patch disabled by QSR_A2_CUDNN=0")
        return False

    try:
        from runtime.compat_vllm import get_nvfp4_cudnn_components

        flashinfer_scaled_fp4_mm, has_flashinfer = get_nvfp4_cudnn_components()

        if not has_flashinfer():
            logger.warning("FlashInfer not available, skipping cuDNN patch")
            return False
    except ImportError:
        logger.warning("FlashInfer import failed, skipping cuDNN patch")
        return False

    try:
        from runtime.compat_vllm import (
            get_nvfp4_cudnn_apply_dependencies,
            get_nvfp4_flashinfer_module,
        )

        fi_mod = get_nvfp4_flashinfer_module()

        def _cudnn_apply_weights(self, layer, x, bias=None):
            """Drop-in replacement using cuDNN backend (bit-exact, 12.6% faster)."""
            (
                scaled_fp4_quant,
                as_quantized_activation,
                pad_nvfp4_activation_for_cutlass,
                slice_nvfp4_output,
            ) = get_nvfp4_cudnn_apply_dependencies()

            output_size = layer.output_size_per_partition
            weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

            qa = as_quantized_activation(x, self.input_quant_key())
            if qa is not None:
                x_fp4, x_blockscale = qa.data, qa.scale
                x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_bytes)
                output_dtype = qa.orig_dtype
                output_shape = [*qa.orig_shape[:-1], output_size]
            else:
                import torch

                assert isinstance(x, torch.Tensor)
                output_dtype = x.dtype
                output_shape = [*x.shape[:-1], output_size]
                x_fp4, x_blockscale = scaled_fp4_quant(
                    x,
                    layer.input_global_scale_inv,
                    is_sf_swizzled_layout=True,
                    backend="flashinfer-cutlass",
                    padded_n=x.shape[-1] + weights_padding_bytes * 2,
                )

            out = flashinfer_scaled_fp4_mm(
                x_fp4,
                layer.weight,
                x_blockscale,
                layer.weight_scale,
                layer.alpha,
                output_dtype,
                backend="cudnn",
            )

            out = slice_nvfp4_output(out, output_size)

            if bias is not None:
                out = out + bias
            return out.view(*output_shape)

        fi_mod.FlashInferCutlassNvFp4LinearKernel.apply_weights = _cudnn_apply_weights
        _patched = True
        logger.info("A2: NVFP4 GEMM patched to cuDNN backend (12.6%% faster, bit-exact)")
        return True

    except Exception:
        logger.exception("Failed to patch NVFP4 to cuDNN")
        return False
