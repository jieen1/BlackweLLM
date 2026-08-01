"""Self-built ``default_weight_loader`` -- 任务#41 (vLLM removal plan
阶段8). Shared by laguna_model.py and laguna_dflash_model.py (both used
vLLM's ``vllm.model_executor.model_loader.weight_utils`` version before
this).

Split into its own module (not defined in laguna_model.py, despite that
being where the usage originated) so laguna_dflash_model.py can import it
too without a circular import, same reasoning as ``_prefix.py``.

Track A step 6 (``docs/architecture.md`` §3.5.5): this module used to also
hold ``remap_kv_scale_name``, compressed-tensors' own KV-scale checkpoint-
key naming convention. That moved to
``runtime/loading/compressed_tensors.py`` (body unchanged) -- it is
quantization-format-specific knowledge, unlike ``default_weight_loader``
below, which is a plain shape-matched tensor copy with nothing format- or
model-specific about it and stays here.
"""

from __future__ import annotations

import torch


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Verbatim port of vLLM's ``default_weight_loader`` (vllm/model_
    executor/model_loader/weight_utils.py) -- plain shape-matched copy,
    nothing vLLM-specific about it."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        # Scalar values aren't always considered tensors with shapes, so
        # if both are scalars, reshape to match before copying.
        param.data.copy_(loaded_weight.view(param.shape))
    else:
        assert param.size() == loaded_weight.size(), (
            f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})"
        )
        param.data.copy_(loaded_weight)
