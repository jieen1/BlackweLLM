"""Self-built ``default_weight_loader``/kv-scale name remapping -- 任务#41
(vLLM removal plan 阶段8). Shared by laguna_model.py and
laguna_dflash_model.py (both used vLLM's ``vllm.model_executor.
model_loader.weight_utils`` versions before this).

Split into its own module (not defined in laguna_model.py, despite that
being where the usage originated) so laguna_dflash_model.py can import it
too without a circular import, same reasoning as ``_prefix.py``.
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
            f"Attempted to load weight ({loaded_weight.size()}) "
            f"into parameter ({param.size()})"
        )
        param.data.copy_(loaded_weight)


def remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Narrowed replacement for vLLM's ``maybe_remap_kv_scale_name``
    (vllm/model_executor/model_loader/weight_utils.py) -- only the one
    real pattern this runtime's checkpoints ever produce: a checkpoint
    key ending directly in ``.k_scale``/``.v_scale`` (e.g. ``model.
    layers.N.self_attn.k_scale``, verified directly against the real
    checkpoint's safetensors), remapped to ``...self_attn.attn.k_scale``
    to match ``SelfBuiltAttentionPlaceholder``'s ``self.attn`` submodule
    nesting (runtime/model/plain_attention.py).

    Every other real pattern vLLM's version covers -- the deprecated
    ``.kv_scale`` format, ModelOpt/QKV-proj/Qwen3-MoE/NemotronH/HYV3
    checkpoint naming conventions, ``q_scale``/zero-point suffixes, MLA's
    ``mla_attn.mla_attn`` prefix -- is provably unreachable for this
    checkpoint (verified directly, not assumed: only ``self_attn.
    {k,v}_scale`` exist per layer, no ``_proj``/``qkv_proj``/etc in
    between) and intentionally not ported. If a future checkpoint needs
    one of those, this needs revisiting, not generalizing in advance --
    same "documented checkpoint-specific assumption, fail loud if wrong"
    stance as model_loading.py's ``_assert_all_params_loaded``. The DFlash
    draft model's checkpoint never has any ``k_scale``/``v_scale`` keys
    at all (verified directly), so this function never even gets called
    with a matching suffix for it -- the ``name in params_dict`` shortcut
    (or plain pass-through) handles every draft-model key.
    """
    if name in params_dict:
        return name
    if name.endswith(".k_scale") or name.endswith(".v_scale"):
        prefix, _, suffix = name.rpartition(".")
        remapped = f"{prefix}.attn.{suffix}"
        return remapped if remapped in params_dict else None
    return name
