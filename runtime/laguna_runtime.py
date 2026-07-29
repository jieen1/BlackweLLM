"""Owned forward-context bridge for graph/runtime call sites.

This module keeps Laguna graph code off ``runtime.compat_vllm`` while the
loaded model stack still reads vLLM's thread-local forward context during
``model.forward()``. Imports are resolved lazily so source-only tests can
import graph modules without requiring a local vLLM checkout.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from typing import Any


@contextmanager
def laguna_forward_context(
    attn_metadata: dict[str, Any],
    runtime_config: Any,
    *,
    slot_mapping: dict[str, Any] | None = None,
    skip_compiled: bool = False,
):
    """Set the model forward context expected by vLLM-loaded modules.

    The graph call sites only need the small subset used during local model
    forward: attention metadata, slot mapping, compile-skip flag, and the
    static layer registries carried on ``runtime_config``.

    Also sets ``vllm.config._current_vllm_config`` because model layers
    (quantization kernels, attention backends) read it via
    ``get_current_vllm_config()``.  Omitting this causes stale config
    reads → incorrect logits → DFlash acceptance collapse.
    """

    vllm_fc = importlib.import_module("vllm.forward_context")
    forward_context_cls = getattr(vllm_fc, "ForwardContext")
    vllm_config_mod = importlib.import_module("vllm.config")
    cudagraph_mode = getattr(vllm_config_mod, "CUDAGraphMode")

    forward_context = forward_context_cls(
        no_compile_layers=runtime_config.compilation_config.static_forward_context,
        all_moe_layers=getattr(runtime_config.compilation_config, "static_all_moe_layers", None),
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=None,
        cudagraph_runtime_mode=cudagraph_mode.NONE,
        batch_descriptor=None,
        ubatch_slices=None,
        skip_compiled=skip_compiled,
        additional_kwargs={},
        is_padding=None,
    )
    previous_fc = getattr(vllm_fc, "_forward_context")
    vllm_fc._forward_context = forward_context

    # Set current VllmConfig (model layers read via get_current_vllm_config)
    set_current = getattr(vllm_config_mod, "set_current_vllm_config", None)
    if set_current is not None:
        config_cm = set_current(runtime_config)
        config_cm.__enter__()
    else:
        config_cm = None

    try:
        yield
    finally:
        if config_cm is not None:
            config_cm.__exit__(None, None, None)
        vllm_fc._forward_context = previous_fc
