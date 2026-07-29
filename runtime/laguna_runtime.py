"""Owned forward-context boundary for Laguna graph call sites.

Laguna graph callers use this module rather than importing ``compat_vllm``
directly.  Until the legacy model layers are moved locally, those layers still
read vLLM's forward context while a CUDA graph is captured.  Keep that bridge
here so the capture contract remains explicit and removable as one unit.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


@contextmanager
def laguna_forward_context(
    attn_metadata: dict[str, Any],
    runtime_config: Any,
    *,
    slot_mapping: dict[str, Any] | None = None,
    skip_compiled: bool = False,
):
    """Scope a graph forward with the legacy model context it still consumes.

    ``BFAttention`` consumes metadata through ``bf_attn_context``.  The
    self-built model's remaining legacy MoE/compile wrappers also consult the
    vLLM forward/config globals during CUDA graph capture; omitting them
    changes the captured execution path and regresses replay throughput.

    This is a migration bridge, not a second call-site dependency: graph
    modules depend only on this owned boundary.  Batch F removes the legacy
    readers, at which point this context becomes a local implementation.
    """
    from runtime.compat_vllm import set_current_vllm_config, set_forward_context

    with set_current_vllm_config(runtime_config):
        with set_forward_context(
            attn_metadata,
            runtime_config,
            slot_mapping=slot_mapping,
            skip_compiled=skip_compiled,
        ):
            yield
