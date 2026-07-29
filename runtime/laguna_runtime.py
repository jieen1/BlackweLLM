"""Owned forward-context boundary for Laguna graph call sites."""

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
    """Scope an owned graph forward.

    ``BFAttention`` consumes metadata through ``bf_attn_context``.  The
    ``BFAttention`` consumes metadata through ``bf_attn_context``. The
    self-built Laguna graph has no remaining vLLM global-state reader.
    """
    del attn_metadata, runtime_config, slot_mapping, skip_compiled
    yield
