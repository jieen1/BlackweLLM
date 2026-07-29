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
    self-built Laguna graph has no remaining vLLM layer that reads a global
    forward context, so this scope exists only to keep eager and graph call
    sites on one explicit runtime boundary.
    """
    del attn_metadata, runtime_config, slot_mapping, skip_compiled
    yield
