"""Owned forward-context boundary for Laguna graph call sites."""

from __future__ import annotations

import socket
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LagunaAttentionMetadata:
    """Owned attention metadata consumed by the SparkInfer adapter.

    The Laguna attention path only reads these fields while converting the
    request layout into :class:`SparkinferAttnMetadata`.  Keeping this small
    value object local prevents a historical FlashInfer/vLLM data class from
    becoming a startup dependency for an otherwise self-built model.
    """

    query_start_loc: Any
    query_start_loc_cpu: Any
    seq_lens: Any
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: Any
    slot_mapping: Any
    causal: bool


def get_open_port() -> int:
    """Return an available loopback TCP port for a one-rank process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def get_distributed_init_method(ip: str, port: int) -> str:
    """Build a torch.distributed TCP init URI."""
    if ":" in ip:
        return f"tcp://[{ip}]:{port}"
    return f"tcp://{ip}:{port}"


def _extract_attention_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """Extract the deterministic attention-cache order from a layer name."""
    indices: list[int] = []
    for part in layer_name.split("."):
        try:
            indices.append(int(part))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(indices) == 1, f"layer name {layer_name} should contain one integer"
        return indices[0]
    assert len(indices) <= 2, f"layer name {layer_name} should contain at most two integers"
    return indices[0] * num_attn_module + indices[1] if len(indices) == 2 else indices[0]


def bind_laguna_kv_cache(
    kv_caches: dict[str, Any],
    attention_layers: dict[str, Any],
    runner_kv_caches: list[Any],
    num_attn_module: int = 1,
) -> None:
    """Bind self-allocated KV caches in stable layer order.

    This is intentionally the complete contract needed by Laguna's
    self-built attention placeholders: populate each layer's ``kv_cache``
    and retain a stable runner list.  No vLLM runtime state participates.
    """
    assert not runner_kv_caches
    index_to_names: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index_to_names[_extract_attention_layer_index(layer_name, num_attn_module)].append(
            layer_name
        )
    for layer_index in sorted(index_to_names):
        for layer_name in index_to_names[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])
    for layer_name, kv_cache in kv_caches.items():
        attention_layers[layer_name].kv_cache = kv_cache


@contextmanager
def laguna_forward_context(
    attn_metadata: dict[str, Any],
    runtime_config: Any,
    *,
    slot_mapping: dict[str, Any] | None = None,
    skip_compiled: bool = False,
):
    """Scope an owned graph forward.

    ``BFAttention`` consumes metadata through ``bf_attn_context``. The
    self-built Laguna graph has no vLLM global-state reader.
    """
    del attn_metadata, runtime_config, slot_mapping, skip_compiled
    yield
