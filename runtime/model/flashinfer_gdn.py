"""Optional FlashInfer GDN extend adapter for SM120.

SGLang's Qwen hybrid path dispatches GDN prefill/extend to FlashInfer's
CuTeDSL chunk kernel.  The local model keeps the recurrent pool in the
historical ``[batch, heads, K, V]`` BF16 layout because that is also the
layout consumed by the recurrent decode and rollback paths.  FlashInfer's
SM120 kernel uses ``[batch, heads, V, K]`` FP32 state, so this adapter owns a
small persistent bridge buffer per GDN layer and copies only the state
boundary around the fused extend call.

The import is deliberately lazy.  CPU tests and installations without the
optional FlashInfer package continue to use the existing FLA implementation.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("qwen_sm120_runtime.flashinfer_gdn")

_CHUNK_GATED_DELTA_RULE = None
_IMPORT_ERROR: BaseException | None = None
_IMPORT_REPORTED = False


def load_chunk_gated_delta_rule():
    """Return FlashInfer's GDN prefill function, or ``None`` if unavailable."""

    global _CHUNK_GATED_DELTA_RULE, _IMPORT_ERROR, _IMPORT_REPORTED
    if _CHUNK_GATED_DELTA_RULE is not None:
        return _CHUNK_GATED_DELTA_RULE
    if _IMPORT_ERROR is not None:
        return None

    # The validated local image has a Python/cubin patch-version mismatch.  The
    # cubin loader is still the same SM120 build used by the benchmark; match
    # the policy of the existing FlashInfer paged-prefill adapter.
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    except BaseException as exc:  # optional dependency; caller falls back
        _IMPORT_ERROR = exc
        if not _IMPORT_REPORTED:
            logger.warning("FlashInfer GDN prefill unavailable; using FLA: %s", exc)
            _IMPORT_REPORTED = True
        return None

    _CHUNK_GATED_DELTA_RULE = chunk_gated_delta_rule
    return _CHUNK_GATED_DELTA_RULE


class FlashInferGDNPrefill:
    """Run one Qwen GDN extend with FlashInfer and bridge its state format."""

    def __init__(self) -> None:
        fn = load_chunk_gated_delta_rule()
        if fn is None:
            raise RuntimeError("FlashInfer GDN prefill is unavailable")
        self._fn = fn
        self._state_workspace: torch.Tensor | None = None
        self._state_shape: tuple[int, int, int, int] | None = None

    def _workspace(
        self,
        recurrent_state: torch.Tensor,
        *,
        num_value_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> torch.Tensor:
        shape = (
            recurrent_state.shape[0],
            num_value_heads,
            head_v_dim,
            head_k_dim,
        )
        if (
            self._state_workspace is None
            or self._state_shape != shape
            or self._state_workspace.device != recurrent_state.device
        ):
            self._state_workspace = torch.empty(
                shape,
                device=recurrent_state.device,
                dtype=torch.float32,
            )
            self._state_shape = shape
        # The runtime pool is [H, K, V]; FlashInfer is [H, V, K].  Keep the
        # conversion explicit so state ownership remains with the caller's
        # fixed pool and no Python reference is rebound during graph work.
        self._state_workspace.copy_(recurrent_state.transpose(-1, -2))
        return self._state_workspace

    def run(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_value_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> torch.Tensor:
        """Return ``[batch, tokens, value_heads, value_dim]`` output.

        ``query``/``key`` retain their native key-head count.  FlashInfer's
        SM120 kernel handles the 16->48 GVA mapping directly; repeating them
        to 48 heads would both waste bandwidth and miss SGLang's dispatch
        contract.
        """

        batch_size, tokens, num_key_heads, _ = query.shape
        state = self._workspace(
            recurrent_state,
            num_value_heads=num_value_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
        )
        query_flat = query.reshape(-1, num_key_heads, head_k_dim).contiguous()
        key_flat = key.reshape(-1, num_key_heads, head_k_dim).contiguous()
        value_flat = value.reshape(-1, num_value_heads, head_v_dim).contiguous()
        log_decay_flat = log_decay.reshape(-1, num_value_heads).contiguous()
        beta_flat = beta.reshape(-1, num_value_heads).contiguous()

        output, _ = self._fn(
            q=query_flat,
            k=key_flat,
            v=value_flat,
            g=torch.exp(log_decay_flat),
            beta=beta_flat,
            initial_state=state,
            output_final_state=True,
            output_state=state,
            cu_seqlens=cu_seqlens,
            # q/k are normalized by the caller, matching SGLang's
            # FlashInferGDNKernel.extend contract.
            use_qk_l2norm_in_kernel=False,
        )
        recurrent_state.copy_(state.transpose(-1, -2))
        return output.reshape(batch_size, tokens, num_value_heads, head_v_dim)
