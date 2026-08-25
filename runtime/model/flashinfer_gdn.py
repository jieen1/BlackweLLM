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
_DECODE_GATED_DELTA_RULE = None
_DECODE_IMPORT_ERROR: BaseException | None = None
_DECODE_IMPORT_REPORTED = False


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


def load_gated_delta_rule_decode():
    """Return FlashInfer's single-token GDN kernel, or ``None``.

    The decode kernel is optional for the same reason as the extend kernel:
    CPU-only tests and installations without the SM120 FlashInfer image must
    retain the FLA implementation.  Keep this import separate from the
    extend loader because the two FlashInfer modules have independent binary
    availability and compile caches.
    """

    global _DECODE_GATED_DELTA_RULE, _DECODE_IMPORT_ERROR, _DECODE_IMPORT_REPORTED
    if _DECODE_GATED_DELTA_RULE is not None:
        return _DECODE_GATED_DELTA_RULE
    if _DECODE_IMPORT_ERROR is not None:
        return None

    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer.gdn_decode import gated_delta_rule_decode
    except BaseException as exc:  # optional dependency; caller falls back
        _DECODE_IMPORT_ERROR = exc
        if not _DECODE_IMPORT_REPORTED:
            logger.warning("FlashInfer GDN decode unavailable; using FLA: %s", exc)
            _DECODE_IMPORT_REPORTED = True
        return None

    _DECODE_GATED_DELTA_RULE = gated_delta_rule_decode
    return _DECODE_GATED_DELTA_RULE


class FlashInferGDNDecode:
    """Run one Qwen GGUF GDN decode step through FlashInfer's native kernel.

    The GGUF contract stores ``ssm_a`` as the negative decay coefficient,
    while FlashInfer exposes the usual ``A_log`` parameter and computes
    ``-exp(A_log) * softplus(a + dt_bias)`` internally.  The cached
    ``log(-ssm_a)`` conversion below is algebraically identical and keeps the
    recurrent state in the runtime's existing contiguous ``[B, H, K, V]``
    layout.  FlashInfer currently consumes BF16 q/k/v/a/b, so the adapter
    rounds only those transient inputs; the persistent state and fixed
    scalars remain F32.
    """

    def __init__(self) -> None:
        fn = load_gated_delta_rule_decode()
        if fn is None:
            raise RuntimeError("FlashInfer GDN decode is unavailable")
        self._fn = fn
        self._a_log_source: torch.Tensor | None = None
        self._a_log_argument: torch.Tensor | None = None

    def _a_log(self, decay: torch.Tensor) -> torch.Tensor:
        if self._a_log_source is not decay or self._a_log_argument is None:
            if torch.any(decay >= 0):
                raise ValueError(
                    "FlashInfer GDN decode requires GGUF negative ssm_a values"
                )
            self._a_log_argument = (-decay).log()
            self._a_log_source = decay
        return self._a_log_argument

    def run(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        beta_logits: torch.Tensor,
        dt_bias: torch.Tensor,
        decay: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(BF16 output, updated state)`` for one decode token."""

        # The current GGUF GVA mapping is expanded by the caller before this
        # boundary.  Passing H == HV avoids the decode kernel's alternate
        # grouped-head mapping, which differs from llama.cpp's GGUF tiling
        # convention even though the ordinary FlashInfer prefill path uses
        # its native H < HV form.
        output, state = self._fn(
            q=query.to(torch.bfloat16),
            k=key.to(torch.bfloat16),
            v=value.to(torch.bfloat16),
            state=recurrent_state,
            A_log=self._a_log(decay),
            a=a.to(torch.bfloat16),
            dt_bias=dt_bias,
            b=beta_logits.to(torch.bfloat16),
            use_qk_l2norm=True,
        )
        return output, state


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
