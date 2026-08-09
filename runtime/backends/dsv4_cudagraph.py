"""DSV4 decode CUDA-Graph driver (Phase 4 item 3).

The kernel-path decode step currently rebuilds every per-layer attention
binding + indices from scratch each token (~20 torch launches x 43
layers = ~170 ms/step of launch + allocation overhead).  A captured
graph collapses the whole decode step into one replay.

The capture-safe contract (mirrors qwen36's ``Qwen36DecodeGraphAttention``):
every tensor the decode forward reads or writes is pre-allocated ONCE and
only its *contents* change between replays.  The fork ``compressed_mla.run``
takes a ``binding`` that pins ``q``/``swa_indices``/``swa_lengths``/
``indexed_indices``/``indexed_lengths`` (all views -- capture safe per the
module docstring) and an optional ``out=`` for direct write, so the graph
can replay with identical addresses every step.

Driver lifecycle:
  - ``prepare(slot)``: pre-allocate every per-layer buffer, warm the
    kernels eagerly (a JIT compile inside capture is not capturable),
    then ``torch.cuda.graph`` the full 43-layer decode.
  - ``replay(slot, token, pos)``: write the token/position into the
    pinned buffers (contents only), replay, return logits.

Not every op in the current forward is buffer-driven yet; the driver
starts with the attention kernel path (the dominant cost) and the
forward restructures the surrounding HC/MoE to reuse fixed buffers.
"""

from __future__ import annotations

import torch

from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer


class Dsv4DecodeGraphDriver:
    """Pre-allocated, capture-safe decode driver for one slot."""

    def __init__(
        self,
        *,
        model,
        kernel_layers: list[Dsv4AttnKernelLayer],
        max_seq_len: int,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.kernel_layers = kernel_layers
        self.max_seq_len = max_seq_len
        self.device = device
        self.hidden = model.config.hidden_size
        self.hc_mult = model.config.hc_mult
        self.graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool: torch.cuda.graph_pool_handle | None = None
        self._pinned: dict = {}

    def _prepare_buffers(self) -> dict:
        """Pre-allocate every per-step buffer (contents mutated per replay)."""
        dev = self.device
        hidden, hcm = self.hidden, self.hc_mult
        config = self.model.config
        d = {
            "input_ids": torch.empty((1,), dtype=torch.long, device=dev),
            # h stream: [1, 1, hcm, hidden] stays fixed; per-layer x reused.
            "h": torch.empty((1, 1, hcm, hidden), dtype=torch.bfloat16, device=dev),
            "x": torch.empty((1, 1, hidden), dtype=torch.bfloat16, device=dev),
            "logits": torch.empty((1, config.vocab_size), dtype=torch.float32, device=dev),
            # per-layer attention pinned buffers
            "attn_q": torch.empty(
                (1, config.num_heads, config.head_dim), dtype=torch.bfloat16, device=dev
            ),
            "swa_indices": torch.empty((1, config.window_size), dtype=torch.int32, device=dev),
            "swa_lengths": torch.empty((1,), dtype=torch.int32, device=dev),
            "indexed_indices": torch.empty((1, config.index_topk), dtype=torch.int32, device=dev),
            "indexed_lengths": torch.empty((1,), dtype=torch.int32, device=dev),
        }
        return d

    def capture(self, slot: int) -> None:
        """Warm kernels eagerly, then capture the decode step into a graph."""
        raise NotImplementedError(
            "capture() is implemented incrementally; the decode forward is "
            "being restructured to reuse these pinned buffers."
        )
