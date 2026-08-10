"""DSV4 decode CUDA-Graph driver (Phase 4 item 3).

The kernel-path decode step currently rebuilds every per-layer attention
binding + indices from scratch each token (~20 torch launches x 43
layers = ~170 ms/step of launch + allocation overhead).  A captured
graph collapses the whole decode step into one replay.

The capture-safe contract (mirrors qwen36's ``Qwen36DecodeGraphAttention``):
the token and position inputs are persistent tensors whose contents change
between replays; intermediate allocations belong to the CUDA graph pool and
therefore keep the addresses captured by ``compressed_mla.run``.  Recursive
compressor/indexer state and packed KV pages are slot-owned persistent
buffers, never replaced after capture.

Driver lifecycle:
  - ``capture(slot)``: warm the kernels eagerly (a JIT compile inside
    capture is not capturable), then graph the full 43-layer decode.
  - ``replay(slot, token, pos)``: write the token/position into the
    pinned buffers (contents only), replay, return logits.

Not every op in the current forward is buffer-driven yet; the driver
starts with the attention kernel path (the dominant cost) and the
forward restructures the surrounding HC/MoE to reuse fixed buffers.
"""

from __future__ import annotations

from typing import Any

import torch

from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer
from runtime.model.dsv4_model import rms_norm

_INDEX_ENTRY_BUCKETS: tuple[int, ...] = (512, 1024, 4096, 16384, 32768)


def _max_ratio4_entries(kernel_layers: list[Dsv4AttnKernelLayer]) -> int:
    caps = [
        int(layer.indexer.kv_cache.shape[1])
        for layer in kernel_layers
        if getattr(layer, "indexer", None) is not None and layer.indexer.kv_cache is not None
    ]
    return max(caps, default=0)


def _index_entry_buckets(kernel_layers: list[Dsv4AttnKernelLayer]) -> tuple[int, ...]:
    max_entries = _max_ratio4_entries(kernel_layers)
    if max_entries == 0:
        return ()
    buckets = [min(cap, max_entries) for cap in _INDEX_ENTRY_BUCKETS if cap < max_entries]
    buckets.append(max_entries)
    return tuple(dict.fromkeys(buckets))


class Dsv4DecodeGraphDriver:
    """Pre-allocated, capture-safe decode driver for one slot."""

    def __init__(
        self,
        *,
        model,
        kernel_layers: list[Dsv4AttnKernelLayer],
        max_seq_len: int,
        max_index_entries: int | None = None,
        device: str = "cuda",
        graph_pool: Any | None = None,
    ) -> None:
        self.model = model
        self.kernel_layers = kernel_layers
        self.max_seq_len = max_seq_len
        self.max_index_entries = max_index_entries
        self.device = device
        self.hidden = model.config.hidden_size
        self.hc_mult = model.config.hc_mult
        self.graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool = graph_pool
        self._input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._position = torch.zeros((1,), dtype=torch.long, device=device)
        self._logits: torch.Tensor | None = None
        self._slot: int | None = None
        # Greedy sampling baked into the graph: the argmax of the final
        # logits is written straight back into ``_input_ids`` during replay,
        # so the host reads the next token with one .item() and never
        # touches the full [1, vocab] logits (Laguna's pattern, mirrors
        # laguna_cuda_graph.py's captured argmax).
        self.greedy = False

    @property
    def graph_pool(self) -> Any | None:
        """CUDA graph memory pool, shareable by serial per-slot graphs."""
        return self._graph_pool

    def _reset_state(self, slot: int, *, clear_pages: bool = False) -> None:
        for layer in self.kernel_layers:
            layer.reset_caches(slot)
            if clear_pages:
                layer.clear_pages(slot)

    def _fill(self, token: int, position: int) -> None:
        if not 0 <= position < self.max_seq_len:
            raise IndexError(
                f"decode position {position} out of range for max_seq_len={self.max_seq_len}"
            )
        self._input_ids.fill_(token)
        self._position.fill_(position)

    def _forward(self) -> torch.Tensor:
        """Full single-token graph body over the slot-owned attention stack."""
        if self._slot is None:
            raise RuntimeError("DSV4 decode graph slot was not set before forward")
        model = self.model
        h = model.embed(self._input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for index, block in enumerate(model.blocks):
            residual = h
            x, post, comb = block.hc_pre(
                h,
                block.hc_attn_fn,
                block.hc_attn_scale,
                block.hc_attn_base,
            )
            x = rms_norm(x, block.attn_norm_weight, block.eps)
            x = self.kernel_layers[index](
                x,
                -1,
                slot=self._slot,
                capture=True,
                pos_tensor=self._position,
                graph_max_index_entries=self.max_index_entries,
            )
            x = block.hc_post(x, residual, post, comb)

            residual = x
            x, post, comb = block.hc_pre(
                x,
                block.hc_ffn_fn,
                block.hc_ffn_scale,
                block.hc_ffn_base,
            )
            x = rms_norm(x, block.ffn_norm_weight, block.eps)
            x = block.moe(x, self._input_ids)
            h = block.hc_post(x, residual, post, comb)
        h = model.hc_head(h)
        logits = model.lm_head(rms_norm(h, model.norm_weight, model.eps))
        if self.greedy:
            # Bake the greedy decision into the graph: write argmax back to
            # the input buffer so replay advances to the next token with no
            # host-side logits round-trip.  The final logits are still
            # produced (they are the sampled distribution's source) but the
            # host only reads ``_input_ids[0]`` after replay.
            self._input_ids.fill_(logits.argmax(dim=-1).squeeze())
        return logits

    def capture(self, slot: int) -> None:
        """Warm kernels eagerly, then capture the decode step into a graph."""
        if self.graph is not None:
            if slot != self._slot:
                raise RuntimeError(f"graph is bound to slot {self._slot}, not slot {slot}")
            return
        if torch.device(self.device).type != "cuda":
            raise RuntimeError("DSV4 decode CUDA Graph requires a CUDA device")
        self._slot = slot

        # Use a boundary shared by ratio-4 and ratio-128 layers so warmup
        # compiles the migration/emission kernels too.  The graph is data-
        # driven and can replay every other position through the same body.
        warm_position = min(self.max_seq_len - 1, 127)
        self._reset_state(slot, clear_pages=True)
        self._fill(0, warm_position)

        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(2):
                self._reset_state(slot, clear_pages=True)
                self._fill(0, warm_position)
                self._forward()
        torch.cuda.current_stream(self.device).wait_stream(side)

        self._reset_state(slot, clear_pages=True)
        self._fill(0, warm_position)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._graph_pool):
            logits = self._forward()
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        self.graph = graph
        self._logits = logits
        # Capture executes once and mutates recursive state.  No real request
        # may inherit that dummy sequence.
        self._reset_state(slot, clear_pages=False)

    @torch.inference_mode()
    def replay(self, slot: int, token: int, position: int) -> torch.Tensor:
        if self.graph is None or self._logits is None:
            raise RuntimeError("DSV4 decode graph has not been captured")
        if slot != self._slot:
            raise RuntimeError(f"graph is bound to slot {self._slot}, not slot {slot}")
        self._fill(token, position)
        self.graph.replay()
        if self.greedy:
            # The graph wrote argmax back into _input_ids; return the next
            # token directly (host reads a scalar, never the full logits).
            return int(self._input_ids[0].item())
        return self._logits


class Dsv4BucketedDecodeGraphDriver:
    """One replay surface over multiple indexer-entry decode graph buckets."""

    def __init__(
        self,
        *,
        model,
        kernel_layers: list[Dsv4AttnKernelLayer],
        max_seq_len: int,
        device: str = "cuda",
        graph_pool: Any | None = None,
    ) -> None:
        self.model = model
        self.kernel_layers = kernel_layers
        self.max_seq_len = max_seq_len
        self.device = device
        self._graph_pool = graph_pool
        self._bucket_caps = _index_entry_buckets(kernel_layers)
        self._drivers: list[tuple[int | None, Dsv4DecodeGraphDriver]] = []

    @property
    def graph_pool(self) -> Any | None:
        return self._graph_pool

    def capture(self, slot: int) -> None:
        if self._drivers:
            return
        caps: tuple[int | None, ...] = self._bucket_caps or (None,)
        pool = self._graph_pool
        drivers: list[tuple[int | None, Dsv4DecodeGraphDriver]] = []
        for cap in caps:
            driver = Dsv4DecodeGraphDriver(
                model=self.model,
                kernel_layers=self.kernel_layers,
                max_seq_len=self.max_seq_len,
                max_index_entries=cap,
                device=self.device,
                graph_pool=pool,
            )
            driver.capture(slot)
            pool = driver.graph_pool
            drivers.append((cap, driver))
        self._graph_pool = pool
        self._drivers = drivers

    def _pick_driver(self, position: int) -> Dsv4DecodeGraphDriver:
        if not self._drivers:
            raise RuntimeError("DSV4 decode graph bucket set has not been captured")
        if not self._bucket_caps:
            return self._drivers[0][1]
        needed_entries = max(1, (position + 1) // 4)
        for cap, driver in self._drivers:
            if cap is not None and needed_entries <= cap:
                return driver
        return self._drivers[-1][1]

    @torch.inference_mode()
    def replay(self, slot: int, token: int, position: int) -> torch.Tensor:
        return self._pick_driver(position).replay(slot, token, position)


class Dsv4BatchedDecodeGraphDriver:
    """Fixed-B decode graph driver over the shared slot arena batch path."""

    def __init__(
        self,
        *,
        backend,
        batch_size: int,
        max_index_entries: int | None = None,
        device: str = "cuda",
        graph_pool: Any | None = None,
        greedy: bool = False,
    ) -> None:
        if batch_size not in (1, 2, 4):
            raise ValueError(f"batch_size must be one of (1, 2, 4), got {batch_size}")
        self.backend = backend
        self.batch_size = batch_size
        self.max_index_entries = max_index_entries
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool = graph_pool
        self.greedy = greedy
        # Pack all dynamic integer inputs into one allocation so production
        # replay needs one pinned H2D copy, not three device allocations plus
        # three D2D copies per token step.
        self._packed_inputs = torch.zeros((3, batch_size), dtype=torch.long, device=device)
        self._input_ids = self._packed_inputs[0].view(batch_size, 1)
        self._positions = self._packed_inputs[1]
        self._slot_ids = self._packed_inputs[2]
        self._host_inputs = torch.empty(
            (3, batch_size),
            dtype=torch.long,
            device="cpu",
            pin_memory=torch.device(device).type == "cuda",
        )
        self._logits: torch.Tensor | None = None

    @property
    def graph_pool(self) -> Any | None:
        return self._graph_pool

    def _assert_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> None:
        if input_ids.shape != (self.batch_size, 1):
            raise ValueError(
                f"input_ids must be [{self.batch_size}, 1], got {tuple(input_ids.shape)}"
            )
        if positions.shape != (self.batch_size,):
            raise ValueError(f"positions must be [{self.batch_size}], got {tuple(positions.shape)}")
        if slot_ids.shape != (self.batch_size,):
            raise ValueError(f"slot_ids must be [{self.batch_size}], got {tuple(slot_ids.shape)}")
        if input_ids.dtype is not torch.long:
            raise ValueError(f"input_ids must be torch.long, got {input_ids.dtype}")
        if positions.dtype is not torch.long:
            raise ValueError(f"positions must be torch.long, got {positions.dtype}")
        if slot_ids.dtype is not torch.long:
            raise ValueError(f"slot_ids must be torch.long, got {slot_ids.dtype}")
        expected_device = self._input_ids.device
        if input_ids.device != expected_device:
            raise ValueError(f"input_ids must be on {expected_device}, got {input_ids.device}")
        if positions.device != expected_device:
            raise ValueError(f"positions must be on {expected_device}, got {positions.device}")
        if slot_ids.device != expected_device:
            raise ValueError(f"slot_ids must be on {expected_device}, got {slot_ids.device}")

    def _copy_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> None:
        self._assert_inputs(input_ids, positions, slot_ids)
        self._input_ids.copy_(input_ids)
        self._positions.copy_(positions)
        self._slot_ids.copy_(slot_ids)

    def _copy_host_inputs(
        self,
        input_ids: list[int],
        positions: list[int],
        slot_ids: list[int],
    ) -> None:
        if not (len(input_ids) == len(positions) == len(slot_ids) == self.batch_size):
            raise ValueError(
                f"host input_ids/positions/slot_ids must all match batch_size={self.batch_size}"
            )
        for row, (token, position, slot) in enumerate(zip(input_ids, positions, slot_ids)):
            self._host_inputs[0, row] = token
            self._host_inputs[1, row] = position
            self._host_inputs[2, row] = slot
        self._packed_inputs.copy_(self._host_inputs, non_blocking=True)

    def _capture_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.batch_size > self.backend.num_slots:
            raise RuntimeError(
                f"cannot capture batch_size={self.batch_size} with only "
                f"{self.backend.num_slots} backend slots"
            )
        sample_positions = torch.arange(
            self.batch_size, dtype=torch.long, device=self._input_ids.device
        )
        if self.max_index_entries is not None:
            boundary = min(self.backend.max_seq_len - 1, self.max_index_entries * 4 - 1)
            sample_positions[-1] = boundary
        sample_slot_ids = torch.arange(
            self.batch_size, dtype=torch.long, device=self._input_ids.device
        )
        sample_input_ids = torch.zeros(
            (self.batch_size, 1), dtype=torch.long, device=self._input_ids.device
        )
        return sample_input_ids, sample_positions, sample_slot_ids

    def _reset_capture_slots(self) -> None:
        for slot in range(self.batch_size):
            self.backend.reset_slot(slot)

    def _forward(self) -> torch.Tensor:
        logits = self.backend._forward_decode_batch(
            self._input_ids,
            self._positions,
            self._slot_ids,
            max_index_entries=self.max_index_entries,
        )
        if self.greedy:
            # Bake the greedy decision into the graph: each batch row's
            # argmax is written back into _input_ids, so replay advances
            # every slot with no host-side logits round-trip.
            self._input_ids.copy_(logits.argmax(dim=-1))
        return logits

    def capture(self) -> None:
        if self.graph is not None:
            return
        if torch.device(self.device).type != "cuda":
            raise RuntimeError("DSV4 batched decode CUDA Graph requires a CUDA device")

        sample_input_ids, sample_positions, sample_slot_ids = self._capture_inputs()
        self._reset_capture_slots()

        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(2):
                self._reset_capture_slots()
                self._copy_inputs(sample_input_ids, sample_positions, sample_slot_ids)
                self._forward()
        torch.cuda.current_stream(self.device).wait_stream(side)

        self._reset_capture_slots()
        self._copy_inputs(sample_input_ids, sample_positions, sample_slot_ids)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._graph_pool):
            logits = self._forward()
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        self.graph = graph
        self._logits = logits
        self._reset_capture_slots()

    @torch.inference_mode()
    def replay(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph is None or self._logits is None:
            raise RuntimeError("DSV4 batched decode graph has not been captured")
        self._copy_inputs(input_ids, positions, slot_ids)
        self.graph.replay()
        if self.greedy:
            # The graph wrote each row's argmax back into _input_ids.
            return self._input_ids[:, 0].clone()
        return self._logits

    @torch.inference_mode()
    def replay_host(
        self,
        input_ids: list[int],
        positions: list[int],
        slot_ids: list[int],
    ) -> torch.Tensor:
        if self.graph is None or self._logits is None:
            raise RuntimeError("DSV4 batched decode graph has not been captured")
        self._copy_host_inputs(input_ids, positions, slot_ids)
        self.graph.replay()
        if self.greedy:
            return self._input_ids[:, 0].clone()
        return self._logits


class Dsv4BucketedBatchedDecodeGraphDriver:
    """Decode graph replay surface over B=1/2/4 and index-entry buckets."""

    _SUPPORTED_BATCH_SIZES: tuple[int, ...] = (1, 2, 4)

    def __init__(
        self,
        *,
        backend,
        device: str = "cuda",
        graph_pool: Any | None = None,
        greedy: bool = False,
    ) -> None:
        self.backend = backend
        self.device = device
        self._graph_pool = graph_pool
        self.greedy = greedy
        self._bucket_caps = _index_entry_buckets(backend.slot_layers)
        self._drivers: dict[int, list[tuple[int | None, Dsv4BatchedDecodeGraphDriver]]] = {}

    @property
    def graph_pool(self) -> Any | None:
        return self._graph_pool

    def _capture_batch_sizes(self) -> tuple[int, ...]:
        return tuple(b for b in self._SUPPORTED_BATCH_SIZES if b <= self.backend.num_slots)

    def capture(self) -> None:
        if self._drivers:
            return
        caps: tuple[int | None, ...] = self._bucket_caps or (None,)
        pool = self._graph_pool
        drivers: dict[int, list[tuple[int | None, Dsv4BatchedDecodeGraphDriver]]] = {}
        for batch_size in self._capture_batch_sizes():
            entries: list[tuple[int | None, Dsv4BatchedDecodeGraphDriver]] = []
            for cap in caps:
                driver = Dsv4BatchedDecodeGraphDriver(
                    backend=self.backend,
                    batch_size=batch_size,
                    max_index_entries=cap,
                    device=self.device,
                    graph_pool=pool,
                    greedy=self.greedy,
                )
                driver.capture()
                pool = driver.graph_pool
                entries.append((cap, driver))
            drivers[batch_size] = entries
        self._graph_pool = pool
        self._drivers = drivers

    def _pick_driver(
        self,
        batch_size: int,
        max_index_entries: int | None,
    ) -> Dsv4BatchedDecodeGraphDriver:
        if not self._drivers:
            raise RuntimeError("DSV4 batched decode graph set has not been captured")
        if batch_size not in self._drivers:
            raise ValueError(f"batch_size {batch_size} was not captured")
        drivers = self._drivers[batch_size]
        if max_index_entries is None:
            return drivers[-1][1]
        for cap, driver in drivers:
            if cap is not None and max_index_entries <= cap:
                return driver
        return drivers[-1][1]

    @torch.inference_mode()
    def replay(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_ids: torch.Tensor,
        *,
        max_index_entries: int | None,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        driver = self._pick_driver(batch_size, max_index_entries)
        return driver.replay(input_ids, positions, slot_ids)

    @torch.inference_mode()
    def replay_host(
        self,
        input_ids: list[int],
        positions: list[int],
        slot_ids: list[int],
        *,
        max_index_entries: int | None,
    ) -> torch.Tensor:
        driver = self._pick_driver(len(input_ids), max_index_entries)
        return driver.replay_host(input_ids, positions, slot_ids)


def build_decode_graph_driver(**kwargs) -> Dsv4BucketedDecodeGraphDriver:
    """Construct the replay surface for one slot's decode CUDA Graphs."""
    return Dsv4BucketedDecodeGraphDriver(**kwargs)


def build_batched_decode_graph_driver(**kwargs) -> Dsv4BucketedBatchedDecodeGraphDriver:
    """Construct the replay surface for the shared-slot batched decode graphs."""
    return Dsv4BucketedBatchedDecodeGraphDriver(**kwargs)
