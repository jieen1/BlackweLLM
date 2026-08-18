"""Optional FlashInfer FA2 paged-prefill driver.

SGLang uses FlashInfer's ``fa2`` paged prefill kernel for Qwen hybrid
full-attention layers.  The runtime keeps this adapter optional: the SM120
installation used for serving has FlashInfer through the reference vLLM
environment, while CPU tests and installations without it retain the
SparkInfer implementation.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

import torch

logger = logging.getLogger("qwen_sm120_runtime.flashinfer_prefill")

_WRAPPER_TYPE = None
_IMPORT_ERROR: BaseException | None = None
_IMPORT_REPORTED = False


def load_batch_prefill_wrapper():
    """Return FlashInfer's paged-prefill wrapper type, or ``None``.

    The local reference environment has a patch-level Python/cubin mismatch
    (0.6.16.post3 vs 0.6.13).  The cubins are already validated on this SM120
    machine, so the version gate is disabled unless the caller supplied an
    explicit policy.
    """

    global _WRAPPER_TYPE, _IMPORT_ERROR, _IMPORT_REPORTED
    if _WRAPPER_TYPE is not None:
        return _WRAPPER_TYPE
    if _IMPORT_ERROR is not None:
        return None

    venv_ninja = os.path.join(os.path.dirname(sys.executable), "ninja")
    if os.path.isfile(venv_ninja) and shutil.which("ninja") is None:
        os.environ["PATH"] = os.path.dirname(venv_ninja) + os.pathsep + os.environ.get(
            "PATH", ""
        )
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except BaseException as exc:  # optional dependency; caller deliberately falls back
        _IMPORT_ERROR = exc
        if not _IMPORT_REPORTED:
            logger.warning("FlashInfer FA2 prefill unavailable; using SparkInfer: %s", exc)
            _IMPORT_REPORTED = True
        return None
    _WRAPPER_TYPE = BatchPrefillWithPagedKVCacheWrapper
    return _WRAPPER_TYPE


class FlashInferPagedPrefill:
    """SGLang-equivalent causal paged prefill for a uniform ``B x Q`` batch.

    The query rows are the newly appended tokens and the paged cache contains
    the complete prefix plus those rows.  Metadata is planned once per batch
    request and reused by every full-attention layer; only the layer's K/V
    cache and output change between calls.
    """

    def __init__(
        self,
        *,
        batch: int,
        tokens_per_slot: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pages_per_slot: int,
        num_cache_pages: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        wrapper_type = load_batch_prefill_wrapper()
        if wrapper_type is None:
            raise RuntimeError("FlashInfer is unavailable")
        if device.type != "cuda":
            raise ValueError("FlashInfer paged prefill requires CUDA")
        if batch < 1 or tokens_per_slot < 1 or pages_per_slot < 1:
            raise ValueError("FlashInfer prefill geometry must be positive")
        if num_cache_pages < batch * pages_per_slot:
            raise ValueError(
                "FlashInfer prefill cache capacity is smaller than the batch page table: "
                f"{num_cache_pages} < {batch * pages_per_slot}"
            )

        self.batch = int(batch)
        self.tokens_per_slot = int(tokens_per_slot)
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.pages_per_slot = int(pages_per_slot)
        self.num_cache_pages = int(num_cache_pages)
        self.dtype = dtype
        self.kv_dtype = kv_dtype
        self.device = device
        workspace_bytes = max(
            64 * 1024 * 1024,
            int(
                os.environ.get(
                    "QSR_QWEN36_FLASHINFER_WORKSPACE_BYTES", str(256 * 1024 * 1024)
                )
            ),
        )
        self.workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self._kv_indptr = torch.empty(batch + 1, dtype=torch.int32, device=device)
        self._kv_indices = torch.empty(
            batch * pages_per_slot, dtype=torch.int32, device=device
        )
        self._last_page_len = torch.empty(batch, dtype=torch.int32, device=device)
        self._kv_indptr_host = torch.empty(batch + 1, dtype=torch.int32, pin_memory=True)
        self._last_page_len_host = torch.empty(batch, dtype=torch.int32, pin_memory=True)
        self._wrapper = wrapper_type(
            self.workspace_buffer,
            kv_layout="NHD",
            backend=os.environ.get("QSR_QWEN36_FLASHINFER_BACKEND", "fa2"),
        )
        self._planned_generation = -1
        self._scale_cache: dict[int, float] = {}

    def _scale_as_float(self, scale: torch.Tensor | float | None) -> float:
        """Convert a one-element checkpoint scale once per layer.

        This FlashInfer build's TVM FFI signature accepts a Python ``float``
        for FP8 calibration, not a scalar CUDA tensor.  The scale tensors are
        immutable after checkpoint loading, so caching the first host read
        keeps the conversion out of subsequent prefill calls.
        """
        if scale is None:
            return 1.0
        if not isinstance(scale, torch.Tensor):
            return float(scale)
        if scale.numel() != 1:
            raise ValueError(f"FlashInfer FP8 scale must have one element, got {scale.shape}")
        key = scale.data_ptr()
        cached = self._scale_cache.get(key)
        if cached is None:
            cached = float(scale.detach().cpu().item())
            self._scale_cache[key] = cached
        return cached

    def _prepare(
        self,
        *,
        page_table: torch.Tensor,
        kv_lengths: tuple[int, ...],
        qo_indptr: torch.Tensor,
        generation: int,
    ) -> None:
        if generation == self._planned_generation:
            return
        if len(kv_lengths) != self.batch:
            raise ValueError(
                f"FlashInfer prefill expected {self.batch} KV lengths, got {len(kv_lengths)}"
            )
        if tuple(page_table.shape) != (self.batch, self.pages_per_slot):
            raise ValueError(
                "FlashInfer prefill page-table shape mismatch: "
                f"expected {(self.batch, self.pages_per_slot)}, got {tuple(page_table.shape)}"
            )

        cumulative = [0]
        page_counts: list[int] = []
        last_page_lens: list[int] = []
        for length in kv_lengths:
            if length <= 0 or length > self.pages_per_slot * self.page_size:
                raise ValueError(
                    "FlashInfer prefill KV length exceeds capacity: "
                    f"{length} > {self.pages_per_slot * self.page_size}"
                )
            pages = (length + self.page_size - 1) // self.page_size
            page_counts.append(pages)
            cumulative.append(cumulative[-1] + pages)
            last_page_lens.append(length - (pages - 1) * self.page_size)
        for index, value in enumerate(cumulative):
            self._kv_indptr_host[index] = value
        for index, value in enumerate(last_page_lens):
            self._last_page_len_host[index] = value
        self._kv_indptr.copy_(self._kv_indptr_host, non_blocking=True)
        self._last_page_len.copy_(self._last_page_len_host, non_blocking=True)
        for row, pages in enumerate(page_counts):
            start = cumulative[row]
            self._kv_indices[start : start + pages].copy_(
                page_table[row, :pages], non_blocking=True
            )

        self._wrapper.plan(
            qo_indptr,
            self._kv_indptr,
            self._kv_indices,
            self._last_page_len,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            head_dim_vo=self.head_dim,
            causal=True,
            sm_scale=self.head_dim**-0.5,
            q_data_type=self.dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        self._planned_generation = generation

    def run(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        kv_lengths: tuple[int, ...],
        qo_indptr: torch.Tensor,
        generation: int,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
    ) -> None:
        self._prepare(
            page_table=page_table,
            kv_lengths=kv_lengths,
            qo_indptr=qo_indptr,
            generation=generation,
        )
        if k_cache.dtype == torch.uint8:
            k_cache = k_cache.view(torch.float8_e4m3fn)
            v_cache = v_cache.view(torch.float8_e4m3fn)
        self._wrapper.run(
            q,
            (k_cache, v_cache),
            k_scale=self._scale_as_float(k_scale),
            v_scale=self._scale_as_float(v_scale),
            out=output,
        )
