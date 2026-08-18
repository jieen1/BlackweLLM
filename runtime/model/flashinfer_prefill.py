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
from typing import Any

import torch

logger = logging.getLogger("qwen_sm120_runtime.flashinfer_prefill")

_WRAPPER_TYPE = None
_IMPORT_ERROR: BaseException | None = None
_IMPORT_REPORTED = False

# Verify graphs for different batch buckets replay one at a time.  FlashInfer's
# float workspace is therefore safe to share across the 16 full-attention
# layers in a bucket, just like the SparkInfer workspace registry does.  Keep
# the tensors alive here instead of allocating one 128 MiB arena per layer.
_VERIFY_WORKSPACES: dict[tuple[Any, ...], torch.Tensor] = {}


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


class FlashInferVerifyAttention:
    """CUDA-graph causal paged prefill for target speculative verification.

    Qwen's target verify is a causal append: each request contributes an
    ``anchor + draft`` suffix, and the cache already contains that suffix when
    attention runs.  FlashInfer's causal paged-prefill contract is exactly the
    SGLang target-verify contract.  The query and page metadata buffers remain
    fixed-address graph inputs; the host staging arrays are used to refresh
    FlashInfer's split-KV schedule outside the captured region.

    ``q`` is always capacity-sized (``batch * max_verify_tokens``) in the
    runtime, even when the request-local lengths are ragged.  The device
    ``qo_indptr`` buffer excludes the scratch tail, while ``_qo_indptr_last``
    stays at capacity so FlashInfer accepts the graph-stable query tensor.
    """

    def __init__(
        self,
        *,
        batch: int,
        verify_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pages_per_slot: int,
        num_cache_pages: int,
        max_seq_len: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        wrapper_type = load_batch_prefill_wrapper()
        if wrapper_type is None:
            raise RuntimeError("FlashInfer is unavailable")
        if device.type != "cuda":
            raise ValueError("FlashInfer verify attention requires CUDA")
        if batch < 1 or verify_tokens < 1 or page_size < 1 or pages_per_slot < 1:
            raise ValueError("FlashInfer verify geometry must be positive")
        # ``pages_per_slot`` is a logical row width.  The serving pool may use
        # a compact dynamic arena whose physical page count is smaller than
        # ``batch * pages_per_slot``; page-table entries still carry valid
        # physical indices.  FlashInfer only needs a compact CSR index buffer,
        # so do not impose an identity-mapping capacity check here.
        if num_cache_pages < 1:
            raise ValueError("FlashInfer verify cache must contain at least one physical page")

        self.batch = int(batch)
        self.verify_tokens = int(verify_tokens)
        self.capacity = self.batch * self.verify_tokens
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.pages_per_slot = int(pages_per_slot)
        self.max_seq_len = int(max_seq_len)
        self.dtype = dtype
        if kv_dtype not in {
            torch.float8_e4m3fn,
            torch.bfloat16,
            torch.float16,
        }:
            raise ValueError(
                "FlashInfer verify supports BF16/FP16/FP8 KV only, got "
                f"{kv_dtype}"
            )
        self.kv_dtype = kv_dtype
        self.device = device
        self.disable_split_kv = os.environ.get(
            "QSR_QWEN36_VERIFY_FLASHINFER_DISABLE_SPLIT_KV", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.replan_page_counts_only = os.environ.get(
            "QSR_QWEN36_FLASHINFER_REPLAN_PAGE_COUNTS_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        workspace_key = (
            device.index if device.index is not None else torch.cuda.current_device(),
            self.batch,
            self.verify_tokens,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.pages_per_slot,
            self.max_seq_len,
            str(self.dtype),
            str(self.kv_dtype),
        )
        workspace = _VERIFY_WORKSPACES.get(workspace_key)
        if workspace is None:
            workspace_bytes = max(
                64 * 1024 * 1024,
                int(
                    os.environ.get(
                        "QSR_QWEN36_FLASHINFER_VERIFY_WORKSPACE_BYTES",
                        str(256 * 1024 * 1024),
                    )
                ),
            )
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            _VERIFY_WORKSPACES[workspace_key] = workspace
        self.workspace_buffer = workspace

        self._qo_indptr = torch.arange(
            self.batch + 1, dtype=torch.int32, device=device
        ) * self.verify_tokens
        self._kv_indptr = torch.arange(
            self.batch + 1, dtype=torch.int32, device=device
        ) * self.pages_per_slot
        self._kv_indices = torch.arange(
            self.batch * self.pages_per_slot, dtype=torch.int32, device=device
        )
        self._last_page_len = torch.full(
            (self.batch,), self.page_size, dtype=torch.int32, device=device
        )
        self._qo_indptr_host = torch.arange(
            self.batch + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ) * self.verify_tokens
        self._kv_indptr_host = torch.arange(
            self.batch + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ) * self.pages_per_slot
        self._kv_lens_host = torch.full(
            (self.batch,), self.max_seq_len, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._last_page_len_host = torch.full(
            (self.batch,), self.page_size, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._wrapper = wrapper_type(
            self.workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self._qo_indptr,
            paged_kv_indptr_buf=self._kv_indptr,
            paged_kv_indices_buf=self._kv_indices,
            paged_kv_last_page_len_buf=self._last_page_len,
            backend=os.environ.get("QSR_QWEN36_FLASHINFER_BACKEND", "fa2"),
        )
        self._planned_key: object | None = None
        # One verify adapter is shared by all full-attention layers in a
        # graph bucket.  The backend supplies a host-side generation key so
        # the first layer refreshes the CSR/length buffers and the remaining
        # fifteen layers can reuse the exact same metadata without repeating
        # the D2D copies or private planner call.
        self._metadata_key: object | None = None
        self._scale_cache: dict[int, float] = {}
        self._plan_initial()

    @staticmethod
    def _host_ints(values: object) -> list[int]:
        if isinstance(values, torch.Tensor):
            return [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
        return [int(value) for value in values]  # type: ignore[union-attr]

    def _scale_as_float(self, scale: torch.Tensor | float | None) -> float:
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

    def _plan_initial(self) -> None:
        """Compile the graph-capacity kernel once before the first capture."""

        self._wrapper.plan(
            self._qo_indptr,
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
            max_token_per_sequence=self.verify_tokens,
            disable_split_kv=self.disable_split_kv,
        )
        # Runtime q is capacity-sized, including for ragged replay.  The
        # public plan above sees the same capacity-sized initial indptr, so
        # this assignment is also a guard for older FlashInfer builds whose
        # private plan leaves the cached row count untouched.
        self._wrapper._qo_indptr_last = self.capacity  # noqa: SLF001

    def _fast_plan(self, *, max_q_len: int, max_kv_len: int) -> None:
        wrapper = self._wrapper
        cached_module = getattr(wrapper, "_cached_module", None)
        if getattr(wrapper, "_backend", None) != "fa2" or cached_module is None:
            self._wrapper.plan(
                self._qo_indptr,
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
                max_token_per_sequence=max_q_len,
                disable_split_kv=self.disable_split_kv,
            )
            wrapper._qo_indptr_last = self.capacity  # noqa: SLF001
            return

        wrapper._qo_indptr_last = self.capacity  # noqa: SLF001
        wrapper._max_q_len = int(max_q_len)  # noqa: SLF001
        wrapper._max_kv_len = int(max_kv_len)  # noqa: SLF001
        wrapper._max_total_num_rows = self.capacity  # noqa: SLF001
        wrapper._batch_size = self.batch  # noqa: SLF001
        wrapper._num_qo_heads = self.num_q_heads  # noqa: SLF001
        wrapper._num_kv_heads = self.num_kv_heads  # noqa: SLF001
        wrapper._prefix_len_ptr = None  # noqa: SLF001
        wrapper._token_pos_in_items_ptr = None  # noqa: SLF001
        wrapper._token_pos_in_items_len = 0  # noqa: SLF001
        wrapper._max_item_len_ptr = None  # noqa: SLF001
        wrapper._cached_q_data_type = self.dtype  # noqa: SLF001
        wrapper._cached_kv_data_type = self.kv_dtype  # noqa: SLF001
        wrapper._cached_o_data_type = self.dtype  # noqa: SLF001
        wrapper._block_tables = None  # noqa: SLF001
        wrapper._plan_info = cached_module.plan(  # noqa: SLF001
            wrapper._float_workspace_buffer,  # noqa: SLF001
            wrapper._int_workspace_buffer,  # noqa: SLF001
            wrapper._pin_memory_int_workspace_buffer,  # noqa: SLF001
            self._qo_indptr_host,
            self._kv_indptr_host,
            self._kv_lens_host,
            self.capacity,
            self.batch,
            self.num_q_heads,
            self.num_kv_heads,
            self.page_size,
            wrapper.is_cuda_graph_enabled,
            self.head_dim,
            self.head_dim,
            True,
            -1,
            -1,
            self.disable_split_kv,
            0,
            0,
        )

    def update_metadata(
        self,
        page_table: torch.Tensor,
        *,
        host_cache_seqlens: object,
        host_cu_seqlens_q: object | None = None,
        metadata_key: object | None = None,
    ) -> None:
        """Refresh graph buffers and replan only when scheduling inputs change."""

        # ``Qwen36MTPRaggedVerifyCudaGraph`` calls every full-attention layer
        # with the same page table, lengths, and query indptr.  Once the first
        # layer has refreshed this shared adapter, the rest of the layer loop
        # must not repeat the same host parsing and device metadata copies.
        # The key is produced by the graph fill after page-table/version and
        # request-length validation; callers without a key retain the old
        # standalone-adapter behavior and run the full validation below.
        if metadata_key is not None and metadata_key == self._metadata_key:
            return

        lengths = self._host_ints(host_cache_seqlens)
        if len(lengths) != self.batch:
            raise ValueError(
                f"FlashInfer verify expected {self.batch} cache lengths, got {len(lengths)}"
            )
        if host_cu_seqlens_q is None:
            qo = [index * self.verify_tokens for index in range(self.batch + 1)]
        else:
            qo = self._host_ints(host_cu_seqlens_q)
        if len(qo) != self.batch + 1 or qo[0] != 0 or qo[-1] > self.capacity:
            raise ValueError(
                "FlashInfer verify query indptr does not fit the graph capacity: "
                f"{qo} vs capacity {self.capacity}"
            )
        query_lengths = [end - start for start, end in zip(qo, qo[1:])]
        if any(length < 1 or length > self.verify_tokens for length in query_lengths):
            raise ValueError(
                "FlashInfer verify query lengths must be in "
                f"[1,{self.verify_tokens}], got {query_lengths}"
            )
        if any(length <= 0 or length > self.max_seq_len for length in lengths):
            raise ValueError(
                "FlashInfer verify cache lengths exceed capacity: "
                f"{lengths} vs {self.max_seq_len}"
            )
        if tuple(page_table.shape) != (self.batch, self.pages_per_slot):
            raise ValueError(
                "FlashInfer verify page-table shape mismatch: "
                f"expected {(self.batch, self.pages_per_slot)}, got {tuple(page_table.shape)}"
            )

        page_counts = [
            (length + self.page_size - 1) // self.page_size for length in lengths
        ]
        cumulative = [0]
        for count in page_counts:
            cumulative.append(cumulative[-1] + count)
        for index, value in enumerate(qo):
            self._qo_indptr_host[index] = value
        for index, value in enumerate(cumulative):
            self._kv_indptr_host[index] = value
        for index, value in enumerate(lengths):
            self._kv_lens_host[index] = value
            self._last_page_len_host[index] = value - (page_counts[index] - 1) * self.page_size

        # ``run`` passes FlashInfer's device-side ``_kv_lens_buffer`` to the
        # kernel separately from the paged indptr/last-page buffers.  The
        # private fast-plan path below bypasses ``Wrapper.plan()``, so it does
        # not perform the public plan's usual copy into that buffer.  Keep it
        # synchronized with the live cache lengths; leaving the constructor's
        # max-sequence values here makes every replay schedule against the
        # logical arena width instead of the actual request lengths.
        self._wrapper._kv_lens_buffer[: self.batch].copy_(  # noqa: SLF001
            self._kv_lens_host,
            non_blocking=True,
        )

        self._qo_indptr.copy_(self._qo_indptr_host, non_blocking=True)
        self._kv_indptr.copy_(self._kv_indptr_host, non_blocking=True)
        self._last_page_len.copy_(self._last_page_len_host, non_blocking=True)
        for row, count in enumerate(page_counts):
            start = cumulative[row]
            self._kv_indices[start : start + count].copy_(
                page_table[row, :count], non_blocking=True
            )

        # FlashInfer receives the exact lengths through graph-resident
        # ``_kv_lens_buffer`` and ``last_page_len`` on every replay.  Its
        # split-KV worklist only changes when the page geometry or the query
        # shape changes.  SGLang keeps this distinction in its shared target
        # verify wrapper; the opt-in path lets us measure the same invariant
        # locally without making the default path rely on a private planner
        # contract before the A/B result is recorded.
        if self.replan_page_counts_only:
            plan_key = (tuple(query_lengths), tuple(page_counts))
        else:
            plan_key = (tuple(qo), tuple(lengths))
        self._wrapper._max_q_len = max(query_lengths)  # noqa: SLF001
        self._wrapper._max_kv_len = max(lengths)  # noqa: SLF001
        if plan_key != self._planned_key:
            self._fast_plan(
                max_q_len=max(query_lengths),
                max_kv_len=max(lengths),
            )
            self._planned_key = plan_key
        self._metadata_key = (
            metadata_key
            if metadata_key is not None
            else (tuple(qo), tuple(lengths), tuple(page_counts))
        )

    def run(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
    ) -> None:
        if q.shape[0] != self.capacity:
            raise ValueError(
                f"FlashInfer verify expects capacity-sized q={self.capacity}, got {q.shape[0]}"
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
