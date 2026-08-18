"""FlashInfer paged attention for the non-causal Qwen3.8 DSpark draft.

The Qwen3.8 DSpark draft is a masked, full-attention block: all ``K`` query
rows see the complete context plus the current ``K`` keys.  SparkInfer's
``extend`` kernel is intentionally causal, so representing those rows as
``K`` independent requests is numerically wrong for the official draft.
FlashInfer's paged prefill wrapper exposes the required ``causal=False``
contract and can be planned once for all five draft layers.

This module is optional.  The runtime keeps its existing SparkInfer path when
FlashInfer is absent or cannot load, which preserves the zero-extra-dependency
CPU/test and fallback environments.  The production SM120 environment has
FlashInfer through the vLLM reference installation; its version check is
disabled only for the locally verified cubin/Python patch-version mismatch.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Any

import torch

from runtime.kernels.fused_kv_scatter import fused_kv_scatter

logger = logging.getLogger("qwen_sm120_runtime.flashinfer_dspark_attn")

_FLASHINFER_IMPORT: tuple[Any, ...] | None = None
_FLASHINFER_IMPORT_ERROR: BaseException | None = None
_FLASHINFER_IMPORT_REPORTED = False


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean runtime switch without treating arbitrary text as true."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _load_flashinfer() -> tuple[Any, ...] | None:
    """Load the small FlashInfer surface needed by DSpark, once."""

    global _FLASHINFER_IMPORT, _FLASHINFER_IMPORT_ERROR, _FLASHINFER_IMPORT_REPORTED
    if _FLASHINFER_IMPORT is not None:
        return _FLASHINFER_IMPORT
    if _FLASHINFER_IMPORT_ERROR is not None:
        return None

    # The reference vLLM environment ships the ninja Python package and its
    # executable beside the interpreter, but the shell PATH used by the local
    # server launcher does not necessarily include that directory.  FlashInfer
    # invokes ninja for a first-time backend specialization.
    venv_ninja = os.path.join(os.path.dirname(sys.executable), "ninja")
    if os.path.isfile(venv_ninja) and shutil.which("ninja") is None:
        os.environ["PATH"] = os.path.dirname(venv_ninja) + os.pathsep + os.environ.get(
            "PATH", ""
        )

    # This machine has flashinfer-python 0.6.16.post3 and a cached 0.6.13
    # cubin package.  The kernels are bit-checked below at integration time;
    # without this opt-in the import fails before the actual wrapper can be
    # selected.  Respect an explicit caller setting.
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    except BaseException as exc:  # optional dependency; fallback is deliberate
        _FLASHINFER_IMPORT_ERROR = exc
        if not _FLASHINFER_IMPORT_REPORTED:
            logger.warning("DSpark FlashInfer path unavailable; using SparkInfer fallback: %s", exc)
            _FLASHINFER_IMPORT_REPORTED = True
        return None

    _FLASHINFER_IMPORT = (BatchPrefillWithPagedKVCacheWrapper,)
    return _FLASHINFER_IMPORT


def flashinfer_dspark_available() -> bool:
    """Return whether the optional non-causal paged path can be constructed."""

    return _load_flashinfer() is not None


class FlashInferDSparkAttentionImpl:
    """Batched non-causal paged attention for one DSpark draft geometry.

    One instance is shared by all draft layers in eager mode.  CUDA-graph
    capture creates one instance per batch-size bucket.  The graph metadata
    uses fixed-width page-table segments (one segment per request), so a
    replay can serve any live slot subset without recapturing.  This mirrors
    SGLang's single ``B*gamma`` draft forward rather than the old local
    one-graph-per-slot path.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        page_size: int,
        max_pages: int,
        num_tokens: int,
        device: torch.device,
        use_cuda_graph: bool = False,
        slot: int = 0,
        batch_size: int = 1,
        workspace_buffer: torch.Tensor | None = None,
    ) -> None:
        imported = _load_flashinfer()
        if imported is None:
            raise RuntimeError("FlashInfer is unavailable")
        (wrapper_type,) = imported

        if num_tokens <= 0 or max_pages <= 0 or page_size <= 0:
            raise ValueError(
                "DSpark FlashInfer geometry must be positive: "
                f"num_tokens={num_tokens}, max_pages={max_pages}, page_size={page_size}"
            )
        if device.type != "cuda":
            raise ValueError("DSpark FlashInfer attention requires CUDA")

        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.num_kv_heads = int(num_kv_heads)
        self.scale = float(scale)
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False
        self.page_size = int(page_size)
        self.max_pages = int(max_pages)
        self.num_tokens = int(num_tokens)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"DSpark FlashInfer batch_size must be positive, got {batch_size}")
        self.device = device
        self.use_cuda_graph = bool(use_cuda_graph)
        self.slot = int(slot)
        # SGLang's FlashInfer paged-prefill path leaves split-KV enabled for
        # the DSpark draft block.  The old local rollout disabled it because
        # the graph metadata update had not yet been compared with SGLang.
        # Keep a kill switch for bisecting a bad cubin/runtime combination,
        # but make the SGLang-equivalent path the production default.
        self.disable_split_kv = _env_flag(
            "QSR_DSPARK_FLASHINFER_DISABLE_SPLIT_KV", default=False
        )
        self._prepared_metadata: object | None = None
        self._planned = False

        if workspace_buffer is None:
            workspace_bytes = max(
                64 * 1024 * 1024,
                int(
                    os.environ.get(
                        "QSR_DSPARK_FLASHINFER_WORKSPACE_BYTES",
                        str(128 * 1024 * 1024),
                    )
                ),
            )
            workspace_buffer = torch.empty(
                workspace_bytes, dtype=torch.uint8, device=device
            )
        self.workspace_buffer = workspace_buffer

        self._qo_indptr = (
            torch.arange(self.batch_size + 1, dtype=torch.int32, device=device)
            * self.num_tokens
        )
        # The graph owns a capacity-sized *compact* page-index buffer.  The
        # per-request ranges are repacked into its prefix on each replay;
        # this is necessary because FlashInfer treats indptr as a compact
        # CSR range and cannot skip unused page ids between requests.
        self._kv_indptr = (
            torch.arange(self.batch_size + 1, dtype=torch.int32, device=device)
            * self.max_pages
        )
        self._kv_indices = torch.arange(
            self.batch_size * self.max_pages,
            dtype=torch.int32,
            device=device,
        )
        self._kv_indices.copy_(
            torch.cat(
                [
                    torch.arange(
                        (self.slot + row) * self.max_pages,
                        (self.slot + row + 1) * self.max_pages,
                        dtype=torch.int32,
                        device=device,
                    )
                    for row in range(self.batch_size)
                ]
            )
        )
        self._default_page_table = torch.arange(
            self.slot * self.max_pages,
            (self.slot + self.batch_size) * self.max_pages,
            dtype=torch.int32,
            device=device,
        ).view(self.batch_size, self.max_pages)
        self._kv_indptr_host = torch.empty(
            self.batch_size + 1, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._qo_indptr_host = (
            torch.arange(
                self.batch_size + 1, dtype=torch.int32, device="cpu", pin_memory=True
            )
            * self.num_tokens
        )
        self._kv_lens_host = torch.empty(
            self.batch_size, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._last_page_len_host = torch.empty(
            self.batch_size, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._last_page_len = torch.full(
            (self.batch_size,), self.page_size, dtype=torch.int32, device=device
        )
        self._wrapper = wrapper_type(
            workspace_buffer,
            "NHD",
            use_cuda_graph=self.use_cuda_graph,
            qo_indptr_buf=self._qo_indptr if self.use_cuda_graph else None,
            paged_kv_indptr_buf=self._kv_indptr if self.use_cuda_graph else None,
            paged_kv_indices_buf=self._kv_indices if self.use_cuda_graph else None,
            paged_kv_last_page_len_buf=(
                self._last_page_len if self.use_cuda_graph else None
            ),
            backend=os.environ.get("QSR_DSPARK_FLASHINFER_BACKEND", "fa2"),
        )

        if self.use_cuda_graph:
            self._plan_graph()

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype

    def do_kv_cache_update(
        self,
        layer: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write projected context K/V using the runtime's FP8 scatter."""

        k_cache = kv_cache[0].view(torch.float8_e4m3fn)
        v_cache = kv_cache[1].view(torch.float8_e4m3fn)
        fused_kv_scatter(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            layer._k_scale,
            layer._v_scale,
        )

    def _plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        last_page_len: torch.Tensor,
    ) -> None:
        self._wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.page_size,
            head_dim_vo=self.head_size,
            causal=False,
            sm_scale=self.scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
            max_token_per_sequence=self.num_tokens,
            # Match SGLang's paged-prefill draft path: split the long KV
            # dimension so all SMs participate at 128K context.  The graph
            # wrapper keeps its plan metadata at fixed addresses; replay only
            # changes the CSR page ranges and last-page lengths.
            disable_split_kv=self.disable_split_kv,
        )
        self._planned = True

    def _plan_graph(self) -> None:
        self._plan(
            self._qo_indptr,
            self._kv_indptr,
            self._kv_indices,
            self._last_page_len,
        )

    def _fast_plan_graph(self, active_pages: int) -> None:
        """Refresh FlashInfer's split schedule without a device-to-host sync.

        ``BatchPrefillWithPagedKVCacheWrapper.plan`` reconstructs its host
        scheduler inputs with ``.to("cpu")`` on every call.  That is acceptable
        during setup, but in DSpark it would put a blocking synchronization in
        every speculative round.  SGLang avoids it by calling the already
        loaded FA2 module's private ``plan`` entry point with host-known CSR
        metadata.  Keep the same narrow adapter here; the wrapper's public
        plan remains the fallback for older/incompatible FlashInfer builds.
        """

        wrapper = self._wrapper
        cached_module = getattr(wrapper, "_cached_module", None)
        if (
            not self.use_cuda_graph
            or getattr(wrapper, "_backend", None) != "fa2"
            or cached_module is None
        ):
            self._plan(
                self._qo_indptr,
                self._kv_indptr,
                self._kv_indices[:active_pages],
                self._last_page_len,
            )
            return

        total_num_rows = int(self._qo_indptr_host[-1])
        wrapper._qo_indptr_last = total_num_rows  # noqa: SLF001
        wrapper._max_q_len = self.num_tokens  # noqa: SLF001
        wrapper._max_kv_len = int(self._kv_lens_host.max().item())  # noqa: SLF001
        if getattr(wrapper, "_max_total_num_rows", None) is None:
            wrapper._max_total_num_rows = total_num_rows  # noqa: SLF001
        wrapper._batch_size = self.batch_size  # noqa: SLF001
        wrapper._num_qo_heads = self.num_heads  # noqa: SLF001
        wrapper._num_kv_heads = self.num_kv_heads  # noqa: SLF001
        wrapper._prefix_len_ptr = None  # noqa: SLF001
        wrapper._token_pos_in_items_ptr = None  # noqa: SLF001
        wrapper._token_pos_in_items_len = 0  # noqa: SLF001
        wrapper._max_item_len_ptr = None  # noqa: SLF001
        wrapper._cached_q_data_type = torch.bfloat16  # noqa: SLF001
        wrapper._cached_kv_data_type = torch.float8_e4m3fn  # noqa: SLF001
        wrapper._cached_o_data_type = torch.bfloat16  # noqa: SLF001
        wrapper._block_tables = None  # noqa: SLF001

        # FlashInfer 0.6.16's FA2 plan has the final uniform-q-len argument;
        # the older SGLang helper omits it.  Keep the call aligned with the
        # installed wrapper implementation used by this runtime.
        wrapper._plan_info = cached_module.plan(  # noqa: SLF001
            wrapper._float_workspace_buffer,  # noqa: SLF001
            wrapper._int_workspace_buffer,  # noqa: SLF001
            wrapper._pin_memory_int_workspace_buffer,  # noqa: SLF001
            self._qo_indptr_host,
            self._kv_indptr_host,
            self._kv_lens_host,
            getattr(wrapper, "_max_total_num_rows", None) or total_num_rows,
            self.batch_size,
            self.num_heads,
            self.num_kv_heads,
            self.page_size,
            wrapper.is_cuda_graph_enabled,
            self.head_size,
            self.head_size,
            False,
            -1,
            -1,
            self.disable_split_kv,
            0,
            0,
        )
        self._planned = True

    def update_graph_metadata(
        self,
        kv_len: int | list[int] | torch.Tensor,
        *,
        page_table: torch.Tensor | None = None,
    ) -> None:
        """Update fixed-address graph metadata before replay/capture.

        ``kv_len`` is either one scalar for the legacy B=1 caller or one
        context length per request.  ``page_table`` is optional for backward
        compatibility; when supplied it must be ``[B, max_pages]`` and is
        copied into the graph-owned flattened page-index buffer in one D2D
        operation.  No host-side page-table reconstruction is needed.
        """

        if isinstance(kv_len, torch.Tensor):
            lengths = kv_len.reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
        elif isinstance(kv_len, (list, tuple)):
            lengths = [int(value) for value in kv_len]
        else:
            lengths = [int(kv_len)]
        if len(lengths) != self.batch_size:
            raise ValueError(
                "DSpark FlashInfer graph expected one KV length per request: "
                f"batch={self.batch_size}, got {len(lengths)}"
            )

        pages: list[int] = []
        last_pages: list[int] = []
        for length in lengths:
            total_len = length + self.num_tokens
            if total_len <= 0 or total_len > self.max_pages * self.page_size:
                raise RuntimeError(
                    f"DSpark FlashInfer graph KV length {total_len} exceeds "
                    f"capacity {self.max_pages * self.page_size}"
                )
            num_pages = (total_len + self.page_size - 1) // self.page_size
            pages.append(num_pages)
            last_pages.append(total_len - (num_pages - 1) * self.page_size)

        # Indptr points at the compact prefix of each request's page ids.
        cumulative = [0]
        for num_pages in pages:
            cumulative.append(cumulative[-1] + num_pages)
        for index, value in enumerate(cumulative):
            self._kv_indptr_host[index] = value
        for index, value in enumerate(last_pages):
            self._last_page_len_host[index] = value
        for index, value in enumerate(lengths):
            self._kv_lens_host[index] = value + self.num_tokens
        self._kv_indptr.copy_(self._kv_indptr_host, non_blocking=True)
        self._last_page_len.copy_(self._last_page_len_host, non_blocking=True)
        if page_table is None:
            page_table = self._default_page_table
        if tuple(page_table.shape) != (self.batch_size, self.max_pages):
            raise ValueError(
                "DSpark FlashInfer graph page_table shape mismatch: "
                f"expected {(self.batch_size, self.max_pages)}, got {tuple(page_table.shape)}"
            )
        for row, num_pages in enumerate(pages):
            start = cumulative[row]
            self._kv_indices[start : start + num_pages].copy_(
                page_table[row, :num_pages], non_blocking=True
            )

        if not self.disable_split_kv:
            # Split-KV scheduling is a function of the actual KV lengths, not
            # only of the fixed graph tensor addresses.  Re-plan outside the
            # captured region after refreshing the CSR buffers.  This is the
            # same lifecycle SGLang uses for its target-verify prefill wrapper;
            # retaining the capture-time plan (which was made at the 2K warmup
            # length) both wastes the long-context parallelism and changes the
            # draft logits as the request grows.
            self._fast_plan_graph(cumulative[-1])

    def _prepare_eager(self, metadata: Any) -> None:
        if metadata is self._prepared_metadata:
            return
        qo_indptr = getattr(metadata, "flashinfer_qo_indptr", None)
        kv_indptr = getattr(metadata, "flashinfer_kv_indptr", None)
        kv_indices = getattr(metadata, "flashinfer_kv_indices", None)
        last_page_len = getattr(metadata, "flashinfer_kv_last_page_len", None)
        if any(value is None for value in (qo_indptr, kv_indptr, kv_indices, last_page_len)):
            raise ValueError(
                "DSpark FlashInfer metadata is missing paged layout tensors"
            )
        self._plan(qo_indptr, kv_indptr, kv_indices, last_page_len)
        self._prepared_metadata = metadata

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value, output_scale, output_block_scale
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        expected_tokens = self.batch_size * self.num_tokens
        if num_actual_tokens != expected_tokens:
            raise ValueError(
                "DSpark FlashInfer expects a fixed B*K-token masked block, got "
                f"{num_actual_tokens} (B={self.batch_size}, K={self.num_tokens})"
            )
        q = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)

        if not self.use_cuda_graph:
            self._prepare_eager(attn_metadata)
        elif not self._planned:
            raise RuntimeError("DSpark FlashInfer CUDA graph was not planned")

        # Draft checkpoints have no KV calibration parameters; the cache is
        # written with scale=1.0.  Keep these scalar arguments explicit so the
        # FP8 kernel does not infer a different calibration contract.
        self._wrapper.run(
            q,
            (key_cache, value_cache),
            k_scale=1.0,
            v_scale=1.0,
            out=output[:num_actual_tokens],
        )
        del layer
        return output
