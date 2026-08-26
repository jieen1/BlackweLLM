"""Eager DSpark round driver for the Qwen3.8 target.

The target model owns the hybrid Qwen3.8 KV/GDN state.  DSpark adds one
small, separate dense draft model whose KV cache is populated from target
hidden-state taps.  This module intentionally starts with the simplest
complete contract:

* one fixed FP8 paged draft cache per serving slot;
* ``bonus + mask*(gamma-1)`` draft input and Markov correction;
* target verify over ``anchor + gamma drafts``;
* target GDN/KV rollback followed by target-hidden KV injection for the
  accepted prefix;
* CUDA-Graph-backed target verify and greedy ``B*K`` draft blocks;
* one fixed-width target verify for the active batch, followed by a compact
  accepted-prefix hidden/KV injection into the draft cache;
* a small host-facing acceptance/commit epilogue, matching SGLang's state
  transitions while keeping the large model work batched on device.

The shape and state transitions mirror SGLang's Qwen3.8 DSpark path.  In
particular, the trailing target bonus is client-visible but is not in the
target KV yet; it becomes the next draft input after the accepted target
prefix has been injected into the draft cache.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from runtime.backends.bf_attention import bf_attn_context, replace_laguna_attention
from runtime.backends.flashinfer_dspark_attn import (
    FlashInferDSparkAttentionImpl,
    compact_dflash_window,
    flashinfer_dspark_available,
)
from runtime.backends.laguna_sparkinfer_attn import (
    SparkinferAttentionImpl,
    SparkinferAttnMetadata,
)
from runtime.backends.qwen36_mtp import Qwen36MTPGDNRows
from runtime.dspark_schedule import (
    build_uninitialized_sps_table,
    choose_verify_widths,
    compute_verify_token_budget,
    load_sps_table,
    schedule_verify_widths_topk,
    survival_prefix,
)
from runtime.laguna_runtime import bind_laguna_kv_cache
from runtime.logprobs import compute_logprobs
from runtime.mtp_accept import (
    determine_accept_reject_batch,
    sample_accept_reject,
    sample_accept_reject_sparse,
)
from runtime.round_profile import round_profile
from runtime.sampling import (
    SamplingParams,
    compute_sampling_distribution,
    make_generator,
    sample_from_logits,
)

if TYPE_CHECKING:
    from runtime.backends.qwen36 import Qwen36Backend
    from runtime.model.qwen36_dspark import Qwen36DSparkDraftForCausalLM


def _flatten_target_taps(
    target_taps: list[torch.Tensor], *, expected_features: int
) -> tuple[torch.Tensor, int]:
    """Concatenate ``[1, Q, H]`` target taps into ``[Q, L*H]``."""

    if not target_taps:
        raise ValueError("DSpark target hidden capture returned no layer taps")
    rows: list[torch.Tensor] = []
    query_len: int | None = None
    for tap in target_taps:
        if tap.ndim == 3:
            if tap.shape[0] != 1:
                raise ValueError(
                    f"DSpark target hidden taps support batch size 1 only; got {tuple(tap.shape)}"
                )
            tap = tap[0]
        if tap.ndim != 2:
            raise ValueError(f"DSpark target hidden tap must be rank 2/3, got {tap.ndim}")
        if query_len is None:
            query_len = int(tap.shape[0])
        elif int(tap.shape[0]) != query_len:
            raise ValueError("DSpark target hidden taps have different query lengths")
        rows.append(tap)
    combined = torch.cat(rows, dim=-1)
    if combined.shape[-1] != expected_features:
        raise ValueError(
            "DSpark target hidden feature dim mismatch: "
            f"expected {expected_features}, got {combined.shape[-1]}"
        )
    assert query_len is not None
    return combined, query_len


def _flatten_target_taps_ragged(
    target_taps: list[torch.Tensor],
    *,
    batch_size: int,
    accepted_counts: list[int],
    expected_features: int,
    verify_lens: list[int] | None = None,
) -> torch.Tensor:
    """Pack accepted target taps as one request-major ragged matrix.

    SGLang's ``inject_ragged`` consumes one compact hidden matrix plus one
    compact position/cache-location vector.  The target graph keeps a dense
    ``[B, K+1, H]`` tap per selected layer; this helper performs the same
    compacting without padding rejected rows into the draft KV projection.
    """

    if len(accepted_counts) != batch_size:
        raise ValueError("accepted_counts must contain one entry per DSpark request")
    if verify_lens is not None:
        if len(verify_lens) != batch_size:
            raise ValueError("verify_lens must contain one entry per DSpark request")
        if any(int(length) < 1 for length in verify_lens):
            raise ValueError("verify_lens must be positive")
        if any(
            int(accepted) < 0 or int(accepted) > int(length)
            for accepted, length in zip(accepted_counts, verify_lens, strict=True)
        ):
            raise ValueError("accepted_counts cannot exceed verify_lens")
    normalized: list[torch.Tensor] = []
    for tap in target_taps:
        if tap.ndim == 2:
            if verify_lens is None:
                if tap.shape[0] % batch_size:
                    raise ValueError("flattened DSpark target tap is not divisible by batch size")
                tap = tap.view(batch_size, -1, tap.shape[-1])
            elif tap.shape[0] < sum(int(length) for length in verify_lens):
                raise ValueError("compact DSpark target tap is shorter than verify_lens")
        if tap.ndim != 3 or tap.shape[0] != batch_size:
            if verify_lens is None or tap.ndim != 2:
                raise ValueError(
                    f"DSpark batched target tap must have shape [B, Q, H], got {tuple(tap.shape)}"
                )
        normalized.append(tap)
    if not normalized:
        raise ValueError("DSpark target hidden capture returned no layer taps")

    if verify_lens is None and all(tap.ndim == 3 for tap in normalized):
        # The fixed-width graph is the common static DFlash2 path.  Build the
        # feature-concatenated tensor once, then compact request rows from it.
        # The old request-major loop performed one feature ``cat`` per request
        # and one more ``cat`` per tap, even when every request accepted the
        # full graph width.  That allocation/launch fan-out sits directly in
        # ``target_hidden_sync`` after every target verify.
        combined = torch.cat(normalized, dim=-1)
        if combined.shape[-1] != expected_features:
            raise ValueError(
                "DSpark target hidden feature dim mismatch: "
                f"expected {expected_features}, got {combined.shape[-1]}"
            )
        query_len = int(combined.shape[1])
        if any(int(count) < 0 or int(count) > query_len for count in accepted_counts):
            raise ValueError(
                "DSpark accepted counts must fit the fixed verify width: "
                f"counts={accepted_counts}, width={query_len}"
            )
        if all(int(count) == query_len for count in accepted_counts):
            # ``combined`` is contiguous after cat, so this is a view and
            # avoids a second device allocation for the usual full-accept
            # DFlash2 round.
            return combined.reshape(-1, expected_features)
        rows = [
            combined[row, : int(count)]
            for row, count in enumerate(accepted_counts)
            if int(count) > 0
        ]
        if not rows:
            return combined.new_empty((0, expected_features))
        return torch.cat(rows, dim=0)

    rows: list[torch.Tensor] = []
    compact_offset = 0
    for request, accepted_count in enumerate(accepted_counts):
        length = int(accepted_count)
        if length <= 0:
            if verify_lens is not None:
                compact_offset += int(verify_lens[request])
            continue
        request_rows: list[torch.Tensor] = []
        for tap in normalized:
            if tap.ndim == 2:
                assert verify_lens is not None
                request_rows.append(tap[compact_offset : compact_offset + length])
            else:
                if tap.shape[1] < length:
                    raise ValueError("DSpark target tap is shorter than the accepted prefix")
                request_rows.append(tap[request, :length])
        rows.append(torch.cat(request_rows, dim=-1))
        if verify_lens is not None:
            compact_offset += int(verify_lens[request])
    if not rows:
        return normalized[0].new_empty((0, expected_features))
    combined = torch.cat(rows, dim=0)
    if combined.shape[-1] != expected_features:
        raise ValueError(
            "DSpark target hidden feature dim mismatch: "
            f"expected {expected_features}, got {combined.shape[-1]}"
        )
    return combined


class Qwen36DSparkEngine:
    """Own the external Qwen3.8 DSpark draft and its per-slot KV cache."""

    def __init__(
        self,
        backend: Qwen36Backend,
        draft_model: Qwen36DSparkDraftForCausalLM,
    ) -> None:
        if backend.device.type != "cuda":
            raise ValueError("Qwen36DSparkEngine requires a CUDA device")
        if not draft_model.config.target_layer_ids:
            raise ValueError("DSpark draft has no target hidden-state taps")
        expected_gamma = (
            draft_model.config.block_size - 1
            if getattr(draft_model, "is_dflash2", False)
            else draft_model.config.block_size
        )
        if draft_model.gamma != expected_gamma:
            raise ValueError(
                "external Qwen draft gamma does not match its block contract; "
                f"got gamma={draft_model.gamma}, expected={expected_gamma}, "
                f"block_size={draft_model.config.block_size}"
            )

        self.backend = backend
        self.model = backend.model
        self.draft_model = draft_model
        self.device = backend.device
        self.dtype = backend.dtype
        self.k = int(draft_model.gamma)
        # DFlash2 evaluates the anchor plus K masked rows. Legacy DSpark's
        # query width equals its proposal width, so keep this distinction out
        # of the target verify/accept code below.
        self._draft_query_tokens = (
            self.k + 1 if getattr(draft_model, "is_dflash2", False) else self.k
        )
        if getattr(draft_model, "is_dflash2", False):
            set_block_size = getattr(draft_model.model, "set_block_size", None)
            if set_block_size is not None:
                set_block_size(self._draft_query_tokens)
        self.page_size = backend.pool.page_size
        self.pages_per_slot = backend.pool.pages_per_slot
        self.max_seq_len = backend.max_seq_len
        self._draft_probs: dict[int, torch.Tensor] = {}
        self._draft_sparse_probs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Confidence is produced by the same draft graph that produces the
        # token block.  Keep a graph-owned view until the following target
        # verify chooses its width; unlike tokens, it must never be converted
        # to Python during proposal generation.
        self._draft_confidence: dict[int, torch.Tensor] = {}
        # The draft CUDA graph owns its output row on device.  Keep one
        # persistent ``[anchor + K]`` verify-input row per slot so the target
        # verify graph can consume that output with a D2D copy.  The old path
        # called ``.tolist()`` on every draft block, synchronizing the stream
        # before the target graph could even be filled.
        self._verify_tokens_buf: dict[int, torch.Tensor] = {}
        self._verify_host: dict[int, torch.Tensor] = {}
        self._verify_host_views: dict[int, object] = {}
        for slot in range(backend.num_slots):
            self._verify_tokens_buf[slot] = torch.zeros(
                1, self.k + 1, dtype=torch.long, device=self.device
            )
            host = torch.zeros(1, self.k + 1, dtype=torch.long, device="cpu", pin_memory=True)
            self._verify_host[slot] = host
            self._verify_host_views[slot] = host.numpy()
        self._verify_tokens_batch: dict[int, torch.Tensor] = {}
        self._verify_host_batch: dict[int, torch.Tensor] = {}
        self._verify_host_batch_views: dict[int, object] = {}
        self._verify_tokens_batch_by_width: dict[tuple[int, int], torch.Tensor] = {}
        self._verify_host_batch_by_width: dict[tuple[int, int], torch.Tensor] = {}
        self._verify_host_batch_views_by_width: dict[tuple[int, int], object] = {}
        for batch_size in range(1, backend.num_slots + 1):
            self._verify_tokens_batch[batch_size] = torch.zeros(
                batch_size, self.k + 1, dtype=torch.long, device=self.device
            )
            host = torch.zeros(
                batch_size,
                self.k + 1,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            self._verify_host_batch[batch_size] = host
            self._verify_host_batch_views[batch_size] = host.numpy()
            for width in range(self.k):
                key = (batch_size, width)
                self._verify_tokens_batch_by_width[key] = torch.zeros(
                    batch_size, width + 1, dtype=torch.long, device=self.device
                )
                width_host = torch.zeros(
                    batch_size,
                    width + 1,
                    dtype=torch.long,
                    device="cpu",
                    pin_memory=True,
                )
                self._verify_host_batch_by_width[key] = width_host
                self._verify_host_batch_views_by_width[key] = width_host.numpy()
        # DSpark uses the same permanent candidate-row topology as MTP.  The
        # target verify graph already knows how to fill these rows; keeping
        # the allocation here also makes the eager fallback use identical
        # rollback semantics instead of silently switching to cloned GDN
        # snapshots when capture is disabled.
        self._spec_rows = Qwen36MTPGDNRows(backend, self.k)
        self._spec_state_col = [0] * (backend.num_slots + 1)
        self.backend.stats["dspark_verify_width_histogram"] = [0] * (self.k + 1)
        self.capture_aux_hidden_states = True
        self.capture_device_accept = os.environ.get("QSR_QWEN36_DSPARK_FUSED_ACCEPT", "1") != "0"
        # Every production caller below constructs a contiguous range from
        # host-known scalar bounds.  Keep that fact explicit so cache mapping
        # does not synchronize the GPU just to rediscover the same bounds.
        # ``0`` is a diagnostic rollback for comparisons against the old
        # generic arbitrary-position validation path.
        self._fast_slot_mapping = (
            os.environ.get("QSR_QWEN36_DSPARK_FAST_SLOT_MAPPING", "1") != "0"
        )
        # Ordinary SGLang DSpark pools inject accepted hidden rows after the
        # ragged verify graph.  Keep the graph-folded variant opt-in: it is a
        # specialized pool optimization, not the standard DSpark contract,
        # and this runtime's generic cache writer is not that pool API.
        self.fuse_context_kv = os.environ.get("QSR_QWEN36_DSPARK_FUSED_CONTEXT_KV", "0") != "0"
        self.verify_mode = os.environ.get("QSR_QWEN36_DSPARK_VERIFY_MODE", "static").strip().lower()
        if self.verify_mode not in {"static", "cap-accept", "compact"}:
            raise ValueError(
                "QSR_QWEN36_DSPARK_VERIFY_MODE must be one of "
                "static, cap-accept, compact; "
                f"got {self.verify_mode!r}"
            )
        threshold_raw = os.environ.get("QSR_QWEN36_DSPARK_CONFIDENCE_THRESHOLD")
        self._confidence_threshold = None if threshold_raw is None else float(threshold_raw)
        if self._confidence_threshold is not None and not 0.0 <= self._confidence_threshold <= 1.0:
            raise ValueError(
                "QSR_QWEN36_DSPARK_CONFIDENCE_THRESHOLD must be in [0,1], "
                f"got {self._confidence_threshold}"
            )
        self._sps_table_path = os.environ.get("QSR_QWEN36_DSPARK_SPS_TABLE")
        # Compact mode is the SGLang-shaped ragged verify path.  A deployment
        # without an explicit SPS table or confidence threshold still verifies
        # the full K block (SGLang's static/no-profile fallback); it must not
        # pay for a confidence-head projection and a per-request GPU->CPU
        # transfer whose result is known in advance.  Dynamic planning is an
        # explicit opt-in through one of those two knobs.
        self._dynamic_planner = self.verify_mode != "static" and (
            self._sps_table_path is not None or self._confidence_threshold is not None
        )
        self.capture_confidence = self._dynamic_planner
        self._load_confidence_sts()
        if self._sps_table_path:
            self._sps_table = load_sps_table(self._sps_table_path)
        elif self.verify_mode in {"cap-accept", "compact"} and self._confidence_threshold is None:
            # This is intentionally the same flat fallback as SGLang: without
            # a measured table the objective is monotonically increasing and
            # the top-k scheduler selects every candidate.
            self._sps_table = build_uninitialized_sps_table(
                max_batch_tokens=backend.num_slots * (self.k + 1)
            )
        else:
            self._sps_table = None
        self._min_verify_len = int(os.environ.get("QSR_QWEN36_DSPARK_MIN_VERIFY_LEN", "1"))
        self._max_verify_len = int(
            os.environ.get("QSR_QWEN36_DSPARK_MAX_VERIFY_LEN", str(self.k + 1))
        )
        if not 0 <= self._min_verify_len <= self._max_verify_len <= self.k + 1:
            raise ValueError(
                "QSR_QWEN36_DSPARK_*_VERIFY_LEN must satisfy "
                f"0 <= min <= max <= {self.k + 1}, got "
                f"{self._min_verify_len}, {self._max_verify_len}"
            )
        self._survival_eps = float(os.environ.get("QSR_QWEN36_DSPARK_SURVIVAL_EPS", "1e-6"))
        if self._survival_eps < 0.0:
            raise ValueError(
                f"QSR_QWEN36_DSPARK_SURVIVAL_EPS must be non-negative, got {self._survival_eps}"
            )
        self._use_cuda_graph = (
            self.device.type == "cuda"
            and os.environ.get("QSR_QWEN36_DSPARK_CUDA_GRAPH", "1") != "0"
        )
        if getattr(draft_model, "is_dflash2", False) and not self._use_cuda_graph:
            raise RuntimeError("DFlash2 requires CUDA Graphs; set QSR_QWEN36_DSPARK_CUDA_GRAPH=1")
        self._verify_cg = None
        self._verify_ragged_cg = None
        self._verify_cg_by_width: dict[int, Any] = {}
        # Draft graphs are keyed by active batch size, matching SGLang's
        # B*K proposer contract.  The old per-slot shape was semantically
        # correct but serialized the draft model across slots.
        self._draft_cg: dict[int, Any] = {}
        self._draft_graph_pool: object | None = None
        self._cg_captured = False
        self.cg_status: dict[str, str] = {}
        self.use_flashinfer_draft = (
            os.environ.get("QSR_QWEN36_DSPARK_FLASHINFER", "1") != "0"
            and flashinfer_dspark_available()
        )
        if getattr(draft_model, "is_dflash2", False) and not self.use_flashinfer_draft:
            raise RuntimeError(
                "DFlash2 requires the FlashInfer draft attention path: its layers are "
                "non-causal sliding-window attention, which the legacy causal SparkInfer "
                "fallback cannot represent"
            )
        self._flashinfer_workspace_buffer: torch.Tensor | None = None
        self._flashinfer_draft_impl: FlashInferDSparkAttentionImpl | None = None
        self._draft_window_left = int(
            getattr(draft_model.config, "sliding_window", -1) - 1
            if getattr(draft_model, "is_dflash2", False)
            else -1
        )
        if self._draft_window_left < -1:
            raise ValueError(
                f"DFlash2 sliding_window must be -1 or non-negative, got {self._draft_window_left}"
            )
        self._use_compact_draft_cache = (
            getattr(draft_model, "is_dflash2", False)
            and self._draft_window_left >= 0
            and os.environ.get("QSR_QWEN36_DSPARK_COMPACT_DRAFT_CACHE", "1") != "0"
        )
        # Narrow physical rows for the draft KV family. In compact mode the
        # sliding-window view only ever reads the last ``window + page``
        # tokens, so each row keeps just enough pages for that window plus
        # one slack page and every absolute page id maps to
        # ``id % draft_row_pages``. The mapping is a pure function of the
        # absolute page number, so prefix preserve/restore stays consistent
        # without tracking physical history. Saves ~12.4 GiB at 4 slots
        # (12.50 -> ~0.12 GiB for the 5-layer DFlash2 draft).
        self._draft_narrow_rows = (
            self._use_compact_draft_cache
            and os.environ.get("QSR_QWEN36_DSPARK_DRAFT_NARROW_ROWS", "1") != "0"
        )
        if self._draft_narrow_rows:
            # Window span can reach window+slack+query pages; add one more
            # page and RESERVE the last one exclusively for the fused-context
            # epilogue's scratch rows, whose addresses must stay unique.
            window_tokens = self._draft_window_left + 1 + self._draft_query_tokens
            self.draft_row_pages = -(-window_tokens // self.page_size) + 2
            self.draft_ring_pages = self.draft_row_pages - 1
        else:
            self.draft_row_pages = self.pages_per_slot
            self.draft_ring_pages = self.pages_per_slot
        if self.use_flashinfer_draft:
            workspace_bytes = max(
                64 * 1024 * 1024,
                int(
                    os.environ.get(
                        "QSR_DSPARK_FLASHINFER_WORKSPACE_BYTES",
                        str(128 * 1024 * 1024),
                    )
                ),
            )
            self._flashinfer_workspace_buffer = torch.empty(
                workspace_bytes, dtype=torch.uint8, device=self.device
            )

        self._draft_layer_names: list[str] = []
        self._draft_attn_layers: dict[str, Any] = {}
        for layer in draft_model.model.layers:
            attn = layer.self_attn.attn
            name = attn.layer_name
            self._draft_layer_names.append(name)
            self._draft_attn_layers[name] = attn
        if not self._draft_layer_names:
            raise ValueError("DSpark draft has no attention layers")

        # The target persistent-prefix family has one scratch row. DSpark is
        # a separate causal cache family, so it needs its own scratch row to
        # make an exact prompt restore valid after the source slot continues
        # decoding or is reused by another request.
        self.scratch_row = backend.num_slots
        total_blocks = (backend.num_slots + 1) * self.draft_row_pages
        self._draft_kv_caches: dict[str, torch.Tensor] = {}
        for name, attn in self._draft_attn_layers.items():
            shape = (
                2,
                total_blocks,
                self.page_size,
                attn.num_kv_heads,
                attn.head_size,
            )
            cache_dtype = getattr(attn, "kv_cache_torch_dtype", torch.float8_e4m3fn)
            if cache_dtype not in (torch.float8_e4m3fn, torch.bfloat16):
                raise ValueError(
                    "unsupported DSpark draft KV cache dtype: "
                    f"{cache_dtype!r}; expected FP8 or BF16"
                )
            self._draft_kv_caches[name] = torch.zeros(shape, dtype=cache_dtype, device=self.device)
        bind_laguna_kv_cache(self._draft_kv_caches, self._draft_attn_layers, [])
        self._patch_attention()

        # The target prefix cache and the DSpark draft cache are independent
        # cache families. Keeping only the target pages on reset would make a
        # warm target hit feed the draft model from an empty/old context. The
        # live length is reset with the slot; the cached length survives until
        # the backend explicitly drops that prefix or overwrites it on reset.
        self._draft_kv_len = [0] * backend.num_slots
        self._cached_prefix_len = [0] * backend.num_slots
        self._scratch_valid_pages: set[int] = set()

        self.stats: dict[str, int] = {
            "rounds": 0,
            "sampled_rounds": 0,
            "draft_forwards": 0,
            "draft_graph_replays": 0,
            "verify_graph_replays": 0,
            "target_hidden_injections": 0,
        }

        if self._use_cuda_graph and self.fuse_context_kv:
            # Build the stacked projection views before graph capture.  The
            # first-call lazy path creates/contiguates GPU tensors and is not
            # part of the captured steady-state region.
            self.draft_model.model._build_fused_kv_buffers()  # noqa: SLF001

        if self._use_cuda_graph:
            self.capture_cuda_graphs()

    def _load_confidence_sts(self) -> None:
        """Load SGLang-compatible per-position confidence temperatures."""

        path = os.environ.get("QSR_QWEN36_DSPARK_CONFIDENCE_STS_PATH")
        if not path:
            return
        head = self.draft_model.confidence_head
        if head is None:
            raise ValueError(
                "QSR_QWEN36_DSPARK_CONFIDENCE_STS_PATH requires a DSpark confidence head"
            )
        try:
            with open(path, encoding="utf-8") as stream:
                calibration = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid DSpark STS calibration file: {path}") from exc
        temperatures = calibration.get("temperatures") if isinstance(calibration, Mapping) else None
        if not isinstance(temperatures, list) or len(temperatures) != self.k:
            raise ValueError(
                "DSpark STS calibration temperatures must contain exactly "
                f"K={self.k} values; got {temperatures!r}"
            )
        values = [float(value) for value in temperatures]
        if any(value <= 0.0 for value in values):
            raise ValueError("DSpark STS calibration temperatures must be positive")
        head.sts_temperatures = torch.tensor(values, dtype=torch.float32, device=self.device)

    @property
    def kv_bytes(self) -> int:
        return sum(cache.numel() * cache.element_size() for cache in self._draft_kv_caches.values())

    def _validate_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.backend.num_slots:
            raise ValueError(f"invalid DSpark slot {slot}")

    def preserve_prefix(self, slot: int, kv_len: int) -> bool:
        """Retain the draft KV prefix before the owning target slot resets."""

        self._validate_slot(slot)
        kv_len = int(kv_len)
        if kv_len < 0 or kv_len > self.max_seq_len:
            raise ValueError(f"invalid DSpark prefix length {kv_len}")
        if self._draft_kv_len[slot] < kv_len:
            # Do not advertise a target-only prefix. The caller will safely
            # fall back to a full prefill when this family is incomplete.
            self._cached_prefix_len[slot] = 0
            return False
        self._cached_prefix_len[slot] = kv_len
        return True

    def can_restore_prefix(self, slot: int, kv_len: int) -> bool:
        self._validate_slot(slot)
        kv_len = int(kv_len)
        return 0 <= kv_len <= self._cached_prefix_len[slot]

    def restore_prefix(self, slot: int, kv_len: int) -> None:
        if not self.can_restore_prefix(slot, kv_len):
            raise RuntimeError(
                f"DSpark prefix is not restorable: slot={slot}, kv_len={kv_len}, "
                f"cached={self._cached_prefix_len[slot]}"
            )
        self._draft_kv_len[slot] = int(kv_len)

    def copy_prefix(self, source_slot: int, target_slot: int, kv_len: int) -> None:
        """Copy retained draft pages for a cross-slot target prefix hit."""

        self._validate_slot(source_slot)
        self._validate_slot(target_slot)
        kv_len = int(kv_len)
        if not self.can_restore_prefix(source_slot, kv_len):
            raise RuntimeError(
                f"DSpark source prefix is not restorable: slot={source_slot}, kv_len={kv_len}"
            )
        if source_slot == target_slot:
            self.restore_prefix(target_slot, kv_len)
            return
        pages = (kv_len + self.page_size - 1) // self.page_size
        if pages > self.pages_per_slot:
            raise RuntimeError(
                f"DSpark prefix exceeds slot page capacity: pages={pages}, "
                f"capacity={self.pages_per_slot}"
            )
        # Narrow rows: only the trailing window survives physically; older
        # pages are invisible to the sliding window either way.
        tail = min(pages, self.draft_row_pages)
        first = pages - tail
        source_start = source_slot * self.draft_row_pages
        target_start = target_slot * self.draft_row_pages
        if self._draft_narrow_rows:
            offs = torch.arange(
                first, pages, dtype=torch.long, device=self.device
            ) % (self.draft_ring_pages)
            source_ids = source_start + offs
            target_ids = target_start + offs
            for cache in self._draft_kv_caches.values():
                cache[:, target_ids] = cache[:, source_ids]
        else:
            source_start += first
            target_start += first
            for cache in self._draft_kv_caches.values():
                cache[:, target_start : target_start + tail].copy_(
                    cache[:, source_start : source_start + tail]
                )
        self._draft_kv_len[target_slot] = kv_len
        self._cached_prefix_len[target_slot] = 0

    def drop_prefix(self, slot: int) -> None:
        self._validate_slot(slot)
        self._cached_prefix_len[slot] = 0

    def snapshot_prefix_to_scratch(
        self,
        source_slot: int,
        kv_len: int,
        *,
        scratch_pages: tuple[int, ...],
    ) -> bool:
        """Copy a retained draft prefix into the persistent scratch row."""

        self._validate_slot(source_slot)
        kv_len = int(kv_len)
        if self._draft_kv_len[source_slot] < kv_len:
            return False
        pages = (kv_len + self.page_size - 1) // self.page_size
        if len(scratch_pages) != pages or len(set(scratch_pages)) != pages:
            raise ValueError("DSpark scratch prefix needs one distinct page per logical page")
        if any(page < 0 or page >= self.pages_per_slot for page in scratch_pages):
            raise ValueError("DSpark scratch prefix page is outside the scratch row")
        source_start = source_slot * self.draft_row_pages
        scratch_start = self.scratch_row * self.draft_row_pages
        if self._draft_narrow_rows:
            tail = min(pages, self.draft_row_pages)
            first = pages - tail
            kept = scratch_pages[first:]
            offs = torch.arange(first, pages, dtype=torch.long, device=self.device) % (
                self.draft_ring_pages
            )
            source_ids = source_start + offs
            scratch_ids = torch.tensor(
                [scratch_start + (page % (
                self.draft_ring_pages
            )) for page in kept],
                dtype=torch.long,
                device=self.device,
            )
            for cache in self._draft_kv_caches.values():
                cache[:, scratch_ids] = cache[:, source_ids]
        else:
            source_pages = torch.arange(
                source_start, source_start + pages, dtype=torch.long, device=self.device
            )
            scratch_page_tensor = torch.tensor(
                [scratch_start + page for page in scratch_pages], dtype=torch.long,
                device=self.device,
            )
            for cache in self._draft_kv_caches.values():
                cache[:, scratch_page_tensor] = cache[:, source_pages]
        self._scratch_valid_pages.update(scratch_pages)
        return True

    def restore_prefix_from_scratch(
        self,
        target_slot: int,
        kv_len: int,
        *,
        scratch_pages: tuple[int, ...],
    ) -> bool:
        """Restore a persistent draft prefix into a live target slot."""

        self._validate_slot(target_slot)
        kv_len = int(kv_len)
        pages = (kv_len + self.page_size - 1) // self.page_size
        if len(scratch_pages) != pages or len(set(scratch_pages)) != pages:
            raise ValueError("DSpark scratch prefix needs one distinct page per logical page")
        if any(page < 0 or page >= self.pages_per_slot for page in scratch_pages):
            raise ValueError("DSpark scratch prefix page is outside the scratch row")
        if not set(scratch_pages).issubset(self._scratch_valid_pages):
            return False
        target_start = target_slot * self.draft_row_pages
        scratch_start = self.scratch_row * self.draft_row_pages
        if self._draft_narrow_rows:
            tail = min(pages, self.draft_row_pages)
            first = pages - tail
            kept = scratch_pages[first:]
            offs = torch.arange(first, pages, dtype=torch.long, device=self.device) % (
                self.draft_ring_pages
            )
            target_ids = target_start + offs
            scratch_ids = torch.tensor(
                [scratch_start + (page % (
                self.draft_ring_pages
            )) for page in kept],
                dtype=torch.long,
                device=self.device,
            )
            for cache in self._draft_kv_caches.values():
                cache[:, target_ids] = cache[:, scratch_ids]
        else:
            target_pages = torch.arange(
                target_start, target_start + pages, dtype=torch.long, device=self.device
            )
            scratch_page_tensor = torch.tensor(
                [scratch_start + page for page in scratch_pages], dtype=torch.long,
                device=self.device,
            )
            for cache in self._draft_kv_caches.values():
                cache[:, target_pages] = cache[:, scratch_page_tensor]
        self._draft_kv_len[target_slot] = kv_len
        self._cached_prefix_len[target_slot] = 0
        return True

    def release_prefix_scratch(self, scratch_pages: tuple[int, ...]) -> None:
        self._scratch_valid_pages.difference_update(scratch_pages)

    def capture_cuda_graphs(self) -> None:
        """Capture DSpark's target verify and greedy draft graphs.

        A failed capture is loud and recorded, then the already-correct eager
        path remains available unless ``QSR_QWEN36_DSPARK_REQUIRE_CG=1``.
        The production comparison harness sets that variable so a run cannot
        accidentally report eager numbers as CUDA-Graph numbers.
        """

        if not self._use_cuda_graph or self._cg_captured:
            return
        from runtime.backends.qwen36_dspark_cudagraph import (
            Qwen36DSparkDraftBatchCudaGraph,
        )
        from runtime.backends.qwen36_mtp_cudagraph import attempt_mtp_cg_capture

        strict = getattr(self.draft_model, "is_dflash2", False) or (
            os.environ.get("QSR_QWEN36_DSPARK_REQUIRE_CG", "0") == "1"
        )
        draft_statuses: list[str] = []
        for batch_size in range(1, self.backend.num_slots + 1):

            def _capture_draft(batch_size: int = batch_size) -> None:
                graph = Qwen36DSparkDraftBatchCudaGraph(self, batch_size)
                graph.capture()
                self._draft_cg[batch_size] = graph

            status = attempt_mtp_cg_capture(
                f"dspark_draft_b{batch_size}", _capture_draft, strict=strict
            )
            self.cg_status[f"draft_b{batch_size}"] = status
            draft_statuses.append(status)
            if status == "failed":
                self._draft_cg.pop(batch_size, None)
        self.cg_status["draft"] = (
            "captured"
            if draft_statuses and all(s == "captured" for s in draft_statuses)
            else "failed"
        )

        if self.verify_mode == "compact":
            # Compact mode is intentionally a single ragged graph family.
            # Do not capture or dispatch a fixed-width graph here: doing so
            # makes request-local widths turn into the old grouping behavior.
            from runtime.backends.qwen36_mtp_cudagraph import (
                Qwen36MTPRaggedVerifyCudaGraph,
            )

            def _capture_ragged_verify() -> None:
                graph = Qwen36MTPRaggedVerifyCudaGraph(self)
                graph.capture()
                self._verify_ragged_cg = graph

            status = attempt_mtp_cg_capture(
                "dspark_verify_ragged", _capture_ragged_verify, strict=strict
            )
            self.cg_status["verify_ragged"] = status
            self.cg_status["verify"] = status
            if status == "failed":
                self._verify_ragged_cg = None
        else:

            def _capture_verify() -> None:
                from runtime.backends.qwen36_mtp_cudagraph import Qwen36MTPVerifyCudaGraph

                graph = Qwen36MTPVerifyCudaGraph(self)
                graph.capture()
                self._verify_cg = graph

            status = attempt_mtp_cg_capture("dspark_verify", _capture_verify, strict=strict)
            self.cg_status["verify"] = status
            if status == "failed":
                self._verify_cg = None

        # Width-specialized graphs are retained only for the legacy
        # cap-accept path. Compact mode above never populates this mapping.
        if (
            self.verify_mode == "cap-accept"
            and (self._confidence_threshold is not None or self._sps_table is not None)
            and self._verify_cg is not None
        ):
            from runtime.backends.qwen36_mtp_cudagraph import Qwen36MTPVerifyCudaGraph

            raw_widths = os.environ.get(
                "QSR_QWEN36_DSPARK_COMPACT_WIDTHS",
                ",".join(str(width) for width in range(self.k)),
            )
            try:
                widths = sorted(
                    {int(value.strip()) for value in raw_widths.split(",") if value.strip()}
                )
            except ValueError as exc:
                raise ValueError(
                    "QSR_QWEN36_DSPARK_COMPACT_WIDTHS must be a comma-separated "
                    f"list of integers, got {raw_widths!r}"
                ) from exc
            if any(width < 0 or width >= self.k for width in widths):
                raise ValueError(
                    "QSR_QWEN36_DSPARK_COMPACT_WIDTHS must contain widths in "
                    f"[0,{self.k - 1}], got {widths}"
                )
            for width in widths:

                def _capture_width(width: int = width) -> None:
                    graph = Qwen36MTPVerifyCudaGraph(self, verify_num_speculative_tokens=width)
                    graph.capture()
                    self._verify_cg_by_width[width] = graph

                width_status = attempt_mtp_cg_capture(
                    f"dspark_verify_w{width}", _capture_width, strict=strict
                )
                self.cg_status[f"verify_w{width}"] = width_status
                if width_status == "failed":
                    self._verify_cg_by_width.pop(width, None)
            self._verify_cg_by_width[self.k] = self._verify_cg
        elif self._verify_cg is not None:
            self._verify_cg_by_width[self.k] = self._verify_cg
        self._cg_captured = True

    def cuda_graphs_healthy(self) -> bool:
        # Match the MTP contract: an engine that did not attempt capture is
        # healthy, while a partial/failed capture is not.  This matters for
        # CPU-facing construction tests and for callers that intentionally
        # disable DSpark CG via the environment.
        return all(status == "captured" for status in self.cg_status.values())

    def _patch_attention(self) -> None:
        """Replace draft placeholders with the runtime attention path."""

        for attn in self._draft_attn_layers.values():
            attn.impl = SparkinferAttentionImpl(
                num_heads=attn.num_heads,
                head_size=attn.head_size,
                scale=attn.head_size**-0.5,
                num_kv_heads=attn.num_kv_heads,
                window_left=self._draft_window_left,
            )

        parents_by_name = {
            layer.self_attn.attn.layer_name: layer.self_attn
            for layer in self.draft_model.model.layers
        }

        def resolve_parent(layer_name: str) -> tuple[Any, str]:
            return parents_by_name[layer_name], "attn"

        replace_laguna_attention(
            self.draft_model,
            self._draft_attn_layers,
            self._draft_kv_caches,
            resolve_parent=resolve_parent,
            prefill_capacity_by_window_left={
                self._draft_window_left: (
                    self._draft_query_tokens,
                    self.draft_row_pages,
                ),
            },
            max_batch=self._draft_query_tokens,
        )

        if self.use_flashinfer_draft:
            first_attn = next(iter(self._draft_attn_layers.values()))
            self._flashinfer_draft_impl = FlashInferDSparkAttentionImpl(
                num_heads=first_attn.num_heads,
                head_size=first_attn.head_size,
                scale=first_attn.scale,
                num_kv_heads=first_attn.num_kv_heads,
                page_size=self.page_size,
                max_pages=self.draft_row_pages,
                num_tokens=self._draft_query_tokens,
                device=self.device,
                window_left=self._draft_window_left,
                kv_cache_dtype=first_attn.kv_cache_dtype,
                workspace_buffer=self._flashinfer_workspace_buffer,
            )
            for attn in self._draft_attn_layers.values():
                attn.impl = self._flashinfer_draft_impl

    def _page_table(self, slot: int) -> torch.Tensor:
        if slot < 0 or slot >= self.backend.num_slots:
            raise ValueError(f"invalid DSpark slot {slot}")
        base = slot * self.draft_row_pages
        width = self.draft_row_pages if self._draft_narrow_rows else self.pages_per_slot
        return torch.arange(
            base,
            base + width,
            dtype=torch.int32,
            device=self.device,
        ).view(1, -1)

    def _draft_attention_page_table(
        self, slot: int, kv_len: int, num_tokens: int
    ) -> tuple[torch.Tensor, int]:
        """Build the local paged view used by a DFlash sliding-window block."""

        page_table = self._page_table(slot)
        if not self._use_compact_draft_cache:
            return page_table, int(kv_len)
        local_len, first_page = compact_dflash_window(
            int(kv_len),
            window_size=self._draft_window_left + 1,
            page_size=self.page_size,
        )
        total_len = local_len + int(num_tokens)
        num_pages = (total_len + self.page_size - 1) // self.page_size
        if self._draft_narrow_rows:
            if num_pages > self.draft_row_pages:
                raise RuntimeError(
                    "DFlash compact page view exceeds the narrow draft row: "
                    f"slot={slot}, kv_len={kv_len}, pages={num_pages}, "
                    f"capacity={self.draft_row_pages}"
                )
            base = slot * self.draft_row_pages
            ring = (first_page + torch.arange(num_pages, device=self.device)) % (
                self.draft_ring_pages
            )
            compact = torch.zeros_like(page_table)
            compact[0, :num_pages] = base + ring
            return compact, local_len
        if first_page + num_pages > self.pages_per_slot:
            raise RuntimeError(
                "DFlash compact page view exceeds the draft slot: "
                f"slot={slot}, kv_len={kv_len}, local_len={local_len}, "
                f"first_page={first_page}, pages={num_pages}, capacity={self.pages_per_slot}"
            )
        compact = torch.zeros_like(page_table)
        compact[0, :num_pages].copy_(page_table[0, first_page : first_page + num_pages])
        return compact, local_len

    def _slot_mapping(
        self,
        slot: int,
        positions: torch.Tensor,
        *,
        position_start: int | None = None,
        position_count: int | None = None,
    ) -> torch.Tensor:
        positions = positions.to(device=self.device, dtype=torch.long).reshape(-1)
        if self._fast_slot_mapping and (position_start is not None or position_count is not None):
            if position_start is None or position_count is None:
                raise ValueError("position_start and position_count must be provided together")
            start = int(position_start)
            count = int(position_count)
            if count != positions.numel():
                raise ValueError(
                    "DSpark slot-mapping range does not match positions: "
                    f"count={count}, positions={positions.numel()}"
                )
            end = start + count
            if start < 0 or end > self.max_seq_len:
                raise RuntimeError(
                    f"DSpark draft position outside cache: slot={slot}, "
                    f"range=[{start}, {end - 1}], max_seq_len={self.max_seq_len}"
                )
        elif positions.numel() and (
            int(positions.min().item()) < 0 or int(positions.max().item()) >= self.max_seq_len
        ):
            raise RuntimeError(
                f"DSpark draft position outside cache: slot={slot}, "
                f"range=[{int(positions.min().item())}, {int(positions.max().item())}], "
                f"max_seq_len={self.max_seq_len}"
            )
        if self._draft_narrow_rows:
            page_ids = positions // self.page_size
            intra = positions - page_ids * self.page_size
            narrow_page = (page_ids + slot * self.pages_per_slot) % (
                self.draft_ring_pages
            )
            return (slot * self.draft_row_pages + narrow_page) * self.page_size + intra
        base = slot * self.pages_per_slot * self.page_size
        return base + positions

    def prepare_dspark_context_kv_metadata(
        self,
        *,
        slots: list[int],
        past_lens: list[int],
        verify_lens: list[int],
        host_positions: torch.Tensor,
        host_slot_mapping: torch.Tensor,
    ) -> None:
        """Fill stable host metadata for the fused ragged context epilogue.

        The target graph's auxiliary taps are compact request-major rows.  A
        valid row maps to the corresponding live DSpark slot and absolute
        position; graph-capacity tail rows map to the DSpark scratch row so
        the fixed-shape projector can run without ever touching live KV.
        """

        if not self.fuse_context_kv:
            raise RuntimeError("fused DSpark context KV metadata is disabled")
        if not (len(slots) == len(past_lens) == len(verify_lens)):
            raise ValueError("DSpark context metadata arguments must have equal lengths")
        capacity = len(slots) * (self.k + 1)
        if tuple(host_positions.shape) != (capacity,) or tuple(host_slot_mapping.shape) != (
            capacity,
        ):
            raise ValueError(
                "DSpark context metadata buffers must have shape "
                f"[{capacity}], got {tuple(host_positions.shape)} and "
                f"{tuple(host_slot_mapping.shape)}"
            )
        if any(int(length) < 1 or int(length) > self.k + 1 for length in verify_lens):
            raise ValueError(f"invalid DSpark verify lengths: {verify_lens}")

        host_positions.zero_()
        host_slot_mapping.zero_()
        compact_offset = 0
        row_stride = self.draft_row_pages * self.page_size
        for slot, past_len, verify_len in zip(slots, past_lens, verify_lens, strict=True):
            self._validate_slot(int(slot))
            past_len = int(past_len)
            verify_len = int(verify_len)
            for column in range(verify_len):
                position = past_len + column
                if position < 0 or position >= self.max_seq_len:
                    raise RuntimeError(
                        "DSpark context position outside cache: "
                        f"slot={slot}, position={position}, max_seq_len={self.max_seq_len}"
                    )
                index = compact_offset + column
                host_positions[index] = position
                if self._draft_narrow_rows:
                    pos = int(position)
                    page_ids = pos // self.page_size
                    intra = pos - page_ids * self.page_size
                    narrow_page = (page_ids + int(slot) * self.pages_per_slot) % (
                        self.draft_row_pages
                    )
                    host_slot_mapping[index] = (
                        int(slot) * self.draft_row_pages + narrow_page
                    ) * self.page_size + intra
                else:
                    host_slot_mapping[index] = int(slot) * row_stride + position
            compact_offset += verify_len

        scratch_base = self.scratch_row * row_stride
        # Narrow rows: epilogue scratch lives on the row's RESERVED last page,
        # contiguous and disjoint from every ring page.
        scratch_reserved_base = self.scratch_row * self.draft_row_pages * self.page_size + (
            (self.draft_row_pages - 1) * self.page_size
        )

        def _scratch_flat(scratch_position: int) -> int:
            if self._draft_narrow_rows:
                return scratch_reserved_base + scratch_position
            return scratch_base + scratch_position

        for index in range(compact_offset, capacity):
            # Invalid target rows are never consumed by ragged attention, but
            # they still pass through the fixed-shape MLP/projection body.
            # Give each one a unique scratch address to avoid scatter races.
            scratch_position = index - compact_offset
            host_positions[index] = scratch_position
            host_slot_mapping[index] = _scratch_flat(scratch_position)

    def inject_dspark_context_kv(
        self,
        target_taps: list[torch.Tensor],
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
    ) -> None:
        """Run the DSpark context projector inside a captured verify graph."""

        if not self.fuse_context_kv:
            raise RuntimeError("fused DSpark context KV injection is disabled")
        if not target_taps:
            raise RuntimeError("DSpark graph context epilogue returned no target taps")
        combined = torch.cat(target_taps, dim=-1)
        self.draft_model.precompute_and_store_context_kv(
            combined,
            context_positions,
            context_slot_mapping,
        )

    def _forward_draft(
        self,
        slot: int,
        anchor_token: int,
        kv_len: int,
        *,
        params: SamplingParams | None = None,
    ) -> list[int] | torch.Tensor:
        """Run one ``bonus + mask`` draft block and return gamma tokens."""

        if kv_len < 0 or kv_len + self._draft_query_tokens > self.max_seq_len:
            raise RuntimeError(
                f"DSpark draft block does not fit: kv_len={kv_len}, "
                f"query_tokens={self._draft_query_tokens}, gamma={self.k}, "
                f"max_seq_len={self.max_seq_len}"
            )
        graph = self._draft_cg.get(1)
        if graph is not None and (params is None or params.is_greedy):
            self._draft_probs.pop(slot, None)
            self._draft_sparse_probs.pop(slot, None)
            self.stats["draft_graph_replays"] += 1
            return self._replay_draft_graph(graph, slot, anchor_token, kv_len)
        input_ids = torch.full(
            (1, self._draft_query_tokens),
            self.draft_model.config.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        input_ids[0, 0] = int(anchor_token)
        positions = torch.arange(
            kv_len, kv_len + self._draft_query_tokens, dtype=torch.long, device=self.device
        )
        page_table, attention_kv_len = self._draft_attention_page_table(
            slot, kv_len, self._draft_query_tokens
        )
        slot_mapping = self._slot_mapping(
            slot,
            positions,
            position_start=kv_len,
            position_count=self._draft_query_tokens,
        )
        if self.use_flashinfer_draft:
            total_len = attention_kv_len + self._draft_query_tokens
            pages = (total_len + self.page_size - 1) // self.page_size
            last_page_len = total_len - (pages - 1) * self.page_size
            metadata = SparkinferAttnMetadata(
                mode="extend",
                page_table=page_table,
                cache_seqlens=torch.tensor([total_len], dtype=torch.int32, device=self.device),
                cu_seqlens_q=torch.tensor(
                    [0, self._draft_query_tokens], dtype=torch.int32, device=self.device
                ),
                num_actual_tokens=self._draft_query_tokens,
                window_left=self._draft_window_left,
                flashinfer_qo_indptr=torch.tensor(
                    [0, self._draft_query_tokens], dtype=torch.int32, device=self.device
                ),
                flashinfer_kv_indptr=torch.tensor(
                    [0, pages], dtype=torch.int32, device=self.device
                ),
                flashinfer_kv_indices=page_table[0, :pages],
                flashinfer_kv_last_page_len=torch.tensor(
                    [last_page_len], dtype=torch.int32, device=self.device
                ),
            )
        else:
            # SparkInfer's paged extend kernel is causal within each request,
            # so the fallback models the query rows as independent one-token
            # requests. DFlash2 is rejected before reaching this branch when
            # FlashInfer is unavailable because it needs non-causal attention.
            page_table = page_table.expand(self._draft_query_tokens, -1)
            metadata = SparkinferAttnMetadata(
                mode="extend",
                page_table=page_table,
                cache_seqlens=torch.tensor(
                    [kv_len + self._draft_query_tokens] * self._draft_query_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                cu_seqlens_q=torch.arange(
                    self._draft_query_tokens + 1, dtype=torch.int32, device=self.device
                ),
                num_actual_tokens=self._draft_query_tokens,
                window_left=self._draft_window_left,
            )
        attn_metadata = {name: metadata for name in self._draft_layer_names}
        slot_mappings = {name: slot_mapping for name in self._draft_layer_names}
        with bf_attn_context(attn_metadata, slot_mappings):
            draft_hidden = self.draft_model(input_ids, positions)
        draft_hidden = draft_hidden.reshape(1, self._draft_query_tokens, -1)
        if getattr(self.draft_model, "is_dflash2", False):
            # The first row is the anchor/noise row. DFlash2 proposes from the
            # following K masked rows, matching the reference block_size=K+1
            # contract.
            draft_hidden = draft_hidden[:, 1:, :]
        if params is None or params.is_greedy:

            def sample_fn(logits: torch.Tensor, _step: int) -> torch.Tensor:
                return logits.argmax(dim=-1)

            self._draft_probs.pop(slot, None)
            self._draft_sparse_probs.pop(slot, None)
            if hasattr(self.draft_model, "set_sampling_context"):
                self.draft_model.set_sampling_context(temperature=0.0, generator=None)
        else:
            generator = make_generator(params.seed, str(self.device))

            def sample_fn(logits: torch.Tensor, _step: int) -> torch.Tensor:
                return sample_from_logits(logits, params, generator=generator)

            if hasattr(self.draft_model, "set_sampling_context"):
                self.draft_model.set_sampling_context(
                    temperature=params.temperature,
                    generator=generator,
                )

        draft_tokens, corrected_logits, confidence = self.draft_model.sample_block(
            draft_hidden,
            anchor_tokens=torch.tensor([int(anchor_token)], dtype=torch.long, device=self.device),
            sampler=sample_fn,
            capture_confidence=self.capture_confidence,
        )
        if params is not None and not params.is_greedy:
            sparse_indices = getattr(self.draft_model, "last_candidate_indices", None)
            sparse_probs = getattr(self.draft_model, "last_candidate_probs", None)
            if sparse_indices is not None and sparse_probs is not None:
                self._draft_sparse_probs[slot] = (
                    sparse_indices[0].detach(),
                    sparse_probs[0].detach(),
                )
                self._draft_probs.pop(slot, None)
            else:
                self._draft_probs[slot] = compute_sampling_distribution(
                    corrected_logits[0], params
                ).detach()
        if confidence is None:
            self._draft_confidence.pop(slot, None)
        else:
            self._draft_confidence[slot] = confidence[0]
        self.stats["draft_forwards"] += 1
        self._draft_kv_len[slot] = max(self._draft_kv_len[slot], kv_len + self._draft_query_tokens)
        return [int(token) for token in draft_tokens[0].tolist()]

    def _replay_draft_graph(
        self, graph: Any, slot: int, anchor_token: int, kv_len: int
    ) -> torch.Tensor:
        if hasattr(graph, "batch_size"):
            tokens = graph.replay([slot], [anchor_token], [kv_len])
            tokens = tokens[0]
            confidence = graph.confidence
            if confidence is None:
                self._draft_confidence.pop(slot, None)
            else:
                self._draft_confidence[slot] = confidence[0]
        else:
            tokens = graph.replay(anchor_token, kv_len)
        self.backend.stats.setdefault("dspark_draft_graph_replays", 0)
        self.backend.stats["dspark_draft_graph_replays"] += 1
        self._draft_kv_len[slot] = max(
            self._draft_kv_len[slot], int(kv_len) + self._draft_query_tokens
        )
        return tokens

    @staticmethod
    def _decisions_from_device_accept(
        slots: list[int],
        accepted: torch.Tensor,
        committed: torch.Tensor,
        gamma: int,
    ) -> dict[int, dict[str, Any]]:
        """Materialize the compact graph epilogue at the scheduler boundary."""

        # The graph has already done argmax and prefix scanning.  Keep one
        # small host synchronization for the scheduler's unavoidable output
        # publication, but do not pull logits/predictions or run another
        # GPU-side cumulative scan outside the graph.
        combined = torch.cat([accepted.to(dtype=torch.long).unsqueeze(1), committed], dim=1)
        rows = combined.tolist()
        decisions: dict[int, dict[str, Any]] = {}
        for slot, row in zip(slots, rows, strict=True):
            num_accepted = int(row[0])
            if not 0 <= num_accepted <= gamma:
                raise RuntimeError(
                    f"DSpark graph accept length outside [0,{gamma}]: {num_accepted}"
                )
            decisions[slot] = {
                "num_accepted": num_accepted,
                "committed": [int(token) for token in row[1 : num_accepted + 2]],
                "rejected_at": num_accepted if num_accepted < gamma else None,
            }
        return decisions

    def _apply_thinking_forces_to_logits(
        self,
        logits: torch.Tensor,
        slots: list[int],
        *,
        row_starts: list[int],
        row_limits: list[int],
        force_positions: Mapping[int, int],
        force_token_ids: Mapping[int, int],
    ) -> bool:
        """Bias budget-boundary rows while retaining the batched verify graph.

        The graph's accept epilogue is captured against the unconstrained
        logits.  A thinking budget can change the first rejected position, so
        forced rounds must recompute the small greedy accept decision from the
        graph-owned logits after applying the one-entry bias.  This is a rare
        boundary path; ordinary DSpark rounds continue to use the fused device
        accept outputs without this extra reduction.

        ``row_starts``/``row_limits`` describe request-major layouts for both
        fixed-width ``[B, Q, V]`` output (after flattening) and compact ragged
        ``[sum(Q), V]`` output.
        """
        if not force_positions:
            return False
        if set(force_positions) != set(force_token_ids):
            raise ValueError("thinking force positions and token ids must cover the same slots")
        if len(row_starts) != len(slots) or len(row_limits) != len(slots):
            raise ValueError("DSpark thinking force row metadata must match slots")
        if logits.ndim == 3:
            vocab_size = int(logits.shape[-1])
            flat_logits = logits.reshape(-1, vocab_size)
        elif logits.ndim == 2:
            vocab_size = int(logits.shape[-1])
            flat_logits = logits
        else:
            raise ValueError(
                "DSpark thinking force expects rank-2 or rank-3 logits, "
                f"got shape={tuple(logits.shape)}"
            )

        flat_rows: list[int] = []
        forced_tokens: list[int] = []
        for index, slot in enumerate(slots):
            if slot not in force_positions:
                continue
            position = int(force_positions[slot])
            token_id = int(force_token_ids[slot])
            if not 0 <= position < int(row_limits[index]):
                raise ValueError(
                    "thinking force position is outside the DSpark verify rows: "
                    f"slot={slot}, position={position}, rows={row_limits[index]}"
                )
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    "thinking force token is outside the DSpark model vocabulary: "
                    f"slot={slot}, token={token_id}, vocab={vocab_size}"
                )
            flat_rows.append(int(row_starts[index]) + position)
            forced_tokens.append(token_id)

        if not flat_rows:
            return False
        row_indices = torch.tensor(flat_rows, dtype=torch.long, device=logits.device)
        token_indices = torch.tensor(forced_tokens, dtype=torch.long, device=logits.device)
        flat_logits.index_put_(
            (row_indices, token_indices),
            flat_logits.new_full((len(flat_rows),), 1.0e9),
        )
        self.backend.stats.setdefault("dspark_thinking_force_batched_replays", 0)
        self.backend.stats["dspark_thinking_force_batched_replays"] += 1
        return True

    def _full_verify_drafts(
        self,
        slots: list[int],
        anchors: Mapping[int, int],
        drafts_by_slot: Mapping[int, list[int] | torch.Tensor],
    ) -> torch.Tensor:
        """Materialize ``anchor + K drafts`` for a forced accept reduction."""
        full = torch.empty((len(slots), self.k + 1), dtype=torch.long, device=self.device)
        full[:, 0] = torch.tensor(
            [int(anchors[slot]) for slot in slots], dtype=torch.long, device=self.device
        )
        for row, slot in enumerate(slots):
            draft = drafts_by_slot[slot]
            if isinstance(draft, torch.Tensor):
                if draft.ndim == 2 and tuple(draft.shape) == (1, self.k):
                    draft = draft[0]
                if draft.ndim != 1 or draft.numel() != self.k:
                    raise ValueError(
                        "DSpark forced verify drafts must have shape [K] or [1, K]; "
                        f"got {tuple(draft.shape)}"
                    )
                full[row, 1:].copy_(draft.to(device=self.device, dtype=torch.long))
            else:
                values = [int(token) for token in draft]
                if len(values) != self.k:
                    raise ValueError(
                        f"DSpark forced verify received {len(values)} drafts; expected {self.k}"
                    )
                full[row, 1:] = torch.tensor(values, dtype=torch.long, device=self.device)
        return full

    def _sync_target_hidden(
        self,
        slot: int,
        target_taps: list[torch.Tensor],
        *,
        position_offset: int,
    ) -> None:
        """Inject target tap rows into the draft cache at absolute positions."""

        dump_path = os.environ.get("QSR_DSPARK_DUMP_TAPS")
        if dump_path and not os.path.exists(dump_path):
            # A bounded, opt-in diagnostic seam for cross-engine semantic
            # comparison.  Dump only the first few rows: saving a full 128K
            # prompt here would turn a hidden-state check into a multi-GB I/O
            # experiment.  The vLLM comparator uses the same row cap.
            row_cap = max(1, int(os.environ.get("QSR_DSPARK_DUMP_TAPS_ROWS", "8")))
            torch.save(
                [tap.detach()[..., :row_cap, :].cpu() for tap in target_taps],
                dump_path,
            )

        all_layers_path = os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS")
        all_layers = getattr(self.model.model, "_debug_last_all_layer_hidden", None)
        if all_layers_path and all_layers_path != dump_path and all_layers:
            if not os.path.exists(all_layers_path):
                torch.save([layer.cpu() for layer in all_layers], all_layers_path)
            # The diagnostic is intentionally one-shot.  Holding the device
            # tensors after the dump would pin a full layer-row snapshot until
            # the next request and obscure the production memory profile.
            self.model.model._debug_last_all_layer_hidden = None

        combined, query_len = _flatten_target_taps(
            target_taps,
            expected_features=self.draft_model.model.fc.input_size,
        )
        if query_len <= 0:
            return
        positions = torch.arange(
            position_offset,
            position_offset + query_len,
            dtype=torch.long,
            device=self.device,
        )
        slot_mapping = self._slot_mapping(
            slot,
            positions,
            position_start=position_offset,
            position_count=query_len,
        )
        self.draft_model.precompute_and_store_context_kv(
            combined,
            positions,
            slot_mapping,
        )
        self._draft_kv_len[slot] = max(self._draft_kv_len[slot], position_offset + query_len)
        self.stats["target_hidden_injections"] += query_len

    def sync_prefill_context(
        self,
        slot: int,
        target_taps: list[torch.Tensor],
        *,
        position_offset: int,
    ) -> None:
        self._sync_target_hidden(slot, target_taps, position_offset=position_offset)
        self._spec_rows.sync_from_live(slot)
        self._spec_state_col[slot] = 0

    def sync_prefill_context_batch(
        self,
        slots: list[int],
        target_taps: list[torch.Tensor],
        *,
        position_offsets: list[int],
        query_len: int,
    ) -> None:
        """Inject one homogeneous target-prefill batch into draft KV.

        The target batch returns each selected tap as ``[B, Q, H]``.  Pack
        those rows once and run the draft projector/KV scatter once, matching
        SGLang's ``commit_hidden`` boundary.  The serial method remains the
        fallback for ragged admissions and for callers with a single slot.
        """

        if not slots:
            return
        if len(position_offsets) != len(slots):
            raise ValueError("DSpark batch prefill offsets must match slots")
        if query_len <= 0:
            raise ValueError("DSpark batch prefill query_len must be positive")
        batch = len(slots)
        for tap in target_taps:
            if tap.ndim != 3 or tap.shape[0] != batch or tap.shape[1] != query_len:
                raise ValueError(
                    "DSpark batch prefill taps must have shape "
                    f"[B={batch}, Q={query_len}, H], got {tuple(tap.shape)}"
                )

        self._sync_target_hidden_batch(
            slots,
            target_taps,
            position_offsets=position_offsets,
            accepted_counts=[query_len] * batch,
        )
        for slot in slots:
            self._spec_rows.sync_from_live(slot)
            self._spec_state_col[slot] = 0

    def draft_after_prefill(
        self, slot: int, anchor_token: int, *, params: SamplingParams | None = None
    ) -> list[int] | torch.Tensor:
        return self._forward_draft(
            slot,
            anchor_token,
            self.backend.pool.slot_state(slot).num_tokens_seen,
            params=params,
        )

    def reset_slot(self, slot: int) -> None:
        self._validate_slot(slot)
        self._draft_probs.pop(slot, None)
        self._draft_sparse_probs.pop(slot, None)
        self._draft_confidence.pop(slot, None)
        self._spec_rows.reset_slot(slot)
        self._spec_state_col[slot] = 0
        self._draft_kv_len[slot] = 0

    def _verify_widths(self, slots: list[int]) -> list[int]:
        """Choose SGLang-style compact verify widths for the active slots."""

        if self.verify_mode not in {"cap-accept", "compact"}:
            return [self.k] * len(slots)
        if not self._dynamic_planner:
            # Full-width static verify is still request-major ragged: every
            # request contributes K+1 rows to the one batch graph.  Avoid the
            # confidence transfer/planner entirely when no dynamic policy was
            # requested.
            return [self.k] * len(slots)
        rows: list[list[float]] = []
        for slot in slots:
            confidence = self._draft_confidence.get(slot)
            if confidence is None:
                rows.append([1.0] * self.k)
                continue
            if confidence.ndim != 1 or confidence.numel() != self.k:
                raise RuntimeError(
                    "DSpark confidence row has the wrong shape: "
                    f"slot={slot}, shape={tuple(confidence.shape)}, K={self.k}"
                )
            # This is the only host read in the compact planner.  SGLang
            # overlaps it with the next scheduler step; local single-process
            # serving has no overlap worker, so keep the transfer to the tiny
            # K-element confidence row and never pull logits/tokens back.
            rows.append(confidence.float().cpu().tolist())
        if self._sps_table is not None:
            survival = [survival_prefix(row, epsilon=0.0) for row in rows]
            decision = compute_verify_token_budget(
                survival,
                sps_table=self._sps_table,
                min_verify_len=self._min_verify_len,
                max_verify_len=self._max_verify_len,
                survival_eps=self._survival_eps,
            )
            widths = schedule_verify_widths_topk(
                rows,
                budget=decision.budget,
                min_verify_len=self._min_verify_len,
                max_verify_len=self._max_verify_len,
                survival_eps=self._survival_eps,
            )
            self.backend.stats["dspark_verify_budget"] = decision.budget
            self.backend.stats["dspark_verify_predicted_step_us"] = int(
                (decision.predicted_step_seconds or 0.0) * 1_000_000
            )
            return widths
        if self._confidence_threshold is None:
            return [self.k] * len(slots)
        # Keep the threshold knob as a small debugging policy.  Production
        # compact scheduling should use the profiled SPS objective above.
        return choose_verify_widths(
            rows,
            max_width=self.k,
            survival_threshold=self._confidence_threshold,
        )

    @staticmethod
    def _apply_accept_caps(
        decisions: dict[int, dict[str, Any]], caps: Mapping[int, int], gamma: int
    ) -> None:
        """Apply SGLang CAP_ACCEPT's cutoff after a full target verify.

        The full target graph has already produced every position's greedy
        prediction.  If the confidence planner caps a request before the
        actual rejection, the draft token at the cap equals that prediction;
        truncating the graph epilogue therefore has the same commit/bonus
        semantics as SGLang's ``cutoff_verify_lens`` without another logits
        readback or target pass.
        """

        for slot, decision in decisions.items():
            cap = min(max(int(caps.get(slot, gamma)), 0), gamma)
            accepted = int(decision["num_accepted"])
            if accepted <= cap:
                continue
            decision["num_accepted"] = cap
            decision["committed"] = decision["committed"][: cap + 1]
            decision["rejected_at"] = cap

    def round(
        self,
        slot: int,
        anchor_token: int,
        drafts: list[int] | torch.Tensor,
        *,
        params: SamplingParams | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
        _accept_cap: int | None = None,
        thinking_force_position: int | None = None,
        thinking_force_token_id: int | None = None,
    ) -> dict[str, Any]:
        """Verify one DSpark block, commit its accepted prefix, and redraft."""

        draft_row: torch.Tensor | None = None
        draft_values: list[int] | None = None
        if isinstance(drafts, torch.Tensor):
            if drafts.ndim == 2 and tuple(drafts.shape) == (1, self.k):
                draft_row = drafts[0]
            elif drafts.ndim == 1 and drafts.numel() == self.k:
                draft_row = drafts
            else:
                raise ValueError(
                    "DSpark round device drafts must have shape [K] or [1, K]; "
                    f"got {tuple(drafts.shape)}"
                )
            if draft_row.device != self.device:
                draft_row = draft_row.to(device=self.device)
        else:
            draft_values = [int(token) for token in drafts]
            if len(draft_values) != self.k:
                raise ValueError(
                    f"DSpark round received {len(draft_values)} drafts; expected {self.k}"
                )

        state = self.backend.pool.slot_state(slot)
        past_len = state.num_tokens_seen
        verify_tokens = self._verify_tokens_buf[slot]
        if draft_row is not None:
            # Same-stream D2D copies preserve ordering with the draft graph;
            # no host readback is needed before target verify starts.
            host_anchor = self._verify_host[slot][0, :1]
            host_anchor[0] = int(anchor_token)
            verify_tokens[0, :1].copy_(host_anchor, non_blocking=True)
            verify_tokens[0, 1:].copy_(draft_row, non_blocking=True)
        else:
            host_view = self._verify_host_views[slot]
            host_view.reshape(-1)[:] = [int(anchor_token), *draft_values]
            verify_tokens.copy_(self._verify_host[slot], non_blocking=True)

        round_profile.begin_round()
        round_profile.phase("setup")
        if self._verify_cg is not None:
            graph_result = self._verify_cg.replay_with_aux(
                [slot],
                verify_tokens,
                [past_len],
                return_accept=self.capture_device_accept,
            )
            all_hiddens, logits_batch, target_taps = graph_result[:3]
            all_logits = logits_batch[0]
            gdn_snapshots = None
            self.stats["verify_graph_replays"] += 1
            self.backend.stats.setdefault("dspark_verify_graph_replays", 0)
            self.backend.stats["dspark_verify_graph_replays"] += 1
        else:
            all_hiddens, gdn_snapshots, target_taps = self.model.verify_forward(
                verify_tokens,
                state,
                spec_state_rows=self._spec_rows.rows_for_slot(slot),
                capture_aux_hidden_states=True,
            )
            all_logits = self.model.compute_logits(all_hiddens)[0]
        round_profile.phase("verify_replay")

        if thinking_force_position is not None:
            if thinking_force_token_id is None:
                raise ValueError("thinking force position requires a token id")
            if not 0 <= thinking_force_position < all_logits.shape[0]:
                raise ValueError(
                    "thinking force position is outside the DSpark verify block: "
                    f"position={thinking_force_position}, rows={all_logits.shape[0]}"
                )
            if not 0 <= thinking_force_token_id < all_logits.shape[-1]:
                raise ValueError(
                    "thinking force token is outside the model vocabulary: "
                    f"token={thinking_force_token_id}, vocab={all_logits.shape[-1]}"
                )
            forced_row = all_logits[thinking_force_position]
            # The force is applied after the verify graph; touch only the
            # selected entry instead of rewriting the whole vocabulary row.
            forced_row[thinking_force_token_id] = 1.0e9

        sampled = params is not None and not params.is_greedy
        if sampled:
            if draft_values is None:
                # Non-greedy draft graphs are intentionally eager today, but
                # keep this fallback correct if a caller supplies a device
                # row from a custom sampler.
                draft_values = [int(token) for token in draft_row.tolist()]
            draft_probs = self._draft_probs.pop(slot, None)
            target_probs = compute_sampling_distribution(all_logits, params)
            sparse = self._draft_sparse_probs.pop(slot, None)
            if sparse is not None:
                draft_indices, sparse_probs = sparse
                decision = sample_accept_reject_sparse(
                    draft_values,
                    draft_indices,
                    sparse_probs,
                    target_probs,
                    generator=make_generator(params.seed, str(self.device)),
                )
            else:
                if draft_probs is None:
                    raise RuntimeError("DSpark sampled verify has no draft probability block")
                decision = sample_accept_reject(
                    draft_values,
                    draft_probs,
                    target_probs,
                    generator=make_generator(params.seed, str(self.device)),
                )
            self.stats["sampled_rounds"] += 1
        else:
            # Greedy acceptance is vectorized on device and drains only the
            # tiny combined decision tensor.  In particular, the K draft
            # tokens and target verify input never cross back to Python.
            if (
                thinking_force_position is None
                and self.capture_device_accept
                and self._verify_cg is not None
            ):
                decision = self._decisions_from_device_accept(
                    [slot], graph_result[3], graph_result[4], self.k
                )[slot]
            else:
                decision = determine_accept_reject_batch(
                    [slot], verify_tokens, all_logits.unsqueeze(0), self.k
                )[slot]
        if _accept_cap is not None:
            self._apply_accept_caps({slot: decision}, {slot: _accept_cap}, self.k)
        round_profile.phase("accept_decision")

        accepted = int(decision["num_accepted"])
        committed: list[int] = decision["committed"]
        new_anchor = int(committed[-1])

        self.model.commit_verify(
            state,
            gdn_snapshots,
            past_len=past_len,
            accepted_count=accepted + 1,
        )
        self._spec_rows.activate(slot, accepted)
        self._spec_state_col[slot] = accepted
        self.backend.pool.slot_kv_len[slot] = state.num_tokens_seen
        self.backend.pool.slot_committed_tokens[slot].extend(committed)
        self.backend._maybe_checkpoint(slot)  # noqa: SLF001 - backend-owned policy
        round_profile.phase("commit")

        accepted_taps = [tap[:, : accepted + 1] for tap in target_taps]
        self._sync_target_hidden(slot, accepted_taps, position_offset=past_len)
        round_profile.phase("target_hidden_sync")
        next_drafts = self._forward_draft(
            slot,
            new_anchor,
            state.num_tokens_seen,
            params=params,
        )
        round_profile.phase("draft")

        self.stats["rounds"] += 1
        output: dict[str, Any] = {
            "committed": committed,
            "num_accepted": accepted,
            "next_anchor": new_anchor,
            "next_draft_tokens": next_drafts,
        }
        if return_logprobs:
            output["logprobs"] = [
                compute_logprobs(
                    all_logits[position : position + 1],
                    [committed[position]],
                    top_k=top_logprobs,
                )[0]
                for position in range(len(committed))
            ]
        round_profile.end_round(label="dspark_round")
        return output

    def _sync_target_hidden_batch(
        self,
        slots: list[int],
        target_taps: list[torch.Tensor],
        *,
        position_offsets: list[int],
        accepted_counts: list[int],
        verify_lens: list[int] | None = None,
    ) -> None:
        """Inject accepted target rows for all requests in one projection.

        SGLang's ``TargetHiddenKvInjector.inject_ragged`` packs the accepted
        prefixes into one request-major matrix before projecting them into the
        draft KV family.  ``verify_lens`` identifies the request-local spans
        when the target graph itself is compact/ragged; the legacy fixed-width
        caller leaves it unset.
        """

        if not (len(slots) == len(position_offsets) == len(accepted_counts)):
            raise ValueError("DSpark batch hidden injection arguments must have equal lengths")
        if not slots:
            return
        combined = _flatten_target_taps_ragged(
            target_taps,
            batch_size=len(slots),
            accepted_counts=accepted_counts,
            expected_features=self.draft_model.model.fc.input_size,
            verify_lens=verify_lens,
        )
        total_rows = sum(int(count) for count in accepted_counts)
        if combined.shape[0] != total_rows:
            raise RuntimeError(
                "DSpark compact target-tap row count mismatch: "
                f"expected {total_rows}, got {combined.shape[0]}"
            )
        if total_rows <= 0:
            return

        position_parts: list[torch.Tensor] = []
        mapping_parts: list[torch.Tensor] = []
        for slot, offset, count in zip(slots, position_offsets, accepted_counts, strict=True):
            positions = torch.arange(
                int(offset),
                int(offset) + int(count),
                dtype=torch.long,
                device=self.device,
            )
            position_parts.append(positions)
            mapping_parts.append(
                self._slot_mapping(
                    slot,
                    positions,
                    position_start=int(offset),
                    position_count=int(count),
                )
            )
        positions = torch.cat(position_parts)
        slot_mapping = torch.cat(mapping_parts)
        self.draft_model.precompute_and_store_context_kv(
            combined,
            positions,
            slot_mapping,
        )
        for slot, offset, count in zip(slots, position_offsets, accepted_counts, strict=True):
            self._draft_kv_len[slot] = max(self._draft_kv_len[slot], int(offset) + int(count))
        self.stats["target_hidden_injections"] += total_rows

    def _forward_draft_batch(
        self,
        slots: list[int],
        anchors: list[int],
        kv_lens: list[int],
        *,
        params_per_slot: Mapping[int, SamplingParams] | None = None,
        device_anchors: torch.Tensor | None = None,
        device_kv_lens: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor | list[int]]:
        """Produce one draft block per slot using the matching ``B*K`` graph.

        The captured graph is greedy by design.  A mixed or sampled batch is
        routed through the already-correct per-slot eager path; this keeps
        sampling semantics exact while the common serving case gets the
        SGLang-shaped batched proposer.
        """

        if not slots:
            return {}
        if (device_anchors is None) != (device_kv_lens is None):
            raise ValueError("DSpark device draft metadata must include anchors and kv_lens")
        if device_anchors is not None and (
            tuple(device_anchors.shape) != (len(slots),)
            or device_kv_lens is None
            or tuple(device_kv_lens.shape) != (len(slots),)
        ):
            raise ValueError("DSpark device draft metadata must have shape [batch]")
        params_per_slot = params_per_slot or {}
        all_greedy = all(
            params_per_slot.get(slot) is None or params_per_slot[slot].is_greedy for slot in slots
        )
        graph = self._draft_cg.get(len(slots)) if all_greedy else None
        if graph is not None:
            if device_anchors is None:
                tokens = graph.replay(slots, anchors, kv_lens)
            else:
                assert device_kv_lens is not None
                tokens = graph.replay_device(
                    slots,
                    device_anchors,
                    device_kv_lens,
                    host_kv_lens=kv_lens,
                )
            confidence = graph.confidence
            if confidence is None:
                for slot in slots:
                    self._draft_confidence.pop(slot, None)
            else:
                for row, slot in enumerate(slots):
                    self._draft_confidence[slot] = confidence[row]
            self.stats["draft_graph_replays"] += 1
            self.backend.stats.setdefault("dspark_draft_graph_replays", 0)
            self.backend.stats["dspark_draft_graph_replays"] += 1
            for slot, kv_len in zip(slots, kv_lens, strict=True):
                self._draft_kv_len[slot] = max(
                    self._draft_kv_len[slot], int(kv_len) + self._draft_query_tokens
                )
            return {slot: tokens[row] for row, slot in enumerate(slots)}

        return {
            slot: self._forward_draft(
                slot,
                int(anchor),
                int(kv_len),
                params=params_per_slot.get(slot),
            )
            for slot, anchor, kv_len in zip(slots, anchors, kv_lens, strict=True)
        }

    def _round_batch_ragged(
        self,
        slots: list[int],
        anchors: Mapping[int, int],
        drafts_by_slot: Mapping[int, list[int] | torch.Tensor],
        *,
        params_per_slot: Mapping[int, SamplingParams],
        return_logprobs: bool,
        top_logprobs: int,
        thinking_force_positions: Mapping[int, int],
        thinking_force_token_ids: Mapping[int, int],
    ) -> dict[int, dict[str, Any]]:
        """Run one unified request-major ragged DSpark verify round."""

        forced = bool(thinking_force_positions)
        # A force can land at any position in the K+1 target block.  Keep the
        # whole active batch at full width for that rare boundary round so the
        # corrected decision can use the same vectorized K-token reducer as a
        # normal fixed-width verify.  The steady-state planner is untouched.
        widths = [self.k] * len(slots) if forced else self._verify_widths(slots)
        verify_lens = [int(width) + 1 for width in widths]
        for width in widths:
            self.backend.stats["dspark_verify_width_histogram"][width] += 1

        graph = self._verify_ragged_cg
        all_greedy = all(
            params_per_slot.get(slot) is None or params_per_slot[slot].is_greedy for slot in slots
        )
        if graph is None or not self.capture_device_accept or not all_greedy:
            # This is only a capture-failure/sampling fallback.  It is kept
            # serial and explicit; compact mode never groups requests by a
            # common width or recursively launches one fixed-width batch.
            return {
                slot: self.round(
                    slot,
                    anchors[slot],
                    drafts_by_slot[slot],
                    params=params_per_slot.get(slot),
                    return_logprobs=return_logprobs,
                    top_logprobs=top_logprobs,
                    _accept_cap=width,
                    thinking_force_position=thinking_force_positions.get(slot),
                    thinking_force_token_id=thinking_force_token_ids.get(slot),
                )
                for slot, width in zip(slots, widths, strict=True)
            }

        states = [self.backend.pool.slot_state(slot) for slot in slots]
        past_lens = [state.num_tokens_seen for state in states]
        anchors_list = [int(anchors[slot]) for slot in slots]
        drafts = [drafts_by_slot[slot] for slot in slots]
        round_profile.begin_round()
        round_profile.phase("setup")
        graph_result = graph.replay_with_aux(
            slots,
            anchors_list,
            drafts,
            past_lens,
            verify_lens,
            return_accept=True,
        )
        _all_hiddens, logits_flat, target_taps, accepted, committed = graph_result
        self.stats["verify_graph_replays"] += 1
        self.backend.stats.setdefault("dspark_verify_graph_replays", 0)
        self.backend.stats["dspark_verify_graph_replays"] += 1
        compact_starts: list[int] = []
        compact_offset = 0
        for verify_len in verify_lens:
            compact_starts.append(compact_offset)
            compact_offset += int(verify_len)
        force_applied = self._apply_thinking_forces_to_logits(
            logits_flat,
            slots,
            row_starts=compact_starts,
            row_limits=verify_lens,
            force_positions=thinking_force_positions,
            force_token_ids=thinking_force_token_ids,
        )
        if force_applied:
            full_drafts = self._full_verify_drafts(slots, anchors, drafts_by_slot)
            decisions = determine_accept_reject_batch(
                slots,
                full_drafts,
                logits_flat[: len(slots) * (self.k + 1)],
                self.k,
            )
        else:
            decisions = self._decisions_from_device_accept(slots, accepted, committed, self.k)
        round_profile.phase("accept_decision")

        accepted_counts: list[int] = []
        committed_by_slot: dict[int, list[int]] = {}
        for slot, state, past_len in zip(slots, states, past_lens, strict=True):
            decision = decisions[slot]
            accepted_count = int(decision["num_accepted"])
            committed_row = [int(token) for token in decision["committed"]]
            self.model.commit_verify(
                state,
                None,
                past_len=past_len,
                accepted_count=accepted_count + 1,
            )
            self._spec_rows.activate(slot, accepted_count)
            self._spec_state_col[slot] = accepted_count
            self.backend.pool.slot_kv_len[slot] = state.num_tokens_seen
            self.backend.pool.slot_committed_tokens[slot].extend(committed_row)
            self.backend._maybe_checkpoint(slot)  # noqa: SLF001 - backend-owned policy
            accepted_counts.append(accepted_count + 1)
            committed_by_slot[slot] = committed_row
        round_profile.phase("commit")

        if graph.context_kv_fused and not force_applied:
            # The verify CUDA graph already projected/scattered every
            # request-local candidate row into the draft cache.  Only the
            # accepted prefix is live for the next draft block; rejected
            # candidate rows remain harmless scratch/stale cache data.
            self.stats["target_hidden_injections"] += sum(accepted_counts)
            for slot, past_len, accepted_count in zip(
                slots, past_lens, accepted_counts, strict=True
            ):
                self._draft_kv_len[slot] = max(
                    self._draft_kv_len[slot], int(past_len) + int(accepted_count)
                )
        else:
            self._sync_target_hidden_batch(
                slots,
                target_taps,
                position_offsets=past_lens,
                accepted_counts=accepted_counts,
                verify_lens=verify_lens,
            )
        round_profile.phase("target_hidden_sync")

        next_anchors = [committed_by_slot[slot][-1] for slot in slots]
        next_lens = [self.backend.pool.slot_state(slot).num_tokens_seen for slot in slots]
        if force_applied:
            # The graph-owned accept outputs were produced before the budget
            # bias.  Host decisions above are authoritative for this boundary
            # round, so do not feed stale graph metadata into the next draft.
            device_next_anchors = device_next_lens = None
        else:
            device_next_anchors, device_next_lens = graph.next_draft_inputs(
                len(slots), accepted, committed
            )
        next_drafts_by_slot = self._forward_draft_batch(
            slots,
            next_anchors,
            next_lens,
            params_per_slot=params_per_slot,
            device_anchors=device_next_anchors,
            device_kv_lens=device_next_lens,
        )
        round_profile.phase("draft_batch")
        self.stats["rounds"] += len(slots)

        output: dict[int, dict[str, Any]] = {}
        compact_offset = 0
        for row, (slot, verify_len) in enumerate(zip(slots, verify_lens, strict=True)):
            decision = decisions[slot]
            committed_row = committed_by_slot[slot]
            item: dict[str, Any] = {
                "committed": committed_row,
                "num_accepted": int(decision["num_accepted"]),
                "next_anchor": committed_row[-1],
                "next_draft_tokens": next_drafts_by_slot[slot],
            }
            if return_logprobs:
                logits = logits_flat[compact_offset : compact_offset + verify_len]
                item["logprobs"] = [
                    compute_logprobs(
                        logits[position : position + 1],
                        [committed_row[position]],
                        top_k=top_logprobs,
                    )[0]
                    for position in range(len(committed_row))
                ]
            output[slot] = item
            compact_offset += verify_len
        round_profile.end_round(label=f"dspark_ragged_round_b{len(slots)}")
        return output

    def round_batch(
        self,
        slots: list[int],
        anchors: Mapping[int, int],
        drafts_by_slot: Mapping[int, list[int] | torch.Tensor],
        *,
        params_per_slot: Mapping[int, SamplingParams] | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
        _verify_gamma: int | None = None,
        thinking_force_positions: Mapping[int, int] | None = None,
        thinking_force_token_ids: Mapping[int, int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Verify and redraft one DSpark round for the active request batch.

        Compact mode follows SGLang's single request-major ragged verify
        contract.  The legacy fixed-width path remains available for static
        and cap-accept deployments, but compact mode never partitions requests
        by accepted width.
        """

        if not slots:
            return {}
        if len(set(slots)) != len(slots):
            raise ValueError(f"DSpark batch contains duplicate slots: {slots}")
        if any(slot not in anchors or slot not in drafts_by_slot for slot in slots):
            raise ValueError("DSpark batch is missing an anchor or draft row")

        params_per_slot = params_per_slot or {}
        position_keys = set(thinking_force_positions or ())
        token_keys = set(thinking_force_token_ids or ())
        if position_keys != token_keys:
            raise ValueError("thinking force positions and token ids must cover the same slots")
        forced_slots = position_keys.intersection(slots)
        if forced_slots != position_keys:
            raise ValueError("thinking force maps contain a slot outside the active DSpark batch")
        if _verify_gamma is None and self.verify_mode == "compact":
            return self._round_batch_ragged(
                slots,
                anchors,
                drafts_by_slot,
                params_per_slot=params_per_slot,
                return_logprobs=return_logprobs,
                top_logprobs=top_logprobs,
                thinking_force_positions=thinking_force_positions or {},
                thinking_force_token_ids=thinking_force_token_ids or {},
            )

        caps: dict[int, int] | None = None
        if forced_slots:
            # Force positions are defined against the full K+1 target verify
            # block.  A boundary round must not be shortened by the dynamic
            # width planner before the constrained row is visible.
            verify_gamma = self.k
        elif _verify_gamma is None and self.verify_mode in {"cap-accept", "compact"}:
            widths = self._verify_widths(slots)
            if self.verify_mode == "compact":
                raise RuntimeError("compact DSpark dispatch must enter _round_batch_ragged")
            else:
                caps = dict(zip(slots, widths, strict=True))

        if not forced_slots:
            verify_gamma = self.k if _verify_gamma is None else int(_verify_gamma)
        if not 0 <= verify_gamma <= self.k:
            raise ValueError(f"DSpark verify width must be in [0,{self.k}], got {verify_gamma}")
        self.backend.stats["dspark_verify_width_histogram"][verify_gamma] += len(slots)
        requested_verify_gamma = verify_gamma
        verify_graph = self._verify_cg_by_width.get(verify_gamma)
        if verify_graph is None and verify_gamma != self.k:
            # A compact width without a captured graph is a performance
            # fallback, never a correctness change.  Use the proven full
            # graph for that round rather than silently dropping speculation.
            verify_gamma = self.k
            verify_graph = self._verify_cg_by_width.get(self.k)
        if verify_graph is None or any(
            params_per_slot.get(slot) is not None and not params_per_slot[slot].is_greedy
            for slot in slots
        ):
            fallback_caps = caps
            if _verify_gamma is not None and requested_verify_gamma < self.k:
                fallback_caps = {slot: requested_verify_gamma for slot in slots}
            return {
                slot: self.round(
                    slot,
                    anchors[slot],
                    drafts_by_slot[slot],
                    params=params_per_slot.get(slot),
                    return_logprobs=return_logprobs,
                    top_logprobs=top_logprobs,
                    _accept_cap=(fallback_caps.get(slot) if fallback_caps is not None else None),
                    thinking_force_position=(
                        thinking_force_positions.get(slot)
                        if thinking_force_positions is not None
                        else None
                    ),
                    thinking_force_token_id=(
                        thinking_force_token_ids.get(slot)
                        if thinking_force_token_ids is not None
                        else None
                    ),
                )
                for slot in slots
            }

        batch = len(slots)
        if verify_gamma == self.k:
            verify_tokens = self._verify_tokens_batch[batch]
            verify_host = self._verify_host_batch[batch]
            verify_host_view = self._verify_host_batch_views[batch]
        else:
            width_key = (batch, verify_gamma)
            verify_tokens = self._verify_tokens_batch_by_width[width_key]
            verify_host = self._verify_host_batch_by_width[width_key]
            verify_host_view = self._verify_host_batch_views_by_width[width_key]

        # Stage host rows in the pinned buffer first.  Device draft rows from
        # the previous graph replay are copied over those rows afterward, so
        # graph-owned output views never need a D2H round trip.
        for row, slot in enumerate(slots):
            verify_host_view[row, 0] = int(anchors[slot])
            draft_row = drafts_by_slot[slot]
            if isinstance(draft_row, torch.Tensor):
                if draft_row.ndim == 2 and tuple(draft_row.shape) == (1, self.k):
                    draft_row = draft_row[0]
                if draft_row.ndim != 1 or draft_row.numel() != self.k:
                    raise ValueError(
                        "DSpark batch device drafts must have shape [K] or [1, K]; "
                        f"got {tuple(draft_row.shape)}"
                    )
                # Fill the host row with a harmless value before the D2D
                # overwrite.  This keeps the pinned staging buffer complete
                # for mixed list/device batches without synchronizing the
                # graph-owned tensor.
                verify_host_view[row, 1:] = 0
            else:
                values = [int(token) for token in draft_row]
                if len(values) != self.k:
                    raise ValueError(
                        f"DSpark batch received {len(values)} drafts; expected {self.k}"
                    )
                verify_host_view[row, 1:] = 0
                verify_host_view[row, 1 : 1 + verify_gamma] = values[:verify_gamma]
        verify_tokens.copy_(verify_host, non_blocking=True)
        for row, slot in enumerate(slots):
            draft_row = drafts_by_slot[slot]
            if isinstance(draft_row, torch.Tensor):
                if draft_row.ndim == 2:
                    draft_row = draft_row[0]
                verify_tokens[row, 1:].copy_(
                    draft_row[:verify_gamma].to(device=self.device, dtype=torch.long),
                    non_blocking=True,
                )

        states = [self.backend.pool.slot_state(slot) for slot in slots]
        past_lens = [state.num_tokens_seen for state in states]
        round_profile.begin_round()
        round_profile.phase("setup")
        graph_result = verify_graph.replay_with_aux(
            slots,
            verify_tokens,
            past_lens,
            return_accept=self.capture_device_accept,
        )
        _all_hiddens, logits_batch, target_taps = graph_result[:3]
        self.stats["verify_graph_replays"] += 1
        self.backend.stats.setdefault("dspark_verify_graph_replays", 0)
        self.backend.stats["dspark_verify_graph_replays"] += 1
        round_profile.phase("verify_replay")

        force_applied = self._apply_thinking_forces_to_logits(
            logits_batch,
            slots,
            row_starts=[index * (verify_gamma + 1) for index in range(batch)],
            row_limits=[verify_gamma + 1] * batch,
            force_positions=thinking_force_positions or {},
            force_token_ids=thinking_force_token_ids or {},
        )
        if force_applied:
            full_drafts = self._full_verify_drafts(slots, anchors, drafts_by_slot)
            decisions = determine_accept_reject_batch(
                slots,
                full_drafts,
                logits_batch.reshape(-1, logits_batch.shape[-1]),
                self.k,
            )
        elif self.capture_device_accept and verify_graph is not None:
            decisions = self._decisions_from_device_accept(
                slots, graph_result[3], graph_result[4], verify_gamma
            )
        else:
            decisions = determine_accept_reject_batch(
                slots,
                verify_tokens,
                logits_batch,
                verify_gamma,
            )
        if caps is not None:
            self._apply_accept_caps(decisions, caps, self.k)
        round_profile.phase("accept_decision")

        accepted_counts: list[int] = []
        committed_by_slot: dict[int, list[int]] = {}
        for slot, state, past_len in zip(slots, states, past_lens, strict=True):
            decision = decisions[slot]
            accepted = int(decision["num_accepted"])
            committed = [int(token) for token in decision["committed"]]
            self.model.commit_verify(
                state,
                None,
                past_len=past_len,
                accepted_count=accepted + 1,
            )
            self._spec_rows.activate(slot, accepted)
            self._spec_state_col[slot] = accepted
            self.backend.pool.slot_kv_len[slot] = state.num_tokens_seen
            self.backend.pool.slot_committed_tokens[slot].extend(committed)
            self.backend._maybe_checkpoint(slot)  # noqa: SLF001 - backend-owned policy
            accepted_counts.append(accepted + 1)
            committed_by_slot[slot] = committed
        round_profile.phase("commit")

        self._sync_target_hidden_batch(
            slots,
            target_taps,
            position_offsets=past_lens,
            accepted_counts=accepted_counts,
        )
        round_profile.phase("target_hidden_sync")

        next_anchors = [committed_by_slot[slot][-1] for slot in slots]
        next_lens = [self.backend.pool.slot_state(slot).num_tokens_seen for slot in slots]
        next_drafts_by_slot = self._forward_draft_batch(
            slots,
            next_anchors,
            next_lens,
            params_per_slot=params_per_slot,
        )
        round_profile.phase("draft_batch")

        self.stats["rounds"] += batch
        output: dict[int, dict[str, Any]] = {}
        for row, (slot, past_len) in enumerate(zip(slots, past_lens, strict=True)):
            decision = decisions[slot]
            committed = committed_by_slot[slot]
            item: dict[str, Any] = {
                "committed": committed,
                "num_accepted": int(decision["num_accepted"]),
                "next_anchor": committed[-1],
                "next_draft_tokens": next_drafts_by_slot[slot],
            }
            if return_logprobs:
                logits = logits_batch[row]
                item["logprobs"] = [
                    compute_logprobs(
                        logits[position : position + 1],
                        [committed[position]],
                        top_k=top_logprobs,
                    )[0]
                    for position in range(len(committed))
                ]
            output[slot] = item
        round_profile.end_round(label=f"dspark_round_b{batch}")
        return output
