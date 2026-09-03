"""Serving backend for the Qwen3.8 Flash-Next checkpoint.

The Flash-Next model graph lives under :mod:`runtime.model.flashnext`.  This
module is the deliberately small serving adapter around that graph: one
process, one SM120 GPU, and one live slot.  It keeps the scheduler-facing
contract independent from the model implementation while preserving the
native target CUDA graph and the in-checkpoint MTP verify path.

Flash-Next has two state families (GDN/PLE recurrent state and QSA KV), so a
cold slot reset must clear both. Every slot owns an independent
state/pool/graph set; the model weights and MTP module are shared read-only
across slots. Prefix reuse is implemented as a co-keyed checkpoint: fixed
QSA/MTP pools stay in place and only the bounded recurrent/metadata state is
copied. A cache miss invalidates the checkpoint before writing new tokens, so
the correctness boundary is explicit rather than an optimistic KV-only hit.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from runtime.backends.protocol import (
    BackendCapabilities,
    BackendSnapshot,
    PrefixHit,
    PrefixSnapshot,
    SlotSnapshot,
)
from runtime.block_pool import ChunkedPrefillState
from runtime.logprobs import compute_logprobs
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

if TYPE_CHECKING:
    from runtime.model.flashnext.model import (
        FlashNextGraphEngine,
        FlashNextModel,
        FlashNextSession,
    )
    from runtime.model.flashnext.spec import FlashNextSpecEngine

logger = logging.getLogger(__name__)

_UNCACHEABLE_VISION_PREFIX_KEY = "<uncacheable-vision-prefix>"
_PREFIX_CACHE_MODE_MARKER = "<flashnext-prefix-cache-mode>"
_PREFIX_CACHE_TEXT_KEY = "<text-prefix>"


def _flashnext_gdn_projection_mode(model: FlashNextModel) -> str:
    """Return the target-verify GDN projection execution contract.

    ``qwen4_exp`` Flash-Next checkpoints intentionally keep the two large GDN
    projections in BF16.  The old serving gate treated every BF16 pair as an
    unsafe compatibility fallback, which left the production MTP verify path
    doing four independent M=1 GEMMs per GDN layer even though the rest of
    verify already runs at M=K+1.  The exact qwen4_exp + FP32-GDN-state
    contract has a long-run end-to-end quality gate (see
    ``notes/2026-08-27-flashnext-runtime-support-plan.md`` §4.16), so it is
    now the ``auto`` production path.

    Unknown Flash-Next-like checkpoints remain conservative.  They can opt in
    to the BF16 large-M path only with the explicitly named validation switch;
    the standalone diagnostic variable ``FN_BATCH_GDN_PROJECTIONS`` is never
    consulted by serving.
    """

    requested = os.environ.get("QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS", "auto").strip().lower()
    if requested not in {"auto", "0", "1"}:
        logger.warning(
            "ignoring invalid QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS=%r; expected auto, 0, or 1",
            requested,
        )
        return "per_row"
    if requested == "0":
        return "disabled"

    gdn_layers = [
        layer for layer in getattr(model, "layers", ()) if not getattr(layer, "is_qsa", False)
    ]
    if not gdn_layers:
        return "disabled"

    # Keep these imports lazy: importing the serving adapter must remain
    # torch-only for registry/tests that do not load a checkpoint.
    from runtime.model.compressed_tensors_linear import (
        CompressedTensorsFP8ChannelLinear,
        fp8_channel_raw_execution_uses_all_layers,
    )
    from runtime.model.modelopt_linear import ModelOptNVFP4W4A4Linear

    def pair_is_qualified(attn: object) -> bool:
        qkvz = getattr(attn, "in_proj_qkvz", None)
        out_proj = getattr(attn, "out_proj", None)
        modelopt_w4a4 = isinstance(qkvz, ModelOptNVFP4W4A4Linear) and isinstance(
            out_proj, ModelOptNVFP4W4A4Linear
        )
        raw_fp8 = (
            fp8_channel_raw_execution_uses_all_layers()
            and isinstance(qkvz, CompressedTensorsFP8ChannelLinear)
            and isinstance(out_proj, CompressedTensorsFP8ChannelLinear)
        )
        return modelopt_w4a4 or raw_fp8

    if all(pair_is_qualified(layer.attn) for layer in gdn_layers):
        return "batched_quantized"

    cfg = getattr(model, "cfg", None)
    # This is deliberately a structural contract, not a blanket BF16 opt-in:
    # qwen4_exp's SGLang-compatible verify path persists FP32 GDN state and
    # has a dedicated long-run quality record.  A model_type match without
    # the FP32 state contract is not enough to select the batched reduction
    # order automatically.
    validated_bf16_contract = (
        str(getattr(cfg, "model_type", "")).casefold() == "qwen4_exp"
        and str(getattr(cfg, "mamba_ssm_dtype", "")).casefold() == "float32"
    )
    if validated_bf16_contract and requested in {"auto", "1"}:
        logger.info(
            "Flash-Next qwen4_exp FP32-GDN contract enables batched BF16 "
            "large projections for target verify"
        )
        return "batched_bf16"

    # Keep an explicit escape hatch for a newly validated checkpoint while
    # preventing a generic force flag from silently changing numerics.
    if not validated_bf16_contract:
        allow_bf16 = os.environ.get(
            "QSR_FLASHNEXT_ALLOW_BF16_BATCH_PROJECTIONS", "0"
        ).strip().lower() in {"1", "true", "on"}
        if requested == "1" and allow_bf16:
            logger.warning(
                "Flash-Next GDN large-projection batching enabled for the explicit "
                "BF16 validation override; this path is not numerically validated "
                "for every workload"
            )
            return "batched_bf16_override"
        message = (
            "Flash-Next GDN large-projection batching is disabled: the loaded "
            "checkpoint does not expose a validated W4A4/raw-FP8 projection "
            "contract; using the BF16 per-row verify path"
        )
        if requested == "1":
            logger.warning(
                "%s; ignoring explicit QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS=1",
                message,
            )
        else:
            logger.info(message)
    return "per_row"


def _flashnext_batch_gdn_projections_enabled(model: FlashNextModel) -> bool:
    """Compatibility bool for callers/tests that only need the fast-path bit."""

    return _flashnext_gdn_projection_mode(model).startswith("batched")


def _cuda_tensor_storage_bytes(tensor: torch.Tensor) -> tuple[int, int]:
    """Return a storage-level dedup key plus byte size for one tensor.

    Flash-Next's verify buffers keep many row views into a shared backing
    allocation. Counting ``tensor.data_ptr()`` or ``numel()*itemsize`` would
    therefore over-count the same storage once per row. Storage accounting
    keeps the debug surface truthful about real resident bytes.
    """
    storage = tensor.untyped_storage()
    return int(storage.data_ptr()), int(storage.nbytes())


@dataclass(frozen=True)
class FlashNextSlotStateView:
    """Immutable scheduler view of one Flash-Next sequence."""

    kv_len: int
    committed_tokens: tuple[int, ...]

    @property
    def is_fresh(self) -> bool:
        return self.kv_len == 0 and not self.committed_tokens


@dataclass
class FlashNextPrefixSnapshot:
    """Co-keyed prompt checkpoint for one Flash-Next slot.

    The large QSA/FP8 pools are deliberately *not* cloned.  They remain in
    the slot's fixed-address allocation across ``reset_slot`` and are reused
    in place.  Only the length-independent recurrent/PLE state, plus the
    small MTP scalars and first-token result, is copied.  This keeps a prefix
    hit bounded to tens of MiB instead of allocating another 256K KV cache.
    """

    token_ids: tuple[int, ...]
    kv_len: int
    gdn: dict[str, tuple[torch.Tensor, torch.Tensor, bool]]
    ple_conv_state: torch.Tensor | None
    rope_next: torch.Tensor | None
    window: tuple[int, ...]
    anchor: int
    draft_tokens: tuple[int, ...]
    anchor_logits: torch.Tensor
    vision_cache_key: tuple[str, ...] | None = None
    mtp_sync_len: int = 0
    mtp_pos: int = 0
    mtp_ready: bool = False
    # Historical checkpoints are captured after target prefill and real-token
    # MTP teacher sync, before the final proposal is made.  They therefore
    # carry a valid real-prefix MTP state but no speculative tail.  Keeping
    # this bit separate from ``mtp_ready`` lets an extension restore the MTP
    # pool cursor without pretending that cached drafts are reusable.
    mtp_prefix_ready: bool = False
    # Target-only checkpoints created by the bounded historical cache may be
    # used for both greedy and sampled requests.  Older target-only entries
    # keep the conservative sampled-only behaviour for compatibility.
    target_only_reusable: bool = False
    # The mode is part of the checkpoint contract.  Greedy and sampled MTP
    # share the recurrent prefix but not the proposal RNG/distribution state;
    # never restore one mode's speculative tail for the other.
    decode_mode: str | None = None
    # One target hidden row is enough to re-teacher-force the first sampled
    # anchor after a full prefix hit.  The fixed MTP pools themselves remain
    # in the slot allocation and are rewound to ``mtp_sync_len`` on restore.
    mtp_teacher_hidden: torch.Tensor | None = None


class FlashNextBackend:
    """Flash-Next backend with one independent target/MTP graph set per slot."""

    _CAPABILITIES = BackendCapabilities(
        speculative_decode=True,
        prefix_cache=True,
        cuda_graph=True,
        chunked_prefill=True,
        warm_continue=False,
        kv_reservation=False,
        prefix_cache_dedup=False,
    )
    # Sampled MTP uses eager stochastic proposal/acceptance around the same
    # fixed-width target verify graph.  Greedy requests still use the captured
    # proposal graphs; the capability bit only controls scheduler routing.
    supports_sampled_speculative_decode = True

    def __init__(
        self,
        model: FlashNextModel,
        *,
        num_slots: int = 1,
        max_seq_len: int = 262_144,
        device: str = "cuda",
        checkpoint_path: str | None = None,
        enable_mtp: bool = True,
        mtp_num_speculative_tokens: int = 3,
        enable_prefix_cache: bool = True,
    ) -> None:
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if mtp_num_speculative_tokens <= 0:
            raise ValueError(
                f"mtp_num_speculative_tokens must be positive, got {mtp_num_speculative_tokens}"
            )

        from runtime.model.flashnext.model import new_session, prepare_graph_buffers

        self.model = model
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path or os.environ.get("QSR_FLASHNEXT_CHECKPOINT", "")
        self.enable_mtp = bool(enable_mtp)
        self.mtp_num_speculative_tokens = int(mtp_num_speculative_tokens)
        self.enable_prefix_cache = bool(enable_prefix_cache)
        # Flash-Next's fixed-address pools are token-indexed rather than
        # paged by the generic BlockPool.  Its recurrent checkpoint is safe
        # at any token boundary, so the coordinator must not floor it to the
        # server's attention page size (typically 128).
        self.prefix_cache_block_size = 1
        self._targets: list[FlashNextGraphEngine] = []
        self._specs: list[FlashNextSpecEngine | None] = []
        self._slot_tokens: list[list[int]] = [[] for _ in range(num_slots)]
        self._last_logits: list[torch.Tensor | None] = [None for _ in range(num_slots)]
        self._prefix_cache: list[FlashNextPrefixSnapshot | None] = [None] * num_slots
        # A single latest checkpoint cannot serve a prompt that is shorter
        # than the last request (the common shape after client compaction).
        # Keep a small bounded history of recurrent checkpoints so any
        # authenticated common boundary can be resumed without replaying the
        # whole cold prefix.  Six entries/slot costs roughly 0.9 GiB/slot on
        # the production checkpoint and is below the measured two-slot headroom.
        try:
            checkpoint_limit = int(
                os.environ.get("QSR_FLASHNEXT_PREFIX_CHECKPOINTS_PER_SLOT", "6")
            )
        except ValueError:
            checkpoint_limit = 6
            logger.warning(
                "invalid QSR_FLASHNEXT_PREFIX_CHECKPOINTS_PER_SLOT; using 6"
            )
        self.prefix_cache_checkpoints_per_slot = max(1, min(checkpoint_limit, 16))
        try:
            checkpoint_interval = int(
                os.environ.get("QSR_FLASHNEXT_PREFIX_CHECKPOINT_INTERVAL", "8192")
            )
        except ValueError:
            checkpoint_interval = 8192
            logger.warning(
                "invalid QSR_FLASHNEXT_PREFIX_CHECKPOINT_INTERVAL; using 8192"
            )
        self.prefix_cache_checkpoint_interval = max(1, min(checkpoint_interval, max_seq_len))
        self._prefix_cache_history: list[list[FlashNextPrefixSnapshot]] = [
            [] for _ in range(num_slots)
        ]
        # These two side tables mirror the names consumed by the existing
        # /debug/stats compatibility surface.
        self._prefix_cache_tokens: list[list[int] | None] = [None] * num_slots
        self._prefix_cache_kv_len: list[int] = [0] * num_slots
        # Admission reconciliation can inspect all fresh slots before the
        # scheduler starts each assigned prefill.  Keep the request identity
        # alongside the depth so a hit found for another request/slot can
        # never be consumed blindly after multi-slot admission.
        self._pending_prefix_hits: dict[
            int, tuple[int, tuple[int, ...], tuple[str, ...] | None]
        ] = {}
        self._captured = False
        # Diagnostic escape hatch for isolating verify CUDA-Graph state from
        # the eager correctness path.  It never changes the model or token
        # budget; when enabled, target verification remains eager while the
        # ordinary decode graph can still be captured.
        self._mtp_verify_eager = os.environ.get(
            "QSR_FLASHNEXT_MTP_VERIFY_EAGER", "0"
        ).strip().lower() in {"1", "true", "on"}
        self._cg_status: dict[str, str] = {}
        self.stats: dict[str, int] = {
            "prefill_requests": 0,
            "decode_rounds": 0,
            "decode_tokens": 0,
            "mtp_rounds": 0,
            "mtp_accepted_tokens": 0,
            # Wall-clock substage timings returned by the verify driver.  The
            # counters make cold PLE/page-fault stalls visible without adding
            # per-round allocations to the normal serving path.
            "mtp_ple_ns": 0,
            "mtp_verify_ns": 0,
            "mtp_proposal_ns": 0,
            "mtp_last_ple_ns": 0,
            "mtp_last_verify_ns": 0,
            "mtp_last_proposal_ns": 0,
            "prefix_cache_history_hits": 0,
            "prefix_cache_history_entries": 0,
            "prefix_cache_full_hits": 0,
            "prefix_cache_stores": 0,
            "prefix_cache_restores": 0,
            "prefill_chunks": 0,
            "prefill_tokens": 0,
            "prefill_target_ns": 0,
            "prefill_mtp_sync_ns": 0,
            "prefill_mtp_draft_ns": 0,
            "prefill_trim_ns": 0,
            "prefill_last_chunks": 0,
            "prefill_last_tokens": 0,
            "prefill_last_target_ns": 0,
            "prefill_last_mtp_sync_ns": 0,
            "prefill_last_mtp_draft_ns": 0,
            "prefill_last_trim_ns": 0,
        }

        shared_verify_buffers = None
        share_verify_buffers = (
            os.environ.get("QSR_FLASHNEXT_SHARE_VERIFY_BUFFERS", "1").strip().lower()
        )
        if share_verify_buffers not in {"0", "false", "off", "1", "true", "on"}:
            raise ValueError(
                f"QSR_FLASHNEXT_SHARE_VERIFY_BUFFERS must be 0 or 1, got {share_verify_buffers!r}"
            )
        gdn_projection_mode = _flashnext_gdn_projection_mode(model)
        batch_gdn_projections = gdn_projection_mode.startswith("batched")
        # Surface the actual contract through BackendSnapshot and
        # /debug/stats.  A boolean alone cannot distinguish the validated
        # qwen4_exp BF16 path from the explicit unknown-checkpoint override,
        # which made prior performance comparisons needlessly ambiguous.
        self._cg_status["gdn_projections"] = gdn_projection_mode
        for _slot in range(num_slots):
            sess = new_session(model, self.device)
            prepare_graph_buffers(
                model,
                sess,
                self.device,
                max_seq=max_seq_len,
                fixed_index_rows=max(self.mtp_num_speculative_tokens + 1, 1),
            )
            from runtime.model.flashnext.model import FlashNextGraphEngine

            target = FlashNextGraphEngine(model, sess, self.device)
            self._targets.append(target)
            if self.enable_mtp:
                from runtime.model.flashnext.mtp import load_flashnext_mtp
                from runtime.model.flashnext.spec import FlashNextSpecEngine

                # The MTP head is shared by all slots.  Each slot still owns
                # an independent MTP cache/session and verify/proposal graphs.
                if not hasattr(self, "_mtp_model"):
                    if not self.checkpoint_path:
                        raise ValueError(
                            "checkpoint_path is required when Flash-Next MTP is enabled"
                        )
                    self._mtp_model = load_flashnext_mtp(
                        self.checkpoint_path,
                        model.cfg,
                        model,
                        device=device,
                    )
                spec = FlashNextSpecEngine(
                    model,
                    self._mtp_model,
                    sess,
                    max_seq=max_seq_len,
                    device=self.device,
                    k=self.mtp_num_speculative_tokens,
                    exact_row_math=os.environ.get("QSR_FLASHNEXT_EXACT_ROW_MATH", "0") == "1",
                    batch_lm_head=os.environ.get("QSR_FLASHNEXT_BATCH_LM_HEAD", "1") == "1",
                    batch_gdn_recurrence=os.environ.get("QSR_FLASHNEXT_BATCH_GDN_RECURRENCE", "1")
                    == "1",
                    batch_gdn_projections=batch_gdn_projections,
                    sequential_qsa=os.environ.get("QSR_FLASHNEXT_SEQUENTIAL_QSA_VERIFY", "0")
                    == "1",
                    recompute_recurrent_state=os.environ.get(
                        # Recomputing all four FP32 GDN rows outside a graph
                        # saves ~0.4 GiB but adds 36 launches per round. Keep
                        # it an explicit capacity experiment until the commit
                        # path has a captured/fused implementation.
                        "QSR_FLASHNEXT_RECOMPUTE_VERIFY_STATE",
                        "0",
                    )
                    == "1",
                    mtp_continuation_graph=os.environ.get(
                        "QSR_FLASHNEXT_MTP_CONTINUATION_GRAPH", "0"
                    )
                    == "1",
                    mtp_sparse_graph=os.environ.get("QSR_FLASHNEXT_MTP_SPARSE_GRAPH", "0") == "1",
                    verify_buffers=(
                        shared_verify_buffers
                        if share_verify_buffers in {"1", "true", "on"}
                        else None
                    ),
                )
                if shared_verify_buffers is None and share_verify_buffers in {
                    "1",
                    "true",
                    "on",
                }:
                    shared_verify_buffers = spec.verify.buffers
                self._specs.append(spec)
            else:
                self._specs.append(None)

    @property
    def capabilities(self) -> BackendCapabilities:
        if getattr(self, "enable_prefix_cache", True):
            return self._CAPABILITIES
        return BackendCapabilities(
            speculative_decode=self._CAPABILITIES.speculative_decode,
            prefix_cache=False,
            cuda_graph=self._CAPABILITIES.cuda_graph,
            chunked_prefill=self._CAPABILITIES.chunked_prefill,
            warm_continue=self._CAPABILITIES.warm_continue,
            kv_reservation=self._CAPABILITIES.kv_reservation,
            prefix_cache_dedup=False,
        )

    @property
    def has_speculative_decode(self) -> bool:
        return self.enable_mtp

    def _target_session(self, slot: int) -> FlashNextSession:
        return self._targets[slot].sess

    def prefix_cache_key_for_vision_inputs(
        self, vision_inputs: object | None
    ) -> tuple[str, ...] | None:
        if vision_inputs is None:
            return None
        raw = getattr(vision_inputs, "image_cache_keys", None)
        if raw is None:
            return (_UNCACHEABLE_VISION_PREFIX_KEY,)
        try:
            keys = tuple(str(item) for item in raw)
        except TypeError:
            return (_UNCACHEABLE_VISION_PREFIX_KEY,)
        if not keys or any(not key for key in keys):
            return (_UNCACHEABLE_VISION_PREFIX_KEY,)
        return keys

    @classmethod
    def prefix_cache_key_for_sampling(
        cls,
        vision_cache_key: tuple[str, ...] | None,
        *,
        sampled: bool,
    ) -> tuple[str, ...]:
        """Add the decode contract to the admission-time cache key.

        Greedy and sampled continuations retain different speculative tails
        (sampled also carries a request-local RNG/distribution contract).  The
        scheduler reconciles prefixes before calling
        ``prefill_chunked_begin``; carrying this small mode marker in the
        existing opaque key lets the backend enforce that boundary without
        changing the shared coordinator ABI.
        """
        mode = "sampled" if sampled else "greedy"
        vision_parts = vision_cache_key or (_PREFIX_CACHE_TEXT_KEY,)
        return (_PREFIX_CACHE_MODE_MARKER, mode, *vision_parts)

    @staticmethod
    def _split_prefix_cache_key(
        prefix_cache_key: object | None,
    ) -> tuple[str | None, tuple[str, ...] | None]:
        """Return ``(decode_mode, vision_key)`` from an admission cache key.

        Direct backend callers and older tests pass the original vision key;
        those retain the conservative pre-sampled behaviour (``mode=None``).
        """
        if isinstance(prefix_cache_key, tuple) and len(prefix_cache_key) >= 3:
            if prefix_cache_key[0] == _PREFIX_CACHE_MODE_MARKER and prefix_cache_key[1] in {
                "greedy",
                "sampled",
            }:
                raw_vision = tuple(str(item) for item in prefix_cache_key[2:])
                vision_key = None if raw_vision == (_PREFIX_CACHE_TEXT_KEY,) else raw_vision
                return str(prefix_cache_key[1]), vision_key
        if prefix_cache_key is None:
            return None, None
        if isinstance(prefix_cache_key, tuple):
            return None, tuple(str(item) for item in prefix_cache_key)
        return None, (str(prefix_cache_key),)

    @staticmethod
    def _vision_cache_matches(
        current: tuple[str, ...] | None,
        cached: tuple[str, ...] | None,
    ) -> bool:
        if current is None:
            return cached is None
        if current == (_UNCACHEABLE_VISION_PREFIX_KEY,):
            return False
        if cached is None:
            return False
        if cached == (_UNCACHEABLE_VISION_PREFIX_KEY,):
            return False
        return len(current) >= len(cached) and current[: len(cached)] == cached

    def _shift_teacher_force_embeds(
        self,
        input_embeds: torch.Tensor,
        anchor: int,
    ) -> torch.Tensor:
        if input_embeds.ndim != 2 or input_embeds.shape[0] <= 0:
            raise ValueError(
                "Flash-Next teacher-force embeddings must be a [tokens, hidden] tensor, "
                f"got {tuple(input_embeds.shape)}"
            )
        if input_embeds.shape[0] > 1:
            input_embeds[:-1].copy_(input_embeds[1:].clone())
        anchor_embed = self.model.embed_tokens(
            torch.tensor([int(anchor)], dtype=torch.long, device=self.device)
        )[0].to(device=input_embeds.device, dtype=input_embeds.dtype)
        input_embeds[-1].copy_(anchor_embed)
        return input_embeds

    def _reset_mtp_state(
        self,
        spec: FlashNextSpecEngine | None,
        *,
        clear_cache: bool = True,
    ) -> None:
        if spec is None:
            return
        sess = spec.mtp_session
        if clear_cache:
            for name in (
                "mtp_k_pool",
                "mtp_v_pool",
                "mtp_idx_k_pool",
                "mtp_pooled_k_pool",
            ):
                tensor = getattr(sess, name, None)
                if torch.is_tensor(tensor):
                    tensor.zero_()
            for name in ("mtp_k_scale_pool", "mtp_v_scale_pool"):
                tensor = getattr(sess, name, None)
                if torch.is_tensor(tensor):
                    tensor.fill_(1.0)
        sess.sync_len = 0
        sess.pos = 0
        sess.shared_sparse_captured_len = 0
        if torch.is_tensor(sess.shared_sparse_indices):
            sess.shared_sparse_indices.zero_()
        if torch.is_tensor(sess.shared_sparse_valid):
            sess.shared_sparse_valid.zero_()
        sparse = getattr(sess, "sparse_graph_buffers", None)
        if sparse is not None:
            for value in vars(sparse).values():
                if torch.is_tensor(value):
                    value.zero_()
        # A previous round normally clears this itself.  Clearing explicitly
        # makes cancellation and a failed request safe too.
        spec.pending_draft_probs = None
        spec.verify._last_tokens = None  # noqa: SLF001 - lifecycle reset

    def _reset_runtime(self, slot: int, *, preserve_prefix: bool = False) -> None:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        target = self._targets[slot]
        sess = target.sess
        keep_prefix = preserve_prefix and bool(self._prefix_entries_for_slot(slot))
        target._zero_state(clear_kv=not keep_prefix)  # noqa: SLF001 - graph-owned state lifecycle
        # Eager decode uses append-only dictionaries rather than the graph
        # pools.  Clearing them prevents a no-CG fallback from inheriting a
        # previous sequence's QSA rows.
        sess.qsa_k.clear()
        sess.qsa_v.clear()
        sess.qsa_idx_k.clear()
        self._reset_mtp_state(self._specs[slot], clear_cache=not keep_prefix)
        self._slot_tokens[slot] = []
        self._last_logits[slot] = None

    def reset_slot(self, slot: int) -> None:
        # Keep the prompt history's fixed-address KV and recurrent checkpoints
        # available for the next admission.  A future cold miss explicitly
        # calls _reset_runtime(preserve_prefix=False), so stale rows can never
        # leak into a different prompt.
        self._reset_runtime(slot, preserve_prefix=self.enable_prefix_cache)

    def _prefix_entries_for_slot(self, slot: int) -> tuple[FlashNextPrefixSnapshot, ...]:
        """Return the bounded history, with a legacy latest-entry fallback."""
        history = getattr(self, "_prefix_cache_history", None)
        if isinstance(history, list) and 0 <= slot < len(history) and history[slot]:
            return tuple(history[slot])
        entries = getattr(self, "_prefix_cache", ())
        if not isinstance(entries, (list, tuple)) or not 0 <= slot < len(entries):
            return ()
        entry = entries[slot]
        return (entry,) if entry is not None else ()

    def _prefix_checkpoint_limit(self) -> int:
        configured = getattr(self, "prefix_cache_checkpoints_per_slot", None)
        if configured is not None:
            return max(1, min(int(configured), 16))
        return 6

    def _prefix_checkpoint_interval(self) -> int:
        configured = getattr(self, "prefix_cache_checkpoint_interval", None)
        if configured is not None:
            return max(1, int(configured))
        return 8192

    def _select_prefix_entry(
        self,
        token_ids: list[int],
        slot: int,
        *,
        prefix_cache_key: tuple[str, ...] | None = None,
        exact_kv_len: int | None = None,
    ) -> FlashNextPrefixSnapshot | None:
        """Find the deepest authenticated checkpoint that fits this prompt."""
        if not getattr(self, "enable_prefix_cache", True) or not token_ids:
            return None
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        cache_mode, vision_cache_key = self._split_prefix_cache_key(prefix_cache_key)
        specs = getattr(self, "_specs", ())
        has_spec = slot < len(specs) and specs[slot] is not None
        candidates: list[FlashNextPrefixSnapshot] = []
        for entry in self._prefix_entries_for_slot(slot):
            if not entry.token_ids or not self._vision_cache_matches(
                vision_cache_key, entry.vision_cache_key
            ):
                continue
            if (
                cache_mode in {"greedy", "sampled"}
                and entry.decode_mode in {"greedy", "sampled"}
                and cache_mode != entry.decode_mode
                and not (not entry.mtp_ready and entry.target_only_reusable)
            ):
                # The target-only historical state is mode agnostic; a full
                # MTP proposal remains mode-specific.
                continue
            if (
                entry.mtp_ready
                and cache_mode == entry.decode_mode
                and entry.decode_mode in {"greedy", "sampled"}
                and entry.mtp_teacher_hidden is None
            ):
                # Full MTP checkpoints need the boundary teacher row to
                # repair the shifted anchor on an extension/sample hit.
                continue
            if (
                has_spec
                and not entry.mtp_ready
                and not entry.target_only_reusable
                and cache_mode != "sampled"
            ):
                # Preserve the old conservative direct/greedy contract for
                # legacy checkpoints that have no mode-neutral marker.
                continue
            if len(token_ids) < entry.kv_len:
                # A shorter request may still match an older checkpoint; do
                # not reject the whole slot because its newest entry is later.
                continue
            if exact_kv_len is not None and entry.kv_len != exact_kv_len:
                continue
            if tuple(token_ids[: entry.kv_len]) != entry.token_ids:
                continue
            if len(token_ids) == entry.kv_len and not entry.anchor_logits.numel():
                continue
            candidates.append(entry)
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.kv_len)

    def _prefix_hit_for_slot(
        self,
        token_ids: list[int],
        slot: int,
        *,
        prefix_cache_key: tuple[str, ...] | None = None,
    ) -> PrefixHit:
        """Return an exact safe checkpoint boundary for one idle slot.

        Flash-Next has both QSA and recurrent state.  A token match is useful
        only when the recurrent snapshot was taken at that *same* boundary;
        therefore this deliberately exposes the deepest stored checkpoint
        length rather than rounding to an attention page or claiming a
        partial KV-only hit.
        The fixed QSA pools make arbitrary token boundaries safe.
        """
        if not getattr(self, "enable_prefix_cache", True) or not token_ids:
            return PrefixHit(kv_hit=0, state_hit=0)
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        entry = self._select_prefix_entry(
            token_ids,
            slot,
            prefix_cache_key=prefix_cache_key,
        )
        if entry is None:
            return PrefixHit(kv_hit=0, state_hit=0)
        latest = getattr(self, "_prefix_cache", ())
        if (
            slot >= len(latest)
            or latest[slot] is not entry
        ):
            stats = getattr(self, "stats", {})
            stats["prefix_cache_history_hits"] = stats.get(
                "prefix_cache_history_hits", 0
            ) + 1
        return PrefixHit(kv_hit=entry.kv_len, state_hit=entry.kv_len)

    def prefix_hit_for_slot(self, token_ids: list[int], slot: int) -> PrefixHit:
        """Coordinator hook for the two-cache-family admission path."""
        return self._prefix_hit_for_slot(token_ids, slot)

    def prefix_hit_for_slot_with_key(
        self,
        token_ids: list[int],
        slot: int,
        prefix_cache_key: tuple[str, ...] | None,
    ) -> PrefixHit:
        return self._prefix_hit_for_slot(token_ids, slot, prefix_cache_key=prefix_cache_key)

    def _capture_prefix_snapshot(
        self,
        slot: int,
        prompt_ids: list[int],
        *,
        anchor: int,
        draft_tokens: list[int],
        anchor_logits: torch.Tensor,
        mtp_ready: bool,
        mtp_prefix_ready: bool = False,
        target_only_reusable: bool = False,
        vision_cache_key: tuple[str, ...] | None = None,
        decode_mode: str | None = None,
        mtp_teacher_hidden: torch.Tensor | None = None,
    ) -> None:
        """Save only state needed to resume the just-computed prompt.

        QSA and MTP K/V/index pools stay in their existing fixed allocations;
        cloning them here would add several GiB at a 256K context and defeat
        the reason for this cache.  The GDN/PLE checkpoint is length
        independent and is copied once per slot.  MTP's prefix rows remain in
        its pool and are addressed again after restoring ``sync_len``.
        """
        if not getattr(self, "enable_prefix_cache", True) or not prompt_ids:
            return
        if not hasattr(self, "_prefix_cache"):
            return
        target = self._targets[slot]
        sess = target.sess
        gdn = {
            name: (
                state.conv_state.detach().clone(),
                state.recurrent_state.detach().clone(),
                bool(state.has_previous_state),
            )
            for name, state in sess.gdn.items()
        }
        ple_state = (
            sess.ple_conv_state.detach().clone() if torch.is_tensor(sess.ple_conv_state) else None
        )
        rope_next = sess.rope_next.detach().clone() if torch.is_tensor(sess.rope_next) else None
        spec = self._specs[slot]
        retain_mtp_prefix = spec is not None and (mtp_ready or mtp_prefix_ready)
        mtp_sess = spec.mtp_session if retain_mtp_prefix else None
        entry = FlashNextPrefixSnapshot(
            token_ids=tuple(int(token) for token in prompt_ids),
            kv_len=len(prompt_ids),
            gdn=gdn,
            ple_conv_state=ple_state,
            rope_next=rope_next,
            window=tuple(int(token) for token in sess.window),
            anchor=int(anchor),
            draft_tokens=tuple(int(token) for token in draft_tokens),
            anchor_logits=anchor_logits.detach().clone(),
            vision_cache_key=vision_cache_key,
            mtp_sync_len=int(mtp_sess.sync_len) if mtp_sess is not None else 0,
            mtp_pos=int(mtp_sess.pos) if mtp_sess is not None else 0,
            # ``mtp_ready`` means that this snapshot also owns a proposal
            # anchored at the boundary.  A historical checkpoint captured
            # after teacher-forcing a real prefix only retains the MTP
            # cursor; it must stay target-only so restore does not rewind a
            # nonexistent proposal row or apply mode-specific sampling.
            mtp_ready=bool(mtp_ready and mtp_sess is not None),
            mtp_prefix_ready=bool(mtp_prefix_ready and mtp_sess is not None),
            target_only_reusable=bool(target_only_reusable),
            decode_mode=decode_mode,
            mtp_teacher_hidden=(
                mtp_teacher_hidden.detach().clone() if torch.is_tensor(mtp_teacher_hidden) else None
            ),
        )
        history = getattr(self, "_prefix_cache_history", None)
        if isinstance(history, list) and slot < len(history):
            slot_history = history[slot]
            slot_history[:] = [old for old in slot_history if old.kv_len != entry.kv_len]
            slot_history.append(entry)
            limit = self._prefix_checkpoint_limit()
            if len(slot_history) > limit:
                # Keep the earliest authenticated boundary as a cheap anchor
                # for aggressively compacted sessions, plus the newest tail
                # where most same-session edits land.
                first = slot_history[0]
                tail = slot_history[-(limit - 1) :] if limit > 1 else []
                slot_history[:] = [first, *[old for old in tail if old is not first]]
            self.stats["prefix_cache_history_entries"] = sum(
                len(items) for items in history
            )
        self._prefix_cache[slot] = entry
        self._prefix_cache_tokens[slot] = list(entry.token_ids)
        self._prefix_cache_kv_len[slot] = entry.kv_len
        self.stats["prefix_cache_stores"] = self.stats.get("prefix_cache_stores", 0) + 1

    def _drop_prefix_snapshot(self, slot: int) -> None:
        """Invalidate one cache entry before a cold write can overwrite it."""
        history = getattr(self, "_prefix_cache_history", None)
        if isinstance(history, list) and slot < len(history):
            history[slot].clear()
            stats = getattr(self, "stats", {})
            stats["prefix_cache_history_entries"] = sum(len(items) for items in history)
        self._prefix_cache[slot] = None
        self._prefix_cache_tokens[slot] = None
        self._prefix_cache_kv_len[slot] = 0

    def _restore_prefix_snapshot(
        self,
        slot: int,
        prompt_ids: list[int],
        hit: int,
        *,
        entry: FlashNextPrefixSnapshot | None = None,
    ) -> None:
        """Restore recurrent/metadata state at an already retained KV prefix."""
        if entry is None:
            entry = next(
                (
                    candidate
                    for candidate in self._prefix_entries_for_slot(slot)
                    if candidate.kv_len == hit
                    and tuple(prompt_ids[:hit]) == candidate.token_ids
                ),
                None,
            )
        if entry is None or hit != entry.kv_len:
            raise RuntimeError(
                f"Flash-Next prefix restore has no checkpoint at hit={hit} for slot={slot}"
            )
        if tuple(prompt_ids[:hit]) != entry.token_ids:
            raise RuntimeError("Flash-Next prefix checkpoint token authentication failed")
        target = self._targets[slot]
        sess = target.sess
        for name, (conv, recurrent, has_previous) in entry.gdn.items():
            state = sess.gdn[name]
            state.conv_state.copy_(conv)
            state.recurrent_state.copy_(recurrent)
            state.has_previous_state = has_previous
        if sess.ple_conv_state is not None:
            if entry.ple_conv_state is None:
                raise RuntimeError("Flash-Next prefix checkpoint is missing PLE state")
            sess.ple_conv_state.copy_(entry.ple_conv_state)
        sess.window = list(entry.window)
        sess.pos = hit
        sess.rope_next = entry.rope_next.detach().clone() if entry.rope_next is not None else None
        spec = self._specs[slot]
        if spec is not None:
            if entry.mtp_prefix_ready or entry.mtp_ready:
                mtp_sess = spec.mtp_session
                if (
                    entry.mtp_ready
                    and entry.decode_mode in {"greedy", "sampled"}
                    and len(prompt_ids) > hit
                ):
                    # The cached proposal consumed the old anchor at L-1.
                    # An extended prompt replaces that row with the first
                    # suffix token, so rewind one row for the shifted
                    # teacher sync below.
                    mtp_sess.sync_len = max(entry.mtp_sync_len - 1, 0)
                    mtp_sess.pos = mtp_sess.sync_len
                elif entry.mtp_ready and entry.decode_mode == "sampled":
                    # The cached proposal already consumed the old anchor at
                    # position L-1.  Rewind that one row so a new request can
                    # overwrite it with its own sampled anchor before
                    # proposing fresh drafts.  The target hidden row for
                    # exactly this boundary is kept in the snapshot.
                    mtp_sess.sync_len = max(entry.mtp_sync_len - 1, 0)
                    mtp_sess.pos = mtp_sess.sync_len
                else:
                    mtp_sess.sync_len = entry.mtp_sync_len
                    mtp_sess.pos = entry.mtp_pos
                spec.pending_draft_probs = None
                # Any previous verify candidate is invalid after a slot reset;
                # the next proposal overwrites it before use.
                spec.verify._last_tokens = None  # noqa: SLF001 - lifecycle reset
            else:
                # Sampled decode uses only the target state.  The slot reset
                # intentionally retained fixed-address pools for the target;
                # clear the stale speculative pools before the next sampled
                # request so a later mode switch cannot observe them.
                self._reset_mtp_state(spec, clear_cache=True)
        self.stats["prefix_cache_restores"] = self.stats.get("prefix_cache_restores", 0) + 1

    def _prepare_prefill_prefix(
        self,
        slot: int,
        prompt_ids: list[int],
        *,
        prefix_cache_key: tuple[str, ...] | None = None,
    ) -> tuple[int, FlashNextPrefixSnapshot | None]:
        """Reset a slot cold or retain/restore its exact prompt prefix."""
        # Lightweight unit fixtures from before the cache fields existed
        # intentionally monkey-patch _reset_runtime(slot). Keep that test
        # double ABI while production instances take the cache-aware path.
        if not hasattr(self, "_prefix_cache"):
            self._reset_runtime(slot)
            return 0, None
        pending = self._pending_prefix_hits.pop(slot, None)
        hit: int | None = None
        if pending is not None:
            pending_hit, pending_tokens, pending_key = pending
            if pending_tokens == tuple(prompt_ids) and pending_key == prefix_cache_key:
                hit = int(pending_hit)
        if hit is None:
            hit = self._prefix_hit_for_slot(
                prompt_ids,
                slot,
                prefix_cache_key=prefix_cache_key,
            ).effective
        if hit > 0:
            entry = self._select_prefix_entry(
                prompt_ids,
                slot,
                prefix_cache_key=prefix_cache_key,
                exact_kv_len=hit,
            )
            if entry is None:
                hit = 0
        if hit > 0:
            self._reset_runtime(slot, preserve_prefix=True)
            self._restore_prefix_snapshot(slot, prompt_ids, hit, entry=entry)
            return hit, entry
        self._drop_prefix_snapshot(slot)
        self._reset_runtime(slot, preserve_prefix=False)
        return 0, None

    def slot_state(self, slot: int) -> FlashNextSlotStateView:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        sess = self._target_session(slot)
        return FlashNextSlotStateView(
            kv_len=int(sess.pos),
            committed_tokens=tuple(self._slot_tokens[slot]),
        )

    def snapshot(self) -> BackendSnapshot:
        return BackendSnapshot(
            slots=tuple(
                SlotSnapshot(
                    slot=slot,
                    kv_len=int(self._target_session(slot).pos),
                    is_fresh=self.slot_state(slot).is_fresh,
                )
                for slot in range(self.num_slots)
            ),
            prefix=tuple(
                PrefixSnapshot(
                    slot=slot,
                    cached_kv_len=self._prefix_cache_kv_len[slot],
                    cached_tokens=len(self._prefix_cache_tokens[slot] or ()),
                    head=tuple((self._prefix_cache_tokens[slot] or ())[:8]),
                )
                for slot in range(self.num_slots)
            ),
            dflash_cg_status=tuple(sorted(self._cg_status.items())),
            runtime_stats=tuple(sorted(self.stats.items())),
        )

    def memory_breakdown(self) -> dict[str, int]:
        """Byte-level CUDA accounting for Flash-Next's model and session state."""
        if torch.device(self.device).type != "cuda":
            return {}
        device_type = torch.device(self.device).type
        seen_storages: set[int] = set()
        totals: dict[str, int] = {
            "model_parameters": 0,
            "model_buffers": 0,
            "model_auxiliary_tensors": 0,
            "mtp_model_parameters": 0,
            "mtp_model_buffers": 0,
            "mtp_auxiliary_tensors": 0,
            "target_session_kv": 0,
            "target_session_recurrent": 0,
            "target_session_io": 0,
            "prefix_cache_snapshots": 0,
            "target_graph_static": 0,
            "mtp_session_kv": 0,
            "mtp_sparse_graph": 0,
            "mtp_verify_state": 0,
            "mtp_verify_outputs": 0,
            "mtp_graph_static": 0,
        }

        def add_tensor(target: str, tensor: torch.Tensor | None) -> None:
            if tensor is None or tensor.device.type != device_type:
                return
            key, nbytes = _cuda_tensor_storage_bytes(tensor)
            if key in seen_storages:
                return
            seen_storages.add(key)
            totals[target] += nbytes

        def add_tree(target: str, value: object) -> None:
            # A global visited set is required for LINEAR cost: a path-local
            # set (with the id removed on unwind) re-expands every shared DAG
            # subtree once per parent path, which made one /debug/stats scrape
            # block for 37 s on the live Flash-Next server.  Global ids are
            # only safe if objects cannot be freed and their ids recycled
            # mid-traversal (that reuse once made a real 256K session
            # disappear from the report), so pin every visited object for the
            # duration of this traversal.
            visited: set[int] = set()
            pinned: list[object] = []

            def visit(item: object) -> None:
                if item is None:
                    return
                if torch.is_tensor(item):
                    add_tensor(target, item)
                    return
                obj_id = id(item)
                if obj_id in visited:
                    return
                visited.add(obj_id)
                pinned.append(item)
                if isinstance(item, Mapping):
                    for child in item.values():
                        visit(child)
                    return
                if isinstance(item, (list, tuple, set)):
                    for child in item:
                        visit(child)
                    return
                if not hasattr(item, "__dict__"):
                    return
                for child in vars(item).values():
                    visit(child)

            visit(value)

        def tree_bytes(value: object) -> int:
            """Count one object tree with storage-level de-duplication.

            ``memory_breakdown`` uses a process-wide set for the authoritative
            totals above.  Per-slot diagnostics need an independent set so a
            shared view is not reported as zero for every slot after the first
            one; this helper intentionally does not mutate that authoritative
            set.
            """

            slot_storages: set[int] = set()
            slot_objects: set[int] = set()

            def visit(item: object) -> int:
                if item is None:
                    return 0
                if torch.is_tensor(item):
                    if item.device.type != device_type:
                        return 0
                    key, nbytes = _cuda_tensor_storage_bytes(item)
                    if key in slot_storages:
                        return 0
                    slot_storages.add(key)
                    return nbytes
                obj_id = id(item)
                if obj_id in slot_objects:
                    return 0
                slot_objects.add(obj_id)
                if isinstance(item, Mapping):
                    return sum(visit(child) for child in item.values())
                if isinstance(item, (list, tuple, set)):
                    return sum(visit(child) for child in item)
                if not hasattr(item, "__dict__"):
                    return 0
                return sum(visit(child) for child in vars(item).values())

            return visit(value)

        for tensor in self.model.parameters():
            add_tensor("model_parameters", tensor)
        for tensor in self.model.buffers():
            add_tensor("model_buffers", tensor)
        # Expert adapters and workspace arenas deliberately are lightweight
        # Python wrappers rather than nn.Module children.  Walk the complete
        # object graph after registered parameters so packed NVFP4 weights,
        # block scales, and shared arenas are not mistaken for allocator
        # overhead.  Storage-level de-duplication keeps this exact even when
        # a tensor is reachable through several layer wrappers.
        add_tree("model_auxiliary_tensors", self.model)
        mtp_model = getattr(self, "_mtp_model", None)
        if mtp_model is not None:
            for tensor in mtp_model.parameters():
                add_tensor("mtp_model_parameters", tensor)
            for tensor in mtp_model.buffers():
                add_tensor("mtp_model_buffers", tensor)
            add_tree("mtp_auxiliary_tensors", mtp_model)

        for slot_index, target in enumerate(self._targets):
            sess = target.sess
            session_kv = (
                sess.qsa_k_pool,
                sess.qsa_v_pool,
                sess.qsa_idx_k_pool,
                sess.qsa_pooled_k_pool,
                getattr(sess, "qsa_k_scale_pool", None),
                getattr(sess, "qsa_v_scale_pool", None),
                sess.qsa_k,
                sess.qsa_v,
                sess.qsa_idx_k,
            )
            session_recurrent = (sess.gdn, sess.ple_conv_state)
            session_io = (
                sess.token_buf,
                sess.pos_buf,
                sess.ends_buf,
                sess.hc_hidden_buf,
                sess.ple_emb_buf,
            )
            add_tree("target_session_kv", session_kv)
            add_tree("target_session_recurrent", session_recurrent)
            add_tree("target_session_io", session_io)
            prefix_entries = self._prefix_entries_for_slot(slot_index)
            add_tree("prefix_cache_snapshots", prefix_entries)
            add_tree("target_graph_static", target._logits)  # noqa: SLF001 - debug accounting
            # These fields deliberately use a per-slot storage set.  They are
            # diagnostic, not part of ``explicit_tensor_bytes``: if a future
            # implementation shares a backing arena, the global totals above
            # remain de-duplicated while this view still exposes each slot's
            # logical reservation.
            totals[f"target_session_kv_slot_{slot_index}"] = tree_bytes(session_kv)
            totals[f"target_session_recurrent_slot_{slot_index}"] = tree_bytes(session_recurrent)
            totals[f"target_session_io_slot_{slot_index}"] = tree_bytes(session_io)
            totals[f"prefix_cache_snapshots_slot_{slot_index}"] = tree_bytes(prefix_entries)

        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            add_tree(
                "target_graph_static",
                (
                    getattr(mlp, "_router_weights", None),
                    getattr(mlp, "_router_ids", None),
                    getattr(mlp, "_prefill_graphs", None),
                ),
            )

        for slot_index, spec in enumerate(self._specs):
            if spec is None:
                continue
            mtp_sess = spec.mtp_session
            mtp_session_kv = (
                mtp_sess.mtp_k_pool,
                mtp_sess.mtp_v_pool,
                mtp_sess.mtp_idx_k_pool,
                mtp_sess.mtp_pooled_k_pool,
                getattr(mtp_sess, "mtp_k_scale_pool", None),
                getattr(mtp_sess, "mtp_v_scale_pool", None),
                getattr(mtp_sess, "shared_sparse_indices", None),
                getattr(mtp_sess, "shared_sparse_valid", None),
            )
            mtp_sparse_graph = getattr(mtp_sess, "sparse_graph_buffers", None)
            add_tree("mtp_session_kv", mtp_session_kv)
            add_tree("mtp_sparse_graph", mtp_sparse_graph)
            verify = spec.verify
            gdn_commit_inputs = getattr(verify, "_gdn_commit_inputs", None)
            add_tree(
                "mtp_verify_state",
                (verify.buffers, gdn_commit_inputs),
            )
            add_tree(
                "mtp_verify_outputs",
                (verify._hc_hidden, verify._logits),  # noqa: SLF001 - debug accounting
            )
            add_tree(
                "mtp_graph_static",
                (
                    spec.mtp_continuation_graph,
                    tuple(spec.mtp_proposal_graphs.values()),
                ),
            )
            totals[f"mtp_session_kv_slot_{slot_index}"] = tree_bytes(mtp_session_kv)
            totals[f"mtp_sparse_graph_slot_{slot_index}"] = tree_bytes(mtp_sparse_graph)
            totals[f"mtp_verify_state_slot_{slot_index}"] = tree_bytes(
                (verify.buffers, gdn_commit_inputs)
            )
            totals[f"mtp_verify_outputs_slot_{slot_index}"] = tree_bytes(
                (verify._hc_hidden, verify._logits)  # noqa: SLF001 - debug accounting
            )

        model_total = (
            totals["model_parameters"]
            + totals["model_buffers"]
            + totals["model_auxiliary_tensors"]
            + totals["mtp_model_parameters"]
            + totals["mtp_model_buffers"]
            + totals["mtp_auxiliary_tensors"]
        )
        target_total = (
            totals["target_session_kv"]
            + totals["target_session_recurrent"]
            + totals["target_session_io"]
            + totals["target_graph_static"]
        )
        prefix_total = totals["prefix_cache_snapshots"]
        mtp_total = (
            totals["mtp_session_kv"]
            + totals["mtp_sparse_graph"]
            + totals["mtp_verify_state"]
            + totals["mtp_verify_outputs"]
            + totals["mtp_graph_static"]
        )
        explicit_total = model_total + target_total + mtp_total + prefix_total
        totals["model_tensor_bytes"] = model_total
        totals["target_session_tensor_bytes"] = target_total
        totals["mtp_session_tensor_bytes"] = mtp_total
        totals["session_tensor_bytes"] = target_total + mtp_total + prefix_total
        totals["explicit_tensor_bytes"] = explicit_total
        totals["torch_allocated"] = torch.cuda.memory_allocated(device_type)
        totals["torch_reserved"] = torch.cuda.memory_reserved(device_type)
        totals["torch_reserved_unattributed"] = max(
            0, totals["torch_reserved"] - totals["explicit_tensor_bytes"]
        )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_type)
            totals["driver_free_bytes"] = int(free_bytes)
            totals["driver_total_bytes"] = int(total_bytes)
            totals["driver_used_bytes"] = int(total_bytes - free_bytes)
            totals["driver_used_non_torch_bytes"] = max(
                0, totals["driver_used_bytes"] - totals["torch_reserved"]
            )
        except Exception:  # pragma: no cover - best-effort observability
            pass
        try:
            stats = torch.cuda.memory_stats(device_type)
            totals["torch_active_bytes"] = int(stats.get("active_bytes.all.current", 0))
            totals["torch_inactive_split_bytes"] = int(
                stats.get("inactive_split_bytes.all.current", 0)
            )
            totals["torch_non_releasable_bytes"] = int(
                stats.get("non_releasable_bytes.all.current", 0)
            )
            totals["torch_peak_allocated"] = int(stats.get("allocated_bytes.all.peak", 0))
            totals["torch_peak_reserved"] = int(stats.get("reserved_bytes.all.peak", 0))
        except Exception:  # pragma: no cover - best-effort observability
            pass
        return totals

    def ple_stats(self) -> dict[str, int | float | str | bool]:
        """Expose the live PLE cache counters for performance attribution."""
        table = getattr(self.model, "ple_table", None)
        if table is None:
            return {}
        snapshot = getattr(table, "stats_snapshot", None)
        if snapshot is None:
            return {}
        return snapshot()

    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        if params.is_greedy:
            return int(logits.argmax(dim=-1).item())
        generator = make_generator(params.seed)
        return int(sample_from_logits(logits.unsqueeze(0), params, generator=generator).item())

    def _trim_prefill_cuda_cache(self, prompt_tokens: int) -> None:
        """Return one-shot large-M allocator blocks after a long prefill.

        Flash-Next's target MoE and the initial MTP teacher sync both use
        eager grouped kernels whose temporary workspaces are sized to the
        prompt.  The CUDA caching allocator otherwise keeps those blocks
        resident for the process lifetime; after a few different long
        prompts that can consume the graph's remaining headroom even though
        no live tensor references them.  The fixed target/MTP graph pools are
        already live and are unaffected by ``empty_cache``.  Keep short
        prompts on the hot path and make the threshold/operator explicit for
        production tuning.
        """
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        threshold = max(
            0,
            int(os.environ.get("QSR_FLASHNEXT_TRIM_PREFILL_CACHE_TOKENS", "2048")),
        )
        if prompt_tokens < threshold or os.environ.get(
            "QSR_FLASHNEXT_TRIM_PREFILL_CACHE", "1"
        ).strip().lower() in {"0", "false", "off"}:
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _prefill_timestamp_ns(self) -> int:
        """Return a wall timestamp, optionally after draining GPU work.

        CUDA launches are asynchronous, so ordinary host timers only measure
        enqueue time.  The profiling switch makes the per-stage counters in
        ``/debug/stats`` authoritative while leaving production runs free of
        the synchronization overhead.
        """
        enabled = os.environ.get("QSR_FLASHNEXT_PROFILE_PREFILL", "0").strip().lower()
        if enabled not in {"", "0", "false", "off", "1", "true", "on"}:
            raise ValueError(f"QSR_FLASHNEXT_PROFILE_PREFILL must be 0 or 1, got {enabled!r}")
        if enabled in {"1", "true", "on"} and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter_ns()

    def _prefill_slot(
        self,
        slot: int,
        prompt_ids: list[int],
        *,
        params: SamplingParams | None = None,
        forced_token: int | None = None,
        chunk_size: int = 0,
        vision_inputs: object | None = None,
    ) -> dict[str, object]:
        if not prompt_ids:
            raise ValueError("Flash-Next prefill needs at least one token")
        if len(prompt_ids) > self.max_seq_len:
            raise ValueError(
                f"prompt length {len(prompt_ids)} exceeds max_seq_len={self.max_seq_len}"
            )
        vision_cache_key = self.prefix_cache_key_for_vision_inputs(vision_inputs)
        cacheable_vision = vision_cache_key != (_UNCACHEABLE_VISION_PREFIX_KEY,)
        if vision_inputs is not None and not cacheable_vision:
            # Token ids alone do not authenticate image pixels.  Do not reuse
            # a text prefix for a multimodal request (or publish one for a
            # later request); a vision-aware cache key can be added separately
            # without weakening this correctness boundary.
            self._drop_prefix_snapshot(slot)
            self._pending_prefix_hits.pop(slot, None)
            prefix_hit, prefix_entry = 0, None
            self._reset_runtime(slot, preserve_prefix=False)
        else:
            request_cache_key = self.prefix_cache_key_for_sampling(
                vision_cache_key,
                sampled=params is not None and not params.is_greedy,
            )
            prefix_hit, prefix_entry = self._prepare_prefill_prefix(
                slot,
                prompt_ids,
                prefix_cache_key=request_cache_key,
            )
        suffix_ids = prompt_ids[prefix_hit:]
        target = self._targets[slot]
        spec = self._specs[slot]
        sampled = params is not None and not params.is_greedy
        decode_mode = "sampled" if sampled else "greedy"
        prefix_mtp_resync = (
            spec is not None
            and prefix_hit > 0
            and prefix_entry is not None
            and prefix_entry.mtp_ready
            and prefix_entry.decode_mode == decode_mode
            and prefix_entry.mtp_teacher_hidden is not None
        )
        # Historical target-only checkpoints normally retain the real-prefix
        # MTP cursor as well.  Very old entries may not; serving those from a
        # target checkpoint is still correct, but MTP must stay disabled for
        # that request because its causal prefix would otherwise be missing.
        prefix_mtp_state_ready = bool(
            prefix_entry is not None
            and (prefix_entry.mtp_prefix_ready or prefix_entry.mtp_ready)
        )
        use_mtp = spec is not None and (prefix_hit == 0 or prefix_mtp_state_ready)

        if prefix_hit == len(prompt_ids):
            if prefix_entry is None:
                raise RuntimeError("Flash-Next full prefix hit has no checkpoint")
            logits = prefix_entry.anchor_logits
            anchor = (
                int(forced_token)
                if forced_token is not None
                else self._sample(logits, params or SamplingParams())
            )
            greedy = params is None or params.is_greedy
            if (
                sampled
                and spec is not None
                and prefix_entry.mtp_ready
                and prefix_entry.decode_mode == "sampled"
            ):
                teacher_hidden = prefix_entry.mtp_teacher_hidden
                if teacher_hidden is None:
                    raise RuntimeError(
                        "Flash-Next sampled prefix hit is missing the target teacher hidden row"
                    )
                # Reuse the retained real MTP prefix rows, but sample a fresh
                # anchor-conditioned draft and q distribution for this
                # request.  Replaying cached sampled token ids would couple
                # independent requests to the old request's RNG stream.
                drafts = spec.sync_and_propose(
                    [anchor],
                    teacher_hidden,
                    params=params,
                )
            elif (
                spec is not None
                and use_mtp
                and prefix_entry.mtp_prefix_ready
                and prefix_entry.mtp_teacher_hidden is not None
            ):
                # A historical target-only checkpoint has the real-prefix
                # MTP cursor but no cached proposal. Re-teacher-force the
                # boundary row so an exact compacted-prefix hit still keeps
                # speculative decode instead of falling back to one-token
                # target decode.
                drafts = spec.sync_and_propose(
                    [anchor],
                    prefix_entry.mtp_teacher_hidden,
                    params=params if sampled else None,
                )
            else:
                drafts = (
                    list(prefix_entry.draft_tokens)
                    if forced_token is None and greedy and prefix_entry.mtp_ready
                    else []
                )
            self._slot_tokens[slot] = list(prompt_ids)
            self._last_logits[slot] = logits
            self.stats["prefill_requests"] += 1
            self.stats["prefill_tokens"] += len(prompt_ids)
            self.stats["prefill_last_chunks"] = 0
            self.stats["prefill_last_tokens"] = len(prompt_ids)
            self.stats["prefix_cache_full_hits"] = self.stats.get("prefix_cache_full_hits", 0) + 1
            return {"anchor": anchor, "draft_tokens": drafts}

        multimodal_embeds: torch.Tensor | None = None
        has_multimodal = vision_inputs is not None
        suffix_embeds: torch.Tensor | None = None
        suffix_rope_positions: torch.Tensor | None = None
        if has_multimodal:
            multimodal_embeds = self.model.encode_multimodal(prompt_ids, vision_inputs)
            suffix_embeds = multimodal_embeds[prefix_hit:]
            rope_positions = getattr(vision_inputs, "rope_positions", None)
            if rope_positions is not None:
                suffix_rope_positions = torch.as_tensor(rope_positions)[:, prefix_hit:]
        # Keep activation and NEXTN teacher-sync working sets bounded for
        # 256K prompts.  Each target chunk advances the same persistent
        # recurrent/QSA state as one full prefill, so no state is duplicated.
        effective_chunk = int(chunk_size)
        if effective_chunk < 0:
            raise ValueError(f"prefill chunk_size must be non-negative, got {chunk_size}")
        chunked = effective_chunk > 0 and effective_chunk < len(suffix_ids)
        if suffix_embeds is not None and suffix_embeds.device.type == "cuda" and chunked:
            # The target prefill consumes one chunk at a time.  Keep the
            # multimodal rows on the host between chunks, but retain the
            # explicit ``has_multimodal`` flag below: dropping the CUDA owner
            # must not make the target silently fall back to token embeddings
            # for image-marker rows.
            suffix_embeds = suffix_embeds.to("cpu")
            multimodal_embeds = None
        final_hidden: torch.Tensor | None = None
        final_chunk_start = 0
        mtp_teacher_hidden: torch.Tensor | None = None
        target_ns = 0
        mtp_sync_ns = 0
        mtp_draft_ns = 0
        chunks = 1
        prefill_layer_ns: dict[str, int] = {}
        prefill_op_ns: dict[str, int] = {}

        def merge_prefill_layer_profile() -> None:
            for layer_name, elapsed_ns in getattr(target, "_last_prefill_layer_ns", {}).items():
                prefill_layer_ns[layer_name] = prefill_layer_ns.get(layer_name, 0) + int(elapsed_ns)
            for op_name, elapsed_ns in getattr(target, "_last_prefill_op_ns", {}).items():
                prefill_op_ns[op_name] = prefill_op_ns.get(op_name, 0) + int(elapsed_ns)

        if not chunked:
            started = self._prefill_timestamp_ns()
            if not has_multimodal:
                logits, final_hidden = target.prefill(suffix_ids)
            else:
                prefill_kwargs = {"input_embeds": suffix_embeds}
                if suffix_rope_positions is not None:
                    prefill_kwargs["rope_positions"] = suffix_rope_positions
                rope_next_position = getattr(vision_inputs, "next_rope_position", None)
                if rope_next_position is not None:
                    prefill_kwargs["rope_next_position"] = rope_next_position
                logits, final_hidden = target.prefill(suffix_ids, **prefill_kwargs)
            merge_prefill_layer_profile()
            target_ns += self._prefill_timestamp_ns() - started
        else:
            logits = None
            mtp_enabled = use_mtp
            # The full fused embedding matrix is only needed for one target
            # chunk at a time. Move it back to host memory for long image
            # prompts so a 256K multimodal request does not retain another
            # ~1.25 GiB BF16 allocation on the already-full card.
            if suffix_embeds is not None and suffix_embeds.device.type == "cuda":
                suffix_embeds = suffix_embeds.to("cpu")
            chunks = (len(suffix_ids) + effective_chunk - 1) // effective_chunk
            # PLE reads are submitted one chunk at a time by the model's
            # standalone prefill API.  On a long cold prompt that leaves the
            # NVMe request for chunk N+1 idle until all of chunk N (including
            # any teacher sync) has completed.  Keep exactly one immutable
            # pending gather ahead of the GPU: the worker processes the
            # current read first, then the next read can overlap the current
            # transformer pass without unbounded host/page-cache growth.
            ahead_raw = os.environ.get("QSR_FLASHNEXT_PLE_AHEAD_PREFETCH", "1")
            ahead_enabled = ahead_raw.strip().lower() not in {"0", "false", "off", "no"}
            start_ple_prefetch = (
                getattr(target, "start_ple_prefetch", None) if ahead_enabled else None
            )
            ple_pending = None

            def schedule_ple(start_offset: int, end_offset: int):
                if not callable(start_ple_prefetch):
                    return None
                ngram_size = int(getattr(self.model.cfg, "ngram_size", 0))
                if start_offset == 0:
                    history_tokens = [
                        *getattr(target.sess, "window", ()),
                        *suffix_ids[:end_offset],
                    ]
                else:
                    history_start = max(0, start_offset - ngram_size)
                    history_tokens = suffix_ids[history_start:end_offset]
                return start_ple_prefetch(
                    suffix_ids[start_offset:end_offset],
                    history_tokens=history_tokens,
                    prefix_hint=start_offset == 0,
                )

            first_end = min(effective_chunk, len(suffix_ids))
            ple_pending = schedule_ple(0, first_end)
            for start in range(0, len(suffix_ids), effective_chunk):
                end = min(start + effective_chunk, len(suffix_ids))
                final = end == len(suffix_ids)
                next_start = end
                next_pending = None
                if next_start < len(suffix_ids):
                    next_end = min(next_start + effective_chunk, len(suffix_ids))
                    # Submit before launching the current target work.  The
                    # PLE executor is FIFO, so this remains bounded at one
                    # chunk while allowing the next read to run during the
                    # current chunk's layer-0..47 compute.
                    next_pending = schedule_ple(next_start, next_end)
                started = self._prefill_timestamp_ns()
                prefill_kwargs = {}
                if ple_pending is not None:
                    prefill_kwargs["_ple_pending"] = ple_pending
                if not has_multimodal:
                    logits, hidden_rows = target.prefill(suffix_ids[start:end], **prefill_kwargs)
                else:
                    prefill_kwargs = {"input_embeds": suffix_embeds[start:end]}
                    if ple_pending is not None:
                        prefill_kwargs["_ple_pending"] = ple_pending
                    if suffix_rope_positions is not None:
                        prefill_kwargs["rope_positions"] = suffix_rope_positions[:, start:end]
                    if final:
                        rope_next_position = getattr(vision_inputs, "next_rope_position", None)
                        if rope_next_position is not None:
                            prefill_kwargs["rope_next_position"] = rope_next_position
                    logits, hidden_rows = target.prefill(suffix_ids[start:end], **prefill_kwargs)
                merge_prefill_layer_profile()
                target_ns += self._prefill_timestamp_ns() - started
                ple_pending = next_pending
                if mtp_enabled and not final:
                    # The next real token is available for teacher forcing;
                    # discard the first draft/hidden immediately and retain
                    # only the MTP session's updated state.
                    started = self._prefill_timestamp_ns()
                    sync_kwargs = {}
                    sync_tokens = suffix_ids[start + 1 : end + 1]
                    sync_hidden = hidden_rows
                    if prefix_mtp_resync and start == 0:
                        # The cached proposal's final row represented the old
                        # anchor.  Replace it with the first real suffix token
                        # before syncing the remaining rows of this chunk.
                        sync_tokens = suffix_ids[: end + 1]
                        sync_hidden = torch.cat(
                            [prefix_entry.mtp_teacher_hidden, hidden_rows], dim=0
                        )
                    if suffix_embeds is not None:
                        if prefix_mtp_resync and start == 0:
                            sync_kwargs["input_embeds"] = suffix_embeds[: end + 1]
                        else:
                            sync_kwargs["input_embeds"] = suffix_embeds[start + 1 : end + 1]
                    spec.sync_real_suffix(
                        sync_tokens,
                        sync_hidden,
                        **sync_kwargs,
                    )
                    mtp_sync_ns += self._prefill_timestamp_ns() - started
                    absolute_end = prefix_hit + end
                    absolute_start = prefix_hit + start
                    checkpoint_interval = self._prefix_checkpoint_interval()
                    crossed_checkpoint = (
                        absolute_end >= checkpoint_interval
                        and absolute_end // checkpoint_interval
                        > absolute_start // checkpoint_interval
                    )
                    if crossed_checkpoint and cacheable_vision:
                        # At this point target state and real-prefix MTP rows
                        # are both at ``absolute_end``.  Do not run a draft
                        # proposal here: capturing it would consume an anchor
                        # that the next chunk still needs.  The checkpoint is
                        # therefore explicitly target-only, while preserving
                        # the MTP teacher cursor for a later extension.
                        self._capture_prefix_snapshot(
                            slot,
                            prompt_ids[:absolute_end],
                            anchor=int(logits.argmax(dim=-1).item()),
                            draft_tokens=[],
                            anchor_logits=logits,
                            vision_cache_key=vision_cache_key,
                            mtp_ready=False,
                            mtp_prefix_ready=True,
                            target_only_reusable=True,
                            decode_mode=decode_mode,
                            mtp_teacher_hidden=hidden_rows[-1:].detach(),
                        )
                elif final:
                    final_hidden = hidden_rows
                    final_chunk_start = start
                del hidden_rows
            assert logits is not None
            if mtp_enabled and final_hidden is None:
                raise RuntimeError("chunked Flash-Next prefill produced no final hidden rows")
        if forced_token is not None:
            anchor = int(forced_token)
        else:
            anchor = self._sample(logits, params or SamplingParams())

        drafts: list[int] = []
        if use_mtp:
            if final_hidden is not None:
                # The target hidden row at the prompt boundary lets a full
                # prefix hit re-teacher-force the first sampled anchor while
                # retaining the MTP real-prefix pools in place.
                mtp_teacher_hidden = final_hidden[-1:].detach().clone()
            if suffix_embeds is not None and suffix_embeds.device.type == "cuda":
                # Target prefill is complete.  Keep only a host copy for the
                # teacher-forced MTP sync so the full multimodal matrix does
                # not remain resident on the GPU while MTP allocates its
                # temporary rows.
                suffix_embeds = suffix_embeds.to("cpu")
                multimodal_embeds = None
            if not chunked:
                assert final_hidden is not None
                started = self._prefill_timestamp_ns()
                if prefix_mtp_resync:
                    shifted = [*suffix_ids, anchor]
                    sync_hidden = torch.cat([prefix_entry.mtp_teacher_hidden, final_hidden], dim=0)
                else:
                    shifted = [*suffix_ids[1:], anchor]
                    sync_hidden = final_hidden
                sync_kwargs = {}
                if suffix_embeds is not None:
                    if prefix_mtp_resync:
                        anchor_embed = self.model.embed_tokens(
                            torch.tensor([int(anchor)], dtype=torch.long, device=self.device)
                        )[0].to(device=suffix_embeds.device, dtype=suffix_embeds.dtype)
                        sync_kwargs["input_embeds"] = torch.cat(
                            [suffix_embeds, anchor_embed.unsqueeze(0)], dim=0
                        )
                    else:
                        sync_kwargs["input_embeds"] = self._shift_teacher_force_embeds(
                            suffix_embeds,
                            anchor,
                        )
                if sampled:
                    drafts = spec.sync_and_propose(
                        shifted,
                        sync_hidden,
                        params=params,
                        **sync_kwargs,
                    )
                else:
                    drafts = spec.sync_and_propose(shifted, sync_hidden, **sync_kwargs)
                mtp_sync_ns += self._prefill_timestamp_ns() - started
            else:
                # NEXTN consumes the hidden row for each target position and
                # the token immediately following it.  Sync one chunk at a
                # time so a 256K prompt never materialises a multi-GB hidden
                # matrix or retains allocator blocks of that size.
                assert final_hidden is not None
                if prefix_mtp_resync and final_chunk_start == 0:
                    shifted = suffix_ids + [anchor]
                    sync_hidden = torch.cat([prefix_entry.mtp_teacher_hidden, final_hidden], dim=0)
                else:
                    shifted = suffix_ids[final_chunk_start + 1 :] + [anchor]
                    sync_hidden = final_hidden
                started = self._prefill_timestamp_ns()
                sync_kwargs = {}
                if suffix_embeds is not None:
                    if prefix_mtp_resync and final_chunk_start == 0:
                        anchor_embed = self.model.embed_tokens(
                            torch.tensor([int(anchor)], dtype=torch.long, device=self.device)
                        )[0].to(device=suffix_embeds.device, dtype=suffix_embeds.dtype)
                        sync_kwargs["input_embeds"] = torch.cat(
                            [suffix_embeds, anchor_embed.unsqueeze(0)], dim=0
                        )
                    else:
                        sync_kwargs["input_embeds"] = self._shift_teacher_force_embeds(
                            suffix_embeds[final_chunk_start:],
                            anchor,
                        )
                if sampled:
                    drafts = spec.sync_and_propose(
                        shifted,
                        sync_hidden,
                        params=params,
                        **sync_kwargs,
                    )
                    mtp_sync_ns += self._prefill_timestamp_ns() - started
                else:
                    first, first_hidden = spec.sync_real_suffix(
                        shifted,
                        sync_hidden,
                        **sync_kwargs,
                    )
                    mtp_sync_ns += self._prefill_timestamp_ns() - started
                    started = self._prefill_timestamp_ns()
                    drafts = spec.continue_draft(first, first_hidden)
                    mtp_draft_ns += self._prefill_timestamp_ns() - started
                del final_hidden
        del multimodal_embeds
        started = self._prefill_timestamp_ns()
        self._trim_prefill_cuda_cache(len(prompt_ids))
        trim_ns = self._prefill_timestamp_ns() - started
        self._slot_tokens[slot] = list(prompt_ids)
        self._last_logits[slot] = logits
        mtp_ready = use_mtp
        if cacheable_vision:
            self._capture_prefix_snapshot(
                slot,
                prompt_ids,
                anchor=anchor,
                draft_tokens=drafts,
                anchor_logits=logits,
                vision_cache_key=vision_cache_key,
                mtp_ready=mtp_ready,
                mtp_prefix_ready=use_mtp,
                target_only_reusable=not use_mtp and prefix_hit > 0,
                decode_mode=decode_mode,
                mtp_teacher_hidden=mtp_teacher_hidden,
            )
        self.stats["prefill_requests"] += 1
        self.stats["prefill_chunks"] += chunks
        self.stats["prefill_tokens"] += len(prompt_ids)
        self.stats["prefill_target_ns"] += target_ns
        self.stats["prefill_mtp_sync_ns"] += mtp_sync_ns
        self.stats["prefill_mtp_draft_ns"] += mtp_draft_ns
        self.stats["prefill_trim_ns"] += trim_ns
        self.stats["prefill_last_chunks"] = chunks
        self.stats["prefill_last_tokens"] = len(prompt_ids)
        self.stats["prefill_last_target_ns"] = target_ns
        self.stats["prefill_last_mtp_sync_ns"] = mtp_sync_ns
        self.stats["prefill_last_mtp_draft_ns"] = mtp_draft_ns
        self.stats["prefill_last_trim_ns"] = trim_ns
        if prefill_layer_ns:
            self.stats["prefill_last_layer_ns"] = prefill_layer_ns
        if prefill_op_ns:
            self.stats["prefill_last_op_ns"] = prefill_op_ns
        return {"anchor": anchor, "draft_tokens": drafts}

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        return int(self._prefill_slot(slot, list(prompt_ids))["anchor"])

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
        force_token_ids: dict[int, int] | None = None,
        vision_inputs_per_slot: dict[int, object] | None = None,
    ) -> ChunkedPrefillState:
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        if not slots:
            return ChunkedPrefillState(done=True, result={})
        result: dict[int, dict[str, object]] = {}
        params_per_slot = params_per_slot or {}
        force_token_ids = force_token_ids or {}
        vision_inputs_per_slot = vision_inputs_per_slot or {}
        for slot, prompt in zip(slots, prompts_per_slot):
            result[slot] = self._prefill_slot(
                slot,
                list(prompt),
                params=params_per_slot.get(slot),
                forced_token=force_token_ids.get(slot),
                chunk_size=chunk_size,
                vision_inputs=vision_inputs_per_slot.get(slot),
            )
        return ChunkedPrefillState(done=True, result=result)

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        return state.done

    def _decode_one(self, slot: int, token_id: int) -> torch.Tensor:
        target = self._targets[slot]
        if target.graph is None:
            from runtime.model.flashnext.model import decode_step

            return decode_step(self.model, int(token_id), target.sess)
        return target.step(int(token_id))

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
        force_token_ids: list[int | None] | None = None,
    ) -> list[int] | tuple[list[int], list[dict]]:
        if not (len(slot_ids) == len(token_ids) == len(kv_lengths) == len(params_list)):
            raise ValueError("decode batch fields must have equal lengths")
        force_token_ids = force_token_ids or [None] * len(slot_ids)
        if len(force_token_ids) != len(slot_ids):
            raise ValueError("force_token_ids must match the decode batch length")

        logits_rows: list[torch.Tensor] = []
        next_tokens: list[int] = []
        for index, (slot, token_id, expected_len, params) in enumerate(
            zip(slot_ids, token_ids, kv_lengths, params_list)
        ):
            actual_len = int(self._target_session(slot).pos)
            if expected_len != actual_len:
                raise RuntimeError(
                    f"slot {slot}: scheduler says kv_len={expected_len}, backend has {actual_len}"
                )
            logits = self._decode_one(slot, int(token_id)).float()
            logits_rows.append(logits)
            forced = force_token_ids[index]
            if forced is not None:
                token = int(forced)
            else:
                token = self._sample(logits, params)
            next_tokens.append(token)
            self._slot_tokens[slot].append(int(token_id))

        self.stats["decode_rounds"] += 1
        self.stats["decode_tokens"] += len(next_tokens)
        if not return_logprobs:
            return next_tokens
        logprobs = [
            compute_logprobs(logits.unsqueeze(0), [token], top_k=top_logprobs)[0]
            for logits, token in zip(logits_rows, next_tokens)
        ]
        return next_tokens, logprobs

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        drafts: dict[int, list[int]],
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
        thinking_force_positions: dict[int, int] | None = None,
        thinking_force_token_ids: dict[int, int] | None = None,
    ) -> dict[int, dict]:
        params_per_slot = params_per_slot or {}
        thinking_force_positions = thinking_force_positions or {}
        thinking_force_token_ids = thinking_force_token_ids or {}
        if set(thinking_force_positions) != set(thinking_force_token_ids):
            raise ValueError(
                "Flash-Next MTP thinking forcing requires matching position/token maps"
            )
        result: dict[int, dict] = {}
        for slot in slots:
            params = params_per_slot.get(slot)
            spec = self._specs[slot]
            if spec is None:
                raise RuntimeError("Flash-Next MTP is disabled")
            decision = spec.round(
                int(anchors[slot]),
                drafts[slot],
                use_graph=self._captured and not self._mtp_verify_eager,
                return_verify_logits=return_logprobs,
                params=params,
                thinking_force_position=thinking_force_positions.get(slot),
                thinking_force_token_id=thinking_force_token_ids.get(slot),
            )
            committed = [int(token) for token in decision["committed"]]
            self._slot_tokens[slot].extend(committed)
            self.stats["mtp_rounds"] += 1
            self.stats["mtp_accepted_tokens"] += int(decision.get("num_accepted", 0))
            timing = decision.get("timing")
            if isinstance(timing, Mapping):
                for source, target, last_target in (
                    ("ple", "mtp_ple_ns", "mtp_last_ple_ns"),
                    ("verify", "mtp_verify_ns", "mtp_last_verify_ns"),
                    ("mtp", "mtp_proposal_ns", "mtp_last_proposal_ns"),
                ):
                    seconds = max(0.0, float(timing.get(source, 0.0)))
                    elapsed_ns = int(seconds * 1_000_000_000)
                    self.stats[target] += elapsed_ns
                    self.stats[last_target] = elapsed_ns
            if return_logprobs:
                logits = decision.get("verify_logits")
                if not torch.is_tensor(logits):
                    raise RuntimeError("Flash-Next verify did not return logits for logprobs")
                decision["logprobs"] = compute_logprobs(
                    logits,
                    committed,
                    top_k=top_logprobs,
                )
            result[slot] = decision
        return result

    def capture_decode_cuda_graph(self) -> int | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        try:
            for target in self._targets:
                target.capture()
            for spec in self._specs:
                if spec is not None:
                    spec.capture_verify()
            for slot in range(self.num_slots):
                self._reset_runtime(slot)
            self._captured = True
            self._cg_status["decode"] = "captured"
            if self.enable_mtp:
                self._cg_status["mtp_verify"] = "captured"
            return self.num_slots
        except Exception:
            self._captured = False
            self._cg_status["decode"] = "failed"
            if self.enable_mtp:
                self._cg_status["mtp_verify"] = "failed"
            logger.exception("Flash-Next CUDA Graph capture failed; using eager fallback")
            for slot in range(self.num_slots):
                try:
                    self._reset_runtime(slot)
                except Exception:
                    logger.exception("Flash-Next reset failed after graph capture error")
            return None

    def capture_prefill_mlp_graphs(self, rows: int) -> None:
        """Capture the shared fixed-row MLP graph used by target prefill."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Flash-Next prefill MLP CUDA Graph requires CUDA")
        if not self._targets:
            raise RuntimeError("Flash-Next prefill MLP CUDA Graph has no target slots")
        self._targets[0].capture_prefill_mlp_graphs(rows)

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        return self.reconcile_prefix_hit_with_key(token_ids, None)

    def reconcile_prefix_hit_with_key(
        self,
        token_ids: list[int],
        prefix_cache_key: tuple[str, ...] | None,
    ) -> PrefixHit:
        """Match the deepest retained same-slot Flash-Next checkpoint."""
        if not self.enable_prefix_cache or not token_ids:
            return PrefixHit(kv_hit=0, state_hit=0)
        best_slot = -1
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in range(self.num_slots):
            if not self.slot_state(slot).is_fresh:
                continue
            hit = self._prefix_hit_for_slot(token_ids, slot, prefix_cache_key=prefix_cache_key)
            if hit.effective > best.effective:
                best_slot, best = slot, hit
        if best_slot >= 0 and best.effective > 0:
            self._pending_prefix_hits[best_slot] = (
                best.effective,
                tuple(token_ids),
                prefix_cache_key,
            )
            self.stats["prefix_cache_hit_tokens"] = (
                self.stats.get("prefix_cache_hit_tokens", 0) + best.effective
            )
        else:
            self.stats["prefix_cache_misses"] = self.stats.get("prefix_cache_misses", 0) + 1
        return best

    def find_best_slot_for_prompt(
        self,
        token_ids: list[int],
        free_slots: list[int],
    ) -> tuple[int, int]:
        return self.find_best_slot_for_prompt_with_key(token_ids, free_slots, None)

    def find_best_slot_for_prompt_with_key(
        self,
        token_ids: list[int],
        free_slots: list[int],
        prefix_cache_key: tuple[str, ...] | None,
    ) -> tuple[int, int]:
        if not free_slots:
            raise ValueError("find_best_slot_for_prompt requires a free slot")
        if not self.enable_prefix_cache:
            return free_slots[0], 0
        best_slot = free_slots[0]
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in free_slots:
            hit = self._prefix_hit_for_slot(token_ids, slot, prefix_cache_key=prefix_cache_key)
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best_slot, best = slot, hit
        return best_slot, best.effective

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if close is not None:
            close()
