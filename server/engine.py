"""``ServerEngine``: continuous-batching engine with a dedicated GPU thread.

Architecture (vLLM V1 / SGLang inspired, optimized for maximum throughput):

- A **dedicated engine thread** owns the CUDA context and runs ALL GPU
  operations (model load, prefill, MTP verify/commit). The asyncio event
  loop (FastAPI/HTTP) NEVER blocks on GPU work.
- **Request channel** (asyncio → engine): lock-free ``collections.deque``
  + ``os.pipe()`` wakeup. The engine thread blocks on ``os.read(pipe)``
  when idle — zero CPU, instant wakeup on new request.
- **Stream channel** (engine → asyncio): per-request ``deque`` buffer +
  shared ``os.pipe()`` + ``loop.add_reader()`` for minimum-latency token
  delivery to SSE generators.
- **Future resolution**: ``loop.call_soon_threadsafe()`` (unavoidable for
  asyncio futures, ~12μs per call — negligible vs. ~30ms GPU round).
- Engine thread runs back-to-back GPU rounds with ZERO asyncio overhead
  when active, maximizing MTP verify/commit throughput.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import select
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.architecture import ArchitectureSpec, parse_architecture
from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS
from runtime.model_registry import IMPLEMENTED_BACKENDS
from runtime.round_profile import round_profile
from runtime.sampling import SamplingParams
from runtime.slot_resource_manager import SlotResourceManager
from runtime.thinking_budget import ThinkingBudgetConfig, ThinkingBudgetState
from server import metrics
from server.formats.stop import find_earliest_stop_match, trim_ambiguous_stop_tail
from server.formats.stream import StreamProcessor
from server.formats.thinking import apply_qwen_default_reasoning_effort
from server.tracing import tracer

# N2: stop-sequence matching must exclude the reasoning phase (OpenAI's
# reasoning_content is not truncated by `stop` -- see _activate_slot /
# _stop_check_token). Laguna's chat template never injects <think> into
# the prompt, so thinking_capable=False here -- mirrors
# server/app.py::SERVER_THINKING_CAPABLE, which is hardcoded the same way
# for the same (currently single-model) reason. If a future model needs
# template-injected thinking, both constants need to move together.
_STOP_TRACKER_THINKING_CAPABLE = False

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# QSR_PROFILE_ADMISSION=1: per-request admission phase timings (queue wait,
# slot match, reset, reconcile, prefill, activate). Wall-clock only, zero
# allocations on the hot path when disabled -- same contract as
# runtime/round_profile.py.
_ADMISSION_PROFILE = os.environ.get("QSR_PROFILE_ADMISSION") == "1"
_adm_logger = logging.getLogger("qwen_sm120.round_profile")


def _adm_start(req: GenerationRequest) -> None:
    if _ADMISSION_PROFILE:
        req._adm_phases = []  # type: ignore[attr-defined]
        req._adm_t0 = time.perf_counter()  # type: ignore[attr-defined]
        req._adm_start_at = req._adm_t0  # type: ignore[attr-defined]


def _adm_phase(req: GenerationRequest, name: str) -> None:
    if _ADMISSION_PROFILE:
        now = time.perf_counter()
        req._adm_phases.append(  # type: ignore[attr-defined]
            (name, round((now - req._adm_t0) * 1000.0, 3))  # type: ignore[attr-defined]
        )
        req._adm_t0 = now  # type: ignore[attr-defined]


def _adm_end(req: GenerationRequest) -> None:
    if not _ADMISSION_PROFILE or not hasattr(req, "_adm_phases"):
        return
    _adm_phase(req, "activate")
    _adm_logger.info(
        json.dumps(
            {
                "label": "admission",
                "request_id": req.request_id,
                "prompt_tokens": len(req.prompt_ids),
                "wait_ms": round(
                    (req._adm_start_at - req._admitted_at) * 1000.0,
                    3,  # type: ignore[attr-defined]
                ),
                "phases": req._adm_phases,  # type: ignore[attr-defined]
            }
        )
    )


# A3 step 7-g (docs/a3-cache-coordinator-design.md §7 row 7-g): fallback for
# constructing a ServerEngine without an explicit `architecture_spec` --
# every existing test does this (they either never reach model loading, or
# hand ServerEngine a fake runner directly, bypassing `_load_laguna_model`
# entirely), and `server/app.py`'s production `lifespan()` is the only
# caller that has a real one to pass (`resolve_checkpoint(...).spec`).
#
# `SlotResourceManager.__init__` requires a real `ArchitectureSpec` -- it has
# no `None` case (see `tests/test_slot_resource_manager.py`, which never
# constructs one with `None`) -- so `self.architecture_spec` must never be
# `None` by the time `self.slot_resources` is read. A single paged-KV layer
# forces `needs_two_cache_families=False`, matching every backend this
# runtime ships today (`IMPLEMENTED_BACKENDS == frozenset({"laguna"})`,
# `tests/test_architecture_spec.py::test_laguna_has_no_recurrent_layers`),
# so the coordinator's pure-forward branch (§5 point 3) is what a caller who
# does not pass a spec gets -- i.e. the same answer `self.runner` itself
# would have given directly, never `NotImplementedError`. Built via
# `parse_architecture` (not a hand-built `ArchitectureSpec(...)`) so it goes
# through the same validated construction path production checkpoints do,
# same pattern `tests/test_slot_resource_manager.py`'s own `_spec()` helper
# uses.
_DEFAULT_ARCHITECTURE_SPEC: ArchitectureSpec = parse_architecture(
    {
        "architectures": ["ServerEngineDefaultForCausalLM"],
        "model_type": "server-engine-default",
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "hidden_size": 8,
        "vocab_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
)


# -- D1/C5: Pure detection predicates (extracted for unit-testability) --------


def find_timed_out_slots(active: dict[int, dict], now: float, timeout_s: float) -> list[int]:
    """Return slot IDs whose requests have exceeded the timeout duration."""
    if timeout_s <= 0:
        return []
    return [s for s, st in active.items() if now - st.get("start_time", now) > timeout_s]


def find_stale_slots(
    active: dict[int, dict], current_round: int, max_stale_rounds: int
) -> list[int]:
    """Return slot IDs that made no progress for too many consecutive rounds."""
    if max_stale_rounds <= 0:
        return []
    return [
        s
        for s, st in active.items()
        if current_round - st.get("last_progress_round", 0) > max_stale_rounds
    ]


def classify_decode_slots(
    active_slots: list[int],
    active: dict[int, dict],
    grammar_slots: list[int],
    mtp_capable: bool,
    sampled_mtp_capable: bool = True,
) -> tuple[list[int], list[int]]:
    """Split one round's active slots into (mtp_slots, plain_sampled_slots).

    Before E2-b (docs/e2e-and-quality-plan.md §2.2), speculative verify/
    commit only ever applied to a GREEDY request on a backend whose runner
    reports ``spec.has_mtp`` -- every ``temperature>0`` request was routed
    to the plain per-step ``decode_batch_sampled`` path instead, silently
    losing all speculative acceleration the moment a caller asked for
    sampling (roadmap Track E, S8 -- "the most obvious functional gap").
    E2-b closed that: ``DFlashEngine.dflash_round`` now resolves accept/
    reject via rejection sampling (``runtime.mtp_accept.sample_accept_reject``)
    for non-greedy requests instead of an argmax comparison, so ``mtp_slots``
    below carries BOTH greedy and sampled requests whenever the backend is
    MTP-capable -- greedy vs sampled is no longer a routing distinction here.

    Only grammar-constrained requests (structured output has no mask hook
    into the speculative verify step yet -- see E-N1,
    ``docs/api-layer-design.md`` §7.1), sampled requests on a backend whose
    verify implementation cannot safely sample, or every request on a
    backend with no MTP capability at all go through the plain
    ``decode_batch_sampled`` path (``plain_sampled_slots``).
    """
    if not mtp_capable:
        return [], list(active_slots)
    mtp_slots = [
        s
        for s in active_slots
        if s not in grammar_slots
        and active[s].get("speculative_enabled", True)
        and (sampled_mtp_capable or not active[s].get("sampled", False))
    ]
    plain_sampled_slots = [s for s in active_slots if s not in mtp_slots]
    return mtp_slots, plain_sampled_slots


logger = logging.getLogger("qwen_sm120_server.engine")

_PREFIX_OVERLAP_HISTORY = 64
_PREFIX_OVERLAP_SAMPLES_KEPT = 200
_PREFIX_CACHE_HIT_SAMPLES_KEPT = 200
_SESSION_WARM_CONTINUATION_SAMPLES_KEPT = 200


def _cuda_graph_extra_slots(
    *, backend: str, enable_cudagraph: bool, enable_dflash: bool
) -> int:
    """Return dedicated server slots required only for graph capture."""
    if not enable_cudagraph or enable_dflash:
        return 0
    if backend in {"deepseek_v4", "qwen36", "flashnext"}:
        return 0
    return 1


def _qwen_kv_bundle_bytes(model: Any, *, include_mtp: bool, page_size: int = 128) -> int:
    """Return exact K+V bytes carried by one Qwen arena bundle.

    This mirrors the actual tensor constructors in ``Qwen36SlotPool`` and
    ``build_pooled_mtp_caches``. It intentionally inspects the loaded model's
    dtypes/geometries instead of baking the current Qwen3.8 constants into a
    deployment knob.
    """
    import torch

    def attention_bytes(attn: Any) -> int:
        element_size = torch.empty((), dtype=attn.kv_cache_dtype).element_size()
        return 2 * page_size * attn.num_kv_heads * attn.head_dim * element_size

    total = sum(
        attention_bytes(layer.self_attn)
        for layer in model.model.layers
        if layer.layer_type != "linear_attention"
    )
    if include_mtp:
        if model.mtp is None:
            raise ValueError("MTP KV budgeting requested but the loaded model has no MTP head")
        total += attention_bytes(model.mtp.layers[0].self_attn)
    if total <= 0:
        raise ValueError("loaded Qwen model has no paged-attention KV tensors to budget")
    return total


def _longest_common_prefix_len(a: list[int], b: list[int], cap: int | None = None) -> int:
    """Length of the common prefix of ``a`` and ``b``, optionally capped.

    ``cap`` exists for the admission-overlap diagnostic below: at
    131072-token prompts a full Python-level scan costs ~3.5 ms per pair,
    and scanning every recent prompt on the serving path made TTFT drift
    upward wave over wave (measured 2026-08-06: activate phase 24 -> 197 ms
    as the 64-entry history filled). The diagnostic only decides whether
    overlap crosses ``block_size``, so a capped scan is semantically
    identical for every consumer of these stats.
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
        if cap is not None and n >= cap:
            break
    return n


def _drain_pipe(fd: int) -> None:
    """Drain all pending bytes from a non-blocking pipe fd."""
    try:
        while os.read(fd, 65536):
            pass
    except (BlockingIOError, OSError):
        pass


@dataclass
class GenerationRequest:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    future: Any
    session_id: str | None = None
    stream_channel: StreamChannel | None = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    stop_sequences: list[str] | None = None
    logprobs: bool = False
    top_logprobs: int = 0
    thinking_budget: ThinkingBudgetConfig | None = None
    stop_on_tool_call: bool = False
    # Prepared CPU-side visual patches for Flash-Next. Image requests use the
    # target path only until multimodal MTP state is implemented.
    vision_inputs: Any | None = None


class StreamChannel:
    """High-performance token delivery channel (engine thread → asyncio).

    Uses a GIL-atomic deque buffer + asyncio.Event for wakeup. The engine
    thread appends token batches to the deque and signals via
    call_soon_threadsafe(event.set). The asyncio consumer awaits the event
    and drains the deque.
    """

    __slots__ = ("_buf", "_event", "_closed", "request_id")

    def __init__(self) -> None:
        self._buf: collections.deque = collections.deque()
        self._event: asyncio.Event | None = None
        self._closed = False
        self.request_id: str | None = None

    def put(self, item: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Engine thread: append item and wake up the asyncio consumer."""
        self._buf.append(item)
        if self._event is not None:
            loop.call_soon_threadsafe(self._event.set)

    def close(self, loop: asyncio.AbstractEventLoop) -> None:
        """Engine thread: signal end-of-stream."""
        self._closed = True
        self._buf.append(None)
        if self._event is not None:
            loop.call_soon_threadsafe(self._event.set)

    async def get(self) -> Any:
        """Asyncio thread: get next item (blocks until available)."""
        while not self._buf:
            if self._event is None:
                self._event = asyncio.Event()
            self._event.clear()
            await self._event.wait()
        return self._buf.popleft()


class ServerEngine:
    """Owns the one ``LagunaBackend`` instance, plus the admission and
    speculative verify/commit bookkeeping for a live, continuously-batched
    service.

    Threading: a dedicated engine thread owns the CUDA context and runs all
    GPU operations. The asyncio event loop communicates via lock-free deques
    and os.pipe() wakeups for maximum throughput and minimum latency.

    ``model``/``backend`` used to be the class attributes ``MODEL``/
    ``BACKEND`` (a single hardcoded Laguna checkpoint and backend name).
    Track A migration step 5 (docs/architecture.md §3.5.5) deleted them:
    ``model`` is now a constructor default carrying the same value, and
    ``backend`` is validated against :data:`runtime.model_registry.
    IMPLEMENTED_BACKENDS` rather than a single string -- the registry's own
    notion of "which backends actually exist" instead of a private copy of
    it. ``server/app.py``'s ``lifespan()`` is registry's first real
    production consumer: it resolves ``backend`` from the checkpoint's
    ``config.json`` via ``model_registry.resolve_checkpoint`` rather than
    hardcoding it, and passes the result in here.

    A3 step 7-g (docs/a3-cache-coordinator-design.md §7 row 7-g):
    ``architecture_spec`` is the same call's ``Resolution.spec`` -- the model
    structure fact (parsed from ``config.json``) that decides whether
    :attr:`slot_resources` (a :class:`runtime.slot_resource_manager.
    SlotResourceManager`) needs a second cache-family allocator. ``None``
    (the default -- every test today, since none of them constructs a real
    ``ArchitectureSpec``) falls back to ``_DEFAULT_ARCHITECTURE_SPEC``, a
    single-paged-KV-layer stand-in that forces
    ``needs_two_cache_families=False``, matching every backend this runtime
    ships (Laguna). See that module-level constant's own comment for why
    this must never be ``None`` by the time :attr:`slot_resources` is read.
    """

    def __init__(
        self,
        *,
        model: str = "poolside/Laguna-S-2.1-NVFP4",
        backend: str = "laguna",
        architecture_spec: ArchitectureSpec | None = None,
        capacity: int = 1,
        num_slots: int = 2,
        block_size: int = 64,
        blocks_per_slot: int = 4096,
        kv_cache_dtype: str = "auto",
        enable_cudagraph: bool = True,
        enable_prefix_cache: bool = False,
        enable_session_affinity: bool = False,
        session_ttl_s: float = 30.0,
        enable_dflash: bool = False,
        enable_mtp: bool = False,
        mtp_num_speculative_tokens: int = 4,
        mtp_resync: bool | None = None,
        enable_dspark: bool = False,
        dspark_draft_model: str = "RadixArk/Qwen3.8-27B-DSpark",
        dspark_num_speculative_tokens: int = 7,
        enable_dflash2: bool = False,
        dflash2_draft_model: str = "/home/bot/models/Qwen3.8-27B-DFlash2",
        dflash2_num_speculative_tokens: int = 7,
        checkpoint_budget_multiple: int | None = None,
        qwen_kv_mode: str = "legacy",
        qwen_kv_pool_bytes: int = 0,
        qwen_kv_watermark_bundles: int = 8,
        qwen_kv_full_sequence_must_fit: bool = True,
        qwen_kv_extensible: bool = False,
        qwen_kv_commit_buffer_gb: float = 10.0,
        gpu_memory_utilization: float = 0.85,
        idle_sleep_s: float = 0.005,
        production: bool = True,
        watchdog_max_stale_rounds: int = 200,
        request_timeout_s: float = 600.0,
    ) -> None:
        if backend not in IMPLEMENTED_BACKENDS:
            raise ValueError(
                f"backend={backend!r} is unsupported; implemented backends are "
                f"{sorted(IMPLEMENTED_BACKENDS)}"
            )
        self.backend_name = backend
        self.architecture_spec = (
            architecture_spec if architecture_spec is not None else _DEFAULT_ARCHITECTURE_SPEC
        )
        self.enable_dflash = enable_dflash
        # B3 (2026-08-03): MTP speculative decode, qwen36 only -- symmetric
        # with enable_dflash's own Laguna-only scope. Rejected here, at
        # construction and before any GPU work, rather than left to
        # Qwen36Backend (which does not exist for a Laguna instance) --
        # same "fail loud at construction, not deep inside a request" rule
        # enable_session_affinity's own guard above follows.
        if enable_mtp and backend not in {"qwen36", "flashnext"}:
            raise ValueError(
                f"enable_mtp requires a Qwen-family backend (got {backend!r}); "
                "MTP is implemented by qwen36 and flashnext, not Laguna "
                "(that backend uses DFlash)"
            )
        self.enable_mtp = enable_mtp
        self.mtp_num_speculative_tokens = mtp_num_speculative_tokens
        self.mtp_resync = mtp_resync
        if enable_dspark and backend != "qwen36":
            raise ValueError(
                f"enable_dspark requires backend='qwen36' (got {backend!r}); "
                "DSpark is the Qwen3.x external draft path"
            )
        if enable_dflash2 and backend != "qwen36":
            raise ValueError(
                f"enable_dflash2 requires backend='qwen36' (got {backend!r}); "
                "DFlash2 is the Qwen3.x external draft path"
            )
        if enable_dspark and enable_dflash2:
            raise ValueError("DSpark and DFlash2 are mutually exclusive")
        if enable_dspark and (enable_dflash or enable_mtp):
            raise ValueError("DSpark is mutually exclusive with DFlash and MTP")
        if enable_dflash2 and (enable_dflash or enable_mtp):
            raise ValueError("DFlash2 is mutually exclusive with DFlash and MTP")
        if enable_dflash2 and not enable_cudagraph:
            raise ValueError("DFlash2 requires CUDA Graphs; do not disable enable_cudagraph")
        if dspark_num_speculative_tokens <= 0:
            raise ValueError("dspark_num_speculative_tokens must be positive")
        if dflash2_num_speculative_tokens <= 0:
            raise ValueError("dflash2_num_speculative_tokens must be positive")
        if (enable_dspark or enable_dflash2) and enable_session_affinity:
            raise ValueError("external Qwen speculative decoding does not support session affinity")
        self.enable_dspark = enable_dspark
        self.dspark_draft_model = dspark_draft_model
        self.dspark_num_speculative_tokens = dspark_num_speculative_tokens
        self.enable_dflash2 = enable_dflash2
        self.dflash2_draft_model = dflash2_draft_model
        self.dflash2_num_speculative_tokens = dflash2_num_speculative_tokens
        self._external_qwen_spec_enabled = enable_dspark or enable_dflash2
        self.checkpoint_budget_multiple = checkpoint_budget_multiple
        self.qwen_kv_extensible = qwen_kv_extensible
        self.qwen_kv_commit_buffer_gb = qwen_kv_commit_buffer_gb
        if qwen_kv_mode not in {"legacy", "strict", "elastic"}:
            raise ValueError(
                f"qwen_kv_mode={qwen_kv_mode!r} must be 'legacy', 'strict', or 'elastic'"
            )
        if qwen_kv_mode != "legacy" and backend != "qwen36":
            raise ValueError(f"qwen_kv_mode={qwen_kv_mode!r} requires backend='qwen36'")
        if qwen_kv_pool_bytes < 0:
            raise ValueError("qwen_kv_pool_bytes must be non-negative")
        if qwen_kv_watermark_bundles < 0:
            raise ValueError("qwen_kv_watermark_bundles must be non-negative")
        if qwen_kv_mode == "elastic" and qwen_kv_pool_bytes <= 0:
            raise ValueError("qwen_kv_mode='elastic' requires qwen_kv_pool_bytes > 0")
        if qwen_kv_extensible and qwen_kv_mode == "legacy":
            raise ValueError(
                "qwen_kv_extensible=True requires qwen_kv_mode='strict' or 'elastic' "
                "(the VMM physical pool only makes sense over the arena's bundle pool)"
            )
        if qwen_kv_commit_buffer_gb < 0:
            raise ValueError("qwen_kv_commit_buffer_gb must be non-negative")
        if qwen_kv_mode != "legacy" and not qwen_kv_full_sequence_must_fit:
            raise ValueError(
                "dynamic Qwen KV currently requires full-sequence reservation; "
                "chunk-only overcommit has no safe preemption/recompute path"
            )
        mtp_graph_pool_enabled = (
            enable_cudagraph
            and os.environ.get("QSR_QWEN36_MTP_CUDA_GRAPH", "1") != "0"
        )
        if qwen_kv_mode != "legacy" and enable_mtp and not mtp_graph_pool_enabled:
            raise ValueError(
                "dynamic Qwen KV with MTP requires CUDA Graph pooled MTP caches; "
                "the eager MTP path still allocates fixed per-slot cache rows"
            )
        self.qwen_kv_mode = qwen_kv_mode
        self.qwen_kv_pool_bytes = qwen_kv_pool_bytes
        self.qwen_kv_watermark_bundles = qwen_kv_watermark_bundles
        self.qwen_kv_full_sequence_must_fit = qwen_kv_full_sequence_must_fit
        self.MODEL = model
        self.vision_enabled = False
        self.vision_checkpoint: str | None = None
        self.image_token_id: int | None = None
        if backend == "flashnext":
            vision_env = os.environ.get("QSR_FLASHNEXT_VISION", "1").strip().lower()
            if vision_env not in {"0", "1", "false", "true", "off", "on"}:
                raise ValueError(
                    "QSR_FLASHNEXT_VISION must be 0 or 1, "
                    f"got {vision_env!r}"
                )
            self.vision_enabled = vision_env in {"1", "true", "on"}
        self.K = 0

        if enable_dflash:
            # DFlashEngine's draft/verify CUDA Graphs are captured against ONE
            # set of scratch buffers (runtime/backends/laguna_dflash_cudagraph.py
            # / laguna_cuda_graph.py LagunaCudaGraphVerify), not a batch-shaped
            # buffer like decode CG's LagunaCudaGraphDecode. Concurrency
            # (capacity>1) is supported by re-addressing those shared buffers
            # per call (_fill_buffers(slot, ...) recomputes every offset from
            # `slot` before each replay) and processing the round's active
            # slots with one sequential CUDA Graph replay per slot -- correct
            # per-slot isolation (verified: notes/2026-07-27-dflash-multi-slot-
            # concurrency.md), but N sequential single-token-batch replays,
            # not one batched N-wide replay like decode CG. No capacity cap.
            # NUM_SPECULATIVE_TOKENS is imported at module scope. Importing it
            # here as well made it a *local* of __init__ for the whole function
            # body -- Python binds a name locally if it is assigned anywhere in
            # the function -- so the histogram sizing below raised
            # UnboundLocalError whenever this branch was not taken. Caught by
            # the full suite (25 failures), not by review.
            self.K = NUM_SPECULATIVE_TOKENS
        elif enable_mtp:
            # Same headroom reasoning as the DFlash branch above:
            # capacity_ok() below must reserve room for up to K drafted-but-
            # not-yet-committed tokens per slot.
            self.K = mtp_num_speculative_tokens
        elif enable_dspark:
            # The official Qwen3.8 DSpark draft uses gamma=7. The loader checks
            # the draft config again before allocation; this early value keeps
            # admission from accepting a request whose speculative tail cannot
            # fit before the draft checkpoint is loaded.
            self.K = dspark_num_speculative_tokens
        elif enable_dflash2:
            self.K = dflash2_num_speculative_tokens

        # CUDA Graph slot budget:
        # - Laguna decode CG (M=1) captures against ONE dedicated slot (the
        #   last), not capacity slots.  After capture the slot is reset before
        #   real use.
        # - Qwen captures every decode batch against the real rows before the
        #   server accepts work, then reset_all() clears them.  Its MTP anchor
        #   graph uses Qwen36SlotPool.scratch_row.  Reserving another server
        #   slot therefore duplicates a full KV/GDN row (9.3 GiB at 256K,
        #   FP8 backbone KV + BF16 MTP KV, K=3) without protecting any graph.
        # - DFlash draft/verify CGs use shared scratch buffers and replay
        #   sequentially per slot; they do NOT need extra physical slots.
        # - DSV4 decode CG is a shared batched driver (B=1/2/4) whose slot ids
        #   and positions are persistent tensor inputs, so it is NOT bound to
        #   a concrete slot and needs no dedicated warmup slot (capacity plan
        #   Phase 2: cg_extra=0, exactly num_slots=capacity).
        # So: +1 only for Laguna's non-DFlash decode CG; +0 for Qwen, DFlash,
        # and DSV4.
        cg_extra = _cuda_graph_extra_slots(
            backend=backend,
            enable_cudagraph=enable_cudagraph,
            enable_dflash=enable_dflash,
        )
        if production:
            min_slots = capacity + cg_extra
        else:
            min_slots = 3 * capacity + cg_extra
        if num_slots < min_slots:
            raise ValueError(
                f"num_slots={num_slots} must be >= {min_slots} for capacity={capacity}, "
                f"enable_cudagraph={enable_cudagraph}, enable_dflash={enable_dflash}"
            )
        if enable_session_affinity and not enable_prefix_cache:
            raise ValueError("enable_session_affinity requires enable_prefix_cache")
        if enable_session_affinity:
            # N8 (docs/architecture.md §3.5.6): mtp_prefill_warm_continue has
            # no LagunaBackend implementation -- it survives only under
            # oracle/qwen36_vllm/. _step_sync used to call it anyway inside a
            # bare `except Exception`, so every warm-continue attempt raised,
            # was swallowed, and silently fell back to a cold prefill: output
            # stayed correct, session_warm_continuations stayed at zero, and
            # nothing told the operator their flag was a no-op. Reject here,
            # at construction time and before any GPU work, instead. Hardcodes
            # LagunaBackend rather than going through the backend registry
            # because IMPLEMENTED_BACKENDS == {"laguna"} today, so the
            # validation above already guarantees self.backend_name ==
            # "laguna" is the only value reachable here -- revisit once
            # Track B adds a second implemented backend.
            from runtime.backends.laguna import LagunaBackend

            if not LagunaBackend.capabilities.fget(None).warm_continue:
                raise ValueError(
                    "enable_session_affinity requires a backend with warm_continue "
                    f"capability; backend={self.backend_name!r} does not support it "
                    "(see docs/architecture.md §3.5.6, N8)"
                )

        # -- config --
        self.capacity = capacity
        self.production = production
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.capacity_tokens_per_slot = block_size * blocks_per_slot
        self.idle_sleep_s = idle_sleep_s
        self.watchdog_max_stale_rounds = watchdog_max_stale_rounds
        self.request_timeout_s = request_timeout_s
        self._kv_cache_dtype = kv_cache_dtype
        self._enable_cudagraph = enable_cudagraph
        self.enable_prefix_cache = enable_prefix_cache
        self.enable_session_affinity = enable_session_affinity
        self.session_ttl_s = session_ttl_s
        self._gpu_memory_utilization = gpu_memory_utilization

        # -- tokenizer (CPU-only, thread-safe for reads) --
        from transformers import AutoTokenizer

        if backend == "deepseek_v4":
            # DSV4 is served from a GGUF weight file, which carries no
            # tokenizer for the transformers loader; the official HF
            # tokenizer dir (tokenizer.json + tokenizer_config.json) is a
            # standard PreTrainedTokenizerFast -- no trust_remote_code, no
            # custom classes.  Default is the vendored reference dir; a
            # deployment points QSR_DSV4_TOKENIZER_DIR at its own copy.
            tokenizer_dir = os.environ.get(
                "QSR_DSV4_TOKENIZER_DIR",
                str(Path(__file__).resolve().parent.parent / "notes" / "dsv4flash-ref"),
            )
            if not (Path(tokenizer_dir) / "tokenizer.json").is_file():
                raise RuntimeError(
                    f"DSV4 tokenizer dir {tokenizer_dir!r} lacks tokenizer.json; "
                    "set QSR_DSV4_TOKENIZER_DIR"
                )
            self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
            # DSV4 serving contract: EOS=1, no BOS added (plan §7.2).
            self.eos_token_id = 1
            self.eos_token_ids = frozenset({1})
        else:
            # Laguna ships a custom AutoConfig/tokenizer class
            # (configuration_laguna.py) that transformers only loads with
            # trust_remote_code=True; without it, config validation falls
            # onto a generic path that chokes on Laguna's yarn
            # rope_parameters (KeyError: 'original_max_position_embeddings').
            tokenizer_source = self.MODEL
            is_qwen_gguf = backend == "qwen36" and Path(self.MODEL).suffix.casefold() == ".gguf"
            if is_qwen_gguf:
                tokenizer_source = os.environ.get("QSR_SERVER_TOKENIZER_PATH", "")
                if not tokenizer_source:
                    raise RuntimeError(
                        "Qwen GGUF serving requires QSR_SERVER_TOKENIZER_PATH pointing "
                        "to a compatible HuggingFace tokenizer directory"
                    )
                if not (Path(tokenizer_source) / "tokenizer.json").is_file():
                    raise RuntimeError(
                        f"Qwen GGUF tokenizer path {tokenizer_source!r} lacks tokenizer.json; "
                        "set QSR_SERVER_TOKENIZER_PATH to a local tokenizer directory"
                    )
            self.tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
            if backend in {"qwen36", "flashnext"}:
                configured_effort = apply_qwen_default_reasoning_effort(self.tok)
                if configured_effort is not None:
                    logger.info(
                        "Qwen reasoning template default configured to effort=%s "
                        "(backend=%s); "
                        "request-level effort remains explicit-only",
                        configured_effort,
                        backend,
                    )
            self.eos_token_id = self.tok.eos_token_id
            if backend == "flashnext":
                configured_image_token_id = getattr(self.tok, "image_token_id", None)
                if configured_image_token_id is not None:
                    self.image_token_id = int(configured_image_token_id)
            try:
                from transformers import GenerationConfig

                gen_cfg_eos = GenerationConfig.from_pretrained(tokenizer_source).eos_token_id
            except Exception:
                gen_cfg_eos = self.eos_token_id
            if isinstance(gen_cfg_eos, (list, tuple, set)):
                self.eos_token_ids = frozenset(int(e) for e in gen_cfg_eos)
            else:
                self.eos_token_ids = frozenset({int(gen_cfg_eos)})

        # -- high-performance request channel (asyncio → engine thread) --
        # deque is GIL-atomic for append/popleft; pipe provides instant wakeup
        self._req_deque: collections.deque[GenerationRequest] = collections.deque()
        self._req_pipe_r, self._req_pipe_w = os.pipe()
        os.set_blocking(
            self._req_pipe_r, False
        )  # non-blocking by default; set blocking only for idle wait
        os.set_blocking(self._req_pipe_w, False)  # asyncio thread never blocks on write

        # -- engine thread state --
        self._ready_event = threading.Event()
        self._load_error: BaseException | None = None
        self._engine_thread: threading.Thread | None = None
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._stop = False
        self._cancel_set: set[str] = set()

        # -- slot management (only mutated from engine thread after start) --
        self.free_slots: list[int] = list(range(capacity))
        self.active: dict[int, dict[str, Any]] = {}
        self.waiting: list[GenerationRequest] = []
        # A5/B4: incremental chunked prefill state (None = no prefill in progress)
        self._pending_prefill = None  # ChunkedPrefillState | None
        self._pending_prefill_reqs: list[tuple[int, GenerationRequest]] = []
        # SGLang's radix scheduler avoids putting exact duplicate prompts in
        # the same cold extend wave: the first request publishes the prefix,
        # and the remaining requests restore it on the next wave.  Keep only
        # keys whose first request is currently being admitted.  They are
        # released as soon as that prefill commits, so all duplicates can
        # restore together without serializing the cheap cache-hit path.
        self._prefix_dedup_inflight: set[tuple[int, ...]] = set()
        self._prefix_dedup_published: set[tuple[int, ...]] = set()
        # SGLang-style wave admission: when the GPU is idle and the first
        # request of a concurrent wave has arrived, briefly collect wakeups
        # already in flight before starting a long prefill.  Keep the default
        # off outside the measured Qwen DSpark profile; that profile enables
        # it because a 128K c4 wave otherwise
        # entered as [1] + [3] and paid two separate prefill schedules.
        configured_coalesce_ms = os.environ.get("QSR_ADMISSION_COALESCE_MS", "0")
        try:
            self._admission_coalesce_s = float(configured_coalesce_ms) / 1000.0
        except ValueError as exc:
            raise ValueError(
                "QSR_ADMISSION_COALESCE_MS must be a non-negative number"
            ) from exc
        if self._admission_coalesce_s < 0:
            raise ValueError("QSR_ADMISSION_COALESCE_MS must be non-negative")
        # A dynamic-arena capacity miss cannot improve while the same active
        # requests keep their pages. Avoid re-hashing a long waiting prompt on
        # every decode token; completion/cancel clears this immediately, and
        # the bounded retry covers any less common capacity transition.
        self._kv_admission_retry_round = 0
        self.retained: dict[str, dict[str, Any]] = {}
        self.ref_slot_for = {p: capacity + p for p in range(capacity)}
        self.diag_slot_for = {p: 2 * capacity + p for p in range(capacity)}

        self._recent_prompts: collections.deque[tuple[str, list[int]]] = collections.deque(
            maxlen=_PREFIX_OVERLAP_HISTORY
        )

        self.stats: dict[str, Any] = {
            "rounds": 0,
            "admissions": 0,
            "admission_batch_sizes": [],
            "admission_coalesce_waits": 0,
            "admission_coalesce_wait_ms": [],
            "prefix_cache_dedup_deferrals": 0,
            "kv_admission_waits": 0,
            "round_batch_sizes": [],
            "bootstrap_checks_ok": 0,
            "bootstrap_checks_failed": 0,
            "bootstrap_failures": [],
            "requests_completed": 0,
            "prefix_overlap_samples": [],
            "prefix_overlap_same_round_events": 0,
            "prefix_overlap_history_events": 0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "prefix_cache_hit_rate": 0.0,
            "prefix_cache_hit_L_samples": [],
            "prefix_cache_hit_tokens_saved": 0,
            "session_warm_continuations": 0,
            "session_warm_continuation_samples": [],
            "session_retentions": 0,
            "session_expirations": 0,
            "session_warm_fallbacks": 0,
            "cancellations": 0,
            "timeouts": 0,
            "watchdog_triggers": 0,
            "watchdog_events": [],
            # Width follows NUM_SPECULATIVE_TOKENS (+1 for the 0 bucket, +1 for
            # overflow). It was a bare 5 while K was 15, so `elif 0 <= na <
            # len(...)` silently dropped every round accepting 5 or more --
            # i.e. the healthier acceptance was, the less of it got recorded.
            "mtp_acceptance_histogram": [0] * (NUM_SPECULATIVE_TOKENS + 2),
            "sampled_decode_rounds": 0,
            # E2-b (docs/e2e-and-quality-plan.md §2.2): non-greedy MTP rounds'
            # acceptance is tracked SEPARATELY from mtp_acceptance_histogram
            # above (which is greedy-only and has always been) -- the
            # completion criterion is "recorded separately from the greedy
            # path's acceptance rate, not required to be the same" (rejection
            # sampling's acceptance probability depends on how much the draft
            # and target distributions agree, which is a different question
            # from greedy top-1 agreement). Accumulated as raw totals rather
            # than a fixed-size histogram bucketed by num_accepted, both to
            # avoid mtp_acceptance_histogram's own pre-existing "silently
            # drops any num_accepted >= 4" quirk (its 5 buckets look sized for
            # a different, smaller-K backend) and because "accepted/drafted"
            # is the number this criterion actually asks for -- the same
            # ratio ``DFlashEngine.generate_verify_only``'s own
            # ``stats["acceptance_rate"]`` already reports for the greedy
            # standalone-generate path.
            "mtp_sampled_total_accepted": 0,
            "mtp_sampled_total_draft": 0,
            "mtp_sampled_rounds": 0,
            # DSpark uses the same scheduler branch as MTP, but its K and
            # acceptance semantics belong to the external draft model. Keep
            # a DSpark-native surface so benchmark artifacts state exactly
            # which speculative engine produced them.
            "dspark_acceptance_histogram": (
                [0]
                * (
                    dflash2_num_speculative_tokens + 2
                    if enable_dflash2
                    else dspark_num_speculative_tokens + 2
                )
                if enable_dspark or enable_dflash2
                else []
            ),
            "dspark_rounds": 0,
            "dspark_accepted_tokens": 0,
            "dspark_committed_tokens": 0,
        }

        self.runner = None
        self._prefill_chunk_size = 512
        self._near_tie_margin_diag = None
        self.near_tie_logit_margin = 0.0

    # -- A3 step 7-g: cache coordinator -------------------------------------
    @property
    def slot_resources(self) -> SlotResourceManager:
        """The coordinator admission and the decode round read instead of
        calling ``self.runner`` directly for the two ``capabilities.
        prefix_cache`` members (docs/a3-cache-coordinator-design.md §7 row
        7-g).

        Constructed fresh on every access rather than cached on ``self`` at
        load time, on purpose: several tests (e.g. ``tests/
        test_engine_prefix_cache_admission.py``) set ``engine.runner =
        <fake>`` directly, bypassing ``_load_laguna_model`` entirely, and a
        cached attribute built once in ``_load_laguna_model`` would then
        keep pointing at whatever ``self.runner`` was at construction time
        (``None``) instead of the fake -- silently wrong, not an error.
        ``SlotResourceManager.__init__`` is two attribute assignments, so
        rebuilding it per call costs nothing and this property is always
        "constructed after ``self.runner`` exists" for whatever caller reads
        it, real load or fake.
        """
        # Flash-Next owns fixed token-indexed QSA pools and can checkpoint its
        # recurrent state at any token boundary; unlike paged Qwen/DSV4
        # caches, it must not be floored to the server's 128-token attention
        # page when reconciling the second (GDN/PLE) cache family.
        cache_block_size = getattr(self.runner, "prefix_cache_block_size", self.block_size)
        return SlotResourceManager(
            self.runner,
            self.architecture_spec,
            block_size=cache_block_size,
        )

    # -- model loading (engine thread only) --------------------------------
    def _load_model(self) -> None:
        """Load model + create the runner. MUST run on engine thread.

        Dispatches on ``self.backend_name``, which ``server/app.py``'s
        ``lifespan()`` resolved from the checkpoint's own ``config.json``
        (``runtime.model_registry.resolve_checkpoint``) -- the engine does
        not re-derive it. Track B / B2 added the second branch; before it
        there was one backend and the dispatch was the absence of one.
        """
        if self.backend_name == "qwen36":
            self._load_qwen36_model()
        elif self.backend_name == "flashnext":
            self._load_flashnext_model()
        elif self.backend_name == "deepseek_v4":
            self._load_deepseek_model()
        else:
            self._load_laguna_model()

    def _load_flashnext_model(self) -> None:
        """Load the native Qwen3.8 Flash-Next text graph.

        Flash-Next is not a Qwen3.6 checkpoint with a different name: its
        ``Qwen4ExpForConditionalGeneration`` graph has 48 GDN/QSA layers, a
        PLE table, and a separate in-checkpoint MTP block.  Keep its loader
        and state ownership in the dedicated backend so a path typo cannot
        silently dispatch through ``Qwen36Backend``.
        """
        from runtime.backends.flashnext import FlashNextBackend
        from runtime.laguna_config import _resolve_laguna_model_dir
        from runtime.model.flashnext.model import load_flashnext_model

        if self.enable_dflash or self.enable_dspark or self.enable_dflash2:
            raise ValueError(
                "Flash-Next uses its in-checkpoint MTP head; DFlash, DSpark, "
                "and DFlash2 are not valid with backend='flashnext'"
            )
        if self.qwen_kv_mode != "legacy":
            raise ValueError(
                "Flash-Next currently owns QSA/GDN state directly and requires "
                "qwen_kv_mode='legacy'"
            )
        checkpoint = Path(self.MODEL)
        if not checkpoint.is_dir():
            checkpoint = Path(_resolve_laguna_model_dir(self.MODEL))
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"Flash-Next checkpoint directory does not contain config.json: {checkpoint}"
            )
        max_model_len = self.blocks_per_slot * self.block_size
        # Flash-Next's large-M recurrent/QSA path is only numerically gated
        # through 1024-token chunks.  Larger chunks are faster in a warm
        # microbenchmark, but they change the b12x/GDN reduction order and
        # fail the same-prompt state/logit gate (2048: cosine ~= 0.876/0.779;
        # 8192 is worse).  Keep the quality-safe value as the production
        # default; an explicit opt-in is required for experiments so a
        # generic QSR_PREFILL_CHUNK=8192 from another backend cannot silently
        # corrupt Flash-Next generations.
        configured_prefill_chunk = os.environ.get("QSR_PREFILL_CHUNK", "1024")
        try:
            self._prefill_chunk_size = int(configured_prefill_chunk)
        except ValueError as exc:
            raise ValueError(
                "QSR_PREFILL_CHUNK must be a positive integer, "
                f"got {configured_prefill_chunk!r}"
            ) from exc
        if self._prefill_chunk_size <= 0:
            raise ValueError(
                "QSR_PREFILL_CHUNK must be a positive integer, "
                f"got {self._prefill_chunk_size}"
            )
        allow_unsafe_chunk = os.environ.get(
            "QSR_FLASHNEXT_ALLOW_UNSAFE_PREFILL_CHUNK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self._prefill_chunk_size > 1024 and not allow_unsafe_chunk:
            logger.warning(
                "QSR_PREFILL_CHUNK=%d is not numerically gated for Flash-Next; "
                "using 1024 (set QSR_FLASHNEXT_ALLOW_UNSAFE_PREFILL_CHUNK=1 "
                "only for explicit experiments)",
                self._prefill_chunk_size,
            )
            self._prefill_chunk_size = 1024
        ple_resident = os.environ.get("QSR_FLASHNEXT_PLE_RESIDENT", "0") == "1"
        # A row contains one 160-byte FP8 vector for one hash head.  The
        # production context is 256K and the checkpoint emits 16 heads per
        # token, so 4M rows cover one complete context's lookup working set
        # while remaining a small host-RAM reservation (~1 GiB worst case with
        # Python object overhead).  Keep smaller-context deployments bounded
        # instead of paying the 4M-entry list cost unconditionally.
        default_ple_cache_rows = min(4_194_304, max(131_072, max_model_len * 16))
        ple_cache_rows = int(
            os.environ.get("QSR_FLASHNEXT_PLE_CACHE_ROWS", str(default_ple_cache_rows))
        )
        ple_cache_pages = int(os.environ.get("QSR_FLASHNEXT_PLE_CACHE_PAGES", "0"))
        ple_io_workers = int(os.environ.get("QSR_FLASHNEXT_PLE_IO_WORKERS", "32"))
        logger.info(
            "loading Qwen3.8 Flash-Next target from %s (max_context=%d, MTP=%s, "
            "PLE=%s, ple_cache_rows=%d, ple_cache_pages=%d, vision=%s, "
            "image_max_pixels=%s, io=%s, prefill_chunk=%d)",
            checkpoint,
            max_model_len,
            self.enable_mtp,
            "resident" if ple_resident else "stream",
            ple_cache_rows,
            ple_cache_pages,
            self.vision_enabled,
            os.environ.get("QSR_FLASHNEXT_IMAGE_MAX_PIXELS", "1048576"),
            os.environ.get("QSR_FLASHNEXT_PLE_IO", "auto"),
            self._prefill_chunk_size,
        )

        def progress(done: int, total: int) -> None:
            if done == total or done == 1 or done % 8 == 0:
                logger.info("Flash-Next weight load: %d/%d layers", done, total)

        model = load_flashnext_model(
            checkpoint,
            device="cuda",
            enable_vision=self.vision_enabled,
            ple_resident=ple_resident,
            ple_cache_rows=ple_cache_rows,
            ple_cache_pages=ple_cache_pages,
            ple_io_workers=ple_io_workers,
            progress=progress,
        )
        self.vision_checkpoint = str(checkpoint)
        if self.image_token_id is None:
            configured_image_token_id = getattr(model.cfg, "image_token_id", None)
            if configured_image_token_id is not None:
                self.image_token_id = int(configured_image_token_id)
        self.runner = FlashNextBackend(
            model,
            num_slots=self.num_slots,
            max_seq_len=max_model_len,
            device="cuda",
            checkpoint_path=str(checkpoint),
            enable_mtp=self.enable_mtp,
            mtp_num_speculative_tokens=self.mtp_num_speculative_tokens,
            enable_prefix_cache=self.enable_prefix_cache,
        )
        if self._enable_cudagraph:
            graph_batch_size = self.runner.capture_decode_cuda_graph()
            if graph_batch_size is not None:
                logger.info(
                    "Flash-Next target%s CUDA Graph captured (batch_size=%d)",
                    " + MTP verify" if self.enable_mtp else "",
                    graph_batch_size,
                )
            else:
                logger.warning(
                    "Flash-Next CUDA Graph capture failed; falling back to eager target decode"
                )
        # Emit the resolved memory/precision profile once the graph has been
        # captured.  This is intentionally based on live tensor ownership,
        # not only nvidia-smi: the latter also includes driver and allocator
        # reservations and made the previous BF16-vs-FP8 capacity decision
        # unnecessarily opaque.
        memory = self.runner.memory_breakdown()
        mtp_model = getattr(self.runner, "_mtp_model", None)
        mtp_mlp = getattr(mtp_model, "mlp", None)
        mtp_dtype = getattr(mtp_mlp, "expert_dtype", None)
        logger.info(
            "Flash-Next memory profile: mtp_expert_dtype=%s, target_model=%.2f GiB, "
            "mtp_model=%.2f GiB, sessions=%.2f GiB, explicit=%.2f GiB, "
            "torch_reserved=%.2f GiB, driver_free=%.2f GiB",
            mtp_dtype,
            memory.get("model_tensor_bytes", 0) / 2**30,
            (
                memory.get("mtp_model_parameters", 0)
                + memory.get("mtp_model_buffers", 0)
                + memory.get("mtp_auxiliary_tensors", 0)
            )
            / 2**30,
            memory.get("session_tensor_bytes", 0) / 2**30,
            memory.get("explicit_tensor_bytes", 0) / 2**30,
            memory.get("torch_reserved", 0) / 2**30,
            memory.get("driver_free_bytes", 0) / 2**30,
        )
        logger.info(
            "Qwen3.8 Flash-Next model loaded: backend=flashnext, num_slots=%d, "
            "max_context=%d, mtp=%s(K=%d)",
            self.num_slots,
            max_model_len,
            self.enable_mtp,
            self.mtp_num_speculative_tokens,
        )

    def _warmup_qwen36_full_forward(self) -> None:
        """One real short prefill through the whole model, then reset the slot.

        `Qwen36ForCausalLMSelfBuilt.warmup_attention_shapes` warms **only the
        attention kernels** -- its own docstring says so, and explains the
        reason: GDN's recurrent state is order-dependent (B0-5), so a real
        forward would risk leaving a warmed-up state behind.

        The consequence is that everything else stays cold until a user's
        first request pays for it: the w4a16 fused MoE across all 56 NVFP4 MLP
        layers (including `_ensure_w4a16_fused_ready`'s one-time fused-weight
        preparation), the GDN recurrent kernels, `lm_head`. Measured, that is
        the whole of the first-request TTFT anomaly -- 4.67s on the first
        request against 0.25s on every one after it.

        The historical vLLM-era runtime did exactly this and no more:
        `oracle/qwen36_vllm/direct_model_runner.py:860`'s `_warmup` runs
        `self.prefill(0, [0, 0, 0, 0, 0])` inside a `try/finally` whose
        `finally` is `self.reset_slot(0)`. That is also the answer to the GDN
        objection above: `reset_slot` zeroes the recurrent state, which is the
        one operational requirement B0-5 attached to its capture-safe verdict
        -- so a real forward is safe as long as the slot is reset afterwards.

        Deliberately before CUDA Graph capture, matching the order the
        Laguna+DFlash path already uses: capture wants the kernels it will
        record to have been compiled already.

        Failure here is logged and swallowed. A warmup that cannot run is a
        latency problem on the first request, not a correctness one, and
        refusing to start the server over it would trade a 4-second stall for
        a total outage.
        """
        try:
            state = self.runner.prefill_chunked_begin([0], [[0, 0, 0, 0, 0]])
            while not state.done:
                self.runner.prefill_chunked_step(state)
        except Exception:
            logger.exception("Qwen3.6 full-forward warmup failed; first request will be slow")
        finally:
            try:
                self.runner.reset_slot(0)
            except Exception:
                logger.exception("reset_slot(0) failed after warmup")

    def _load_qwen36_model(self) -> None:
        """Load ``Qwen36Backend`` (Track B / B2, B3). MUST run on engine thread.

        The qwen36 backend itself is reachable (``runtime.model_registry
        .IMPLEMENTED_BACKENDS`` includes ``"qwen36"`` as of ``0e83b52``,
        "B2: Qwen3.6 is servable"); this docstring used to say otherwise --
        that was already stale before this B3 change touched the file.

        No DFlash for this backend (DFlash is Laguna's own draft model).
        Qwen3.6's native speculative story is MTP, while the measured Qwen3.x
        default uses the separate external DSpark draft wired below via
        ``self.enable_dspark``. MTP remains an explicit rollback path.
        DSpark captures fixed-width target-verify and greedy-draft CUDA Graphs;
        its separate draft KV remains a legacy physical allocation even when
        the target uses the dynamic arena. The draft-KV family participates in
        the persistent prefix snapshot/restore path, which is why the measured
        Qwen default can keep prefix caching enabled.

        ``enable_cudagraph`` captures here, before ``start()``'s admission
        loop can hand out a slot, for the same reason Laguna does -- plus
        one Laguna does not have: capture runs real forwards, which write
        real recurrent state, and a recurrent state left behind is read by
        the next sequence rather than ignored (B0-5).
        ``capture_decode_cuda_graph`` zeroes every slot on the way out for
        exactly that reason. Qwen3.6 MTP is wired BEFORE capture: its
        historical K+1 GDN rows extend the same pooled storage and column
        zero must alias the ordinary prefill/decode row. Capturing first
        would bake the old state addresses into the decode graph. Both MTP's
        own graphs and the ordinary decode graphs therefore still capture
        while every slot is definitionally empty.
        """
        import torch  # local: this module stays importable without torch

        from runtime.backends.qwen36 import Qwen36Backend
        from runtime.laguna_config import _resolve_laguna_model_dir
        from runtime.model_loading import (
            load_qwen36_dspark_draft_model,
            load_qwen36_model,
            load_qwen38_dflash2_draft_model,
            load_qwen38_gguf_model,
        )

        if self.enable_dflash:
            # capabilities.speculative_decode is True for this backend (B3),
            # but DFlash specifically is Laguna's own draft-model mechanism,
            # not qwen36's (that is enable_mtp). Refusing here rather than
            # silently downgrading: a deployment that asked for speculative
            # decoding and got ordinary decoding without being told is how
            # an acceptance-rate counter sits at zero unnoticed (N8).
            raise ValueError(
                "enable_dflash is not supported by the qwen36 backend "
                "(DFlash is Laguna's draft model; qwen36's is MTP -- "
                "enable_mtp/QSR_SERVER_ENABLE_MTP). "
                "Start the server with QSR_SERVER_ENABLE_DFLASH=0."
            )

        max_model_len = self.blocks_per_slot * self.block_size
        requested_model_path = Path(self.MODEL)
        if requested_model_path.suffix.casefold() == ".gguf":
            if not requested_model_path.is_file():
                raise FileNotFoundError(
                    f"Qwen GGUF checkpoint does not exist: {requested_model_path}"
                )
            if self.enable_dspark:
                raise ValueError(
                    "legacy DSpark cannot be paired with a GGUF Qwen target; "
                    "use enable_dflash2 with the Qwen3.8 DFlash2 checkpoint"
                )
            target_model_path = requested_model_path
            model = load_qwen38_gguf_model(
                target_model_path,
                device="cuda",
                dtype=torch.float32,
                max_seq_len=max_model_len,
                warmup_attention=False,
            )
            logger.info("Qwen GGUF target loaded from %s", target_model_path)
        else:
            target_model_path = _resolve_laguna_model_dir(self.MODEL)
            model = load_qwen36_model(
                target_model_path,
                device="cuda",
                dtype=torch.bfloat16,
                max_seq_len=max_model_len,
                enable_mtp=self.enable_mtp,
            )
        dynamic_arena = self.qwen_kv_mode != "legacy"
        pool_bundles = None
        bundle_bytes = _qwen_kv_bundle_bytes(model, include_mtp=self.enable_mtp)
        pages_per_slot = (max_model_len + 127) // 128
        if self.qwen_kv_mode == "strict":
            # Null bundle + every addressable server slot at its declared
            # ceiling + an explicit emergency/COW watermark. The scratch row
            # is logical-only in dynamic mode and borrows pages during load-
            # time capture rather than pinning another 256K row forever.
            pool_bundles = (
                1 + self.num_slots * pages_per_slot + self.qwen_kv_watermark_bundles
            )
        elif self.qwen_kv_mode == "elastic":
            pool_bundles = self.qwen_kv_pool_bytes // bundle_bytes
            min_bundles = 1 + pages_per_slot + self.qwen_kv_watermark_bundles
            if pool_bundles < min_bundles:
                raise ValueError(
                    "Qwen elastic KV pool cannot fit one maximum-length request plus "
                    f"null/watermark: bytes={self.qwen_kv_pool_bytes}, "
                    f"bundle_bytes={bundle_bytes}, bundles={pool_bundles}, "
                    f"required_at_least={min_bundles}"
                )
        # A5/B4: this is live again. Qwen36Backend.prefill_chunked_step now
        # advances one chunk per round and returns done=False until the prompt
        # is consumed, so the incremental branch below is reachable and a long
        # admission no longer blocks active slots' decode.
        #
        # 2048, not 512, and the difference is not cosmetic. Measured on a 60k
        # prompt against an already-decoding request (capacity=2, CG on):
        #
        #   one-shot     max stall 24,939 ms   prefill 25.7s   decode ITL   35 ms
        #   chunk 512    max stall     688 ms  prefill 56.1s   decode ITL  367 ms
        #
        # Chunking at 512 does remove the stall -- 36x -- but turns 8 forwards
        # into 118, so the prefill itself takes 2.2x longer and the concurrent
        # request's steady-state ITL degrades 10x. Totalled over the short
        # request's 220 tokens that is a net loss, despite the stall being
        # gone: ~33s one-shot versus ~80s at chunk 512.
        #
        # So the chunk size is the actual knob here, trading stall length
        # against per-forward overhead. Picking it from the measurements
        # rather than by feel: prefill runs at ~2335 tok/s (60000/25.7s), so
        # bounding a chunk at roughly one second of prefill gives ~2048
        # tokens, i.e. ~30 rounds for a 60k prompt instead of 118.
        #
        # Worth stating because it revises something this repo already
        # concluded: notes/2026-07-20-comprehensive-optimization-plan.md
        # records intra-admission chunking (Phase A) as worth only -10.7%,
        # which is true and is about chunking WITHOUT interleaving. It does
        # not mean chunk size is irrelevant once interleaving exists -- here
        # it is the parameter that decides whether interleaving pays at all.
        configured_prefill_chunk = os.environ.get("QSR_PREFILL_CHUNK")
        if configured_prefill_chunk is None:
            self._prefill_chunk_size = 2048
        else:
            try:
                self._prefill_chunk_size = int(configured_prefill_chunk)
            except ValueError as exc:
                raise ValueError(
                    "QSR_PREFILL_CHUNK must be a positive integer, "
                    f"got {configured_prefill_chunk!r}"
                ) from exc
            if self._prefill_chunk_size <= 0:
                raise ValueError(
                    "QSR_PREFILL_CHUNK must be a positive integer, "
                    f"got {self._prefill_chunk_size}"
                )
        model_config = getattr(model, "config", {})
        target_dtype = (
            torch.bfloat16
            if model_config.get("weight_format") == "gguf"
            and model_config.get("gguf_compute_dtype") == "bfloat16"
            else torch.float32
            if model_config.get("weight_format") == "gguf"
            else torch.bfloat16
        )
        self.runner = Qwen36Backend(
            model,
            num_slots=self.num_slots,
            max_seq_len=max_model_len,
            block_size=self.block_size,
            device="cuda",
            dtype=target_dtype,
            enable_prefix_cache=self.enable_prefix_cache,
            enable_persistent_prefix_cache=None,
            checkpoint_budget_multiple=self.checkpoint_budget_multiple,
            dynamic_arena=dynamic_arena,
            pool_bundles=pool_bundles,
            watermark_bundles=self.qwen_kv_watermark_bundles,
            extensible_kv=self.qwen_kv_extensible,
        )
        if self.qwen_kv_extensible:
            # Phase 5.5: the pool reserves full VA but commits physical
            # pages incrementally. Everything that writes KV before the
            # final commit must have its pages backed first: warmup (one
            # 128-token page), MTP verify (K+1 pages per slot) and decode
            # graph capture (one page per slot). One page per slot beyond
            # that is already guaranteed by enable_mtp's ensure_kv_blocks(1)
            # and the capture path; ensure the worst case up front.
            speculative_k = (
                self.mtp_num_speculative_tokens
                if self.enable_mtp
                else self.K
                if self._external_qwen_spec_enabled
                else 0
            )
            self.runner.ensure_kv_blocks(1 + self.num_slots * (speculative_k + 2))
        self._warmup_qwen36_full_forward()
        # MTP extends the shared GDN state allocation so its column zero is
        # the exact ordinary decode/prefill row.  This must precede decode
        # CUDA Graph capture: a graph captured against the pre-MTP pool would
        # retain stale recurrent-state addresses after the extension.
        if self.enable_mtp:
            self.runner.enable_mtp(
                num_speculative_tokens=self.mtp_num_speculative_tokens,
                enable_resync=self.mtp_resync,
            )
            logger.info(
                "Qwen3.6 MTP speculative decode wired: K=%d, resync=%s, cg_status=%s",
                self.mtp_num_speculative_tokens,
                self.runner._mtp.enable_resync,  # noqa: SLF001 -- log-only introspection
                self.runner._mtp.cg_status,  # noqa: SLF001 -- log-only introspection
            )
        if self.enable_dspark:
            draft_model_path = _resolve_laguna_model_dir(self.dspark_draft_model)
            draft_model = load_qwen36_dspark_draft_model(
                model,
                target_model_path=str(target_model_path),
                draft_model_path=str(draft_model_path),
                device="cuda",
                dtype=torch.bfloat16,
            )
            if draft_model.gamma != self.dspark_num_speculative_tokens:
                raise ValueError(
                    "DSpark draft gamma does not match the server reservation: "
                    f"draft={draft_model.gamma}, configured={self.dspark_num_speculative_tokens}"
                )
            self.runner.enable_dspark(draft_model)
            logger.info(
                "Qwen DSpark speculative decode wired: draft=%s K=%d cg_status=%s",
                self.dspark_draft_model,
                draft_model.gamma,
                self.runner._dspark.cg_status,  # noqa: SLF001 -- load-time observability
            )
        if self.enable_dflash2:
            draft_model = load_qwen38_dflash2_draft_model(
                model,
                draft_model_path=self.dflash2_draft_model,
                device="cuda",
                dtype=torch.bfloat16,
            )
            if draft_model.gamma != self.dflash2_num_speculative_tokens:
                raise ValueError(
                    "DFlash2 draft block size does not match the server reservation: "
                    f"draft={draft_model.gamma}, configured={self.dflash2_num_speculative_tokens}"
                )
            self.runner.enable_dflash2(draft_model)
            logger.info(
                "Qwen DFlash2 speculative decode wired: draft=%s K=%d cg_status=%s",
                self.dflash2_draft_model,
                draft_model.gamma,
                self.runner._dspark.cg_status,  # noqa: SLF001 -- load-time observability
            )
        if self._enable_cudagraph:
            graph_batch_size = self.runner.capture_decode_cuda_graph()
            if self.enable_dflash2 and graph_batch_size != self.num_slots:
                raise RuntimeError(
                    "DFlash2 requires a captured target decode CUDA Graph for every slot; "
                    f"captured={graph_batch_size}, required={self.num_slots}"
                )
            if graph_batch_size is not None:
                logger.info(
                    "Qwen3.6 decode CUDA Graph captured at load (max batch_size=%d)",
                    graph_batch_size,
                )
            else:
                logger.warning(
                    "Qwen3.6 decode CUDA Graph capture failed or unavailable; "
                    "falling back to eager batched decode"
                )
        logger.info(
            "Qwen3.6 model loaded on engine thread: num_slots=%d max_context=%d tokens/slot, "
            "recurrent state %.1f MiB/slot, KV %.1f MiB/slot",
            self.num_slots,
            self.runner.max_seq_len,
            self.runner.pool.geometry.recurrent_bytes_per_slot / 2**20,
            self.runner.pool.geometry.kv_bytes_per_slot / 2**20,
        )
        if self.qwen_kv_extensible:
            self._commit_extensible_kv_pool(model)

    def _commit_extensible_kv_pool(self, model) -> None:
        """Commit the final extensible KV pool size from measured memory.

        Runs after warmup + MTP + decode CUDA Graph capture, when every
        non-KV allocation is settled. The pool's VA capacity was fixed at
        construction (``pool_bundles``); this commits the physical prefix
        to min(capacity, measured) bundles, leaving ``commit_buffer_gb``
        of headroom. If measured memory cannot cover the configured
        capacity, the arena's admission gate keeps refusing new requests
        rather than OOMing -- logged loudly here, not silently.
        """
        import torch  # local: this module stays importable without torch

        capacity = self.runner.pool.pool_bundles
        bundle_bytes = _qwen_kv_bundle_bytes(model, include_mtp=self.enable_mtp)
        torch.cuda.synchronize()
        free_memory, _ = torch.cuda.mem_get_info()
        measured = int(
            (free_memory - int(self.qwen_kv_commit_buffer_gb * 2**30))
            // bundle_bytes
        )
        final = min(capacity, measured)
        committed = self.runner.commit_kv_cache(final)
        logger.info(
            "Extensible KV pool committed: capacity=%d bundles (VA), "
            "committed=%d (%d MiB physical), measured-free %.1f GiB, "
            "bundle_bytes=%d",
            capacity,
            committed,
            self.runner.pool.physical_kv_bytes / 2**20,
            free_memory / 2**30,
            bundle_bytes,
        )
        if final < capacity:
            logger.warning(
                "Extensible KV pool committed %d/%d bundles: measured memory "
                "(%.1f GiB free, %d GiB buffer) cannot cover the configured "
                "capacity; long-context requests may be rejected by admission",
                final,
                capacity,
                free_memory / 2**30,
                self.qwen_kv_commit_buffer_gb,
            )

    def _load_deepseek_model(self) -> None:
        """Load ``DeepseekV4Backend`` (Phase 4). MUST run on engine thread.

        The checkpoint is a single GGUF weight file (``self.MODEL`` is the
        file path, resolved by ``resolve_gguf_checkpoint`` in
        ``server/app.py``'s lifespan); the eager ``Dsv4Transformer`` built
        by ``load_dsv4_from_gguf`` owns the weights, and the serving
        backend stacks one kernel-path attention layer set per slot on
        top of them.

        Memory discipline: the DSV4-Flash IQ2_XS checkpoint is 81.9 GiB
        packed; the per-slot kernel layers add page buffers + MLA scratch
        (order 0.5 GiB/slot at 128K ctx) on top, and the eager oracle's
        own caches live for the process.  The plan's 2-slot x 128K budget
        lands at ~87 GiB; ``capacity``/``num_slots``/``blocks_per_slot``
        are sized from that in the server env (QSR_SERVER_*).
        """
        from runtime.backends.dsv4 import DeepseekV4Backend
        from runtime.model.dsv4_model import load_dsv4_from_gguf

        if self.enable_dflash:
            raise ValueError(
                "enable_dflash is not supported by the deepseek_v4 backend "
                "(DFlash is Laguna's draft model; DSV4's speculative story "
                "is DSpark, which is not served). "
                "Start the server with QSR_SERVER_ENABLE_DFLASH=0."
            )
        if self.enable_mtp:
            raise ValueError(
                "enable_mtp is not supported by the deepseek_v4 backend "
                "(MTP is Qwen3.6's mechanism; DSV4's is DSpark, not served). "
                "Start the server with QSR_SERVER_ENABLE_MTP=0."
            )
        if self._external_qwen_spec_enabled:
            raise ValueError(
                "external Qwen speculative decoding is not supported by the deepseek_v4 backend; "
                "DSpark/DFlash2 are not wired for DSV4 in this runtime"
            )

        max_model_len = self.blocks_per_slot * self.block_size
        model, count = load_dsv4_from_gguf(self.MODEL, max_seq_len=max_model_len, device="cuda")
        logger.info(
            "DeepSeek-V4-Flash GGUF loaded: %d tensors, max_context=%d tokens/slot",
            count,
            max_model_len,
        )
        self.runner = DeepseekV4Backend(
            model,
            model.config,
            num_slots=self.num_slots,
            max_seq_len=max_model_len,
            # The MLA scratch is planned for a bounded prefill chunk and is
            # shared across all 43 layers.  At 64 rows it is 0.376 GiB and
            # does not scale with slot count; 64 rows measured ~35% faster
            # than 32 while 96 rows regressed.  Long prompts are chunked at
            # min(max_q_rows, 128), so keep the measured optimum as default.
            max_q_rows=int(os.environ.get("QSR_DSV4_PREFILL_ROWS", "64")),
            device="cuda",
        )
        # Serving-inactive memory release, BEFORE decode-graph capture: the
        # kernel-path RoPE tables must be shared with the eager graph's BEFORE
        # capture so every captured kernel reads the shared buffer address
        # (sharing after capture would leave the graph pointing at storage
        # that gets GC'd -- measured 2026-08-13: graph replay returns garbage
        # tokens like 124208 ' buruj' instead of the eager EOS).  The eager
        # oracle KV arenas are freed AFTER capture (capture never touches
        # them; freeing them only gives prefill scratch headroom).
        try:
            freed_freqs = self.runner._share_rope_freqs()  # noqa: SLF001
            freed_freqs_bytes = freed_freqs.get("kernel_freqs", 0)
            if freed_freqs_bytes:
                logger.info(
                    "DeepSeek-V4 shared RoPE tables: %.2f GiB reclaimed",
                    freed_freqs_bytes / 2**30,
                )
        except Exception:  # pragma: no cover - release must not block startup
            logger.exception("DeepSeek-V4 RoPE table sharing failed")
        if self._enable_cudagraph:
            graph_batches = self.runner.capture_decode_cuda_graph()
            if graph_batches is not None:
                logger.info(
                    "DeepSeek-V4 decode CUDA Graph captured at load for %d batch buckets",
                    graph_batches,
                )
            else:
                logger.warning(
                    "DeepSeek-V4 decode CUDA Graph capture failed or unavailable; "
                    "falling back to eager decode"
                )
        try:
            freed = self.runner._free_eager_oracle_caches()  # noqa: SLF001
            freed_kv_bytes = freed.get("eager_oracle_kv", 0)
            if freed_kv_bytes:
                logger.info(
                    "DeepSeek-V4 freed eager-oracle KV: %.2f GiB reclaimed",
                    freed_kv_bytes / 2**30,
                )
        except Exception:  # pragma: no cover - release must not block startup
            logger.exception("DeepSeek-V4 eager-oracle KV release failed")
        if self._enable_cudagraph:
            try:
                prefill_graph_ok = self.runner.capture_prefill_cuda_graph()
                if prefill_graph_ok:
                    logger.info("DeepSeek-V4 prefill CUDA Graph captured (43-layer K32 MoE)")
            except Exception:  # pragma: no cover - must not block startup
                logger.exception("DeepSeek-V4 prefill CUDA Graph capture failed")
        logger.info(
            "DeepSeek-V4-Flash model loaded on engine thread: num_slots=%d "
            "max_context=%d tokens/slot",
            self.num_slots,
            max_model_len,
        )

    def _load_laguna_model(self) -> None:
        """Load LagunaBackend. MUST run on engine thread.

        No speculative draft model unless DFlash is enabled, and no
        persistent prefix cache / session affinity unless the caller opts
        in. ``enable_prefix_cache`` / ``enable_session_affinity`` are
        honored as passed, not overridden here -- ServerEngine stays a
        thin, honest pass-through of whatever config the caller chose.

        ``enable_cudagraph`` now does something for Laguna: when set, the
        M=1 decode CUDA Graph is captured right here, before ``start()``'s
        admission loop can hand out any slot -- capture scribbles dummy
        warmup data into the last slot's physical KV-cache range, which is
        only safe while every slot is still definitionally empty. Capturing
        later (e.g. lazily on first decode call, as the standalone benchmark
        helper does) risks corrupting a live request's cache if that slot
        happens to be in use. See ``decode_batch_sampled`` /
        ``_decode_cg_batch_eligible`` in runtime/backends/laguna.py for the
        replay side.
        """
        from runtime.backends.laguna import LagunaBackend
        from runtime.laguna_config import build_laguna_config

        max_model_len = self.blocks_per_slot * self.block_size
        runtime_config = build_laguna_config(
            model=self.MODEL,
            max_model_len=max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            dtype="bfloat16",
        )
        self._prefill_chunk_size = 512  # unused: Laguna prefill is one-shot (see LagunaBackend)
        self.runner = LagunaBackend(
            runtime_config,
            num_slots=self.num_slots,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
        )
        if os.environ.get("QSR_SERVER_WARMUP_PAGED_ATTENTION", "1") != "0":
            # Before any slot can be claimed by a real request (same "every
            # slot is still definitionally empty" window the CUDA Graph/
            # DFlash captures below rely on): pay the paged-attention JIT's
            # per-block-table-width compile cost up front instead of on
            # whichever real request happens to hit each width first. See
            # LagunaBackend.warmup_paged_attention_shapes's docstring.
            self.runner.warmup_paged_attention_shapes()
        if self._enable_cudagraph:
            graph_batch_size = self.runner.capture_decode_cuda_graph()
            if graph_batch_size is not None:
                logger.info(
                    "Laguna decode CUDA Graph captured at load (batch_size=%d)",
                    graph_batch_size,
                )
            else:
                logger.warning(
                    "Laguna decode CUDA Graph capture failed or disabled "
                    "(QSR_DECODE_CUDA_GRAPH); falling back to eager decode"
                )
        if self.enable_dflash:
            # Wire DFlash speculative decoding into the shared MTP-shaped
            # decode path: runner.has_speculative_decode flips True so
            # classify_decode_slots (server/engine.py) routes requests to
            # mtp_verify_and_commit_batch, which LagunaBackend now
            # implements by delegating to DFlashEngine.dflash_round per
            # slot. E2-b (docs/e2e-and-quality-plan.md S2.2) extended this
            # to non-greedy requests too -- dflash_round resolves
            # accept/reject via rejection sampling instead of argmax for
            # them, so classify_decode_slots routes them here as well, not
            # to decode_batch_sampled (only grammar-constrained requests,
            # or every request when DFlash is disabled, still go there).
            # Must run before start()'s admission loop (same "capture
            # before any real slot use" requirement as _ensure_decode_cg
            # above): DFlashEngine.__init__ captures its own draft/verify
            # CUDA Graphs, which scribble dummy warmup data into physical
            # slots.
            dflash_cuda_graph = self.runner.enable_dflash(num_speculative_tokens=self.K)
            logger.info(
                "DFlash speculative decoding wired: K=%d, cuda_graph=%s",
                self.K,
                dflash_cuda_graph,
            )
        logger.info(
            "Laguna model loaded on engine thread: num_slots=%d blocks_per_slot=%d "
            "max_context=%d tokens/slot",
            self.num_slots,
            self.blocks_per_slot,
            self.capacity_tokens_per_slot,
        )

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Spawn the dedicated engine thread; blocks until model is ready."""
        self._asyncio_loop = asyncio.get_running_loop()
        self._engine_thread = threading.Thread(
            target=self._engine_thread_main, daemon=True, name="blackwellm-engine"
        )
        self._engine_thread.start()
        # 2400s (40min), not 600s: on a clean ~/.cache/sparkinfer, model
        # load (~2.5min) plus LagunaBackend.warmup_paged_attention_shapes
        # (one CuTe compile per attention layer group -- full-attention
        # plus each distinct SWA window, commonly tens of seconds apiece)
        # adds a few extra minutes at most. Generous headroom here in case
        # a given machine/toolchain compiles slower than observed. Only the
        # *first* boot on a given machine pays this -- every later restart
        # replays the on-disk compile cache in well under a second total.
        startup_timeout_s = int(os.environ.get("QSR_SERVER_STARTUP_TIMEOUT_S", "2400"))
        if not self._ready_event.wait(timeout=startup_timeout_s):
            raise RuntimeError(
                f"Engine thread failed to initialize model within {startup_timeout_s}s"
            )
        if self._load_error is not None:
            raise RuntimeError("Engine thread failed during model loading") from self._load_error

    async def stop(self) -> None:
        self._stop = True
        # Wake up engine thread if blocked on pipe read
        try:
            os.write(self._req_pipe_w, b"\x00")
        except (BlockingIOError, OSError):
            pass
        if self._engine_thread is not None:
            self._engine_thread.join(timeout=30)
        os.close(self._req_pipe_r)
        os.close(self._req_pipe_w)

    # -- request-facing API (asyncio thread) --------------------------------
    def capacity_ok(self, prompt_len: int, max_tokens: int) -> bool:
        return prompt_len + max_tokens + self.K <= self.capacity_tokens_per_slot

    def advertised_input_capacity(self, max_output_tokens: int) -> int:
        """Return the largest prompt length that still fits this slot budget."""
        return max(1, self.capacity_tokens_per_slot - max(0, int(max_output_tokens)) - self.K)

    def prefix_cache_key_for_request(self, req: GenerationRequest) -> object | None:
        build = getattr(self.runner, "prefix_cache_key_for_vision_inputs", None)
        if build is None:
            return None
        return build(req.vision_inputs)

    async def submit(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        session_id: str | None = None,
        sampling_params: SamplingParams | None = None,
        stop_sequences: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        thinking_budget: ThinkingBudgetConfig | None = None,
        stop_on_tool_call: bool = False,
        vision_inputs: Any | None = None,
    ) -> dict:
        """Submit a generation request. Resolves when generation completes."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        req = GenerationRequest(
            request_id=str(uuid.uuid4()),
            prompt_ids=list(prompt_ids),
            max_tokens=max_tokens,
            future=fut,
            session_id=session_id,
            sampling_params=sampling_params or SamplingParams(),
            stop_sequences=stop_sequences,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            thinking_budget=thinking_budget,
            stop_on_tool_call=stop_on_tool_call,
            vision_inputs=vision_inputs,
        )
        req._admitted_at = time.perf_counter()
        self._req_deque.append(req)
        try:
            os.write(self._req_pipe_w, b"\x00")
        except (BlockingIOError, OSError):
            pass
        return await fut

    async def submit_stream(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        session_id: str | None = None,
        sampling_params: SamplingParams | None = None,
        cancel_ref: list | None = None,
        stop_sequences: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        thinking_budget: ThinkingBudgetConfig | None = None,
        stop_on_tool_call: bool = False,
        vision_inputs: Any | None = None,
    ):
        """Submit a streaming generation request. Yields token-id lists as
        each MTP round commits them. Final yield is the result dict."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        channel = StreamChannel()
        request_id = str(uuid.uuid4())
        channel.request_id = request_id
        if cancel_ref is not None:
            cancel_ref[0] = request_id
        req = GenerationRequest(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            max_tokens=max_tokens,
            future=fut,
            session_id=session_id,
            stream_channel=channel,
            sampling_params=sampling_params or SamplingParams(),
            stop_sequences=stop_sequences,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            thinking_budget=thinking_budget,
            stop_on_tool_call=stop_on_tool_call,
            vision_inputs=vision_inputs,
        )
        req._admitted_at = time.perf_counter()
        self._req_deque.append(req)
        try:
            os.write(self._req_pipe_w, b"\x00")
        except (BlockingIOError, OSError):
            pass
        while True:
            item = await channel.get()
            if item is None:
                break
            if item:
                yield item
        # The stream carries token batches through the channel, while the
        # completion metadata is resolved on the request future.  Returning
        # that result here keeps the streaming contract identical to
        # ``submit`` and gives protocol adapters the authoritative finish
        # reason, token usage, and merged logprobs.
        yield await fut

    def cancel(self, request_id: str) -> None:
        """Request cancellation from any thread (asyncio-safe).

        The engine thread will reclaim the slot on its next round.
        """
        self._cancel_set.add(request_id)
        try:
            os.write(self._req_pipe_w, b"\x00")
        except (BlockingIOError, OSError):
            pass

    # -- thread-safe asyncio callbacks (engine thread → asyncio) -----------
    def _resolve_future(self, fut: asyncio.Future, result: Any) -> None:
        if not fut.done():
            self._asyncio_loop.call_soon_threadsafe(fut.set_result, result)

    def _fail_future(self, fut: asyncio.Future, exc: BaseException) -> None:
        if not fut.done():
            self._asyncio_loop.call_soon_threadsafe(fut.set_exception, exc)

    def _stream_put(self, channel: StreamChannel, item: Any) -> None:
        channel.put(item, self._asyncio_loop)

    def _stream_close(self, channel: StreamChannel) -> None:
        channel.close(self._asyncio_loop)

    # -- admission-time correctness check (engine thread) --------------------
    def _admission_bootstrap_check(self, slot: int, req: GenerationRequest, anchor: int) -> None:
        ref_slot = self.ref_slot_for[slot]
        if not self.runner.slot_state(ref_slot).is_fresh:
            self.runner.reset_slot(ref_slot)
        ref_first = self.runner.prefill(ref_slot, req.prompt_ids)
        if ref_first == anchor:
            self.stats["bootstrap_checks_ok"] += 1
            return
        diag_slot = self.diag_slot_for[slot]
        if not self.runner.slot_state(diag_slot).is_fresh:
            self.runner.reset_slot(diag_slot)
        diag = self._near_tie_margin_diag(self.runner, diag_slot, req.prompt_ids, anchor)
        if diag["within_tolerance"]:
            self.stats["bootstrap_checks_ok"] += 1
        else:
            self.stats["bootstrap_checks_failed"] += 1
            self.stats["bootstrap_failures"].append(
                {"request_id": req.request_id, "slot": slot, "diag": diag}
            )
            logger.warning("bootstrap check FAILED for %s: %s", req.request_id, diag)

    # -- observability (engine thread) ---------------------------------------
    def _log_prefix_overlap(self, admit_now: list[tuple[int, GenerationRequest]]) -> None:
        # The overlap stats are advisory only (they decide whether a batch
        # shares >= block_size tokens, nothing more), so every scan is
        # capped at block_size. Before the cap this was O(B*(B+H)*L) Python
        # work per admission with L=131072 -- a growing per-wave TTFT tax on
        # the serving path, not a benchmark artifact.
        new_prompts = [(req.request_id, req.prompt_ids) for _, req in admit_now]
        for i, (rid, prompt) in enumerate(new_prompts):
            same_round_best = 0
            for j, (_, other_prompt) in enumerate(new_prompts):
                if j == i:
                    continue
                same_round_best = max(
                    same_round_best,
                    _longest_common_prefix_len(prompt, other_prompt, cap=self.block_size),
                )
            history_best = 0
            history_best_rid: str | None = None
            for other_rid, other_prompt in self._recent_prompts:
                overlap = _longest_common_prefix_len(prompt, other_prompt, cap=self.block_size)
                if overlap > history_best:
                    history_best = overlap
                    history_best_rid = other_rid
            self.stats["prefix_overlap_samples"].append(
                {
                    "request_id": rid,
                    "prompt_tokens": len(prompt),
                    "same_round_overlap_tokens": same_round_best,
                    "history_overlap_tokens": history_best,
                    "history_overlap_source": history_best_rid,
                }
            )
            if len(self.stats["prefix_overlap_samples"]) > _PREFIX_OVERLAP_SAMPLES_KEPT:
                self.stats["prefix_overlap_samples"].pop(0)
            if same_round_best >= self.block_size:
                self.stats["prefix_overlap_same_round_events"] += 1
            if history_best >= self.block_size:
                self.stats["prefix_overlap_history_events"] += 1
        for rid, prompt in new_prompts:
            self._recent_prompts.append((rid, prompt))

    def _record_prefix_cache_hits(
        self, admit_now: list[tuple[int, GenerationRequest]], hit_depths: list[int]
    ) -> None:
        for (_slot, req), L in zip(admit_now, hit_depths):
            # C6: store per-request hit depth for usage reporting
            req._prefix_cache_hit_tokens = L
            if L > 0:
                self.stats["prefix_cache_hits"] += 1
                self.stats["prefix_cache_hit_tokens_saved"] += L
                self.stats["prefix_cache_hit_L_samples"].append(
                    {"request_id": req.request_id, "prompt_tokens": len(req.prompt_ids), "hit_L": L}
                )
                if len(self.stats["prefix_cache_hit_L_samples"]) > _PREFIX_CACHE_HIT_SAMPLES_KEPT:
                    self.stats["prefix_cache_hit_L_samples"].pop(0)
            else:
                self.stats["prefix_cache_misses"] += 1
        total = self.stats["prefix_cache_hits"] + self.stats["prefix_cache_misses"]
        self.stats["prefix_cache_hit_rate"] = (
            (self.stats["prefix_cache_hits"] / total) if total else 0.0
        )

    def _expire_retained_slots(self) -> None:
        now = time.perf_counter()
        for sid in [s for s, r in self.retained.items() if r["expire_t"] <= now]:
            ret = self.retained.pop(sid)
            try:
                self.runner.reset_slot(ret["slot"])
            except Exception:
                logger.exception("reset_slot(%d) failed expiring session %s", ret["slot"], sid)
            self.free_slots.append(ret["slot"])
            self.stats["session_expirations"] += 1

    def _release_all_retained(self) -> None:
        for sid in list(self.retained.keys()):
            ret = self.retained.pop(sid)
            try:
                self.runner.reset_slot(ret["slot"])
            except Exception:
                logger.exception("reset_slot(%d) failed releasing session %s", ret["slot"], sid)

    def _thinking_decode_kwargs(self, slots: list[int]) -> dict[str, object]:
        """Build the optional one-token logits constraint for plain decode."""
        forced: dict[int, int] = {}
        for slot in slots:
            state = self.active[slot].get("thinking_state")
            if state is None:
                continue
            decision = state.force_for(1)
            if decision is not None:
                position, token_id = decision
                if position != 0:  # force_for(1) can only return position zero
                    raise RuntimeError(f"invalid one-token thinking force position: {position}")
                forced[slot] = token_id
        if not forced:
            return {}
        return {"force_token_ids": [forced.get(slot) for slot in slots]}

    def _thinking_mtp_kwargs(self, slots: list[int]) -> dict[str, object]:
        """Build per-slot force maps for an MTP target verify block."""
        positions: dict[int, int] = {}
        token_ids: dict[int, int] = {}
        for slot in slots:
            state = self.active[slot].get("thinking_state")
            if state is None:
                continue
            decision = state.force_for(self.K + 1)
            if decision is not None:
                positions[slot], token_ids[slot] = decision
        if not positions:
            return {}
        return {
            "thinking_force_positions": positions,
            "thinking_force_token_ids": token_ids,
        }

    @staticmethod
    def _thinking_prefill_kwargs(
        admissions: list[tuple[int, GenerationRequest]],
    ) -> dict[str, object]:
        """Force an already-exhausted budget at the initial anchor sample."""
        forced: dict[int, int] = {}
        for slot, req in admissions:
            if req.thinking_budget is None:
                continue
            state = ThinkingBudgetState(req.prompt_ids, req.thinking_budget)
            decision = state.force_for(1)
            if decision is not None:
                position, token_id = decision
                if position != 0:
                    raise RuntimeError(f"invalid prefill thinking force position: {position}")
                forced[slot] = token_id
        if not forced:
            return {}
        return {"force_token_ids": forced}

    # -- slot lifecycle (engine thread) --------------------------------------
    def _activate_slot(
        self, slot: int, req: GenerationRequest, anchor: int, drafts: list[int]
    ) -> None:
        req._prefill_done_at = time.perf_counter()
        _adm_end(req)
        thinking_state = (
            ThinkingBudgetState(req.prompt_ids, req.thinking_budget)
            if req.thinking_budget is not None
            else None
        )
        forced_anchor: int | None = None
        if thinking_state is not None:
            forced = thinking_state.force_for(1)
            if forced is not None:
                if forced[0] != 0:
                    raise RuntimeError(f"invalid prefill thinking force position: {forced[0]}")
                forced_anchor = forced[1]
                if anchor != forced_anchor:
                    raise RuntimeError(
                        "prefill did not honor the thinking-token constraint: "
                        f"expected anchor {forced_anchor}, got {anchor}"
                    )
        # A forced anchor is intentionally not the ordinary greedy reference
        # token, so do not compare it against the unconstrained bootstrap
        # check. Qwen prefill already verified the token at the logits boundary
        # before creating speculative drafts.
        if (
            not self.production
            and req.sampling_params.is_greedy
            and forced_anchor is None
            and req.vision_inputs is None
        ):
            self._admission_bootstrap_check(slot, req, anchor)

        if anchor in self.eos_token_ids:
            if req.stream_channel is not None:
                self._stream_close(req.stream_channel)
            self._finish_request(slot, req, committed_tokens=[], finish_reason="stop")
            return

        stop_sequences = req.stop_sequences or None
        self.active[slot] = {
            "req": req,
            "anchor": anchor,
            "drafts": drafts,
            "committed_tokens": [],
            "sampled": not req.sampling_params.is_greedy,
            # An empty draft list is a valid fallback (e.g. a graph/capacity
            # edge or a non-cacheable multimodal request).  Routing that slot
            # to MTP solely because it is greedy makes ``round`` fail with
            # "requires K drafts" instead of using the correct target decode.
            "speculative_enabled": bool(drafts),
            "last_token": anchor,
            "last_progress_round": self.stats["rounds"],
            "start_time": time.perf_counter(),
            "stop_sequences": stop_sequences,
            "thinking_state": thinking_state,
        }
        st = self.active[slot]
        if req.stop_on_tool_call:
            # Keep tool completion detection in the same parser/state machine
            # used by the HTTP response layer.  It is intentionally separate
            # from the client-facing StreamProcessor: the latter may already
            # have emitted ordinary text and freezes only once the tool XML
            # begins, while this tracker owns the scheduler terminal decision.
            st["tool_tracker"] = StreamProcessor(
                self.tok, thinking_capable=_STOP_TRACKER_THINKING_CAPABLE
            )
        if stop_sequences:
            # N2: private tracker, separate from the client-facing
            # StreamProcessor app.py builds from the same token stream --
            # this one only exists to know (a) whether the reasoning phase
            # has ended (stop must not fire on reasoning text) and (b)
            # which raw tokens are still withheld from the stream channel
            # because their content contribution is an unresolved prefix
            # of a configured stop sequence. See _stop_check_token.
            st["stop_tracker"] = StreamProcessor(
                self.tok, thinking_capable=_STOP_TRACKER_THINKING_CAPABLE
            )
            st["stop_pending_ids"] = []
            st["stop_pending_text"] = ""
        tracer.request_admitted(req.request_id, slot, len(req.prompt_ids))
        # The trace entry is created here because this is the first point at
        # which the request has a validated anchor.  Preserve the actual
        # prefill start separately so ``prefill_ms`` does not silently stay at
        # zero (the old call site never invoked ``prefill_done``) and does not
        # mislabel queue/reset time as GPU prefill.
        prefill_started_at = getattr(
            req,
            "_prefill_started_at",
            getattr(req, "_admitted_at", req._prefill_done_at),
        )
        tracer.prefill_done(
            req.request_id,
            max(0.0, (req._prefill_done_at - prefill_started_at) * 1000.0),
        )

        # The anchor is the request's first generated token -- it must go
        # through the same stop-sequence check as every later token (a
        # single-token stop sequence could match here), and MUST be fed to
        # the tracker even when it doesn't match, or all later matching
        # would be silently missing this token's contribution.
        st["committed_tokens"].append(anchor)
        if thinking_state is not None:
            thinking_state.add_output([anchor])
        matched = self._stop_check_token(st, anchor) if stop_sequences else None
        if matched is not None:
            self._drop_stop_pending_from_committed(st)
            self._finish_request(
                slot, req, st["committed_tokens"], "stop", matched_stop_sequence=matched
            )
            del self.active[slot]
            return

        tool_complete = self._tool_call_check_token(st, anchor)
        if tool_complete:
            if stop_sequences:
                self._flush_stop_pending(st)
            elif req.stream_channel is not None:
                self._stream_put(req.stream_channel, [anchor])
            self._finish_request(slot, req, st["committed_tokens"], "tool_calls")
            del self.active[slot]
            return

        if not stop_sequences and req.stream_channel is not None:
            self._stream_put(req.stream_channel, [anchor])

        if len(st["committed_tokens"]) >= req.max_tokens:
            self._flush_stop_pending(st)
            self._finish_request(slot, req, st["committed_tokens"], finish_reason="length")
            del self.active[slot]
            return

    def _tool_call_check_token(self, st: dict, tok: int) -> bool:
        """Feed one token and report a newly complete parsed tool call."""
        tracker: StreamProcessor | None = st.get("tool_tracker")
        if tracker is None:
            return False
        tracker.add_tokens([tok])
        return bool(tracker.complete_tool_calls())

    def _stop_check_token(self, st: dict, tok: int) -> str | None:
        """Feed one just-committed token through the slot's stop-sequence
        tracker (N2). ``tok`` must already be the last entry appended to
        ``st["committed_tokens"]`` when this is called.

        Returns the matched stop string if committing ``tok`` completes a
        configured stop sequence -- the caller MUST then pop the last
        ``len(st["stop_pending_ids"])`` entries off ``committed_tokens``
        (that count includes ``tok`` itself) and stop generating for this
        slot: none of those tokens were ever flushed to the stream
        channel, so there is nothing to retract, but they must not remain
        in the authoritative committed/result token list either.

        Returns ``None`` when generation should continue normally. ``tok``
        may still be sitting in the pending buffer at that point, withheld
        from the stream channel until a later call resolves the
        ambiguity -- see ``_flush_stop_pending``.
        """
        tracker: StreamProcessor = st["stop_tracker"]
        stop_sequences: list[str] = st["stop_sequences"]
        tracker.add_tokens([tok])
        st["stop_pending_ids"].append(tok)
        st["stop_pending_text"] += "".join(tracker.drain_content())

        if not tracker.thinking_done:
            # Still inside (or ambiguously might still become) the
            # reasoning phase: `drain_content()` reveals nothing here
            # (stop_pending_text stays ""), so there is no content for a
            # stop sequence to match against yet. Nothing is being
            # withheld FOR STOP-MATCHING PURPOSES either -- flush
            # immediately so reasoning keeps streaming to the client with
            # the same latency as a request with no stop_sequences
            # configured (only content, never reasoning, is held back by
            # this method -- see the module docstring's reasoning/content
            # rationale).
            self._flush_stop_pending(st)
            return None

        match = find_earliest_stop_match(st["stop_pending_text"], stop_sequences)
        if match is not None:
            return match[1]

        trimmed = trim_ambiguous_stop_tail(st["stop_pending_text"], stop_sequences)
        if trimmed == st["stop_pending_text"]:
            self._flush_stop_pending(st)
        return None

    def _drop_stop_pending_from_committed(self, st: dict) -> None:
        """After ``_stop_check_token`` returns a match: pop
        ``len(st["stop_pending_ids"])`` entries off the tail of
        ``committed_tokens`` (never flushed, must not appear in the
        result) and, in lockstep, off ``logprobs_acc`` if logprobs were
        requested.

        The one narrow exception: the very first pending token can be the
        request's anchor (fed through the tracker in ``_activate_slot``),
        which has no ``logprobs_acc`` entry of its own. If an ambiguous
        stop-sequence prefix happens to start at the anchor and only
        resolves into a match rounds later, this can trim one entry more
        than strictly necessary from ``logprobs_acc``. That is the safe
        direction (never leaves ``logprobs_acc`` misaligned with the
        *kept* tokens, at worst a couple of legitimate entries short) and
        is accepted rather than tracked precisely -- see docs/api-layer-design.md.
        """
        n_drop = len(st["stop_pending_ids"])
        del st["committed_tokens"][-n_drop:]
        logprobs_acc = st.get("logprobs_acc")
        if logprobs_acc:
            del logprobs_acc[-n_drop:]
        st["stop_pending_ids"] = []
        st["stop_pending_text"] = ""

    def _flush_stop_pending(self, st: dict) -> None:
        """Release any tokens withheld by ``_stop_check_token`` to the
        stream channel, once their content is confirmed free of stop-
        sequence ambiguity (or generation is ending for an unrelated
        reason -- EOS/max_tokens -- so no further tokens can ever arrive
        to complete a match)."""
        ids = st.get("stop_pending_ids")
        if not ids:
            return
        req: GenerationRequest = st["req"]
        if req.stream_channel is not None:
            self._stream_put(req.stream_channel, list(ids))
        st["stop_pending_ids"] = []
        st["stop_pending_text"] = ""

    def _finish_request(
        self,
        slot: int,
        req: GenerationRequest,
        committed_tokens: list[int],
        finish_reason: str,
        logprobs_data: list[dict] | None = None,
        matched_stop_sequence: str | None = None,
    ) -> None:
        if self.runner.capabilities.kv_reservation:
            # A request may stop at EOS long before its declared max_tokens.
            # Release the unmaterialized tail even when session affinity keeps
            # the live prefix/slot resident and therefore skips reset_slot().
            self.runner.release_kv_reservation(slot)
            self._kv_admission_retry_round = 0
        tracer.request_finished(req.request_id, finish_reason)
        prefill_elapsed = max(
            0.0,
            getattr(req, "_prefill_done_at", 0.0) - getattr(req, "_admitted_at", 0.0),
        )
        result = {
            "committed_token_ids": committed_tokens,
            "finish_reason": finish_reason,
            "matched_stop_sequence": matched_stop_sequence,
            "prompt_tokens": len(req.prompt_ids),
            "completion_tokens": len(committed_tokens),
            "prefix_cache_hit_tokens": getattr(req, "_prefix_cache_hit_tokens", 0),
            "prefill_elapsed_s": prefill_elapsed,
        }
        if logprobs_data is not None:
            result["logprobs"] = logprobs_data
        if req.stream_channel is not None:
            self._stream_close(req.stream_channel)
        self._resolve_future(req.future, result)
        self.stats["requests_completed"] += 1
        if self.enable_session_affinity and req.session_id and self.enable_prefix_cache:
            old = self.retained.get(req.session_id)
            if old is not None and old["slot"] != slot:
                self.runner.reset_slot(old["slot"])
                self.free_slots.append(old["slot"])
            slot_state = self.runner.slot_state(slot)
            prior_len = slot_state.kv_len
            committed_full = list(slot_state.committed_tokens)
            self.retained[req.session_id] = {
                "slot": slot,
                "expire_t": time.perf_counter() + self.session_ttl_s,
                "prior_len": prior_len,
                "committed_full": committed_full,
            }
            self.stats["session_retentions"] += 1
            return
        self.runner.reset_slot(slot)
        self.free_slots.append(slot)

    # -- engine thread -------------------------------------------------------
    def _engine_thread_main(self) -> None:
        """Dedicated engine thread entry. Loads model (CUDA context created
        here), then runs the continuous-batching loop until stopped."""
        try:
            self._load_model()
        except Exception as exc:
            logger.exception("FATAL: model loading failed on engine thread")
            # Bug found 2026-07-27 testing DFlash capacity>1 (an oversized
            # blocks_per_slot triggered a real draft-model-load CUDA OOM):
            # this branch set _ready_event without recording the failure, so
            # start()'s wait() returned normally and the server came up
            # "healthy" (/health -> 200) with the engine thread already
            # exited -- every real request would hang forever, since
            # _step_sync never runs. Record the exception so start() can
            # re-raise and fail the server startup loudly instead.
            self._load_error = exc
            self._ready_event.set()
            return
        self._ready_event.set()
        logger.info("engine thread started")

        while not self._stop:
            try:
                self._step_sync()
            except Exception as exc:
                logger.exception("engine round failed, failing active requests")
                for slot, st in list(self.active.items()):
                    self._fail_future(st["req"].future, exc)
                    if st["req"].stream_channel is not None:
                        self._stream_close(st["req"].stream_channel)
                    try:
                        self.runner.reset_slot(slot)
                    except Exception:
                        logger.exception("reset_slot(%d) failed in error recovery", slot)
                    self.free_slots.append(slot)
                self.active.clear()
                self._release_all_retained()
                time.sleep(0.05)

        self._release_all_retained()
        logger.info("engine thread stopped")

    def _drain_requests(self) -> None:
        """Drain all pending requests from the lock-free deque."""
        while self._req_deque:
            self.waiting.append(self._req_deque.popleft())

    def _coalesce_admission_wave(self) -> None:
        """Collect a just-arriving idle wave before launching prefill.

        The HTTP tasks are created together, but the asyncio loop can wake the
        engine after the first append and before the remaining tasks have
        appended.  Starting a 128K prefill at that point permanently splits
        the wave: the other requests cannot join while ``_pending_prefill`` is
        active.  A bounded ``select`` on the existing request pipe lets the
        engine sleep without polling and preserves the zero-CPU idle path.

        This is deliberately limited to an idle engine with spare slots.  It
        never delays a decode round or an already-running incremental prefill,
        and it exits early as soon as all currently free slots have a request.
        """
        # Catch the small race between the normal top-of-round drain and the
        # pipe drain, but only for the opt-in feature.  With the feature off,
        # keep the historical ordering intact; the idle branch's re-drain is
        # the existing fix for that race.
        if self._admission_coalesce_s <= 0.0:
            return
        self._drain_requests()
        if (
            self.active
            or self._pending_prefill is not None
            or not self.waiting
            or len(self.waiting) >= len(self.free_slots)
        ):
            return

        target = len(self.free_slots)
        started = time.perf_counter()
        deadline = started + self._admission_coalesce_s
        while len(self.waiting) < target:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                readable, _, _ = select.select([self._req_pipe_r], [], [], remaining)
            except (OSError, ValueError):
                # Shutdown can close the pipe while a request is being
                # admitted.  The normal admission path remains correct.
                break
            if not readable:
                break
            _drain_pipe(self._req_pipe_r)
            self._drain_requests()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.stats["admission_coalesce_waits"] += 1
        samples = self.stats["admission_coalesce_wait_ms"]
        samples.append(round(elapsed_ms, 3))
        if len(samples) > 128:
            del samples[:-128]

    def _select_admission_requests(self, limit: int) -> list[GenerationRequest]:
        """Select the next admission wave, applying in-batch prefix dedup.

        SGLang's waiting-queue radix tree deliberately deprioritizes exact
        duplicate prompts when they have no prior device-cache match.  The
        first request gets to publish the full prefix; the duplicates then
        enter a later extend wave as persistent-cache hits.  Doing the same
        here is both cheaper and safer than trying to share live slot state
        across requests: Qwen's target KV, GDN checkpoint, and DSpark KV are
        restored through the existing persistent-prefix protocol.

        Deduplication is restricted to block-aligned prompts.  Qwen only
        publishes an exact full-prompt persistent checkpoint at that
        boundary; holding an unaligned duplicate would otherwise turn a
        useful batch into repeated cold prefills.  The environment switch is
        an A/B escape hatch for profiling and incident response.
        """
        if limit <= 0 or not self.waiting:
            return []
        dedup_enabled = (
            os.environ.get("QSR_PREFIX_CACHE_IN_BATCH_DEDUP", "1") != "0"
            and self.runner.capabilities.prefix_cache_dedup
        )
        if not dedup_enabled:
            selected = self.waiting[:limit]
            del self.waiting[:limit]
            return selected

        selected: list[GenerationRequest] = []
        deferred: list[GenerationRequest] = []
        selected_keys: set[tuple[int, ...]] = set()
        inflight_at_start = set(self._prefix_dedup_inflight)
        for req in self.waiting:
            key = tuple(req.prompt_ids)
            aligned = (
                len(req.prompt_ids) >= self.block_size
                and len(req.prompt_ids) % self.block_size == 0
            )
            duplicate_inflight = aligned and key in inflight_at_start
            duplicate_selected = (
                aligned
                and key in selected_keys
                and key not in self._prefix_dedup_published
            )
            if len(selected) < limit and not duplicate_inflight and not duplicate_selected:
                selected.append(req)
                if aligned:
                    selected_keys.add(key)
                    self._prefix_dedup_inflight.add(key)
            else:
                deferred.append(req)
                if duplicate_inflight or duplicate_selected:
                    self.stats["prefix_cache_dedup_deferrals"] += 1
        self.waiting = deferred
        return selected

    def _release_prefix_dedup_keys(
        self,
        requests: list[GenerationRequest] | list[tuple[int, GenerationRequest]],
        *,
        published: bool,
    ) -> None:
        """Release a key after success, deferral, or admission failure."""
        for item in requests:
            req = item[1] if isinstance(item, tuple) else item
            aligned = (
                len(req.prompt_ids) >= self.block_size
                and len(req.prompt_ids) % self.block_size == 0
            )
            if aligned:
                key = tuple(req.prompt_ids)
                self._prefix_dedup_inflight.discard(key)
                if published:
                    self._prefix_dedup_published.add(key)

    def _step_sync(self) -> None:
        """One engine round. Runs entirely on the engine thread."""
        # -- drain request deque + pipe (non-blocking) --
        self._drain_requests()
        _drain_pipe(self._req_pipe_r)
        # A request can be appended between the two operations above.  Drain
        # once more before deciding whether this is the first request of an
        # idle concurrent wave, then use the pipe as a bounded coalescing wait.
        self._coalesce_admission_wave()

        # -- process cancellations (asyncio thread → engine thread) --
        if self._cancel_set and self.active:
            cancelled_slots = []
            for s, st in list(self.active.items()):
                if st["req"].request_id in self._cancel_set:
                    cancelled_slots.append(s)
            for s in cancelled_slots:
                st = self.active.pop(s)
                req = st["req"]
                self._cancel_set.discard(req.request_id)
                self.stats["cancellations"] += 1
                logger.info(
                    "cancelled request %s on slot %d (%d tokens committed)",
                    req.request_id,
                    s,
                    len(st["committed_tokens"]),
                )
                if req.stream_channel is not None:
                    self._stream_close(req.stream_channel)
                self._fail_future(req.future, asyncio.CancelledError("request cancelled by client"))
                try:
                    self.runner.reset_slot(s)
                except Exception:
                    logger.exception("cancel reset_slot(%d) failed", s)
                self.free_slots.append(s)
                self._kv_admission_retry_round = 0
            # Also remove from waiting queue
            if self._cancel_set:
                self.waiting = [r for r in self.waiting if r.request_id not in self._cancel_set]
                self._cancel_set.clear()

        # -- P4b: expire retained warm slots --
        self._expire_retained_slots()

        # -- P4b warm-continue admissions --
        if self.enable_session_affinity and self.retained:
            for req in list(self.waiting):
                if not req.session_id or req.session_id not in self.retained:
                    continue
                ret = self.retained.pop(req.session_id)
                self.waiting.remove(req)
                slot, prior_len = ret["slot"], ret["prior_len"]
                committed_full = ret["committed_full"]
                match = (
                    len(req.prompt_ids) > prior_len
                    and req.prompt_ids[:prior_len] == committed_full[:prior_len]
                )
                if not match:
                    self.runner.reset_slot(slot)
                    self.free_slots.append(slot)
                    self.stats["session_warm_fallbacks"] += 1
                    self.waiting.insert(0, req)
                    continue
                req._prefill_started_at = time.perf_counter()
                try:
                    res = self.runner.mtp_prefill_warm_continue(slot, req.prompt_ids, prior_len)
                except Exception:
                    logger.exception("warm-continue failed for session %s", req.session_id)
                    self.runner.reset_slot(slot)
                    self.free_slots.append(slot)
                    self.stats["session_warm_fallbacks"] += 1
                    self.waiting.insert(0, req)
                    continue
                self.stats["session_warm_continuations"] += 1
                self.stats["session_warm_continuation_samples"].append(
                    {
                        "request_id": req.request_id,
                        "session_id": req.session_id,
                        "slot": slot,
                        "prior_len": prior_len,
                        "prompt_tokens": len(req.prompt_ids),
                        "suffix_len": len(req.prompt_ids) - prior_len,
                    }
                )
                if (
                    len(self.stats["session_warm_continuation_samples"])
                    > _SESSION_WARM_CONTINUATION_SAMPLES_KEPT
                ):
                    self.stats["session_warm_continuation_samples"].pop(0)
                self._activate_slot(slot, req, res["anchor"], res["draft_tokens"])

        # -- A5/B4: advance pending incremental prefill (one chunk per round) --
        if self._pending_prefill is not None:
            if _ADMISSION_PROFILE:
                _prefill_step_t0 = time.perf_counter()
            try:
                done = self.runner.prefill_chunked_step(self._pending_prefill)
            except Exception as exc:
                logger.exception("incremental prefill step failed")
                self._release_prefix_dedup_keys(self._pending_prefill_reqs, published=False)
                for slot, req in self._pending_prefill_reqs:
                    self._fail_future(req.future, exc)
                    if req.stream_channel is not None:
                        self._stream_close(req.stream_channel)
                    try:
                        self.runner.reset_slot(slot)
                    except Exception:
                        logger.exception("reset_slot(%d) failed in prefill recovery", slot)
                    self.free_slots.append(slot)
                self._pending_prefill = None
                self._pending_prefill_reqs = []
            else:
                if _ADMISSION_PROFILE:
                    _adm_logger.info(
                        json.dumps(
                            {
                                "label": "prefill_step",
                                "ms": round(
                                    (time.perf_counter() - _prefill_step_t0) * 1000.0,
                                    3,
                                ),
                                "done": done,
                            }
                        )
                    )
                if done:
                    prefill_result = self._pending_prefill.result
                    admit_now = self._pending_prefill_reqs
                    self._release_prefix_dedup_keys(admit_now, published=True)
                    self.stats["admissions"] += 1
                    self.stats["admission_batch_sizes"].append(len(admit_now))
                    for slot, req in admit_now:
                        anchor = prefill_result[slot]["anchor"]
                        drafts = prefill_result[slot]["draft_tokens"]
                        self._activate_slot(slot, req, anchor, drafts)
                    self._pending_prefill = None
                    self._pending_prefill_reqs = []

        # -- normal admission (starts incremental prefill, non-blocking) --
        if (
            self._pending_prefill is None
            and self.free_slots
            and self.waiting
            and self.stats["rounds"] >= self._kv_admission_retry_round
        ):
            n = min(len(self.free_slots), len(self.waiting))
            admission_reqs = self._select_admission_requests(n)
            # Cache-aware slot assignment: match each prompt to the free
            # slot with the deepest warm KV prefix hit (same-slot reuse).
            admit_now = []
            remaining_slots = list(self.free_slots)
            for req in admission_reqs:
                _adm_start(req)
                # A3 step 7-b (docs/a3-cache-coordinator-design.md §1.1):
                # capability-bit query, not a hasattr probe -- the pattern
                # protocol.py's own module docstring names as the one to
                # eliminate ("the scheduler must consult that BEFORE calling
                # into a family -- never try/except AttributeError"; a bare
                # hasattr probe is the same anti-pattern's sibling). The gate
                # itself stays on ``self.runner`` (capabilities describe the
                # backend, not the coordinator -- SlotResourceManager's own
                # docstring: "Deliberately does NOT re-check ... that gate
                # belongs to the caller"); only the call once gated moves to
                # the coordinator (step 7-g).
                if self.runner.capabilities.prefix_cache and remaining_slots:
                    prefix_cache_key = self.prefix_cache_key_for_request(req)
                    best_slot, _hit = self.slot_resources.find_best_slot_for_prompt(
                        req.prompt_ids,
                        remaining_slots,
                        prefix_cache_key=prefix_cache_key,
                    )
                    remaining_slots.remove(best_slot)
                else:
                    best_slot = remaining_slots.pop(0)
                admit_now.append((best_slot, req))
            self.free_slots = remaining_slots
            for _slot, req in admit_now:
                _adm_phase(req, "slot_match")
            try:
                for slot, _ in admit_now:
                    if not self.runner.slot_state(slot).is_fresh:
                        self.runner.reset_slot(slot)
                for _slot, req in admit_now:
                    _adm_phase(req, "reset")
                if self.runner.capabilities.kv_reservation:
                    # Match vLLM's full_sequence_must_fit admission gate, but
                    # retain an explicit per-request remainder because our
                    # chunked prefill spans engine rounds. The first capacity
                    # miss preserves FCFS: it and the unexamined tail go back
                    # to waiting, and no GPU write has happened yet.
                    first_wait = None
                    for index, (slot, req) in enumerate(admit_now):
                        total_tokens = len(req.prompt_ids) + req.max_tokens + self.K
                        if not self.runner.reserve_kv_capacity(slot, total_tokens):
                            first_wait = index
                            break
                    if first_wait is not None:
                        deferred = admit_now[first_wait:]
                        admit_now = admit_now[:first_wait]
                        self.waiting = [req for _slot, req in deferred] + self.waiting
                        self._release_prefix_dedup_keys(deferred, published=False)
                        self.free_slots.extend(slot for slot, _req in deferred)
                        self.stats["kv_admission_waits"] += 1
                        self._kv_admission_retry_round = self.stats["rounds"] + 16
                        for _slot, req in deferred:
                            _adm_phase(req, "kv_wait")

                new_slots = [s for s, _ in admit_now]
                new_prompts = [r.prompt_ids for _, r in admit_now]
                prefill_state = None
                hit_depths = []
                if admit_now:
                    # reconcile_prefix_hit returns a PrefixHit (runtime/backends/
                    # protocol.py); .effective is the length safe to skip prefill
                    # for (== state_hit, never kv_hit -- see PrefixHit's own
                    # docstring and docs/a3-cache-coordinator-design.md §3).
                    # A3 step 7-g: routed through the coordinator, not
                    # self.runner directly -- see self.slot_resources's docstring.
                    hit_depths = [
                        self.slot_resources.reconcile_prefix_hit(
                            prompt,
                            prefix_cache_key=self.prefix_cache_key_for_request(req),
                        ).effective
                        for prompt, (_slot, req) in zip(new_prompts, admit_now)
                    ]
                    for _slot, req in admit_now:
                        _adm_phase(req, "reconcile")
                    # E2-b: only non-greedy requests carry an entry -- see the
                    # MTP-round call site's identical comment on why a missing
                    # entry preserves prior (greedy) behavior byte-for-byte.
                    params_per_slot = {
                        slot: req.sampling_params
                        for slot, req in admit_now
                        if not req.sampling_params.is_greedy
                    }
                    prefill_kwargs = self._thinking_prefill_kwargs(admit_now)
                    vision_inputs_per_slot = {
                        slot: req.vision_inputs
                        for slot, req in admit_now
                        if req.vision_inputs is not None
                    }
                    if vision_inputs_per_slot:
                        prefill_kwargs["vision_inputs_per_slot"] = vision_inputs_per_slot
                    prefill_started_at = time.perf_counter()
                    for _slot, req in admit_now:
                        req._prefill_started_at = prefill_started_at
                    prefill_state = self.runner.prefill_chunked_begin(
                        new_slots,
                        new_prompts,
                        chunk_size=self._prefill_chunk_size,
                        params_per_slot=params_per_slot,
                        **prefill_kwargs,
                    )
                    for _slot, req in admit_now:
                        _adm_phase(req, "prefill_begin")
            except Exception as exc:
                logger.exception("admission failed for %d request(s)", len(admit_now))
                self._release_prefix_dedup_keys(admit_now, published=False)
                for slot, req in admit_now:
                    self._fail_future(req.future, exc)
                    if req.stream_channel is not None:
                        self._stream_close(req.stream_channel)
                    try:
                        self.runner.reset_slot(slot)
                    except Exception:
                        logger.exception("reset_slot(%d) failed in admission recovery", slot)
                    self.free_slots.append(slot)
            else:
                if prefill_state is None:
                    pass
                elif prefill_state.done:
                    # Short prompt: prefill completed immediately
                    self._release_prefix_dedup_keys(admit_now, published=True)
                    self._log_prefix_overlap(admit_now)
                    self._record_prefix_cache_hits(admit_now, hit_depths)
                    self.stats["admissions"] += 1
                    self.stats["admission_batch_sizes"].append(len(admit_now))
                    prefill_result = prefill_state.result
                    for slot, req in admit_now:
                        anchor = prefill_result[slot]["anchor"]
                        drafts = prefill_result[slot]["draft_tokens"]
                        self._activate_slot(slot, req, anchor, drafts)
                else:
                    # Long prompt: prefill will be advanced incrementally
                    self._log_prefix_overlap(admit_now)
                    self._record_prefix_cache_hits(admit_now, hit_depths)
                    self._pending_prefill = prefill_state
                    self._pending_prefill_reqs = admit_now

        # -- idle: block on pipe (zero CPU, instant wakeup) --
        # Only block when BOTH active and waiting are empty.
        # If waiting has requests (e.g. admission failed and re-queued),
        # we must loop back to retry admission, NOT block on the pipe.
        if not self.active and not self.waiting and self._pending_prefill is None:
            # Re-drain immediately before blocking. `self.waiting` was filled
            # at the top of this round and `_drain_pipe` right after it ate
            # every wakeup byte then pending -- so a request appended between
            # those two lines is in `_req_deque` with its wakeup byte already
            # consumed, and the emptiness test above cannot see it. Blocking
            # here would then sleep forever on a request that has already
            # arrived.
            #
            # Observed live 2026-08-01: an agent's follow-up turn landed 152 ms
            # after the previous one finished -- precisely while the engine was
            # winding down into this branch -- and the engine stopped stepping
            # (`rounds` frozen, GPU idle, `active`/`waiting` both empty) until
            # the client timed out. Not intermittent: a conversational client
            # sends its next turn in exactly this window every time.
            #
            # Ordering is what makes this airtight, so keep it: a request that
            # arrives before the re-drain is seen here; one that arrives after
            # it still has its wakeup byte in the pipe, because nothing drains
            # the pipe between the re-drain and the blocking read.
            self._drain_requests()
            if self.waiting:
                return

            # Set pipe to blocking mode for efficient idle wait
            os.set_blocking(self._req_pipe_r, True)
            try:
                os.read(self._req_pipe_r, 1)  # blocks until request or stop
            except OSError:
                pass
            os.set_blocking(self._req_pipe_r, False)
            if self._stop:
                return
            self._drain_requests()
            _drain_pipe(self._req_pipe_r)
            return
        elif not self.active and self.waiting and self._pending_prefill is None:
            # Have waiting requests but no active slots — retry admission
            # next round without blocking. Brief sleep to avoid hot-spin
            # if admission keeps failing (e.g. OOM).
            self._kv_admission_retry_round = 0
            time.sleep(0.01)
            return

        # -- decode round (hot path, zero wait) --
        active_slots = list(self.active.keys())
        # N1: structured output (json_object/json_schema) is rejected at
        # the API layer (server/app.py::_reject_unsupported_response_format)
        # rather than routed here -- see docs/api-layer-design.md §5.1 for
        # why grammar-masking has no reachable hook in this decode loop.
        # grammar_slots stays permanently empty; classify_decode_slots keeps
        # the parameter (still covered by tests/test_laguna_server_integration.py)
        # for the day a real implementation lands.
        # E2-b: a backend with no MTP (e.g. Laguna without DFlash) routes
        # every slot through the plain sampled path regardless of is_greedy;
        # an MTP-capable backend (Laguna+DFlash) now routes BOTH greedy and
        # sampled requests through the MTP branch -- see classify_decode_slots.
        mtp_slots, plain_sampled_slots = classify_decode_slots(
            active_slots,
            self.active,
            [],
            self.runner.has_speculative_decode,
            sampled_mtp_capable=getattr(
                self.runner,
                "supports_sampled_speculative_decode",
                not self._external_qwen_spec_enabled,
            ),
        )

        self.stats["rounds"] += 1
        self.stats["round_batch_sizes"].append(len(active_slots))

        newly_finished: list[int] = []

        # -- plain sampled decode (no MTP, simple autoregressive) --
        if plain_sampled_slots:
            self.stats["sampled_decode_rounds"] += 1
            slot_ids = plain_sampled_slots
            token_ids = [self.active[s]["last_token"] for s in slot_ids]
            kv_lengths = [self.runner.slot_state(s).kv_len for s in slot_ids]
            params_list = [self.active[s]["req"].sampling_params for s in slot_ids]
            any_lp = any(self.active[s]["req"].logprobs for s in slot_ids)
            top_lp = (
                max(
                    (self.active[s]["req"].top_logprobs for s in slot_ids),
                    default=0,
                )
                if any_lp
                else 0
            )
            decode_result = self.runner.decode_batch_sampled(
                slot_ids,
                token_ids,
                kv_lengths,
                params_list,
                return_logprobs=any_lp,
                top_logprobs=top_lp,
                **self._thinking_decode_kwargs(slot_ids),
            )
            if any_lp:
                next_tokens, lp_batch = decode_result
            else:
                next_tokens = decode_result
                lp_batch = None
            for i, (s, tok) in enumerate(zip(slot_ids, next_tokens)):
                st = self.active[s]
                req: GenerationRequest = st["req"]
                if len(st["committed_tokens"]) >= req.max_tokens:
                    self._flush_stop_pending(st)
                    self._finish_request(
                        s,
                        req,
                        st["committed_tokens"],
                        "length",
                        logprobs_data=st.get("logprobs_acc"),
                    )
                    newly_finished.append(s)
                    continue
                if tok in self.eos_token_ids:
                    self._flush_stop_pending(st)
                    self._finish_request(
                        s,
                        req,
                        st["committed_tokens"],
                        "stop",
                        logprobs_data=st.get("logprobs_acc"),
                    )
                    newly_finished.append(s)
                    continue
                st["committed_tokens"].append(tok)
                if st.get("thinking_state") is not None:
                    st["thinking_state"].add_output([tok])
                st["last_token"] = tok
                st["last_progress_round"] = self.stats["rounds"]
                if lp_batch is not None and st["req"].logprobs:
                    st.setdefault("logprobs_acc", []).append(lp_batch[i])
                # N2: stop-sequence check (must run before streaming tok --
                # see _stop_check_token/_flush_stop_pending).
                stop_sequences = st.get("stop_sequences")
                if stop_sequences:
                    matched = self._stop_check_token(st, tok)
                    if matched is not None:
                        self._drop_stop_pending_from_committed(st)
                        self._finish_request(
                            s,
                            req,
                            st["committed_tokens"],
                            "stop",
                            logprobs_data=st.get("logprobs_acc"),
                            matched_stop_sequence=matched,
                        )
                        newly_finished.append(s)
                        continue
                elif req.stream_channel is not None:
                    self._stream_put(req.stream_channel, [tok])
                if self._tool_call_check_token(st, tok):
                    if stop_sequences:
                        self._flush_stop_pending(st)
                    self._finish_request(
                        s,
                        req,
                        st["committed_tokens"],
                        "tool_calls",
                        logprobs_data=st.get("logprobs_acc"),
                    )
                    newly_finished.append(s)
                    continue
                if len(st["committed_tokens"]) >= req.max_tokens:
                    self._flush_stop_pending(st)
                    self._finish_request(
                        s,
                        req,
                        st["committed_tokens"],
                        "length",
                        logprobs_data=st.get("logprobs_acc"),
                    )
                    newly_finished.append(s)

        # -- MTP verify/commit round (E2-b: greedy AND sampled) --
        if mtp_slots:
            _round_t0 = time.perf_counter()
            any_lp_g = any(self.active[s]["req"].logprobs for s in mtp_slots)
            top_lp_g = (
                max(
                    (self.active[s]["req"].top_logprobs for s in mtp_slots),
                    default=0,
                )
                if any_lp_g
                else 0
            )
            # Only slots actually marked "sampled" (temperature>0) carry a
            # SamplingParams entry -- a missing entry (or None) makes
            # dflash_round/mtp_verify_and_commit_batch take the exact prior
            # (greedy) code path for that slot, byte-for-byte.
            params_per_slot = {
                s: self.active[s]["req"].sampling_params
                for s in mtp_slots
                if self.active[s].get("sampled")
            }
            decisions = self.runner.mtp_verify_and_commit_batch(
                mtp_slots,
                {s: self.active[s]["anchor"] for s in mtp_slots},
                {s: self.active[s]["drafts"] for s in mtp_slots},
                params_per_slot=params_per_slot,
                return_logprobs=any_lp_g,
                top_logprobs=top_lp_g,
                **self._thinking_mtp_kwargs(mtp_slots),
            )
            _round_ms = (time.perf_counter() - _round_t0) * 1000
            _bookkeep_t0 = time.perf_counter()

            for s in mtp_slots:
                st = self.active[s]
                req = st["req"]
                decision = decisions[s]
                new_tokens = decision["committed"]
                na = decision.get("num_accepted", 0)
                if req.logprobs and "logprobs" in decision:
                    st.setdefault("logprobs_acc", []).extend(decision["logprobs"])
                if self._external_qwen_spec_enabled:
                    self.stats["dspark_rounds"] += 1
                    self.stats["dspark_accepted_tokens"] += na
                    dspark_hist = self.stats["dspark_acceptance_histogram"]
                    dspark_hist[min(na, len(dspark_hist) - 1)] += 1
                if st.get("sampled"):
                    # E2-b: recorded separately from mtp_acceptance_histogram
                    # (greedy-only) -- see the stats dict's own comment for why.
                    self.stats["mtp_sampled_total_accepted"] += na
                    self.stats["mtp_sampled_total_draft"] += len(st["drafts"])
                    self.stats["mtp_sampled_rounds"] += 1
                elif na >= 0:
                    # Clamp into a final overflow bucket rather than dropping:
                    # a discarded sample is indistinguishable from one that
                    # never happened, which is how this went unnoticed.
                    hist = self.stats["mtp_acceptance_histogram"]
                    hist[min(na, len(hist) - 1)] += 1
                    metrics.record_mtp_acceptance(na)

                # N2: a single MTP round can commit several draft tokens at
                # once -- a stop sequence can land anywhere inside that
                # batch, so tokens are appended to committed_tokens (and,
                # for stop-configured slots, fed through the tracker) ONE
                # AT A TIME rather than batched via `kept` + extend-at-end,
                # so the loop can stop exactly at the match and discard
                # everything the backend drafted past it.
                stop_sequences = st.get("stop_sequences")
                matched_stop: str | None = None
                tool_complete = False
                finish_reason: str | None = None
                kept: list[int] = []
                for t in new_tokens:
                    if len(st["committed_tokens"]) >= req.max_tokens:
                        finish_reason = "length"
                        break
                    if t in self.eos_token_ids:
                        finish_reason = "stop"
                        break
                    st["committed_tokens"].append(t)
                    kept.append(t)
                    if st.get("thinking_state") is not None:
                        st["thinking_state"].add_output([t])
                    if stop_sequences:
                        matched_stop = self._stop_check_token(st, t)
                        if matched_stop is not None:
                            finish_reason = "stop"
                            break
                    tool_complete = self._tool_call_check_token(st, t)
                    if tool_complete:
                        finish_reason = "tool_calls"
                        break
                if kept:
                    st["last_progress_round"] = self.stats["rounds"]
                    if self._external_qwen_spec_enabled:
                        self.stats["dspark_committed_tokens"] += len(kept)
                if matched_stop is not None:
                    self._drop_stop_pending_from_committed(st)
                elif stop_sequences and tool_complete:
                    self._flush_stop_pending(st)
                elif not stop_sequences and kept and req.stream_channel is not None:
                    self._stream_put(req.stream_channel, kept)
                if finish_reason is None and len(st["committed_tokens"]) >= req.max_tokens:
                    finish_reason = "length"

                if finish_reason is None:
                    st["anchor"] = decision["next_anchor"]
                    st["drafts"] = decision["next_draft_tokens"]
                    tracer.decode_round(req.request_id, self.stats["rounds"], len(kept), _round_ms)
                    continue

                if stop_sequences and matched_stop is None:
                    self._flush_stop_pending(st)
                self._finish_request(
                    s,
                    req,
                    st["committed_tokens"],
                    finish_reason,
                    logprobs_data=st.get("logprobs_acc"),
                    matched_stop_sequence=matched_stop,
                )
                newly_finished.append(s)

            round_profile.engine_step(_round_ms, (time.perf_counter() - _bookkeep_t0) * 1000)

        for s in newly_finished:
            del self.active[s]

        # -- request timeout: reclaim slots exceeding max duration --
        if self.request_timeout_s > 0 and self.active:
            now = time.perf_counter()
            timed_out = find_timed_out_slots(self.active, now, self.request_timeout_s)
            for s in timed_out:
                st = self.active.pop(s)
                req = st["req"]
                elapsed = now - st.get("start_time", now)
                self.stats["timeouts"] += 1
                logger.warning(
                    "TIMEOUT: slot %d request %s exceeded %.0fs limit (%.0fs elapsed, "
                    "%d tokens committed)",
                    s,
                    req.request_id,
                    self.request_timeout_s,
                    elapsed,
                    len(st["committed_tokens"]),
                )
                if req.stream_channel is not None:
                    self._stream_close(req.stream_channel)
                self._fail_future(
                    req.future,
                    TimeoutError(
                        f"request timed out after {elapsed:.0f}s "
                        f"(limit {self.request_timeout_s:.0f}s)"
                    ),
                )
                try:
                    self.runner.reset_slot(s)
                except Exception:
                    logger.exception("timeout reset_slot(%d) failed", s)
                self.free_slots.append(s)

        # -- watchdog: force-reclaim slots that made no progress --
        if self.watchdog_max_stale_rounds > 0 and self.active:
            current_round = self.stats["rounds"]
            stale_slots = find_stale_slots(
                self.active, current_round, self.watchdog_max_stale_rounds
            )
            for s in stale_slots:
                st = self.active.pop(s)
                req = st["req"]
                kv_len = self.runner.slot_state(s).kv_len if self.runner else -1
                committed = len(st["committed_tokens"])
                event = {
                    "slot": s,
                    "round": current_round,
                    "stale_rounds": current_round - st.get("last_progress_round", 0),
                    "kv_len": kv_len,
                    "committed_tokens": committed,
                    "request_id": req.request_id,
                }
                self.stats["watchdog_triggers"] += 1
                self.stats["watchdog_events"].append(event)
                if len(self.stats["watchdog_events"]) > 50:
                    self.stats["watchdog_events"].pop(0)
                logger.error(
                    "WATCHDOG: slot %d wedged (no progress for %d rounds, "
                    "kv_len=%d, committed=%d) — force-reclaiming",
                    s,
                    event["stale_rounds"],
                    kv_len,
                    committed,
                )
                self._fail_future(
                    req.future,
                    RuntimeError(
                        f"slot {s} watchdog: no progress for {event['stale_rounds']} rounds"
                    ),
                )
                if req.stream_channel is not None:
                    self._stream_close(req.stream_channel)
                try:
                    self.runner.reset_slot(s)
                except Exception:
                    logger.exception("watchdog reset_slot(%d) failed", s)
                self.free_slots.append(s)

        # Yield GIL to asyncio event loop so HTTP requests (health, SSE)
        # can be processed between GPU rounds. Without this, the engine
        # thread starves the event loop during long generations.
        time.sleep(0)
