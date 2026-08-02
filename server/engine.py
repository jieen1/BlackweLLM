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
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from runtime.architecture import ArchitectureSpec, parse_architecture
from runtime.model_registry import IMPLEMENTED_BACKENDS
from runtime.sampling import SamplingParams
from runtime.slot_resource_manager import SlotResourceManager
from server.formats.stop import find_earliest_stop_match, trim_ambiguous_stop_tail
from server.formats.stream import StreamProcessor
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
    ``docs/api-layer-design.md`` §7.1) or every request on a backend with
    no MTP capability at all still go through the plain
    ``decode_batch_sampled`` path (``plain_sampled_slots``).
    """
    if not mtp_capable:
        return [], list(active_slots)
    mtp_slots = [s for s in active_slots if s not in grammar_slots]
    plain_sampled_slots = [s for s in active_slots if s in grammar_slots]
    return mtp_slots, plain_sampled_slots


logger = logging.getLogger("qwen_sm120_server.engine")

_PREFIX_OVERLAP_HISTORY = 64
_PREFIX_OVERLAP_SAMPLES_KEPT = 200
_PREFIX_CACHE_HIT_SAMPLES_KEPT = 200
_SESSION_WARM_CONTINUATION_SAMPLES_KEPT = 200


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
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
        self.MODEL = model
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
            from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS

            self.K = NUM_SPECULATIVE_TOKENS

        # CUDA Graph slot budget:
        # - Decode CG (M=1) captures against ONE slot (the last), not capacity
        #   slots.  After capture the slot is reset before real use.
        # - DFlash draft/verify CGs use shared scratch buffers and replay
        #   sequentially per slot; they do NOT need extra physical slots.
        # So: +1 slot for decode CG warmup (non-DFlash only), +0 for DFlash.
        cg_extra = 0
        if enable_cudagraph and not enable_dflash:
            cg_extra = 1  # single warmup slot for M=1 decode CG capture
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

        # Laguna ships a custom AutoConfig/tokenizer class (configuration_laguna.py)
        # that transformers only loads with trust_remote_code=True; without it,
        # config validation falls onto a generic path that chokes on Laguna's
        # yarn rope_parameters (KeyError: 'original_max_position_embeddings').
        self.tok = AutoTokenizer.from_pretrained(self.MODEL, trust_remote_code=True)
        self.eos_token_id = self.tok.eos_token_id
        try:
            from transformers import GenerationConfig

            gen_cfg_eos = GenerationConfig.from_pretrained(self.MODEL).eos_token_id
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
            "mtp_acceptance_histogram": [0] * 5,
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
        return SlotResourceManager(self.runner, self.architecture_spec)

    # -- model loading (engine thread only) --------------------------------
    def _load_model(self) -> None:
        """Load model + create the Laguna runner. MUST run on engine thread."""
        self._load_laguna_model()

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

    async def submit(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        session_id: str | None = None,
        sampling_params: SamplingParams | None = None,
        stop_sequences: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
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
        )
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
        )
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
        new_prompts = [(req.request_id, req.prompt_ids) for _, req in admit_now]
        for i, (rid, prompt) in enumerate(new_prompts):
            same_round_best = 0
            for j, (_, other_prompt) in enumerate(new_prompts):
                if j == i:
                    continue
                same_round_best = max(
                    same_round_best, _longest_common_prefix_len(prompt, other_prompt)
                )
            history_best = 0
            history_best_rid: str | None = None
            for other_rid, other_prompt in self._recent_prompts:
                overlap = _longest_common_prefix_len(prompt, other_prompt)
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

    # -- slot lifecycle (engine thread) --------------------------------------
    def _activate_slot(
        self, slot: int, req: GenerationRequest, anchor: int, drafts: list[int]
    ) -> None:
        if not self.production and req.sampling_params.is_greedy:
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
            "last_token": anchor,
            "last_progress_round": self.stats["rounds"],
            "start_time": time.perf_counter(),
            "stop_sequences": stop_sequences,
        }
        st = self.active[slot]
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

        # The anchor is the request's first generated token -- it must go
        # through the same stop-sequence check as every later token (a
        # single-token stop sequence could match here), and MUST be fed to
        # the tracker even when it doesn't match, or all later matching
        # would be silently missing this token's contribution.
        st["committed_tokens"].append(anchor)
        matched = self._stop_check_token(st, anchor) if stop_sequences else None
        if matched is not None:
            self._drop_stop_pending_from_committed(st)
            self._finish_request(
                slot, req, st["committed_tokens"], "stop", matched_stop_sequence=matched
            )
            del self.active[slot]
            return

        if not stop_sequences and req.stream_channel is not None:
            self._stream_put(req.stream_channel, [anchor])

        if len(st["committed_tokens"]) >= req.max_tokens:
            self._flush_stop_pending(st)
            self._finish_request(slot, req, st["committed_tokens"], finish_reason="length")
            del self.active[slot]
            return

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
        tracer.request_finished(req.request_id, finish_reason)
        result = {
            "committed_token_ids": committed_tokens,
            "finish_reason": finish_reason,
            "matched_stop_sequence": matched_stop_sequence,
            "prompt_tokens": len(req.prompt_ids),
            "completion_tokens": len(committed_tokens),
            "prefix_cache_hit_tokens": getattr(req, "_prefix_cache_hit_tokens", 0),
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

    def _step_sync(self) -> None:
        """One engine round. Runs entirely on the engine thread."""
        # -- drain request deque + pipe (non-blocking) --
        self._drain_requests()
        _drain_pipe(self._req_pipe_r)

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
            try:
                done = self.runner.prefill_chunked_step(self._pending_prefill)
            except Exception as exc:
                logger.exception("incremental prefill step failed")
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
                if done:
                    prefill_result = self._pending_prefill.result
                    admit_now = self._pending_prefill_reqs
                    self.stats["admissions"] += 1
                    self.stats["admission_batch_sizes"].append(len(admit_now))
                    for slot, req in admit_now:
                        anchor = prefill_result[slot]["anchor"]
                        drafts = prefill_result[slot]["draft_tokens"]
                        self._activate_slot(slot, req, anchor, drafts)
                    self._pending_prefill = None
                    self._pending_prefill_reqs = []

        # -- normal admission (starts incremental prefill, non-blocking) --
        if self._pending_prefill is None and self.free_slots and self.waiting:
            n = min(len(self.free_slots), len(self.waiting))
            # Cache-aware slot assignment: match each prompt to the free
            # slot with the deepest warm KV prefix hit (same-slot reuse).
            admit_now = []
            remaining_slots = list(self.free_slots)
            for _ in range(n):
                req = self.waiting.pop(0)
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
                    best_slot, _hit = self.slot_resources.find_best_slot_for_prompt(
                        req.prompt_ids,
                        remaining_slots,
                    )
                    remaining_slots.remove(best_slot)
                else:
                    best_slot = remaining_slots.pop(0)
                admit_now.append((best_slot, req))
            self.free_slots = remaining_slots
            new_slots = [s for s, _ in admit_now]
            new_prompts = [r.prompt_ids for _, r in admit_now]
            try:
                for slot, _ in admit_now:
                    if not self.runner.slot_state(slot).is_fresh:
                        self.runner.reset_slot(slot)
                # reconcile_prefix_hit returns a PrefixHit (runtime/backends/
                # protocol.py); .effective is the length safe to skip prefill
                # for (== state_hit, never kv_hit -- see PrefixHit's own
                # docstring and docs/a3-cache-coordinator-design.md §3).
                # A3 step 7-g: routed through the coordinator, not
                # self.runner directly -- see self.slot_resources's docstring.
                hit_depths = [
                    self.slot_resources.reconcile_prefix_hit(p).effective for p in new_prompts
                ]
                # E2-b: only non-greedy requests carry an entry -- see the
                # MTP-round call site's identical comment on why a missing
                # entry preserves prior (greedy) behavior byte-for-byte.
                # Harmless/ignored on a backend without DFlash enabled
                # (LagunaBackend.prefill_chunked_begin's non-DFlash branch
                # never looks at it -- those requests already go through
                # decode_batch_sampled exclusively, per classify_decode_slots).
                params_per_slot = {
                    slot: req.sampling_params
                    for slot, req in admit_now
                    if not req.sampling_params.is_greedy
                }
                prefill_state = self.runner.prefill_chunked_begin(
                    new_slots,
                    new_prompts,
                    chunk_size=self._prefill_chunk_size,
                    params_per_slot=params_per_slot,
                )
            except Exception as exc:
                logger.exception("admission failed for %d request(s)", len(admit_now))
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
                self._log_prefix_overlap(admit_now)
                self._record_prefix_cache_hits(admit_now, hit_depths)
                if prefill_state.done:
                    # Short prompt: prefill completed immediately
                    self.stats["admissions"] += 1
                    self.stats["admission_batch_sizes"].append(len(admit_now))
                    prefill_result = prefill_state.result
                    for slot, req in admit_now:
                        anchor = prefill_result[slot]["anchor"]
                        drafts = prefill_result[slot]["draft_tokens"]
                        self._activate_slot(slot, req, anchor, drafts)
                else:
                    # Long prompt: prefill will be advanced incrementally
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
            active_slots, self.active, [], self.runner.has_speculative_decode
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
            )
            _round_ms = (time.perf_counter() - _round_t0) * 1000

            for s in mtp_slots:
                st = self.active[s]
                req = st["req"]
                decision = decisions[s]
                new_tokens = decision["committed"]
                na = decision.get("num_accepted", 0)
                if req.logprobs and "logprobs" in decision:
                    st.setdefault("logprobs_acc", []).extend(decision["logprobs"])
                if st.get("sampled"):
                    # E2-b: recorded separately from mtp_acceptance_histogram
                    # (greedy-only) -- see the stats dict's own comment for why.
                    self.stats["mtp_sampled_total_accepted"] += na
                    self.stats["mtp_sampled_total_draft"] += len(st["drafts"])
                    self.stats["mtp_sampled_rounds"] += 1
                elif 0 <= na < len(self.stats["mtp_acceptance_histogram"]):
                    self.stats["mtp_acceptance_histogram"][na] += 1

                # N2: a single MTP round can commit several draft tokens at
                # once -- a stop sequence can land anywhere inside that
                # batch, so tokens are appended to committed_tokens (and,
                # for stop-configured slots, fed through the tracker) ONE
                # AT A TIME rather than batched via `kept` + extend-at-end,
                # so the loop can stop exactly at the match and discard
                # everything the backend drafted past it.
                stop_sequences = st.get("stop_sequences")
                matched_stop: str | None = None
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
                    if stop_sequences:
                        matched_stop = self._stop_check_token(st, t)
                        if matched_stop is not None:
                            finish_reason = "stop"
                            break
                if kept:
                    st["last_progress_round"] = self.stats["rounds"]
                if matched_stop is not None:
                    self._drop_stop_pending_from_committed(st)
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
