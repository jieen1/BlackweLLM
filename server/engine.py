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

from runtime.sampling import SamplingParams
from runtime.structured_output import GrammarState, ResponseFormat
from server.tracing import tracer

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


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
    """Split one round's active slots into (greedy_speculative, sampled).

    Speculative verify/commit only ever applies to a greedy request on a
    backend whose runner reports ``spec.has_mtp``. Everything else --
    explicit sampling, grammar-constrained structured output, or Laguna
    without DFlash wired up -- goes through the plain per-step
    ``decode_batch_sampled`` path. Greedy there is just temperature=0, so
    this is only a routing distinction.
    """
    if not mtp_capable:
        return [], list(active_slots)
    greedy_slots = [
        s for s in active_slots if not active[s].get("sampled") and s not in grammar_slots
    ]
    sampled_slots = [s for s in active_slots if active[s].get("sampled") or s in grammar_slots]
    return greedy_slots, sampled_slots


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
    response_format: dict | None = None
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
    """

    MODEL = "poolside/Laguna-S-2.1-NVFP4"
    K = 0
    BACKEND = "laguna"

    def __init__(
        self,
        *,
        backend: str = "laguna",
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
        if backend != self.BACKEND:
            raise ValueError(f"backend={backend!r} is unsupported; expected {self.BACKEND!r}")
        self.backend_name = self.BACKEND
        self.enable_dflash = enable_dflash
        self.MODEL = self.__class__.MODEL
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
        }

        self.runner = None
        self._prefill_chunk_size = 512
        self._near_tie_margin_diag = None
        self.near_tie_logit_margin = 0.0

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
            # classify_decode_slots (server/engine.py) routes greedy
            # requests to mtp_verify_and_commit_batch, which
            # LagunaBackend now implements by delegating to
            # DFlashEngine.dflash_round per slot. Non-greedy requests are
            # unaffected -- classify_decode_slots still routes them through
            # decode_batch_sampled. Must run before start()'s admission loop
            # (same "capture before any real slot use" requirement as
            # _ensure_decode_cg above): DFlashEngine.__init__ captures its
            # own draft/verify CUDA Graphs, which scribble dummy warmup data
            # into physical slots.
            dflash_cuda_graph = self.runner.enable_dflash(
                num_speculative_tokens=self.K
            )
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
        if not self._ready_event.wait(timeout=600):
            raise RuntimeError("Engine thread failed to initialize model within 600s")
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
        response_format: dict | None = None,
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
            response_format=response_format,
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
        response_format: dict | None = None,
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
            response_format=response_format,
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
        committed_tokens = [anchor]
        if req.stream_channel is not None:
            self._stream_put(req.stream_channel, [anchor])
        if len(committed_tokens) >= req.max_tokens:
            self._finish_request(
                slot, req, committed_tokens=committed_tokens, finish_reason="length"
            )
            return
        self.active[slot] = {
            "req": req,
            "anchor": anchor,
            "drafts": drafts,
            "committed_tokens": committed_tokens,
            "sampled": not req.sampling_params.is_greedy,
            "last_token": anchor,
            "last_progress_round": self.stats["rounds"],
            "start_time": time.perf_counter(),
        }
        tracer.request_admitted(req.request_id, slot, len(req.prompt_ids))
        # C3: initialize grammar state for structured output
        fmt = ResponseFormat.from_api(req.response_format)
        if fmt.is_constrained:
            self.active[slot]["grammar"] = GrammarState(fmt, self.tok)
        else:
            self.active[slot]["grammar"] = None

    def _finish_request(
        self,
        slot: int,
        req: GenerationRequest,
        committed_tokens: list[int],
        finish_reason: str,
        logprobs_data: list[dict] | None = None,
    ) -> None:
        tracer.request_finished(req.request_id, finish_reason)
        result = {
            "committed_token_ids": committed_tokens,
            "finish_reason": finish_reason,
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
            admit_now = [(self.free_slots.pop(0), self.waiting.pop(0)) for _ in range(n)]
            new_slots = [s for s, _ in admit_now]
            new_prompts = [r.prompt_ids for _, r in admit_now]
            try:
                for slot, _ in admit_now:
                    if not self.runner.slot_state(slot).is_fresh:
                        self.runner.reset_slot(slot)
                hit_depths = [self.runner.reconcile_prefix_hit(p) for p in new_prompts]
                prefill_state = self.runner.prefill_chunked_begin(
                    new_slots, new_prompts, chunk_size=self._prefill_chunk_size
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
        # C3: structured output requests use sampled path with grammar masking
        grammar_slots = [s for s in active_slots if self.active[s].get("grammar") is not None]
        # E1: a backend with no MTP (e.g. Laguna) routes every slot through
        # the sampled path regardless of is_greedy -- see classify_decode_slots.
        greedy_slots, sampled_slots = classify_decode_slots(
            active_slots, self.active, grammar_slots, self.runner.has_speculative_decode
        )

        self.stats["rounds"] += 1
        self.stats["round_batch_sizes"].append(len(active_slots))

        newly_finished: list[int] = []

        # -- sampled decode (no MTP, simple autoregressive) --
        if sampled_slots:
            self.stats["sampled_decode_rounds"] += 1
            slot_ids = sampled_slots
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
                # C3: advance grammar state
                grammar = st.get("grammar")
                if grammar is not None:
                    grammar.accept(tok)
                if req.stream_channel is not None:
                    self._stream_put(req.stream_channel, [tok])
                if len(st["committed_tokens"]) >= req.max_tokens:
                    self._finish_request(
                        s,
                        req,
                        st["committed_tokens"],
                        "length",
                        logprobs_data=st.get("logprobs_acc"),
                    )
                    newly_finished.append(s)

        # -- MTP verify/commit round (greedy path, unchanged) --
        if greedy_slots:
            _round_t0 = time.perf_counter()
            any_lp_g = any(self.active[s]["req"].logprobs for s in greedy_slots)
            top_lp_g = (
                max(
                    (self.active[s]["req"].top_logprobs for s in greedy_slots),
                    default=0,
                )
                if any_lp_g
                else 0
            )
            decisions = self.runner.mtp_verify_and_commit_batch(
                greedy_slots,
                {s: self.active[s]["anchor"] for s in greedy_slots},
                {s: self.active[s]["drafts"] for s in greedy_slots},
                return_logprobs=any_lp_g,
                top_logprobs=top_lp_g,
            )
            _round_ms = (time.perf_counter() - _round_t0) * 1000

            for s in greedy_slots:
                st = self.active[s]
                req = st["req"]
                decision = decisions[s]
                new_tokens = decision["committed"]
                na = decision.get("num_accepted", 0)
                if req.logprobs and "logprobs" in decision:
                    st.setdefault("logprobs_acc", []).extend(decision["logprobs"])
                if 0 <= na < len(self.stats["mtp_acceptance_histogram"]):
                    self.stats["mtp_acceptance_histogram"][na] += 1

                finish_reason: str | None = None
                kept: list[int] = []
                for t in new_tokens:
                    if len(st["committed_tokens"]) + len(kept) >= req.max_tokens:
                        finish_reason = "length"
                        break
                    if t in self.eos_token_ids:
                        finish_reason = "stop"
                        break
                    kept.append(t)
                st["committed_tokens"].extend(kept)
                if kept:
                    st["last_progress_round"] = self.stats["rounds"]
                if kept and req.stream_channel is not None:
                    self._stream_put(req.stream_channel, kept)
                if finish_reason is None and len(st["committed_tokens"]) >= req.max_tokens:
                    finish_reason = "length"

                if finish_reason is None:
                    st["anchor"] = decision["next_anchor"]
                    st["drafts"] = decision["next_draft_tokens"]
                    # C3: advance grammar state for committed tokens
                    grammar = st.get("grammar")
                    if grammar is not None:
                        for _t in kept:
                            grammar.accept(_t)
                    tracer.decode_round(req.request_id, self.stats["rounds"], len(kept), _round_ms)
                    continue

                self._finish_request(
                    s,
                    req,
                    st["committed_tokens"],
                    finish_reason,
                    logprobs_data=st.get("logprobs_acc"),
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
