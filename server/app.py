"""OpenAI + Anthropic compatible HTTP server for BlackweLLM runtime.

Wraps ``server/engine.py`` (continuous-batching engine) with full
OpenAI ``/v1/chat/completions`` and Anthropic ``/v1/messages`` APIs.

Capabilities (B1/C1 采样全链路 + streaming + tool calling):

- ``POST /v1/chat/completions``, ``POST /v1/completions``,
  ``POST /v1/messages`` (Anthropic format).
- Streaming (SSE) and non-streaming responses.
- Full sampling: temperature, top_p, top_k, seed (``runtime/sampling.py``).
  ``temperature == 0`` selects greedy with MTP speculative verification.
- Tool calling via chat template (``convert_tools_to_chat_template``).
- Configurable capacity (default 4 slots, 256K context per slot).
- Prefix cache with session affinity for warm multi-turn.
- CUDA Graph accelerated decode.
- FP8 KV cache (2× capacity vs BF16).
- Prometheus metrics at ``/metrics``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from runtime.sampling import PersistentSeed, SamplingParams
from runtime.structured_output import ResponseFormat
from server import metrics
from server.engine import ServerEngine
from server.formats import anthropic as anthropic_format
from server.formats import convert_tools_to_chat_template
from server.formats import openai as openai_format
from server.formats import responses as responses_format
from server.formats.stream import StreamProcessor
from server.tracing import tracer

logger = logging.getLogger("qwen_sm120_server.app")

# uvicorn only configures its own loggers; without an explicit handler this
# logger's INFO records (e.g. the Anthropic debug capture below) are dropped
# silently. Attach a stderr handler so they reach the service log file.
logger.setLevel(logging.INFO)
if not logger.handlers:
    _stderr_handler = logging.StreamHandler()
    _stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_stderr_handler)
logger.propagate = False

# Verbose raw request/response capture for ALL endpoints (OpenAI + Anthropic).
# Default ON so real client traffic (e.g. Claude Desktop) is captured for
# debugging and regression fixtures; set QSR_DEBUG_REQUESTS=0 (or the legacy
# QSR_DEBUG_ANTHROPIC=0) to disable. Logs the raw request body, the parsed
# messages, the decoded prompt (exact model input), and the raw model output.
DEBUG_REQUESTS = (
    os.environ.get("QSR_DEBUG_REQUESTS", os.environ.get("QSR_DEBUG_ANTHROPIC", "1")) != "0"
)

DEFAULT_MAX_TOKENS = 16384

# CLI/launcher (``python -m server.app``) sets these via env vars before
# ``uvicorn.run`` triggers the lifespan startup below -- kept as module-
# level constants (not argparse-threaded into the FastAPI app object
# directly) since uvicorn's import-string app-loading convention
# (``uvicorn.run("server.app:app", ...)``) needs ``app`` importable with
# no constructor arguments.
# The production server is Laguna-only.  The archived Qwen tenant has no
# serving entry point, so launcher defaults state Laguna's actual geometry
# directly instead of retaining unreachable alternate defaults.
#
# Track A migration step 5 (docs/architecture.md §3.5.5): SERVER_MODEL_BACKEND
# used to be hardcoded here ("laguna") and fed straight into ServerEngine.
# It is now *resolved*, in lifespan() below, by runtime.model_registry --
# the registry's first real production consumer (previously it had only
# shadow-mode tests). SERVER_MODEL_PATH is the one thing that still has to be
# stated somewhere: the registry resolves a checkpoint's config.json into a
# backend, it does not invent which checkpoint to serve.
SERVER_MODEL_PATH = os.environ.get("QSR_SERVER_MODEL_PATH", "poolside/Laguna-S-2.1-NVFP4")

SERVER_CAPACITY = int(os.environ.get("QSR_SERVER_CAPACITY", "1"))
# Laguna default bumped 1->2: ServerEngine requires num_slots >= capacity +
# (capacity if enable_cudagraph else 0), and enable_cudagraph now defaults
# on for Laguna (see SERVER_ENABLE_CUDAGRAPH below) -- capacity=1 needs the
# extra slot for the CG capture's warmup writes.
SERVER_NUM_SLOTS = int(os.environ.get("QSR_SERVER_NUM_SLOTS", "2"))
SERVER_BLOCK_SIZE = int(os.environ.get("QSR_SERVER_BLOCK_SIZE", "64"))
# Laguna's SparkInfer attention uses 64-token pages.  The default below is
# currently 128K per slot.
# The KV cache pool size is now determined by GPU memory profiling (see
# server/engine.py _load_model → profile_kv_cache_blocks), NOT by the old
# fixed formula (num_slots + 1) * blocks_per_slot. blocks_per_slot is the
# per-slot MAXIMUM context ceiling; the actual pool is sized to fit the GPU.
# The E2E check sets its OWN smaller blocks_per_slot (its prompts are moderate),
# so it does not pay for the full long-context pool.
# Laguna default (2048 × 64 = 128K/slot) is conservative pending the SWA
# ring-buffer optimization above -- see notes/2026-07-23-laguna-server-
# integration-plan.md for the memory math.
SERVER_BLOCKS_PER_SLOT = int(os.environ.get("QSR_SERVER_BLOCKS_PER_SLOT", "2048"))
# Laguna default flipped 0->1 (2026-07-27): decode CUDA Graph is now wired
# into decode_batch_sampled (runtime/backends/laguna.py's
# _decode_cg_batch_eligible) and verified end-to-end over a real HTTP
# request (notes/2026-07-27-p1-http-e2e-and-thinking-strip-bug.md) --
# eager decode_batch_sampled is no longer the only path exercised in
# production. QSR_SERVER_ENABLE_CUDAGRAPH=0 / --no-cudagraph still rolls
# back to eager.
SERVER_ENABLE_CUDAGRAPH = os.environ.get("QSR_SERVER_ENABLE_CUDAGRAPH", "1") != "0"
# P4a (notes/prefix-cache-design.md sec 5-P4): the prefix-cache rollback
# spine, plumbed straight into ServerEngine(enable_prefix_cache=...). Default
# ON (this is THE product value -- warm prefix hits served across requests);
# `python -m server.app --no-prefix-cache` (or QSR_SERVER_ENABLE_PREFIX_CACHE=0)
# turns it off => byte-for-byte the old server.
#
# The default was "0" until 2026-08-02, contradicting every other signal
# around it: this comment, the existence of a `--no-prefix-cache` opt-out
# (an opt-out only makes sense against a default of ON), and the deployed
# launcher pinning `QSR_SERVER_ENABLE_PREFIX_CACHE:=1`. It was a fossil --
# `8f27f59` collapsed `"0" if _IS_LAGUNA else "1"` to `"0"` when the Qwen
# branches were removed, preserving Laguna's then-experimental default,
# and nobody flipped it once the prefix cache became the product. The
# launcher's pin had been patching over it at the deployment layer, so
# `python -m server.app` with no flags silently ran without the cache while
# `--no-prefix-cache` was a no-op. `test_prefix_cache_default_is_on` now
# pins this so it cannot drift back unnoticed.
SERVER_ENABLE_PREFIX_CACHE = os.environ.get("QSR_SERVER_ENABLE_PREFIX_CACHE", "1") != "0"
# P4b session affinity (notes/2026-07-20-p4b-session-affinity-plan.md): opt-in
# warm-slot retention. Default OFF => byte-for-byte P4a (without a session_id, or
# with the flag off, _finish_request does the unconditional reset_slot). Requires
# the prefix cache -- ServerEngine raises ValueError if affinity is on but prefix
# cache is off (warm-continue needs the persistent content-hash cache).
SERVER_ENABLE_SESSION_AFFINITY = os.environ.get("QSR_SERVER_ENABLE_SESSION_AFFINITY", "0") != "0"
SERVER_SESSION_TTL_S = float(os.environ.get("QSR_SERVER_SESSION_TTL_S", "30.0"))
# Laguna default: ``auto``. FP8 KV has not been validated for Laguna, so an
# explicit override remains required before it can be used in production.
SERVER_KV_CACHE_DTYPE = os.environ.get("QSR_SERVER_KV_CACHE_DTYPE", "auto")
SERVER_GPU_MEM_UTIL = float(os.environ.get("QSR_SERVER_GPU_MEM_UTIL", "0.85"))
SERVER_PRODUCTION = os.environ.get("QSR_SERVER_PRODUCTION", "1") != "0"
# DFlash speculative decoding (2026-07-27, notes/2026-07-27-dflash-server-
# integration.md): default OFF even for Laguna. Unlike SERVER_ENABLE_CUDAGRAPH
# above, this is a same-day integration of a capability that loads an extra
# draft model (real additional GPU memory) and hard-requires capacity=1
# (ServerEngine raises otherwise) -- opt-in via QSR_SERVER_ENABLE_DFLASH=1
# until it has run in production for a while, not flipped on by default yet.
SERVER_ENABLE_DFLASH = os.environ.get("QSR_SERVER_ENABLE_DFLASH", "0") != "0"
# MTP speculative decoding (2026-08-03, Track B / B3): qwen36's own draft
# head, the sibling of SERVER_ENABLE_DFLASH above (Laguna's). Default OFF
# for the same reason: this landing is the first time it is reachable from
# the serving path at all, so main's shipping behaviour must stay unchanged
# (plain per-token decode) until MTP has run through the same acceptance/
# correctness gate DFlash did before its own default flip. See
# runtime/backends/qwen36_mtp.py for the round driver and the (token,
# hidden) pairing fix this landing carries.
SERVER_ENABLE_MTP = os.environ.get("QSR_SERVER_ENABLE_MTP", "0") != "0"
SERVER_MTP_K = int(os.environ.get("QSR_SERVER_MTP_K", "4"))
# Per-round KV resync (runtime/backends/qwen36_mtp.py's Qwen36MTPEngine
# docstring): independently toggleable from MTP itself so it can be A/B
# measured on real hardware separately from the pairing fix. Only consulted
# when SERVER_ENABLE_MTP is set; "unset" (None) lets Qwen36MTPEngine fall
# back to its own QSR_SERVER_MTP_RESYNC-driven default (also off).
_mtp_resync_env = os.environ.get("QSR_SERVER_MTP_RESYNC")
SERVER_MTP_RESYNC = None if _mtp_resync_env is None else _mtp_resync_env != "0"
SERVER_REQUEST_TIMEOUT_S = float(
    os.environ.get("QSR_SERVER_REQUEST_TIMEOUT_S", "600"))
# T0-3/E4 (docs/roadmap.md §7 D1): reasoning/thinking contract. "expose"
# (default) surfaces a <think> block as OpenAI message.reasoning_content /
# delta.reasoning_content, and Anthropic's non-standard top-level
# reasoning_content field / reasoning_content_delta SSE event (see
# server/formats/anthropic.py's build_response docstring for why NOT the
# spec `thinking` content block). "strip" discards it (bandwidth-saving
# opt-out); content/text NEVER carries reasoning either way.
SERVER_REASONING_MODE = os.environ.get("QSR_REASONING_MODE", "expose")
if SERVER_REASONING_MODE not in ("expose", "strip"):
    raise RuntimeError(f"QSR_REASONING_MODE={SERVER_REASONING_MODE!r} must be 'expose' or 'strip'")
# Laguna's chat template DOES inject <think>. This was recorded the other way
# round until a live server run on 2026-08-01 showed the prompt ending in:
#
#   ...</system>\n<user>...</user>\n<assistant><think>
#
# and every single generated completion starting with a bare closing tag:
#
#   RAW OUTPUT (10 tokens, finish=stop): </think>Hello! How can I help you today?
#
# So generation begins *inside* a think block. With the flag off, no <think>
# sits at position 0 of the generated text, the anchored span rule finds no
# reasoning segment, and the orphan </think> is served as the first characters
# of message.content. Chat and messages endpoints only -- /v1/completions
# applies no chat template and still returns generated text verbatim, which is
# the distinction the 2026-07-27 incident (notes/2026-07-27-p1-http-e2e-and-
# thinking-strip-bug.md) turned on: prepending unconditionally, on the endpoint
# that has no template, ate whole responses.
SERVER_THINKING_CAPABLE = os.environ.get("QSR_THINKING_CAPABLE", "1") != "0"

# Selects which server/formats/tool_parsers/ shape to decode tool calls
# with -- mirrors vLLM's --tool-call-parser NAME. Default matches this
# project's currently (and so far only) production model, poolside/
# Laguna-S-2.1-NVFP4. A model with a differently-shaped tool-call output
# needs its own ToolCallParser registered there, then selected here.
SERVER_TOOL_CALL_PARSER = os.environ.get("QSR_TOOL_CALL_PARSER", "poolside_v1")

engine: ServerEngine | None = None


async def _tokenize_chat(engine_ref, messages, tools=None, chat_template_kwargs=None):
    """Run apply_chat_template in a thread to avoid blocking the event loop.

    ``chat_template_kwargs`` is forwarded verbatim to the Jinja template, so the
    official Qwen3.6 ``{"enable_thinking": False}`` toggle (and any other template
    option) is honored exactly as in stock vLLM. Without this the template always
    defaults to thinking mode and the toggle sent by clients is silently ignored.
    """
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        engine_ref.tok.apply_chat_template,
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **(chat_template_kwargs or {}),
    )
    return await loop.run_in_executor(None, fn)


async def _tokenize_encode(engine_ref, text):
    """Run tokenizer encode in a thread.

    Laguna requires BOS (``add_special_tokens=True``).
    """
    loop = asyncio.get_running_loop()
    fn = functools.partial(engine_ref.tok.encode, text, add_special_tokens=True)
    return await loop.run_in_executor(None, fn)


async def _tokenize_decode(engine_ref, token_ids):
    """Run tokenizer decode in a thread."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(engine_ref.tok.decode, token_ids, skip_special_tokens=True)
    return await loop.run_in_executor(None, fn)


def _format_logprobs_openai(
    engine_ref,
    raw_logprobs: list[dict] | None,
) -> dict | None:
    """Format raw logprobs data into OpenAI API logprobs structure."""
    if not raw_logprobs:
        return None
    tok = engine_ref.tok
    content_list = []
    for entry in raw_logprobs:
        token_str = tok.decode([entry["token_id"]])
        item: dict = {
            "token": token_str,
            "logprob": entry["logprob"],
            "bytes": list(token_str.encode("utf-8")),
        }
        top = []
        for alt in entry.get("top_logprobs", []):
            alt_str = tok.decode([alt["token_id"]])
            top.append(
                {
                    "token": alt_str,
                    "logprob": alt["logprob"],
                    "bytes": list(alt_str.encode("utf-8")),
                }
            )
        item["top_logprobs"] = top
        content_list.append(item)
    return {"content": content_list}


def _endpoint_from_path(path: str) -> str:
    """Map a request path to a low-cardinality metrics endpoint label."""
    if path.startswith("/v1/chat/completions"):
        return "chat"
    if path.startswith("/v1/completions"):
        return "completions"
    if path.startswith("/v1/messages"):
        return "messages"
    return "other"


async def _debug_log_input(tag: str, body: dict, parsed_messages, prompt_ids: list[int]) -> None:
    """Capture the full raw request, the parsed messages, and the decoded
    prompt (the exact input the model receives). Gated on DEBUG_REQUESTS."""
    if not DEBUG_REQUESTS:
        return
    try:
        _raw = json.dumps(body, ensure_ascii=False, default=str)
        logger.info("%s RAW REQUEST (%d bytes): %s", tag, len(_raw), _raw)
        logger.info(
            "%s PARSED MESSAGES: %s",
            tag,
            json.dumps(parsed_messages, ensure_ascii=False, default=str),
        )
        _loop = asyncio.get_running_loop()
        _prompt_text = await _loop.run_in_executor(
            None,
            functools.partial(engine.tok.decode, prompt_ids, skip_special_tokens=False),
        )
        logger.info(
            "%s DECODED PROMPT (%d ids, %d chars): %s",
            tag,
            len(prompt_ids),
            len(_prompt_text),
            _prompt_text,
        )
    except Exception:
        logger.exception("%s debug input capture failed", tag)


def _debug_log_output(
    tag: str, raw_text: str, visible_text: str, finish_reason: str, gen_tokens: int
) -> None:
    """Capture the raw model output and the visible (thinking-stripped) output
    for a NON-streaming response. Gated on DEBUG_REQUESTS."""
    if not DEBUG_REQUESTS:
        return
    try:
        logger.info(
            "%s RAW OUTPUT (%d tokens, finish=%s, %d chars): %s",
            tag,
            gen_tokens,
            finish_reason,
            len(raw_text),
            raw_text,
        )
        logger.info("%s VISIBLE OUTPUT (%d chars): %s", tag, len(visible_text), visible_text)
    except Exception:
        logger.exception("%s debug output capture failed", tag)


async def _debug_log_stream_output(
    tag: str, proc, visible_text: str, tool_calls, finish_reason: str
) -> None:
    """Capture the raw + visible model output for a STREAMING response by
    decoding the full committed token list. Gated on DEBUG_REQUESTS."""
    if not DEBUG_REQUESTS:
        return
    try:
        gen_tokens = len(proc.all_ids)
        _loop = asyncio.get_running_loop()
        _raw = await _loop.run_in_executor(
            None,
            functools.partial(engine.tok.decode, proc.all_ids, skip_special_tokens=False),
        )
        logger.info(
            "%s RAW OUTPUT (%d tokens, finish=%s, %d chars): %s",
            tag,
            gen_tokens,
            finish_reason,
            len(_raw),
            _raw,
        )
        logger.info("%s VISIBLE OUTPUT (%d chars): %s", tag, len(visible_text), visible_text)
        if tool_calls:
            logger.info(
                "%s TOOL CALLS: %s",
                tag,
                json.dumps(tool_calls, ensure_ascii=False, default=str),
            )
    except Exception:
        logger.exception("%s debug output capture failed", tag)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Validated and applied before the (slow) model load, so a typo in
    # QSR_TOOL_CALL_PARSER/--tool-call-parser fails in <1s instead of after
    # minutes of loading weights.
    from server.formats.tool_parsers import set_active_parser

    set_active_parser(SERVER_TOOL_CALL_PARSER)
    logger.info("tool_call_parser=%s", SERVER_TOOL_CALL_PARSER)

    # Track A migration step 5 (docs/architecture.md §3.5.5): the backend
    # name used to be the hardcoded constant SERVER_MODEL_BACKEND. It is now
    # resolved from the checkpoint's own config.json -- registry's first
    # real production consumer, not just shadow-mode tests (see
    # runtime/model_registry.py). Resolution reads only config.json (fast,
    # no weights), so this still runs before the slow model load below, same
    # as the tool_call_parser check above it.
    from runtime.laguna_config import _resolve_laguna_model_dir
    from runtime.model_registry import resolve_checkpoint

    resolution = resolve_checkpoint(_resolve_laguna_model_dir(SERVER_MODEL_PATH))
    logger.info(
        "loading model=%s backend=%s (registry-resolved; this can take a while: "
        "model load + KV cache alloc)...",
        SERVER_MODEL_PATH,
        resolution.backend,
    )
    engine = ServerEngine(
        model=SERVER_MODEL_PATH,
        backend=resolution.backend,
        # A3 step 7-g (docs/a3-cache-coordinator-design.md §7 row 7-g):
        # `resolution.spec` used to be read only for `.backend` above and
        # discarded -- it also carries `needs_two_cache_families`, which
        # `ServerEngine.slot_resources` (a `SlotResourceManager`) needs to
        # decide whether a second cache-family allocator is required. For
        # every checkpoint this runtime serves today that is `False`, so
        # this plumbs the already-computed fact through rather than
        # re-deriving a fallback inside ServerEngine.
        architecture_spec=resolution.spec,
        capacity=SERVER_CAPACITY,
        num_slots=SERVER_NUM_SLOTS,
        block_size=SERVER_BLOCK_SIZE,
        blocks_per_slot=SERVER_BLOCKS_PER_SLOT,
        kv_cache_dtype=SERVER_KV_CACHE_DTYPE,
        enable_cudagraph=SERVER_ENABLE_CUDAGRAPH,
        enable_prefix_cache=SERVER_ENABLE_PREFIX_CACHE,
        enable_session_affinity=SERVER_ENABLE_SESSION_AFFINITY,
        session_ttl_s=SERVER_SESSION_TTL_S,
        enable_dflash=SERVER_ENABLE_DFLASH,
        enable_mtp=SERVER_ENABLE_MTP,
        mtp_num_speculative_tokens=SERVER_MTP_K,
        mtp_resync=SERVER_MTP_RESYNC,
        request_timeout_s=SERVER_REQUEST_TIMEOUT_S,
        gpu_memory_utilization=SERVER_GPU_MEM_UTIL,
        production=SERVER_PRODUCTION,
    )
    engine.start()
    logger.info(
        "engine ready: backend=%s model=%s capacity=%d num_slots=%d capacity_tokens_per_slot=%d "
        "cudagraph=%s prefix_cache=%s session_affinity=%s ttl=%.1fs dflash=%s "
        "mtp=%s(K=%d,resync=%s)",
        engine.backend_name,
        engine.MODEL,
        engine.capacity,
        engine.num_slots,
        engine.capacity_tokens_per_slot,
        SERVER_ENABLE_CUDAGRAPH,
        SERVER_ENABLE_PREFIX_CACHE,
        SERVER_ENABLE_SESSION_AFFINITY,
        SERVER_SESSION_TTL_S,
        SERVER_ENABLE_DFLASH,
        SERVER_ENABLE_MTP,
        SERVER_MTP_K,
        SERVER_MTP_RESYNC,
    )
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(title="qwen-sm120-runtime server", lifespan=lifespan)


@app.head("/")
@app.get("/")
async def root():
    return {"status": "ok", "service": "blackwellm"}


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    import time as _time

    t0 = _time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (_time.perf_counter() - t0) * 1000
    if elapsed_ms > 100:
        logger.info(
            "SLOW %s %s -> %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


# -- schemas (loose OpenAI-compatible subset -- see module docstring for
# the explicit, intentional deviations: greedy-only, non-streaming, plus
# a debug-only extra field). --


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    n: int | None = None
    stream: bool | None = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    session_id: str | None = None
    response_format: dict | None = None
    stop: str | list[str] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = None
    # Forwarded to the chat template (e.g. {"enable_thinking": False} for
    # non-thinking mode). Mirrors vLLM's chat_template_kwargs request field.
    chat_template_kwargs: dict | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    n: int | None = None
    stream: bool | None = False
    response_format: dict | None = None
    stop: str | list[str] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = None
    # P4b session affinity (opt-in) -- see ChatCompletionRequest.session_id.
    session_id: str | None = None


def _invalid_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=400, detail={"error": {"message": message, "type": "invalid_request_error"}}
    )


def _build_sampling_params(
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    n: int | None = None,
) -> SamplingParams:
    """Validate and build SamplingParams from API request fields.

    ``temperature == 0`` (or ``None``) selects greedy decode. Both greedy
    and ``temperature > 0`` (true sampling) get MTP speculative
    verification when the backend has DFlash enabled -- E2-b
    (docs/e2e-and-quality-plan.md §2.2) closed the gap where sampling used
    to silently fall back to non-speculative autoregressive decode
    (``DFlashEngine.dflash_round`` resolves accept/reject via rejection
    sampling instead of an argmax comparison for non-greedy requests; see
    ``server/engine.py::classify_decode_slots``). A backend without DFlash
    enabled, or a grammar-constrained request, still always uses the plain
    autoregressive path regardless of temperature.
    """
    if n is not None and n != 1:
        raise _invalid_request(
            f"n={n!r} is not supported: only a single completion (n=1) per request."
        )
    temp = temperature if temperature is not None else 0.0
    if temp < 0:
        raise _invalid_request(f"temperature must be >= 0, got {temp}")
    resolved_top_p = top_p if top_p is not None else 1.0
    if not (0.0 < resolved_top_p <= 1.0):
        raise _invalid_request(f"top_p must be in (0, 1], got {resolved_top_p}")
    resolved_top_k = top_k if top_k is not None else 0
    if resolved_top_k < 0:
        raise _invalid_request(f"top_k must be >= 0, got {resolved_top_k}")
    return SamplingParams(
        temperature=temp,
        top_p=resolved_top_p,
        top_k=resolved_top_k,
        # N3: wrap in PersistentSeed so make_generator() advances ONE
        # generator across this request's decode rounds instead of
        # reseeding an identical initial RNG state at every token -- see
        # PersistentSeed's docstring (runtime/sampling.py). A fresh
        # instance per request/per call means two different requests that
        # happen to pass the same integer seed never share RNG state.
        seed=PersistentSeed(seed) if seed is not None else None,
    )


def _validate_and_resolve_max_tokens(max_tokens: int | None) -> int:
    resolved = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if resolved <= 0:
        raise _invalid_request(f"max_tokens={max_tokens!r} must be >= 1.")
    return resolved


def _reject_unsupported_response_format(response_format: dict | None) -> None:
    """N1: structured output (``json_object`` / ``json_schema``) has no
    working enforcement path in this runtime -- see
    docs/api-layer-design.md §7.1. The only reachable masking hook
    (``runtime/sampling.py::sample_from_logits``) is never reached by:

    - the prefill anchor token (the FIRST token of every request is a raw
      unconstrained argmax inside ``runtime/backends/laguna.py``'s
      ``prefill_chunked_begin``/``_forward``, with no ``SamplingParams``
      involved at all);
    - the CUDA-Graph decode replay path (greedy argmax is baked into the
      captured graph itself);
    - the plain eager ``if params.is_greedy: argmax(...)`` shortcut in
      ``decode_batch_sampled`` (bypasses ``sample_from_logits`` entirely).

    Since this runtime's default temperature is 0.0 (greedy) when a client
    doesn't set one explicitly, EVERY one of those unreachable paths is
    exactly the path a typical "give me guaranteed JSON" request (no
    explicit temperature) takes, for every token including the first.
    Wiring only the narrow reachable slice (temperature > 0, decode tokens
    2+) would silently leave the common/default case completely
    unconstrained while looking wired-in -- the same silent-failure shape
    this check exists to eliminate, just relocated. Reject loudly instead.
    """
    fmt = ResponseFormat.from_api(response_format)
    if fmt.is_constrained:
        raise _invalid_request(
            f"response_format type={fmt.type!r} is not supported: this runtime "
            "does not enforce structured output (JSON mode / json_schema) during "
            "generation -- passing it would silently return unconstrained plain "
            "text, not a JSON guarantee. Omit response_format and validate/parse "
            "JSON on the client side instead."
        )


def _normalize_stop(
    stop: str | list[str] | None, *, max_count: int | None = None
) -> list[str] | None:
    """Normalize OpenAI's ``stop`` (string or list of strings) / Anthropic's
    ``stop_sequences`` (list of strings) into one shared shape.

    Empty strings are dropped (an empty stop sequence trivially "matches"
    at position 0 of any output and has no sensible use); an all-empty
    result normalizes to ``None`` (no stop sequences configured) rather
    than an empty list, so callers can treat ``None``/``[]`` as one case.
    ``max_count`` enforces OpenAI's documented limit of 4; Anthropic's
    ``stop_sequences`` has no such documented cap, so callers for that
    protocol pass ``max_count=None``.
    """
    if stop is None:
        return None
    seqs = [stop] if isinstance(stop, str) else list(stop)
    seqs = [s for s in seqs if s]
    if not seqs:
        return None
    if max_count is not None and len(seqs) > max_count:
        raise _invalid_request(f"stop supports at most {max_count} sequences, got {len(seqs)}")
    return seqs


def _validate_capacity(prompt_ids: list[int], max_tokens: int) -> None:
    # metrics.record_error is NOT called here: _http_exception_handler
    # records it once, uniformly, for every raised HTTPException -- an
    # explicit call here would double-count.
    assert engine is not None
    if not engine.capacity_ok(len(prompt_ids), max_tokens):
        raise _invalid_request(
            f"prompt_tokens({len(prompt_ids)}) + max_tokens({max_tokens}) = "
            f"{len(prompt_ids) + max_tokens} exceeds this runtime's per-slot capacity of "
            f"{engine.capacity_tokens_per_slot} tokens (blocks_per_slot * block_size). "
            "Reduce the prompt length or max_tokens and retry."
        )


def _protocol_error_body(path: str, err: dict) -> dict:
    """Shape a ``{"message": ..., "type": ...}`` error for the protocol
    the failing request actually used."""
    if path.startswith("/v1/messages"):
        return {"type": "error", "error": err}
    return {"error": err}


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    """E1 (docs/roadmap.md Track E, error-code semantics): FastAPI's
    default HTTPException handling wraps whatever ``detail`` a handler
    raised in an extra ``{"detail": ...}`` envelope -- verified empirically
    (see docs/api-layer-design.md): a 400 raised via ``_invalid_request()``
    actually reached the client as
    ``{"detail": {"error": {"message": ..., "type": ...}}}``, matching
    NEITHER OpenAI's ``{"error": {...}}`` NOR Anthropic's
    ``{"type": "error", "error": {...}}``. Unwrap it and reshape for
    whichever protocol was actually called.
    """
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        err = detail["error"]
    else:
        err = {"message": str(detail), "type": "invalid_request_error"}
    metrics.record_error(_endpoint_from_path(request.url.path), exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content=_protocol_error_body(request.url.path, err),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """Same fix as ``_http_exception_handler``, for the OTHER shape FastAPI
    produces on its own: a request body that fails pydantic validation
    (e.g. a malformed/missing field) gets FastAPI's default 422
    ``{"detail": [{"loc": ..., "msg": ..., "type": ...}, ...]}`` -- also
    matching neither protocol. This is a common real client mistake (typo'd
    field, wrong type), not an edge case.
    """
    messages = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    metrics.record_error(_endpoint_from_path(request.url.path), 422)
    return JSONResponse(
        status_code=422,
        content=_protocol_error_body(
            request.url.path, {"message": messages, "type": "invalid_request_error"}
        ),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    # A defensive net so an unexpected runtime error (e.g. the engine's own
    # error-recovery path in server/engine.py's _loop) surfaces as a clean
    # 500 JSON body instead of an unhandled-exception stack trace / crash.
    logger.exception("unhandled exception serving %s", request.url.path)
    metrics.record_error(_endpoint_from_path(request.url.path), 500)
    err = {"message": str(exc), "type": "internal_error"}
    return JSONResponse(status_code=500, content=_protocol_error_body(request.url.path, err))


@app.get("/health")
async def health():
    assert engine is not None
    return {
        "status": "ok",
        "capacity": engine.capacity,
        "free_slots": len(engine.free_slots),
        "active": len(engine.active),
        "waiting": len(engine.waiting),
    }


@app.get("/debug/stats")
async def debug_stats():
    """Non-standard, this-project-only endpoint: exposes the engine's own
    round/admission counters so the E2E validation script (and any curious
    human) can directly confirm real multi-request batching happened
    (``admission_batch_sizes``/``round_batch_sizes`` containing entries
    > 1), rather than inferring it indirectly from timing alone."""
    assert engine is not None
    snapshot = _backend_snapshot(engine.runner)
    if snapshot is not None:
        engine.stats["_prefix_cache_dbg"] = {
            f"slot_{p.slot}": {
                "cached_len": p.cached_tokens,
                "kv_len": p.cached_kv_len,
                "head": list(p.head) if p.head else None,
            }
            for p in snapshot.prefix
        }
        # B3 step 0 (docs/implementation-plan.md §7.3 C7-2): CUDA Graph
        # capture success/failure, made observable here rather than only via
        # /metrics' Prometheus gauges -- curl-able without a Prometheus text
        # parser, and this is the exact question B3's throughput numbers are
        # unattributable without answering ("did the decode CUDA Graph
        # actually capture in THIS server process, or is decode silently
        # running eager"). () when the backend has not attempted any capture
        # yet (matches BackendSnapshot.dflash_cg_status's own "never missing"
        # contract), not omitted from the response, so a caller can
        # distinguish "no capture attempted" from "field doesn't exist".
        engine.stats["_cuda_graph_dbg"] = dict(snapshot.dflash_cg_status)
    # 2026-08-02 CG audit (docs/implementation-plan.md §7.3 C7-2, "activity
    # confirmation"): `_cuda_graph_dbg` above proves capture succeeded, not
    # that a real decode round actually replayed the captured graph instead
    # of silently falling back to eager every time. `Qwen36Backend.stats`
    # already counts `decode_graph_replays` (runtime/backends/qwen36.py);
    # surfacing it here turns "did CUDA Graph actually engage for real
    # traffic" into the same curl-able signal `_cuda_graph_dbg` is, without
    # inventing a new counter. `getattr` because this is backend-specific --
    # `LagunaBackend` has no `.stats` attribute at all (its decode-CG replay
    # path, `runtime/backends/laguna_cuda_graph.py`'s `replay()`, carries no
    # counter to expose) and must not be assumed present.
    backend_stats = getattr(engine.runner, "stats", None)
    if backend_stats is not None:
        engine.stats["_backend_stats_dbg"] = dict(backend_stats)
    return engine.stats


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    assert engine is not None
    sampling_params = _build_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        n=req.n,
    )
    _reject_unsupported_response_format(req.response_format)
    stop_sequences = _normalize_stop(req.stop, max_count=4)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    t0 = time.perf_counter()

    # Parse messages through the format layer (handles string | array content)
    chat_messages = openai_format.parse_chat_messages(req.model_dump())

    # Convert tools for the chat template
    tools = convert_tools_to_chat_template(req.tools)

    prompt_ids = await _tokenize_chat(
        engine,
        chat_messages,
        tools=tools,
        chat_template_kwargs=req.chat_template_kwargs,
    )
    await _debug_log_input(
        "OPENAI /v1/chat/completions", req.model_dump(), chat_messages, prompt_ids
    )
    _validate_capacity(prompt_ids, max_tokens)

    model_name = req.model or engine.MODEL

    if req.stream:
        import json as _json

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def _sse():
            proc = StreamProcessor(engine.tok, thinking_capable=SERVER_THINKING_CAPABLE)
            final_result = None
            first_token_t = None
            # First chunk: role announcement (matches vLLM format)
            first_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json.dumps(first_chunk)}\n\n"
            _cancel_ref: list[str | None] = [None]
            async for item in engine.submit_stream(
                prompt_ids,
                max_tokens,
                session_id=req.session_id,
                sampling_params=sampling_params,
                cancel_ref=_cancel_ref,
                stop_sequences=stop_sequences,
                logprobs=bool(req.logprobs),
                top_logprobs=req.top_logprobs or 0,
            ):
                if await request.is_disconnected():
                    if _cancel_ref[0]:
                        engine.cancel(_cancel_ref[0])
                    return
                if isinstance(item, dict):
                    final_result = item
                    break
                proc.add_tokens(item)
                if first_token_t is None and item:
                    first_token_t = time.perf_counter()
                if SERVER_REASONING_MODE == "expose":
                    for rdelta in proc.drain_thinking():
                        rchunk = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_content": rdelta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(rchunk)}\n\n"
                for delta in proc.drain_content():
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {_json.dumps(chunk)}\n\n"
                # C4: stream tool call deltas incrementally
                for td in proc.drain_tool_deltas():
                    if td["type"] == "name":
                        tc_chunk = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": td["index"],
                                                "id": td["id"],
                                                "type": "function",
                                                "function": {"name": td["name"]},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(tc_chunk)}\n\n"
                    elif td["type"] == "arguments_delta":
                        tc_chunk = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": td["index"],
                                                "function": {"arguments": td["delta"]},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(tc_chunk)}\n\n"
            finish = final_result["finish_reason"] if final_result else "stop"
            visible_text, tool_calls = proc.finalize()
            if tool_calls:
                finish = "tool_calls"
            _lp_final = None
            if req.logprobs and final_result:
                _lp_final = _format_logprobs_openai(
                    engine,
                    final_result.get("logprobs"),
                )
            done = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish, "logprobs": _lp_final}
                ],
            }
            yield f"data: {_json.dumps(done)}\n\n"
            metrics.record_request(
                "chat",
                len(prompt_ids),
                len(proc.all_ids),
                finish,
                time.perf_counter() - t0,
                (first_token_t - t0) if first_token_t is not None else None,
            )
            await _debug_log_stream_output(
                "OPENAI /v1/chat/completions", proc, visible_text, tool_calls, finish
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    # Non-streaming path
    result = await engine.submit(
        prompt_ids,
        max_tokens,
        session_id=req.session_id,
        sampling_params=sampling_params,
        stop_sequences=stop_sequences,
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs or 0,
    )
    raw_text = await _tokenize_decode(engine, result["committed_token_ids"])
    # Same state machine as the streaming path (server/formats/stream.py) --
    # not a second, independently-written parser for the non-streaming case.
    proc = StreamProcessor(engine.tok, thinking_capable=SERVER_THINKING_CAPABLE)
    proc.add_tokens(result["committed_token_ids"])
    text = proc.content_text()
    reasoning_content = proc.reasoning_content() if SERVER_REASONING_MODE == "expose" else None
    metrics.record_request(
        "chat",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
    )
    _debug_log_output(
        "OPENAI /v1/chat/completions",
        raw_text,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
    resp = openai_format.build_response(
        model=model_name,
        text=text,
        finish_reason=result["finish_reason"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        committed_token_ids=result["committed_token_ids"],
        prompt_token_ids=list(prompt_ids),
        reasoning_content=reasoning_content,
    )
    if req.logprobs:
        resp["choices"][0]["logprobs"] = _format_logprobs_openai(
            engine,
            result.get("logprobs"),
        )
    return resp


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request):
    assert engine is not None
    sampling_params = _build_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        n=req.n,
    )
    _reject_unsupported_response_format(req.response_format)
    stop_sequences = _normalize_stop(req.stop, max_count=4)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    t0 = time.perf_counter()
    prompt_ids = await _tokenize_encode(engine, req.prompt)
    await _debug_log_input("OPENAI /v1/completions", req.model_dump(), req.prompt, prompt_ids)
    _validate_capacity(prompt_ids, max_tokens)

    result = await engine.submit(
        prompt_ids,
        max_tokens,
        session_id=req.session_id,
        sampling_params=sampling_params,
        stop_sequences=stop_sequences,
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs or 0,
    )
    _raw_comp = await _tokenize_decode(engine, result["committed_token_ids"])
    # Legacy text-completions has no chat-message/reasoning_content concept
    # to route a <think> block into (OpenAI's real /v1/completions has no
    # such field either) and no chat template is applied here at all, so
    # there is nothing to split -- return the generated text verbatim
    # (replacement-char cleanup only). This endpoint is the exact site of
    # the original P1 empty-output bug (notes/2026-07-27-p1-http-e2e-and-
    # thinking-strip-bug.md): unconditionally wrapping raw completion output
    # in a synthetic <think> prefix before stripping ate the entire response.
    text = _raw_comp.replace("�", "").strip()
    metrics.record_request(
        "completions",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
        ttft_seconds=result.get("prefill_elapsed_s"),
    )
    _debug_log_output(
        "OPENAI /v1/completions",
        _raw_comp,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or engine.MODEL,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": result["finish_reason"],
                "logprobs": (
                    _format_logprobs_openai(engine, result.get("logprobs"))
                    if req.logprobs
                    else None
                ),
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "debug_committed_token_ids": result["committed_token_ids"],
        "debug_prompt_token_ids": list(prompt_ids),
    }


def _run_startup_preflight() -> None:
    """Validate the environment before any weights are loaded, and abort on a
    fatal mismatch (roadmap T0-3 / D3).

    ``runtime.preflight`` deliberately never prints or exits -- it returns a
    structured report and leaves presentation and policy to its caller. This
    is that caller: it renders one line per check, blocks on fatal failures,
    and lets warnings through with their remediation text.

    The checkpoint checks want a local directory, but ``SERVER_MODEL_PATH``
    is a HuggingFace repo id. ``_resolve_laguna_model_dir`` is the resolver
    the loader itself uses (offline-only, no network fetch); importing the
    private name is deliberate -- duplicating two lines of resolution logic
    here would be free to drift away from what actually gets loaded.
    """
    import sys

    from runtime.laguna_config import _resolve_laguna_model_dir
    from runtime.preflight import run_preflight

    try:
        checkpoint_dir = _resolve_laguna_model_dir(SERVER_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - any resolution failure is fatal here
        print(
            f"preflight: cannot resolve a local checkpoint for {SERVER_MODEL_PATH!r}: {exc}\n"
            f"           Download it first, or point QSR_SERVER_MODEL_PATH at a local path.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    report = run_preflight(checkpoint_dir)
    for check in report.checks:
        mark = "ok  " if check.passed else ("FAIL" if check.severity == "fatal" else "warn")
        print(f"preflight [{mark}] {check.name}: {check.actual}", file=sys.stderr)
    for check in report.warnings:
        if check.remediation:
            print(f"preflight        -> {check.remediation}", file=sys.stderr)
    if report.ok:
        return
    print("preflight: refusing to start.", file=sys.stderr)
    for check in report.fatal_failures:
        print(
            f"  {check.name}: expected {check.expected}, got {check.actual}",
            file=sys.stderr,
        )
        if check.remediation:
            print(f"    -> {check.remediation}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capacity", type=int, default=SERVER_CAPACITY)
    parser.add_argument("--num-slots", type=int, default=SERVER_NUM_SLOTS)
    parser.add_argument("--blocks-per-slot", type=int, default=SERVER_BLOCKS_PER_SLOT)
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Skip the startup environment checks (GPU architecture, CUDA, torch, "
            "SparkInfer, checkpoint). Escape hatch for deliberately unusual setups; "
            "a fatal check normally means the server would fail later and less clearly."
        ),
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable the persistent prefix cache (rollback to the pre-P4a server).",
    )
    parser.add_argument(
        "--session-affinity",
        action="store_true",
        help="Enable opt-in session-affinity warm-slot retention (P4b). Requires the prefix cache.",
    )
    parser.add_argument(
        "--session-ttl-s",
        type=float,
        default=SERVER_SESSION_TTL_S,
        help="Warm-slot retention TTL in seconds for session affinity (P4b). Default 30.0.",
    )
    parser.add_argument(
        "--dflash",
        action="store_true",
        help=(
            "Enable DFlash speculative decoding (Laguna backend only). Any "
            "--capacity is supported; DFlash's CUDA Graphs replay once per "
            "active slot per round (not batched), so throughput scales "
            "sub-linearly with capacity -- see notes/2026-07-27-dflash-"
            "multi-slot-concurrency.md."
        ),
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        help=(
            "Enable MTP speculative decoding (qwen36 backend only). Sibling "
            "of --dflash for Qwen3.6's own draft head -- see "
            "runtime/backends/qwen36_mtp.py."
        ),
    )
    parser.add_argument(
        "--mtp-k",
        type=int,
        default=SERVER_MTP_K,
        help="MTP draft tokens per round (only used with --mtp). Default 4.",
    )
    parser.add_argument(
        "--mtp-resync",
        action="store_true",
        help=(
            "Enable per-round MTP KV resync (only used with --mtp) -- see "
            "runtime/backends/qwen36_mtp.py's Qwen36MTPEngine docstring. "
            "Default off, independently A/B-measurable from --mtp itself."
        ),
    )
    from server.formats.tool_parsers import available_parsers

    parser.add_argument(
        "--tool-call-parser",
        choices=available_parsers(),
        default=SERVER_TOOL_CALL_PARSER,
        help=(
            "Tool-call output shape to decode (mirrors vLLM's "
            "--tool-call-parser). One per model family -- see "
            "server/formats/tool_parsers/. Default matches the currently "
            "loaded model."
        ),
    )
    args = parser.parse_args()

    # P4b: refuse --session-affinity together with --no-prefix-cache -- a clean
    # startup error, not a runtime crash (warm-continue needs the persistent
    # content-hash cache; ServerEngine.__init__ raises the same way as a backstop).
    if args.session_affinity and args.no_prefix_cache:
        parser.error(
            "--session-affinity requires the prefix cache (cannot combine with --no-prefix-cache)"
        )

    # N8 (docs/architecture.md §3.5.6): fail before setting any env var or
    # touching uvicorn, not after minutes of model loading. ServerEngine.
    # __init__ enforces the identical check as a backstop for callers that
    # construct it directly (tests, embedding) rather than through this CLI.
    if args.session_affinity:
        from runtime.backends.laguna import LagunaBackend

        if not LagunaBackend.capabilities.fget(None).warm_continue:
            parser.error(
                "--session-affinity requires a backend with warm_continue capability; "
                "the laguna backend does not support it yet "
                "(see docs/architecture.md §3.5.6, N8)"
            )

    os.environ["QSR_SERVER_CAPACITY"] = str(args.capacity)
    os.environ["QSR_SERVER_NUM_SLOTS"] = str(args.num_slots)
    os.environ["QSR_SERVER_BLOCKS_PER_SLOT"] = str(args.blocks_per_slot)
    if args.no_cudagraph:
        os.environ["QSR_SERVER_ENABLE_CUDAGRAPH"] = "0"
    if args.no_prefix_cache:
        os.environ["QSR_SERVER_ENABLE_PREFIX_CACHE"] = "0"
    if args.session_affinity:
        os.environ["QSR_SERVER_ENABLE_SESSION_AFFINITY"] = "1"
    os.environ["QSR_SERVER_SESSION_TTL_S"] = str(args.session_ttl_s)
    if args.dflash:
        os.environ["QSR_SERVER_ENABLE_DFLASH"] = "1"
    if args.mtp:
        os.environ["QSR_SERVER_ENABLE_MTP"] = "1"
    os.environ["QSR_SERVER_MTP_K"] = str(args.mtp_k)
    if args.mtp_resync:
        os.environ["QSR_SERVER_MTP_RESYNC"] = "1"
    os.environ["QSR_TOOL_CALL_PARSER"] = args.tool_call_parser

    # Runs before uvicorn imports the app module, so the model is not loaded
    # yet -- a fatal environment mismatch costs seconds, not a failed load.
    if not args.skip_preflight:
        _run_startup_preflight()

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()


@app.get("/v1/models")
async def list_models():
    served = os.environ.get("QSR_SERVED_MODEL_NAME", engine.MODEL)
    names = served.split()
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen-sm120-runtime",
                "root": engine.MODEL,
                "parent": None,
                "max_model_len": engine.capacity_tokens_per_slot if engine else 0,
                "permission": [
                    {
                        "id": f"modelperm-{uuid.uuid4().hex[:24]}",
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            }
            for name in names
        ],
    }


def _backend_snapshot(runner):
    """Ask the backend to describe itself; None if it cannot.

    Replaces the previous per-attribute reads. Those had to guess at internal
    shape -- ``slot_kv_len`` as list or mapping, ``_prefix_cache_*`` by name --
    and guessing wrong is what returned 500 from /metrics twice on 2026-08-01.
    A backend that implements the contract now answers for its own shape.

    Still tolerant of a backend that has no ``snapshot`` at all, and still
    swallows failure, because the older constraint has not changed: an
    observability endpoint that takes the monitoring signal down with it is
    worse than one reporting a slightly wrong number. The difference is that
    degrading now requires a backend outside the contract, rather than merely
    one that stores its slot lengths differently.
    """
    snapshot = getattr(runner, "snapshot", None)
    if snapshot is None:
        return None
    try:
        return snapshot()
    except Exception:  # pragma: no cover - defensive; see docstring
        logger.exception("backend snapshot failed; reporting degraded metrics")
        return None


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics in the BlackweLLM namespace."""
    assert engine is not None
    runner = engine.runner
    # LagunaBackend uses static block allocation (num_slots × blocks_per_slot),
    # not a dynamic BlockPool. Compute KV usage from active slot count.
    total_blocks = engine.num_slots * engine.blocks_per_slot
    active_slots = len(engine.active)
    # This sum previously indexed the backend's own `slot_kv_len` and assumed
    # it was a mapping; it is a list, so a busy server (non-empty
    # `engine.active`) returned 500 while an idle one scraped fine. Observed
    # live 2026-08-01 during a 68 s request. The snapshot removes the shape
    # assumption rather than tolerating it; the fallback below is for a
    # backend that does not implement the contract at all.
    snapshot = _backend_snapshot(runner)
    if snapshot is None:
        used_blocks = active_slots * engine.blocks_per_slot
    else:
        by_slot = {s.slot: s.kv_len for s in snapshot.slots}
        used_blocks = sum(
            (by_slot.get(s, 0) + engine.block_size - 1) // engine.block_size for s in engine.active
        )
        # Feed the per-slot gauge here, where the per-slot numbers already
        # exist. `record_slot_kv_usage` had zero callers, so D2's
        # `slot_kv_usage_fraction` series was exported and never populated --
        # and an empty series is omitted entirely, so it read as "no data yet"
        # rather than as a broken pipe.
        for slot in engine.active:
            metrics.record_slot_kv_usage(
                slot,
                (by_slot.get(slot, 0) + engine.block_size - 1) // engine.block_size,
                engine.blocks_per_slot,
            )
    kv_usage = used_blocks / total_blocks if total_blocks > 0 else 0.0

    num_running = len(engine.active)
    num_waiting = len(engine.waiting)
    num_free_slots = len(engine.free_slots)

    lines = [
        "# HELP blackwellm:num_requests_running Number of requests currently running.",
        "# TYPE blackwellm:num_requests_running gauge",
        f'blackwellm:num_requests_running{{model_name="{engine.MODEL}"}} {num_running}',
        "# HELP blackwellm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE blackwellm:num_requests_waiting gauge",
        f'blackwellm:num_requests_waiting{{model_name="{engine.MODEL}"}} {num_waiting}',
        "# HELP blackwellm:kv_cache_usage_perc KV cache usage percentage.",
        "# TYPE blackwellm:kv_cache_usage_perc gauge",
        f'blackwellm:kv_cache_usage_perc{{model_name="{engine.MODEL}"}} {kv_usage:.4f}',
        "# HELP blackwellm:num_free_slots Number of free production slots.",
        "# TYPE blackwellm:num_free_slots gauge",
        f'blackwellm:num_free_slots{{model_name="{engine.MODEL}"}} {num_free_slots}',
        "# HELP blackwellm:capacity_tokens_per_slot Max tokens per slot.",
        "# TYPE blackwellm:capacity_tokens_per_slot gauge",
        f'blackwellm:capacity_tokens_per_slot{{model_name="{engine.MODEL}"}} '
        f"{engine.capacity_tokens_per_slot}",
        "# HELP blackwellm:requests_completed_total Total completed requests.",
        "# TYPE blackwellm:requests_completed_total counter",
        f'blackwellm:requests_completed_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('requests_completed', 0)}",
        "# HELP blackwellm:prefix_cache_hit_rate Prefix cache hit rate.",
        "# TYPE blackwellm:prefix_cache_hit_rate gauge",
        f'blackwellm:prefix_cache_hit_rate{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_hit_rate', 0.0):.4f}",
        "# HELP blackwellm:prefix_cache_hits_total Prefix cache hits.",
        "# TYPE blackwellm:prefix_cache_hits_total counter",
        f'blackwellm:prefix_cache_hits_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_hits', 0)}",
        "# HELP blackwellm:prefix_cache_misses_total Prefix cache misses.",
        "# TYPE blackwellm:prefix_cache_misses_total counter",
        f'blackwellm:prefix_cache_misses_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_misses', 0)}",
        "# HELP blackwellm:kv_cache_total_blocks Total KV cache blocks.",
        "# TYPE blackwellm:kv_cache_total_blocks gauge",
        f'blackwellm:kv_cache_total_blocks{{model_name="{engine.MODEL}"}} {total_blocks}',
        "# HELP blackwellm:kv_cache_used_blocks Used KV cache blocks.",
        "# TYPE blackwellm:kv_cache_used_blocks gauge",
        f'blackwellm:kv_cache_used_blocks{{model_name="{engine.MODEL}"}} {used_blocks}',
    ]

    # DFlash CUDA Graph capture health, one gauge per graph DFlash has
    # actually attempted to capture ("verify"/"draft"/"decode"). Before this,
    # a capture failure was observable only by grepping startup logs for one
    # exact line -- Prometheus never saw it, so "degrade but loud" was loud
    # only to someone already looking. See
    # notes/2026-08-01-c1-c2-gpu-investigation.md and
    # BackendSnapshot.dflash_cg_status's docstring.
    #
    # Always emits the HELP/TYPE header even with zero series below (DFlash
    # disabled, snapshot unavailable, or no capture attempted yet) -- that is
    # valid Prometheus exposition format and keeps this metric's presence
    # stable across scrapes instead of appearing/disappearing. Never raises:
    # `snapshot` is already None-safe from `_backend_snapshot` above, and
    # `dflash_cg_status` is `()` (not missing, not an attribute error) in
    # every case that isn't "at least one real capture attempt happened".
    lines.append(
        "# HELP blackwellm:dflash_cg_captured Whether a DFlash CUDA Graph is "
        "captured (1) or degraded to its eager fallback (0)."
    )
    lines.append("# TYPE blackwellm:dflash_cg_captured gauge")
    cg_status = snapshot.dflash_cg_status if snapshot is not None else ()
    for graph_name, status in cg_status:
        captured = 1 if status == "captured" else 0
        lines.append(
            f'blackwellm:dflash_cg_captured{{model_name="{engine.MODEL}",graph="{graph_name}"}} '
            f"{captured}"
        )

    # Accuracy/correctness signal: the admission bootstrap check re-runs each
    # speculative prefill on an independent reference slot and compares the
    # first committed token. A non-zero failure count means the MTP path
    # diverged from the greedy reference output (a real correctness problem).
    lines.append(
        "# HELP blackwellm:bootstrap_checks_ok_total "
        "Speculative prefills matching the reference prefill."
    )
    lines.append("# TYPE blackwellm:bootstrap_checks_ok_total counter")
    lines.append(
        f'blackwellm:bootstrap_checks_ok_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('bootstrap_checks_ok', 0)}"
    )
    lines.append(
        "# HELP blackwellm:bootstrap_checks_failed_total "
        "Speculative prefills diverged from reference."
    )
    lines.append("# TYPE blackwellm:bootstrap_checks_failed_total counter")
    lines.append(
        f'blackwellm:bootstrap_checks_failed_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('bootstrap_checks_failed', 0)}"
    )

    # App-layer request metrics: latency, TTFT, TPOT, token throughput, and
    # success/error counters (recorded per request in the handlers above).
    lines.extend(metrics.render(engine.MODEL))

    # D2: runtime-internal metrics (MTP acceptance, prefix cache depth, per-slot KV)
    lines.append(metrics.render_d2_metrics(engine.MODEL))

    # D3: request-level tracing stats
    lines.append(tracer.render_prometheus(engine.MODEL))

    from fastapi.responses import PlainTextResponse

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; charset=utf-8")


@app.get("/debug/traces")
async def debug_traces(request_id: str | None = None, slow: bool = False, limit: int = 20):
    """D3: Request-level tracing debug endpoint.

    Query params:
      - request_id: get trace for a specific request
      - slow=true: get recent slow requests
      - limit: max number of traces to return (default 20)
    """
    if request_id:
        trace = tracer.get_trace(request_id)
        if trace is None:
            return {"error": "trace not found", "request_id": request_id}
        return trace
    if slow:
        return {"slow_requests": tracer.get_slow_requests(limit)}
    return {"recent": tracer.get_recent(limit), "stats": tracer.get_stats()}


@app.api_route("/v1", methods=["GET", "POST"])
async def v1_root():
    return {
        "object": "api_info",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions", "/metrics"],
        "model": engine.MODEL,
    }


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Anthropic token counting endpoint (Claude Desktop requires this)."""
    assert engine is not None
    body = await request.json()
    from server.formats import convert_tools_to_chat_template
    from server.formats.anthropic import parse_messages

    chat_messages = parse_messages(body)
    tools = convert_tools_to_chat_template(body.get("tools"))
    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    if DEBUG_REQUESTS:
        logger.info(
            "ANTHROPIC count_tokens: msgs=%d system_chars=%d tools=%d -> input_tokens=%d",
            len(body.get("messages", [])),
            len(str(body.get("system") or "")),
            len(body.get("tools", [])),
            len(prompt_ids),
        )
    return {"input_tokens": len(prompt_ids)}


# -- Anthropic Messages API (/v1/messages) ---------------------------------
# Full format handling delegated to server/formats.py.
# This handler only does: parse -> tokenize -> engine.submit -> format response.


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    assert engine is not None
    body = await request.json()
    t0 = time.perf_counter()

    # Diagnostic: log request shape for Claude Desktop debugging
    _sys = body.get("system")
    _sys_len = len(str(_sys)) if _sys else 0
    _msgs = body.get("messages", [])
    _tools_n = len(body.get("tools", []))
    _stream = body.get("stream", False)
    logger.info(
        "ANTHROPIC REQ: system_chars=%d msgs=%d tools=%d stream=%s max_tokens=%s qs=%s",
        _sys_len,
        len(_msgs),
        _tools_n,
        _stream,
        body.get("max_tokens"),
        str(request.url.query) if request.url.query else "-",
    )

    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    model_name = body.get("model") or engine.MODEL
    stream = body.get("stream", False)
    sampling_params = _build_sampling_params(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
    )
    # Anthropic's stop_sequences has no documented count limit (unlike
    # OpenAI's stop, capped at 4) -- see _normalize_stop's docstring.
    stop_sequences = _normalize_stop(body.get("stop_sequences"))

    # Parse through the Anthropic format layer (handles array content, tool_use, tool_result)
    chat_messages = anthropic_format.parse_messages(body)
    if not chat_messages:
        # _invalid_request()'s shape is reshaped per-protocol by
        # _http_exception_handler (below), so this endpoint no longer needs
        # its own hand-rolled Anthropic-shaped JSONResponse for validation.
        raise _invalid_request("no messages provided")

    # Convert tools for the chat template
    tools = convert_tools_to_chat_template(body.get("tools"))

    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    await _debug_log_input("ANTHROPIC /v1/messages", body, chat_messages, prompt_ids)

    effective_max = min(max_tokens, engine.capacity_tokens_per_slot - len(prompt_ids) - 1)
    if effective_max < 1:
        raise _invalid_request("prompt too long for requested max_tokens")

    if stream:
        import json as _json

        async def _anthropic_sse():
            proc = StreamProcessor(engine.tok, thinking_capable=SERVER_THINKING_CAPABLE)
            final_result = None
            first_token_t = None
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            msg_start = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": len(prompt_ids),
                        "output_tokens": 0,
                        "cache_read_input_tokens": (
                            final_result.get("prefix_cache_hit_tokens", 0) if final_result else 0
                        ),
                        "cache_creation_input_tokens": 0,
                    },
                },
            }
            yield f"event: message_start\ndata: {_json.dumps(msg_start)}\n\n"
            yield "event: ping\ndata: " + _json.dumps({"type": "ping"}) + "\n\n"

            block_index = 0
            text_open = False

            _cancel_ref: list[str | None] = [None]
            async for item in engine.submit_stream(
                prompt_ids,
                effective_max,
                sampling_params=sampling_params,
                cancel_ref=_cancel_ref,
                stop_sequences=stop_sequences,
            ):
                if await request.is_disconnected():
                    if _cancel_ref[0]:
                        engine.cancel(_cancel_ref[0])
                    return
                if isinstance(item, dict):
                    final_result = item
                    break
                proc.add_tokens(item)
                if first_token_t is None and item:
                    first_token_t = time.perf_counter()

                # Reasoning is exposed via a custom, non-spec SSE event --
                # NOT a `thinking` content block. We cannot produce the
                # cryptographic signature real Anthropic thinking blocks
                # carry; shipping one anyway previously broke Claude
                # Desktop (it drops every content block, including
                # tool_use, that follows an invalid thinking block -- see
                # commit f13fd4a). An `event:` name outside Anthropic's
                # documented set is safe: compliant SSE consumers switch on
                # known event names and ignore the rest.
                if SERVER_REASONING_MODE == "expose":
                    for rdelta in proc.drain_thinking():
                        rd = {"type": "reasoning_content_delta", "delta": rdelta}
                        yield f"event: reasoning_content_delta\ndata: {_json.dumps(rd)}\n\n"

                for delta in proc.drain_content():
                    if not text_open:
                        text_open = True
                        bs = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                    d = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": delta},
                    }
                    yield f"event: content_block_delta\ndata: {_json.dumps(d)}\n\n"

            if text_open:
                yield (
                    "event: content_block_stop\ndata: "
                    + _json.dumps({"type": "content_block_stop", "index": block_index})
                    + "\n\n"
                )
                block_index += 1

            finish = final_result["finish_reason"] if final_result else "stop"
            matched_stop_sequence = (
                final_result.get("matched_stop_sequence") if final_result else None
            )
            if matched_stop_sequence:
                stop_reason = "stop_sequence"
            else:
                stop_reason = "end_turn" if finish == "stop" else "max_tokens"
            visible_text, tool_calls = proc.finalize()
            out_tokens = len(proc.all_ids)
            if tool_calls:
                stop_reason = "tool_use"
                from server.formats.tools import format_tool_calls_anthropic

                for tc in format_tool_calls_anthropic(tool_calls):
                    bs = {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": {},
                        },
                    }
                    yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                    delta_ev = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": _json.dumps(tc["input"]),
                        },
                    }
                    yield f"event: content_block_delta\ndata: {_json.dumps(delta_ev)}\n\n"
                    yield (
                        "event: content_block_stop\ndata: "
                        + _json.dumps({"type": "content_block_stop", "index": block_index})
                        + "\n\n"
                    )
                    block_index += 1

            if not text_open and not tool_calls:
                bs = {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                yield f"event: content_block_start\ndata: {_json.dumps(bs)}\n\n"
                yield (
                    "event: content_block_stop\ndata: "
                    + _json.dumps({"type": "content_block_stop", "index": 0})
                    + "\n\n"
                )

            msg_delta = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop_sequence},
                "usage": {"output_tokens": out_tokens},
            }
            yield f"event: message_delta\ndata: {_json.dumps(msg_delta)}\n\n"
            metrics.record_request(
                "messages",
                len(prompt_ids),
                len(proc.all_ids),
                finish,
                time.perf_counter() - t0,
                (first_token_t - t0) if first_token_t is not None else None,
            )
            await _debug_log_stream_output(
                "ANTHROPIC /v1/messages", proc, visible_text, tool_calls, finish
            )
            yield "event: message_stop\ndata: " + _json.dumps({"type": "message_stop"}) + "\n\n"

        return StreamingResponse(_anthropic_sse(), media_type="text/event-stream")

    # Non-streaming path
    result = await engine.submit(
        prompt_ids,
        effective_max,
        sampling_params=sampling_params,
        stop_sequences=stop_sequences,
    )
    _raw_anth = await _tokenize_decode(engine, result["committed_token_ids"])
    # Same state machine as the streaming path (server/formats/stream.py).
    proc = StreamProcessor(engine.tok, thinking_capable=SERVER_THINKING_CAPABLE)
    proc.add_tokens(result["committed_token_ids"])
    text = proc.content_text()
    reasoning_content = proc.reasoning_content() if SERVER_REASONING_MODE == "expose" else None
    metrics.record_request(
        "messages",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
    )
    _debug_log_output(
        "ANTHROPIC /v1/messages",
        _raw_anth,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
    return anthropic_format.build_response(
        model=model_name,
        text=text,
        finish_reason=result["finish_reason"],
        input_tokens=result["prompt_tokens"],
        output_tokens=result["completion_tokens"],
        cache_read_input_tokens=result.get("prefix_cache_hit_tokens", 0),
        reasoning_content=reasoning_content,
        stop_sequence=result.get("matched_stop_sequence"),
    )


# -- OpenAI Responses API (/v1/responses) ----------------------------------
# Adapter for Codex CLI 0.146+ (which removed `wire_api = "chat"`): the
# Responses protocol is translated onto the same engine pipeline as
# /v1/chat/completions. Request parsing and response building live in
# server/formats/responses.py; this handler only does parse -> tokenize ->
# engine.submit -> format, mirroring the Anthropic handler above.


@app.post("/v1/responses")
async def responses_api(request: Request):
    assert engine is not None
    body = await request.json()
    t0 = time.perf_counter()

    max_tokens = _validate_and_resolve_max_tokens(body.get("max_output_tokens"))
    model_name = body.get("model") or engine.MODEL
    stream = body.get("stream", False)
    sampling_params = _build_sampling_params(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
    )
    chat_messages = responses_format.parse_input(body)
    if not chat_messages:
        raise _invalid_request("no messages provided")
    tools = convert_tools_to_chat_template(body.get("tools"))
    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    await _debug_log_input("OPENAI /v1/responses", body, chat_messages, prompt_ids)
    _validate_capacity(prompt_ids, max_tokens)

    if stream:
        import json as _json

        async def _responses_sse():
            proc = StreamProcessor(
                engine.tok, thinking_capable=SERVER_THINKING_CAPABLE
            )
            final_result = None
            first_token_t = None
            resp_id = f"resp_{uuid.uuid4().hex[:24]}"
            created_at = int(time.time())
            output_index = 0
            text_item_id = f"msg_{uuid.uuid4().hex[:24]}"

            in_progress = responses_format.snapshot(
                resp_id, created_at, model_name, "in_progress", [], None
            )
            # Responses lifecycle events carry the full snapshot under a
            # ``response`` key with a top-level ``type`` (the OpenAI SSE
            # contract).  Sending the bare snapshot made Codex's client treat
            # the terminal events as unknown and reconnect ("stream closed
            # before response.completed").
            yield (
                "event: response.created\ndata: "
                + _json.dumps({"type": "response.created", "response": in_progress})
                + "\n\n"
            )
            yield (
                "event: response.in_progress\ndata: "
                + _json.dumps({"type": "response.in_progress", "response": in_progress})
                + "\n\n"
            )
            yield (
                "event: response.output_item.added\ndata: "
                + _json.dumps(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": text_item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                        },
                    }
                )
                + "\n\n"
            )
            yield (
                "event: response.content_part.added\ndata: "
                + _json.dumps(
                    {
                        "type": "response.content_part.added",
                        "item_id": text_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
                + "\n\n"
            )
            text_started = False
            last_send = time.monotonic()
            _cancel_ref: list[str | None] = [None]
            async for item in engine.submit_stream(
                prompt_ids,
                max_tokens,
                sampling_params=sampling_params,
                cancel_ref=_cancel_ref,
                stop_sequences=None,
            ):
                if await request.is_disconnected():
                    if _cancel_ref[0]:
                        engine.cancel(_cancel_ref[0])
                    return
                if isinstance(item, dict):
                    final_result = item
                    break
                proc.add_tokens(item)
                if first_token_t is None and item:
                    first_token_t = time.perf_counter()
                for delta in proc.drain_content():
                    text_started = True
                    ev = {
                        "type": "response.output_text.delta",
                        "item_id": text_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": delta,
                    }
                    yield f"event: response.output_text.delta\ndata: {_json.dumps(ev)}\n\n"
                    last_send = time.monotonic()
                # Long reasoning-only stretches emit no delta events; an SSE
                # comment keeps the connection alive for clients with an idle
                # read timeout (comments are ignored by spec-compliant SSE
                # parsers, including Codex's).
                if time.monotonic() - last_send >= 15:
                    yield ": keepalive\n\n"
                    last_send = time.monotonic()

            finish = final_result["finish_reason"] if final_result else "stop"
            visible_text, tool_calls = proc.finalize()
            output_items = []
            text = visible_text or ""
            output_items.append(
                responses_format.message_item(text_item_id, text)
            )
            if text_started:
                yield (
                    "event: response.output_text.done\ndata: "
                    + _json.dumps(
                        {
                            "type": "response.output_text.done",
                            "item_id": text_item_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "text": text,
                        }
                    )
                    + "\n\n"
                )
            yield (
                "event: response.content_part.done\ndata: "
                + _json.dumps(
                    {
                        "type": "response.content_part.done",
                        "item_id": text_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        },
                    }
                )
                + "\n\n"
            )
            yield (
                "event: response.output_item.done\ndata: "
                + _json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": output_items[-1],
                    }
                )
                + "\n\n"
            )
            output_index += 1
            for tc in tool_calls:
                fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                fc_item = responses_format.function_call_item(
                    fc_id,
                    tc["name"],
                    _json.dumps(tc["arguments"], ensure_ascii=False),
                )
                output_items.append(fc_item)
                yield (
                    "event: response.output_item.added\ndata: "
                    + _json.dumps(
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": {
                                "id": fc_id,
                                "type": "function_call",
                                "call_id": fc_item["call_id"],
                                "name": tc["name"],
                                "arguments": "",
                            },
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "event: response.function_call_arguments.delta\ndata: "
                    + _json.dumps(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": fc_id,
                            "output_index": output_index,
                            "delta": fc_item["arguments"],
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "event: response.function_call_arguments.done\ndata: "
                    + _json.dumps(
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": fc_id,
                            "output_index": output_index,
                            "arguments": fc_item["arguments"],
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "event: response.output_item.done\ndata: "
                    + _json.dumps(
                        {
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": fc_item,
                        }
                    )
                    + "\n\n"
                )
                output_index += 1

            usage = responses_format.build_usage(
                len(prompt_ids),
                len(proc.all_ids),
                final_result.get("prefix_cache_hit_tokens", 0)
                if final_result
                else 0,
            )
            completed = responses_format.snapshot(
                resp_id, created_at, model_name, "completed", output_items, usage
            )
            metrics.record_request(
                "responses",
                len(prompt_ids),
                len(proc.all_ids),
                finish,
                time.perf_counter() - t0,
                (first_token_t - t0) if first_token_t is not None else None,
            )
            await _debug_log_stream_output(
                "OPENAI /v1/responses", proc, visible_text, tool_calls, finish
            )
            yield (
                "event: response.completed\ndata: "
                + _json.dumps({"type": "response.completed", "response": completed})
                + "\n\n"
            )
            yield (
                "event: response.done\ndata: "
                + _json.dumps({"type": "response.done", "response": completed})
                + "\n\n"
            )

        return StreamingResponse(_responses_sse(), media_type="text/event-stream")

    result = await engine.submit(
        prompt_ids,
        max_tokens,
        sampling_params=sampling_params,
        stop_sequences=None,
    )
    raw_text = await _tokenize_decode(engine, result["committed_token_ids"])
    proc = StreamProcessor(engine.tok, thinking_capable=SERVER_THINKING_CAPABLE)
    proc.add_tokens(result["committed_token_ids"])
    text = proc.content_text()
    reasoning_content = (
        proc.reasoning_content() if SERVER_REASONING_MODE == "expose" else None
    )
    metrics.record_request(
        "responses",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
    )
    _debug_log_output(
        "OPENAI /v1/responses",
        raw_text,
        text,
        result["finish_reason"],
        result["completion_tokens"],
    )
    return responses_format.build_response(
        model=model_name,
        text=text,
        finish_reason=result["finish_reason"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        committed_token_ids=result["committed_token_ids"],
        reasoning_content=reasoning_content,
        prefix_cache_hit_tokens=result.get("prefix_cache_hit_tokens", 0),
    )
