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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from runtime.sampling import SamplingParams
from server import metrics
from server.engine import ServerEngine
from server.formats import anthropic as anthropic_format
from server.formats import convert_tools_to_chat_template, strip_thinking
from server.formats import openai as openai_format
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
# E1: which model backend to serve. "qwen36" (default) is the original,
# unmodified production path. "laguna" drives the new LagunaBackend second
# tenant (roadmap Track E / L2) -- it has no CUDA Graph integration yet
# (Lane 2 GPU work lives in runtime/backends/laguna_cuda_graph.py, not
# wired into the engine), no persistent prefix cache, and no session
# affinity, so those three default OFF for it below unless the operator
# explicitly opts back in via the same env vars. It also has no SWA
# ring-buffer KV yet (roadmap L2 TODO) -- every discovered attention layer,
# including the 36 sliding-window ones, currently gets a KV cache sized for
# the FULL context ceiling, so per-token memory cost is ~4x the roadmap L0
# budget note's estimate (which assumed that optimization already existed).
# SERVER_BLOCKS_PER_SLOT's Laguna default is sized conservatively for this
# reality, not for the eventual ring-buffer-optimized budget.
SERVER_MODEL_BACKEND = os.environ.get("QSR_SERVER_MODEL_BACKEND", "qwen36")
_IS_LAGUNA = SERVER_MODEL_BACKEND == "laguna"

SERVER_CAPACITY = int(os.environ.get("QSR_SERVER_CAPACITY", "1" if _IS_LAGUNA else "4"))
SERVER_NUM_SLOTS = int(os.environ.get("QSR_SERVER_NUM_SLOTS", "1" if _IS_LAGUNA else "8"))
SERVER_BLOCK_SIZE = int(os.environ.get("QSR_SERVER_BLOCK_SIZE", "64" if _IS_LAGUNA else "16"))
# 256K context support: Qwen uses 16384 × 16; Laguna's sparkinfer attention
# uses 4096 × 64.  The Laguna default below is currently 128K/slot.
# The KV cache pool size is now determined by GPU memory profiling (see
# server/engine.py _load_model → profile_kv_cache_blocks), NOT by the old
# fixed formula (num_slots + 1) * blocks_per_slot. blocks_per_slot is the
# per-slot MAXIMUM context ceiling; the actual pool is sized to fit the GPU.
# The E2E check sets its OWN smaller blocks_per_slot (its prompts are moderate),
# so it does not pay for the full long-context pool.
# Laguna default (2048 × 64 = 128K/slot) is conservative pending the SWA
# ring-buffer optimization above -- see notes/2026-07-23-laguna-server-
# integration-plan.md for the memory math.
SERVER_BLOCKS_PER_SLOT = int(
    os.environ.get("QSR_SERVER_BLOCKS_PER_SLOT", "2048" if _IS_LAGUNA else "16384")
)
SERVER_ENABLE_CUDAGRAPH = (
    os.environ.get("QSR_SERVER_ENABLE_CUDAGRAPH", "0" if _IS_LAGUNA else "1") != "0"
)
# P4a (notes/prefix-cache-design.md sec 5-P4): the prefix-cache rollback
# spine, plumbed straight into ServerEngine(enable_prefix_cache=...). Default
# ON (this is THE product value -- warm prefix hits served across requests);
# `python -m server.app --no-prefix-cache` (or QSR_SERVER_ENABLE_PREFIX_CACHE=0)
# turns it off => byte-for-byte the old server.
SERVER_ENABLE_PREFIX_CACHE = (
    os.environ.get("QSR_SERVER_ENABLE_PREFIX_CACHE", "0" if _IS_LAGUNA else "1") != "0"
)
# P4b session affinity (notes/2026-07-20-p4b-session-affinity-plan.md): opt-in
# warm-slot retention. Default OFF => byte-for-byte P4a (without a session_id, or
# with the flag off, _finish_request does the unconditional reset_slot). Requires
# the prefix cache -- ServerEngine raises ValueError if affinity is on but prefix
# cache is off (warm-continue needs the persistent content-hash cache).
SERVER_ENABLE_SESSION_AFFINITY = os.environ.get("QSR_SERVER_ENABLE_SESSION_AFFINITY", "0") != "0"
SERVER_SESSION_TTL_S = float(os.environ.get("QSR_SERVER_SESSION_TTL_S", "30.0"))
# Laguna default: "auto" (vLLM's native/default KV dtype). FP8 KV has only
# been validated for Qwen's path in this codebase; Laguna's Lane 2 CUDA
# Graph benchmarks so far ran without an explicit override (see
# benchmarks/laguna_cudagraph_bench.py) -- don't claim fp8_e4m3 works here
# until that's actually exercised on GPU.
SERVER_KV_CACHE_DTYPE = os.environ.get(
    "QSR_SERVER_KV_CACHE_DTYPE", "auto" if _IS_LAGUNA else "fp8_e4m3"
)
SERVER_GPU_MEM_UTIL = float(os.environ.get("QSR_SERVER_GPU_MEM_UTIL", "0.85"))
SERVER_PRODUCTION = os.environ.get("QSR_SERVER_PRODUCTION", "1") != "0"

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

    Laguna requires BOS (add_special_tokens=True, the default).
    Qwen3.6 does not use BOS (add_special_tokens=False).
    """
    loop = asyncio.get_running_loop()
    fn = functools.partial(engine_ref.tok.encode, text, add_special_tokens=_IS_LAGUNA)
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
    logger.info(
        "loading model backend=%s (this can take a while: model load + KV cache alloc)...",
        SERVER_MODEL_BACKEND,
    )
    engine = ServerEngine(
        backend=SERVER_MODEL_BACKEND,
        capacity=SERVER_CAPACITY,
        num_slots=SERVER_NUM_SLOTS,
        block_size=SERVER_BLOCK_SIZE,
        blocks_per_slot=SERVER_BLOCKS_PER_SLOT,
        kv_cache_dtype=SERVER_KV_CACHE_DTYPE,
        enable_cudagraph=SERVER_ENABLE_CUDAGRAPH,
        enable_prefix_cache=SERVER_ENABLE_PREFIX_CACHE,
        enable_session_affinity=SERVER_ENABLE_SESSION_AFFINITY,
        session_ttl_s=SERVER_SESSION_TTL_S,
        gpu_memory_utilization=SERVER_GPU_MEM_UTIL,
        production=SERVER_PRODUCTION,
    )
    engine.start()
    logger.info(
        "engine ready: backend=%s model=%s capacity=%d num_slots=%d capacity_tokens_per_slot=%d "
        "cudagraph=%s prefix_cache=%s session_affinity=%s ttl=%.1fs",
        SERVER_MODEL_BACKEND,
        engine.MODEL,
        engine.capacity,
        engine.num_slots,
        engine.capacity_tokens_per_slot,
        SERVER_ENABLE_CUDAGRAPH,
        SERVER_ENABLE_PREFIX_CACHE,
        SERVER_ENABLE_SESSION_AFFINITY,
        SERVER_SESSION_TTL_S,
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

    ``temperature == 0`` (or ``None``) selects greedy decode with MTP
    speculative verification.  ``temperature > 0`` enables true sampling
    (autoregressive, no MTP).
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
        seed=seed,
    )


def _validate_and_resolve_max_tokens(max_tokens: int | None) -> int:
    resolved = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if resolved <= 0:
        raise _invalid_request(f"max_tokens={max_tokens!r} must be >= 1.")
    return resolved


def _validate_capacity(prompt_ids: list[int], max_tokens: int, endpoint: str = "request") -> None:
    assert engine is not None
    if not engine.capacity_ok(len(prompt_ids), max_tokens):
        metrics.record_error(endpoint, 400)
        raise _invalid_request(
            f"prompt_tokens({len(prompt_ids)}) + max_tokens({max_tokens}) = "
            f"{len(prompt_ids) + max_tokens} exceeds this runtime's per-slot capacity of "
            f"{engine.capacity_tokens_per_slot} tokens (blocks_per_slot * block_size). "
            "Reduce the prompt length or max_tokens and retry."
        )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    # A defensive net so an unexpected runtime error (e.g. the engine's own
    # error-recovery path in server/engine.py's _loop) surfaces as a clean
    # 500 JSON body instead of an unhandled-exception stack trace / crash.
    logger.exception("unhandled exception serving %s", request.url.path)
    metrics.record_error(_endpoint_from_path(request.url.path), 500)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


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
    _validate_capacity(prompt_ids, max_tokens, "chat")

    model_name = req.model or engine.MODEL

    if req.stream:
        import json as _json

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def _sse():
            proc = StreamProcessor(engine.tok)
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
                response_format=req.response_format,
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
                # Stream thinking as reasoning_content (vLLM compatible)
                for td in proc.drain_thinking():
                    chunk = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"reasoning_content": td}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {_json.dumps(chunk)}\n\n"
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
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs or 0,
    )
    raw_text = await _tokenize_decode(engine, result["committed_token_ids"])
    _THINK_OPEN = chr(60) + "think" + chr(62)
    _THINK_CLOSE = chr(60) + "/think" + chr(62)
    # Non-thinking mode (chat_template_kwargs={"enable_thinking": False}): the
    # template already emitted a closed empty  block, so the generated
    # tokens ARE the answer -- there is no reasoning to strip. In thinking mode
    # the tokens start with the reasoning body (the opening  tag was
    # injected by the prompt), which we wrap and strip below.
    _non_thinking = bool(
        req.chat_template_kwargs and req.chat_template_kwargs.get("enable_thinking") is False
    ) or engine.backend_name != "qwen36"
    reasoning_content = None
    if _non_thinking:
        text = raw_text.replace("\ufffd", "").strip()
    else:
        _raw_for_strip = (
            raw_text if raw_text.startswith(_THINK_OPEN) else (_THINK_OPEN + "\n" + raw_text)
        )
        text = strip_thinking(_raw_for_strip)
        _raw_with_think = (
            raw_text if raw_text.startswith(_THINK_OPEN) else _THINK_OPEN + "\n" + raw_text
        )
        if _THINK_CLOSE in _raw_with_think:
            start = _raw_with_think.index(_THINK_OPEN) + len(_THINK_OPEN)
            if start < len(_raw_with_think) and _raw_with_think[start] == "\n":
                start += 1
            end = _raw_with_think.index(_THINK_CLOSE)
            reasoning_content = _raw_with_think[start:end].strip().replace("\ufffd", "")
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
    )
    if reasoning_content:
        resp["choices"][0]["message"]["reasoning_content"] = reasoning_content
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
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    t0 = time.perf_counter()
    prompt_ids = await _tokenize_encode(engine, req.prompt)
    await _debug_log_input("OPENAI /v1/completions", req.model_dump(), req.prompt, prompt_ids)
    _validate_capacity(prompt_ids, max_tokens, "completions")

    result = await engine.submit(
        prompt_ids,
        max_tokens,
        session_id=req.session_id,
        sampling_params=sampling_params,
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs or 0,
    )
    _raw_comp = await _tokenize_decode(engine, result["committed_token_ids"])
    if engine.backend_name != "qwen36":
        # Non-thinking backends (e.g. Laguna) never emit a <think> block, so
        # there is no reasoning to strip -- wrapping in a synthetic <think>
        # prefix here would make strip_thinking's unclosed-block rule eat the
        # entire response (see notes/2026-07-27-*.md for the real request
        # that surfaced this as an empty completion).
        text = _raw_comp.replace("�", "").strip()
    else:
        _raw_comp_full = (
            _raw_comp
            if _raw_comp.startswith(chr(60) + "think" + chr(62))
            else (chr(60) + "think" + chr(62) + "\n" + _raw_comp)
        )
        text = strip_thinking(_raw_comp_full)
    metrics.record_request(
        "completions",
        result["prompt_tokens"],
        result["completion_tokens"],
        result["finish_reason"],
        time.perf_counter() - t0,
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
    args = parser.parse_args()

    # P4b: refuse --session-affinity together with --no-prefix-cache -- a clean
    # startup error, not a runtime crash (warm-continue needs the persistent
    # content-hash cache; ServerEngine.__init__ raises the same way as a backstop).
    if args.session_affinity and args.no_prefix_cache:
        parser.error(
            "--session-affinity requires the prefix cache (cannot combine with --no-prefix-cache)"
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


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics (vLLM naming convention)."""
    assert engine is not None
    runner = engine.runner
    pool = runner.block_pool
    total_blocks = pool.num_blocks - pool.reserved
    free_blocks = len(pool._free_queue)
    used_blocks = total_blocks - free_blocks
    kv_usage = used_blocks / total_blocks if total_blocks > 0 else 0.0

    num_running = len(engine.active)
    num_waiting = len(engine.waiting)
    num_free_slots = len(engine.free_slots)

    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running.",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="{engine.MODEL}"}} {num_running}',
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{model_name="{engine.MODEL}"}} {num_waiting}',
        "# HELP vllm:kv_cache_usage_perc KV cache usage percentage.",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f'vllm:kv_cache_usage_perc{{model_name="{engine.MODEL}"}} {kv_usage:.4f}',
        "# HELP vllm:num_free_slots Number of free production slots.",
        "# TYPE vllm:num_free_slots gauge",
        f'vllm:num_free_slots{{model_name="{engine.MODEL}"}} {num_free_slots}',
        "# HELP vllm:capacity_tokens_per_slot Max tokens per slot.",
        "# TYPE vllm:capacity_tokens_per_slot gauge",
        f'vllm:capacity_tokens_per_slot{{model_name="{engine.MODEL}"}} '
        f"{engine.capacity_tokens_per_slot}",
        "# HELP vllm:requests_completed_total Total completed requests.",
        "# TYPE vllm:requests_completed_total counter",
        f'vllm:requests_completed_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('requests_completed', 0)}",
        "# HELP vllm:prefix_cache_hit_rate Prefix cache hit rate.",
        "# TYPE vllm:prefix_cache_hit_rate gauge",
        f'vllm:prefix_cache_hit_rate{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_hit_rate', 0.0):.4f}",
        "# HELP vllm:prefix_cache_hits_total Prefix cache hits.",
        "# TYPE vllm:prefix_cache_hits_total counter",
        f'vllm:prefix_cache_hits_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_hits', 0)}",
        "# HELP vllm:prefix_cache_misses_total Prefix cache misses.",
        "# TYPE vllm:prefix_cache_misses_total counter",
        f'vllm:prefix_cache_misses_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('prefix_cache_misses', 0)}",
        "# HELP vllm:kv_cache_total_blocks Total KV cache blocks.",
        "# TYPE vllm:kv_cache_total_blocks gauge",
        f'vllm:kv_cache_total_blocks{{model_name="{engine.MODEL}"}} {total_blocks}',
        "# HELP vllm:kv_cache_used_blocks Used KV cache blocks.",
        "# TYPE vllm:kv_cache_used_blocks gauge",
        f'vllm:kv_cache_used_blocks{{model_name="{engine.MODEL}"}} {used_blocks}',
    ]

    # Accuracy/correctness signal: the admission bootstrap check re-runs each
    # speculative prefill on an independent reference slot and compares the
    # first committed token. A non-zero failure count means the MTP path
    # diverged from the greedy reference output (a real correctness problem).
    lines.append(
        "# HELP vllm:bootstrap_checks_ok_total Speculative prefills matching the reference prefill."
    )
    lines.append("# TYPE vllm:bootstrap_checks_ok_total counter")
    lines.append(
        f'vllm:bootstrap_checks_ok_total{{model_name="{engine.MODEL}"}} '
        f"{engine.stats.get('bootstrap_checks_ok', 0)}"
    )
    lines.append(
        "# HELP vllm:bootstrap_checks_failed_total Speculative prefills diverged from reference."
    )
    lines.append("# TYPE vllm:bootstrap_checks_failed_total counter")
    lines.append(
        f'vllm:bootstrap_checks_failed_total{{model_name="{engine.MODEL}"}} '
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

    # Parse through the Anthropic format layer (handles array content, tool_use, tool_result)
    chat_messages = anthropic_format.parse_messages(body)
    if not chat_messages:
        metrics.record_error("messages", 400)
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "no messages provided"},
            },
        )

    # Convert tools for the chat template
    tools = convert_tools_to_chat_template(body.get("tools"))

    prompt_ids = await _tokenize_chat(engine, chat_messages, tools=tools)
    await _debug_log_input("ANTHROPIC /v1/messages", body, chat_messages, prompt_ids)

    effective_max = min(max_tokens, engine.capacity_tokens_per_slot - len(prompt_ids) - 1)
    if effective_max < 1:
        metrics.record_error("messages", 400)
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt too long for requested max_tokens",
                },
            },
        )

    if stream:
        import json as _json

        async def _anthropic_sse():
            proc = StreamProcessor(engine.tok)
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

                # Advance the thinking state machine but do NOT emit thinking
                # blocks.  We cannot produce the cryptographic signature that
                # the official Anthropic API attaches via signature_delta;
                # Claude Desktop validates it and DROPS every content block
                # that follows an invalid thinking block -- including tool_use
                # (e.g. AskUserQuestion), which is why the user's selection
                # was lost as "(no content)".
                proc.drain_thinking()

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
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
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
    )
    _raw_anth = await _tokenize_decode(engine, result["committed_token_ids"])
    if engine.backend_name != "qwen36":
        # Non-thinking backends (e.g. Laguna) never emit a <think> block --
        # see the matching guard in /v1/completions for why unconditionally
        # wrapping in a synthetic <think> prefix would eat the whole response.
        text = _raw_anth.replace("�", "").strip()
    else:
        _raw_anth_full = (
            _raw_anth
            if _raw_anth.startswith(chr(60) + "think" + chr(62))
            else (chr(60) + "think" + chr(62) + "\n" + _raw_anth)
        )
        text = strip_thinking(_raw_anth_full)
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
    )
