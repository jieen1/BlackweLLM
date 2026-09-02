"""OpenAI + Anthropic compatible HTTP server for BlackweLLM runtime.

Wraps ``server/engine.py`` (continuous-batching engine) with full
OpenAI ``/v1/chat/completions`` and Anthropic ``/v1/messages`` APIs.

Capabilities (B1/C1 采样全链路 + streaming + tool calling):

- ``POST /v1/chat/completions``, ``POST /v1/completions``,
  ``POST /v1/messages`` (Anthropic format).
- Streaming (SSE) and non-streaming responses.
- Full sampling: temperature, top_p, top_k, seed (``runtime/sampling.py``).
  Explicit ``temperature == 0`` selects greedy; omitted sampler fields use the
  loaded model's profile (Qwen3.8 Flash-Next follows the official thinking vs
  non-thinking recommendations).
- Tool calling via chat template (``convert_tools_to_chat_template``).
- Configurable capacity (default 4 slots, 256K context per slot).
- Prefix cache with session affinity for warm multi-turn.
- CUDA Graph accelerated decode.
- FP8 KV cache (2× capacity vs BF16).
- Prometheus metrics at ``/metrics``.
"""

from __future__ import annotations

import asyncio
import fcntl
import functools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

from runtime.sampling import PersistentSeed, SamplingParams
from runtime.structured_output import ResponseFormat
from runtime.thinking_budget import ThinkingBudgetConfig
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


def _is_flashnext_model(model_path: str) -> bool:
    """Recognize the distinct Qwen3.8 Flash-Next checkpoint family."""
    normalized = model_path.casefold().replace("-", ".").replace("_", ".")
    return "flash.next" in normalized or "flashnext" in normalized or (
        os.environ.get("QSR_SERVER_BACKEND", "").casefold() == "flashnext"
    )


def _is_qwen36_family_model(model_path: str) -> bool:
    """Recognize Qwen3.6/Qwen3.8 checkpoints handled by ``qwen36``.

    The backend name is resolved from ``config.json`` later, but an explicit
    ``QSR_SERVER_BACKEND=qwen36`` is also enough to identify a local snapshot
    whose directory name carries no model-family hint. This only chooses safe
    launcher defaults before weights load; Laguna keeps its conservative ones.
    """
    if _is_flashnext_model(model_path):
        return False
    normalized = model_path.casefold().replace("-", ".").replace("_", ".")
    backend_hint = os.environ.get("QSR_SERVER_BACKEND", "").casefold()
    return (
        "qwen3.6" in normalized
        or "qwen3.8" in normalized
        or backend_hint == "qwen36"
    )


_FLASHNEXT_DEFAULT_PROFILE = _is_flashnext_model(SERVER_MODEL_PATH)
_QWEN_DSPARK_DEFAULT_PROFILE = _is_qwen36_family_model(SERVER_MODEL_PATH)
_QWEN_DSPARK_POOL_BYTES = 19_629_342_720

# Qwen3.8-Flash-Next publishes separate sampler recommendations for its two
# template modes.  Keep these at the API boundary rather than changing
# ``SamplingParams``'s constructor defaults: direct runtime callers and the
# legacy Laguna/Qwen36 paths still rely on an explicit greedy default, while a
# Flash-Next request with omitted fields must not silently become temperature
# zero.  The model card also recommends min_p/presence/repetition penalties;
# this runtime currently exposes only temperature/top_p/top_k, so only the
# supported fields are resolved here and client-supplied values still win.
_LEGACY_SAMPLING_DEFAULTS: tuple[float, float, int] = (0.0, 1.0, 0)
_FLASHNEXT_THINKING_SAMPLING_DEFAULTS: tuple[float, float, int] = (1.0, 0.95, 20)
_FLASHNEXT_INSTRUCT_SAMPLING_DEFAULTS: tuple[float, float, int] = (0.7, 0.80, 20)

SERVER_CAPACITY = int(
    os.environ.get(
        "QSR_SERVER_CAPACITY",
        "1" if _FLASHNEXT_DEFAULT_PROFILE else "4" if _QWEN_DSPARK_DEFAULT_PROFILE else "1",
    )
)
# Qwen's measured DSpark profile uses four live slots. Laguna's non-DFlash
# decode graph keeps its conservative two-slot launcher default.
SERVER_NUM_SLOTS = int(
    os.environ.get(
        "QSR_SERVER_NUM_SLOTS",
        "1" if _FLASHNEXT_DEFAULT_PROFILE else "4" if _QWEN_DSPARK_DEFAULT_PROFILE else "2",
    )
)
# Qwen3.8's latest same-toolchain DSpark parity run used 128-token pages;
# Laguna's SparkInfer attention continues to require 64-token pages.
SERVER_BLOCK_SIZE = int(
    os.environ.get(
        "QSR_SERVER_BLOCK_SIZE",
        "128" if (_FLASHNEXT_DEFAULT_PROFILE or _QWEN_DSPARK_DEFAULT_PROFILE) else "64",
    )
)
# The KV cache pool size is now determined by GPU memory profiling (see
# server/engine.py _load_model → profile_kv_cache_blocks), NOT by the old
# fixed formula (num_slots + 1) * blocks_per_slot. blocks_per_slot is the
# per-slot MAXIMUM context ceiling; the actual pool is sized to fit the GPU.
# The Qwen DSpark default is 2048 × 128 = 256K per slot; Laguna remains
# 2048 × 64 = 128K per slot pending its SWA ring-buffer optimization.
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
SERVER_ENABLE_PREFIX_CACHE = os.environ.get(
    "QSR_SERVER_ENABLE_PREFIX_CACHE",
    "1",
) != "0"
# P4b session affinity (notes/2026-07-20-p4b-session-affinity-plan.md): opt-in
# warm-slot retention. Default OFF => byte-for-byte P4a (without a session_id, or
# with the flag off, _finish_request does the unconditional reset_slot). Requires
# the prefix cache -- ServerEngine raises ValueError if affinity is on but prefix
# cache is off (warm-continue needs the persistent content-hash cache).
SERVER_ENABLE_SESSION_AFFINITY = os.environ.get("QSR_SERVER_ENABLE_SESSION_AFFINITY", "0") != "0"
SERVER_SESSION_TTL_S = float(os.environ.get("QSR_SERVER_SESSION_TTL_S", "30.0"))
# Laguna default: ``auto``. FP8 KV has not been validated for Laguna, so an
# explicit override remains required before it can be used in production.
SERVER_KV_CACHE_DTYPE = os.environ.get(
    "QSR_SERVER_KV_CACHE_DTYPE",
    "fp8_e4m3" if _QWEN_DSPARK_DEFAULT_PROFILE else "auto",
)
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
# head. It remains an explicit rollback path now that the measured external
# DSpark profile is the Qwen default. See runtime/backends/qwen36_mtp.py for
# the round driver and the (token, hidden) pairing fix this path carries.
SERVER_ENABLE_MTP = os.environ.get(
    "QSR_SERVER_ENABLE_MTP", "1" if _FLASHNEXT_DEFAULT_PROFILE else "0"
) != "0"
SERVER_MTP_K = int(
    os.environ.get("QSR_SERVER_MTP_K", "3" if _FLASHNEXT_DEFAULT_PROFILE else "4")
)
# Qwen3.8 DFlash2 is an explicit, isolated opt-in. It loads a separate
# checkpoint; greedy draft/verify uses the captured native path, while sampled
# draft sampling remains eager because its request-local RNG is not graph-safe.
# DFlash2 never silently replaces the existing NVFP4+DSpark service profile.
SERVER_ENABLE_DFLASH2 = os.environ.get("QSR_SERVER_ENABLE_DFLASH2", "0") != "0"
SERVER_DFLASH2_DRAFT_MODEL = os.environ.get(
    "QSR_SERVER_DFLASH2_DRAFT_MODEL", "/home/bot/models/Qwen3.8-27B-DFlash2"
)
SERVER_DFLASH2_K = int(os.environ.get("QSR_SERVER_DFLASH2_K", "7"))
# Qwen3.6/Qwen3.8 DSpark uses a separate RadixArk draft checkpoint. The
# measured Qwen profile is now the default for a Qwen checkpoint; Laguna and
# generic imports remain unchanged. Set QSR_SERVER_ENABLE_DSPARK=0 (or use
# --mtp) for the explicit native-MTP rollback.
SERVER_ENABLE_DSPARK = os.environ.get(
    "QSR_SERVER_ENABLE_DSPARK",
    "0"
    if SERVER_ENABLE_DFLASH2 or _FLASHNEXT_DEFAULT_PROFILE
    else "1"
    if _QWEN_DSPARK_DEFAULT_PROFILE
    else "0",
) != "0"
SERVER_DSPARK_DRAFT_MODEL = os.environ.get(
    "QSR_SERVER_DSPARK_DRAFT_MODEL", "RadixArk/Qwen3.8-27B-DSpark"
)
SERVER_DSPARK_K = int(os.environ.get("QSR_SERVER_DSPARK_K", "7"))
SERVER_DSPARK_VERIFY_MODE = os.environ.get(
    "QSR_QWEN36_DSPARK_VERIFY_MODE",
    "compact" if _QWEN_DSPARK_DEFAULT_PROFILE else "static",
).strip().lower()
SERVER_DSPARK_REQUIRE_CG = os.environ.get(
    "QSR_QWEN36_DSPARK_REQUIRE_CG",
    "1" if _QWEN_DSPARK_DEFAULT_PROFILE and SERVER_ENABLE_CUDAGRAPH else "0",
) != "0"


def _apply_qwen_dspark_runtime_defaults() -> None:
    """Install the measured DSpark knobs before the backend is constructed.

    These are environment-backed because the same low-level switches are
    useful to the standalone profiling runbooks. ``setdefault`` preserves an
    operator's explicit A/B choice and keeps the Laguna process untouched.
    """
    # GGUF Q6 keeps the native linear weights in F32.  FlashInfer's GDN
    # prefill kernel only accepts fp16/bf16, so it cannot be the implicit
    # default for this checkpoint family.  Keep the measured FlashInfer path
    # for NVFP4 and select the compatible FLA path for GGUF without changing
    # any operator-supplied override.
    is_gguf = Path(SERVER_MODEL_PATH).suffix.casefold() == ".gguf"
    if is_gguf:
        # The server's production target path should use the SM120 BF16 graph
        # by default.  The loader retains F32 as its standalone bring-up
        # default, and an operator can explicitly restore it for diagnosis.
        os.environ.setdefault("QSR_GGUF_COMPUTE_DTYPE", "bf16")
    if not (SERVER_ENABLE_DSPARK or SERVER_ENABLE_DFLASH2):
        return
    gdn_prefill_backend = "fla" if is_gguf else "flashinfer"
    defaults = {
        "QSR_QWEN36_DSPARK_CUDA_GRAPH": "1" if SERVER_ENABLE_CUDAGRAPH else "0",
        "QSR_QWEN36_DSPARK_REQUIRE_CG": (
            "1" if SERVER_ENABLE_DFLASH2 else "1" if SERVER_DSPARK_REQUIRE_CG else "0"
        ),
        "QSR_QWEN36_DSPARK_VERIFY_MODE": SERVER_DSPARK_VERIFY_MODE,
        "QSR_PREFILL_CHUNK": "8192",
        "QSR_ADMISSION_COALESCE_MS": "10",
        "QSR_QWEN36_PREFILL_ATTN_BACKEND": "flashinfer",
        "QSR_QWEN36_GDN_PREFILL_BACKEND": gdn_prefill_backend,
        "QSR_QWEN36_MLP_FP4_QUANT": "flashinfer",
        "QSR_QWEN36_MLP_W4A4": "1",
        "QSR_QWEN36_MLP_W4A4_ALL": "1",
        "QSR_PREFIX_CACHE_IN_BATCH_DEDUP": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if is_gguf and SERVER_ENABLE_DFLASH2:
        # Keep packed GGUF weights resident and dequantize only genuine
        # prefill-sized batches into a short-lived BF16 matrix.  The fresh
        # SM120 A/B is materially faster than resident BF16 for the Q6 target
        # because it removes the packed M>1 prefill bottleneck without paying
        # the resident model-sized BF16 cache.  The helper keeps M=8 DFlash2
        # eager warmups and all graph replay on the packed path.
        # ``setdefault`` preserves explicit resident-BF16 and packed rollback
        # choices made by an operator.
        os.environ.setdefault("QSR_GGUF_DEQUANTIZE_WEIGHTS", "0")
        os.environ.setdefault("QSR_GGUF_NATIVE_PREFILL_DEQUANT", "1")


# Qwen dynamic KV Phase 4. ``legacy`` remains the rollback/default until the
# full 4x256K GPU matrix (Phase 5) is green. ``strict`` guarantees every
# configured slot's full context; ``elastic`` takes an explicit byte budget
# but still uses conservative full-sequence admission (no unsafe overcommit).
SERVER_QWEN_KV_MODE = os.environ.get(
    "QSR_QWEN_KV_MODE",
    (
        "legacy"
        if _FLASHNEXT_DEFAULT_PROFILE
        else "elastic"
        if _QWEN_DSPARK_DEFAULT_PROFILE
        else "legacy"
    ),
)
SERVER_QWEN_KV_POOL_BYTES = int(
    os.environ.get(
        "QSR_QWEN_KV_POOL_BYTES",
        str(_QWEN_DSPARK_POOL_BYTES)
        if _QWEN_DSPARK_DEFAULT_PROFILE and not _FLASHNEXT_DEFAULT_PROFILE
        else "0",
    )
)
SERVER_QWEN_KV_WATERMARK_BUNDLES = int(
    os.environ.get("QSR_QWEN_KV_WATERMARK_BUNDLES", "8")
)
# Phase 5.5: VMM-backed extensible physical KV (reserve full VA, commit the
# final pool size from measured post-capture memory). Requires a dynamic
# qwen_kv_mode; see notes/2026-08-16-vllm-extensible-kv-cache.md.
SERVER_QWEN_KV_EXTENSIBLE = os.environ.get("QSR_QWEN_KV_EXTENSIBLE", "0") == "1"
SERVER_QWEN_KV_COMMIT_BUFFER_GB = float(
    os.environ.get("QSR_QWEN_KV_COMMIT_BUFFER_GB", "10")
)
SERVER_QWEN_KV_FULL_SEQUENCE_MUST_FIT = (
    os.environ.get("QSR_QWEN_KV_FULL_SEQUENCE_MUST_FIT", "1") != "0"
)
# Recurrent-checkpoint budget, in multiples of one checkpoint
# (pool.recurrent_checkpoint_nbytes()). Default 0 keeps the backend
# default (2x). At 128K with a block_size=16 boundary a checkpoint is
# taken nearly every round per slot, and a budget that only holds two
# checkpoints forces constant evict+realloc of 96-tensor clones --
# measured 2026-08-06 as sporadic 50-85 ms GPU-queue stalls on ~17%
# of rounds. Raise it (e.g. 2*num_slots+2) to stop the churn.
SERVER_CHECKPOINT_BUDGET_MULTIPLE = int(
    os.environ.get("QSR_SERVER_CHECKPOINT_BUDGET_MULTIPLE", "0")
)
# Per-round KV resync (runtime/backends/qwen36_mtp.py's Qwen36MTPEngine
# docstring): independently toggleable from MTP itself so it can be A/B
# measured on real hardware separately from the pairing fix. Only consulted
# when SERVER_ENABLE_MTP is set; "unset" (None) lets Qwen36MTPEngine fall
# back to its own QSR_SERVER_MTP_RESYNC-driven default (also off).
_mtp_resync_env = os.environ.get("QSR_SERVER_MTP_RESYNC")
SERVER_MTP_RESYNC = None if _mtp_resync_env is None else _mtp_resync_env != "0"
SERVER_REQUEST_TIMEOUT_S = float(os.environ.get("QSR_SERVER_REQUEST_TIMEOUT_S", "600"))
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

# Qwen3.8's template uses ``reasoning_effort`` as a Jinja variable.  OpenAI
# clients, however, commonly put it at the request root (and the Responses
# API puts it under ``reasoning.effort``).  Keep the root-level compatibility
# mapping here instead of making every endpoint know the template details.
#
# ``high`` is deliberately not a runtime effort level.  The Anthropic relay
# currently translates its ``max`` setting to OpenAI ``high``; the Qwen
# adapter downgrades that compatibility value to ``medium`` before this
# resolver runs.  This prevents the old high -> xhigh alias from silently
# activating Qwen3.8's effectively unbounded thinking path.
_REASONING_EFFORT_VALUES = ("low", "medium", "xhigh", "none")
# The Flash-Next tokenizer is deliberately narrower than the generic
# OpenAI/OpenCode effort vocabulary: its template implements exactly
# ``low``, ``medium`` and ``xhigh`` (plus the separate ``enable_thinking``
# switch).  Keep the public compatibility aliases at the adapter boundary so
# clients can use their normal ``high``/``max`` controls without making the
# Jinja template reject the request.  Qwen36 has a different compatibility
# contract and still uses ``_qwen_compat_effort`` below.
_FLASHNEXT_REASONING_EFFORT_ALIASES = {
    "minimal": "low",
    "high": "xhigh",
    "max": "xhigh",
}
_QWEN_UNSUPPORTED_REASONING_EFFORTS = frozenset({"high", "xhigh", "max"})


def _flashnext_preserve_thinking_default() -> bool:
    """Return whether Flash-Next should replay old hidden reasoning by default.

    OpenCode sends assistant ``reasoning_content`` back in every subsequent
    request.  Flash-Next's template wraps that field in a new ``<think>``
    block when ``preserve_thinking`` is true; replaying a long tool history
    can therefore make a continuation deterministically re-enter the same
    plan.  Keep the safer, context-efficient policy as the default while
    retaining an explicit opt-in for clients that need full hidden-history
    replay.
    """
    return os.environ.get("QSR_FLASHNEXT_PRESERVE_THINKING", "0") != "0"


def _default_served_model_name(engine_ref) -> str:
    """Return the stable public model id, independent of snapshot paths."""
    configured = os.environ.get("QSR_SERVED_MODEL_NAME")
    if configured:
        return configured.split()[0]
    if engine_ref is None:
        return "qwen3.8-flash-next" if _FLASHNEXT_DEFAULT_PROFILE else "qwen3.8"
    if getattr(engine_ref, "backend_name", None) == "qwen36":
        return "qwen3.8"
    if getattr(engine_ref, "backend_name", None) == "flashnext":
        return "qwen3.8-flash-next"
    return engine_ref.MODEL


def _served_max_output_tokens(engine_ref) -> int:
    default = (
        32_000
        if getattr(engine_ref, "backend_name", None) == "flashnext"
        else DEFAULT_MAX_TOKENS
    )
    raw = os.environ.get("QSR_SERVED_MAX_OUTPUT_TOKENS", str(default)).strip()
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"QSR_SERVED_MAX_OUTPUT_TOKENS must be an integer, got {raw!r}") from exc
    if resolved <= 0:
        raise RuntimeError(
            f"QSR_SERVED_MAX_OUTPUT_TOKENS must be positive, got {resolved}"
        )
    capacity = getattr(engine_ref, "capacity_tokens_per_slot", 0) or 0
    speculative = max(0, int(getattr(engine_ref, "K", 0)))
    if capacity > 0:
        return min(resolved, max(1, capacity - speculative))
    return resolved


def _served_input_token_limit(engine_ref) -> int:
    output_limit = _served_max_output_tokens(engine_ref)
    advertised = getattr(engine_ref, "advertised_input_capacity", None)
    if advertised is not None:
        return int(advertised(output_limit))
    capacity = getattr(engine_ref, "capacity_tokens_per_slot", 0) or 0
    speculative = max(0, int(getattr(engine_ref, "K", 0)))
    if capacity > 0:
        return max(1, capacity - output_limit - speculative)
    return max(1, output_limit)


def _qwen_compat_effort(value: object) -> object:
    """Map unsupported high-effort compatibility values to medium.

    Claude/Anthropic clients use ``max`` while the local OpenAI-compatible
    relay currently forwards that as ``reasoning_effort=high``.  The legacy
    Qwen36 checkpoint has no validated high-effort path, so accepting that
    value and then translating it to xhigh is a dangerous silent behavior.
    Keep the client-facing request working, but make the Qwen36 runtime
    contract unambiguous: it executes low, medium, or disabled thinking, and
    the relay's high compatibility value executes as medium.  Flash-Next is
    handled separately because its native template *does* define xhigh.
    """
    if isinstance(value, str) and value.lower() in _QWEN_UNSUPPORTED_REASONING_EFFORTS:
        return "medium"
    return value


def _resolve_engine_chat_template_kwargs(
    engine_ref,
    chat_template_kwargs: dict | None,
    *,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    reasoning: dict | None = None,
    thinking: dict | None = None,
) -> dict | None:
    """Resolve request controls after applying the loaded model's contract."""
    requested_effort = reasoning_effort
    backend_name = getattr(engine_ref, "backend_name", None)
    if backend_name == "qwen36":
        if chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            for key in ("effort", "reasoning_effort", "level"):
                if key in chat_template_kwargs:
                    chat_template_kwargs[key] = _qwen_compat_effort(
                        chat_template_kwargs[key]
                    )
        reasoning_effort = _qwen_compat_effort(reasoning_effort)
        if isinstance(reasoning, dict):
            reasoning = {
                **reasoning,
                **{
                    key: _qwen_compat_effort(reasoning[key])
                    for key in ("effort", "reasoning_effort", "level")
                    if key in reasoning
                },
            }
        if isinstance(thinking, dict):
            thinking = {
                **thinking,
                **{
                    key: _qwen_compat_effort(thinking[key])
                    for key in ("effort", "reasoning_effort", "level")
                    if key in thinking
                },
            }
    elif backend_name == "flashnext":
        # Qwen3.8 Flash-Next's shipped template validates the exact set
        # ``low|medium|xhigh``.  OpenCode's standard variants and several
        # OpenAI-compatible relays use ``minimal|high|max`` instead; normalize
        # those aliases before the generic resolver reaches the Jinja layer.
        def _flashnext_effort(value: object) -> object:
            if isinstance(value, str):
                return _FLASHNEXT_REASONING_EFFORT_ALIASES.get(
                    value.lower(), value
                )
            return value

        if chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            for key in ("effort", "reasoning_effort", "level"):
                if key in chat_template_kwargs:
                    chat_template_kwargs[key] = _flashnext_effort(
                        chat_template_kwargs[key]
                    )
        reasoning_effort = _flashnext_effort(reasoning_effort)
        if isinstance(reasoning, dict):
            reasoning = {
                **reasoning,
                **{
                    key: _flashnext_effort(reasoning[key])
                    for key in ("effort", "reasoning_effort", "level")
                    if key in reasoning
                },
            }
        if isinstance(thinking, dict):
            thinking = {
                **thinking,
                **{
                    key: _flashnext_effort(thinking[key])
                    for key in ("effort", "reasoning_effort", "level")
                    if key in thinking
                },
            }
        # OpenCode echoes every prior ``reasoning_content`` field.  The
        # shipped Flash-Next template defaults ``preserve_thinking`` to true,
        # which feeds those hidden plans back into the next generation prompt
        # and is the direct cause of the observed repeated-thinking loop.
        # Drop historical reasoning by default; an explicit request kwarg or
        # QSR_FLASHNEXT_PRESERVE_THINKING=1 remains an escape hatch.
        if not chat_template_kwargs or "preserve_thinking" not in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs or {})
            chat_template_kwargs["preserve_thinking"] = (
                _flashnext_preserve_thinking_default()
            )
    resolved = _resolve_chat_template_kwargs(
        chat_template_kwargs,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
        reasoning=reasoning,
        thinking=thinking,
    )
    if DEBUG_REQUESTS and backend_name in {"qwen36", "flashnext"}:
        logger.info(
            "Qwen reasoning resolved: backend=%s requested=%r effective=%r "
            "thinking=%r preserve_history=%r",
            backend_name,
            requested_effort,
            resolved.get("reasoning_effort") if resolved else None,
            resolved.get("enable_thinking") if resolved else None,
            resolved.get("preserve_thinking") if resolved else None,
        )
    return resolved


def _resolve_chat_template_kwargs(
    chat_template_kwargs: dict | None,
    *,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    reasoning: dict | None = None,
    thinking: dict | None = None,
) -> dict | None:
    """Merge an API-level reasoning effort into chat-template kwargs.

    ``chat_template_kwargs`` is the escape hatch used by vLLM and remains the
    most explicit request-level control.  Therefore an explicit
    ``enable_thinking`` or ``reasoning_effort`` in that mapping wins over the
    OpenAI-compatible root field.  If the request does not select an effort,
    the tokenizer's configured model default is left untouched.
    ``none`` is represented by the Qwen hard switch because Qwen3.8's template
    accepts ``low|medium|xhigh``, not ``none``.  The engine-aware wrapper
    removes the unusable high-effort compatibility path before this function
    sees it.
    """
    resolved = dict(chat_template_kwargs or {})
    if "enable_thinking" in resolved or "reasoning_effort" in resolved:
        return resolved or None
    for controls in (reasoning, thinking):
        if not isinstance(controls, dict):
            continue
        if reasoning_effort is None:
            for key in ("effort", "reasoning_effort", "level"):
                candidate = controls.get(key)
                if candidate is not None:
                    reasoning_effort = candidate
                    break
        if enable_thinking is None:
            if "enable_thinking" in controls:
                enable_thinking = controls["enable_thinking"]
            elif controls.get("type") in ("disabled", "off", "none"):
                enable_thinking = False
            elif controls.get("type") in ("enabled", "on"):
                enable_thinking = True
    if enable_thinking is not None:
        if not isinstance(enable_thinking, bool):
            raise _invalid_request(
                f"enable_thinking must be a boolean, got {enable_thinking!r}"
            )
        resolved["enable_thinking"] = enable_thinking
        if enable_thinking is False:
            return resolved
    if reasoning_effort is None:
        return resolved or None
    if not isinstance(reasoning_effort, str):
        raise _invalid_request(
            "reasoning_effort must be one of "
            f"{', '.join(_REASONING_EFFORT_VALUES)}, got {reasoning_effort!r}"
        )
    effort = reasoning_effort.lower()
    if effort not in _REASONING_EFFORT_VALUES:
        raise _invalid_request(
            "reasoning_effort must be one of "
            f"{', '.join(_REASONING_EFFORT_VALUES)}, got {reasoning_effort!r}"
        )
    if effort == "none":
        resolved["enable_thinking"] = False
    else:
        resolved["reasoning_effort"] = effort
        # This mirrors vLLM's automatic activation for a request-level
        # effort.  Qwen3.8 already defaults to thinking, while other Qwen
        # templates may require the explicit switch.
        resolved["enable_thinking"] = True
    return resolved

# Selects which server/formats/tool_parsers/ shape to decode tool calls
# with -- mirrors vLLM's --tool-call-parser NAME. Qwen3.8 (including the
# Flash-Next NVFP4 checkpoint) emits the qwen3_coder XML shape; Laguna emits
# poolside_v1. An operator-supplied parser always wins.
_DEFAULT_TOOL_CALL_PARSER = (
    "qwen3_coder"
    if _is_flashnext_model(SERVER_MODEL_PATH)
    or Path(SERVER_MODEL_PATH).suffix.casefold() == ".gguf"
    else "poolside_v1"
)
SERVER_TOOL_CALL_PARSER = os.environ.get("QSR_TOOL_CALL_PARSER", _DEFAULT_TOOL_CALL_PARSER)

engine: ServerEngine | None = None
_GPU_PROCESS_LOCK_FD: int | None = None


def _acquire_gpu_process_lock() -> int:
    """Reserve this host's single GPU before preflight/model loading.

    The runtime is deliberately single-node/single-GPU. Starting a second
    Flash-Next process while the first one owns a 256K session pool otherwise
    lets both loaders consume nearly the whole card before either can report a
    useful error. ``flock`` is released by the kernel when the process exits,
    so a crashed service cannot leave a stale lock behind.
    """
    lock_path = os.environ.get("QSR_GPU_LOCK_PATH", "/tmp/qsr-gpu.lock")
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot open GPU process lock {lock_path!r}: {exc}") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(
            f"another qwen-sm120-runtime process already owns the GPU lock "
            f"{lock_path!r}; refusing a second model load to prevent OOM"
        ) from exc
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()}\n".encode())
    return fd


def _new_stream_processor(tokenizer, chat_template_kwargs: dict | None = None) -> StreamProcessor:
    """Create a response parser that matches this request's rendered prompt.

    ``QSR_THINKING_CAPABLE`` describes whether the loaded model's template can
    leave generation inside an open ``<think>`` block.  Qwen3.6 and Qwen3.8
    override that behavior per request: ``enable_thinking=False`` emits a
    *closed* ``<think></think>`` block in the prompt, so generated tokens begin
    with visible content and must not be parsed as an implicit reasoning body.
    """
    request_opens_thinking = SERVER_THINKING_CAPABLE and not (
        chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False
    )
    return StreamProcessor(tokenizer, thinking_capable=request_opens_thinking)


async def _tokenize_chat(engine_ref, messages, tools=None, chat_template_kwargs=None):
    """Run apply_chat_template in a thread to avoid blocking the event loop.

    ``chat_template_kwargs`` is forwarded verbatim to the Jinja template, so the
    official Qwen3.6 ``{"enable_thinking": False}`` toggle (and any other template
    option) is honored exactly as in stock vLLM. Without this the template always
    defaults to thinking mode and the toggle sent by clients is silently ignored.

    The deepseek_v4 backend does not carry a Jinja chat template at all --
    the official ``encoding_dsv4.py`` message encoder is the contract (plan
    §7.2 / D9).  Its ``encode_messages`` returns the prompt string; the
    standard tokenizer then tokenizes it.  For DSV4 the adapter maps
    ``enable_thinking`` and ``reasoning_effort`` onto the official encoder.
    """
    loop = asyncio.get_running_loop()
    if getattr(engine_ref, "backend_name", None) == "deepseek_v4":
        from server.formats.dsv4_encoding import encode_messages_dsv4

        encode = functools.partial(
            encode_messages_dsv4,
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        prompt = await loop.run_in_executor(None, encode)
        return engine_ref.tok.encode(prompt, add_special_tokens=False)
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


async def _tokenize_multimodal_chat(
    engine_ref,
    messages,
    tools=None,
    chat_template_kwargs=None,
):
    """Tokenize text and prepare bounded Flash-Next image inputs.

    The normal text path remains byte-for-byte unchanged. For an image
    request, the chat template emits one image marker per source image; the
    processor then supplies the exact patch-token count and this helper
    expands each marker before capacity admission.
    """

    from runtime.model.flashnext.vision import (
        build_mrope_positions,
        expand_image_tokens,
        extract_image_blocks,
        has_video_blocks,
        prepare_image_inputs,
    )

    image_blocks = extract_image_blocks(messages)
    if has_video_blocks(messages):
        raise ValueError(
            "video inputs are not enabled yet; send still images to the Flash-Next runtime"
        )
    if image_blocks and getattr(engine_ref, "backend_name", None) != "flashnext":
        raise ValueError(
            "image inputs are currently supported only by the Flash-Next backend"
        )

    prompt_ids = await _tokenize_chat(
        engine_ref,
        messages,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not image_blocks:
        return list(prompt_ids), None
    if not getattr(engine_ref, "vision_enabled", False):
        raise ValueError(
            "Flash-Next vision is disabled; set QSR_FLASHNEXT_VISION=1 and restart the server"
        )
    image_token_id = getattr(engine_ref, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(engine_ref.tok, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("the loaded tokenizer does not define an image token id")
    checkpoint = (
        getattr(engine_ref, "vision_checkpoint", None)
        or getattr(engine_ref, "MODEL", None)
    )
    loop = asyncio.get_running_loop()
    prepare = functools.partial(
        prepare_image_inputs,
        messages,
        checkpoint=checkpoint,
    )
    try:
        prepared = await loop.run_in_executor(None, prepare)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    expanded = expand_image_tokens(
        list(prompt_ids),
        int(image_token_id),
        prepared.image_token_counts,
    )
    vision_start_token_id = getattr(engine_ref.tok, "vision_start_token_id", None)
    if vision_start_token_id is None:
        loaded_model = getattr(getattr(engine_ref, "runner", None), "model", None)
        loaded_cfg = getattr(loaded_model, "cfg", None)
        vision_start_token_id = getattr(loaded_cfg, "vision_start_token_id", None)
    rope_positions, next_rope_position = build_mrope_positions(
        expanded,
        image_token_id=int(image_token_id),
        vision_start_token_id=vision_start_token_id,
        image_grid_thw=prepared.image_grid_thw,
        spatial_merge_size=int(getattr(prepared, "spatial_merge_size", 2)),
    )
    prepared = replace(
        prepared,
        rope_positions=rope_positions,
        next_rope_position=next_rope_position,
    )
    logger.info(
        "Flash-Next image request: images=%d source=%s resized=%s visual_tokens=%d "
        "max_pixels=%d",
        len(image_blocks),
        prepared.source_sizes,
        prepared.resized_sizes,
        prepared.total_image_tokens,
        prepared.max_pixels,
    )
    return expanded, prepared


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

    # Apply the measured Qwen DSpark profile before model construction. The
    # low-level backend reads these switches while allocating/capturing its
    # graphs, so changing them after ServerEngine.start() would be too late.
    _apply_qwen_dspark_runtime_defaults()

    # Track A migration step 5 (docs/architecture.md §3.5.5): the backend
    # name used to be the hardcoded constant SERVER_MODEL_BACKEND. It is now
    # resolved from the checkpoint's own config.json -- registry's first
    # real production consumer, not just shadow-mode tests (see
    # runtime/model_registry.py). Resolution reads only config.json (fast,
    # no weights), so this still runs before the slow model load below, same
    # as the tool_call_parser check above it.
    from runtime.laguna_config import _resolve_laguna_model_dir
    from runtime.model_registry import resolve_checkpoint

    model_path = Path(SERVER_MODEL_PATH)
    if model_path.is_file() and model_path.suffix == ".gguf":
        resolution = resolve_checkpoint(model_path)
    else:
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
        enable_dspark=SERVER_ENABLE_DSPARK,
        dspark_draft_model=SERVER_DSPARK_DRAFT_MODEL,
        dspark_num_speculative_tokens=SERVER_DSPARK_K,
        enable_dflash2=SERVER_ENABLE_DFLASH2,
        dflash2_draft_model=SERVER_DFLASH2_DRAFT_MODEL,
        dflash2_num_speculative_tokens=SERVER_DFLASH2_K,
        checkpoint_budget_multiple=(SERVER_CHECKPOINT_BUDGET_MULTIPLE or None),
        qwen_kv_mode=SERVER_QWEN_KV_MODE,
        qwen_kv_pool_bytes=SERVER_QWEN_KV_POOL_BYTES,
        qwen_kv_watermark_bundles=SERVER_QWEN_KV_WATERMARK_BUNDLES,
        qwen_kv_full_sequence_must_fit=SERVER_QWEN_KV_FULL_SEQUENCE_MUST_FIT,
        qwen_kv_extensible=SERVER_QWEN_KV_EXTENSIBLE,
        qwen_kv_commit_buffer_gb=SERVER_QWEN_KV_COMMIT_BUFFER_GB,
        request_timeout_s=SERVER_REQUEST_TIMEOUT_S,
        gpu_memory_utilization=SERVER_GPU_MEM_UTIL,
        production=SERVER_PRODUCTION,
    )
    engine.start()
    logger.info(
        "engine ready: backend=%s served_model=%s checkpoint=%s capacity=%d "
        "num_slots=%d capacity_tokens_per_slot=%d "
        "cudagraph=%s prefix_cache=%s session_affinity=%s ttl=%.1fs dflash=%s "
        "mtp=%s(K=%d,resync=%s) dspark=%s(K=%d,draft=%s,verify=%s,require_cg=%s) "
        "dflash2=%s(K=%d,draft=%s)",
        engine.backend_name,
        _default_served_model_name(engine),
        engine.MODEL,
        engine.capacity,
        engine.num_slots,
        engine.capacity_tokens_per_slot,
        SERVER_ENABLE_CUDAGRAPH,
        engine.enable_prefix_cache,
        SERVER_ENABLE_SESSION_AFFINITY,
        SERVER_SESSION_TTL_S,
        SERVER_ENABLE_DFLASH,
        SERVER_ENABLE_MTP,
        SERVER_MTP_K,
        SERVER_MTP_RESYNC,
        SERVER_ENABLE_DSPARK,
        SERVER_DSPARK_K,
        SERVER_DSPARK_DRAFT_MODEL,
        SERVER_DSPARK_VERIFY_MODE,
        SERVER_DSPARK_REQUIRE_CG,
        SERVER_ENABLE_DFLASH2,
        SERVER_DFLASH2_K,
        SERVER_DFLASH2_DRAFT_MODEL,
    )
    # Cyclic-GC pauses land in the decode hot loop as 50-150 ms host stalls:
    # the 2026-08-06 128K/c4 node trace measured 667 GPU-idle gaps >0.5 ms
    # (33% of steady-state wall) and the round profile's worst rounds put
    # 100-150 ms inside accept_decision/draft_batch with the GPU idle --
    # the classic full-heap collection signature on a process holding a
    # 27B-parameter object graph. vLLM disables GC in its engine core for
    # the same reason; serving objects here are acyclic (request/response
    # trees freed by refcount), so the generational collector only costs.
    # QSR_DISABLE_GC=0 restores the default for comparison runs.
    if os.environ.get("QSR_DISABLE_GC", "1") == "1":
        import gc

        gc.collect()
        gc.disable()
        logger.info("QSR_DISABLE_GC=1: cyclic garbage collection disabled for the serving loop")
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


# -- schemas (loose OpenAI-compatible subset; the debug-only extra fields are
# documented in the endpoint models below). --


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
    # OpenAI-compatible request-level reasoning control.  It must be mapped
    # to the model's chat-template kwargs before tokenization; leaving this
    # field out makes pydantic silently ignore the client's setting and lets
    # Qwen3.8 fall back to its default xhigh instruction.
    reasoning_effort: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reasoning_effort", "reasoningEffort"),
    )
    # Qwen's model-specific sampler-level thinking budget. This is deliberately
    # separate from ``max_tokens``: the scheduler forces ``</think>`` at the
    # token boundary while preserving one continuous generation request.
    thinking_token_budget: int | None = None
    # OpenAI-compatible clients also send nested reasoning controls.  Keep
    # these explicit so pydantic cannot silently discard the user's choice.
    reasoning: dict | None = None
    enable_thinking: bool | None = None
    # Anthropic/agent clients sometimes use the same request body on this
    # endpoint; accepting the shape is harmless and keeps all adapters aligned.
    thinking: dict | None = None
    # Forwarded to the chat template (e.g. {"enable_thinking": False} for
    # non-thinking mode). Mirrors vLLM's chat_template_kwargs request field.
    chat_template_kwargs: dict | None = None


def _message_text_for_protocol_detection(content: object) -> str:
    """Return text from an OpenAI message content value.

    OpenCode's title request uses plain strings today, but accepting the
    standard text-part form keeps the protocol check stable if its client
    serializer changes.  Non-text parts are intentionally ignored: an image
    or tool payload must never make an unrelated request look like a title
    request.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
    return "".join(text_parts)


def _is_opencode_title_request(req: ChatCompletionRequest) -> bool:
    """Identify OpenCode's internal title-generation protocol request.

    OpenCode submits this as a normal chat completion before the first user
    turn.  Its system contract explicitly says to output only a short title,
    but the request omits ``enable_thinking``; Flash-Next therefore enters its
    default xhigh ``<think>`` path and can occupy the single runtime slot for
    a long time.  This is a protocol-mode correction, not a generation cap:
    the title request keeps the caller's ``max_tokens`` and terminates via the
    model's normal EOS/stream completion.
    """
    if req.tools or req.tool_choice not in (None, "none"):
        return False
    system_text = "\n".join(
        _message_text_for_protocol_detection(message.get("content"))
        for message in req.messages
        if message.get("role") == "system"
    ).casefold()
    user_text = "\n".join(
        _message_text_for_protocol_detection(message.get("content"))
        for message in req.messages
        if message.get("role") == "user"
    ).casefold()
    return (
        "you are a title generator" in system_text
        and "generate a title for this conversation" in user_text
    )


class CompletionRequest(BaseModel):
    model: str | None = None
    # OpenAI-compatible: prompt may be a string OR a list of token ids.
    # The token-id form exists for exact-workload parity benchmarks (the
    # historical 128K/c4 numbers were measured on synthetic token-id
    # fixtures whose text round-trip is not identity-preserving).
    prompt: str | list[int]
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


def _invalid_request(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> HTTPException:
    error = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return HTTPException(
        status_code=400,
        detail={"error": error},
    )


def _sampling_defaults_for_request(
    engine_ref,
    chat_template_kwargs: dict | None = None,
) -> tuple[float, float, int]:
    """Return sampler defaults for the loaded model and template mode.

    Qwen3.8 Flash-Next runs in thinking mode unless the chat template is
    explicitly switched off.  Its model card recommends ``(1.0, 0.95, 20)``
    for thinking and ``(0.7, 0.80, 20)`` for instruct/non-thinking.  Resolve
    this only when the request omitted an individual field; callers can still
    override temperature, top-p, or top-k independently in
    :func:`_build_sampling_params`.

    ``SERVER_MODEL_PATH`` is only a pre-load hint.  Once an engine exists its
    resolved backend name is authoritative, which prevents a stale launcher
    hint from applying Flash-Next sampling to a Qwen36 or Laguna engine.
    """
    backend_name = getattr(engine_ref, "backend_name", None)
    is_flashnext = backend_name == "flashnext" or (
        backend_name is None and _FLASHNEXT_DEFAULT_PROFILE
    )
    if not is_flashnext:
        return _LEGACY_SAMPLING_DEFAULTS
    if chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False:
        return _FLASHNEXT_INSTRUCT_SAMPLING_DEFAULTS
    return _FLASHNEXT_THINKING_SAMPLING_DEFAULTS


def _build_sampling_params(
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    n: int | None = None,
    *,
    defaults: tuple[float, float, int] | None = None,
) -> SamplingParams:
    """Validate and build SamplingParams from API request fields.

    Explicit ``temperature == 0`` selects greedy decode. An omitted field
    uses ``defaults`` (the legacy fallback is greedy; the Flash-Next HTTP
    adapters pass the model-card profile for the request's thinking mode).
    Both greedy and ``temperature > 0`` (true sampling) get MTP speculative
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
    default_temperature, default_top_p, default_top_k = (
        defaults if defaults is not None else _LEGACY_SAMPLING_DEFAULTS
    )
    temp = temperature if temperature is not None else default_temperature
    if temp < 0:
        raise _invalid_request(f"temperature must be >= 0, got {temp}")
    resolved_top_p = top_p if top_p is not None else default_top_p
    if not (0.0 < resolved_top_p <= 1.0):
        raise _invalid_request(f"top_p must be in (0, 1], got {resolved_top_p}")
    resolved_top_k = top_k if top_k is not None else default_top_k
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
    resolved = max_tokens if max_tokens is not None else _served_max_output_tokens(engine)
    if resolved <= 0:
        raise _invalid_request(f"max_tokens={max_tokens!r} must be >= 1.")
    return resolved


def _validate_thinking_token_budget(value: object | None) -> int | None:
    """Validate the request-level low-level thinking budget."""
    if value is None:
        return None
    # bool is an int subclass, but accepting true as a one-token budget is a
    # particularly surprising API failure mode.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _invalid_request(
            f"thinking_token_budget must be a positive integer, got {value!r}"
        )
    return value


def _resolve_thinking_token_budget(
    value: object | None,
    chat_template_kwargs: dict | None,
    *,
    reasoning: dict | None = None,
    thinking: dict | None = None,
) -> int | None:
    """Resolve only an explicitly requested thinking budget.

    Reasoning effort is a model/template control, not a hidden token quota.
    In particular, this function must never synthesize a budget from the
    server profile, ``max_tokens``, or an effort label.  A caller may still
    send ``thinking_token_budget``/``budget_tokens`` when it intentionally
    wants the low-level forced-end-marker contract.
    """
    budget = _validate_thinking_token_budget(value)
    if budget is None:
        for controls in (reasoning, thinking):
            if not isinstance(controls, dict):
                continue
            for key in ("thinking_token_budget", "budget_tokens"):
                candidate = controls.get(key)
                if candidate is not None:
                    budget = _validate_thinking_token_budget(candidate)
                    break
            if budget is not None:
                break
    if budget is not None and chat_template_kwargs:
        if chat_template_kwargs.get("enable_thinking") is False:
            raise _invalid_request(
                "thinking_token_budget requires thinking mode; remove "
                "enable_thinking=false or the budget"
            )
    return budget


async def _tokenize_thinking_budget_config(
    engine_ref, thinking_token_budget: int | None
) -> ThinkingBudgetConfig | None:
    """Resolve Qwen's marker ids for the scheduler-owned token state."""
    if thinking_token_budget is None:
        return None
    backend_name = getattr(engine_ref, "backend_name", None)
    if backend_name not in (None, "qwen36"):
        raise _invalid_request(
            "thinking_token_budget is supported only by the Qwen reasoning backend"
        )
    loop = asyncio.get_running_loop()
    start_fn = functools.partial(engine_ref.tok.encode, "<think>", add_special_tokens=False)
    end_fn = functools.partial(engine_ref.tok.encode, "</think>", add_special_tokens=False)
    start_ids, end_ids = await asyncio.gather(
        loop.run_in_executor(None, start_fn),
        loop.run_in_executor(None, end_fn),
    )
    start_ids = tuple(int(token_id) for token_id in start_ids)
    end_ids = tuple(int(token_id) for token_id in end_ids)
    if not start_ids or not end_ids:
        raise _invalid_request(
            "thinking_token_budget requires tokenizer ids for both <think> and </think>"
        )
    return ThinkingBudgetConfig(
        budget=thinking_token_budget,
        start_token_ids=start_ids,
        end_token_ids=end_ids,
    )


def _merge_generation_result(
    result: dict,
    generated_ids: list[int],
    prompt_tokens: int,
    logprobs: list[dict] | None,
) -> dict:
    """Normalize scheduler output while preserving authoritative token usage."""
    merged = dict(result)
    merged["committed_token_ids"] = list(generated_ids)
    merged["completion_tokens"] = len(generated_ids)
    merged["prompt_tokens"] = prompt_tokens
    if logprobs is not None:
        merged["logprobs"] = logprobs
    return merged


async def _submit_with_thinking_budget(
    engine_ref,
    prompt_ids: list[int],
    max_tokens: int,
    *,
    thinking_budget: ThinkingBudgetConfig | None,
    processor: StreamProcessor,
    session_id: str | None = None,
    sampling_params: SamplingParams | None = None,
    stop_sequences: list[str] | None = None,
    logprobs: bool = False,
    top_logprobs: int = 0,
    stop_on_tool_call: bool = False,
    vision_inputs=None,
) -> dict:
    """Submit one request with a sampler-level thinking constraint."""
    submit_kwargs = {
        "session_id": session_id,
        "sampling_params": sampling_params,
        "stop_sequences": stop_sequences,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "thinking_budget": thinking_budget,
        "stop_on_tool_call": stop_on_tool_call,
    }
    if vision_inputs is not None:
        submit_kwargs["vision_inputs"] = vision_inputs
    result = await engine_ref.submit(prompt_ids, max_tokens, **submit_kwargs)
    generated_ids = list(result.get("committed_token_ids", []))
    processor.add_tokens(generated_ids)
    generation_logprobs = list(result.get("logprobs") or []) if logprobs else None
    return _merge_generation_result(
        result, generated_ids, len(prompt_ids), generation_logprobs
    )


async def _submit_stream_with_thinking_budget(
    engine_ref,
    prompt_ids: list[int],
    max_tokens: int,
    *,
    thinking_budget: ThinkingBudgetConfig | None,
    processor: StreamProcessor,
    session_id: str | None = None,
    sampling_params: SamplingParams | None = None,
    cancel_ref: list | None = None,
    stop_sequences: list[str] | None = None,
    logprobs: bool = False,
    top_logprobs: int = 0,
    stop_on_tool_call: bool = False,
    vision_inputs=None,
):
    """Streaming counterpart of :func:`_submit_with_thinking_budget`."""
    generated_ids: list[int] = []
    stream_ids: list[int] = []
    result: dict | None = None
    submit_kwargs = {
        "session_id": session_id,
        "sampling_params": sampling_params,
        "cancel_ref": cancel_ref,
        "stop_sequences": stop_sequences,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "thinking_budget": thinking_budget,
        "stop_on_tool_call": stop_on_tool_call,
    }
    if vision_inputs is not None:
        submit_kwargs["vision_inputs"] = vision_inputs
    async for item in engine_ref.submit_stream(prompt_ids, max_tokens, **submit_kwargs):
        if isinstance(item, dict):
            result = item
            break
        processor.add_tokens(item)
        stream_ids.extend(item)
        yield item

    if result is None:
        return
    generated_ids = list(result.get("committed_token_ids", stream_ids))
    # Test doubles may yield only the terminal result.  The real engine emits
    # token batches first, so do not double-feed the normal stream.
    if not stream_ids:
        processor.add_tokens(generated_ids)
    yield _merge_generation_result(
        result,
        generated_ids,
        len(prompt_ids),
        (list(result.get("logprobs") or []) if logprobs else None),
    )


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

    Legacy/Laguna requests still default to temperature 0.0 (greedy) when a
    client does not set one explicitly.  Flash-Next requests instead resolve
    the model-card sampler profile at the API boundary, but structured-output
    masking is still absent for either profile.  Wiring only the narrow
    reachable slice (temperature > 0, decode tokens 2+) would silently leave
    the common/default case unconstrained while looking wired-in -- the same
    silent-failure shape this check exists to eliminate, just relocated.
    Reject loudly instead.
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


def _validate_capacity(
    prompt_ids: list[int], max_tokens: int, *, extra_tokens: int = 0
) -> None:
    # metrics.record_error is NOT called here: _http_exception_handler
    # records it once, uniformly, for every raised HTTPException -- an
    # explicit call here would double-count.
    assert engine is not None
    reserved_tokens = max_tokens + extra_tokens
    speculative_tokens = max(0, int(getattr(engine, "K", 0)))
    capacity_ok = getattr(engine, "capacity_ok", None)
    if capacity_ok is None:
        # Small API test doubles from before the dynamic-capacity helper only
        # expose the public ceiling.  Keep the validation useful for them
        # while production ServerEngine continues to own the exact policy.
        fits = (
            len(prompt_ids) + reserved_tokens + speculative_tokens
            <= engine.capacity_tokens_per_slot
        )
    else:
        fits = capacity_ok(len(prompt_ids), reserved_tokens)
    if not fits:
        parts = [
            f"prompt_tokens({len(prompt_ids)})",
            f"max_tokens({max_tokens})",
        ]
        if extra_tokens:
            parts.append(f"extra_tokens({extra_tokens})")
        if speculative_tokens:
            parts.append(f"speculative_tokens({speculative_tokens})")
        total = len(prompt_ids) + reserved_tokens + speculative_tokens
        available_prompt_tokens = max(
            0,
            int(engine.capacity_tokens_per_slot) - reserved_tokens - speculative_tokens,
        )
        raise _invalid_request(
            "prompt is too long: "
            f"{' + '.join(parts)} = {total} exceeds this runtime's context window of "
            f"{engine.capacity_tokens_per_slot} tokens (blocks_per_slot * block_size). "
            f"At this max_tokens setting, the largest admissible prompt is "
            f"available_prompt_tokens({available_prompt_tokens}). "
            "Reduce the prompt length or max_tokens and retry.",
            error_type="context_length_exceeded",
            code="context_length_exceeded",
        )


def _shrink_max_tokens_to_capacity(
    prompt_ids: list[int], requested_max_tokens: int, *, extra_tokens: int = 0
) -> int:
    """Clamp output budget to the actual remaining per-slot capacity."""
    assert engine is not None
    speculative_tokens = max(0, int(getattr(engine, "K", 0)))
    available = (
        int(engine.capacity_tokens_per_slot)
        - len(prompt_ids)
        - max(0, int(extra_tokens))
        - speculative_tokens
    )
    if available < 1:
        _validate_capacity(prompt_ids, requested_max_tokens, extra_tokens=extra_tokens)
    return min(int(requested_max_tokens), available)


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
    # The collection below walks the backend's full object graph
    # (``memory_breakdown``) and can take tens of seconds.  Running it on the
    # event loop froze EVERY endpoint -- including in-flight streaming chat
    # responses -- for the duration of one scrape (measured 37 s on the live
    # Flash-Next server, with /health unresponsive throughout).  Off-load it
    # to a worker thread so the loop keeps serving while the debug snapshot
    # is being built.
    return await asyncio.to_thread(_collect_debug_stats)


def _collect_debug_stats() -> dict:
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
        engine.stats["_backend_snapshot_stats_dbg"] = dict(snapshot.runtime_stats)
        engine.stats["_cuda_graph_fallback_reasons_dbg"] = dict(snapshot.cg_fallback_reasons)
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
    # Memory probe: per-category CUDA byte accounting from the backend.  Only
    # present when the backend implements it (DeepseekV4Backend does); the
    # field is omitted entirely otherwise so this endpoint stays truthful for
    # backends that have not opted in.
    memory_breakdown = getattr(engine.runner, "memory_breakdown", None)
    if memory_breakdown is not None:
        try:
            engine.stats["_memory_breakdown_dbg"] = memory_breakdown()
        except Exception:  # pragma: no cover - observability must not die
            logger.exception("memory breakdown failed; reporting without it")
    ple_stats = getattr(engine.runner, "ple_stats", None)
    if ple_stats is not None:
        try:
            engine.stats["_ple_stats_dbg"] = ple_stats()
        except Exception:  # pragma: no cover - observability must not die
            logger.exception("PLE stats failed; reporting without it")
    # Phase 0 KV capacity evidence (`.omx/plans/qwen38-dynamic-context-vllm-plan.md`):
    # formula KV bytes, measured tensor storage, and physical row layout from
    # the Qwen pool, present only when the backend opted in. This is the
    # load-time-config surface -- pool size, scratch row, per-slot geometry --
    # that cannot be inferred from a warm engine.
    kv_capacity = getattr(engine.runner, "kv_capacity_snapshot", None)
    if kv_capacity is not None:
        try:
            engine.stats["_qwen_kv_capacity_dbg"] = kv_capacity()
        except Exception:  # pragma: no cover - observability must not die
            logger.exception("KV capacity snapshot failed; reporting without it")
    return engine.stats


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    assert engine is not None
    _reject_unsupported_response_format(req.response_format)
    stop_sequences = _normalize_stop(req.stop, max_count=4)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    t0 = time.perf_counter()

    # Parse messages through the format layer (handles string | array content)
    chat_messages = openai_format.parse_chat_messages(req.model_dump())

    # Convert tools for the chat template
    tools = convert_tools_to_chat_template(req.tools)
    chat_template_kwargs = _resolve_engine_chat_template_kwargs(
        engine,
        req.chat_template_kwargs,
        reasoning_effort=req.reasoning_effort,
        enable_thinking=req.enable_thinking,
        reasoning=req.reasoning,
        thinking=req.thinking,
    )
    if _is_opencode_title_request(req):
        # The title generator is an internal OpenCode protocol turn, not a
        # user reasoning turn.  Its contract is "ONLY a thread title" and it
        # has no tools; leaving Flash-Next's default thinking prompt enabled
        # makes this bookkeeping request spend the single slot in a long
        # reasoning loop before the real user request can be admitted.  Keep
        # the completion window untouched and select the template's explicit
        # non-thinking mode instead.
        chat_template_kwargs = {"enable_thinking": False}
        logger.info("OpenCode title request: using non-thinking template mode")
    thinking_token_budget = _resolve_thinking_token_budget(
        req.thinking_token_budget,
        chat_template_kwargs,
        reasoning=req.reasoning,
        thinking=req.thinking,
    )
    sampling_params = _build_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        n=req.n,
        defaults=_sampling_defaults_for_request(engine, chat_template_kwargs),
    )

    try:
        prompt_ids, vision_inputs = await _tokenize_multimodal_chat(
            engine,
            chat_messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    except ValueError as exc:
        raise _invalid_request(str(exc)) from exc
    await _debug_log_input(
        "OPENAI /v1/chat/completions", req.model_dump(), chat_messages, prompt_ids
    )
    thinking_budget = await _tokenize_thinking_budget_config(engine, thinking_token_budget)
    _validate_capacity(
        prompt_ids,
        max_tokens,
    )

    model_name = req.model or _default_served_model_name(engine)

    if req.stream:
        import json as _json

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def _sse():
            proc = _new_stream_processor(engine.tok, chat_template_kwargs)
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
            async for item in _submit_stream_with_thinking_budget(
                engine,
                prompt_ids,
                max_tokens,
                thinking_budget=thinking_budget,
                processor=proc,
                session_id=req.session_id,
                sampling_params=sampling_params,
                cancel_ref=_cancel_ref,
                stop_sequences=stop_sequences,
                logprobs=bool(req.logprobs),
                top_logprobs=req.top_logprobs or 0,
                # Some agent clients (including OpenCode's OpenAI-compatible
                # adapter) put the tool contract in the prompt and omit the
                # JSON `tools` field.  The active parser is authoritative for
                # the wire format, so stop after a complete parsed call even
                # when the request did not repeat the schema.
                stop_on_tool_call=True,
                vision_inputs=vision_inputs,
            ):
                if await request.is_disconnected():
                    if _cancel_ref[0]:
                        engine.cancel(_cancel_ref[0])
                    return
                if isinstance(item, dict):
                    final_result = item
                    break
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
            stream_prompt_tokens = (
                final_result.get("prompt_tokens") if final_result else None
            )
            stream_completion_tokens = (
                final_result.get("completion_tokens") if final_result else None
            )
            prompt_tokens = (
                int(stream_prompt_tokens)
                if stream_prompt_tokens is not None
                else len(prompt_ids)
            )
            completion_tokens = (
                int(stream_completion_tokens)
                if stream_completion_tokens is not None
                else len(proc.all_ids)
            )
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
                # OpenCode's compaction/overflow guard consumes usage from the
                # streaming chunks.  Keep it on the existing terminal chunk
                # (rather than adding an empty-choice chunk) so clients that
                # assume every chunk has choices remain compatible.
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {_json.dumps(done)}\n\n"
            metrics.record_request(
                "chat",
                prompt_tokens,
                completion_tokens,
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
    proc = _new_stream_processor(engine.tok, chat_template_kwargs)
    result = await _submit_with_thinking_budget(
        engine,
        prompt_ids,
        max_tokens,
        thinking_budget=thinking_budget,
        processor=proc,
        session_id=req.session_id,
        sampling_params=sampling_params,
        stop_sequences=stop_sequences,
        logprobs=bool(req.logprobs),
        top_logprobs=req.top_logprobs or 0,
        stop_on_tool_call=True,
        vision_inputs=vision_inputs,
    )
    raw_text = await _tokenize_decode(engine, result["committed_token_ids"])
    # Same state machine as the streaming path (server/formats/stream.py) --
    # not a second, independently-written parser for the non-streaming case.
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
        # The legacy endpoint has no chat template and therefore no thinking
        # block; use Flash-Next's instruct/non-thinking profile when its
        # fields are omitted.
        defaults=_sampling_defaults_for_request(engine, {"enable_thinking": False}),
    )
    _reject_unsupported_response_format(req.response_format)
    stop_sequences = _normalize_stop(req.stop, max_count=4)
    max_tokens = _validate_and_resolve_max_tokens(req.max_tokens)
    t0 = time.perf_counter()
    if isinstance(req.prompt, list):
        prompt_ids = list(req.prompt)
    else:
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
        "model": req.model or _default_served_model_name(engine),
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
    is a HuggingFace repo id (or a local .gguf weight file for DSV4).
    ``_resolve_laguna_model_dir`` is the resolver the loader itself uses
    (offline-only, no network fetch) for directories; a local .gguf path
    is passed through as-is.  Importing the private name is deliberate --
    duplicating two lines of resolution logic here would be free to drift
    away from what actually gets loaded.
    """
    import sys

    from runtime.laguna_config import _resolve_laguna_model_dir
    from runtime.preflight import run_preflight

    try:
        model_path = Path(SERVER_MODEL_PATH)
        if model_path.is_file() and model_path.suffix == ".gguf":
            checkpoint_dir = model_path
        else:
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
    global _GPU_PROCESS_LOCK_FD
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capacity", type=int, default=SERVER_CAPACITY)
    parser.add_argument("--num-slots", type=int, default=SERVER_NUM_SLOTS)
    parser.add_argument("--blocks-per-slot", type=int, default=SERVER_BLOCKS_PER_SLOT)
    parser.add_argument(
        "--qwen-kv-mode",
        choices=("legacy", "strict", "elastic"),
        default=SERVER_QWEN_KV_MODE,
        help=(
            "Qwen KV allocation mode. legacy=fixed rows; strict=dynamic arena sized "
            "for every slot ceiling; elastic=dynamic arena with --qwen-kv-pool-bytes."
        ),
    )
    parser.add_argument(
        "--qwen-kv-pool-bytes",
        type=int,
        default=SERVER_QWEN_KV_POOL_BYTES,
        help="Physical Qwen KV byte budget for --qwen-kv-mode=elastic.",
    )
    parser.add_argument(
        "--qwen-kv-watermark-bundles",
        type=int,
        default=SERVER_QWEN_KV_WATERMARK_BUNDLES,
        help="Emergency/COW page bundles excluded from dynamic admission. Default 8.",
    )
    parser.add_argument(
        "--qwen-kv-extensible",
        action="store_true",
        default=SERVER_QWEN_KV_EXTENSIBLE,
        help=(
            "Phase 5.5: VMM-backed extensible physical KV -- reserve the full "
            "pool VA at load, commit the final size from measured post-capture "
            "memory. Requires --qwen-kv-mode strict/elastic."
        ),
    )
    parser.add_argument(
        "--qwen-kv-commit-buffer-gb",
        type=float,
        default=SERVER_QWEN_KV_COMMIT_BUFFER_GB,
        help="GiB of free GPU memory kept uncommitted after the extensible KV "
        "final commit. Default 10.",
    )
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
            "Enable native MTP speculative decoding for the qwen36 or "
            "flashnext backend. Qwen3.6 uses its draft head; Flash-Next uses "
            "the qwen4_exp MTP path."
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
    parser.add_argument(
        "--dspark",
        action="store_true",
        help=(
            "Enable Qwen3.6/Qwen3.8 DSpark speculative decoding (qwen36 backend only). "
            "Loads a separate five-layer draft checkpoint and uses gamma=7 by "
            "default; persistent prefix caching is supported, and MTP is "
            "disabled for this path."
        ),
    )
    parser.add_argument(
        "--dspark-draft-model",
        default=SERVER_DSPARK_DRAFT_MODEL,
        help=(
            "Local path or cached Hugging Face id for the external DSpark draft. "
            f"Default: {SERVER_DSPARK_DRAFT_MODEL}."
        ),
    )
    parser.add_argument(
        "--dspark-k",
        type=int,
        default=SERVER_DSPARK_K,
        help="DSpark draft tokens per round; must match draft config block_size. Default 7.",
    )
    parser.add_argument(
        "--dflash2",
        action="store_true",
        help=(
            "Enable Qwen3.8 DFlash2 speculative decoding with an explicit local "
            "GGUF target and DFlash2 draft checkpoint. This is opt-in and is "
            "mutually exclusive with --dspark/--mtp."
        ),
    )
    parser.add_argument(
        "--dflash2-draft-model",
        default=SERVER_DFLASH2_DRAFT_MODEL,
        help=(
            "Local DFlash2 draft directory or cached Hugging Face id. "
            f"Default: {SERVER_DFLASH2_DRAFT_MODEL}."
        ),
    )
    parser.add_argument(
        "--dflash2-k",
        type=int,
        default=SERVER_DFLASH2_K,
        help="DFlash2 speculative tokens per round; block_size=8 reserves 7 proposals. Default 7.",
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

    if (args.dflash2 or SERVER_ENABLE_DFLASH2) and (args.dspark or args.mtp or args.dflash):
        parser.error("--dflash2 is mutually exclusive with --dspark, --mtp, and --dflash")

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
    if args.dspark and (args.mtp or args.dflash):
        parser.error("--dspark is mutually exclusive with --mtp and --dflash")
    if args.dspark and args.session_affinity:
        parser.error("--dspark currently does not support --session-affinity")

    os.environ["QSR_SERVER_CAPACITY"] = str(args.capacity)
    os.environ["QSR_SERVER_NUM_SLOTS"] = str(args.num_slots)
    os.environ["QSR_SERVER_BLOCKS_PER_SLOT"] = str(args.blocks_per_slot)
    os.environ["QSR_QWEN_KV_MODE"] = args.qwen_kv_mode
    os.environ["QSR_QWEN_KV_POOL_BYTES"] = str(args.qwen_kv_pool_bytes)
    os.environ["QSR_QWEN_KV_WATERMARK_BUNDLES"] = str(
        args.qwen_kv_watermark_bundles
    )
    os.environ["QSR_QWEN_KV_EXTENSIBLE"] = "1" if args.qwen_kv_extensible else "0"
    os.environ["QSR_QWEN_KV_COMMIT_BUFFER_GB"] = str(args.qwen_kv_commit_buffer_gb)
    if args.no_cudagraph:
        os.environ["QSR_SERVER_ENABLE_CUDAGRAPH"] = "0"
    if args.no_prefix_cache:
        os.environ["QSR_SERVER_ENABLE_PREFIX_CACHE"] = "0"
    if args.session_affinity:
        os.environ["QSR_SERVER_ENABLE_SESSION_AFFINITY"] = "1"
    os.environ["QSR_SERVER_SESSION_TTL_S"] = str(args.session_ttl_s)
    if args.dflash:
        os.environ["QSR_SERVER_ENABLE_DFLASH"] = "1"
        os.environ["QSR_SERVER_ENABLE_DFLASH2"] = "0"
    if args.mtp:
        os.environ["QSR_SERVER_ENABLE_MTP"] = "1"
        os.environ["QSR_SERVER_ENABLE_DSPARK"] = "0"
        os.environ["QSR_SERVER_ENABLE_DFLASH2"] = "0"
    os.environ["QSR_SERVER_MTP_K"] = str(args.mtp_k)
    if args.mtp_resync:
        os.environ["QSR_SERVER_MTP_RESYNC"] = "1"
    if args.dspark:
        os.environ["QSR_SERVER_ENABLE_DSPARK"] = "1"
        os.environ["QSR_SERVER_ENABLE_MTP"] = "0"
        os.environ["QSR_SERVER_ENABLE_DFLASH2"] = "0"
    os.environ["QSR_SERVER_DSPARK_DRAFT_MODEL"] = args.dspark_draft_model
    os.environ["QSR_SERVER_DSPARK_K"] = str(args.dspark_k)
    if args.dflash2:
        os.environ["QSR_SERVER_ENABLE_DFLASH2"] = "1"
        os.environ["QSR_SERVER_ENABLE_DSPARK"] = "0"
        os.environ["QSR_SERVER_ENABLE_MTP"] = "0"
    os.environ["QSR_SERVER_DFLASH2_DRAFT_MODEL"] = args.dflash2_draft_model
    os.environ["QSR_SERVER_DFLASH2_K"] = str(args.dflash2_k)
    os.environ["QSR_TOOL_CALL_PARSER"] = args.tool_call_parser

    # Runs before uvicorn imports the app module, so the model is not loaded
    # yet -- a fatal environment mismatch costs seconds, not a failed load.
    try:
        _GPU_PROCESS_LOCK_FD = _acquire_gpu_process_lock()
    except RuntimeError as exc:
        parser.error(str(exc))

    if not args.skip_preflight:
        _run_startup_preflight()

    uvicorn.run("server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()


@app.get("/v1/models")
async def list_models():
    configured = os.environ.get("QSR_SERVED_MODEL_NAME")
    served = configured or _default_served_model_name(engine)
    names = served.split()
    max_model_len = engine.capacity_tokens_per_slot if engine else 0
    max_output_tokens = _served_max_output_tokens(engine)
    input_token_limit = _served_input_token_limit(engine)
    speculative_tokens = int(getattr(engine, "K", 0)) if engine else 0
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen-sm120-runtime",
                "root": _default_served_model_name(engine),
                "parent": None,
                "max_model_len": max_model_len,
                "context_length": max_model_len,
                "input_token_limit": input_token_limit,
                "max_output_tokens": max_output_tokens,
                "speculative_reserved_tokens": speculative_tokens,
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
    served_model = _default_served_model_name(engine)
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
        f'blackwellm:num_requests_running{{model_name="{served_model}"}} {num_running}',
        "# HELP blackwellm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE blackwellm:num_requests_waiting gauge",
        f'blackwellm:num_requests_waiting{{model_name="{served_model}"}} {num_waiting}',
        "# HELP blackwellm:kv_cache_usage_perc KV cache usage percentage.",
        "# TYPE blackwellm:kv_cache_usage_perc gauge",
        f'blackwellm:kv_cache_usage_perc{{model_name="{served_model}"}} {kv_usage:.4f}',
        "# HELP blackwellm:num_free_slots Number of free production slots.",
        "# TYPE blackwellm:num_free_slots gauge",
        f'blackwellm:num_free_slots{{model_name="{served_model}"}} {num_free_slots}',
        "# HELP blackwellm:capacity_tokens_per_slot Max tokens per slot.",
        "# TYPE blackwellm:capacity_tokens_per_slot gauge",
        f'blackwellm:capacity_tokens_per_slot{{model_name="{served_model}"}} '
        f"{engine.capacity_tokens_per_slot}",
        "# HELP blackwellm:requests_completed_total Total completed requests.",
        "# TYPE blackwellm:requests_completed_total counter",
        f'blackwellm:requests_completed_total{{model_name="{served_model}"}} '
        f"{engine.stats.get('requests_completed', 0)}",
        "# HELP blackwellm:prefix_cache_hit_rate Prefix cache hit rate.",
        "# TYPE blackwellm:prefix_cache_hit_rate gauge",
        f'blackwellm:prefix_cache_hit_rate{{model_name="{served_model}"}} '
        f"{engine.stats.get('prefix_cache_hit_rate', 0.0):.4f}",
        "# HELP blackwellm:prefix_cache_hits_total Prefix cache hits.",
        "# TYPE blackwellm:prefix_cache_hits_total counter",
        f'blackwellm:prefix_cache_hits_total{{model_name="{served_model}"}} '
        f"{engine.stats.get('prefix_cache_hits', 0)}",
        "# HELP blackwellm:prefix_cache_misses_total Prefix cache misses.",
        "# TYPE blackwellm:prefix_cache_misses_total counter",
        f'blackwellm:prefix_cache_misses_total{{model_name="{served_model}"}} '
        f"{engine.stats.get('prefix_cache_misses', 0)}",
        "# HELP blackwellm:kv_cache_total_blocks Total KV cache blocks.",
        "# TYPE blackwellm:kv_cache_total_blocks gauge",
        f'blackwellm:kv_cache_total_blocks{{model_name="{served_model}"}} {total_blocks}',
        "# HELP blackwellm:kv_cache_used_blocks Used KV cache blocks.",
        "# TYPE blackwellm:kv_cache_used_blocks gauge",
        f'blackwellm:kv_cache_used_blocks{{model_name="{served_model}"}} {used_blocks}',
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
        "captured (1) or degraded to its eager fallback (0); for Flash-Next "
        "gdn_projections the fast batched projection contract reports 1."
    )
    lines.append("# TYPE blackwellm:dflash_cg_captured gauge")
    cg_status = snapshot.dflash_cg_status if snapshot is not None else ()
    for graph_name, status in cg_status:
        # Flash-Next surfaces the GDN projection execution CONTRACT as a mode
        # string ("batched_bf16"/"batched_quantized"/"per_row"/"disabled")
        # instead of a capture verdict (runtime/backends/flashnext.py sets
        # _cg_status["gdn_projections"] to the mode, never to "captured").
        # Any "batched*" mode is the fast path; only "captured" or "batched*"
        # count as not-degraded here.
        captured = 1 if status == "captured" or str(status).startswith("batched") else 0
        lines.append(
            f'blackwellm:dflash_cg_captured{{model_name="{served_model}",graph="{graph_name}"}} '
            f"{captured}"
        )

    snapshot_stats = dict(snapshot.runtime_stats) if snapshot is not None else {}
    for stat_name in (
        "decode_graph_capture_attempts",
        "decode_graph_capture_successes",
        "decode_graph_capture_failures",
        "decode_graph_replays",
        "decode_eager_fallbacks",
    ):
        lines.append(f"# HELP blackwellm:{stat_name}_total Backend {stat_name.replace('_', ' ')}.")
        lines.append(f"# TYPE blackwellm:{stat_name}_total counter")
        lines.append(
            f'blackwellm:{stat_name}_total{{model_name="{served_model}"}} '
            f"{snapshot_stats.get(stat_name, 0)}"
        )
    lines.append(
        "# HELP blackwellm:decode_eager_fallback_reason_total "
        "Decode tokens using eager fallback, by reason."
    )
    lines.append("# TYPE blackwellm:decode_eager_fallback_reason_total counter")
    fallback_reasons = snapshot.cg_fallback_reasons if snapshot is not None else ()
    for reason, count in fallback_reasons:
        lines.append(
            f'blackwellm:decode_eager_fallback_reason_total{{model_name="{served_model}",'
            f'reason="{reason}"}} {count}'
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
        f'blackwellm:bootstrap_checks_ok_total{{model_name="{served_model}"}} '
        f"{engine.stats.get('bootstrap_checks_ok', 0)}"
    )
    lines.append(
        "# HELP blackwellm:bootstrap_checks_failed_total "
        "Speculative prefills diverged from reference."
    )
    lines.append("# TYPE blackwellm:bootstrap_checks_failed_total counter")
    lines.append(
        f'blackwellm:bootstrap_checks_failed_total{{model_name="{served_model}"}} '
        f"{engine.stats.get('bootstrap_checks_failed', 0)}"
    )

    # App-layer request metrics: latency, TTFT, TPOT, token throughput, and
    # success/error counters (recorded per request in the handlers above).
    lines.extend(metrics.render(served_model))

    # D2: runtime-internal metrics (MTP acceptance, prefix cache depth, per-slot KV)
    lines.append(metrics.render_d2_metrics(served_model))

    # Phase 0 Qwen KV capacity gauges: load-time pool geometry, measured only
    # when the backend opted in (kv_capacity_snapshot). Absent for Laguna, so
    # this surface never claims a Qwen-only number for another backend.
    kv_capacity = getattr(runner, "kv_capacity_snapshot", None)
    if kv_capacity is not None:
        try:
            metrics.record_qwen_kv_capacity(kv_capacity())
        except Exception:  # pragma: no cover - observability must not die
            logger.exception("KV capacity snapshot failed; omitting Qwen gauges")
        _render_kv = getattr(metrics, "_render_qwen_kv_capacity")
        _render_kv(lines, served_model)

    # D3: request-level tracing stats
    lines.append(tracer.render_prometheus(served_model))

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
        "model": _default_served_model_name(engine),
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
    chat_template_kwargs = _resolve_engine_chat_template_kwargs(
        engine,
        body.get("chat_template_kwargs"),
        reasoning_effort=body.get("reasoning_effort"),
        enable_thinking=body.get("enable_thinking"),
        reasoning=body.get("reasoning"),
        thinking=body.get("thinking"),
    )
    try:
        prompt_ids, _vision_inputs = await _tokenize_multimodal_chat(
            engine,
            chat_messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    except ValueError as exc:
        raise _invalid_request(str(exc)) from exc
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

    max_tokens = _validate_and_resolve_max_tokens(body.get("max_tokens"))
    model_name = body.get("model") or _default_served_model_name(engine)
    stream = body.get("stream", False)
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
    chat_template_kwargs = _resolve_engine_chat_template_kwargs(
        engine,
        body.get("chat_template_kwargs"),
        reasoning_effort=body.get("reasoning_effort"),
        enable_thinking=body.get("enable_thinking"),
        reasoning=body.get("reasoning"),
        thinking=body.get("thinking"),
    )
    anthropic_thinking = body.get("thinking")
    thinking_token_budget = _resolve_thinking_token_budget(
        body.get("thinking_token_budget")
        if body.get("thinking_token_budget") is not None
        else (
            anthropic_thinking.get("budget_tokens")
            if isinstance(anthropic_thinking, dict)
            else None
        ),
        chat_template_kwargs,
        reasoning=body.get("reasoning"),
        thinking=anthropic_thinking,
    )
    sampling_params = _build_sampling_params(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
        defaults=_sampling_defaults_for_request(engine, chat_template_kwargs),
    )

    try:
        prompt_ids, vision_inputs = await _tokenize_multimodal_chat(
            engine,
            chat_messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    except ValueError as exc:
        raise _invalid_request(str(exc)) from exc
    await _debug_log_input("ANTHROPIC /v1/messages", body, chat_messages, prompt_ids)

    thinking_budget = await _tokenize_thinking_budget_config(engine, thinking_token_budget)
    _validate_capacity(prompt_ids, max_tokens)
    effective_max = max_tokens

    if stream:
        import json as _json

        async def _anthropic_sse():
            proc = _new_stream_processor(engine.tok, chat_template_kwargs)
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
            async for item in _submit_stream_with_thinking_budget(
                engine,
                prompt_ids,
                effective_max,
                thinking_budget=thinking_budget,
                processor=proc,
                sampling_params=sampling_params,
                cancel_ref=_cancel_ref,
                stop_sequences=stop_sequences,
                stop_on_tool_call=True,
                vision_inputs=vision_inputs,
            ):
                if await request.is_disconnected():
                    if _cancel_ref[0]:
                        engine.cancel(_cancel_ref[0])
                    return
                if isinstance(item, dict):
                    final_result = item
                    break
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
            out_tokens = (
                final_result.get("completion_tokens", len(proc.all_ids))
                if final_result
                else len(proc.all_ids)
            )
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
                out_tokens,
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
    proc = _new_stream_processor(engine.tok, chat_template_kwargs)
    result = await _submit_with_thinking_budget(
        engine,
        prompt_ids,
        effective_max,
        thinking_budget=thinking_budget,
        processor=proc,
        sampling_params=sampling_params,
        stop_sequences=stop_sequences,
        stop_on_tool_call=True,
        vision_inputs=vision_inputs,
    )
    _raw_anth = await _tokenize_decode(engine, result["committed_token_ids"])
    # Same state machine as the streaming path (server/formats/stream.py).
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

    requested_max_tokens = _validate_and_resolve_max_tokens(body.get("max_output_tokens"))
    truncation_mode = str(body.get("truncation") or "disabled")
    model_name = body.get("model") or _default_served_model_name(engine)
    stream = body.get("stream", False)
    chat_messages = responses_format.parse_input(body)
    if not chat_messages:
        raise _invalid_request("no messages provided")
    tools = convert_tools_to_chat_template(body.get("tools"))
    response_reasoning = body.get("reasoning")
    response_effort = (
        response_reasoning.get("effort") if isinstance(response_reasoning, dict) else None
    )
    chat_template_kwargs = _resolve_engine_chat_template_kwargs(
        engine,
        body.get("chat_template_kwargs"),
        reasoning_effort=body.get("reasoning_effort", response_effort),
        enable_thinking=body.get("enable_thinking"),
        reasoning=response_reasoning,
        thinking=body.get("thinking"),
    )
    responses_thinking_budget = body.get("thinking_token_budget")
    if responses_thinking_budget is None and isinstance(response_reasoning, dict):
        responses_thinking_budget = response_reasoning.get("thinking_token_budget")
        if responses_thinking_budget is None:
            responses_thinking_budget = response_reasoning.get("budget_tokens")
    thinking_token_budget = _resolve_thinking_token_budget(
        responses_thinking_budget,
        chat_template_kwargs,
        reasoning=response_reasoning,
        thinking=body.get("thinking"),
    )
    sampling_params = _build_sampling_params(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
        defaults=_sampling_defaults_for_request(engine, chat_template_kwargs),
    )
    try:
        prompt_ids, vision_inputs = await _tokenize_multimodal_chat(
            engine,
            chat_messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    except ValueError as exc:
        raise _invalid_request(str(exc)) from exc
    await _debug_log_input("OPENAI /v1/responses", body, chat_messages, prompt_ids)
    thinking_budget = await _tokenize_thinking_budget_config(engine, thinking_token_budget)
    if truncation_mode == "auto":
        max_tokens = _shrink_max_tokens_to_capacity(prompt_ids, requested_max_tokens)
    else:
        max_tokens = requested_max_tokens
        _validate_capacity(
            prompt_ids,
            max_tokens,
        )

    if stream:

        async def _responses_sse():
            proc = _new_stream_processor(engine.tok, chat_template_kwargs)
            final_result = None
            stream_error: BaseException | None = None
            first_token_t = None
            resp_id = f"resp_{uuid.uuid4().hex[:24]}"
            created_at = int(time.time())
            output_index = 0
            text_item_id = f"msg_{uuid.uuid4().hex[:24]}"
            sequence_number = 0

            def emit(event_type: str, payload: dict) -> str:
                nonlocal sequence_number
                event = responses_format.sse_event(event_type, sequence_number, payload)
                sequence_number += 1
                return event

            in_progress = responses_format.snapshot(
                resp_id,
                created_at,
                model_name,
                "in_progress",
                [],
                None,
                max_output_tokens=max_tokens,
                truncation=truncation_mode,
            )
            # Responses lifecycle events carry the full snapshot under a
            # ``response`` key with a top-level ``type`` (the OpenAI SSE
            # contract).  Sending the bare snapshot made Codex's client treat
            # the terminal events as unknown and reconnect ("stream closed
            # before response.completed").
            yield emit("response.created", {"response": in_progress})
            yield emit("response.in_progress", {"response": in_progress})
            yield emit(
                "response.output_item.added",
                {
                    "output_index": output_index,
                    "item": {
                        "id": text_item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
            yield emit(
                "response.content_part.added",
                {
                    "item_id": text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
            last_send = time.monotonic()
            _cancel_ref: list[str | None] = [None]
            try:
                async for item in _submit_stream_with_thinking_budget(
                    engine,
                    prompt_ids,
                    max_tokens,
                    thinking_budget=thinking_budget,
                    processor=proc,
                    sampling_params=sampling_params,
                    cancel_ref=_cancel_ref,
                    stop_sequences=None,
                    stop_on_tool_call=True,
                    vision_inputs=vision_inputs,
                ):
                    if await request.is_disconnected():
                        if _cancel_ref[0]:
                            engine.cancel(_cancel_ref[0])
                        return
                    if isinstance(item, dict):
                        final_result = item
                        break
                    if first_token_t is None and item:
                        first_token_t = time.perf_counter()
                    for delta in proc.drain_content():
                        yield emit(
                            "response.output_text.delta",
                            {
                                "item_id": text_item_id,
                                "output_index": output_index,
                                "content_index": 0,
                                "delta": delta,
                                "logprobs": [],
                            },
                        )
                        last_send = time.monotonic()
                    # Long reasoning-only stretches emit no delta events; an SSE
                    # comment keeps the connection alive for clients with an idle
                    # read timeout (comments are ignored by spec-compliant SSE
                    # parsers, including Codex's).
                    if time.monotonic() - last_send >= 15:
                        yield ": keepalive\n\n"
                        last_send = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Once StreamingResponse has sent HTTP 200, the global
                # exception handler cannot turn an engine failure into JSON.
                # Without an in-band terminal event the client sees a valid
                # prefix followed by EOF and commonly reports "one sentence
                # then stopped".  Preserve the partial text but terminate the
                # Responses state machine explicitly as response.failed.
                stream_error = exc
                logger.exception("Responses generation stream failed")

            if final_result is None and stream_error is None:
                # A stream that closes without its authoritative result is an
                # engine protocol violation, not a successful EOS.  Do not
                # silently convert it into response.completed.
                stream_error = RuntimeError(
                    "generation stream ended before completion metadata"
                )
                logger.error("Responses generation stream ended without a result")

            try:
                visible_text, tool_calls = proc.finalize()
            except Exception as exc:
                if stream_error is None:
                    stream_error = exc
                logger.exception("Responses output finalization failed")
                visible_text, tool_calls = "", []

            finish = "error" if stream_error is not None else final_result["finish_reason"]
            output_items = []
            text = visible_text or ""
            output_items.append(responses_format.message_item(text_item_id, text))
            # Finalization events are required even when the model spent its
            # entire allowance thinking and therefore produced no visible
            # delta.  Omitting output_text.done leaves Responses clients
            # waiting for a content item that the terminal event then closes.
            yield emit(
                "response.output_text.done",
                {
                    "item_id": text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                },
            )
            yield emit(
                "response.content_part.done",
                {
                    "item_id": text_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    },
                },
            )
            yield emit(
                "response.output_item.done",
                {"output_index": output_index, "item": output_items[-1]},
            )
            output_index += 1
            for tc in tool_calls:
                fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                fc_item = responses_format.function_call_item(
                    fc_id,
                    tc["name"],
                    json.dumps(tc["arguments"], ensure_ascii=False),
                )
                output_items.append(fc_item)
                yield emit(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {
                            "id": fc_id,
                            "type": "function_call",
                            "call_id": fc_item["call_id"],
                            "name": tc["name"],
                            "arguments": "",
                        },
                    },
                )
                yield emit(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": fc_id,
                        "output_index": output_index,
                        "delta": fc_item["arguments"],
                    },
                )
                yield emit(
                    "response.function_call_arguments.done",
                    {
                        "item_id": fc_id,
                        "output_index": output_index,
                        "name": fc_item["name"],
                        "arguments": fc_item["arguments"],
                    },
                )
                yield emit(
                    "response.output_item.done",
                    {"output_index": output_index, "item": fc_item},
                )
                output_index += 1

            usage = responses_format.build_usage(
                len(prompt_ids),
                (
                    final_result.get("completion_tokens", len(proc.all_ids))
                    if final_result
                    else len(proc.all_ids)
                ),
                final_result.get("prefix_cache_hit_tokens", 0) if final_result else 0,
            )
            status, incomplete_details = responses_format.terminal_status(finish)
            response_error = (
                {
                    "code": "server_error",
                    "message": "The model failed to generate a response.",
                }
                if stream_error is not None
                else None
            )
            completed = responses_format.snapshot(
                resp_id,
                created_at,
                model_name,
                status,
                output_items,
                usage,
                max_output_tokens=max_tokens,
                incomplete_details=incomplete_details,
                error=response_error,
                truncation=truncation_mode,
            )
            completion_tokens = (
                final_result.get("completion_tokens", len(proc.all_ids))
                if final_result
                else len(proc.all_ids)
            )
            if stream_error is None:
                metrics.record_request(
                    "responses",
                    len(prompt_ids),
                    completion_tokens,
                    finish,
                    time.perf_counter() - t0,
                    (first_token_t - t0) if first_token_t is not None else None,
                )
            else:
                metrics.record_error("responses", 500)
            await _debug_log_stream_output(
                "OPENAI /v1/responses", proc, visible_text, tool_calls, finish
            )
            terminal_event = (
                "response.failed"
                if status == "failed"
                else "response.incomplete"
                if status == "incomplete"
                else "response.completed"
            )
            yield emit(terminal_event, {"response": completed})

        return StreamingResponse(
            _responses_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    proc = _new_stream_processor(engine.tok, chat_template_kwargs)
    result = await _submit_with_thinking_budget(
        engine,
        prompt_ids,
        max_tokens,
        thinking_budget=thinking_budget,
        processor=proc,
        sampling_params=sampling_params,
        stop_sequences=None,
        stop_on_tool_call=True,
        vision_inputs=vision_inputs,
    )
    raw_text = await _tokenize_decode(engine, result["committed_token_ids"])
    text = proc.content_text()
    reasoning_content = proc.reasoning_content() if SERVER_REASONING_MODE == "expose" else None
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
        max_output_tokens=max_tokens,
        truncation=truncation_mode,
    )
