"""``EngineProvider``: the pluggable engine abstraction that makes bfdiag's
warm daemon developable without a GPU.

Two concrete providers:

* ``FakeEngineProvider`` -- pure Python, no torch, deterministic. Used for
  every test in this task (``tests/test_bfdiag_daemon.py``,
  ``tests/test_bfdiag_canary.py``): daemon lifecycle, concurrency,
  timeout/forced-abandon, crash recovery, and the canary-failure ->
  refuse-exec -> restart path are all exercised end to end against this
  class, never against real hardware.
* ``LagunaEngineProvider`` -- the real implementation (Laguna-S-2.1 backend
  + DFlash speculative decoding + CUDA Graph capture). Written against the
  current on-disk ``runtime/`` source (verified via direct reads, not
  guessed) but, per this task's hard no-GPU constraint, never executed.
  Every torch/runtime import is deferred into method bodies so that this
  module -- and therefore the whole bfdiag daemon package -- stays
  importable in a torch-free environment; only *calling*
  ``LagunaEngineProvider.load()`` requires torch/CUDA/the model weights.

See ``notes/2026-07-27-bfdiag-warm-daemon.md`` for the full GPU-validation
TODO list this class needs before it can be trusted in production use.

This module also draws the hot/cold boundary that the rest of bfdiag has
to respect: ``LOAD_TIME_CONFIG_KEYS`` / ``LOAD_TIME_ENV_VARS`` name the
configuration that is fixed the moment ``LagunaBackend``/``DFlashEngine``
are constructed (block/slot layout, dtype, memory budget, and a handful of
env vars read exactly once inside their ``__init__``), and
``requires_cold_restart`` is the pure function that turns "does this
sweep/reconfiguration actually need a fresh process" into a yes/no list of
keys, instead of a silent, wrong measurement in an engine that never
actually picked up the change.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_DEFAULT_LAGUNA_MODEL_PATH = (
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
_DEFAULT_DSV4_MODEL_PATH = (
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/"
    "DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
_DEFAULT_DSV4_TOKENIZER_PATH = str(
    Path(__file__).resolve().parents[2] / "notes" / "dsv4flash-ref"
)

#: LagunaEngineProvider constructor kwargs that are fixed the instant
#: LagunaBackend/DFlashEngine are constructed -- changing any of these on a
#: live daemon requires a fresh process (``bf daemon stop`` + ``start``, or
#: ``bf run --cold``), never a hot re-``load()``.
LOAD_TIME_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "model_path",
        "num_slots",
        "blocks_per_slot",
        "block_size",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "dflash_model_path",
        "tokenizer_path",
        "prefill_rows",
        "enable_cudagraph",
    }
)

#: Environment variables confirmed (by reading the current runtime/
#: source, not guessed) to be read exactly once inside
#: LagunaBackend.__init__/DFlashEngine.__init__ -- runtime/backends/
#: laguna.py:305 (QSR_PREFILL_CHUNK), :342 (QSR_DECODE_CUDA_GRAPH);
#: runtime/backends/laguna_dflash.py:168 (QSR_DFLASH_CUDA_GRAPH), :384
#: (QSR_VERIFY_CUDA_GRAPH), and (added alongside the C-1 capacity fix, see
#: notes/2026-08-01-c1-c2-gpu-investigation.md) QSR_DFLASH_REQUIRE_CG --
#: read once into DFlashEngine._require_cg, governs whether a CUDA Graph
#: capture failure refuses to finish construction or degrades to that
#: path's eager fallback. Defaults to "1" (refuse to start): the eager
#: verify fallback, while capacity-correct, was found to diverge from the
#: CG-verify path's real output at kv_len>=400 (not yet root-caused, see
#: notes/2026-08-02-eager-verify-cg-verify-divergence.md) -- see
#: DFlashEngine.__init__'s comment on self._require_cg for the live
#: reasoning. -- and QSR_DFLASH_DEBUG_FORCE_CG_FAIL, a debug-only fault
#: injector (comma-separated subset of "verify","draft","decode") read
#: once into DFlashEngine._debug_force_cg_fail, never set outside
#: diagnosis. Setting any of these on an already-loaded hot daemon has NO
#: effect on the running engine -- see queue.py's sweep guard, which
#: refuses to silently sweep one of these through a hot daemon and produce
#: measurements that never actually changed anything.
LOAD_TIME_ENV_VARS: frozenset[str] = frozenset(
    {
        "QSR_PREFILL_CHUNK",
        "QSR_DECODE_CUDA_GRAPH",
        "QSR_DFLASH_CUDA_GRAPH",
        "QSR_VERIFY_CUDA_GRAPH",
        "QSR_DFLASH_REQUIRE_CG",
        "QSR_DFLASH_DEBUG_FORCE_CG_FAIL",
    }
)

_MISSING = object()

# Modules whose hot-path methods can be rebound on an already-loaded Laguna
# instance.  Keep the dependency order: leaf helpers first, then the backend
# and engine methods that import them at call time.
HOT_RELOAD_MODULES: tuple[str, ...] = (
    "bfdiag.workloads",
    "runtime.backends.laguna_sparkinfer_moe",
    "runtime.backends.laguna_sparkinfer_attn",
    "runtime.backends.laguna_cuda_graph",
    "runtime.backends.laguna_dflash_cudagraph",
    "runtime.backends.laguna_dflash",
    "runtime.backends.laguna",
)


def _rebind_instance_class(instance: Any, replacement: type[Any], label: str) -> bool:
    """Switch a live no-slots instance to its reloaded implementation class.

    CUDA tensors, loaded weights, and captured graph buffers remain owned by
    the original Python object.  Only its method dispatch changes.  A class
    layout mismatch is intentionally a hard error: silently retaining an old
    implementation after a source edit would make a performance experiment
    report the wrong code revision.
    """
    if instance is None:
        return False
    try:
        instance.__class__ = replacement
    except TypeError as exc:
        raise RuntimeError(
            f"hot reload cannot rebind {label} to {replacement.__module__}."
            f"{replacement.__name__}; restart the daemon to apply this edit"
        ) from exc
    return True


def _rebind_verify_graphs(engine: Any, replacement: type[Any]) -> list[str]:
    """Rebind the fixed M=16 and bounded-final verify graph objects."""
    rebound: list[str] = []
    if _rebind_instance_class(
        getattr(engine, "_verify_cg", None), replacement, "engine._verify_cg"
    ):
        rebound.append("engine._verify_cg")
    for num_tokens, verify_cg in getattr(engine, "_partial_verify_cgs", {}).items():
        label = f"engine._partial_verify_cgs[{num_tokens}]"
        if _rebind_instance_class(verify_cg, replacement, label):
            rebound.append(label)
    return rebound


def requires_cold_restart(
    current_cfg: dict[str, Any],
    requested_cfg: dict[str, Any],
    locked_keys: frozenset[str] | None = None,
) -> list[str]:
    """Pure function: which keys in ``locked_keys`` (default
    ``LOAD_TIME_CONFIG_KEYS``) differ between ``current_cfg`` (e.g. a
    running daemon's ``describe()["load_config"]``) and ``requested_cfg``
    (e.g. the config a new ``bf daemon start`` invocation asked for)?

    A non-empty return means "you cannot hot-swap this into the running
    engine -- it needs a cold restart" (stop the daemon and start a new
    one, or use ``bf run --cold``). An empty return means every locked key
    already matches, so reusing the running daemon as-is is safe.
    """
    keys = locked_keys if locked_keys is not None else LOAD_TIME_CONFIG_KEYS
    changed = [
        key
        for key in keys
        if (key in current_cfg or key in requested_cfg)
        and current_cfg.get(key, _MISSING) != requested_cfg.get(key, _MISSING)
    ]
    return sorted(changed)


@runtime_checkable
class EngineProvider(Protocol):
    """Minimal lifecycle an engine backend must implement to plug into the
    bfdiag warm daemon.

    ``load()`` performs the one-time (per daemon process) expensive setup:
    model weights, draft model, CUDA Graph capture, autotune caches, etc.
    ``reset()`` must return the engine to a pristine post-load state -- see
    ``bfdiag/daemon/session.py`` for the concrete checklist the real engine
    needs. ``describe()`` is a small JSON-safe dict for ``bf daemon
    status``. ``is_healthy()`` is a cheap, side-effect-free liveness probe
    (no forward pass) checked before/after requests.

    Two more methods round out the contract that ``canary.py``/``server.py``
    actually depend on: ``generate()`` (fixed-prompt greedy decoding, used
    by the canary self-check and available to ad-hoc diagnostic code) and
    ``namespace()`` (the bindings injected into exec'd code's globals).

    Providers backed by CUDA must also opt out of in-process recovery after
    a taint.  A timed-out Python thread can still be executing a CUDA call;
    swapping in a second provider in that process would let two runtimes
    contend for the same GPU and cannot reclaim the first CUDA context.
    """

    def load(self, *, on_stage: Callable[[str], None] | None = None) -> None:
        """One-time expensive setup. Must be safe to call exactly once per
        process; the daemon never calls it twice on the same instance."""
        ...

    def reset(self) -> None:
        """Return to a pristine post-load state. Must be safe to call
        before the very first ``exec`` (right after ``load()``) and
        between every pair of experiments."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-safe status dict (kind, model identity, slot config, ...)."""
        ...

    def is_healthy(self) -> bool:
        """Cheap liveness probe; no forward pass, no GPU synchronize."""
        ...

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        temperature: float = 0.0,
    ) -> list[int]:
        """Greedy-by-default token generation. Used by the canary
        self-check (fixed prompt, fixed step count) and available to ad-hoc
        exec'd diagnostic code via ``namespace()['provider']``."""
        ...

    def namespace(self) -> dict[str, Any]:
        """Bindings injected into the globals of ``exec``'d diagnostic
        code, e.g. ``{"engine": ..., "backend": ..., "provider": self}``."""
        ...

    def unload(self) -> None:
        """Release the loaded engine and its GPU memory. Called by the
        daemon's idle-TTL watchdog before the process exits, so the GPU is
        actually handed back rather than merely left idle-but-allocated
        until whatever eventually kills the process gets around to it."""
        ...

    def memory_snapshot(self) -> dict[str, Any]:
        """JSON-safe GPU memory snapshot (allocated/reserved bytes,
        alloc-retry count, a fragmentation indicator, ...), taken by the
        daemon immediately before and after every ``exec``. A tok/s number
        from a long-lived hot daemon is not comparable to another one
        without this: the caching allocator's layout drifts as more
        experiments run -- see notes/2026-07-27-bfdiag-warm-daemon.md's
        memory-visibility section."""
        ...


class FakeEngineProvider:
    """Deterministic, pure-Python, CPU-only fake engine.

    Simulates just enough "engine" behavior to exercise the daemon's full
    lifecycle end to end without GPU/torch: ``generate()`` is a pure
    function of ``(prompt_ids, max_tokens, an internal _dirty counter)``.
    ``_dirty`` models exactly the kind of leftover per-slot/KV-cache state
    this whole task exists to guard against -- it only changes via
    ``pollute()`` (what a careless experiment does) or ``reset()``/a fresh
    instance (what a clean run / daemon restart guarantees). A canary check
    that runs ``generate()`` before and after a ``pollute()`` call will,
    correctly, observe two different token sequences.
    """

    # The fake provider has no external device state or abandoned CUDA work,
    # so its restart-on-taint tests can safely exercise the generic path.
    allow_in_process_recovery_after_taint = True

    def __init__(
        self,
        *,
        model_revision: str = "fake-v1",
        num_slots: int = 1,
        load_delay_s: float = 0.0,
        fail_health_after: int | None = None,
        crash_on_reset: bool = False,
    ) -> None:
        self._model_revision = model_revision
        self._num_slots = num_slots
        self._load_delay_s = load_delay_s
        self._fail_health_after = fail_health_after
        self._crash_on_reset = crash_on_reset
        self._loaded = False
        self._dirty = 0
        self._call_count = 0
        self._load_count = 0
        self._unload_count = 0

    def load(self, *, on_stage: Callable[[str], None] | None = None) -> None:
        """Signature matches ``EngineProvider.load`` exactly (see N7 in
        docs/roadmap.md / notes/2026-08-01-bfdiag-assertion-audit.md):
        before this fix, this method took no ``on_stage`` parameter at
        all, while the Protocol it is supposed to satisfy declares one and
        ``LagunaEngineProvider.load`` implements it. Every current call
        site (``server.py``, ``canary.py``) calls ``load()`` bare, so the
        mismatch was dormant -- passing ``on_stage=`` anywhere would have
        raised ``TypeError`` for this provider specifically. The fake has
        no real multi-phase loading to report sub-stages for, so it just
        calls ``on_stage`` once at the end, matching a caller's minimum
        expectation ("this stage name fires when load() finishes") without
        inventing stage names the real provider doesn't also use.
        """
        if self._load_delay_s:
            time.sleep(self._load_delay_s)
        self._loaded = True
        self._load_count += 1
        self._dirty = 0
        if on_stage is not None:
            on_stage("after_reset")

    def reset(self) -> None:
        if self._crash_on_reset:
            raise RuntimeError("FakeEngineProvider: simulated crash during reset()")
        self._dirty = 0

    def unload(self) -> None:
        self._loaded = False
        self._unload_count += 1

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "fake",
            "loaded": self._loaded,
            "model_revision": self._model_revision,
            "calls": self._call_count,
            "load_count": self._load_count,
            "unload_count": self._unload_count,
            "dirty": self._dirty,
            # Nothing about the fake provider is actually load-time locked
            # (there is no real engine underneath it), but num_slots is
            # exposed here so hot/cold-boundary tests
            # (requires_cold_restart, the daemon-reuse config check) have
            # something real to compare without needing torch/GPU.
            "load_config": {"num_slots": self._num_slots},
        }

    def memory_snapshot(self) -> dict[str, Any]:
        return {
            "kind": "fake",
            "allocated_bytes": None,
            "reserved_bytes": None,
            "num_alloc_retries": None,
            "fragmentation_ratio": None,
        }

    def is_healthy(self) -> bool:
        if not self._loaded:
            return False
        if self._fail_health_after is not None and self._call_count >= self._fail_health_after:
            return False
        return True

    def pollute(self, amount: int = 1) -> None:
        """Simulate a careless experiment leaving residual state behind (a
        stale slot counter, an un-zeroed KV row, ...). Exposed so exec'd
        test code (``namespace()['provider'].pollute()``) can dirty the
        engine the same way a real leaky diagnostic script would."""
        self._dirty += amount

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        temperature: float = 0.0,
    ) -> list[int]:
        if not self._loaded:
            raise RuntimeError("FakeEngineProvider.generate() called before load()")
        self._call_count += 1
        state = (sum(prompt_ids) if prompt_ids else 0) ^ (self._dirty * 2654435761)
        tokens: list[int] = []
        for i in range(max_tokens):
            state = (state * 1103515245 + 12345 + i) & 0x7FFFFFFF
            tokens.append(state % 32768)
        return tokens

    def namespace(self) -> dict[str, Any]:
        return {"provider": self, "engine": self}


class LagunaEngineProvider:
    """Real bfdiag engine provider: Laguna-S-2.1 backend + DFlash
    speculative decoding + CUDA Graph capture.

    GPU-only. Every torch/runtime import is deferred to inside method
    bodies so that merely importing this module (e.g. to reach
    ``FakeEngineProvider`` in the same file) never requires torch/CUDA to
    be installed. This class's code has been written and reviewed against
    the current ``runtime/`` source but, per the task's hard GPU ban, has
    NEVER been executed -- see notes/2026-07-27-bfdiag-warm-daemon.md's
    "需要 GPU 才能验证的待办清单" for the full list of what must be
    checked before trusting this in production use.
    """

    # A timeout can leave a daemon exec thread running against this
    # provider.  Do not load a second runtime in that same process: leave
    # the daemon TAINTED so `bf daemon stop` ends the process and releases
    # the old CUDA context before a fresh daemon is started.
    allow_in_process_recovery_after_taint = False

    def __init__(
        self,
        model_path: str | None = None,
        num_slots: int = 1,
        blocks_per_slot: int = 4096,
        block_size: int = 64,
        dtype: str = "bfloat16",
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.88,
        dflash_model_path: str | None = None,
    ) -> None:
        self._model_path = model_path or _DEFAULT_LAGUNA_MODEL_PATH
        self._num_slots = num_slots
        self._blocks_per_slot = blocks_per_slot
        self._block_size = block_size
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dflash_model_path = dflash_model_path
        self._backend: Any = None
        self._engine: Any = None
        self._tokenizer: Any = None

    def load(self, *, on_stage: Callable[[str], None] | None = None) -> None:
        """Load the Laguna backend + DFlash engine (weights, draft model,
        CUDA Graph capture, autotune) and leave both in a pristine state.

        NEVER RUN. See class docstring.
        """
        import os

        from transformers import AutoTokenizer

        from runtime.backends.laguna import LagunaBackend
        from runtime.backends.laguna_dflash import DFlashEngine
        from runtime.laguna_config import build_laguna_config

        model_path = os.path.expanduser(self._model_path)
        runtime_config = build_laguna_config(
            model_path,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=True,
        )
        self._backend = LagunaBackend(
            runtime_config,
            num_slots=self._num_slots,
            blocks_per_slot=self._blocks_per_slot,
            block_size=self._block_size,
        )
        if on_stage is not None:
            on_stage("after_target_backend")
        self._engine = DFlashEngine(
            self._backend,
            dflash_model_path=self._dflash_model_path,
            defer_cuda_graph_capture=on_stage is not None,
        )
        if on_stage is not None:
            on_stage("after_dflash_eager")
            self._engine.capture_cuda_graphs()
            on_stage("after_dflash_cuda_graphs")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if on_stage is not None:
            on_stage("after_tokenizer")
        # CUDA Graph capture inside DFlashEngine.__init__ warms up using
        # dummy tokens written DIRECTLY into real logical slot 0 (every CG
        # class) and the tail num_slots-batch_size..num_slots logical slots
        # (the M=1 decode CG) -- see session.RESET_CHECKLIST's "CUDA Graph
        # capture-time warmup residue" entry. load() is therefore not
        # pristine by construction; reset() as the last step here is what
        # makes it so.
        self.reset()
        if on_stage is not None:
            on_stage("after_reset")

    def reset(self) -> None:
        if self._engine is None:
            raise RuntimeError("LagunaEngineProvider.reset() called before load()")
        from bfdiag.daemon.session import reset_laguna_engine

        # A historical M=1 decode workload captures a q=1 attention graph by
        # temporarily patching the backend's attention implementations.  A
        # long-lived daemon must never carry that specialized implementation
        # into the next DFlash canary/prefill (which can have q>1).
        self._backend._unpatch_impls_for_prefill()
        reset_laguna_engine(self._engine)

    def hot_reload_code(self) -> dict[str, Any]:
        """Apply hot-path source edits without reloading model weights.

        This is deliberately narrow: it supports performance/correctness
        iteration in the already-allocated Laguna backend, DFlash engine,
        sparkinfer helpers, and CUDA-graph wrappers.  It does *not* support
        constructor/layout changes, checkpoint/model-graph changes, or
        load-time configuration changes; those require a cold daemon restart.

        A fixed greedy canary is run before and after rebinding.  This catches
        the dangerous case where an edit silently changes output while a
        throughput experiment still looks faster.  The engine is reset around
        both passes, so the canary itself cannot leak KV or draft-cache state
        into the following ``bf exec`` experiment.
        """
        if self._backend is None or self._engine is None:
            raise RuntimeError("LagunaEngineProvider.hot_reload_code() called before load()")

        from bfdiag.daemon.canary import DEFAULT_CANARY_PROMPT_IDS, DEFAULT_CANARY_STEPS

        before = self.generate(DEFAULT_CANARY_PROMPT_IDS, DEFAULT_CANARY_STEPS)
        self.reset()
        importlib.invalidate_caches()
        modules = {
            name: importlib.reload(importlib.import_module(name)) for name in HOT_RELOAD_MODULES
        }

        rebound: list[str] = []
        if _rebind_instance_class(
            self._backend, modules["runtime.backends.laguna"].LagunaBackend, "backend"
        ):
            rebound.append("backend")
        if _rebind_instance_class(
            self._engine, modules["runtime.backends.laguna_dflash"].DFlashEngine, "engine"
        ):
            rebound.append("engine")
        if _rebind_instance_class(
            getattr(self._backend, "_decode_cg", None),
            modules["runtime.backends.laguna_cuda_graph"].LagunaCudaGraphDecode,
            "backend._decode_cg",
        ):
            rebound.append("backend._decode_cg")
        rebound.extend(
            _rebind_verify_graphs(
                self._engine,
                modules["runtime.backends.laguna_cuda_graph"].LagunaCudaGraphVerify,
            )
        )
        if _rebind_instance_class(
            getattr(self._engine, "_draft_cg", None),
            modules["runtime.backends.laguna_dflash_cudagraph"].DFlashDraftCudaGraph,
            "engine._draft_cg",
        ):
            rebound.append("engine._draft_cg")

        moe_class = modules["runtime.backends.laguna_sparkinfer_moe"].SparkinferMoELayer
        for index, layer in enumerate(getattr(self._backend, "_moe_sparkinfer_layers", ())):
            _rebind_instance_class(layer, moe_class, f"backend._moe_sparkinfer_layers[{index}]")
        if getattr(self._backend, "_moe_sparkinfer_layers", ()):
            rebound.append("backend._moe_sparkinfer_layers")

        attn_module = modules["runtime.backends.laguna_sparkinfer_attn"]
        for name, layer in getattr(self._backend, "static_forward_context", {}).items():
            implementation = getattr(layer, "impl", None)
            replacement = getattr(attn_module, type(implementation).__name__, None)
            if replacement is not None and type(implementation).__module__ == attn_module.__name__:
                _rebind_instance_class(implementation, replacement, f"attention[{name}].impl")
        rebound.append("attention implementations")

        self.reset()
        after = self.generate(DEFAULT_CANARY_PROMPT_IDS, DEFAULT_CANARY_STEPS)
        if after != before:
            mismatch = next(
                index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]
            )
            raise RuntimeError(
                "hot reload canary mismatch at token "
                f"{mismatch}: before={before[mismatch]}, after={after[mismatch]}; "
                "daemon remains loaded but must not be used for performance claims"
            )
        self.reset()
        return {
            "modules": list(HOT_RELOAD_MODULES),
            "rebound": rebound,
            "canary_tokens": len(after),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "laguna",
            "loaded": self._engine is not None,
            "model_path": self._model_path,
            "model_revision": _extract_revision(self._model_path),
            "num_slots": self._num_slots,
            "blocks_per_slot": self._blocks_per_slot,
            "block_size": self._block_size,
            # See LOAD_TIME_CONFIG_KEYS: every key here is fixed at
            # construction time -- ``requires_cold_restart(this, requested)``
            # is how ``bf daemon start``'s reuse-detection tells "identical
            # config, safe to keep using this daemon" from "you're asking
            # for a different engine, stop this one first".
            "load_config": {
                "model_path": self._model_path,
                "num_slots": self._num_slots,
                "blocks_per_slot": self._blocks_per_slot,
                "block_size": self._block_size,
                "dtype": self._dtype,
                "max_model_len": self._max_model_len,
                "gpu_memory_utilization": self._gpu_memory_utilization,
                "dflash_model_path": self._dflash_model_path,
            },
            # DFlashEngine.cg_status ("verify"/"draft"/"decode" ->
            # "captured"/"failed"): a startup capture failure used to be
            # observable only by grepping logs for one exact line (see
            # notes/2026-08-01-c1-c2-gpu-investigation.md's C-1). Surfacing
            # it here means `bf daemon status` shows whether any path is
            # currently running in its (now capacity-correct, but slower)
            # eager fallback, without needing to know that line exists.
            "cg_status": dict(getattr(self._engine, "cg_status", {})) if self._engine else {},
        }

    def is_healthy(self) -> bool:
        return self._backend is not None and self._engine is not None

    def unload(self) -> None:
        """Drop the loaded engine/backend/tokenizer and release GPU
        memory. NEVER RUN -- see class docstring. Called by the daemon's
        idle-TTL watchdog right before the process exits; the whole point
        is that the GPU is actually freed at that moment rather than only
        whenever the OS eventually reclaims a dead process's memory."""
        import gc

        import torch

        self._engine = None
        self._backend = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def memory_snapshot(self) -> dict[str, Any]:
        """``torch.cuda.memory_stats()``-derived snapshot. NEVER RUN -- see
        class docstring; read together with any tok/s number this daemon
        produces, per notes/2026-07-27-bfdiag-warm-daemon.md."""
        import torch

        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        num_alloc_retries = stats.get("num_alloc_retries", 0)
        fragmentation_ratio = (reserved - allocated) / reserved if reserved else 0.0
        return {
            "kind": "laguna",
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "num_alloc_retries": num_alloc_retries,
            "fragmentation_ratio": fragmentation_ratio,
        }

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        temperature: float = 0.0,
    ) -> list[int]:
        """Fixed-prompt greedy generation via the production DFlash path.

        Deliberately does NOT call ``LagunaBackend.generate()`` (the plain,
        non-DFlash path) -- reading runtime/backends/laguna.py while
        building this task's reset checklist turned up a real bug there
        (``self._decode_cg.reset()`` calls a method that does not exist on
        ``LagunaCudaGraphDecode``, see session.RESET_CHECKLIST's last
        entry). Instead this drives ``DFlashEngine.generate_verify_only()``
        directly -- the actual production DFlash entrypoint -- with
        ``enable_prefix_cache=False`` and an explicit ``reset_slot()``
        bracket on both sides, so this call is a pure function of
        ``prompt_ids`` regardless of what any previous experiment left in
        this slot (see session.RESET_CHECKLIST's prefix-cache-reuse entry
        for why that bracket is not optional).
        """
        if temperature != 0.0:
            raise NotImplementedError(
                "LagunaEngineProvider.generate() is greedy-only (temperature=0); "
                "canary checks and fixed-step diagnostics never need sampling"
            )
        from bfdiag.daemon.session import reset_laguna_engine

        slot = self._num_slots - 1
        self._backend.reset_slot(slot)
        try:
            tokens, _stats = self._engine.generate_verify_only(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=0.0,
                slot=slot,
                enable_prefix_cache=False,
            )
        finally:
            self._backend.reset_slot(slot)
            reset_laguna_engine(self._engine)
        return tokens

    def namespace(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "engine": self._engine,
            "tokenizer": self._tokenizer,
            "provider": self,
        }


class DeepseekV4EngineProvider:
    """Real bfdiag engine provider for the DeepSeek-V4 GGUF backend.

    GPU-only. Like ``LagunaEngineProvider``, all heavy imports are deferred so
    importing ``bfdiag.daemon.provider`` stays torch-free.
    """

    allow_in_process_recovery_after_taint = False

    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        num_slots: int = 1,
        max_model_len: int = 131072,
        prefill_rows: int = 32,
        enable_cudagraph: bool = True,
    ) -> None:
        self._model_path = model_path or _DEFAULT_DSV4_MODEL_PATH
        self._tokenizer_path = tokenizer_path or _DEFAULT_DSV4_TOKENIZER_PATH
        self._num_slots = num_slots
        self._max_model_len = max_model_len
        self._prefill_rows = prefill_rows
        self._enable_cudagraph = enable_cudagraph
        self._backend: Any = None
        self._tokenizer: Any = None
        self._load_timings_s: dict[str, float] = {}

    def load(self, *, on_stage: Callable[[str], None] | None = None) -> None:
        import os

        from transformers import AutoTokenizer

        from runtime.backends.dsv4 import load_deepseek_v4_backend

        model_path = os.path.expanduser(self._model_path)
        tokenizer_path = os.path.expanduser(self._tokenizer_path)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"DeepSeek-V4 GGUF not found: {model_path}")
        if not (Path(tokenizer_path) / "tokenizer.json").is_file():
            raise FileNotFoundError(
                f"DeepSeek-V4 tokenizer dir lacks tokenizer.json: {tokenizer_path}"
            )
        load_started = time.perf_counter()
        stage_started = load_started
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._load_timings_s["tokenizer"] = time.perf_counter() - stage_started
        if on_stage is not None:
            on_stage("after_tokenizer")
        stage_started = time.perf_counter()
        self._backend = load_deepseek_v4_backend(
            model_path,
            num_slots=self._num_slots,
            max_seq_len=self._max_model_len,
            max_q_rows=self._prefill_rows,
            device="cuda",
        )
        # The fixed bfdiag workload refuses to produce a run record without
        # a concrete model identity.  Keep the annotation on the provider-
        # owned backend so ``bf exec`` users cannot accidentally omit it.
        self._backend.bfdiag_model_identity = {
            "path": str(Path(model_path).resolve()),
            "revision": _extract_revision(model_path),
        }
        self._load_timings_s["backend"] = time.perf_counter() - stage_started
        if on_stage is not None:
            on_stage("after_target_backend")
        stage_started = time.perf_counter()
        if self._enable_cudagraph:
            self._backend.capture_decode_cuda_graph()
        self._load_timings_s["cuda_graph_capture"] = time.perf_counter() - stage_started
        if on_stage is not None:
            on_stage("after_decode_cuda_graphs")
        self.reset()
        self._load_timings_s["total"] = time.perf_counter() - load_started
        if on_stage is not None:
            on_stage("after_reset")

    def reset(self) -> None:
        if self._backend is None:
            raise RuntimeError("DeepseekV4EngineProvider.reset() called before load()")
        for slot in range(self._num_slots):
            self._backend.reset_slot(slot)

    def _snapshot_fields(self) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
        if self._backend is None:
            return {}, {}, {}
        try:
            snapshot = self._backend.snapshot()
        except Exception:
            return {}, {}, {}
        return (
            dict(snapshot.dflash_cg_status),
            dict(snapshot.runtime_stats),
            dict(snapshot.cg_fallback_reasons),
        )

    def describe(self) -> dict[str, Any]:
        cg_status, runtime_stats, cg_fallback_reasons = self._snapshot_fields()
        return {
            "kind": "deepseek_v4",
            "loaded": self._backend is not None,
            "model_path": self._model_path,
            "tokenizer_path": self._tokenizer_path,
            "model_revision": _extract_revision(self._model_path),
            "num_slots": self._num_slots,
            "load_config": {
                "model_path": self._model_path,
                "tokenizer_path": self._tokenizer_path,
                "num_slots": self._num_slots,
                "max_model_len": self._max_model_len,
                "prefill_rows": self._prefill_rows,
                "enable_cudagraph": self._enable_cudagraph,
            },
            "cg_status": cg_status,
            "runtime_stats": runtime_stats,
            "cg_fallback_reasons": cg_fallback_reasons,
            "load_timings_s": dict(self._load_timings_s),
        }

    def is_healthy(self) -> bool:
        return self._backend is not None and self._tokenizer is not None

    def unload(self) -> None:
        import gc

        import torch

        self._backend = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def memory_snapshot(self) -> dict[str, Any]:
        import torch

        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        num_alloc_retries = stats.get("num_alloc_retries", 0)
        fragmentation_ratio = (reserved - allocated) / reserved if reserved else 0.0
        return {
            "kind": "deepseek_v4",
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "num_alloc_retries": num_alloc_retries,
            "fragmentation_ratio": fragmentation_ratio,
        }

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        temperature: float = 0.0,
    ) -> list[int]:
        if self._backend is None:
            raise RuntimeError("DeepseekV4EngineProvider.generate() called before load()")
        if temperature != 0.0:
            raise NotImplementedError(
                "DeepseekV4EngineProvider.generate() is greedy-only (temperature=0)"
            )
        if max_tokens <= 0:
            return []

        from runtime.sampling import SamplingParams

        slot = self._num_slots - 1
        self.reset()
        try:
            tokens = [self._backend.prefill(slot, prompt_ids)]
            params = SamplingParams(temperature=0.0)
            while len(tokens) < max_tokens:
                kv_len = self._backend.slot_state(slot).kv_len
                next_token = self._backend.decode_batch_sampled(
                    [slot],
                    [tokens[-1]],
                    [kv_len],
                    [params],
                )[0]
                tokens.append(next_token)
            return tokens
        finally:
            self.reset()

    def namespace(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "engine": self._backend,
            "tokenizer": self._tokenizer,
            "provider": self,
        }


def _extract_revision(model_path: str) -> str:
    """Best-effort HF snapshot hash from a ``.../snapshots/<hash>/`` path,
    for the canary's model-identity fingerprint. Falls back to the whole
    path when it doesn't look like a HF cache layout."""
    parts = [p for p in model_path.rstrip("/").split("/") if p]
    if len(parts) >= 2 and parts[-2] == "snapshots":
        return parts[-1]
    path = Path(model_path).expanduser()
    if path.is_file():
        stat = path.stat()
        # Hashing a tens-of-gigabytes GGUF on every diagnostic run would
        # contaminate cold-start and I/O measurements.  Size + nanosecond
        # mtime is a cheap local-file revision token; the absolute path is
        # recorded separately in ModelInfo.path.
        return f"stat:{stat.st_size}:{stat.st_mtime_ns}"
    return model_path


if __name__ == "__main__":
    fake = FakeEngineProvider(num_slots=2)
    fake.load()
    baseline = fake.generate([1, 2, 3], 8)
    print("baseline:", baseline)
    fake.pollute()
    print("polluted:", fake.generate([1, 2, 3], 8))
    fake.reset()
    print("post-reset matches baseline:", fake.generate([1, 2, 3], 8) == baseline)
    print("memory_snapshot:", fake.memory_snapshot())

    current = fake.describe()["load_config"]
    print("requires_cold_restart (same cfg):", requires_cold_restart(current, current))
    print(
        "requires_cold_restart (num_slots 2 -> 4):",
        requires_cold_restart(current, {"num_slots": 4}),
    )

    fake.unload()
    print("unloaded, is_healthy:", fake.is_healthy())
