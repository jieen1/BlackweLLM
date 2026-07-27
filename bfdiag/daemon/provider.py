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
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

_DEFAULT_LAGUNA_MODEL_PATH = (
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)


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
    """

    def load(self) -> None:
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

    def __init__(
        self,
        *,
        model_revision: str = "fake-v1",
        load_delay_s: float = 0.0,
        fail_health_after: int | None = None,
        crash_on_reset: bool = False,
    ) -> None:
        self._model_revision = model_revision
        self._load_delay_s = load_delay_s
        self._fail_health_after = fail_health_after
        self._crash_on_reset = crash_on_reset
        self._loaded = False
        self._dirty = 0
        self._call_count = 0
        self._load_count = 0

    def load(self) -> None:
        if self._load_delay_s:
            time.sleep(self._load_delay_s)
        self._loaded = True
        self._load_count += 1
        self._dirty = 0

    def reset(self) -> None:
        if self._crash_on_reset:
            raise RuntimeError("FakeEngineProvider: simulated crash during reset()")
        self._dirty = 0

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "fake",
            "loaded": self._loaded,
            "model_revision": self._model_revision,
            "calls": self._call_count,
            "load_count": self._load_count,
            "dirty": self._dirty,
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

    def __init__(
        self,
        model_path: str | None = None,
        num_slots: int = 1,
        blocks_per_slot: int = 4096,
        dtype: str = "bfloat16",
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.88,
        dflash_model_path: str | None = None,
    ) -> None:
        self._model_path = model_path or _DEFAULT_LAGUNA_MODEL_PATH
        self._num_slots = num_slots
        self._blocks_per_slot = blocks_per_slot
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dflash_model_path = dflash_model_path
        self._backend: Any = None
        self._engine: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load the Laguna backend + DFlash engine (weights, draft model,
        CUDA Graph capture, autotune) and leave both in a pristine state.

        NEVER RUN. See class docstring.
        """
        import os

        from transformers import AutoTokenizer

        from runtime.backends.laguna import LagunaBackend
        from runtime.backends.laguna_dflash import DFlashEngine
        from runtime.compat_vllm import EngineArgs

        model_path = os.path.expanduser(self._model_path)
        engine_args = EngineArgs(
            model=model_path,
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=True,
        )
        vllm_config = engine_args.create_engine_config()
        self._backend = LagunaBackend(
            vllm_config,
            num_slots=self._num_slots,
            blocks_per_slot=self._blocks_per_slot,
        )
        self._engine = DFlashEngine(self._backend, dflash_model_path=self._dflash_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # CUDA Graph capture inside DFlashEngine.__init__ warms up using
        # dummy tokens written DIRECTLY into real logical slot 0 (every CG
        # class) and the tail num_slots-batch_size..num_slots logical slots
        # (the M=1 decode CG) -- see session.RESET_CHECKLIST's "CUDA Graph
        # capture-time warmup residue" entry. load() is therefore not
        # pristine by construction; reset() as the last step here is what
        # makes it so.
        self.reset()

    def reset(self) -> None:
        if self._engine is None:
            raise RuntimeError("LagunaEngineProvider.reset() called before load()")
        from bfdiag.daemon.session import reset_laguna_engine

        reset_laguna_engine(self._engine)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "laguna",
            "loaded": self._engine is not None,
            "model_path": self._model_path,
            "model_revision": _extract_revision(self._model_path),
            "num_slots": self._num_slots,
            "blocks_per_slot": self._blocks_per_slot,
        }

    def is_healthy(self) -> bool:
        return self._backend is not None and self._engine is not None

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


def _extract_revision(model_path: str) -> str:
    """Best-effort HF snapshot hash from a ``.../snapshots/<hash>/`` path,
    for the canary's model-identity fingerprint. Falls back to the whole
    path when it doesn't look like a HF cache layout."""
    parts = [p for p in model_path.rstrip("/").split("/") if p]
    if len(parts) >= 2 and parts[-2] == "snapshots":
        return parts[-1]
    return model_path


if __name__ == "__main__":
    fake = FakeEngineProvider()
    fake.load()
    baseline = fake.generate([1, 2, 3], 8)
    print("baseline:", baseline)
    fake.pollute()
    print("polluted:", fake.generate([1, 2, 3], 8))
    fake.reset()
    print("post-reset matches baseline:", fake.generate([1, 2, 3], 8) == baseline)
