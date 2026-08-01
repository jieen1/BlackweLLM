"""Loader adapters, split by quantization format (Track A step 6,
``docs/architecture.md`` §3.2-D / §3.5.5 step 6).

The split:

- :mod:`runtime.loading.common` -- format-agnostic. Sharded safetensors
  streaming, the full-parameter-coverage assertion, the KV-cache-scale
  post-load copy, and the ``language_model_only`` vision-tensor filter
  (B0-1a) all live here because none of them read a checkpoint tensor
  *name* in a way that depends on which quantization library produced it
  (see each function's docstring for why, case by case -- it is not the
  same reason every time).
- :mod:`runtime.loading.compressed_tensors` -- Laguna's format. The one
  real, production-loaded implementation today.
- modelopt (Qwen3.6, Track B / B0-2) does not exist yet. Its checkpoint's
  own naming turns out to need a materially different loading strategy,
  not just different suffix strings -- see
  ``notes/2026-08-02-qwen36-b0-fact-baseline.md`` §1.6-1.7 -- so building a
  placeholder module for it now would not exercise anything real.

``runtime/model_loading.py`` is the orchestrator that imports from here; it
statically imports the compressed-tensors adapter (there is only one real
implementation to choose between today) rather than dispatching on
``runtime.model_registry``'s ``Resolution.loader`` string. That dispatch is
deferred until a second format actually needs the registry to pick between
two working adapters -- see this package's module boundary note above.
"""
