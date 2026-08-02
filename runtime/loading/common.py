"""Format-agnostic loader building blocks (Track A step 6,
``docs/architecture.md`` §3.2-D / §3.5.5 step 6).

Moved here, unchanged, from ``runtime/model_loading.py`` (阶段7/阶段8 of
notes/2026-07-27-vllm-complete-removal-implementation-plan.md), which now
imports from here instead of defining these itself. Every function below
is behavior-preserving relocation, not a rewrite -- the reason each one
belongs in the *common* layer rather than a quantization-format adapter
is explained per function, because it is not the same reason every time:

- :func:`default_torch_dtype` / :func:`iterate_safetensors_checkpoint`
  never look at a checkpoint tensor's name or quantization scheme at all --
  plain dtype context manager and plain sharded-safetensors iteration.
- :func:`assert_all_params_loaded` only compares *model Parameter* names
  (``model.named_parameters()``) against the set the loader touched -- it
  never reads a raw checkpoint key, so which quantization format produced
  the checkpoint is invisible to it.
- :func:`apply_kv_cache_scale_post_load` copies already-loaded
  ``k_scale``/``v_scale`` *Parameters* into the ``_k_scale``/``_v_scale``
  buffers attention actually reads. It never reads a raw checkpoint key
  either: by the time it runs, whatever format-specific adapter did the
  name remapping (e.g. ``runtime.loading.compressed_tensors.
  remap_kv_scale_name``) has already turned the checkpoint's own scale-key
  spelling into these two Parameter names -- names this runtime's own
  model graph chose (``runtime/model/plain_attention.py``'s
  ``SelfBuiltAttentionPlaceholder`` always creates exactly ``k_scale``/
  ``v_scale``, never anything else), not something compressed-tensors or
  modelopt imposes. That is why "KV scale post-load" is common rather than
  per-adapter, even though *which raw checkpoint key* maps to those two
  Parameters is genuinely format-specific and lives in the adapter instead.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Generator, Iterable
from pathlib import Path

import torch
from safetensors import safe_open

from runtime.model.plain_attention import SelfBuiltAttentionPlaceholder

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def default_torch_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """Verbatim port of vLLM's ``set_default_torch_dtype`` (vllm/utils/
    torch_utils.py) -- plain ``torch.set_default_dtype`` set/restore,
    nothing vLLM-specific about it."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def iterate_safetensors_checkpoint(
    model_path: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Direct safetensors reader -- replaces vLLM's ``DefaultModelLoader.
    get_all_weights`` (阶段7, vLLM removal plan). Same pattern already
    proven in this codebase (``runtime/backends/laguna_sparkinfer_moe.py``'s
    ``load_moe_layer_weights``, applied there to one MoE layer's tensors;
    here to the whole checkpoint), not a new one.

    Deliberately NOT the general vLLM loader's feature set: no HF Hub
    download (this runtime always uses an already-cached local snapshot
    path, confirmed by HF_HUB_OFFLINE=1 everywhere), no .bin/.pt format
    fallback, no multi-threading, no expert-parallel weight filtering
    (this runtime is TP=1/EP=1 always). Handles the two real checkpoint
    layouts this runtime actually has (verified directly, not assumed):
    the main model's 15 sharded files with a
    ``model.safetensors.index.json``, and the DFlash draft model's single
    unsharded ``model.safetensors`` with no index file at all.

    Memory-safety note (this is the reason DefaultModelLoader's generality
    isn't just unused complexity -- checked before writing this): the main
    checkpoint is ~67 GiB and this host has ~19 GiB RAM (see the
    "Checkpoint size... Available RAM..." log line vLLM's own loader
    prints). Iterating shard files one at a time -- opening one, yielding
    its tensors, letting it close before the next `safe_open` call -- is
    what actually keeps peak host RAM bounded to roughly one shard's
    worth, not the whole checkpoint; the streaming behavior is load-
    bearing, not incidental, so it must NOT be replaced with e.g. reading
    every shard into one dict first.

    Format-agnostic: nothing here reads ``quantization_config`` or branches
    on tensor names, so this is shared by every loader adapter rather than
    duplicated per quantization format.
    """
    model_dir = Path(model_path)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    for shard_file in shard_files:
        with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def apply_kv_cache_scale_post_load(model: torch.nn.Module) -> None:
    """Copies loaded ``k_scale``/``v_scale`` checkpoint values into the
    ``_k_scale``/``_v_scale`` buffers ``BFAttention``/
    ``SparkinferAttentionImpl``/``fused_kv_scatter`` actually read at
    runtime (see ``runtime/model/plain_attention.py``'s module docstring
    for the full read/write contract).

    Self-built replacement (阶段7) for vLLM's
    ``CompressedTensorsKVCacheMethod.process_weights_after_loading``
    (vllm/model_executor/layers/quantization/compressed_tensors/
    compressed_tensors.py:1121-1147) -- confirmed via real source, not
    the more generic ``BaseKVCacheMethod.process_weights_after_loading``
    (kv_cache.py:74), which looked plausible at first but is never
    actually invoked for this checkpoint's quantization scheme (its
    "positive/duplicate/no-scale" branching does not apply here). The
    real method does a plain reference reassignment
    (``layer._k_scale = layer.k_scale``); this does an equivalent value
    copy instead, since ``_k_scale``/``_v_scale`` are registered buffers
    here rather than being swapped for the loaded Parameter object
    itself -- same end state (real checkpoint value ends up in the
    buffer BFAttention reads), simpler to reason about than replicating
    vLLM's specific buffer/Parameter identity dance.

    Only applies to layers where ``has_checkpoint_kv_scale`` is True (the
    main model) -- the DFlash draft model's checkpoint has no
    quantization_config and never creates ``k_scale``/``v_scale`` at all
    (see plain_attention.py's module docstring), so its ``_k_scale``/
    ``_v_scale`` stay at their construction-time default of 1.0
    permanently; there is nothing to copy for those layers, not a gap.
    For layers where it IS True, safe to run unconditionally after
    ``assert_all_params_loaded`` has already passed: that assertion
    requires ``k_scale``/``v_scale`` to have received a real checkpoint
    tensor (no exception list carves them out), so by the time this runs
    they are guaranteed to hold real values, not their own construction-
    time default of 1.0.

    Common rather than per-adapter despite the vLLM class this replicates
    having "CompressedTensors" in its name: see this module's docstring for
    why -- the checkpoint-format-specific part (which raw checkpoint key
    becomes the ``k_scale``/``v_scale`` Parameter in the first place) has
    already happened by the time this function runs, in whichever adapter
    loaded the weights.
    """
    for module in model.modules():
        if isinstance(module, SelfBuiltAttentionPlaceholder) and module.has_checkpoint_kv_scale:
            module._k_scale.copy_(module.k_scale.detach().to(torch.float32))
            module._v_scale.copy_(module.v_scale.detach().to(torch.float32))


def assert_all_params_loaded(
    model: torch.nn.Module,
    loaded: set[str],
    *,
    context: str,
    expected_unloaded: frozenset[str] = frozenset(),
) -> None:
    """Fail loudly if any parameter was never touched by ``load_weights``.

    vLLM's ``DefaultModelLoader.load_weights`` has an equivalent check
    (``track_weights_loading``, gated behind ``enable_weights_track``,
    default-on for non-quantized models only -- Laguna is NVFP4-quantized
    so vLLM's own check is actually OFF for us today). We check
    unconditionally instead: a silently-uninitialized parameter is exactly
    the kind of bug that stays invisible until it corrupts an output many
    tokens later (see notes/2026-07-27-block-size-128-accept-rate-root-
    cause-CLOSED.md for what that class of bug looks like once it's
    already propagating).

    ``expected_unloaded`` carves out parameters that are legitimately never
    loaded from the checkpoint this ``model`` was built from -- the DFlash
    draft model's ``embed_tokens``/``lm_head``, shared with the target
    model and swapped in by the caller after this returns, is the one real
    user of this today. Passing nothing (the default) is the main model's
    case: no exception list needed (阶段7) -- ``SelfBuiltAttentionPlaceholder``
    (runtime/model/plain_attention.py) only creates the two Parameters this
    checkpoint's real, symmetric per-tensor KV-cache scheme actually
    provides (``k_scale``/``v_scale``) -- unlike real vLLM's
    ``CompressedTensorsKVCacheMethod.create_weights``, which also
    unconditionally creates ``q_scale``/``k_zero_point``/``v_zero_point``/
    ``q_zero_point`` (to support llm-compressor checkpoints that use
    asymmetric/per-query-head schemes this one doesn't) and therefore
    needed an exception list here for the four params no checkpoint tensor
    would ever satisfy. Every parameter this assertion can now see is
    expected to be genuinely loadable, so a real gap fails loud with no
    carve-outs to accidentally widen.

    ``context`` is a plain label (e.g. ``"load_laguna_model"``) folded into
    the error message so a failure names which caller hit it, without this
    function needing to know anything about who its callers are.

    Unified at Track A step 6 from what used to be two near-identical
    private functions in ``runtime/model_loading.py``
    (``_assert_all_params_loaded``/``_assert_draft_params_loaded``) --
    same set-arithmetic either way, ``expected_unloaded`` is the only real
    difference between the two call sites.
    """
    all_param_names = {name for name, _ in model.named_parameters()}
    missing = all_param_names - loaded - expected_unloaded
    if missing:
        raise RuntimeError(
            f"{context}: {len(missing)} parameter(s) never received "
            f"a checkpoint tensor, e.g. {sorted(missing)[:5]!r}. This means "
            "either the checkpoint is missing a tensor the model expects, "
            "or load_weights's name-mapping logic has a bug -- do not "
            "silently proceed with randomly-initialized weights."
        )


def record_checkpoint_tensor_names(
    weights: Iterable[tuple[str, torch.Tensor]],
    sink: set[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Pass ``weights`` through unchanged, recording every name into ``sink``.

    A generator rather than a list because the whole point of
    :func:`iterate_safetensors_checkpoint` is never holding the checkpoint in
    memory at once (see its own memory-safety note). Only names accumulate.

    Wrap this *after* any filtering, not before -- filtered-out tensors were
    dropped on purpose and must not be reported as unconsumed.
    """
    for name, tensor in weights:
        sink.add(name)
        yield name, tensor


def _tensor_family(name: str) -> str:
    """The trailing component that identifies what KIND of tensor this is.

    ``model.layers.0.mlp.gate_proj.input_global_scale`` -> ``input_global_scale``
    """
    return name.rsplit(".", 1)[-1]


def warn_on_unconsumed_tensor_families(
    model: torch.nn.Module,
    seen_checkpoint_names: set[str],
    *,
    context: str,
    expected_unconsumed: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Report checkpoint tensor *families* that reached no model parameter.

    :func:`assert_all_params_loaded` checks one direction only: every model
    Parameter received a checkpoint tensor. The reverse -- a checkpoint tensor
    that no Parameter consumed -- is completely silent, because an unused
    tensor simply never gets read.

    That blind spot is not hypothetical. ``CompressedTensorsNVFP4Linear``
    never created an ``input_global_scale`` Parameter, so the standard
    checkpoint's 56 activation-scale tensors were dropped without a word for
    as long as that class existed. It happened to be harmless -- the W4A16
    path does not need an activation scale -- but nothing said so at the time,
    and the next dropped scale need not be harmless. A missing scale does not
    crash; it quietly changes the arithmetic.

    Compares *families* (trailing name component) rather than full names on
    purpose: loaders remap names, so ``checkpoint_names - loaded_param_names``
    would be meaningless set arithmetic between two different naming schemes.
    A family is scheme-independent -- if a checkpoint ships any
    ``.input_global_scale`` and the model has no parameter whose name ends
    that way, nothing could have consumed it, whatever the remapping.

    **Warns rather than raises**, unlike its sibling. A parameter with no
    tensor is unambiguously a defect -- the weight is random. An unconsumed
    tensor is often legitimate: checkpoints ship tensors for features this
    runtime does not enable (MTP heads with ``enable_mtp=False``, alternate
    quantization schemes). Raising would turn "the checkpoint offers more than
    we use" into a load failure. Being able to *see* it is the whole fix.

    Returns the unconsumed families so callers/tests can assert on them.
    """
    checkpoint_families = {_tensor_family(n) for n in seen_checkpoint_names}
    param_families = {_tensor_family(n) for n, _ in model.named_parameters()}
    buffer_families = {_tensor_family(n) for n, _ in model.named_buffers()}

    unconsumed = checkpoint_families - param_families - buffer_families - expected_unconsumed
    if unconsumed:
        counts = {
            family: sum(1 for n in seen_checkpoint_names if _tensor_family(n) == family)
            for family in unconsumed
        }
        detail = ", ".join(f"{f} x{counts[f]}" for f in sorted(unconsumed))
        logger.warning(
            "%s: %d checkpoint tensor family/families reached no model parameter "
            "or buffer: %s. Nothing consumed them, so whatever they encode is not "
            "affecting this model. That is expected for features this build does "
            "not enable, and a real defect if one of them is a scale the active "
            "quantization scheme should be using.",
            context,
            len(unconsumed),
            detail,
        )
    return frozenset(unconsumed)
