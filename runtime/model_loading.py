"""Self-built replacement for vLLM's ``get_model()`` — Phase 1, extended
阶段7 (weight-loading + post-load processing self-built too; see
runtime/kernels/rope.py and runtime/model/laguna_decoder.py for the
Linear/Embedding/RoPE side of that same phase).

See notes/2026-07-27-vllm-complete-removal-implementation-plan.md.
The ONLY path for both the main model and the DFlash draft model since
阶段1/阶段3 respectively; the ``QSR_LAGUNA_MODEL_LOADER=vllm``/
``QSR_DFLASH_MODEL_LOADER=vllm`` escape hatches back to vLLM's
``get_model()``/``load_dflash_model()`` were removed entirely (任务#46)
once 任务#45's GPU validation confirmed this self-built path bit-exact.

Remaining vLLM dependency here: ``DefaultModelLoader.get_all_weights`` and
``process_weights_after_loading`` are BOTH now replaced
(``_iterate_safetensors_checkpoint``/``_apply_kv_cache_scale_post_load``
below, 阶段7). The first attempt at the latter (calling only
``Attention.process_weights_after_loading(dtype)`` per-module) produced a
confirmed, severe regression on a real GPU e2e run (garbled text,
degenerate repetition, wrong accept rate) versus the established
bit-exact baseline, and was reverted rather than shipped half-understood.
Root-caused afterward (阶段7-补充, prompted by a coordinator challenge
to the "don't replace Attention" call this had led to -- see
notes/2026-07-27-vllm-complete-removal-implementation-plan.md's
"Attention op-dispatch ABC 结论修正"): the missing piece was never
``Attention.process_weights_after_loading`` itself (confirmed a no-op
for this model -- no attention sinks), but
``CompressedTensorsKVCacheMethod.process_weights_after_loading``
(vllm/model_executor/layers/quantization/compressed_tensors/
compressed_tensors.py:1121-1147), which is what actually copies the
loaded ``k_scale``/``v_scale`` checkpoint values into the ``_k_scale``/
``_v_scale`` buffers ``BFAttention`` reads at runtime.
``_apply_kv_cache_scale_post_load`` below replicates exactly that (see
its docstring and runtime/model/plain_attention.py's module docstring
for the full contract), and ``SelfBuiltAttentionPlaceholder``
(runtime/model/plain_attention.py) replaces ``Attention`` construction
itself in runtime/model/laguna_decoder.py. ``VllmConfig`` remains a
pure type-annotation usage in this file -- see the ``TYPE_CHECKING``
import below.

任务#41 (阶段8): ``set_default_torch_dtype`` is now self-built too
(``_default_torch_dtype`` below) -- vLLM's real version (vllm/utils/
torch_utils.py) is a 5-line ``@contextlib.contextmanager`` wrapping
``torch.set_default_dtype``/restore, nothing vLLM-specific about it.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from runtime.model.laguna_dflash_model import LagunaDraftForCausalLMSelfBuilt
from runtime.model.laguna_model import LagunaForCausalLMSelfBuilt
from runtime.model.plain_attention import SelfBuiltAttentionPlaceholder

if TYPE_CHECKING:
    # Type-annotation only -- never instantiated/isinstance-checked here,
    # so this import doesn't need to happen at runtime (verified: no
    # runtime use of the name below beyond parameter/return annotations,
    # and `from __future__ import annotations` above makes annotations
    # lazy strings anyway). Real vllm.config dependency stays only where
    # a VllmConfig instance actually gets constructed (runtime/compat_vllm.py
    # / runtime/backends/laguna.py), not here.
    from vllm.config import VllmConfig


@contextlib.contextmanager
def _default_torch_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """Verbatim port of vLLM's ``set_default_torch_dtype`` (vllm/utils/
    torch_utils.py) -- plain ``torch.set_default_dtype`` set/restore,
    nothing vLLM-specific about it."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _iterate_safetensors_checkpoint(
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


def _apply_kv_cache_scale_post_load(model: torch.nn.Module) -> None:
    """Copies loaded ``k_scale``/``v_scale`` checkpoint values into the
    ``_k_scale``/``_v_scale`` buffers ``BFAttention``/
    ``SparkinferAttentionImpl``/``fused_kv_scatter`` actually read at
    runtime (see runtime/model/plain_attention.py's module docstring for
    the full read/write contract).

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
    ``_assert_all_params_loaded`` has already passed: that assertion
    requires ``k_scale``/``v_scale`` to have received a real checkpoint
    tensor (no exception list carves them out), so by the time this runs
    they are guaranteed to hold real values, not their own construction-
    time default of 1.0.
    """
    for module in model.modules():
        if isinstance(module, SelfBuiltAttentionPlaceholder) and module.has_checkpoint_kv_scale:
            module._k_scale.copy_(module.k_scale.detach().to(torch.float32))
            module._v_scale.copy_(module.v_scale.detach().to(torch.float32))


def load_laguna_model(vllm_config: VllmConfig) -> LagunaForCausalLMSelfBuilt:
    """Construct + load a Laguna model instance without vLLM's ``get_model()``.

    Caller is responsible for the ``set_current_vllm_config(vllm_config)``
    context and distributed init, exactly as ``LagunaBackend.__init__``
    already does around its current ``get_model()`` call -- this function
    only replaces what happens *inside* that context.
    """
    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    device_config = vllm_config.device_config

    load_device = device_config.device if load_config.device is None else load_config.device
    target_device = torch.device(load_device)

    with _default_torch_dtype(model_config.dtype), target_device:
        model = LagunaForCausalLMSelfBuilt(vllm_config=vllm_config)

        loaded_param_names = model.load_weights(
            _iterate_safetensors_checkpoint(model_config.model)
        )
        _assert_all_params_loaded(model, loaded_param_names)

        _apply_kv_cache_scale_post_load(model)

    return model.eval()


def load_laguna_dflash_draft_model(
    target_model: LagunaForCausalLMSelfBuilt,
    draft_vllm_config: VllmConfig,
) -> LagunaDraftForCausalLMSelfBuilt:
    """Construct + load the DFlash draft model without vLLM's ``get_model()``
    / ``load_dflash_model()``. See runtime/model/laguna_dflash_model.py's
    module docstring for the two subtle wiring details this must get right
    (attention layer naming offset, Laguna-specific context-KV projection).

    Caller is responsible for the ``set_current_vllm_config(draft_vllm_config)``
    context, exactly as ``DFlashEngine._load_draft_model`` already does
    around its current ``load_dflash_model()`` call.
    """
    draft_model_config = draft_vllm_config.speculative_config.draft_model_config
    load_config = draft_vllm_config.load_config
    device_config = draft_vllm_config.device_config

    load_device = device_config.device if load_config.device is None else load_config.device
    target_device = torch.device(load_device)

    with _default_torch_dtype(draft_model_config.dtype), target_device:
        model = LagunaDraftForCausalLMSelfBuilt(vllm_config=draft_vllm_config)

        loaded_param_names = model.load_weights(
            _iterate_safetensors_checkpoint(draft_model_config.model)
        )
        _assert_draft_params_loaded(model, loaded_param_names)

        _apply_kv_cache_scale_post_load(model)

    # embed_tokens/lm_head are shared with the target model, not loaded from
    # this checkpoint (verified: it has no such keys at all). Matches
    # vLLM's `_should_share` (vllm/v1/worker/gpu/spec_decode/eagle/utils.py):
    # unconditional sharing, since has_own_embed_tokens/has_own_lm_head are
    # False -- this is an object-reference swap, not a weight copy.
    del model.model.embed_tokens
    model.model.embed_tokens = target_model.model.embed_tokens
    del model.lm_head
    model.lm_head = target_model.lm_head

    # One-time fused-buffer construction for context-KV precompute. Must
    # run after weight loading (and after the embed/lm_head swap, though
    # those aren't read by this step) -- see laguna_dflash_model.py.
    model.model._build_fused_kv_buffers()

    return model.eval()


def _assert_draft_params_loaded(model: torch.nn.Module, loaded: set[str]) -> None:
    """Same rationale as _assert_all_params_loaded, minus the two params
    that are legitimately never loaded from this checkpoint (shared with
    the target model instead, swapped in by the caller after this returns).
    """
    all_param_names = {name for name, _ in model.named_parameters()}
    expected_unloaded = {"model.embed_tokens.weight", "lm_head.weight"}
    missing = all_param_names - loaded - expected_unloaded
    if missing:
        raise RuntimeError(
            f"load_laguna_dflash_draft_model: {len(missing)} parameter(s) "
            f"never received a checkpoint tensor, e.g. {sorted(missing)[:5]!r}."
        )


def _assert_all_params_loaded(model: torch.nn.Module, loaded: set[str]) -> None:
    """Fail loudly if any parameter was never touched by load_weights.

    vLLM's ``DefaultModelLoader.load_weights`` has an equivalent check
    (``track_weights_loading``, gated behind ``enable_weights_track``,
    default-on for non-quantized models only -- Laguna is NVFP4-quantized
    so vLLM's own check is actually OFF for us today). We check
    unconditionally instead: a silently-uninitialized parameter is exactly
    the kind of bug that stays invisible until it corrupts an output many
    tokens later (see notes/2026-07-27-block-size-128-accept-rate-root-
    cause-CLOSED.md for what that class of bug looks like once it's
    already propagating).

    No exception list needed here (阶段7): ``SelfBuiltAttentionPlaceholder``
    (runtime/model/plain_attention.py) only creates the two Parameters
    this checkpoint's real, symmetric per-tensor KV-cache scheme actually
    provides (``k_scale``/``v_scale``) -- unlike real vLLM's
    ``CompressedTensorsKVCacheMethod.create_weights``, which also
    unconditionally creates ``q_scale``/``k_zero_point``/``v_zero_point``/
    ``q_zero_point`` (to support llm-compressor checkpoints that use
    asymmetric/per-query-head schemes this one doesn't) and therefore
    needed an exception list here for the four params no checkpoint tensor
    would ever satisfy. Every parameter this assertion can now see is
    expected to be genuinely loadable, so a real gap fails loud with no
    carve-outs to accidentally widen.
    """
    all_param_names = {name for name, _ in model.named_parameters()}
    missing = all_param_names - loaded
    if missing:
        raise RuntimeError(
            f"load_laguna_model: {len(missing)} parameter(s) never received "
            f"a checkpoint tensor, e.g. {sorted(missing)[:5]!r}. This means "
            "either the checkpoint is missing a tensor the model expects, "
            "or load_weights's name-mapping logic has a bug -- do not "
            "silently proceed with randomly-initialized weights."
        )
