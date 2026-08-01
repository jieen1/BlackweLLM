"""Loader adapter: compressed-tensors format (Track A step 6,
``docs/architecture.md`` §3.2-D / §3.5.5 step 6).

Laguna's checkpoint declares ``quantization_config.quant_method ==
"compressed-tensors"`` (verified:
``tests/test_architecture_spec.py::TestLagunaShadowAgreement::
test_quantization_is_compressed_tensors_with_fp8_kv``) -- this module holds
that format's own naming knowledge, split out of
``runtime/model/laguna_model.py`` and ``runtime/model/_weight_loading.py``
where it used to live inline. The split is two things moving apart, not one:

- the model graph's *structural* weight mapping (``stacked_params_mapping``
  in ``LagunaModelSelfBuilt.load_weights`` -- which of the checkpoint's
  separate ``q_proj``/``k_proj``/``v_proj`` shards merge into this
  runtime's single ``qkv_proj`` Linear) is Laguna's own layer shape. It has
  nothing to do with quantization format and stays where it is.
- the quantization format's own *naming* knowledge -- which suffixes never
  have a matching model Parameter for this checkpoint's real, symmetric
  per-tensor KV-cache scheme (:data:`IGNORE_WEIGHT_SUFFIXES`), and how a
  ``self_attn.{k,v}_scale`` checkpoint key maps onto
  ``SelfBuiltAttentionPlaceholder``'s ``self_attn.attn.{k,v}_scale``
  submodule nesting (:func:`remap_kv_scale_name`) -- is what moved here.

Both pieces below are behavior-preserving relocations, not rewrites: same
tuple, same function body, same call sites' effective behavior, only a new
import path. That is deliberate for this step -- the gate is "same weights,
bit-exact", and the safest way to satisfy it for a pure name-based split is
to not touch the logic that already passed it.

What did **not** move here (a scope decision, not an oversight):
``runtime/backends/laguna_sparkinfer_moe.py``'s own reading of
``weight_packed``/``weight_scale``/``weight_global_scale`` for MoE expert
weights. That is the same quantization format and the same checkpoint, but
it is a second, already-independent loading pipeline -- it bypasses
``load_weights()``/``runtime/model_loading.py`` entirely and feeds
sparkinfer's own MoE kernel prep directly (see ``laguna_model.py``'s module
docstring, "Reimplementing FusedMoE's expert-parallel dispatch is out of
scope for a Linear/Embedding phase" -- the same scoping logic applies here).
Folding it into this adapter would touch sparkinfer-kernel-prep code for no
benefit this step's gate (dense-path tensor checksums) would catch.

Not yet exercised by a second real quantization format. modelopt (Qwen3.6,
Track B / B0-2) is the second real consumer this abstraction is eventually
built to take, and it turns out to need materially different loading logic,
not just different suffix strings -- see
``notes/2026-08-02-qwen36-b0-fact-baseline.md`` §1.6-1.7 (modelopt's
``.weight`` suffix is ambiguous between NVFP4-packed / FP8-unpacked / plain
BF16 depending on ``quantization_config.quantized_layers[name]``; Laguna's
``weight_packed`` suffix is self-describing). Until that adapter exists,
everything below is proven only by continuing to serve Laguna's own
checkpoint bit-exactly, not by two formats actually diverging in a caller's
hands.
"""

from __future__ import annotations

#: compressed-tensors' checkpoint-side suffixes that never have a matching
#: model Parameter for this checkpoint's real, symmetric per-tensor
#: KV-cache scheme (see ``runtime/model/plain_attention.py``'s module
#: docstring for why e.g. ``q_scale``/zero-points are never created here to
#: match against). Line-for-line what ``LagunaModelSelfBuilt.load_weights``
#: already skipped inline before this move -- relocated, not rewritten.
IGNORE_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    ".weight_scale",
    "_weight_scale",
    ".input_scale",
    "_input_scale",
)


def remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Narrowed replacement for vLLM's ``maybe_remap_kv_scale_name``
    (vllm/model_executor/model_loader/weight_utils.py) -- only the one real
    pattern this runtime's checkpoints ever produce: a checkpoint key ending
    directly in ``.k_scale``/``.v_scale`` (e.g. ``model.layers.N.self_attn.
    k_scale``, verified directly against the real checkpoint's safetensors),
    remapped to ``...self_attn.attn.k_scale`` to match
    ``SelfBuiltAttentionPlaceholder``'s ``self.attn`` submodule nesting
    (``runtime/model/plain_attention.py``).

    Every other real pattern vLLM's version covers -- the deprecated
    ``.kv_scale`` format, ModelOpt/QKV-proj/Qwen3-MoE/NemotronH/HYV3
    checkpoint naming conventions, ``q_scale``/zero-point suffixes, MLA's
    ``mla_attn.mla_attn`` prefix -- is provably unreachable for this
    checkpoint (verified directly, not assumed: only ``self_attn.
    {k,v}_scale`` exist per layer, no ``_proj``/``qkv_proj``/etc in
    between) and intentionally not ported. If a future checkpoint needs one
    of those, this needs revisiting, not generalizing in advance -- same
    "documented checkpoint-specific assumption, fail loud if wrong" stance
    as ``runtime/loading/common.py``'s ``assert_all_params_loaded``. The
    DFlash draft model's checkpoint never has any ``k_scale``/``v_scale``
    keys at all (verified directly), so this function never even gets
    called with a matching suffix for it -- the ``name in params_dict``
    shortcut (or plain pass-through) handles every draft-model key.

    Originally ``runtime/model/_weight_loading.py::remap_kv_scale_name``;
    moved here at Track A step 6 (compressed-tensors is the format that
    actually needs this particular naming knowledge, and
    ``_weight_loading.py`` keeps only the format-agnostic
    ``default_weight_loader``). Body unchanged.
    """
    if name in params_dict:
        return name
    if name.endswith(".k_scale") or name.endswith(".v_scale"):
        prefix, _, suffix = name.rpartition(".")
        remapped = f"{prefix}.attn.{suffix}"
        return remapped if remapped in params_dict else None
    return name
