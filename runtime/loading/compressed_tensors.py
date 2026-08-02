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

Not yet exercised by a second real quantization format *of Laguna's own
kind* -- ``IGNORE_WEIGHT_SUFFIXES``/:func:`remap_kv_scale_name` above are
still proven only by Laguna's own checkpoint. modelopt (Qwen3.6, Track B /
B0-2) is a sibling loader adapter, not a second consumer of this module --
see ``runtime/loading/modelopt.py`` for why it needed materially different
loading logic, not just different suffix strings (modelopt's ``.weight``
suffix is ambiguous between NVFP4-packed / FP8-unpacked / plain BF16
depending on ``quantization_config.quantized_layers[name]``; Laguna's
``weight_packed`` suffix is self-describing).

**2026-08-02, Track B "mixed-precision" adapter**: unsloth's
``unsloth/Qwen3.6-27B-NVFP4`` declares ``quant_method: "compressed-tensors"``,
``format: "mixed-precision"`` -- a *second* real compressed-tensors format,
this time landing in this module rather than modelopt.py. It is not one
payload but two, layered via ``quantization_config.config_groups``, each
group carrying its own ``format`` string:

- ``group_0``, ``format: "float-quantized"``: FP8 (E4M3) weights,
  **per-output-channel** scale (``strategy: "channel"``) -- covers every
  ``self_attn.{q,k,v,o}_proj``, ``linear_attn.{in_proj_qkv,in_proj_z,
  out_proj}``, ``lm_head``, and (see the overlap note below) ``mlp.{gate,up,
  down}_proj`` for layers 56-63 only. Checkpoint tensors: ``.weight``
  (``float8_e4m3fn``, ``[out, in]``, unpacked -- verified against real
  safetensors headers, 2026-08-02) and ``.weight_scale`` (``bfloat16``,
  ``[out, 1]`` -- one scale per output row). This is a **different physical
  layout from modelopt's own FP8** (``runtime/loading/modelopt.py``'s
  ``dequantize_fp8``): that one is a single per-*tensor* ``float32`` scalar;
  this one is per-*channel* and stored in ``bfloat16``. Using
  ``dequantize_fp8`` here would silently broadcast the wrong scale per row
  -- see :func:`dequantize_fp8_channel` below, a genuinely different
  function, not a shape-tolerant variant of the modelopt one.
- ``group_1``, ``format: "nvfp4-pack-quantized"`` -- the **same format
  string** ``SUPPORTED_QUANT_FORMATS`` already lists for Laguna, and the
  same physical *byte* layout for the packed weight and block scale:
  ``.weight_packed`` (``uint8``, ``[out, in // 2]``), ``.weight_scale``
  (``float8_e4m3fn``, ``[out, in // 16]``, block size 16, same
  min/max/mean value range as modelopt's own block scale for the identical
  real module). Covers ``mlp.{gate,up,down}_proj`` for layers 0-55.
  ``.weight_global_scale`` (``float32``, shape ``[1]``) is where the
  layouts genuinely diverge, not just in name: **it is the reciprocal of
  modelopt's ``weight_scale_2``, not the same value.** Measured directly
  (2026-08-03, after a GPU run of the naive "reuse the value as-is" version
  produced degenerate ``"!!!!!!!!!!!!"`` output): ``layers.0.mlp.gate_proj``
  has unsloth ``weight_global_scale`` = ``6624.0`` vs. nvidia
  ``weight_scale_2`` = ``0.0002`` for the same module -- ``1/6624 ≈
  0.000151``, the same order of magnitude. This matches, and is explained
  by, ``runtime/backends/laguna_sparkinfer_moe.py``'s own documented
  convention for this exact checkpoint tensor in Laguna's MoE pipeline
  (``w1_global_scale = 1/checkpoint_gs``) -- read before writing this
  adapter, but its implication for *this* dequant path (not just
  sparkinfer's kernel-side alpha) was missed on the first pass. A fourth
  tensor, ``.input_global_scale`` (``float32``, shape ``[1]``, also
  reciprocal-flavored per the same real checkpoint: ``776.0`` here vs.
  modelopt's ``input_scale`` ``0.0016`` for the analogous module), exists
  per module too -- unsloth's NVFP4 group additionally declares
  ``input_activations`` (dynamic, block-quantized, group_size 16), i.e. this
  is really a W4A4 scheme, unlike modelopt's weight-only W4A16. This adapter
  follows B1's existing "dequantize weights to BF16, ignore the activation
  side" simplification (``runtime/model/modelopt_linear.py``'s module
  docstring) and never reads ``.input_global_scale`` -- consistent with, not
  a new exception to, how ``.input_scale`` is already ignored for modelopt.
  The two-level dequant *math* itself (E2M1 LUT, block scale x global
  scale) is identical to modelopt's ``W4A16_NVFP4`` and :func:`dequantize_fp8_channel`'s
  sibling function :func:`~runtime.loading.modelopt.dequantize_nvfp4` is
  reused unchanged, not reimplemented -- but the *caller*
  (``runtime/model/compressed_tensors_linear.py``'s
  ``CompressedTensorsNVFP4Linear``) must reciprocate
  ``weight_global_scale`` before passing it in, precisely because the
  checkpoint-side calling convention is not the same as modelopt's despite
  the shared function.

**The one overlap, resolved by measurement, not by guessing a precedence
rule**: ``group_1``'s target (``re:.*mlp\\.(gate|up|down)_proj$``) matches
*every* layer's MLP, including 56-63, which ``group_0`` also explicitly
targets (``re:.*layers\\.(56|57|58|59|60|61|62|63)\\.mlp\\.(gate|up|down)_proj$``).
Checked directly against the real checkpoint's safetensors headers
(2026-08-02): layer 56's ``mlp.gate_proj`` is ``float8_e4m3fn`` ``.weight`` +
``bfloat16`` ``[17408, 1]`` ``.weight_scale`` -- no ``.weight_packed``
anywhere. FP8 wins the overlap. :class:`MixedPrecisionQuantMap` below checks
``group_0``'s (FP8) targets before ``group_1``'s (NVFP4) to match, rather
than relying on ``config_groups`` dict order (which happens to agree here,
but is not what this module's correctness rests on).

Deliberately torch-free at module scope (see the file's own tests, which
import this module with no ``pytest.importorskip("torch")`` guard): the
classification logic above (:class:`MixedPrecisionQuantMap`) is plain
string/regex work and stays that way, but :func:`dequantize_fp8_channel`
below is genuine tensor arithmetic and imports ``torch`` locally inside the
function body rather than at module scope, so importing this module never
requires torch to be installed. ``runtime/model/compressed_tensors_linear.py``
(a new, torch-heavy sibling of ``runtime/model/modelopt_linear.py``) is
where the Parameter-holding ``nn.Module`` classes that call it live.
"""

from __future__ import annotations

import re
from typing import Any

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


# ---------------------------------------------------------------------------
# "mixed-precision" sub-format (unsloth's Qwen3.6-27B-NVFP4). See module
# docstring for the measured semantics; everything below implements exactly
# that, nothing more.
# ---------------------------------------------------------------------------

#: The two ``config_groups[*].format`` strings unsloth's checkpoint actually
#: declares (verified directly against its ``config.json``, 2026-08-02). A
#: group declaring anything else is refused loudly by
#: :class:`MixedPrecisionQuantMap` rather than silently treated as
#: unquantized -- see its docstring.
MIXED_PRECISION_FORMAT_FP8_CHANNEL = "float-quantized"
MIXED_PRECISION_FORMAT_NVFP4 = "nvfp4-pack-quantized"
_KNOWN_MIXED_PRECISION_FORMATS = (
    MIXED_PRECISION_FORMAT_FP8_CHANNEL,
    MIXED_PRECISION_FORMAT_NVFP4,
)

#: Algo strings :func:`mixed_precision_quant_map` produces, consumed by
#: ``runtime/model/qwen36_model.py``'s ``_make_linear``. Deliberately
#: distinct spellings from ``runtime.loading.modelopt``'s ``QUANT_ALGO_FP8``/
#: ``QUANT_ALGO_NVFP4`` -- same math (NVFP4) or same bit width (FP8), but a
#: different physical layout each time (see module docstring), so
#: ``_make_linear`` must never confuse the two.
QUANT_ALGO_MP_FP8_CHANNEL = "mixed_precision_fp8_channel"
QUANT_ALGO_MP_NVFP4 = "mixed_precision_nvfp4"


def _compile_target(entry: str) -> re.Pattern[str] | str:
    """compressed-tensors' own convention: a ``"re:"``-prefixed target is a
    regex (matched with :func:`re.match` -- every real pattern this
    checkpoint declares already starts with ``.*`` or ``^``, so anchoring at
    position 0 only is equivalent to a search here, not a narrowing);
    anything else is an exact module-name literal (as seen in unsloth's own
    ``ignore`` list, e.g. ``"model.language_model.layers.0.linear_attn.
    in_proj_a"``)."""
    if entry.startswith("re:"):
        return re.compile(entry[len("re:") :])
    return entry


def _matches_any(name: str, patterns: list[re.Pattern[str] | str]) -> bool:
    for pattern in patterns:
        if isinstance(pattern, re.Pattern):
            if pattern.match(name):
                return True
        elif pattern == name:
            return True
    return False


class MixedPrecisionQuantMap:
    """Classifies a dotted module name against unsloth's mixed-precision
    ``quantization_config`` -- duck-types dict's ``.get(name, default)`` so
    ``runtime/model/qwen36_model.py``'s ``_make_linear`` can use one straight
    off either this or ``runtime.loading.modelopt.quantized_layers_map``'s
    plain ``dict[str, str]`` without caring which format produced it.

    A precomputed flat ``dict[str, str]`` (what ``quantized_layers_map``
    returns for modelopt) is not possible here: modelopt's checkpoint lists
    every quantized module explicitly
    (``quantization_config.quantized_layers``); compressed-tensors instead
    declares *regex* ``targets`` per ``config_groups`` entry, so which module
    names exist is never enumerated in the config at all -- only the model
    graph knows that, one dotted name at a time, exactly when it calls
    :meth:`get`. Classifying lazily like this also means correctness never
    depends on re-deriving the graph's own layer/module shape a second time
    outside it.

    ``ignore`` is checked first (unsloth's own opt-out list -- e.g.
    ``linear_attn.in_proj_a``/``in_proj_b``/``norm``, kept plain BF16), then
    the FP8 group's targets, then the NVFP4 group's -- see module docstring
    for why FP8 is checked first (the one real overlap, resolved by reading
    the real checkpoint's tensors rather than assumed from either group's
    position in ``config_groups``).
    """

    def __init__(self, quant_config: dict[str, Any]) -> None:
        self._ignore = [_compile_target(p) for p in (quant_config.get("ignore") or [])]
        groups = quant_config.get("config_groups") or {}
        fp8_targets: list[str] = []
        nvfp4_targets: list[str] = []
        for group_name, group in groups.items():
            fmt = group.get("format")
            targets = list(group.get("targets") or [])
            if fmt == MIXED_PRECISION_FORMAT_FP8_CHANNEL:
                fp8_targets.extend(targets)
            elif fmt == MIXED_PRECISION_FORMAT_NVFP4:
                nvfp4_targets.extend(targets)
            else:
                raise ValueError(
                    f"compressed-tensors mixed-precision config_groups[{group_name!r}] "
                    f"declares format {fmt!r}, which this adapter does not know how to "
                    f"load; known sub-formats are {_KNOWN_MIXED_PRECISION_FORMATS}. Failing "
                    "loudly here beats silently treating an unrecognized layout's weights "
                    "as plain unquantized BF16."
                )
        self._fp8_targets = [_compile_target(p) for p in fp8_targets]
        self._nvfp4_targets = [_compile_target(p) for p in nvfp4_targets]

    def get(self, name: str, default: str | None = None) -> str | None:
        if _matches_any(name, self._ignore):
            return default
        if _matches_any(name, self._fp8_targets):
            return QUANT_ALGO_MP_FP8_CHANNEL
        if _matches_any(name, self._nvfp4_targets):
            return QUANT_ALGO_MP_NVFP4
        return default


def mixed_precision_quant_map(config: dict[str, Any]) -> MixedPrecisionQuantMap | dict:
    """``config`` -> a classifier usable exactly like
    ``runtime.loading.modelopt.quantized_layers_map``'s return value (plain
    ``.get(name, default)``). Empty dict (never ``None``) when the checkpoint
    declares no ``mixed-precision`` quantization at all, matching that
    function's same "callers never need a None-check" contract.
    """
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return {}
    if quant_config.get("format") != "mixed-precision":
        return {}
    return MixedPrecisionQuantMap(quant_config)


def dequantize_fp8_channel(weight_fp8: Any, weight_scale: Any) -> Any:
    """Per-output-channel FP8 (E4M3) weight dequantization to BF16 --
    compressed-tensors' ``"float-quantized"``/``strategy: "channel"`` scheme
    (unsloth's Qwen3.6 checkpoint). **Not** the same layout as
    ``runtime.loading.modelopt.dequantize_fp8``, whose ``weight_scale`` is a
    single per-*tensor* scalar -- this one is one scale per output row and
    must not be conflated with it (a scalar-shaped call into that function
    against this checkpoint's weights would silently apply only the first
    row's scale to every row).

    ``weight_fp8``: ``[out, in]``, ``torch.float8_e4m3fn``, unpacked (one
    byte per element -- verified against real safetensors headers,
    2026-08-02). ``weight_scale``: ``[out, 1]`` (or anything reshaping to
    it), any float dtype -- the real checkpoint stores ``bfloat16``, unlike
    modelopt's ``float32`` scalar.

    Imports ``torch`` locally rather than at module scope -- see this
    module's docstring for why ``runtime/loading/compressed_tensors.py``
    stays importable without torch installed.
    """
    import torch

    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"expected float8_e4m3fn weight, got {weight_fp8.dtype}")
    out_dim = weight_fp8.shape[0]
    if weight_scale.numel() != out_dim:
        raise ValueError(
            f"weight_scale has {weight_scale.numel()} element(s), expected {out_dim} "
            f"(one per output channel) for weight shape {tuple(weight_fp8.shape)}"
        )
    scale = weight_scale.reshape(out_dim, 1).to(torch.float32)
    return (weight_fp8.to(torch.float32) * scale).to(torch.bfloat16)
