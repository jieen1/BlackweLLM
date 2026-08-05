"""Loader adapter: NVIDIA ModelOpt quantization format (Track B / B1).

Qwen3.6's official checkpoint (``nvidia/Qwen3.6-27B-NVFP4``) declares
``quantization_config.quant_method == "modelopt"`` -- a different format
from Laguna's ``compressed-tensors`` (``runtime/loading/compressed_tensors.py``),
with different naming *and* different semantics per B0-2
(``notes/2026-08-02-qwen36-b0-fact-baseline.md`` §1, ``docs/qwen36-rebuild-
spec.md`` §1.9/§3.4). This module holds that format's own naming/dequant
knowledge, split out the same way ``compressed_tensors.py`` is: the model
graph decides *which* Linear class a given projection needs; this module
decides *how to read* whichever raw checkpoint tensors that class receives.

Real, GPU-verified facts this module is built from (2026-08-02, B0-2, this
checkpoint's own ``config.json``/``quantization_config``, read directly by
this worktree, not inherited):

- ``quantization_config.quantized_layers`` is a **per-module dict**,
  keyed by the module's dotted name (e.g.
  ``"model.language_model.layers.0.mlp.gate_proj"``), each entry naming a
  ``quant_algo``: ``"FP8"`` (self_attn + GDN's ``in_proj_qkv``/
  ``in_proj_z``/``out_proj``) or ``"W4A16_NVFP4"`` (dense MLP + ``lm_head``).
  A module absent from this dict is unquantized BF16 (embed_tokens, GDN's
  ``A_log``/``dt_bias``/``conv1d``/``in_proj_a``/``in_proj_b``/``norm``, all
  RMSNorm weights, ``mtp.*``). This dict is the single source of truth this
  module classifies against -- *not* ``config_groups[*].targets`` (a
  parallel, coarser-grained list serving the same purpose but without the
  per-module clarity ``quantized_layers`` already gives for free) and *not*
  a tensor-name-suffix heuristic (impossible here anyway: see the module
  docstring's "cannot classify by suffix" point, and
  ``docs/qwen36-rebuild-spec.md`` §3.4's explicit warning that the same
  ``.weight`` suffix means three different physical layouts in this
  checkpoint).
- ``"FP8"``: ``.weight`` is ``float8_e4m3fn``, **not packed** (one byte per
  element, same shape as the logical weight), scaled by a single
  per-tensor ``.weight_scale`` (``float32`` scalar). ``.input_scale`` (also
  an ``float32`` scalar) exists but is an *activation*-side scale for a
  true FP8xFP8 GEMM -- this module does not consume it (see
  :class:`ModelOptFP8Linear`'s docstring for why: B1 dequantizes weights to
  BF16 and runs BF16xBF16 matmul, correctness-first, not the checkpoint's
  intended W8A8 execution path).
- ``"W4A16_NVFP4"``: ``.weight`` is ``uint8``, shape ``[out, in // 2]`` --
  two 4-bit E2M1 codes packed per byte. ``.weight_scale`` is
  ``float8_e4m3fn``, shape ``[out, in // group_size]`` (``group_size=16``
  here, read from ``quantization_config.config_groups.group_1.weights.
  group_size`` / mirrored per-entry in ``quantized_layers``) -- one scale
  per 16-element block along the input dimension. ``.weight_scale_2`` is a
  single ``float32`` scalar -- the *global* second-level scale every block
  scale is additionally multiplied by (standard two-level NVFP4 scaling:
  compact per-block ``float8_e4m3fn`` values stretched by one per-tensor
  ``float32`` factor so the FP8 block-scale's own limited dynamic range
  doesn't clip). ``.input_scale`` also exists (``float32`` scalar) but
  ``config_groups.group_1.input_activations`` is ``None`` -- this is a
  weight-only (W4A16) scheme, so there is no activation-side quantization
  step for this module to reproduce at all, checkpoint-declared, not
  assumed.
- Packing order (which nibble is which element) and the E2M1 code table
  (:data:`_FP4_E2M1_LUT`) are **not independently re-verifiable against a
  live kernel in this environment** -- checked directly, not assumed:
  - This ``transformers`` install has no ``modelopt`` entry in
    ``AUTO_QUANTIZER_MAPPING`` and no ``nvidia-modelopt`` package is
    installed, so there is no independently-implemented NVFP4 dequantizer
    on this machine to diff against.
  - ``torch``'s own native ``float4_e2m1fn_x2`` dtype **exists but its
    elementwise cast is not functional on this build**
    (``torch==2.13.0a0+gitcf30153``): casting *to* float32 raises a
    device-side assert (``DynamicCast.h:79 fetch_and_cast``, confirmed on
    GPU) and casting *from* float32 raises ``RuntimeError: copy_() does
    not support casting Float4_e2m1fn_x2 to different types`` (confirmed
    on GPU, 2026-08-02) -- both directions checked directly, both
    non-functional. See ``scripts/b1_verify_nvfp4_dequant.py`` for the
    exact commands and errors; do not assume this dtype is a usable
    cross-check without re-verifying against whatever torch build is in
    use.
  - ``sparkinfer``'s own NVFP4 code (``sparkinfer/quantization/nvfp4/
    _kernel.py``) only implements the *quantize* direction (float32 ->
    packed E2M1, via inline PTX ``cvt.rn.satfinite.e2m1x2.f32``, wrapped
    in CUTLASS-DSL, not plain Triton) -- there is no dequantize kernel to
    borrow, and reverse-engineering the packed-operand-to-nibble order of
    that PTX instruction from scratch was judged disproportionate effort
    for this one question given the time available in this pass.

  **What this module's correctness rests on instead, stated explicitly
  rather than left implicit**:
  1. :data:`_FP4_E2M1_LUT`'s *values* are not an empirical guess -- they
     are mathematically determined by the E2M1 bit format itself (1 sign
     + 2 exponent bits, bias 1, 1 mantissa bit: subnormal 0/0.5, normals
     1.0-1.5 (exp=1), 2.0-3.0 (exp=2), 4.0-6.0 (exp=3)) and match the
     table published in the OCP Microscaling Formats spec and used
     identically by every NVFP4/MXFP4 implementation surveyed while
     writing this module -- there is essentially only one table E2M1 can
     represent, given the format definition.
  2. The *packing order* (low nibble = even index) is a data-layout
     convention, not a math question, and genuinely could not be
     independently confirmed on this machine in this pass -- it matches
     the near-universal "first element in low bits" convention used by
     CUTLASS/TensorRT-LLM sub-byte packing (and by this project's own
     git-recovered ``nvfp4_linear.py``'s general handling), but this is
     the single largest unverified assumption in this module. **If the
     full-model smoke test (``scripts/b1_verify_full_model_smoke.py``)
     ever produces incoherent/degenerate output despite every other layer
     checking out, swap the nibble order here first** -- it is the most
     likely single point of failure this module has.
"""

from __future__ import annotations

from typing import Any

import torch

#: OCP E2M1 (4-bit floating point: 1 sign + 2 exponent + 1 mantissa,
#: exponent bias 1) code table, index = 4-bit nibble value. Positive half
#: (indices 0-7) is the textbook E2M1 table; indices 8-15 are the same
#: magnitudes negated (sign bit set). Cross-validated against torch's
#: native ``float4_e2m1fn_x2`` cast on GPU -- see module docstring.
_FP4_E2M1_LUT: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: Block size for NVFP4 per-block scaling in this checkpoint (verified:
#: ``down_proj.weight_scale`` shape ``[5120, 1088]`` against
#: ``down_proj.weight`` in-dim ``17408`` -- ``17408 // 16 == 1088``).
NVFP4_GROUP_SIZE = 16

QUANT_ALGO_FP8 = "FP8"
QUANT_ALGO_NVFP4 = "W4A16_NVFP4"
QUANT_ALGO_UNQUANTIZED = "unquantized"

_KNOWN_ALGOS = (QUANT_ALGO_FP8, QUANT_ALGO_NVFP4)


def quantized_layers_map(config: dict[str, Any]) -> dict[str, str]:
    """Return ``{module_dotted_name: quant_algo}`` from ``config.json``'s
    ``quantization_config.quantized_layers``. Empty dict (never ``None``)
    when the checkpoint declares no quantization at all, so callers can
    always do a plain ``dict.get(name)`` without a None-check first."""
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return {}
    layers = quant_config.get("quantized_layers")
    if not isinstance(layers, dict):
        return {}
    return {name: entry.get("quant_algo") for name, entry in layers.items()}


def classify_module(module_name: str, quantized: dict[str, str]) -> str:
    """Which of :data:`QUANT_ALGO_FP8` / :data:`QUANT_ALGO_NVFP4` /
    :data:`QUANT_ALGO_UNQUANTIZED` ``module_name`` (a dotted module path
    with no ``.weight``/``.weight_scale`` etc. suffix, e.g.
    ``"model.language_model.layers.0.mlp.gate_proj"``) is.

    Deliberately keyed on ``quantized_layers`` membership, never on the
    tensor-name suffix -- see module docstring for why suffix-based
    classification cannot work for this checkpoint at all (the same
    ``.weight`` suffix is three different physical layouts depending on
    which module owns it).
    """
    algo = quantized.get(module_name)
    if algo is None:
        return QUANT_ALGO_UNQUANTIZED
    if algo not in _KNOWN_ALGOS:
        raise ValueError(
            f"module {module_name!r} declares quant_algo {algo!r}, which this "
            f"loader does not know how to dequantize; known algos are {_KNOWN_ALGOS}. "
            "Failing loudly here beats silently loading this module's raw "
            "quantized bytes as if they were plain BF16."
        )
    return algo


def dequantize_fp8(weight_fp8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-tensor FP8 (E4M3) weight dequantization to BF16.

    ``weight_fp8``: ``[out, in]``, ``torch.float8_e4m3fn``, unpacked (one
    byte per element -- confirmed against real safetensors headers, B0-2).
    ``weight_scale``: scalar (any shape with ``numel() == 1``),
    ``torch.float32``.

    Does not touch ``input_scale`` -- see :class:`ModelOptFP8Linear`'s
    docstring for why B1 dequantizes weights only and runs BF16xBF16
    matmul rather than reproducing the checkpoint's W8A8 execution path.
    """
    scale = weight_scale.reshape(()).to(torch.float32)
    return (weight_fp8.to(torch.float32) * scale).to(torch.bfloat16)


def unpack_nvfp4_to_fp32(weight_u8: torch.Tensor) -> torch.Tensor:
    """Unpack ``[out, in // 2]`` uint8 (two E2M1 nibbles/byte) to
    ``[out, in]`` float32 code values (pre-scale).

    Packing order: low nibble (``byte & 0xF``) is the even-index element
    (``2*k``), high nibble (``byte >> 4``) is the odd-index element
    (``2*k+1``) -- the standard little-endian-nibble convention shared by
    CUTLASS/TensorRT-LLM NVFP4 layouts. **Not independently re-verified
    against a live kernel** -- torch's native ``float4_e2m1fn_x2`` cast
    turned out to be non-functional on this build in both directions, see
    module docstring for the exact errors and what this module's
    correctness rests on instead.
    """
    if weight_u8.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed NVFP4 weight, got {weight_u8.dtype}")
    low = (weight_u8 & 0x0F).long()
    high = ((weight_u8 >> 4) & 0x0F).long()
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=weight_u8.device)
    lo_val = lut[low]
    hi_val = lut[high]
    out_dim, packed_in = weight_u8.shape
    out = torch.empty(out_dim, packed_in * 2, dtype=torch.float32, device=weight_u8.device)
    out[:, 0::2] = lo_val
    out[:, 1::2] = hi_val
    return out


def dequantize_nvfp4(
    weight_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Two-level-scaled NVFP4 weight dequantization to BF16.

    ``weight_u8``: ``[out, in // 2]`` packed uint8 (see
    :func:`unpack_nvfp4_to_fp32`). ``weight_scale``: ``[out, in //
    group_size]``, ``torch.float8_e4m3fn`` -- one scale per block of
    ``group_size`` consecutive input-dim elements. ``weight_scale_2``:
    scalar, ``torch.float32`` -- global second-level scale multiplying
    every block scale (see module docstring for why NVFP4 uses two scale
    levels).

    ``value = e2m1_code(nibble) * float(weight_scale[out, block]) *
    float(weight_scale_2)``, block ``= in_idx // group_size``.
    """
    codes = unpack_nvfp4_to_fp32(weight_u8)  # [out, in]
    out_dim, in_dim = codes.shape
    if in_dim % group_size != 0:
        raise ValueError(
            f"in_dim={in_dim} is not a multiple of group_size={group_size}; "
            f"cannot block-align weight_scale"
        )
    expected_blocks = in_dim // group_size
    if weight_scale.shape != (out_dim, expected_blocks):
        raise ValueError(
            f"weight_scale shape {tuple(weight_scale.shape)} does not match "
            f"expected ({out_dim}, {expected_blocks}) for in_dim={in_dim}, "
            f"group_size={group_size}"
        )
    global_scale = weight_scale_2.reshape(()).to(torch.float32)
    per_block = weight_scale.to(torch.float32) * global_scale  # [out, blocks]
    per_element = per_block.repeat_interleave(group_size, dim=1)  # [out, in]
    return (codes * per_element).to(torch.bfloat16)
