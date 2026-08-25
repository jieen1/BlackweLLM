"""Packed GGUF Linear and embedding modules for the Qwen3.8 target.

GGML bytes remain the resident parameter.  On the supported SM120/BF16 path,
forward dispatches directly to the native packed Q/K kernels; the explicit
Torch dequantizer remains an opt-in reference oracle for CPU tests and
numerical diagnosis.  The server's Q6+DFlash2 profile can instead cache BF16
matrices and release the packed payload after warmup.  The native Q8 path also
has an opt-in zero-overhead Q8_0 split layout for experiments; the checkpoint
representation and all non-native/reference paths remain unchanged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from runtime.kernels.gguf_qk import NativeGgufQK
from runtime.loading.gguf import GGUF_BLOCK_BYTES, dequantize_gguf_packed

_logger = logging.getLogger("qwen_sm120_runtime.gguf_linear")
_NATIVE_LIBRARY: NativeGgufQK | None = None
_NATIVE_GGUF_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K", "Q8_0"})
_DEQUANTIZE_WEIGHTS_ENV = "QSR_GGUF_DEQUANTIZE_WEIGHTS"
_DEQUANTIZE_TYPES_ENV = "QSR_GGUF_DEQUANTIZE_TYPES"
_DEQUANTIZE_MODULES_ENV = "QSR_GGUF_DEQUANTIZE_MODULES"
_NATIVE_PREFILL_DEQUANT_ENV = "QSR_GGUF_NATIVE_PREFILL_DEQUANT"
_NATIVE_PREFILL_DEQUANT_MAX_BYTES_ENV = "QSR_GGUF_NATIVE_PREFILL_DEQUANT_MAX_BYTES"
_NATIVE_F32_Q8_MODULES_ENV = "QSR_GGUF_NATIVE_F32_Q8_MODULES"
_NATIVE_MMQ_Q8_MODULES_ENV = "QSR_GGUF_NATIVE_MMQ_Q8_MODULES"
_NATIVE_MXFP6_W6A8_ENV = "QSR_GGUF_MXFP6_W6A8"
_NATIVE_MXFP6_ROWS_ENV = "QSR_GGUF_MXFP6_ROWS"
_NATIVE_MXFP6_MODULES_ENV = "QSR_GGUF_MXFP6_MODULES"
_NATIVE_TC_TILE_MAJOR_ENV = "QSR_GGUF_TC_TILE_MAJOR"
_NATIVE_TC_TILE_MAJOR_ROWS_ENV = "QSR_GGUF_TC_TILE_MAJOR_ROWS"
_NATIVE_TC_TILE_MAJOR_MODULES_ENV = "QSR_GGUF_TC_TILE_MAJOR_MODULES"
_Q6_ALIGNED_BLOCK_BYTES = 224
_Q6_SPLIT_DATA_BLOCK_BYTES = 208
_Q8_SPLIT_DATA_BLOCK_BYTES = 32
_BF16_ELEMENT_BYTES = 2

# A decode step fans one hidden row into several projections (Q/K/V/O, GDN
# gates, and the MLP gate/up pair).  The native Q8 path needs the activation
# quantized once per input row, not once per projection.  The cache is scoped
# by the top-level model call: keeping it global would reuse stale Q8 values
# when a graph's static input buffer is updated on the next replay.  CUDA
# Graph capture runs the Python body once, so cached workspace addresses become
# ordinary graph dependencies on replay.
_Q8_ACTIVATION_CACHE: dict[tuple[int, int, int, int, int, int], torch.Tensor] | None = None
_Q8_ACTIVATION_CACHE_STATS: list[int] | None = None


@contextmanager
def gguf_q8_activation_cache() -> Iterator[None]:
    """Reuse Q8_1 activation scratch within one GGUF model invocation."""

    global _Q8_ACTIVATION_CACHE, _Q8_ACTIVATION_CACHE_STATS
    previous = _Q8_ACTIVATION_CACHE
    previous_stats = _Q8_ACTIVATION_CACHE_STATS
    _Q8_ACTIVATION_CACHE = (
        {} if os.environ.get("QSR_GGUF_Q8_ACTIVATION_CACHE", "1").strip() != "0" else None
    )
    _Q8_ACTIVATION_CACHE_STATS = (
        [0, 0] if os.environ.get("QSR_GGUF_Q8_ACTIVATION_STATS", "0").strip() == "1" else None
    )
    try:
        yield
    finally:
        stats = _Q8_ACTIVATION_CACHE_STATS
        if stats is not None:
            _logger.info(
                "GGUF Q8 activation cache hits=%d misses=%d entries=%d",
                stats[0],
                stats[1],
                len(_Q8_ACTIVATION_CACHE or {}),
            )
        _Q8_ACTIVATION_CACHE = previous
        _Q8_ACTIVATION_CACHE_STATS = previous_stats


def _cached_q8_activation(
    native: NativeGgufQK,
    flat: torch.Tensor,
    *,
    source: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return invocation-scoped Q8 scratch for one source activation.

    ``reshape(...).contiguous()`` is allowed to allocate a fresh view/copy in
    every projection.  Using only that temporary's data pointer therefore
    misses the sharing opportunity, and allocator address reuse could also
    make a pointer-only cache key stale.  The top-level model call keeps the
    source activation alive while all projections consume it, so its Python
    identity plus in-place version is the stable scope key.  Shape/device are
    retained as guards for callers that intentionally reuse one source object
    with different flattened views.
    """

    cache = _Q8_ACTIVATION_CACHE
    if cache is None:
        return None
    device_index = flat.device.index if flat.device.index is not None else 0
    cache_source = source if source is not None else flat
    key = (
        id(cache_source),
        cache_source._version,
        flat.shape[0],
        flat.shape[1],
        device_index,
        flat.element_size(),
    )
    activation_workspace = cache.get(key)
    if activation_workspace is None:
        if _Q8_ACTIVATION_CACHE_STATS is not None:
            _Q8_ACTIVATION_CACHE_STATS[1] += 1
        activation_workspace = native.quantize_q8_1(flat)
        cache[key] = activation_workspace
    elif _Q8_ACTIVATION_CACHE_STATS is not None:
        _Q8_ACTIVATION_CACHE_STATS[0] += 1
    return activation_workspace


def _native_gguf_enabled() -> bool:
    """Return the explicit native/reference switch without changing defaults."""

    return os.environ.get("QSR_GGUF_NATIVE", "1").strip() != "0"


def _native_q8_activation_enabled() -> bool:
    """Select Q8_1 activation quantization versus direct packed GEMM."""

    return os.environ.get("QSR_GGUF_NATIVE_Q8", "1").strip() != "0"


def _native_tensor_core_enabled() -> bool:
    """Select the BF16 tensor-core decoder for large packed GEMMs.

    The tensor-core decoder is a good prefill/verify path, while its
    shape-aware Triton tile keeps the fixed-width verify work at ``BLOCK_M=8``
    and widens true batched work.  Keep the small-M decision at the call site
    so the same explicit switch can accelerate batched work without replacing
    the measured M=1 Q8 path.
    """

    return os.environ.get("QSR_GGUF_NATIVE_TC", "0").strip() != "0"


def _native_tensor_core_tile_major_enabled() -> bool:
    """Select the exact Q/K N-tile-major weight cache for measured shapes."""

    return os.environ.get(_NATIVE_TC_TILE_MAJOR_ENV, "0").strip() != "0"


def _native_tensor_core_tile_major_rows_enabled(rows: int) -> bool:
    """Restrict tile-major allocation to an explicitly measured row count."""

    raw = os.environ.get(_NATIVE_TC_TILE_MAJOR_ROWS_ENV, "8").strip()
    if not raw:
        return False
    try:
        return rows in {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError:
        _logger.warning(
            "invalid %s=%r; disabling GGUF tile-major rows",
            _NATIVE_TC_TILE_MAJOR_ROWS_ENV,
            raw,
        )
        return False


def _native_tensor_core_tile_major_module_enabled(module_name: str | None) -> bool:
    """Restrict tile-major residency to measured projection families."""

    raw = os.environ.get(_NATIVE_TC_TILE_MAJOR_MODULES_ENV, "").strip()
    if not raw:
        return True
    if module_name is None:
        return False
    name = module_name.casefold()
    return any(pattern.strip().casefold() in name for pattern in raw.split(",") if pattern.strip())


def _native_mxfp6_w6a8_enabled() -> bool:
    """Select the opt-in SM120 MX-FP6 W6A8 route for Q6_K weights.

    This path is intentionally separate from the exact packed Q6 decoder.  It
    dequantizes a Q6 row once, quantizes the resulting BF16 matrix to the
    local Sparkinfer MX-FP6 format, and then uses the native mxf8f6f4 MMA
    kernel with per-row FP8 activation quantization.  The model-level quality
    gate must pass before a profile enables it by default.
    """

    return os.environ.get(_NATIVE_MXFP6_W6A8_ENV, "0").strip() != "0"


def _native_mxfp6_rows_enabled(rows: int) -> bool:
    """Restrict MX-FP6 to explicitly measured row counts.

    The first supported shape is DFlash2's fixed M=8 verify.  Keeping the
    allowlist explicit prevents an experimental small-M quantizer from
    silently replacing the exact M=1 decode or arbitrary prefill kernels.
    """

    raw = os.environ.get(_NATIVE_MXFP6_ROWS_ENV, "8").strip().casefold()
    if not raw:
        return False
    try:
        return rows in {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError:
        _logger.warning("invalid %s=%r; disabling MX-FP6 GGUF rows", _NATIVE_MXFP6_ROWS_ENV, raw)
        return False


def _native_mxfp6_module_enabled(module_name: str | None) -> bool:
    """Restrict MX-FP6 to measured projection families during A/B tests.

    The empty value intentionally keeps the explicit feature switch process
    wide.  A comma-separated substring allowlist lets us separate MLP
    up/down, attention, and recurrent projections without baking Qwen-specific
    layer names into the dispatch code.
    """

    raw = os.environ.get(_NATIVE_MXFP6_MODULES_ENV, "").strip()
    if not raw:
        return True
    if module_name is None:
        return False
    name = module_name.casefold()
    return any(pattern.strip().casefold() in name for pattern in raw.split(",") if pattern.strip())


def _native_mmq_enabled() -> bool:
    """Select the opt-in SGLang-style K-quant MMQ verify tile."""

    return os.environ.get("QSR_GGUF_NATIVE_MMQ", "0").strip() != "0"


def _native_mmq_q8_enabled() -> bool:
    """Select the separate Q8_0 MMQ experiment after its quality gate."""

    return os.environ.get("QSR_GGUF_NATIVE_MMQ_Q8", "0").strip() != "0"


def _native_mmq_q8_module_enabled(module_name: str | None) -> bool:
    """Restrict the approximate Q8 MMQ route to named projection families.

    An empty allowlist preserves the process-wide Q8 experiment.  When set,
    entries are case-insensitive substrings of the checkpoint module name.
    This is needed because Q8 MLP projections accumulate noticeably different
    activation error from the recurrent/attention projections; callers can
    therefore gate only the shapes whose quality and kernel timing have been
    measured.
    """

    raw = os.environ.get(_NATIVE_MMQ_Q8_MODULES_ENV, "").strip()
    if not raw:
        return True
    if module_name is None:
        return False
    name = module_name.casefold()
    return any(pattern.strip().casefold() in name for pattern in raw.split(",") if pattern.strip())


def _native_mmq_q5_enabled() -> bool:
    """Select the Q5_K MMQ experiment independently from the Q6 route."""

    return os.environ.get("QSR_GGUF_NATIVE_MMQ_Q5", "0").strip() != "0"


def _native_mmq_lm_head_enabled(module_name: str | None, type_name: str) -> bool:
    """Select the wide Q8_0 LM-head MMQ route independently.

    The vocabulary projection is a different shape from the MLP matrices:
    Qwen3.8 has ``N=248320, K=5120`` and DFlash2 evaluates it at ``M=8``.
    SGLang's shared-weight MMQ tile has enough output-row reuse there to beat
    the packed tensor-core decoder, while its Q8_1 activation quantization
    remains an explicit approximation.  Keep the switch separate from the
    experimental Q8 MLP route so enabling this measured shape cannot silently
    reroute every Q8 projection.
    """

    return (
        module_name == "lm_head"
        and type_name == "Q8_0"
        and os.environ.get("QSR_GGUF_NATIVE_MMQ_LM_HEAD", "0").strip() != "0"
    )


def _native_mmq_rows_enabled(rows: int) -> bool:
    """Use MMQ only for the fixed-width DFlash2 M=8 verify graph.

    The SGLang MMQ tile is useful for this narrow verify shape, but it is not
    a general prefill kernel.  In particular, routing a 4K prefill through
    MMQ adds quantization and DP4A work without enough weight reuse.  Ragged
    tail verifies also stay on the tensor-core path until they have a tuned
    tile of their own.
    """

    return _native_mmq_enabled() and rows == 8


def _native_mmq_q8_rows_enabled(rows: int) -> bool:
    """Allow the Q8-only experiment to run without enabling Q6/Q5 MMQ."""

    return rows == 8 and (_native_mmq_enabled() or _native_mmq_q8_enabled())


def _native_mmq_shape_enabled(output_size: int, input_size: int) -> bool:
    """Use MMQ only for the wide MLP shapes it improves on SM120.

    SGLang's Q6_K MMQ tile amortizes its DP4A/decode work across output rows,
    but the SM120 tensor-core tile remains faster for square attention/GDN
    projections and for the wide-input MLP down projection.  The Qwen3.8
    gate/up projections are the useful shape (N=17408, K=5120); mixed Q5/Q6
    gate-up projections are still dispatched as separate linears, so the
    merged N=34816 form is only a future same-format possibility rather than a
    current production shape.
    """

    return output_size >= 16_384 and output_size >= 2 * input_size


def _native_mmq_storage_enabled(type_name: str) -> bool:
    """Return whether the tuned SM120 MMQ path supports this row storage."""

    # The ABI can validate standard Q8_0 for parity, but its 34-byte stride
    # makes every other row/block unaligned.  The measured fast route uses
    # the zero-overhead 32-byte payload + row-tail layout.
    return type_name in {"Q5_K", "Q6_K_SPLIT", "Q8_0_SPLIT"}


def _native_tensor_core_f32_enabled() -> bool:
    """Select the BF16 tensor-core decoder for large F32 GGUF matmuls.

    Qwen3.8 GGUF keeps the surrounding graph in F32 for the llama.cpp
    numerical contract, but a large prefill matmul does not benefit from the
    scalar F32 packed kernel.  The tensor-core decoder rounds only the
    activation/output boundary to BF16 and converts the result back to F32;
    decode-sized M=1 stays on the exact direct path.  The fixed-width M=8
    DFlash2 verify is large enough to amortize the conversion and uses the
    same graph-safe tensor-core path as prefill.  This is a Q6/Q-K performance
    switch, independent of the existing BF16-only diagnostic knob above.
    """

    return os.environ.get("QSR_GGUF_NATIVE_TC_F32", "1").strip() != "0"


def _native_prefill_dequant_enabled() -> bool:
    """Select transient native dequantization followed by a cuBLAS matmul.

    The packed tensor-core decoder is still the no-materialization fallback.
    This path is intentionally opt-in while its allocator and CUDA Graph
    boundaries are being validated: it creates one BF16 matrix for a single
    projection, uses it immediately, and releases it before the next layer.
    """

    return os.environ.get(_NATIVE_PREFILL_DEQUANT_ENV, "0").strip() != "0"


def _native_prefill_dequant_max_bytes() -> int:
    """Return the transient BF16 workspace limit for one projection."""

    raw = os.environ.get(_NATIVE_PREFILL_DEQUANT_MAX_BYTES_ENV, str(512 * 1024**2))
    try:
        return max(0, int(raw))
    except ValueError:
        _logger.warning(
            "invalid %s=%r; disabling transient GGUF prefill dequantization",
            _NATIVE_PREFILL_DEQUANT_MAX_BYTES_ENV,
            raw,
        )
        return 0


def _native_tensor_core_m1_enabled(*, f32: bool) -> bool:
    """Allow a measured M=1 TC A/B without changing the safe default."""

    variable = "QSR_GGUF_NATIVE_TC_F32_M1" if f32 else "QSR_GGUF_NATIVE_TC_M1"
    return os.environ.get(variable, "0").strip() != "0"


def _native_f32_q8_activation_enabled() -> bool:
    """Select the fast Q8_1 activation path for F32 M=1 decode.

    Qwen3.8's GGUF graph keeps F32 at the Python/module boundary for the
    numerical contract, while the packed Q/K kernel can execute the
    latency-sensitive single-row GEMV through Q8_1 + DP4A.  The kernel
    preserves an F32 output boundary, so this is still an intentionally
    explicit approximation switch; large F32 matrices continue to use the
    BF16 tensor-core decoder above.
    """

    return os.environ.get("QSR_GGUF_NATIVE_F32_Q8", "0").strip() != "0"


def _native_f32_q8_module_enabled(module_name: str | None) -> bool:
    """Restrict approximate F32 Q8 GEMV to an explicit module allowlist.

    An empty allowlist preserves the historical process-wide opt-in.  When
    populated, entries are case-insensitive substrings of the checkpoint
    module name, which makes role-level experiments (for example ``mlp``)
    possible without hardcoding this model's layer count or tensor names.
    """

    raw = os.environ.get(_NATIVE_F32_Q8_MODULES_ENV, "").strip()
    if not raw:
        return True
    if module_name is None:
        return False
    name = module_name.casefold()
    return any(pattern.casefold() in name for pattern in raw.split(",") if pattern.strip())


def _native_f32_gemv_bf16_enabled() -> bool:
    """Run F32 M=1 packed GEMV through the BF16 arithmetic variant.

    Qwen3.8 keeps F32 at the model boundary for its exact bring-up path, but
    the surrounding decode kernels can also consume a BF16 rounded linear
    result.  This switch is deliberately opt-in while the model-level
    top-k/output gate is being established: it changes both activation and
    decoded-weight rounding, unlike the exact F32 direct kernel.
    """

    return os.environ.get("QSR_GGUF_F32_GEMV_BF16", "0").strip() != "0"


def _native_cache_activation_enabled() -> bool:
    """Stage one exact M=1 activation row in shared memory per CTA.

    The path is an A/B switch because the input row is already heavily reused
    through L1 on some GPUs.  It is only meaningful for the exact native GEMV
    path; approximate Q8_1 GEMM has its own reusable activation workspace.
    """

    return os.environ.get("QSR_GGUF_CACHE_ACTIVATION", "0").strip() != "0"


def _native_q8_type_enabled(type_name: str) -> bool:
    """Allow selective exact routing for numerically sensitive GGUF types."""

    excluded = {
        value.strip().upper()
        for value in os.environ.get("QSR_GGUF_NATIVE_Q8_EXCLUDE", "").split(",")
        if value.strip()
    }
    return type_name.upper() not in excluded


def _resident_bf16_weights_enabled(
    type_name: str | None = None,
    module_name: str | None = None,
) -> bool:
    """Select the opt-in resident-BF16 cuBLAS path for GGUF linears.

    This is deliberately separate from the packed native switches.  It
    trades the checkpoint's compact resident representation for a cached
    BF16 matrix, then releases that module's packed storage.  The server
    enables it by default only for the explicit Q6+DFlash2 profile; direct
    module users retain the compact packed default.  ``QSR_GGUF_DEQUANTIZE_TYPES``
    narrows the experiment to a comma-separated set of GGUF formats without
    changing the all-types switch.  ``QSR_GGUF_DEQUANTIZE_MODULES`` can also
    restrict it to comma-separated substrings of the checkpoint module name;
    when both filters are present they are intersected.
    """

    selected = {
        value.strip().upper()
        for value in os.environ.get(_DEQUANTIZE_TYPES_ENV, "").split(",")
        if value.strip()
    }
    selected_modules = tuple(
        value.strip().casefold()
        for value in os.environ.get(_DEQUANTIZE_MODULES_ENV, "").split(",")
        if value.strip()
    )
    if selected and (type_name is None or type_name.upper() not in selected):
        return False
    if selected_modules and (
        module_name is None
        or not any(value in module_name.casefold() for value in selected_modules)
    ):
        return False
    if selected or selected_modules:
        return True
    return os.environ.get(_DEQUANTIZE_WEIGHTS_ENV, "0").strip() != "0"


def _q6_aligned_weights_enabled() -> bool:
    """Select the padded Q6_K storage used by the native BF16 Q8 path."""

    # The padded layout was kept as an explicit experiment after its
    # correctness gate, but the extra 6.7% payload traffic currently costs
    # more than the aligned-load win on SM120.  Standard GGUF storage is the
    # production default until a kernel profile proves otherwise.
    return os.environ.get("QSR_GGUF_Q6_ALIGNED", "0").strip() != "0"


def _q6_split_weights_enabled() -> bool:
    """Select the zero-overhead split Q6_K storage experiment."""

    return os.environ.get("QSR_GGUF_Q6_SPLIT", "0").strip() != "0"


def _q8_split_weights_enabled() -> bool:
    """Select the zero-overhead split Q8_0 storage experiment."""

    return os.environ.get("QSR_GGUF_Q8_SPLIT", "0").strip() != "0"


# M=1 is the latency-sensitive decode case; M=8 is DFlash2's fixed verify
# width and is already large enough for the tensor-core decoder to win.  The
# same gate is used for BF16 prefill and F32 verify so enabling the packed
# tensor-core switch cannot silently replace the M=1 GEMV with a slower tile.
_TENSOR_CORE_MIN_ROWS = 8


def _tensor_core_rows_enabled(rows: int, *, f32: bool) -> bool:
    return rows >= _TENSOR_CORE_MIN_ROWS or (rows == 1 and _native_tensor_core_m1_enabled(f32=f32))


def _native_library() -> NativeGgufQK:
    global _NATIVE_LIBRARY
    if _NATIVE_LIBRARY is None:
        _NATIVE_LIBRARY = NativeGgufQK.load()
    return _NATIVE_LIBRARY


class GgufLinear(nn.Module):
    """A bias-free Linear whose resident weight is one GGML packed tensor."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        type_name: str,
        *,
        cache_dequantized: bool = False,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        if type_name not in GGUF_BLOCK_BYTES:
            raise ValueError(f"unsupported GGUF Linear type {type_name!r}")
        elements_per_block = 32 if type_name == "Q8_0" else 256
        if input_size % elements_per_block:
            raise ValueError(
                f"GGUF {type_name} Linear input size {input_size} is not a multiple of "
                f"the {elements_per_block}-element block size"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.type_name = type_name
        self.module_name = module_name
        self.row_bytes = (input_size // elements_per_block) * GGUF_BLOCK_BYTES[type_name]
        self.cache_dequantized = cache_dequantized
        self.weight = nn.Parameter(
            torch.empty(output_size * self.row_bytes, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self._dequantized_weight: torch.Tensor | None = None
        self._native_packed_weight: torch.Tensor | None = None
        self._native_storage_type: str | None = None
        # Lazy, opt-in MX-FP6 W6A8 representation.  This is deliberately a
        # plain attribute rather than a Parameter: the loader remains the
        # owner of the GGUF bytes and the derived representation is an
        # execution cache, not a checkpoint format.
        self._mxfp6_weight: Any | None = None
        self._tensor_core_tile_major_weights: dict[tuple[int, int, int], torch.Tensor] = {}
        self._packed_weight_released = False

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        expected_numel = self.output_size * self.row_bytes
        if loaded_weight.dtype != torch.uint8 or loaded_weight.numel() != expected_numel:
            raise ValueError(
                f"GGUF {self.type_name} weight expects {expected_numel} uint8 values, got "
                f"{loaded_weight.numel()} {loaded_weight.dtype}"
            )
        # Resident-BF16 mode intentionally shrinks ``param`` to an empty
        # placeholder after warmup.  Keep reload/state restore valid instead
        # of making that one-way representation change a loader footgun.
        if param.numel() != expected_numel:
            param.data = torch.empty(
                expected_numel,
                dtype=torch.uint8,
                device=param.device,
            )
        param.data.copy_(loaded_weight.reshape_as(param))
        self._dequantized_weight = None
        self._native_packed_weight = None
        self._native_storage_type = None
        self._mxfp6_weight = None
        self._tensor_core_tile_major_weights.clear()
        self._packed_weight_released = False

    def _ensure_mxfp6_weight(self) -> Any:
        """Build and cache Sparkinfer's MX-FP6 W6A8 weight from Q6_K bytes.

        Conversion is eager-only because both the native row gather and the
        Sparkinfer quantizer allocate their output buffers.  If an earlier
        decode path already repacked the source into Q6_K_SPLIT, the native
        dequantizer is used to recover the same BF16 matrix without rebuilding
        the standard 210-byte layout on the host.
        """

        if self.type_name != "Q6_K":
            raise RuntimeError(f"MX-FP6 W6A8 only supports Q6_K, got {self.type_name!r}")
        cached = self._mxfp6_weight
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot build MX-FP6 GGUF weights during CUDA Graph capture")

        if self.weight.numel() != 0:
            weight_bf16 = dequantize_gguf_packed(
                self.weight,
                (self.output_size, self.input_size),
                self.type_name,
                dtype=torch.bfloat16,
            ).contiguous()
        else:
            storage = self._tensor_core_storage()
            if storage is None:
                raise RuntimeError(
                    f"GGUF {self.type_name} packed storage was released before MX-FP6 conversion"
                )
            packed, row_bytes, type_name = storage
            row_ids = torch.arange(self.output_size, dtype=torch.int64, device=packed.device)
            weight_bf16 = _native_library().dequant_rows(
                row_ids,
                packed,
                rows=self.output_size,
                k=self.input_size,
                row_bytes=row_bytes,
                type_name=type_name,
                dtype=torch.bfloat16,
            )

        # Keep the import boundary lazy: bfdiag/CPU-only collection must not
        # import Cutlass DSL or Sparkinfer just because GGUF modules exist.
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from b12x.quantization.mxfp6 import quantize_dense_weight_to_fp6

        try:
            cached = quantize_dense_weight_to_fp6(
                weight_bf16,
                source_format="mxfp6_w6a8",
            )
        finally:
            del weight_bf16
        self._mxfp6_weight = cached
        _logger.info(
            "GGUF MX-FP6 W6A8 cache built for %s (%d x %d)",
            self.module_name or "<unnamed>",
            self.output_size,
            self.input_size,
        )
        return cached

    def _mxfp6_forward(self, flat: torch.Tensor) -> torch.Tensor | None:
        """Run a measured M=8 Q6_K projection through MX-FP6 W6A8."""

        if not (
            _native_mxfp6_w6a8_enabled()
            and self.type_name == "Q6_K"
            and flat.dtype == torch.bfloat16
            and _native_mxfp6_rows_enabled(flat.shape[0])
            and _native_mxfp6_module_enabled(self.module_name)
        ):
            return None
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from b12x.quantization.mxfp6 import dense_fp6_linear

        return dense_fp6_linear(flat, self._ensure_mxfp6_weight())

    def _native_q8_storage(self) -> tuple[torch.Tensor, int, str]:
        """Return packed storage and ABI geometry for the native Q8 path.

        The production representation remains the loader/state-dict payload.
        Three explicit native experiments can repack it eagerly: a padded
        224-byte Q6_K block layout, a same-size split Q6_K layout whose
        208-byte payload blocks are followed by a row-tail array of FP16 d
        values, or the analogous Q8_0 layout with 32-byte payload blocks.
        """

        if self.type_name == "Q8_0" and _q8_split_weights_enabled():
            split = self._native_packed_weight
            if split is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("cannot build split Q8_0 storage during CUDA Graph capture")
                if self.weight.numel() == 0:
                    raise RuntimeError("Q8_0 packed storage was released before native repack")
                blocks_per_row = self.input_size // 32
                source = self.weight.view(self.output_size, blocks_per_row, 34)
                split = self.weight.new_empty(self.output_size, self.row_bytes)
                split_data = split[:, : blocks_per_row * _Q8_SPLIT_DATA_BLOCK_BYTES]
                split_data = split_data.view(
                    self.output_size,
                    blocks_per_row,
                    _Q8_SPLIT_DATA_BLOCK_BYTES,
                )
                split_scales = split[:, blocks_per_row * _Q8_SPLIT_DATA_BLOCK_BYTES :]
                split_scales = split_scales.view(self.output_size, blocks_per_row, 2)
                split_data.copy_(source[..., 2:])
                split_scales.copy_(source[..., :2])
                self._native_packed_weight = split.reshape(-1)
                self._native_storage_type = "Q8_0_SPLIT"
                self.weight.data = self.weight.data.new_empty(0)
                self._packed_weight_released = True
            assert self._native_packed_weight is not None
            return (
                self._native_packed_weight,
                self.row_bytes,
                self._native_storage_type or "Q8_0_SPLIT",
            )
        # A merged projection may have released the loader Parameter after
        # copying its rows into one shared packed allocation.  Keep the
        # per-linear slice usable for a later dtype fallback (for example a
        # prefill row that remains F32), instead of returning the empty
        # placeholder below.
        if self._native_packed_weight is not None and self._native_storage_type == self.type_name:
            return self._native_packed_weight, self.row_bytes, self.type_name
        if self.type_name != "Q6_K":
            return self.weight, self.row_bytes, self.type_name
        if _q6_split_weights_enabled():
            split = self._native_packed_weight
            if split is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("cannot build split Q6_K storage during CUDA Graph capture")
                if self.weight.numel() == 0:
                    raise RuntimeError("Q6_K packed storage was released before native repack")
                blocks_per_row = self.input_size // 256
                source = self.weight.view(self.output_size, blocks_per_row, 210)
                split = self.weight.new_empty(self.output_size, self.row_bytes)
                split_data = split[:, : blocks_per_row * _Q6_SPLIT_DATA_BLOCK_BYTES]
                split_data = split_data.view(
                    self.output_size,
                    blocks_per_row,
                    _Q6_SPLIT_DATA_BLOCK_BYTES,
                )
                split_scales = split[:, blocks_per_row * _Q6_SPLIT_DATA_BLOCK_BYTES :]
                split_scales = split_scales.view(self.output_size, blocks_per_row, 2)
                split_data.copy_(source[..., :_Q6_SPLIT_DATA_BLOCK_BYTES])
                split_scales.copy_(source[..., _Q6_SPLIT_DATA_BLOCK_BYTES:])
                self._native_packed_weight = split.reshape(-1)
                self._native_storage_type = "Q6_K_SPLIT"
                self.weight.data = self.weight.data.new_empty(0)
                self._packed_weight_released = True
            assert self._native_packed_weight is not None
            return (
                self._native_packed_weight,
                self.row_bytes,
                self._native_storage_type or "Q6_K_SPLIT",
            )
        if not _q6_aligned_weights_enabled():
            return self.weight, self.row_bytes, self.type_name
        aligned = self._native_packed_weight
        if aligned is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cannot build aligned Q6_K storage during CUDA Graph capture")
            if self.weight.numel() == 0:
                raise RuntimeError("Q6_K packed storage was released before native repack")
            blocks_per_row = self.input_size // 256
            source = self.weight.view(self.output_size, blocks_per_row, 210)
            aligned = self.weight.new_empty(
                self.output_size, blocks_per_row, _Q6_ALIGNED_BLOCK_BYTES
            )
            aligned[..., :210].copy_(source)
            self._native_packed_weight = aligned.reshape(-1)
            self._native_storage_type = "Q6_K_ALIGNED"
            self.weight.data = self.weight.data.new_empty(0)
            self._packed_weight_released = True
            aligned = self._native_packed_weight
        assert aligned is not None
        blocks_per_row = self.input_size // 256
        return (
            aligned,
            blocks_per_row * _Q6_ALIGNED_BLOCK_BYTES,
            "Q6_K_ALIGNED",
        )

    def _tensor_core_storage(self) -> tuple[torch.Tensor, int, str] | None:
        """Return the resident layout that the Triton TC decoder can read.

        Decode graph capture may have already converted a Q6/Q8 source into
        the native split layout and released the loader ``Parameter``.  The
        split payload is still a valid tensor-core input once the Triton
        decoder knows its row-tail ``d`` location, so retain one storage
        selection point instead of blindly passing the now-empty Parameter.
        Padded Q6 is intentionally excluded until it has a matching Triton
        decoder; callers use the native Q8 path in that configuration.
        """

        if self.weight.numel() != 0:
            return self.weight, self.row_bytes, self.type_name
        if self._native_packed_weight is None:
            return None
        if self._native_storage_type in {
            "Q4_K",
            "Q5_K",
            "Q6_K",
            "Q8_0",
            "Q6_K_SPLIT",
            "Q8_0_SPLIT",
        }:
            return (
                self._native_packed_weight,
                self.row_bytes,
                self._native_storage_type,
            )
        return None

    def _tensor_core_tile_major_storage(
        self,
        *,
        rows: int,
    ) -> tuple[torch.Tensor, int, str] | None:
        """Return an exact N-tile-major cache for the measured TC row count.

        The cache preserves the GGML bytes and only changes their physical
        order.  It is built during eager warmup, never during CUDA Graph
        capture; a missing cache therefore falls back to the row-major
        decoder instead of allocating from inside a graph.
        """

        if not (
            _native_tensor_core_tile_major_enabled()
            and _native_tensor_core_tile_major_rows_enabled(rows)
            and _native_tensor_core_tile_major_module_enabled(self.module_name)
        ):
            return None
        storage = self._tensor_core_storage()
        if storage is None:
            return None
        packed, row_bytes, type_name = storage
        from runtime.kernels.gguf_qk_triton import _tensor_core_block_n

        block_n = _tensor_core_block_n(
            type_name=type_name,
            rows=rows,
            n=self.output_size,
            k=self.input_size,
        )
        key = (packed.data_ptr(), packed.numel(), block_n)
        cached = self._tensor_core_tile_major_weights.get(key)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            from runtime.kernels.gguf_qk_triton import gguf_qk_repack_for_tensor_core

            cached, _padded_n = gguf_qk_repack_for_tensor_core(
                packed,
                n=self.output_size,
                k=self.input_size,
                row_bytes=row_bytes,
                type_name=type_name,
                block_n=block_n,
            )
            self._tensor_core_tile_major_weights[key] = cached
            _logger.info(
                "GGUF tile-major cache built for %s (%d x %d, block_n=%d, %.1f MiB)",
                self.module_name or "<unnamed>",
                self.output_size,
                self.input_size,
                block_n,
                cached.numel() / 2**20,
            )
        return cached, block_n, type_name

    def _release_native_q8_storage(self) -> None:
        """Drop a source payload after a merged projection copied it."""

        self._native_packed_weight = None
        if self.weight.numel() != 0:
            self.weight.data = self.weight.data.new_empty(0)
        self._packed_weight_released = True

    def _native_q8_enabled(self) -> bool:
        """Keep the logits projection on exact activation arithmetic by default."""

        return (
            _native_q8_activation_enabled()
            and _native_q8_type_enabled(self.type_name)
            and (
                not self.cache_dequantized
                or os.environ.get("QSR_GGUF_NATIVE_Q8_LM_HEAD", "0").strip() != "0"
            )
        )

    def _native_prefill_bf16(
        self,
        flat: torch.Tensor,
        native: NativeGgufQK,
    ) -> torch.Tensor | None:
        """Run one prefill projection through transient BF16 weights.

        The native row gather and cuBLAS matmul are substantially faster than
        decoding every packed value inside the current M>1 Triton kernel on
        SM120.  The BF16 matrix is a local temporary: unlike resident mode it
        is not cached and the packed checkpoint representation remains the
        long-lived storage.  CUDA Graph capture must use the packed path
        because this branch allocates a variable-size workspace.
        """

        if (
            not _native_prefill_dequant_enabled()
            # DFlash2/DSpark CUDA-Graph warmups also execute M=8 verify
            # batches on an eager side stream before capture. Treating those
            # rows as prefill makes this opt-in path allocate and run cuBLAS
            # during warmup, leaving capture/replay with a different
            # execution/lifetime pattern. Keep transient dequantization on
            # genuinely prefill-sized batches; decode and verify stay packed.
            or flat.shape[0] < 32
            or torch.cuda.is_current_stream_capturing()
            or self.output_size * self.input_size * _BF16_ELEMENT_BYTES
            > _native_prefill_dequant_max_bytes()
        ):
            return None
        storage = self._tensor_core_storage()
        if storage is None:
            return None
        packed, row_bytes, type_name = storage
        row_ids = torch.arange(self.output_size, dtype=torch.int64, device=flat.device)
        weight = native.dequant_rows(
            row_ids,
            packed,
            rows=self.output_size,
            k=self.input_size,
            row_bytes=row_bytes,
            type_name=type_name,
            dtype=torch.bfloat16,
        )
        activation = flat if flat.dtype == torch.bfloat16 else flat.to(torch.bfloat16)
        output = F.linear(activation, weight)
        return output if flat.dtype == torch.bfloat16 else output.to(flat.dtype)

    def _materialize_resident_bf16_weight(self) -> torch.Tensor:
        """Dequantize this module once and release its packed payload.

        The operation is expected during eager prefill/warmup, never while a
        CUDA Graph is being captured.  Keeping the packed ``Parameter`` as an
        empty uint8 tensor preserves the loader/state-dict name while freeing
        its storage; the BF16 matrix is the only resident representation used
        by the cuBLAS path afterwards.
        """

        cached = self._dequantized_weight
        if cached is not None:
            return cached
        if self.weight.numel() == 0:
            raise RuntimeError(
                f"GGUF {self.type_name} packed storage was released before its BF16 weight "
                "was materialized"
            )
        packed = self.weight
        cached = dequantize_gguf_packed(
            packed,
            (self.output_size, self.input_size),
            self.type_name,
            dtype=torch.bfloat16,
        ).contiguous()
        self._dequantized_weight = cached
        # The local ``packed`` reference keeps the old allocation alive only
        # until this method returns.  No model-sized duplicate is retained.
        self.weight.data = packed.new_empty(0)
        self._packed_weight_released = True
        return cached

    def _weight_for(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cached = self._dequantized_weight
        if cached is None or cached.dtype != dtype or cached.device != device:
            cached = dequantize_gguf_packed(
                self.weight,
                (self.output_size, self.input_size),
                self.type_name,
                dtype=dtype,
            )
            if self.cache_dequantized:
                self._dequantized_weight = cached
        return cached

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and _native_mxfp6_w6a8_enabled()
            and self.type_name == "Q6_K"
        ):
            flat = hidden_states.reshape(-1, self.input_size).contiguous()
            mxfp6_output = self._mxfp6_forward(flat)
            if mxfp6_output is not None:
                return mxfp6_output.reshape(*hidden_states.shape[:-1], self.output_size)
        if (
            hidden_states.is_cuda
            and _resident_bf16_weights_enabled(self.type_name, self.module_name)
            and self.type_name in _NATIVE_GGUF_TYPES
            and not _native_mmq_lm_head_enabled(self.module_name, self.type_name)
        ):
            flat = hidden_states.reshape(-1, self.input_size).contiguous()
            weight = self._materialize_resident_bf16_weight()
            if hidden_states.dtype == torch.float32:
                output = F.linear(flat.to(torch.bfloat16), weight).to(torch.float32)
            elif hidden_states.dtype == torch.bfloat16:
                output = F.linear(flat, weight)
            else:
                raise TypeError(
                    "resident GGUF BF16 weights expect BF16 or F32 activations, "
                    f"got {hidden_states.dtype}"
                )
            return output.reshape(*hidden_states.shape[:-1], self.output_size)
        if (
            hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
            and _native_gguf_enabled()
            and self.type_name in _NATIVE_GGUF_TYPES
        ):
            flat = hidden_states.reshape(-1, self.input_size).contiguous()
            native = _native_library()
            tensor_core_storage = (
                self._tensor_core_storage()
                if (
                    _native_tensor_core_enabled()
                    or (hidden_states.dtype == torch.float32 and _native_tensor_core_f32_enabled())
                    or _native_prefill_dequant_enabled()
                )
                else None
            )
            prefill_output = self._native_prefill_bf16(flat, native)
            if prefill_output is not None:
                output = prefill_output
            elif (
                hidden_states.dtype == torch.bfloat16
                and (
                    _native_mmq_rows_enabled(flat.shape[0])
                    or (self.type_name == "Q8_0" and _native_mmq_q8_rows_enabled(flat.shape[0]))
                )
                and _native_mmq_shape_enabled(self.output_size, self.input_size)
                and (
                    (self.type_name == "Q5_K" and _native_mmq_q5_enabled())
                    or (self.type_name == "Q6_K" and _q6_split_weights_enabled())
                    or (
                        self.type_name == "Q8_0"
                        and _native_mmq_q8_enabled()
                        and _native_mmq_q8_module_enabled(self.module_name)
                    )
                    or _native_mmq_lm_head_enabled(self.module_name, self.type_name)
                )
                and (
                    self._native_q8_enabled()
                    or _native_mmq_lm_head_enabled(self.module_name, self.type_name)
                )
            ):
                # SGLang's MMQ tile is an opt-in approximation for the
                # DFlash2 M=8 verify shape.  Q6 uses the split payload used by
                # the native DP4A GEMV; Q8_0 uses the aligned zero-copy split
                # row layout in the tuned route.  Both share the invocation-
                # scoped Q8_1 activation with adjacent linears.
                packed, row_bytes, type_name = self._native_q8_storage()
                if not _native_mmq_storage_enabled(type_name):
                    raise RuntimeError(f"unsupported MMQ storage layout {type_name!r}")
                activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
                if activation_workspace is None:
                    activation_workspace = native.quantize_q8_1(flat)
                output = native.gemm_q8_mmq(
                    activation_workspace,
                    packed,
                    m=flat.shape[0],
                    n=self.output_size,
                    k=self.input_size,
                    row_bytes=row_bytes,
                    type_name=type_name,
                )
            elif (
                _native_tensor_core_enabled()
                and hidden_states.dtype == torch.bfloat16
                and _tensor_core_rows_enabled(flat.shape[0], f32=False)
                and tensor_core_storage is not None
            ):
                tile_major_storage = self._tensor_core_tile_major_storage(rows=flat.shape[0])
                if tile_major_storage is None:
                    packed, row_bytes, type_name = tensor_core_storage
                    output = native.gemm_tensor_core(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=row_bytes,
                        type_name=type_name,
                    )
                else:
                    packed, block_n, type_name = tile_major_storage
                    output = native.gemm_tensor_core_tile_major(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        type_name=type_name,
                        block_n=block_n,
                    )
            elif (
                hidden_states.dtype == torch.float32
                and _native_tensor_core_f32_enabled()
                and _tensor_core_rows_enabled(flat.shape[0], f32=True)
                and tensor_core_storage is not None
            ):
                # The GGUF model graph deliberately exposes F32 tensors so
                # GDN scalars, full-attention reductions, and recurrent state
                # keep their file-format precision.  Only the large packed
                # linear is rounded to BF16 here: this selects the SM120
                # tensor-core decoder without materialising a dequantized
                # weight matrix, then restores the module's F32 contract at
                # the output boundary.  M=1 decode does not enter this
                # branch, preserving its direct arithmetic path and avoiding
                # an activation cast for the latency-sensitive single row.
                packed, row_bytes, type_name = tensor_core_storage
                output = native.gemm_tensor_core(
                    flat.to(torch.bfloat16),
                    packed,
                    m=flat.shape[0],
                    n=self.output_size,
                    k=self.input_size,
                    row_bytes=row_bytes,
                    type_name=type_name,
                ).to(torch.float32)
            elif (
                hidden_states.dtype == torch.float32
                and _native_f32_q8_activation_enabled()
                and _native_f32_q8_module_enabled(self.module_name)
                and _native_q8_type_enabled(self.type_name)
                and (
                    not self.cache_dequantized
                    or os.environ.get("QSR_GGUF_NATIVE_Q8_LM_HEAD", "0").strip() != "0"
                )
                and flat.shape[0] == 1
            ):
                # Decode is a true GEMV.  Quantizing this one activation row
                # to Q8_1 enables the native DP4A K-quant kernel and avoids
                # the scalar per-value decode in ``gemm_direct``.  Match
                # SGLang's GGML path by quantizing the F32 row once per model
                # invocation and retaining an F32 output boundary; the switch
                # remains opt-in because Q8_1 activation quantization is an
                # approximation.
                packed, row_bytes, type_name = self._native_q8_storage()
                activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
                if activation_workspace is None:
                    output = native.gemm_q8_f32(
                        flat,
                        packed,
                        m=1,
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=row_bytes,
                        type_name=type_name,
                    )
                else:
                    output = native.gemm_q8_prequantized(
                        activation_workspace,
                        packed,
                        m=1,
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=row_bytes,
                        type_name=type_name,
                        output_dtype=torch.float32,
                    )
            elif (
                hidden_states.dtype == torch.float32
                and _native_f32_gemv_bf16_enabled()
                and flat.shape[0] == 1
            ):
                direct_storage = self._tensor_core_storage()
                if direct_storage is None:
                    raise RuntimeError(
                        f"GGUF {self.type_name} direct storage was released before use"
                    )
                direct_packed, direct_row_bytes, direct_type_name = direct_storage
                output = native.gemm_direct(
                    flat.to(torch.bfloat16),
                    direct_packed,
                    m=1,
                    n=self.output_size,
                    k=self.input_size,
                    row_bytes=direct_row_bytes,
                    type_name=direct_type_name,
                    cache_activation=_native_cache_activation_enabled(),
                ).to(torch.float32)
            elif hidden_states.dtype == torch.float32 or not self._native_q8_enabled():
                direct_storage = self._tensor_core_storage()
                if direct_storage is None:
                    raise RuntimeError(
                        f"GGUF {self.type_name} direct storage was released before use"
                    )
                direct_packed, direct_row_bytes, direct_type_name = direct_storage
                output = native.gemm_direct(
                    flat,
                    direct_packed,
                    m=flat.shape[0],
                    n=self.output_size,
                    k=self.input_size,
                    row_bytes=direct_row_bytes,
                    type_name=direct_type_name,
                    cache_activation=_native_cache_activation_enabled(),
                )
            else:
                packed, row_bytes, type_name = self._native_q8_storage()
                activation_workspace = _cached_q8_activation(
                    native, flat.to(torch.bfloat16), source=hidden_states
                )
                if activation_workspace is None:
                    output = native.gemm(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=row_bytes,
                        type_name=type_name,
                    )
                else:
                    output = native.gemm_q8_prequantized(
                        activation_workspace,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=row_bytes,
                        type_name=type_name,
                    )
            return output.reshape(*hidden_states.shape[:-1], self.output_size)
        weight = self._weight_for(hidden_states.dtype, hidden_states.device)
        return F.linear(hidden_states, weight)


class GgufMergedLinear(nn.Module):
    """Run projections sharing one input activation through the packed path.

    Same-format projections are physically concatenated and use one GEMM.
    Mixed-format projections cannot share a packed payload, but they still
    share the expensive Q8_1 activation quantization through the native
    prequantized ABI.  The source modules stay attached to their owning model
    for loader/state-dict compatibility.  M=1 uses the dynamic mixed kernel
    by default because it avoids an additional output concat per projection;
    ``QSR_GGUF_NATIVE_Q8_MIXED=0`` selects the format-specialized diagnostic
    fallback.
    """

    def __init__(self, *linears: GgufLinear) -> None:
        super().__init__()
        if len(linears) < 2:
            raise ValueError("GgufMergedLinear needs at least two projections")
        first = linears[0]
        if any(linear.input_size != first.input_size for linear in linears[1:]):
            raise ValueError("GgufMergedLinear projections must share input geometry")
        self._linears = tuple(linears)
        self.input_size = first.input_size
        self.output_size = sum(linear.output_size for linear in linears)
        self.type_name = first.type_name
        self.row_bytes = first.row_bytes
        self._same_type = all(linear.type_name == first.type_name for linear in linears[1:])
        self._q8_packed_weight: torch.Tensor | None = None
        self._tensor_core_packed_weight: torch.Tensor | None = None
        self._native_row_bytes = self.row_bytes
        self._native_type_name = self.type_name
        self._tensor_core_row_bytes = self.row_bytes
        self._tensor_core_type_name = self.type_name
        self._tensor_core_tile_major_weights: dict[tuple[int, int, int], torch.Tensor] = {}
        self._mixed_descriptors: torch.Tensor | None = None

    def _ensure_packed_weight(self, *, q8: bool = True) -> torch.Tensor:
        packed = self._q8_packed_weight if q8 else self._tensor_core_packed_weight
        if not self._same_type:
            raise RuntimeError("mixed-format GGUF projections do not have one packed payload")
        if packed is not None:
            return packed
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot build merged GGUF payload during CUDA Graph capture")
        source_row_bytes: int
        source_type_name: str
        if q8:
            # A same-type merged projection may have been fused by the
            # tensor-core path first (for example during prefill warmup).
            # Its packed representation is already valid for the native Q8
            # decoder, so do not try to read source Parameters that fusion
            # intentionally released.
            tensor_core_type = self._tensor_core_type_name
            if self._tensor_core_packed_weight is not None and tensor_core_type in {
                "Q4_K",
                "Q5_K",
                "Q6_K",
                "Q8_0",
                "Q6_K_SPLIT",
                "Q8_0_SPLIT",
                "Q6_K_ALIGNED",
            }:
                self._q8_packed_weight = self._tensor_core_packed_weight
                self._native_row_bytes = self._tensor_core_row_bytes
                self._native_type_name = tensor_core_type
                self._bind_merged_source_slices(
                    self._q8_packed_weight,
                    row_bytes=self._native_row_bytes,
                    type_name=self._native_type_name,
                )
                return self._q8_packed_weight
            storages = [linear._native_q8_storage() for linear in self._linears]
            packed = torch.cat(tuple(storage[0].detach() for storage in storages), dim=0)
            self._native_row_bytes = storages[0][1]
            self._native_type_name = storages[0][2]
            source_row_bytes = self._native_row_bytes
            source_type_name = self._native_type_name
        else:
            storages = [linear._tensor_core_storage() for linear in self._linears]
            if any(storage is None for storage in storages):
                raise RuntimeError(
                    "merged GGUF source payload has no tensor-core-compatible storage"
                )
            concrete_storages = [storage for storage in storages if storage is not None]
            type_names = {storage[2] for storage in concrete_storages}
            if len(type_names) != 1:
                raise RuntimeError(
                    "merged GGUF tensor-core projections must use one physical layout"
                )
            packed = torch.cat(tuple(storage[0].detach() for storage in concrete_storages), dim=0)
            self._tensor_core_row_bytes = concrete_storages[0][1]
            self._tensor_core_type_name = concrete_storages[0][2]
            source_row_bytes = self._tensor_core_row_bytes
            source_type_name = self._tensor_core_type_name
        packed = packed.contiguous()
        if q8:
            self._q8_packed_weight = packed
            # Split and standard native Q8 layouts are both understood by
            # Triton.  Reuse the merged allocation rather than retaining a
            # second copy solely for prefill/verify.
            if self._native_type_name in {
                "Q4_K",
                "Q5_K",
                "Q6_K",
                "Q8_0",
                "Q6_K_SPLIT",
                "Q8_0_SPLIT",
            }:
                self._tensor_core_packed_weight = packed
                self._tensor_core_row_bytes = self._native_row_bytes
                self._tensor_core_type_name = self._native_type_name
        else:
            self._tensor_core_packed_weight = packed
        self._bind_merged_source_slices(
            packed,
            row_bytes=source_row_bytes,
            type_name=source_type_name,
        )
        return packed

    def _bind_merged_source_slices(
        self,
        packed: torch.Tensor,
        *,
        row_bytes: int,
        type_name: str,
    ) -> None:
        """Keep released source modules addressable through merged storage.

        The merged projection owns one contiguous copy, but its source
        ``GgufLinear`` objects can still be reached by a dtype-specific
        fallback later in the same model (notably an F32 prefill path after a
        BF16 graph warmup).  Bind row-aligned views before releasing each
        loader Parameter; this preserves one packed residency without
        leaving the source module with an unusable zero-byte payload.
        """

        offset = 0
        for linear in self._linears:
            size = linear.output_size * row_bytes
            linear._native_packed_weight = packed.narrow(0, offset, size)  # noqa: SLF001
            linear._native_storage_type = type_name  # noqa: SLF001
            if linear.weight.numel() != 0:
                linear.weight.data = linear.weight.data.new_empty(0)
            linear._packed_weight_released = True  # noqa: SLF001
            offset += size
        if offset != packed.numel():
            raise RuntimeError(
                "merged GGUF packed payload does not match its source row geometry: "
                f"bound={offset} bytes, payload={packed.numel()} bytes"
            )

    def _tensor_core_storage_available(self) -> bool:
        if self._tensor_core_packed_weight is not None:
            return True
        return all(linear._tensor_core_storage() is not None for linear in self._linears)

    def _tensor_core_tile_major_storage(
        self,
        *,
        rows: int,
    ) -> tuple[torch.Tensor, int, str] | None:
        """Return one exact tile-major cache for a same-format merged GEMM."""

        if not (
            self._same_type
            and _native_tensor_core_tile_major_enabled()
            and _native_tensor_core_tile_major_rows_enabled(rows)
            and any(
                _native_tensor_core_tile_major_module_enabled(linear.module_name)
                for linear in self._linears
            )
        ):
            return None
        if self._tensor_core_packed_weight is None and torch.cuda.is_current_stream_capturing():
            return None
        packed = self._ensure_packed_weight(q8=False)
        from runtime.kernels.gguf_qk_triton import _tensor_core_block_n

        block_n = _tensor_core_block_n(
            type_name=self._tensor_core_type_name,
            rows=rows,
            n=self.output_size,
            k=self.input_size,
        )
        key = (packed.data_ptr(), packed.numel(), block_n)
        cached = self._tensor_core_tile_major_weights.get(key)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            from runtime.kernels.gguf_qk_triton import gguf_qk_repack_for_tensor_core

            cached, _padded_n = gguf_qk_repack_for_tensor_core(
                packed,
                n=self.output_size,
                k=self.input_size,
                row_bytes=self._tensor_core_row_bytes,
                type_name=self._tensor_core_type_name,
                block_n=block_n,
            )
            self._tensor_core_tile_major_weights[key] = cached
            _logger.info(
                "GGUF merged tile-major cache built (%d x %d, block_n=%d, %.1f MiB)",
                self.output_size,
                self.input_size,
                block_n,
                cached.numel() / 2**20,
            )
        return cached, block_n, self._tensor_core_type_name

    def _native_prefill_bf16(
        self,
        flat: torch.Tensor,
        native: NativeGgufQK,
    ) -> torch.Tensor | None:
        """Run a merged projection through one transient BF16 matrix."""

        if (
            not _native_prefill_dequant_enabled()
            # See GgufLinear._native_prefill_bf16: the DFlash2 eager warmup
            # uses M=8 before graph capture and must stay on packed weights.
            or flat.shape[0] < 32
            or torch.cuda.is_current_stream_capturing()
            or self.output_size * self.input_size * _BF16_ELEMENT_BYTES
            > _native_prefill_dequant_max_bytes()
        ):
            return None
        if not self._same_type:
            outputs = []
            for linear in self._linears:
                output = linear._native_prefill_bf16(flat, native)  # noqa: SLF001
                if output is None:
                    return None
                outputs.append(output)
            return torch.cat(tuple(outputs), dim=-1)

        packed = self._ensure_packed_weight(q8=False)
        row_ids = torch.arange(self.output_size, dtype=torch.int64, device=flat.device)
        weight = native.dequant_rows(
            row_ids,
            packed,
            rows=self.output_size,
            k=self.input_size,
            row_bytes=self._tensor_core_row_bytes,
            type_name=self._tensor_core_type_name,
            dtype=torch.bfloat16,
        )
        output = F.linear(flat, weight)
        return output

    def _native_q8_enabled(self) -> bool:
        return _native_q8_activation_enabled() and all(
            linear._native_q8_enabled() for linear in self._linears
        )

    def _native_f32_q8_enabled(self) -> bool:
        """Return whether this merged projection is on the F32 Q8 allowlist."""

        return _native_f32_q8_activation_enabled() and all(
            _native_f32_q8_module_enabled(linear.module_name)
            and _native_q8_type_enabled(linear.type_name)
            for linear in self._linears
        )

    def _ensure_mixed_descriptors(self, device: torch.device) -> torch.Tensor:
        descriptors = self._mixed_descriptors
        if descriptors is not None:
            return descriptors
        if self._same_type:
            raise RuntimeError("same-format GGUF projections do not need mixed descriptors")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot build mixed GGUF descriptors during CUDA Graph capture")
        storages = [linear._native_q8_storage() for linear in self._linears]
        descriptors = torch.zeros((len(self._linears), 4), dtype=torch.int64, device=device)
        metadata = descriptors.view(torch.int32).view(len(self._linears), 8)
        offset = 0
        type_ids = {
            "Q4_K": 0,
            "Q5_K": 1,
            "Q6_K": 2,
            "Q8_0": 3,
            "Q6_K_ALIGNED": 4,
            "Q6_K_SPLIT": 5,
            "Q8_0_SPLIT": 6,
        }
        for index, (linear, storage) in enumerate(zip(self._linears, storages)):
            packed, row_bytes, type_name = storage
            if packed.device != device:
                raise RuntimeError("mixed GGUF projections must share one CUDA device")
            descriptors[index, 0] = packed.data_ptr()
            metadata[index, 2] = offset
            metadata[index, 3] = linear.output_size
            metadata[index, 4] = row_bytes
            metadata[index, 5] = type_ids[type_name]
            offset += linear.output_size
        self._mixed_descriptors = descriptors
        return descriptors

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.device.type != "cuda"
            or hidden_states.dtype not in (torch.bfloat16, torch.float32)
            or not _native_gguf_enabled()
            or any(
                _resident_bf16_weights_enabled(linear.type_name, linear.module_name)
                for linear in self._linears
            )
        ):
            return torch.cat(tuple(linear(hidden_states) for linear in self._linears), dim=-1)

        flat = hidden_states.reshape(-1, self.input_size).contiguous()
        native = _native_library()
        prefill_output = self._native_prefill_bf16(flat, native)
        if prefill_output is not None:
            return prefill_output.reshape(*hidden_states.shape[:-1], self.output_size)
        if (
            hidden_states.dtype == torch.bfloat16
            and _native_mxfp6_w6a8_enabled()
            and _native_mxfp6_rows_enabled(flat.shape[0])
        ):
            # Qwen3.8's GGUF MLP gate/up pair is commonly mixed Q5_K + Q6_K.
            # Run the Q6 half through MX-FP6 while retaining each other
            # projection's measured native route.  This keeps the mixed
            # descriptor ABI exact for the default profile and only splits the
            # pair when the explicit W6A8 switch is enabled.
            outputs = []
            for linear in self._linears:
                output = linear._mxfp6_forward(flat)  # noqa: SLF001
                if output is None:
                    # ``flat`` keeps the mixed projection outputs rank-2;
                    # passing the original [B, L, H] tensor here would leave
                    # the non-MXFP6 sibling rank-3 and make the concat fail.
                    output = linear(flat)
                outputs.append(output)
            return torch.cat(tuple(outputs), dim=-1).reshape(
                *hidden_states.shape[:-1], self.output_size
            )
        if not self._same_type:
            if flat.dtype == torch.float32 and flat.shape[0] == 1 and self._native_f32_q8_enabled():
                # Mixed QKV/GDN projections still share one F32 Q8_1 row.
                # Keep this before the exact-F32 guard: SGLang's GGML path
                # does not require the adjacent projections to share a weight
                # format, only the activation quantization and row geometry.
                activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
                if activation_workspace is None:
                    activation_workspace = native.quantize_q8_1(flat)
                output = native.gemm_q8_mixed(
                    activation_workspace,
                    self._ensure_mixed_descriptors(flat.device),
                    projection_count=len(self._linears),
                    total_n=self.output_size,
                    k=self.input_size,
                    output_dtype=torch.float32,
                )
                return output.reshape(*hidden_states.shape[:-1], self.output_size)
            if flat.dtype == torch.float32 or not self._native_q8_enabled():
                # The Q8_1 activation path is an approximation.  When it is
                # disabled for a quality run, preserve that contract for
                # merged mixed-format projections as well; otherwise this
                # branch would silently quantize activations despite the
                # process-wide switch.
                if flat.shape[0] == 1:
                    if flat.dtype == torch.float32 and _native_f32_gemv_bf16_enabled():
                        output = native.gemm_direct_mixed(
                            flat.to(torch.bfloat16),
                            self._ensure_mixed_descriptors(flat.device),
                            projection_count=len(self._linears),
                            total_n=self.output_size,
                            k=self.input_size,
                            cache_activation=_native_cache_activation_enabled(),
                        ).to(torch.float32)
                        return output.reshape(*hidden_states.shape[:-1], self.output_size)
                    output = native.gemm_direct_mixed(
                        flat,
                        self._ensure_mixed_descriptors(flat.device),
                        projection_count=len(self._linears),
                        total_n=self.output_size,
                        k=self.input_size,
                        cache_activation=_native_cache_activation_enabled(),
                    )
                    return output.reshape(*hidden_states.shape[:-1], self.output_size)
                return torch.cat(tuple(linear(hidden_states) for linear in self._linears), dim=-1)
            if _native_tensor_core_enabled() and _tensor_core_rows_enabled(
                flat.shape[0], f32=False
            ):
                return torch.cat(tuple(linear(hidden_states) for linear in self._linears), dim=-1)
            activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
            if activation_workspace is None:
                activation_workspace = native.quantize_q8_1(flat)
            if (
                flat.shape[0] == 1
                and os.environ.get("QSR_GGUF_NATIVE_Q8_MIXED", "1").strip() != "0"
            ):
                output = native.gemm_q8_mixed(
                    activation_workspace,
                    self._ensure_mixed_descriptors(flat.device),
                    projection_count=len(self._linears),
                    total_n=self.output_size,
                    k=self.input_size,
                )
            else:
                outputs = []
                for linear in self._linears:
                    packed, row_bytes, type_name = linear._native_q8_storage()
                    outputs.append(
                        native.gemm_q8_prequantized(
                            activation_workspace,
                            packed,
                            m=flat.shape[0],
                            n=linear.output_size,
                            k=linear.input_size,
                            row_bytes=row_bytes,
                            type_name=type_name,
                        )
                    )
                output = torch.cat(outputs, dim=-1)
        else:
            if flat.dtype == torch.float32:
                if flat.shape[0] == 1 and self._native_f32_q8_enabled():
                    activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
                    if activation_workspace is None:
                        activation_workspace = native.quantize_q8_1(flat)
                    if self._same_type:
                        packed = self._ensure_packed_weight(q8=True)
                        output = native.gemm_q8_prequantized(
                            activation_workspace,
                            packed,
                            m=1,
                            n=self.output_size,
                            k=self.input_size,
                            row_bytes=self._native_row_bytes,
                            type_name=self._native_type_name,
                            output_dtype=torch.float32,
                        )
                    else:
                        output = native.gemm_q8_mixed(
                            activation_workspace,
                            self._ensure_mixed_descriptors(flat.device),
                            projection_count=len(self._linears),
                            total_n=self.output_size,
                            k=self.input_size,
                            output_dtype=torch.float32,
                        )
                    return output.reshape(*hidden_states.shape[:-1], self.output_size)
                # The scalar F32 tiled kernel is a good exact GEMV, but its
                # large-M launch has a different register/cache balance when
                # two projections are concatenated.  Keep prefill/verify on
                # the independently tuned per-projection path; the merged
                # F32 ABI is a decode-only launch reduction.
                if flat.shape[0] != 1:
                    return torch.cat(
                        tuple(linear(hidden_states) for linear in self._linears), dim=-1
                    )
                packed = self._ensure_packed_weight(q8=False)
                if _native_f32_gemv_bf16_enabled():
                    output = native.gemm_direct(
                        flat.to(torch.bfloat16),
                        packed,
                        m=1,
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=self._tensor_core_row_bytes,
                        type_name=self._tensor_core_type_name,
                        cache_activation=_native_cache_activation_enabled(),
                    ).to(torch.float32)
                else:
                    output = native.gemm_direct(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=self._tensor_core_row_bytes,
                        type_name=self._tensor_core_type_name,
                        cache_activation=_native_cache_activation_enabled(),
                    )
            elif (
                (
                    _native_mmq_rows_enabled(flat.shape[0])
                    or (self.type_name == "Q8_0" and _native_mmq_q8_rows_enabled(flat.shape[0]))
                )
                and _native_mmq_shape_enabled(self.output_size, self.input_size)
                and (
                    (self.type_name == "Q5_K" and _native_mmq_q5_enabled())
                    or (self.type_name == "Q6_K" and _q6_split_weights_enabled())
                    or (
                        self.type_name == "Q8_0"
                        and _native_mmq_q8_enabled()
                        and all(
                            _native_mmq_q8_module_enabled(linear.module_name)
                            for linear in self._linears
                        )
                    )
                )
                and self._native_q8_enabled()
            ):
                packed = self._ensure_packed_weight(q8=True)
                if not _native_mmq_storage_enabled(self._native_type_name):
                    raise RuntimeError(f"unsupported MMQ storage layout {self._native_type_name!r}")
                activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
                if activation_workspace is None:
                    activation_workspace = native.quantize_q8_1(flat)
                output = native.gemm_q8_mmq(
                    activation_workspace,
                    packed,
                    m=flat.shape[0],
                    n=self.output_size,
                    k=self.input_size,
                    row_bytes=self._native_row_bytes,
                    type_name=self._native_type_name,
                )
            elif (
                _native_tensor_core_enabled()
                and _tensor_core_rows_enabled(flat.shape[0], f32=False)
                and self._tensor_core_storage_available()
            ):
                tile_major_storage = self._tensor_core_tile_major_storage(rows=flat.shape[0])
                if tile_major_storage is None:
                    packed = self._ensure_packed_weight(q8=False)
                    output = native.gemm_tensor_core(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=self._tensor_core_row_bytes,
                        type_name=self._tensor_core_type_name,
                    )
                else:
                    packed, block_n, type_name = tile_major_storage
                    output = native.gemm_tensor_core_tile_major(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        type_name=type_name,
                        block_n=block_n,
                    )
            elif self._native_q8_enabled():
                packed = self._ensure_packed_weight(q8=True)
                activation_workspace = _cached_q8_activation(native, flat, source=hidden_states)
                if activation_workspace is None:
                    output = native.gemm(
                        flat,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=self._native_row_bytes,
                        type_name=self._native_type_name,
                    )
                else:
                    output = native.gemm_q8_prequantized(
                        activation_workspace,
                        packed,
                        m=flat.shape[0],
                        n=self.output_size,
                        k=self.input_size,
                        row_bytes=self._native_row_bytes,
                        type_name=self._native_type_name,
                    )
            else:
                packed = self._ensure_packed_weight(q8=False)
                output = native.gemm_direct(
                    flat,
                    packed,
                    m=flat.shape[0],
                    n=self.output_size,
                    k=self.input_size,
                    row_bytes=self._tensor_core_row_bytes,
                    type_name=self._tensor_core_type_name,
                    cache_activation=_native_cache_activation_enabled(),
                )
        return output.reshape(*hidden_states.shape[:-1], self.output_size)


class GgufEmbedding(nn.Module):
    """Row-selecting GGUF embedding that never materializes the full table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        type_name: str,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if type_name not in GGUF_BLOCK_BYTES:
            raise ValueError(f"unsupported GGUF embedding type {type_name!r}")
        elements_per_block = 32 if type_name == "Q8_0" else 256
        if embedding_dim % elements_per_block:
            raise ValueError(
                f"GGUF {type_name} embedding dim {embedding_dim} is not a multiple of "
                f"the {elements_per_block}-element block size"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.type_name = type_name
        self.row_bytes = (embedding_dim // elements_per_block) * GGUF_BLOCK_BYTES[type_name]
        self.output_dtype = output_dtype
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * self.row_bytes, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.dtype != torch.uint8 or loaded_weight.numel() != param.numel():
            raise ValueError(
                f"GGUF {self.type_name} embedding expects {param.numel()} uint8 values, got "
                f"{loaded_weight.numel()} {loaded_weight.dtype}"
            )
        param.data.copy_(loaded_weight.reshape_as(param))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.long().reshape(-1)
        if flat_ids.numel() == 0:
            return torch.empty(
                (*input_ids.shape, self.embedding_dim),
                device=input_ids.device,
                dtype=self.output_dtype,
            )
        # ``min()``/``max()`` materialize a device scalar on the host.  That
        # is a useful eager-input guard, but CUDA forbids the synchronization
        # while a graph is being captured.  Graph replay owns fixed buffers
        # whose ids were validated by the eager warmup, so omit only this
        # host-side check during capture.
        if not input_ids.is_cuda or not torch.cuda.is_current_stream_capturing():
            if flat_ids.min() < 0 or flat_ids.max() >= self.num_embeddings:
                raise IndexError("GGUF embedding input id is outside the vocabulary")
        if (
            input_ids.is_cuda
            and self.output_dtype in (torch.bfloat16, torch.float32)
            and _native_gguf_enabled()
            and self.type_name in _NATIVE_GGUF_TYPES
        ):
            return (
                _native_library()
                .dequant_rows(
                    flat_ids.contiguous(),
                    self.weight,
                    rows=flat_ids.numel(),
                    k=self.embedding_dim,
                    row_bytes=self.row_bytes,
                    type_name=self.type_name,
                    dtype=self.output_dtype,
                )
                .reshape(*input_ids.shape, self.embedding_dim)
            )
        rows = self.weight.reshape(self.num_embeddings, self.row_bytes).index_select(0, flat_ids)
        result = dequantize_gguf_packed(
            rows.reshape(-1),
            (flat_ids.numel(), self.embedding_dim),
            self.type_name,
            dtype=self.output_dtype,
        )
        return result.reshape(*input_ids.shape, self.embedding_dim)
