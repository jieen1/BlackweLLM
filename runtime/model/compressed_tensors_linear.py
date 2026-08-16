"""Self-built Linear layers for compressed-tensors "mixed-precision"
checkpoints (Track B, unsloth's ``unsloth/Qwen3.6-27B-NVFP4``). Sibling of
``runtime/model/modelopt_linear.py`` -- same per-Parameter ``weight_loader``
closure idiom, same "dequantize once to BF16, cache, run BF16xBF16
``F.linear``" B1 scope decision (see that module's docstring for why: this
runtime does not reproduce either checkpoint's *intended* quantized GEMM
execution path, correctness-first), different checkpoint naming and,
for FP8, a genuinely different physical scale layout.

Both classes' Parameter attribute names are the checkpoint's own tensor
suffixes verbatim (``weight``/``weight_scale`` for FP8;
``weight_packed``/``weight_scale``/``weight_global_scale`` for NVFP4) --
``runtime/model/qwen36_model.py``'s ``load_weights`` does no per-tensor name
remapping for the backbone, only a fixed top-level prefix strip, so this has
to line up 1:1 exactly like ``ModelOptFP8Linear``/``ModelOptNVFP4Linear``
already do for the other format.

See ``runtime/loading/compressed_tensors.py``'s module docstring for the
measured evidence (real safetensors headers, 2026-08-02) that:

- unsloth's FP8 scale is per-output-*channel* (``[out, 1]``, ``bfloat16``),
  not modelopt's per-*tensor* scalar (``float32``) -- a different function,
  :func:`~runtime.loading.compressed_tensors.dequantize_fp8_channel`, not a
  shape-tolerant variant of ``runtime.loading.modelopt.dequantize_fp8``.
- unsloth's NVFP4 sub-format (``config_groups`` ``format:
  "nvfp4-pack-quantized"``) is the same physical layout as modelopt's
  ``W4A16_NVFP4`` (block size 16, E2M1 codes, two-level scaling) -- so
  :class:`CompressedTensorsNVFP4Linear` reuses
  ``runtime.loading.modelopt.dequantize_nvfp4`` unchanged rather than
  reimplementing the unpack/dequant math, and only ``.input_global_scale``
  (this checkpoint's activation-side scale, never read here -- same "B1
  dequantizes weights only" reasoning as ``.input_scale`` for modelopt) is
  new relative to that format.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from runtime.loading.compressed_tensors import dequantize_fp8_channel
from runtime.loading.modelopt import NVFP4_GROUP_SIZE, dequantize_nvfp4
from runtime.model._weight_loading import default_weight_loader

#: Pre-flight-only diagnostic switch (2026-08-03, FP8 W8A8 B1-R pre-flight --
#: see ``scripts/verify_fp8_w8a8_activation_emulation_single_layer.py`` and
#: ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py``, the
#: two consumers this flag exists for). Default OFF: production must never
#: pay this, and must never even risk paying it
#: through a stale env var left set in a shell -- ``forward()`` below reads
#: this at call time (not cached at import time) specifically so a test can
#: toggle it with ``monkeypatch.setenv``/``delenv`` and see the effect
#: immediately, matching this codebase's existing ``QSR_*`` flag idiom (e.g.
#: ``runtime/backends/laguna.py``'s ``QSR_PROFILE_MOE_PHASES``).
QSR_EMULATE_FP8_ACTIVATION_ENV = "QSR_EMULATE_FP8_ACTIVATION"
QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV = "QSR_TORCH_SCALED_MM_FP8_CHANNEL"
QSR_NATIVE_W8A8_FP8_CHANNEL_ENV = "QSR_NATIVE_W8A8_FP8_CHANNEL"
QSR_NATIVE_W8A8_QUANT_ENV = "QSR_NATIVE_W8A8_QUANT"

_native_w8a8_library: object | None = None
_native_w8a8_quantizer_unavailable = False


def _fp8_activation_emulation_enabled() -> bool:
    return os.environ.get(QSR_EMULATE_FP8_ACTIVATION_ENV) == "1"


def _torch_scaled_mm_fp8_channel_enabled() -> bool:
    # Qwen3.6 declares this dynamic per-token / per-channel E4M3 contract.
    # CUDA serving therefore uses raw W8A8 by default. ``0`` remains solely
    # for CPU tests and explicit fallback diagnostics.
    return os.environ.get(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV, "all") in {"1", "all"}


def _native_w8a8_fp8_channel_enabled() -> bool:
    # Combined historical-kernel mode: W4A4 NVFP4 MLP (sparkinfer) + native
    # W8A8 FP8 dense (historical CUTLASS port). Measured 2026-08-04 W1-S
    # c=4 twice: wall 33.3/33.2 s vs 58.7-60.2 s baseline, acceptance
    # 72.3% (historical anchor 70.29%) -- default ON since then. ``0``
    # stays the diagnostic fallback back to the torch._scaled_mm path.
    if os.environ.get("QSR_NATIVE_W8A8_FP8_CHANNEL", "1") not in {"1", "all"}:
        return False
    return _native_w8a8_artifact_usable()


def _native_w8a8_lm_head_enabled() -> bool:
    """Per-shape native routing for the 248,320-wide lm_head only.

    Measured 2026-08-04 (RTX PRO 6000 Blackwell, M=4/16, real-shape
    microbench): the self-owned kernel streams lm_head's 2.54 GiB weights in
    3.1 ms vs torch._scaled_mm's 3.8-4.7 ms, while the blanket all-shapes
    native switch measured slightly worse e2e -- hence shape-scoped.
    """
    return os.environ.get("QSR_NATIVE_W8A8_LM_HEAD") == "1"


def _native_w8a8_quantization_enabled() -> bool:
    """Return whether to trial the self-owned quantizer before Torch's GEMM.

    This remains diagnostic-only until full-model agreement is established.
    It must never select a BF16 weight materialization.
    """
    return os.environ.get(QSR_NATIVE_W8A8_QUANT_ENV) == "1"


def fp8_channel_raw_execution_uses_all_layers() -> bool:
    """Return whether the experimental W8A8 route owns every FP8 Linear.

    ``1`` is the original narrow MLP-only experiment. ``all`` is the full
    historical W8A8 contract: all FP8-channel modules consume raw E4M3
    weights, so model loading must not create BF16 caches for any of them.
    """
    return (
        _torch_scaled_mm_fp8_channel_enabled()
        and os.environ.get(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV, "all") != "1"
    ) or os.environ.get(QSR_NATIVE_W8A8_FP8_CHANNEL_ENV) == "all"


def _native_w8a8_artifact_usable() -> bool:
    """Whether the self-owned W8A8 artifact can load on this machine.

    A default-ON native route must degrade to the torch._scaled_mm path on
    an unbuilt checkout instead of crashing at the first CUDA forward.
    """
    try:
        _native_w8a8_library_for_cuda()
        return True
    except RuntimeError:
        return False


def _native_w8a8_library_for_cuda() -> object:
    """Load the explicit self-owned W8A8 artifact exactly once per process."""
    global _native_w8a8_library
    if _native_w8a8_library is None:
        from runtime.fp8_w8a8 import NativeFP8W8A8Library

        _native_w8a8_library = NativeFP8W8A8Library.load()
    return _native_w8a8_library


def _native_w8a8_quantizer_for_cuda() -> object | None:
    """Return the self-owned fused quantizer when its optional artifact exists.

    The default GEMM path intentionally remains ``torch._scaled_mm`` because
    it is materially more accurate than the current self-owned GEMM for
    multi-token rows.  Its input quantizer is independent, bit-compatible
    with the historical per-token E4M3 contract, and can therefore be used
    whenever the separately built raw-pointer artifact is present.  A source
    checkout remains runnable before that optional build by using the pure
    Torch quantizer below; it never falls back to a BF16 weight cache.
    """
    global _native_w8a8_quantizer_unavailable
    if _native_w8a8_quantizer_unavailable:
        return None
    try:
        return _native_w8a8_library_for_cuda()
    except RuntimeError:
        _native_w8a8_quantizer_unavailable = True
        return None


def _quantize_fp8_activation_for_torch_scaled_mm(
    x: torch.Tensor, input_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one CUDA W8A8 activation with the historical fused contract.

    The self-owned quantizer writes E4M3 codes and FP32 per-token scales in
    one launch.  It is an explicit diagnostic candidate only: full-model
    agreement must be established before it can replace the default pure-
    Torch quantizer.  Either branch preserves the same raw-FP8 route.
    """
    x_2d = x.reshape(-1, input_size).contiguous()
    if not _native_w8a8_quantization_enabled():
        return quantize_fp8_activation_per_token(x_2d)
    library = _native_w8a8_quantizer_for_cuda()
    if library is None:
        return quantize_fp8_activation_per_token(x_2d)
    x_fp8 = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x.device)
    library.quantize_per_token(x_2d, x_fp8, activation_scale)
    return x_fp8, activation_scale


def emulate_fp8_activation_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Per-token dynamic FP8 (E4M3) activation quantize/dequantize
    round-trip -- emulates the DOMINANT new error term a genuine W8A8 GEMM
    would add over today's dequantize-weight-to-BF16-and-``F.linear`` path,
    without building an FP8xFP8 kernel.

    Matches this checkpoint's own declared scheme for every FP8-channel
    target (verified directly against the standard checkpoint's
    ``config.json``, 2026-08-03: ``quantization_config.config_groups.
    group_0.input_activations`` = ``{num_bits: 8, type: float, strategy:
    "token", dynamic: true, symmetric: true}``) -- one scale per row (per
    token), derived from that row's own max-abs value, never loaded from
    the checkpoint (there is no ``.input_scale`` Parameter for this group;
    see :class:`CompressedTensorsFP8ChannelLinear`'s docstring for why).
    This is the compressed-tensors library's own standard per-token E4M3
    recipe (``scale = amax / fp8_max``, symmetric, round-to-nearest via the
    ``float8_e4m3fn`` cast, no zero-point), the same convention
    ``scripts/verify_fp8_tensor_gemm_single_layer.py::quantize_activation_fp8``
    already used for modelopt's *static per-tensor* FP8 scheme -- here
    applied per-row instead of once for the whole tensor, because this
    checkpoint's activation scale is dynamic and per-token, not static and
    per-tensor (a different checkpoint, a different declared scheme -- see
    that script's own docstring for why the two are not interchangeable).

    This is a **lower bound** on real W8A8's error, not an equivalent of
    it: a genuine ``sparkinfer``-style FP8xFP8 GEMM would also change
    accumulation order relative to today's BF16xBF16 ``F.linear``, which
    this round-trip does not touch (only the activation values change; the
    GEMM itself still runs in BF16). See this module's docstring and
    ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py`` for
    what this is being used to decide (a B1-R gap-error gate) and why a
    lower bound is sufficient for a negative verdict there.
    """
    x_fp8, scale = quantize_fp8_activation_per_token(x)
    return (x_fp8.to(torch.float32) * scale).to(x.dtype)


def quantize_fp8_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 activation codes and their per-token dequant scales.

    The split form is deliberately public because an actual W8A8 GEMM needs
    the codes as its left operand while applying the scale after the raw dot
    product.  Keeping this calculation shared with
    :func:`emulate_fp8_activation_round_trip` prevents the diagnostic and
    kernel-preflight paths from silently using different quantization rules.
    """
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
    x32 = x.to(torch.float32)
    amax = x32.abs().amax(dim=-1, keepdim=True)
    # Match the historical dynamic-FP8 quantizer's nonzero lower scale
    # bound.  It prevents tiny rows from underflowing their reciprocal scale
    # while leaving normal token rows unchanged.
    scale = (amax / fp8_max).clamp_min(1.0 / (fp8_max * 512.0))
    x_fp8 = (x32 / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return x_fp8, scale


class CompressedTensorsFP8ChannelLinear(nn.Module):
    """Per-output-channel E4M3 Linear with raw W8A8 CUDA serving.

    Checkpoint shape (verified against real safetensors headers,
    2026-08-02): ``weight`` is ``[out, in]`` ``float8_e4m3fn`` (unpacked, one
    byte per element -- same physical layout as modelopt's FP8 weight
    tensor); ``weight_scale`` is ``[out, 1]`` ``bfloat16`` -- one scale per
    output row, unlike modelopt's single ``float32`` scalar. ``.input_scale``
    does not exist for this group (``input_activations.dynamic: true`` in
    the checkpoint's own config -- a real per-token dynamic activation
    scale, computed at runtime for the intended FP8xFP8 GEMM this B1
    implementation does not perform), so there is nothing to ignore here the
    way modelopt's ``.input_scale`` is.

    **Legacy BF16 fallback / FP8 W8A8 pre-flight (2026-08-03,**
    ``QSR_EMULATE_FP8_ACTIVATION`` **env flag, default OFF)**: the fallback
    optionally round-trips the
    activation through :func:`emulate_fp8_activation_round_trip` before
    ``F.linear`` -- a cheap way to measure a genuine W8A8 GEMM's *dominant*
    new error source (activation quantization) without building an FP8xFP8
    kernel. CUDA serving uses the checkpoint's fused W8A8 arithmetic and
    preserves raw FP8 matrices instead. See
    :func:`emulate_fp8_activation_round_trip`'s docstring for the scheme
    and why it is a lower, not exact, bound, and
    ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py`` for
    the full-model verdict this flag was built to produce. Default OFF
    means this can never affect production; nothing sets this env var
    except a diagnostic script or a test.
    """

    def __init__(self, input_size: int, output_size: int, *, bias: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, 1, dtype=torch.bfloat16), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        if self.bias is not None:
            self.bias.weight_loader = default_weight_loader

        self._weight_bf16: torch.Tensor | None = None
        # Deliberately separate from the production BF16 cache.  This pack is
        # only built by the explicit single-layer W8A8 preflight below; the
        # normal model never pays its memory or compilation cost.
        self._fp8_channel_packed_weight: object | None = None
        self._fp8_channel_fused_packed_weight: object | None = None
        self._fp8_channel_kernel_weight_scale: torch.Tensor | None = None
        # ``torch._scaled_mm`` uses the same immutable FP32 channel-scale
        # ABI as the native kernel. Do not recast this BF16 vector for every
        # one of Qwen's 233 FP8 projections on every decode/verify replay.
        self._torch_w8a8_weight_scale: torch.Tensor | None = None
        # FP32 is the native epilogue ABI.  This is one scalar per output
        # channel, not a materialized BF16 copy of the E4M3 weight matrix.
        self._native_w8a8_weight_scale: torch.Tensor | None = None
        # The raw ABI deliberately takes caller-owned workspace.  Keying it
        # by stream and launch geometry keeps a graph's workspace address
        # stable and prevents two streams from racing on a process-global
        # scratch allocation.
        self._native_w8a8_workspaces: dict[tuple[int, int, int, int, bool], torch.Tensor] = {}

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            self._weight_bf16 = dequantize_fp8_channel(self.weight.data, self.weight_scale.data)

    def free_fp8_raw_weight(self) -> None:
        """Drop the FP8 ``.weight`` storage once ``_weight_bf16`` exists.

        The FP8 original and its BF16 dequantization are both resident
        otherwise, and :meth:`forward` reads only the BF16 one. Measured on
        the standard checkpoint (``notes/2026-08-03-production-memory-audit.md``):
        237 FP8 weight tensors, 10.73B parameters, so the originals are
        9.99 GiB held for nothing once the cache is built.

        This is the FP8 half of a fix that was only ever applied to the NVFP4
        half. ``free_nvfp4_raw_params`` took the resident set from 76.34 to
        53.08 GiB by doing exactly this for the 56 NVFP4 MLP layers, and
        stopped there -- the other 237 layers kept both copies, unremarked,
        because nothing measured the split until that audit.

        Deliberately NOT called from :meth:`_ensure_ready`. The dequantization
        happens lazily on first forward, and several diagnostic scripts build
        one of these layers precisely to compare the FP8 original against a
        dequantized or emulated path (``scripts/verify_fp8_*``); freeing
        automatically would break them at a distance for a memory win they do
        not need. Model-level code calls this once, after warmup -- same
        division of labour as ``Qwen36MLP._free_raw_nvfp4_weights``.

        Safe to call before the cache exists (materializes it first) and safe
        to call more than once. Reassigns ``.data`` to a 0-element tensor on
        the same device rather than deleting the Parameter, so
        ``named_parameters()`` keeps its shape and nothing that walks the
        module tree trips over a missing entry.
        """
        self._ensure_ready()
        self.weight.data = self.weight.data.new_empty(0)
        self.release_fp8_channel_kernel()

    def release_fp8_channel_kernel(self) -> None:
        """Release the opt-in raw-FP8 GEMM preflight cache, if materialized."""
        self._fp8_channel_packed_weight = None
        self._fp8_channel_fused_packed_weight = None
        self._fp8_channel_kernel_weight_scale = None

    def prepare_fp8_channel_kernel(self) -> None:
        """Pack raw FP8 weights for the explicit dynamic-W8A8 preflight.

        ``tensor_fp8_linear`` normally applies one static scalar in its
        epilogue.  Passing a unit scalar exposes its raw FP8 dot product;
        :meth:`forward_fp8_channel_kernel` then applies this checkpoint's
        dynamic per-token activation scale and per-output-channel weight
        scale outside that operation.  This is *not* wired into
        :meth:`forward`: it exists solely to establish numerical and
        performance evidence before changing the serving route.
        """
        if self.weight.data.numel() == 0:
            raise RuntimeError(
                "raw FP8 weight was released; prepare the experimental kernel before "
                "free_fp8_raw_weight()"
            )
        if self.weight.device.type != "cuda":
            raise RuntimeError("FP8-channel kernel preflight requires CUDA-resident weights")
        if self._fp8_channel_packed_weight is not None:
            return

        from b12x.gemm import tensor_fp8_linear

        unit_output_scale = torch.ones(1, dtype=torch.float32, device=self.weight.device)
        self._fp8_channel_packed_weight = tensor_fp8_linear.pack_weight(
            self.weight.data, unit_output_scale
        )
        weight_scale = self.weight_scale.data.to(torch.float32)
        self._fp8_channel_kernel_weight_scale = weight_scale.reshape(1, self.output_size)
        try:
            from b12x.gemm import tensor_fp8_channel_linear
        except ImportError:
            self._fp8_channel_fused_packed_weight = None
        else:
            self._fp8_channel_fused_packed_weight = tensor_fp8_channel_linear.pack_weight(
                self.weight.data,
                weight_scale.reshape(self.output_size),
            )

    def forward_fp8_channel_kernel(
        self,
        x: torch.Tensor,
        *,
        expected_m: int | None = None,
    ) -> torch.Tensor:
        """Run the explicit raw-FP8 GEMM composition for one preflight call.

        The intended checkpoint arithmetic is ``(round(x / a) @ W_fp8.T) *
        a * w`` where ``a`` is dynamic per token and ``w`` is static per
        output channel.  For single-row ``M=1`` inputs, the optional fused
        SparkInfer channel-scale epilogue applies ``w`` in-kernel.  Larger
        ``M`` falls back to the existing raw-dot-product preflight and
        preserves both scales as explicit post-GEMM multiplies.  This
        remains intentionally unsuitable for graph replay or production
        until a direct full-model correctness gate accepts it.
        """
        if x.device != self.weight.device:
            raise ValueError("activation and FP8-channel weight must share a device")
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"activation K={x.shape[-1]} does not match weight K={self.input_size}"
            )
        self.prepare_fp8_channel_kernel()
        assert self._fp8_channel_packed_weight is not None
        assert self._fp8_channel_kernel_weight_scale is not None

        x_shape = x.shape
        x_2d = x.reshape(-1, self.input_size).contiguous()
        x_fp8, activation_scale = quantize_fp8_activation_per_token(x_2d)
        if x_fp8.shape[0] == 1 and self._fp8_channel_fused_packed_weight is not None:
            from b12x.gemm import tensor_fp8_channel_linear

            output = tensor_fp8_channel_linear.mm(
                x_fp8,
                self._fp8_channel_fused_packed_weight,
                activation_scale.reshape(1),
                out_dtype=torch.bfloat16,
                expected_m=expected_m,
            ).to(x.dtype)
        else:
            from b12x.gemm import tensor_fp8_linear

            raw_output = tensor_fp8_linear.mm(
                x_fp8,
                self._fp8_channel_packed_weight,
                out_dtype=torch.bfloat16,
                expected_m=expected_m,
            )
            output = (
                raw_output.float() * activation_scale * self._fp8_channel_kernel_weight_scale
            ).to(x.dtype)
        output = output.view(*x_shape[:-1], self.output_size)
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward_torch_scaled_mm(self, x: torch.Tensor) -> torch.Tensor:
        """Run CUDA's fused per-token/per-channel FP8 scaled GEMM.

        The GEMM consumes original E4M3 checkpoint storage directly.  Its
        default activation quantizer is pure Torch; the self-owned fused
        implementation is an explicit diagnostic candidate that preserves
        the same raw-FP8 contract but is not default until full-model
        agreement is established.
        """
        if x.device != self.weight.device or x.device.type != "cuda":
            raise RuntimeError("torch scaled_mm FP8-channel path requires CUDA co-resident tensors")
        if self.weight.data.numel() == 0:
            raise RuntimeError("raw FP8 weight was released; scaled_mm path is unavailable")
        x_fp8, activation_scale = _quantize_fp8_activation_for_torch_scaled_mm(x, self.input_size)
        if self._torch_w8a8_weight_scale is None:
            self._torch_w8a8_weight_scale = self.weight_scale.data.t().to(torch.float32)
        output = torch._scaled_mm(
            x_fp8,
            self.weight.data.t(),
            scale_a=activation_scale,
            scale_b=self._torch_w8a8_weight_scale,
            out_dtype=x.dtype,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = output.view(*x.shape[:-1], self.output_size)
        return output if self.bias is None else output + self.bias

    def forward_native_w8a8(self, x: torch.Tensor) -> torch.Tensor:
        """Run the self-owned SM120 W8A8 E4M3 path without BF16 weights.

        Scratch is caller-owned rather than extension-global.  An eager
        warmup keeps it outside graph capture; if PyTorch switches to its
        private capture stream, creating the stream-specific tensor during
        capture is also safe because the graph owns that allocation for all
        replays.
        """
        if x.device != self.weight.device or x.device.type != "cuda":
            raise RuntimeError("native W8A8 FP8-channel path requires CUDA co-resident tensors")
        if self.weight.data.numel() == 0:
            raise RuntimeError("raw FP8 weight was released; native W8A8 path is unavailable")
        library = _native_w8a8_library_for_cuda()
        x_shape = x.shape
        x_2d = x.reshape(-1, self.input_size).contiguous()
        x_fp8 = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
        activation_scale = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x.device)
        library.quantize_per_token(x_2d, x_fp8, activation_scale)
        if self._native_w8a8_weight_scale is None:
            self._native_w8a8_weight_scale = self.weight_scale.data.t().to(torch.float32)
        output = torch.empty((x_2d.shape[0], self.output_size), dtype=x.dtype, device=x.device)
        geometry = (x_2d.shape[0], self.output_size, self.input_size, False)
        stream_id = torch.cuda.current_stream(x.device).cuda_stream
        workspace_key = (stream_id, *geometry)
        workspace = self._native_w8a8_workspaces.get(workspace_key)
        if workspace is None:
            workspace_bytes = library.workspace_bytes(
                m=geometry[0], n=geometry[1], k=geometry[2], batch_invariant=False
            )
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=x.device)
            self._native_w8a8_workspaces[workspace_key] = workspace
        library.launch(
            x_fp8,
            self.weight.data.t(),
            activation_scale,
            self._native_w8a8_weight_scale,
            output,
            workspace,
            batch_invariant=False,
        )
        output = output.view(*x_shape[:-1], self.output_size)
        return output if self.bias is None else output + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``1`` preserves the original narrow 17,408-wide MLP experiment.
        # ``all`` is the complete historical W8A8 candidate and remains
        # opt-in until every small attention/GDN geometry is qualified.
        # Per-shape routing exception: ``QSR_NATIVE_W8A8_LM_HEAD=1`` sends
        # ONLY the 248,320-wide lm_head through the self-owned kernel --
        # measured 2026-08-04 at M=4/16 (decode/verify shapes): 3.1 ms vs
        # torch._scaled_mm 3.8-4.7 ms, while the blanket all-shapes native
        # switch measured slightly WORSE e2e. lm_head M=1 numerics were
        # previously verified max_abs=0 vs the historical cutlass_scaled_mm.
        # Both native branches are CUDA-only routes; CPU tensors must fall
        # through to the BF16 dequant fallback even on a machine where the
        # self-owned artifact is built (the enabled() gates cannot see the
        # input device, and CI/fallback diagnostics run CPU-only).
        if (
            x.device.type == "cuda"
            and _native_w8a8_lm_head_enabled()
            and self.output_size == 248320
        ):
            return self.forward_native_w8a8(x)
        if (
            x.device.type == "cuda"
            and _native_w8a8_fp8_channel_enabled()
            and (self.output_size == 17408 or fp8_channel_raw_execution_uses_all_layers())
        ):
            return self.forward_native_w8a8(x)
        if (
            x.device.type == "cuda"
            and _torch_scaled_mm_fp8_channel_enabled()
            and (self.output_size == 17408 or fp8_channel_raw_execution_uses_all_layers())
        ):
            return self.forward_torch_scaled_mm(x)
        self._ensure_ready()
        if _fp8_activation_emulation_enabled():
            x = emulate_fp8_activation_round_trip(x)
        return F.linear(x, self._weight_bf16, self.bias)


class CompressedTensorsNVFP4Linear(nn.Module):
    """Block-scaled NVFP4 (E2M1) weight-only-quantized Linear, dequantized to
    BF16 on first use -- unsloth's ``config_groups`` ``format:
    "nvfp4-pack-quantized"`` group (same format string, same physical byte
    layout, as Laguna's own NVFP4 -- ``runtime/backends/laguna_sparkinfer_moe.py``
    reads the identical three suffixes for its MoE experts, a separate
    pipeline; this class is the dense/Linear-path consumer for this format).

    Checkpoint shape (verified against real safetensors headers,
    2026-08-02): ``weight_packed`` is ``[out, in // 2]`` ``uint8`` (two
    4-bit E2M1 codes/byte); ``weight_scale`` is ``[out, in // group_size]``
    ``float8_e4m3fn`` (one scale per 16-element input-dim block, identical
    shape/dtype/value-range to modelopt's own NVFP4 block scale -- both
    checkpoints' real ``layers.0.mlp.gate_proj.weight_scale`` measure
    min=4.5/max=448/mean~22-26). ``weight_global_scale`` is shape ``[1]``
    ``float32``. A fourth tensor, ``input_global_scale``, also exists per
    module and is never read here -- see this module's own docstring.

    **``weight_global_scale`` is the RECIPROCAL of modelopt's
    ``weight_scale_2``, not the same value under a different name --
    measured, not assumed (2026-08-03, after a real GPU run of this class
    produced degenerate ``"!!!!!!!!!!!!"`` output; per-layer hidden-state
    hooks showed the very first MLP layer already saturating to ~4e22).**
    Real checkpoint values for the same module (``layers.0.mlp.gate_proj``):
    unsloth's ``weight_global_scale`` = ``6624.0``; nvidia's
    ``weight_scale_2`` = ``0.0002`` -- and ``1 / 6624 ≈ 0.000151``, the same
    order of magnitude, not a coincidence. This matches, and is explained
    by, the convention ``runtime/backends/laguna_sparkinfer_moe.py``'s own
    module docstring already documents for this exact checkpoint-side
    tensor name in Laguna's MoE pipeline: ``w1_global_scale = 1 /
    checkpoint_gs`` (its ``prepare_sparkinfer_layer`` literally computes
    ``w1_alpha = (1.0 / raw["gate_gs"]).float()`` before handing it to
    sparkinfer's kernel as the multiplicative alpha) -- this class's
    ``_ensure_ready`` does the same reciprocal before calling
    :func:`~runtime.loading.modelopt.dequantize_nvfp4`, which itself
    performs a direct multiply (``per_block = weight_scale * global_scale``,
    proven correct for modelopt's checkpoint, whose ``weight_scale_2`` is
    already stored as the direct multiplier). Passing
    ``weight_global_scale`` to that function unreciprocated -- reusing its
    math but not its checkpoint-side calling convention -- is exactly the
    bug this docstring is recording so it cannot regress silently:
    dequantizing unsloth's real ``layers.0.mlp.gate_proj`` weight the wrong
    way measured mean=247.8/std=426744 (nonsense for a neural net weight);
    with the reciprocal applied it lands at std≈0.0097, the same order of
    magnitude as a normal weight. ``default_weight_loader``'s scalar-numel
    special case handles ``weight_global_scale``'s ``[1]``-vs-modelopt's
    ``()`` shape difference transparently -- that part was never the issue.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        group_size: int = NVFP4_GROUP_SIZE,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(f"NVFP4 packs 2 elements/byte; input_size={input_size} must be even")
        if input_size % group_size != 0:
            raise ValueError(
                f"input_size={input_size} must be a multiple of group_size={group_size}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.group_size = group_size

        self.weight_packed = nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(output_size, input_size // group_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_global_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        # Activation-side static global scale for this checkpoint's genuine
        # W4A4 scheme (`config_groups.group_1.input_activations`: num_bits=4,
        # strategy=tensor_group, group_size=16, dynamic="local" -- a
        # calibrated per-tensor scale with the per-block e4m3 scale computed
        # at runtime, exactly `quantize_grouped_nvfp4_torch`'s two-level
        # design). Loaded but not read by this class's own `forward()` --
        # same "checkpoint data present, not this class's job" split as
        # `weight_global_scale` before this Parameter existed (see class
        # docstring). Consumer is `nvfp4_w4a4_components_for_fuse` below,
        # used only by the diagnostic scripts
        # `scripts/verify_nvfp4_w4a4_gemm_single_layer.py` /
        # `scripts/verify_nvfp4_w4a4_gemm_full_model_gap.py`
        # (2026-08-03, ``work/w4a4-20260803``): routing this checkpoint's
        # MLP through a genuine W4A4 ``sparkinfer.gemm.blockscaled.mm`` GEMM
        # measured cosine ~0.988 at the single-layer level (vs ~0.99999 for
        # the production W4A16 path) and **failed B1-R's calibrated
        # gap-error bars** at the full-model level (median/p90/p90-logprob
        # all over their bars; one of three workloads diverged badly enough
        # to overflow the diagnostic's top-1024 capture window). Not wired
        # into `Qwen36MLP.forward` for that reason -- this Parameter and the
        # method below are kept because they are correct, real checkpoint
        # data, and load-bearing evidence for that negative result, not
        # because anything reads them in production.
        self.input_global_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.weight_packed.weight_loader = default_weight_loader
        self.weight_scale.weight_loader = default_weight_loader
        self.weight_global_scale.weight_loader = default_weight_loader
        self.input_global_scale.weight_loader = default_weight_loader
        if self.bias is not None:
            self.bias.weight_loader = default_weight_loader

        self._weight_bf16: torch.Tensor | None = None

    def _ensure_ready(self) -> None:
        if self._weight_bf16 is None:
            # Reciprocal, not a direct pass-through -- see class docstring
            # for the measured evidence (unsloth weight_global_scale=6624.0
            # vs modelopt weight_scale_2=0.0002 for the same real module;
            # 1/6624 ≈ 0.000151, same order of magnitude) and for why this
            # is a real checkpoint-convention difference, not a shape quirk
            # dequantize_nvfp4 already handles.
            reciprocal_global_scale = 1.0 / self.weight_global_scale.data.to(torch.float32)
            self._weight_bf16 = dequantize_nvfp4(
                self.weight_packed.data,
                self.weight_scale.data,
                reciprocal_global_scale,
                group_size=self.group_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        return F.linear(x, self._weight_bf16, self.bias)

    def nvfp4_components_for_fuse(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(packed_weight, block_scale, global_scale)`` for
        ``Qwen36MLP``'s fused w13/w2 W4A16 path (``runtime/model/
        qwen36_model.py``) -- packed weight ``[out, in // 2]`` uint8, block
        scale ``[out, in // group_size]`` float8_e4m3fn, global scale a
        scalar float32 in :class:`~runtime.model.modelopt_linear.
        ModelOptNVFP4Linear`'s convention (a direct multiplier -- see
        :meth:`~runtime.model.modelopt_linear.ModelOptNVFP4Linear.
        nvfp4_components_for_fuse`'s docstring), NOT this checkpoint's own
        ``weight_global_scale`` convention. The reciprocal happens here,
        once, so every caller downstream of this method (in particular
        ``sparkinfer.moe._shared.kernels.w4a16.prepare.
        prepare_w4a16_modelopt_nvfp4_weights``, whose docstring literally
        says "raw ModelOpt weight global scales") can stay written in the
        single convention it already expects, exactly like this class's own
        :meth:`_ensure_ready` already does for its dequant-to-BF16 path --
        see this class's docstring for the measured evidence that skipping
        this reciprocal produces degenerate output (``"!!!!!!!!!!!!"``,
        2026-08-03).
        """
        reciprocal_global_scale = 1.0 / self.weight_global_scale.data.reshape(()).to(torch.float32)
        return self.weight_packed.data, self.weight_scale.data, reciprocal_global_scale

    def nvfp4_w4a4_components_for_fuse(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(packed_weight, block_scale, weight_global_scale,
        input_global_scale)`` **verbatim, unreciprocated**, for a genuine
        W4A4 ``sparkinfer.gemm.blockscaled.mm`` GEMM (both operands
        pre-quantized) -- a different convention from
        :meth:`nvfp4_components_for_fuse` above, which reciprocates
        ``weight_global_scale`` to match ``dequantize_nvfp4``'s
        direct-multiplier convention for the W4A16 dequant-inside-kernel
        path.

        **Not called from ``Qwen36MLP.forward`` or any other production
        path** -- only from the two diagnostic scripts named in this
        Parameter's own declaration comment above. The W4A4 investigation
        this method was built for (2026-08-03, ``work/w4a4-20260803``)
        concluded negatively: correct as far as it goes (this convention IS
        the one that produces sane output, see below), but genuinely
        quantizing the activation to 4-bit NVFP4 costs enough precision,
        compounded over 56 MLP layers x 3 GEMMs each, to fail B1-R's
        calibrated gap-error bars at the full-model level. Kept for the
        record and in case a future, more precise activation-quantization
        recipe revisits this -- not because production reads it.

        ``blockscaled.mm``'s own operand-building convention (matching
        ``sparkinfer._lib.intrinsics.quantize_grouped_nvfp4_torch`` and its
        oracle test, ``tests/gemm/test_blockscaled.py::_make_quantized_operand``)
        is: ``global_scale`` is the value that maps a block's real amax onto
        the e4m3 grid (``scale = global_scale * block_max / 6``, clipped),
        and the GEMM's ``alpha = 1 / (weight_gs * activation_gs)`` undoes
        *both* operands' global scales in one shot after the block-scaled
        MMA has already applied each operand's per-block e4m3 scale
        in-kernel. That is ``weight_global_scale`` used directly, not its
        reciprocal -- confirmed by working the checkpoint's own arithmetic
        backward from :meth:`nvfp4_components_for_fuse` (whose reciprocal is
        independently verified correct for the BF16-dequant path): if
        ``dequantize_nvfp4``'s ``per_block = weight_scale * (1 /
        weight_global_scale)`` is the right per-element multiplier, and
        ``quantize_grouped_nvfp4_torch`` defines that same multiplier as
        ``scale / global_scale``, then ``weight_global_scale ==
        global_scale`` (the un-reciprocated quantizer-side value) --
        checked against a real layer's numbers in
        ``scripts/verify_nvfp4_w4a4_gemm_single_layer.py`` (GPU, real
        checkpoint weights), not just derived on paper.

        ``input_global_scale`` (this checkpoint's genuine W4A4 activation
        scale, ``config_groups.group_1.input_activations``) is assumed to
        follow the **same** un-reciprocated convention as
        ``weight_global_scale`` (both are this checkpoint's own calibration
        output, produced by the same quantization tool) -- also verified
        numerically by the same script rather than only by analogy, per
        this class's own docstring note that guessing this backward
        produces plausible-looking garbage (the ``"!!!!!!!!!!!!"`` bug).
        """
        global_scale = self.weight_global_scale.data.reshape(()).to(torch.float32)
        input_global_scale = self.input_global_scale.data.reshape(()).to(torch.float32)
        return self.weight_packed.data, self.weight_scale.data, global_scale, input_global_scale

    def free_nvfp4_raw_params(self) -> None:
        """Zero out this Linear's raw NVFP4 Parameter storage
        (``.weight_packed``/``.weight_scale``/``.weight_global_scale``/
        ``.input_global_scale``) in place -- called by
        ``Qwen36MLP._free_raw_nvfp4_weights`` once the fused w13/w2
        representation built from :meth:`nvfp4_components_for_fuse` (or, for
        the W4A4 path, :meth:`nvfp4_w4a4_components_for_fuse`) no longer
        needs them. Same discipline as :meth:`~runtime.model.modelopt_linear.
        ModelOptNVFP4Linear.free_nvfp4_raw_params` -- see that method's
        docstring; the only difference is this format's own Parameter
        names (``weight_packed``/``weight_global_scale`` rather than
        ``weight``/``weight_scale_2``, plus this format's extra
        ``input_global_scale``, which modelopt's weight-only checkpoint
        does not have).
        """
        for name in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
            param = getattr(self, name)
            param.data = param.data.new_empty(0)


class FusedFP8ChannelQKV:
    """Fused one-launch QKV W8A8 GEMM over three FP8-channel linears.

    Production verify/decode runs q_proj / k_proj / v_proj as three
    separate native W8A8 GEMMs that each re-quantize the same activation
    and each pay launch + workspace overhead.  Fusing them into one launch
    quantizes the shared input once and streams the three weight fragments
    as one contiguous read, which is what the fused GEMM actually wins:
    the bytes are the same (per-column dots are unchanged), so this is
    bit-exact with the three-GEMM path -- every output column's dot
    product and per-channel scale are identical, and the native per-token
    activation quantizer is deterministic in its input.

    Holds no Parameters (the fused buffers are plain tensors built lazily
    on CUDA from the three linears' raw FP8 weights), so it never appears
    in ``named_parameters()``/``state_dict`` and cannot perturb checkpoint
    loading or the ``free_fp8_raw_weight`` ownership dance.  Build it after
    loading and before the raw FP8 originals are freed (the model frees
    them after warmup; the layer builds this on first forward, which
    precedes warmup by construction).
    """

    def __init__(
        self,
        q_proj: CompressedTensorsFP8ChannelLinear,
        k_proj: CompressedTensorsFP8ChannelLinear,
        v_proj: CompressedTensorsFP8ChannelLinear,
    ) -> None:
        if q_proj.bias is not None or k_proj.bias is not None or v_proj.bias is not None:
            raise ValueError("FusedFP8ChannelQKV requires bias-less projections")
        self._q = q_proj
        self._k = k_proj
        self._v = v_proj
        self._weight: torch.Tensor | None = None
        self._weight_scale: torch.Tensor | None = None
        self._out_split: tuple[int, int] | None = None
        self._workspaces: dict[tuple[int, int, int, int, bool], torch.Tensor] = {}

    @property
    def ready(self) -> bool:
        """True when the fused CUDA buffers have been built."""
        return self._weight is not None

    def _ensure(self) -> None:
        if self._weight is not None:
            return
        wq, wk, wv = self._q.weight.data, self._k.weight.data, self._v.weight.data
        if wq.numel() == 0 or wk.numel() == 0 or wv.numel() == 0:
            raise RuntimeError(
                "FusedFP8ChannelQKV needs raw FP8 weights; build it before "
                "free_fp8_raw_weight() frees them"
            )
        self._weight = torch.cat([wq, wk, wv], dim=0).contiguous()
        sq = self._q.weight_scale.data.float().reshape(-1)
        sk = self._k.weight_scale.data.float().reshape(-1)
        sv = self._v.weight_scale.data.float().reshape(-1)
        self._weight_scale = torch.cat([sq, sk, sv]).contiguous()
        self._out_split = (wq.shape[0], wk.shape[0])

    def forward_native(
        self,
        x: torch.Tensor,
        library: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One quantize + one native W8A8 launch; returns ``(q_gate, k, v)``.

        ``x`` may be any leading-shape ``[..., K]`` BF16 tensor; the output
        splits keep its leading dimensions.  ``library`` is the loaded
        ``NativeFP8W8A8Library`` (same ABI the single-projection path
        uses).
        """
        self._ensure()
        assert self._weight is not None
        assert self._weight_scale is not None
        assert self._out_split is not None
        x_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        m, k = x_2d.shape
        n_total = self._weight.shape[0]
        x_fp8 = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
        activation_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        library.quantize_per_token(x_2d, x_fp8, activation_scale)
        out = torch.empty((m, n_total), dtype=x_2d.dtype, device=x.device)
        geometry = (m, n_total, k, False)
        stream_id = torch.cuda.current_stream(x.device).cuda_stream
        key = (stream_id, *geometry)
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = torch.empty(
                library.workspace_bytes(m=m, n=n_total, k=k, batch_invariant=False),
                dtype=torch.uint8,
                device=x.device,
            )
            self._workspaces[key] = workspace
        library.launch(
            x_fp8,
            self._weight.t(),
            activation_scale,
            self._weight_scale,
            out,
            workspace,
            batch_invariant=False,
        )
        nq, nkv = self._out_split
        lead = x_shape[:-1]
        return (
            out[:, :nq].view(*lead, nq),
            out[:, nq : nq + nkv].view(*lead, nkv),
            out[:, nq + nkv :].view(*lead, nkv),
        )
