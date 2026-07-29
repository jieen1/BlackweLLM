"""Qwen3.6 legacy tenant's single-point consolidation for vLLM dependencies.

Every remaining ``from vllm.*`` import belongs to the separately retained
Qwen3.6/``DirectModelRunner`` tenant and goes through this module.

**Self-written (thin)** — pure dataclasses / constants / trivial utilities
re-implemented here with zero vLLM import.  These survive even if vLLM
is uninstalled.

**Re-exported (medium)** — stable public API symbols that vLLM exposes
and that we consume without modification.  Imported lazily so the module
can be loaded (for its self-written symbols) even without vLLM installed.

**Re-exported (thick)** — model graph construction and MTP loading.
These are the last to be replaced (pulled by A1/A2/A3/E1 evidence).

Migration invariant (architecture.md §3.6): replacing any symbol here
must preserve bit-level parity on the greedy fixed-prompt suite.
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING

# AOT compile bakes fixed input shapes from the first call (M=1 warmup),
# breaking CUDA Graph capture at M=16 (verify).  JIT torch.compile with
# dropped guards handles dynamic shapes correctly.
os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")

if TYPE_CHECKING:
    import torch

__all__ = [
    "EngineArgs",
    "VllmConfig",
    "bind_kv_cache",
    "get_distributed_init_method",
    "get_gemma_rms_norm",
    "get_model",
    "get_open_port",
    "get_vllm_ir",
    "init_worker_distributed_environment",
    "load_eagle_model",
    "set_current_vllm_config",
    "set_forward_context",
    "get_flashinfer_metadata_builder",
    "get_common_attn_metadata_cls",
    "init_flashinfer_workspace",
    "get_nvfp4_b12x_kernel_components",
    "get_nvfp4_cutlass_kernel_components",
    "get_nvfp4_cudnn_components",
    "get_nvfp4_cudnn_apply_dependencies",
    "get_nvfp4_custom_ops",
    "get_nvfp4_cutlass_module",
    "get_nvfp4_flashinfer_module",
    "get_cutlass_scaled_fp4_mm",
]

# ---------------------------------------------------------------------------
# GDNAttentionMetadata/SM120GQAMetadata/AttentionBackendEnum/register_backend/
# FLA chunk helpers/compute_causal_conv1d_metadata moved to
# runtime/compat_vllm_qwen36.py (任务#42) -- every real consumer
# (runtime/metadata_builders.py, runtime/cuda_graphs.py,
# runtime/direct_model_runner.py) is exclusively the qwen36 tenant, never
# Laguna (grep-verified: zero references anywhere in laguna*.py/
# runtime/model/*.py/runtime/kernels/*.py). Having them as unconditional
# module-level imports HERE meant Laguna's own, legitimate
# ``from runtime.compat_vllm import (...)`` transitively required
# vllm.v1.attention.backends.gdn_attn/sm120_gqa/registry and the
# third-party fla package to be importable too, purely because this file
# is shared between both tenants -- not because Laguna's code needs any
# of them. See compat_vllm_qwen36.py's module docstring for the full
# writeup (found via a coordinator cross-check, not this project's own
# earlier audits).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Self-written: network utilities (thin)
# ---------------------------------------------------------------------------


def get_open_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_distributed_init_method(ip: str, port: int) -> str:
    """Build a ``tcp://`` URI for torch.distributed init."""
    if ":" in ip:
        return f"tcp://[{ip}]:{port}"
    return f"tcp://{ip}:{port}"


# ---------------------------------------------------------------------------
# Re-exported: medium/thick dependencies (vLLM public API). 任务#46
# removed Laguna's OWN QSR_LAGUNA_MODEL_LOADER=vllm/QSR_DFLASH_MODEL_
# LOADER=vllm escape hatches (runtime/backends/laguna.py/laguna_dflash.py
# no longer call get_model()/init_worker_distributed_environment/
# load_dflash_model() at all), so these symbols now exist here purely
# for: (1) the separate qwen3.6/DirectModelRunner tenant (runtime/
# direct_model_runner.py, out of scope for this whole effort, 阶段0),
# which unconditionally constructs a real vLLM model graph via
# get_model() and genuinely needs all of config plumbing/distributed
# init/forward-context state the same way Laguna's escape hatch used to;
# (2) EngineArgs specifically, which dozens of benchmarks/*.py diagnostic
# scripts construct directly for one-off real-vLLM comparisons/repros,
# independent of either tenant's production loader. See
# runtime/compat_vllm_qwen36.py for the qwen36-exclusive re-exports this
# file used to also carry (split out 任务#42a).
#
# ForwardContext/CUDAGraphMode/vllm.forward_context specifically: dropped
# 任务#42(b), then RESTORED 任务#45 after a real GPU run of Laguna's
# escape hatch (before its 任务#46 removal) crashed: ``AssertionError:
# Forward context is not set`` inside vLLM's OWN ``vllm/model_executor/
# layers/fused_moe/runner/moe_runner.py``, which calls
# ``get_forward_context()`` internally -- confirmed direct_model_runner.py
# (still in production use) hits the exact same real-vLLM-FusedMoE
# dependency via set_forward_context(), so this stays regardless of
# Laguna's own escape hatch being gone.
# ---------------------------------------------------------------------------
import vllm.forward_context as _vllm_fc  # noqa: E402
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.forward_context import ForwardContext  # noqa: E402
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.worker.gpu_worker import (  # noqa: E402
    init_worker_distributed_environment,  # noqa: E402
)

# bind_kv_cache: self-written (see below)


def load_eagle_model(*args, **kwargs):
    """Thick dependency: MTP model loading (replaced by A3 evidence)."""
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        load_eagle_model as _load_eagle_model,
    )

    return _load_eagle_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Re-exported: vLLM IR ops and model layers (used by norm patches)
# ---------------------------------------------------------------------------


def get_vllm_ir():
    """Lazy import of vLLM's IR op system (used by gemma_norm_patch / triton_norm_ops)."""
    from vllm import ir

    return ir


def get_gemma_rms_norm():
    """Lazy import of GemmaRMSNorm (used by gemma_norm_patch)."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    return GemmaRMSNorm


# ---------------------------------------------------------------------------
# Self-written: bind_kv_cache (B7-V1 薄依赖自写)
#
# 原 vLLM 实现: vllm/v1/worker/utils.py:479
# 纯字典绑定 + extract_layer_index（字符串解析），零 vLLM 依赖。
# ---------------------------------------------------------------------------


def _extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """Extract the integer layer index from a dotted module name.

    Self-written replacement for vLLM's
    ``vllm.model_executor.models.utils.extract_layer_index``.
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, f"layer name {layer_name} should only contain one integer"
        return int_vals[0]
    else:
        assert len(int_vals) <= 2, f"layer name {layer_name} should contain most two integers"
        return int_vals[0] * num_attn_module + int_vals[1] if len(int_vals) == 2 else int_vals[0]


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, object],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Bind allocated KV caches to ModelRunner list and forward context.

    Self-written replacement for vLLM's ``vllm.v1.worker.utils.bind_kv_cache``.
    Pure dict binding + layer-index sorting, zero vLLM dependency.
    """
    from collections import defaultdict

    assert len(runner_kv_caches) == 0

    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index2name[_extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


# ---------------------------------------------------------------------------
# Self-written: set_forward_context (B7-V1 薄依赖自写)
#
# 原 vLLM 实现: vllm/forward_context.py:260
# 简化版：跳过 DP/batch-tracking/cudagraph/platform 逻辑（单 GPU 无需）。
# 仍需 ForwardContext dataclass 并写入 vllm.forward_context._forward_
# context -- 任务#42(b)一度移除过这段状态写入(理由:this runtime自己的
# 代码从不调用get_forward_context()),任务#45一次真实的escape hatch GPU
# 跑通测试(QSR_LAGUNA_MODEL_LOADER=vllm)当场crash在vLLM自己
# FusedMoE runner的get_forward_context()调用上,已恢复且不再重试这个
# 移除方向 -- 完整依据见上面的模块级注释。
# ---------------------------------------------------------------------------

from contextlib import contextmanager  # noqa: E402


@contextmanager
def set_forward_context(
    attn_metadata,
    vllm_config: VllmConfig,
    *,
    slot_mapping=None,
    skip_compiled: bool = False,
    **_ignored_kwargs,
):
    """Simplified forward context manager for single-GPU BlackweLLM.

    Self-written replacement for vLLM's ``set_forward_context``.
    Skips DP coordination, batch-size tracking, cudagraph mode dispatch,
    and platform hooks — none apply to our single-GPU, non-MoE setup.

    Still sets ``vllm.forward_context._forward_context`` because vLLM's
    OWN model/layer code (used unconditionally by both the escape hatch,
    which constructs a real vLLM model graph, and the self-built default
    path's own real ``Attention``/``FusedMoE`` construction side effects)
    calls ``get_forward_context()`` -- see the module comment above for
    the specific real crash this restores a fix for.
    """
    forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        all_moe_layers=getattr(vllm_config.compilation_config, "static_all_moe_layers", None),
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
        ubatch_slices=None,
        skip_compiled=skip_compiled,
        additional_kwargs={},
        is_padding=None,
    )
    prev = _vllm_fc._forward_context
    _vllm_fc._forward_context = forward_context
    try:
        yield
    finally:
        _vllm_fc._forward_context = prev


# ---------------------------------------------------------------------------
# compute_causal_conv1d_metadata moved to runtime/compat_vllm_qwen36.py
# (任务#42) -- causal_conv1d is a GDN/Mamba-kernel concern, its only real
# caller is runtime/metadata_builders.py (qwen36-exclusive, see that
# module's docstring). Already zero-vLLM-dependency (pure numpy/torch),
# so moving it doesn't change any vLLM-importability story -- purely
# keeping qwen36-only functionality out of Laguna's compat layer.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Re-exported: FlashInfer attention metadata (Laguna backend)
#
# Used by runtime/backends/laguna.py for direct model.forward() path.
# FlashInferMetadataBuilder builds per-group attention metadata.
# CommonAttentionMetadata is the input dataclass for the builder.
# init_workspace_manager initializes FlashInfer workspace buffers.
# ---------------------------------------------------------------------------


def get_flashinfer_metadata_builder():
    """Lazy import: FlashInferMetadataBuilder."""
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    return FlashInferMetadataBuilder


def get_common_attn_metadata_cls():
    """Lazy import: CommonAttentionMetadata."""
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    return CommonAttentionMetadata


def init_flashinfer_workspace(device):
    """Lazy import: init_workspace_manager."""
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(device)


# ---------------------------------------------------------------------------
# Re-exported: NVFP4 patch dependencies
#
# The four local NVFP4 tuning modules deliberately own only policy and patch
# logic.  All imports from vLLM stay here, so those modules remain usable as
# thin, self-owned adapters while each upstream symbol is replaced over time.
# ---------------------------------------------------------------------------


def get_nvfp4_b12x_kernel_components():
    """Return the vLLM registry, B12x kernel class, and CUDA platform enum."""
    from vllm.model_executor.kernels.linear import _POSSIBLE_NVFP4_KERNELS
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferB12xNvFp4LinearKernel,
    )
    from vllm.platforms import PlatformEnum

    return _POSSIBLE_NVFP4_KERNELS, FlashInferB12xNvFp4LinearKernel, PlatformEnum


def get_nvfp4_cutlass_kernel_components():
    """Return the vLLM registry, CUTLASS kernel class, and CUDA platform enum."""
    from vllm.model_executor.kernels.linear import _POSSIBLE_NVFP4_KERNELS
    from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
        CutlassNvFp4LinearKernel,
    )
    from vllm.platforms import PlatformEnum

    return _POSSIBLE_NVFP4_KERNELS, CutlassNvFp4LinearKernel, PlatformEnum


def get_nvfp4_cudnn_components():
    """Return FlashInfer's NVFP4 GEMM entry point and availability predicate."""
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

    return flashinfer_scaled_fp4_mm, has_flashinfer


def get_nvfp4_cudnn_apply_dependencies():
    """Return vLLM helpers needed by the cuDNN NVFP4 apply-weights patch."""
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.fusion.quant_activation import (
        as_quantized_activation,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_activation_for_cutlass,
        slice_nvfp4_output,
    )

    return (
        scaled_fp4_quant,
        as_quantized_activation,
        pad_nvfp4_activation_for_cutlass,
        slice_nvfp4_output,
    )


def get_nvfp4_custom_ops():
    """Return vLLM custom ops for a narrow, reversible local patch."""
    import vllm._custom_ops as ops

    return ops


def get_nvfp4_cutlass_module():
    """Return the vLLM CUTLASS NVFP4 module when that upstream layout exists."""
    import vllm.model_executor.kernels.linear.nvfp4.cutlass as cutlass_module

    return cutlass_module


def get_nvfp4_flashinfer_module():
    """Return the vLLM FlashInfer NVFP4 module for a narrow local patch."""
    import vllm.model_executor.kernels.linear.nvfp4.flashinfer as flashinfer_module

    return flashinfer_module


def get_cutlass_scaled_fp4_mm():
    """Return vLLM's fallback NVFP4 CUTLASS operator."""
    from vllm._custom_ops import cutlass_scaled_fp4_mm

    return cutlass_scaled_fp4_mm
