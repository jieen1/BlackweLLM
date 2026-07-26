"""B7-V1: Single-point consolidation for ALL vLLM dependencies.

Every ``from vllm.*`` import in the production path goes through this
module.  Dependencies are classified into three tiers:

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
    "FLA_CHUNK_SIZE",
    "EngineArgs",
    "GDNAttentionMetadata",  # re-exported from vLLM (isinstance-sensitive)
    "SM120GQAMetadata",  # re-exported from vLLM (isinstance-sensitive)
    "VllmConfig",
    "AttentionBackendEnum",
    "bind_kv_cache",
    "compute_causal_conv1d_metadata",
    "get_distributed_init_method",
    "get_gemma_rms_norm",
    "get_model",
    "get_open_port",
    "get_vllm_ir",
    "init_worker_distributed_environment",
    "load_eagle_model",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "set_current_vllm_config",
    "register_backend",
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
# Re-exported: SM120GQAMetadata (thin — vLLM dataclass, isinstance-sensitive)
#
# Cannot self-write: vLLM's SM120GQAImpl.forward() does isinstance() checks
# against this class. Self-writing breaks the check. Will be replaced when
# we own the model graph (V2/E1).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Re-exported: GDNAttentionMetadata (thin — vLLM dataclass, isinstance-sensitive)
#
# Cannot self-write: vLLM's qwen_gdn_attention_core does isinstance() checks.
# Will be replaced when we own the model graph (V2/E1).
# ---------------------------------------------------------------------------
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends.sm120_gqa import SM120GQAMetadata  # noqa: E402

# ---------------------------------------------------------------------------
# Self-written: constants (thin)
# ---------------------------------------------------------------------------

FLA_CHUNK_SIZE: int = 64


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
# Self-written: compute_causal_conv1d_metadata (B7-V1 薄依赖自写)
#
# 原 vLLM 实现纯计算（numpy + torch），已自写替代（见文件末尾）。
# 2026-07-22 实测验证 bit-exact。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Re-exported: medium/thick dependencies (vLLM public API)
#
# These are imported at module level so that ``direct_model_runner.py``
# can do ``from runtime.compat_vllm import X`` at its own module level.
# All vLLM imports in the production path are consolidated here — this
# is the B7-V1 "single point" contract.
# ---------------------------------------------------------------------------
import vllm.forward_context as _vllm_fc  # noqa: E402

# ---------------------------------------------------------------------------
# Re-exported: FLA chunk index helpers (vLLM internal, kernel-coupled)
# ---------------------------------------------------------------------------
# B7-V1: FLA 切上游 — 从 vLLM 内嵌 FLA 切到 flash-linear-attention 上游包
# 2026-07-22 实测验证：上游 FLA 0.5.2 的 prepare_chunk_indices/offsets
# 与 vLLM 内嵌版本 bit-exact 一致（batch 1/2/4 × seq 128/1024/4096 × chunk 64/128）
from fla.ops.utils.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.forward_context import ForwardContext  # noqa: E402
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,  # noqa: E402
    register_backend,  # noqa: E402
)

# B7-V1: compute_causal_conv1d_metadata 已自写（见文件末尾）
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
# 仍需 ForwardContext dataclass（model layers 通过 get_forward_context() 读取）。
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

    Still sets ``vllm.forward_context._forward_context`` because model
    layers (loaded via vLLM's ``get_model()``) call ``get_forward_context()``.
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
# Self-written: compute_causal_conv1d_metadata (B7-V1 薄依赖自写)
#
# 原 vLLM 实现: vllm/v1/attention/backends/utils.py:836
# 纯计算：numpy + torch tensor ops，零 vLLM 依赖。
# 2026-07-22 实测验证 bit-exact（见下方切换注释）。
# ---------------------------------------------------------------------------

_PAD_SLOT_ID = -1


def _is_pin_memory_available() -> bool:
    import torch

    return torch.cuda.is_available() and hasattr(torch.Tensor, "pin_memory")


def _np_to_pinned_tensor(array) -> torch.Tensor:
    import torch

    t = torch.from_numpy(array)
    return t.pin_memory() if _is_pin_memory_available() else t


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor, *, device: torch.device
) -> tuple:
    """Compute chunk metadata for causal_conv1d kernel.

    Self-written replacement for vLLM's
    ``vllm.v1.attention.backends.utils.compute_causal_conv1d_metadata``.
    Pure computation: numpy + torch tensor ops, zero vLLM dependency.
    """
    import numpy as np
    import torch

    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict: dict[int, dict] = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    pin_memory = _is_pin_memory_available()

    for BLOCK_M in [8]:
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = _np_to_pinned_tensor(np.repeat(np.arange(len(nums)), nums.numpy()))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(mlist)
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num.item()))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32, pin_memory=pin_memory)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), _PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), _PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(_PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(_PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist, non_blocking=True)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return nums_dict, batch_ptr, token_chunk_offset_ptr


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
