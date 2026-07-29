"""vLLM attention re-exports needed ONLY by the qwen36/DirectModelRunner tenant.

任务#42 (vLLM removal plan): split out of ``runtime/compat_vllm.py``.
Found by a coordinator cross-check, not by this project's own earlier
audits: ``compat_vllm.py`` had these five symbols as UNCONDITIONAL,
module-level imports, even though every real consumer
(``runtime/metadata_builders.py``, ``runtime/cuda_graphs.py``,
``runtime/direct_model_runner.py`` -- all three explicitly "extracted
from direct_model_runner.py" per their own module docstrings) is
exclusively part of the qwen36 tenant, never the Laguna one (confirmed
via ``server/engine.py``'s ``_load_model()``: ``backend_name ==
"laguna"`` dispatches to ``_load_laguna_model()``/``LagunaBackend``,
never touching ``DirectModelRunner``; qwen3.6/DirectModelRunner was
explicitly put out of scope for this whole vLLM-removal effort at 阶段0
-- "qwen3.6(DirectModelRunner)路径本次不动"). Since ``compat_vllm.py``
is one shared module both tenants import from, Laguna's own (legitimate)
``from runtime.compat_vllm import (...)`` was transitively requiring
``vllm.v1.attention.backends.gdn_attn``/``sm120_gqa``/``registry`` and
the third-party ``fla`` package to be importable -- not because Laguna's
own code needs any of them (grep-verified: zero references anywhere in
``laguna*.py``/``runtime/model/*.py``/``runtime/kernels/*.py``), purely
because of file-sharing. This module exists so Laguna's import chain no
longer drags these in.

Not a "thin/self-written" tier like most of ``compat_vllm.py`` -- these
are real, still-thick vLLM/fla dependencies for the qwen36 tenant, kept
exactly as before (isinstance-sensitive dataclasses, same reasoning
``compat_vllm.py`` used to state for them). No behavior change, pure
relocation -- verified by checking every real importer before moving
anything (``runtime/metadata_builders.py``, ``runtime/cuda_graphs.py``,
``runtime/direct_model_runner.py``), not assumed from the class names.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Re-exported: FLA chunk index helpers (GDN/Mamba linear-attention kernel
# support -- runtime/metadata_builders.py). Not vLLM itself but an
# upstream package vLLM's GDN code also depends on; Laguna has no GDN/
# linear-attention layers, so this dependency is qwen36-exclusive too.
# ---------------------------------------------------------------------------
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exported: FLA chunk index helpers (GDN/Mamba linear-attention kernel
# support -- runtime/metadata_builders.py). Not vLLM itself but an
# upstream package vLLM's GDN code also depends on; Laguna has no GDN/
# linear-attention layers, so this dependency is qwen36-exclusive too.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Re-exported: SM120GQAMetadata / GDNAttentionMetadata (vLLM dataclasses,
# isinstance-sensitive in vLLM's own GDN linear-attention layer code --
# vllm/model_executor/layers/mamba/gdn/{qwen,kimi,olmo}_gdn_linear_attn.py
# do `assert isinstance(attn_metadata, GDNAttentionMetadata)`. Verified
# directly against real vLLM source, not assumed from the docstring that
# used to sit in compat_vllm.py -- SM120GQAMetadata itself turned out to
# have NO isinstance check anywhere in vLLM (SM120GQAImpl.forward() only
# does duck-typed attribute access), but it's kept re-exported here
# anyway since qwen36 is out of scope for re-evaluating this.
# ---------------------------------------------------------------------------
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exported: attention backend registry (DirectModelRunner's own
# SM120GQABackend registration -- runtime/direct_model_runner.py)
# ---------------------------------------------------------------------------
from vllm.v1.attention.backends.registry import (  # noqa: F401
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.sm120_gqa import SM120GQAMetadata  # noqa: F401

FLA_CHUNK_SIZE: int = 64


# ---------------------------------------------------------------------------
# Self-written: compute_causal_conv1d_metadata (moved from compat_vllm.py,
# 任务#42 -- causal_conv1d is a GDN/Mamba-kernel concern, qwen36-only).
#
# 原 vLLM 实现: vllm/v1/attention/backends/utils.py:836
# 纯计算：numpy + torch tensor ops，零 vLLM 依赖。
# 2026-07-22 实测验证 bit-exact。
# ---------------------------------------------------------------------------

_PAD_SLOT_ID = -1


def _is_pin_memory_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch.Tensor, "pin_memory")


def _np_to_pinned_tensor(array) -> torch.Tensor:
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
