"""bfdiag.shapes.harness -- turn shape descriptions into real ``torch`` tensors.

Convenience layer for kernel isolation tests: instead of hand-typing
``torch.empty(48, 128, ...)`` (and getting a number wrong), build the tensor
straight from a derived shape. Depends on ``torch`` only -- no ``vllm``
import anywhere in this package.

**Hard safety rule**: this box has exactly one GPU and a human is using it
right now to debug the block_size=64 vs 128 acceptance-rate question. This
module must never allocate a CUDA tensor. ``device`` defaults to ``"cpu"``
and is otherwise accepted only so call sites can be written the way they'll
eventually look once someone deliberately opts in on a machine where that's
safe (see ``ALLOW_CUDA`` below) -- by default any non-CPU device raises
immediately, no silent redirect to "cpu", so a caller can't accidentally
think they got a GPU tensor.
"""

from __future__ import annotations

import os

import torch

ALLOW_CUDA_ENV = "BF_SHAPES_ALLOW_CUDA"


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cpu" and os.environ.get(ALLOW_CUDA_ENV) != "1":
        raise RuntimeError(
            f"bfdiag.shapes.harness refuses to allocate a {dev.type!r} tensor "
            f"(requested device={device!r}). This package is for GPU-free shape "
            "validation on a machine with a single GPU under active use. "
            f"If you really mean to run on GPU, set {ALLOW_CUDA_ENV}=1 yourself "
            "-- this module will not do it implicitly."
        )
    return dev


def make_empty(
    shape: tuple[int, ...], *, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"
) -> torch.Tensor:
    """``torch.empty(shape, dtype=dtype, device=device)`` with the CUDA guard above."""
    return torch.empty(shape, dtype=dtype, device=_resolve_device(device))


def make_randn(
    shape: tuple[int, ...], *, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"
) -> torch.Tensor:
    """``torch.randn`` for shapes that need non-garbage data (e.g. feeding a
    reference softmax/argmax check on CPU). Integer/uint8/fp8 dtypes fall
    back to a uniform-random fill since ``torch.randn`` doesn't support them.
    """
    dev = _resolve_device(device)
    if dtype in (torch.uint8, torch.int32, torch.int64, torch.int16, torch.int8):
        info = torch.iinfo(dtype) if dtype != torch.uint8 else None
        high = 256 if dtype == torch.uint8 else min(info.max, 2**31 - 1)
        return torch.randint(0, high, shape, dtype=dtype, device=dev)
    try:
        return torch.randn(shape, dtype=dtype, device=dev)
    except RuntimeError:
        # torch.randn doesn't support float8_* directly -- fill via float32 then cast.
        return torch.randn(shape, dtype=torch.float32, device=dev).to(dtype)


def make_zeros(
    shape: tuple[int, ...], *, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"
) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=_resolve_device(device))


_FILLERS = {"empty": make_empty, "randn": make_randn, "zeros": make_zeros}


def make_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    fill: str = "empty",
) -> torch.Tensor:
    """Dispatch to :func:`make_empty`/:func:`make_randn`/:func:`make_zeros` by name."""
    try:
        fn = _FILLERS[fill]
    except KeyError as exc:
        raise ValueError(f"fill must be one of {sorted(_FILLERS)}, got {fill!r}") from exc
    return fn(shape, dtype=dtype, device=device)


def empty_from_shapes(
    shapes: dict[str, tuple[int, ...]],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    dtype_overrides: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]:
    """Realize a ``{name: shape}`` mapping (e.g. a :class:`GemmShape`'s or
    :class:`AttentionCallShape`'s ``.shapes()``) into ``{name: torch.Tensor}``,
    all ``torch.empty``. ``dtype_overrides`` lets a caller give e.g.
    ``page_table``/``cache_seqlens`` int32 while everything else stays bf16.
    """
    overrides = dtype_overrides or {}
    return {
        name: make_empty(shape, dtype=overrides.get(name, dtype), device=device)
        for name, shape in shapes.items()
    }
