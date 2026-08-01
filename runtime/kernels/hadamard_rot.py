"""Hadamard incoherent processing for FP8 attention quality (FA3 §3.3).

Applies a normalized 128×128 Hadamard rotation to Q and K head dimensions
before FP8 quantization.  Spreads outlier magnitudes evenly, reducing
per-tensor FP8 E4M3 quantization error ~2.6x (FlashAttention-3 paper).

Mathematically transparent: Q'K'^T = QHH^TK^T = QK^T since HH^T = I.
"""

import torch

_H128: torch.Tensor | None = None


def get_hadamard_128(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    global _H128
    if _H128 is not None and _H128.device == device and _H128.dtype == dtype:
        return _H128
    H = torch.tensor([[1.0]])
    for _ in range(7):  # 2^7 = 128
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = H / (128.0**0.5)
    _H128 = H.to(device=device, dtype=dtype)
    return _H128


def hadamard_rotate_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Rotate the head_dim axis of a [..., num_heads*head_dim] tensor.

    x: [..., num_heads * head_dim] (flat last dim, as from qkv split)
    Returns: same shape, Hadamard-rotated per head.
    """
    H = get_hadamard_128(x.device, x.dtype)
    orig_shape = x.shape
    # Reshape to [..., num_heads, head_dim]
    x_heads = x.view(*orig_shape[:-1], num_heads, head_dim)
    # Rotate: [..., num_heads, head_dim] @ [head_dim, head_dim]
    x_rot = torch.matmul(x_heads, H)
    return x_rot.view(orig_shape)
