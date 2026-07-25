"""Graph-safe sampling primitives for BlackweLLM.

Implements temperature / top-k / top-p (nucleus) sampling as pure tensor
operations on logits.  ``temperature == 0`` is defined as greedy (argmax)
and is bit-identical to the existing ``logits.argmax(dim=-1)`` path.

Design constraints (roadmap B1):
- All operations use pre-allocated persistent buffers so CUDA Graph replay
  is safe (no host-side allocation in the hot path).
- Greedy path (temperature=0) must remain bit-level identical to the
  current ``argmax`` code path.
- Sampling path runs in eager mode first; graph capture is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request sampling configuration.

    ``temperature == 0`` means greedy (argmax).  All other fields are
    ignored in greedy mode.
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


def sample_from_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample token ids from ``logits`` according to ``params``.

    Args:
        logits: Shape ``[batch, vocab]`` (float32 or bfloat16).
        params: Sampling configuration.
        generator: Optional seeded generator for reproducibility.

    Returns:
        Token ids of shape ``[batch]`` (int64).
    """
    import torch as _torch

    if params.is_greedy:
        return logits.argmax(dim=-1)

    logits_f32 = logits.float()

    if params.temperature != 1.0:
        logits_f32 = logits_f32 / params.temperature

    if params.top_k > 0:
        logits_f32 = _apply_top_k(logits_f32, params.top_k)

    if params.top_p < 1.0:
        logits_f32 = _apply_top_p(logits_f32, params.top_p)

    probs = _torch.softmax(logits_f32, dim=-1)
    if generator is not None and probs.device != generator.device:
        probs = probs.to(generator.device)
        result = _torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        return result.to(logits.device)
    return _torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside the top-k highest values."""
    k = min(k, logits.size(-1))
    top_k_vals = logits.topk(k, dim=-1).values
    threshold = top_k_vals[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass reaches ``p``, zero out the rest."""
    import torch as _torch

    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    sorted_probs = _torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_mask = cumulative_probs - sorted_probs >= p
    sorted_logits[sorted_mask] = float("-inf")

    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def make_generator(seed: int | None, device: str | None = None) -> torch.Generator | None:
    """Create a seeded generator for reproducible sampling.

    Returns ``None`` when ``seed is None`` (non-deterministic sampling).
    The generator is placed on CUDA if available (required by
    ``torch.multinomial`` on CUDA tensors), otherwise CPU.
    """
    if seed is None:
        return None
    import torch as _torch

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    gen = _torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
