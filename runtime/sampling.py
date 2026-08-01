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


class PersistentSeed:
    """Wraps an integer seed so repeated ``make_generator`` calls across a
    single request's decode rounds advance ONE ``torch.Generator`` instead
    of recreating (and re-seeding, i.e. resetting to the same initial
    state) a fresh one every token.

    Root cause this fixes (N3, docs/roadmap.md Track E): every sampled-path
    call site in ``runtime/backends/laguna.py`` does
    ``gen = make_generator(params.seed); sample_from_logits(..., generator=gen)``
    once PER DECODE STEP. When ``seed`` was a plain ``int``, each of those
    calls did ``torch.Generator().manual_seed(seed)`` from scratch, so every
    token drew from the SAME initial RNG state -- "same seed" meant
    "identical random draw at every position" rather than "one
    reproducible stream," which is not what ``seed`` means in any other
    sampling API.

    Fixing the call sites directly is out of scope here (they live in
    ``runtime/backends/laguna.py``, owned by other in-flight work); this
    wrapper fixes it entirely from the caller's side instead, by relying on
    object identity rather than the call site's behavior:

    - ``server/app.py::_build_sampling_params`` creates exactly ONE
      ``PersistentSeed`` instance per HTTP request and stores it as
      ``SamplingParams.seed``.
    - ``GenerationRequest.sampling_params`` (and therefore ``.seed``) is the
      SAME object for the entire lifetime of a request -- every
      ``make_generator(params.seed)`` call across every decode round for
      that request receives the identical ``PersistentSeed`` instance.
    - ``make_generator`` (below) special-cases ``PersistentSeed``: it
      creates a ``torch.Generator`` lazily on first use and returns that
      SAME generator (already advanced by prior draws) on every later
      call, instead of reseeding.

    Two different requests that happen to pass the same integer seed value
    get two independent ``PersistentSeed`` instances (identity-based, not
    value-based) and therefore two independent streams -- no cross-request
    interference, and no risk of two concurrent ``seed=42`` requests
    sharing RNG state.

    Greedy (``temperature <= 0``) decode never calls ``make_generator`` at
    all (see ``SamplingParams.is_greedy`` / the ``is_greedy`` branches in
    ``sample_from_logits`` and ``decode_batch_sampled``), so this is fully
    inert for the greedy path -- required for bit-exact greedy decoding.
    """

    __slots__ = ("_seed", "_generator")

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._generator: torch.Generator | None = None

    def __repr__(self) -> str:
        return f"PersistentSeed({self._seed!r})"

    def generator(self, device: str) -> torch.Generator:
        """Return this request's persistent generator, creating (and
        seeding) it lazily on first use. Subsequent calls -- even with a
        different ``device`` string that resolves to the same device type
        -- return the SAME object, already advanced by prior draws."""
        import torch as _torch

        if self._generator is None or self._generator.device.type != _torch.device(device).type:
            self._generator = _torch.Generator(device=device)
            self._generator.manual_seed(self._seed)
        return self._generator


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request sampling configuration.

    ``temperature == 0`` means greedy (argmax).  All other fields are
    ignored in greedy mode.
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | PersistentSeed | None = None

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


def compute_sampling_distribution(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The proper probability distribution ``sample_from_logits`` samples
    from for non-greedy ``params`` -- temperature scale, then top-k, then
    top-p, then softmax -- stopping short of the final ``multinomial``
    draw. Extracted (2026-08-02, E2-b, docs/e2e-and-quality-plan.md §2.2)
    so a caller that needs the FULL distribution rather than one sample can
    reuse the exact same transform ``sample_from_logits`` uses, instead of
    hand-rolling a second copy that could silently drift out of sync.

    The motivating caller is speculative decoding's rejection-sampling
    accept/reject test (``runtime.mtp_accept.sample_accept_reject``): it
    needs both the draft model's and the target model's own ``p(x)``/``q(x)``
    at specific tokens to compute an acceptance probability and a residual
    distribution, not just "a" sampled token from each -- passing it
    anything other than the SAME distribution ``sample_from_logits`` would
    have sampled from (e.g. raw softmax without the temperature/top-k/top-p
    transform) would silently change what speculative decoding accepts,
    defeating the entire point of matching the non-speculative sampling
    distribution.

    Precondition: ``not params.is_greedy`` -- greedy has no non-degenerate
    distribution (the whole point of greedy is a single deterministic
    argmax), so callers on the greedy path have no use for this and should
    not call it.
    """
    assert not params.is_greedy, "compute_sampling_distribution is only defined for temperature>0"
    import torch as _torch

    logits_f32 = logits.float()

    if params.temperature != 1.0:
        logits_f32 = logits_f32 / params.temperature

    if params.top_k > 0:
        logits_f32 = _apply_top_k(logits_f32, params.top_k)

    if params.top_p < 1.0:
        logits_f32 = _apply_top_p(logits_f32, params.top_p)

    return _torch.softmax(logits_f32, dim=-1)


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

    probs = compute_sampling_distribution(logits, params)
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


def make_generator(
    seed: int | PersistentSeed | None, device: str | None = None
) -> torch.Generator | None:
    """Create (or, for a ``PersistentSeed``, fetch) a seeded generator for
    reproducible sampling.

    Returns ``None`` when ``seed is None`` (non-deterministic sampling).
    The generator is placed on CUDA if available (required by
    ``torch.multinomial`` on CUDA tensors), otherwise CPU.

    A plain ``int`` reseeds a brand-new generator every call (unchanged
    behavior, still used directly by a few tests/benchmarks). A
    ``PersistentSeed`` returns the SAME generator across calls, advanced
    rather than reset -- see its docstring for why that's the fix for N3
    (seed reproducibility) without touching ``runtime/backends/laguna.py``.
    """
    if seed is None:
        return None
    import torch as _torch

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    if isinstance(seed, PersistentSeed):
        return seed.generator(device)
    gen = _torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
