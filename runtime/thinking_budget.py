"""Token-level thinking-budget state for reasoning-model decoding.

The state is deliberately independent of HTTP and torch.  The scheduler owns
one instance per live request and asks it which output position must emit the
next reasoning-end token.  Backends then apply that constraint to their
already-computed target logits, including speculative verify logits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThinkingBudgetConfig:
    """The token markers and budget needed by one reasoning request."""

    budget: int
    start_token_ids: tuple[int, ...]
    end_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError(f"thinking budget must be positive, got {self.budget}")
        if not self.start_token_ids:
            raise ValueError("thinking budget requires a non-empty start marker")
        if not self.end_token_ids:
            raise ValueError("thinking budget requires a non-empty end marker")


class ThinkingBudgetState:
    """Track the current reasoning span and the next forced end-token slot.

    ``prompt_token_ids`` may already contain an open ``<think>`` marker, as
    Qwen's chat template does.  ``output_token_ids`` contains only committed
    model output, including MTP recovery/bonus tokens.  A returned position is
    relative to the next output block: position zero is the next token in a
    normal decode round, while MTP positions ``0..K`` are the target
    predictions for the committed block.

    The budget counts tokens after the latest start marker and before the
    latest complete end marker.  If the end marker itself is multi-token, a
    partial suffix is continued from position zero on the next round.
    """

    __slots__ = ("config", "prompt_token_ids", "output_token_ids")

    def __init__(
        self,
        prompt_token_ids: list[int] | tuple[int, ...],
        config: ThinkingBudgetConfig,
    ) -> None:
        self.config = config
        self.prompt_token_ids = tuple(prompt_token_ids)
        self.output_token_ids: list[int] = []

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]

    def add_output(self, token_ids: list[int] | tuple[int, ...]) -> None:
        self.output_token_ids.extend(int(token_id) for token_id in token_ids)

    @staticmethod
    def _last_sequence_index(sequence: list[int], marker: tuple[int, ...]) -> int:
        width = len(marker)
        if not width or width > len(sequence):
            return -1
        for index in range(len(sequence) - width, -1, -1):
            if tuple(sequence[index : index + width]) == marker:
                return index
        return -1

    def _span(self, sequence: list[int]) -> tuple[int, int] | None:
        start = self._last_sequence_index(sequence, self.config.start_token_ids)
        end = self._last_sequence_index(sequence, self.config.end_token_ids)
        if start < 0 or start < end:
            return None
        return start, end

    def _end_prefix_length(self, sequence: list[int]) -> int:
        """Return the longest incomplete end-marker prefix at the tail."""
        end = self.config.end_token_ids
        max_prefix = min(len(end) - 1, len(sequence))
        for length in range(max_prefix, 0, -1):
            if tuple(sequence[-length:]) == end[:length]:
                return length
        return 0

    def force_for(self, max_output_positions: int) -> tuple[int, int] | None:
        """Return ``(position, token_id)`` for the next decode block.

        ``None`` means the request is not currently inside a reasoning span,
        or the budget boundary is beyond this block.  ``max_output_positions``
        is one for ordinary decode and ``K + 1`` for an MTP/DSpark verify.
        """
        if max_output_positions <= 0:
            raise ValueError("max_output_positions must be positive")

        sequence = self.all_token_ids
        span = self._span(sequence)
        if span is None:
            return None

        _start, end = span
        end_prefix_length = self._end_prefix_length(sequence)
        if end_prefix_length:
            return 0, self.config.end_token_ids[end_prefix_length]

        reasoning_start = _start + len(self.config.start_token_ids)
        reasoning_count = len(sequence) - reasoning_start
        remaining = self.config.budget - reasoning_count
        position = max(0, remaining)
        if position >= max_output_positions:
            return None
        return position, self.config.end_token_ids[0]
