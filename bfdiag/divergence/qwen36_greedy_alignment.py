"""Pure comparison/decision logic for the B1 gate's other half: "与 HF
transformers 贪心逐 token 对齐（>= 3 工作负载 x 512 token）"
(``docs/implementation-plan.md`` §7.1).

Deliberately split from the actual generation loop (which needs a live
model, a GPU, and ~90 seconds of sparkinfer JIT on the first call, see
``runtime/model/qwen36_model.py``'s ``Qwen36AttentionWorkspace``) the same
way ``bfdiag/divergence/scan.py`` is split from
``bfdiag/divergence/capture.py``: this module takes plain ``int`` token-id
sequences and answers "did they align, and does that satisfy the gate" --
nothing here touches ``torch``, a model, or a GPU, so it is exhaustively
unit-testable on CPU (see ``tests/test_bfdiag_qwen36_greedy_alignment.py``)
without needing the real comparison run to exist yet. The real run
(``scripts/b1_verify_greedy_alignment.py``) is GPU-only and was NOT
completed in this pass -- see that script's own docstring for exactly why
and what it needs to do the one time it runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenAlignmentResult:
    """Greedy-token comparison for one workload (one prompt).

    ``mine_token_ids``/``oracle_token_ids`` are the GENERATED tokens only
    (never the prompt) -- comparing prompt tokens would be comparing the
    tokenizer against itself, not the two models.
    """

    workload_name: str
    prompt_token_ids: tuple[int, ...]
    mine_token_ids: tuple[int, ...]
    oracle_token_ids: tuple[int, ...]
    first_divergence_index: int | None  # index into the GENERATED tokens
    num_tokens_compared: int
    num_matched: int

    @property
    def fully_aligned(self) -> bool:
        return self.first_divergence_index is None

    @property
    def match_rate(self) -> float:
        if self.num_tokens_compared == 0:
            return 0.0
        return self.num_matched / self.num_tokens_compared

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_name": self.workload_name,
            "prompt_token_ids": list(self.prompt_token_ids),
            "mine_token_ids": list(self.mine_token_ids),
            "oracle_token_ids": list(self.oracle_token_ids),
            "first_divergence_index": self.first_divergence_index,
            "num_tokens_compared": self.num_tokens_compared,
            "num_matched": self.num_matched,
            "match_rate": self.match_rate,
            "fully_aligned": self.fully_aligned,
        }


def compare_greedy_token_ids(
    workload_name: str,
    prompt_token_ids: list[int] | tuple[int, ...],
    mine_token_ids: list[int] | tuple[int, ...],
    oracle_token_ids: list[int] | tuple[int, ...],
) -> TokenAlignmentResult:
    """Pure comparison: where (if anywhere) do the two greedy generations
    first disagree, over however many tokens both sides actually produced.

    Compares only up to ``min(len(mine), len(oracle))`` -- one side
    stopping early (e.g. hitting EOS) is not itself a divergence signal;
    the caller decides whether unequal lengths matter for its own gate
    (see :func:`passes_b1_gate`'s ``min_tokens_per_workload``).
    """
    mine = tuple(mine_token_ids)
    oracle = tuple(oracle_token_ids)
    compared_len = min(len(mine), len(oracle))

    first_divergence: int | None = None
    num_matched = 0
    for i in range(compared_len):
        if mine[i] == oracle[i]:
            num_matched += 1
        elif first_divergence is None:
            first_divergence = i

    return TokenAlignmentResult(
        workload_name=workload_name,
        prompt_token_ids=tuple(prompt_token_ids),
        mine_token_ids=mine,
        oracle_token_ids=oracle,
        first_divergence_index=first_divergence,
        num_tokens_compared=compared_len,
        num_matched=num_matched,
    )


@dataclass(frozen=True)
class GreedyAlignmentReport:
    workloads: tuple[TokenAlignmentResult, ...] = field(default_factory=tuple)

    @property
    def all_fully_aligned(self) -> bool:
        return all(w.fully_aligned for w in self.workloads)

    @property
    def overall_match_rate(self) -> float:
        """Token-weighted average match rate across every workload (not a
        plain mean of per-workload rates -- a short workload should not
        count as much as a long one)."""
        total_compared = sum(w.num_tokens_compared for w in self.workloads)
        if total_compared == 0:
            return 0.0
        total_matched = sum(w.num_matched for w in self.workloads)
        return total_matched / total_compared

    def to_dict(self) -> dict[str, Any]:
        return {
            "workloads": [w.to_dict() for w in self.workloads],
            "all_fully_aligned": self.all_fully_aligned,
            "overall_match_rate": self.overall_match_rate,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GreedyAlignmentReport:
        data = json.loads(Path(path).read_text())
        workloads = tuple(
            TokenAlignmentResult(
                workload_name=w["workload_name"],
                prompt_token_ids=tuple(w["prompt_token_ids"]),
                mine_token_ids=tuple(w["mine_token_ids"]),
                oracle_token_ids=tuple(w["oracle_token_ids"]),
                first_divergence_index=w["first_divergence_index"],
                num_tokens_compared=w["num_tokens_compared"],
                num_matched=w["num_matched"],
            )
            for w in data["workloads"]
        )
        return cls(workloads=workloads)


def passes_b1_gate(
    report: GreedyAlignmentReport,
    *,
    min_workloads: int = 3,
    min_tokens_per_workload: int = 512,
) -> tuple[bool, str]:
    """Literal B1 gate check: ``docs/implementation-plan.md`` §7.1 reads
    "greedy token-for-token alignment against HF transformers, >= 3
    workloads x 512 tokens" -- interpreted here as a hard requirement
    (every compared token must match, not a fuzzy threshold), because
    that is what "逐 token 对齐" literally says and this project's own
    convention (see ``bf diff``'s bit-exact-first stance,
    ``docs/README.md``) is to not quietly relax a stated bar.

    Returns ``(passed, reason)`` -- ``reason`` explains a failure (or
    confirms a pass) in one line, meant for a human reading a CI log, not
    for programmatic branching (check ``passed`` for that).
    """
    if len(report.workloads) < min_workloads:
        return False, (
            f"only {len(report.workloads)} workload(s) reported, gate requires "
            f">= {min_workloads}"
        )
    short = [
        w for w in report.workloads if w.num_tokens_compared < min_tokens_per_workload
    ]
    if short:
        names = ", ".join(f"{w.workload_name} ({w.num_tokens_compared})" for w in short)
        return False, (
            f"{len(short)} workload(s) compared fewer than {min_tokens_per_workload} "
            f"tokens: {names}"
        )
    if not report.all_fully_aligned:
        diverged = [w for w in report.workloads if not w.fully_aligned]
        names = ", ".join(
            f"{w.workload_name}@token#{w.first_divergence_index}" for w in diverged
        )
        return False, (
            f"{len(diverged)}/{len(report.workloads)} workload(s) diverged from HF "
            f"before {min_tokens_per_workload} tokens: {names} "
            f"(overall_match_rate={report.overall_match_rate:.4f})"
        )
    return True, (
        f"all {len(report.workloads)} workloads fully aligned for "
        f">= {min_tokens_per_workload} tokens each"
    )
