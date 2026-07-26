"""Frozen serving workload definitions.

2026-07-17 extension (Codex-sol review, via the coordinator): the original
Phase 0 contract (below, kept unchanged for `W1`/`W2` -- `tests/test_workloads
.py` pins their `input_tokens`/`concurrency`) only carried length and
concurrency. That is not enough to make an acceptance-rate/throughput
comparison reproducible: an earlier round's W1 acceptance-rate investigation
found THREE real confounds hiding in exactly the fields this file was
missing -- sampling temperature (defaulted differently on each side),
input token distribution (assumed "same formula" was enough; it wasn't
precise enough for source-level reproducibility), and generation
depth/length (this workload's own `output_tokens` interacts with a REAL
acceptance-inflation effect for long unconstrained-greedy generation, see
`W1_S`/`W2_S` below). `Workload` now carries every field an acceptance-rate
or throughput comparison actually needs pinned: the exact tokenizer,
sampling parameters, stop semantics, and (for the synthetic `-S` line) a
path to FROZEN, VERSIONED prompt token ids -- not a regenerate-from-formula
scheme, which is reproducible but not precise (two runs using "the same
formula" can still diverge if `tokenizer.vocab_size`/`all_special_ids`
differ across environments/versions).

Two independent workload lines, per the coordinator's adopted "方案(c)":
- `*_S` (controlled synthetic, e.g. `W1_S`/`W2_S`): mechanism alignment,
  regression, fast debugging. Exact, frozen, versioned prompt token ids
  (see `PromptFixture`/`load_prompt_token_ids`) so both sides -- native
  vLLM and this project's own direct runtime -- are provably running the
  IDENTICAL input, not just "the same distribution." Explicitly NOT
  intended to answer "does this pass the acceptance-rate gate" by
  itself (see `W1_S`/`W2_S`'s own docstring below on the generation-depth
  caveat this line exists to isolate, not paper over).
- `*_R` (representative, e.g. `W1_R`/`W2_R`, NOT YET DEFINED -- design
  only this round, see notes/direct-model-runner-design.md): real
  programming-agent traffic replay, for the actual go/no-go acceptance
  decision. Per `项目实施规划.md`'s own original intent (confirmed
  consistent with the coordinator's framing): the formal gate is
  accepted-tokens/s + ITL + quality regression, NOT acceptance rate by
  itself -- acceptance rate is an explanatory intermediate metric for
  the `-S` line, never the final pass/fail criterion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class Workload:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: tuple[int, ...]


WORKLOADS = {
    "W1": Workload("W1", input_tokens=4096, output_tokens=1024, concurrency=(1, 4)),
    "W2": Workload("W2", input_tokens=32768, output_tokens=1024, concurrency=(1, 4)),
}


@dataclass(frozen=True)
class SamplingConfig:
    """Every field an acceptance-rate comparison must pin identically on
    both sides -- 2026-07-17's temperature confound (native's benchmark
    client silently used a non-zero-by-default temperature while this
    project's own direct runtime is unconditionally greedy) was found
    precisely because this was NOT a tracked, versioned field before."""

    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class StopConfig:
    """`allow_early_eos=False` means force exactly `output_tokens` tokens
    regardless of a real EOS (this project's OWN direct runtime and every
    synthetic-workload measurement so far do this implicitly -- there is
    no real EOS to hit on sequential-token-synthetic input in the first
    place). `allow_early_eos=True` is the representative (`-R`) line's
    mode: a real completion should be allowed to stop early, and forcing
    it to a fixed length would itself distort accepted-tokens/s and
    acceptance-rate statistics relative to real usage."""

    stop_sequences: tuple[str, ...] = ()
    allow_early_eos: bool = True


@dataclass(frozen=True)
class PromptFixture:
    """Pointer to a FROZEN, VERSIONED set of exact prompt token ids for a
    synthetic (`-S`) workload -- committed to the repo (`benchmarks/
    fixtures/<name>.json`), not regenerated from a formula at measurement
    time. This is what makes "both sides ran the identical input" a
    checkable fact instead of an assumption. See
    `benchmarks/generate_synthetic_fixtures.py` for how these were built
    (once, with a recorded seed/formula/tokenizer version) and
    `load_prompt_token_ids()` below for how every measurement script must
    load them (never regenerate)."""

    path: str  # relative to FIXTURES_DIR
    tokenizer: str
    generation_formula: str  # human-readable description of how it was built, for audit
    seed: int
    num_requests: int
    prompt_len: int


def load_prompt_token_ids(fixture: PromptFixture) -> list[list[int]]:
    """Load the exact, frozen prompt token id lists a fixture points to.
    Raises if the file is missing rather than silently regenerating --
    regenerating would defeat the entire point of freezing these."""
    full_path = FIXTURES_DIR / fixture.path
    if not full_path.exists():
        raise FileNotFoundError(
            f"frozen prompt fixture not found: {full_path} -- run "
            "benchmarks/generate_synthetic_fixtures.py first, do not regenerate ad hoc"
        )
    with open(full_path) as f:
        data = json.load(f)
    prompts = data["prompt_token_ids"]
    if len(prompts) != fixture.num_requests or any(len(p) != fixture.prompt_len for p in prompts):
        raise ValueError(
            f"fixture {full_path} does not match its own declared shape "
            f"(num_requests={fixture.num_requests}, prompt_len={fixture.prompt_len})"
        )
    return prompts


# -- Controlled synthetic line (W1-S/W2-S) --
#
# "sequential-token-synthetic" (2026-07-17 rename, per the coordinator's
# explicit instruction): this project's own earlier round called this
# input "random" (matching vLLM's own dataset name, `--dataset-name
# random`) -- but the actual generation formula
# (`allowed_tokens[(offset+index+arange(n)) % len(allowed_tokens)]`, read
# directly from vLLM's `RandomDataset` source) produces a SEQUENTIAL RUN
# of ascending token ids, not i.i.d. random sampling. That distinction
# mattered: it was found to meaningfully raise acceptance rate relative
# to genuine i.i.d. sampling (more locally predictable), so calling it
# "random" was actively misleading about what property of the input was
# actually being measured. Renamed here to describe what it structurally
# is, not what vLLM happens to call it.
W1_S = Workload("W1-S", input_tokens=4096, output_tokens=256, concurrency=(4,))
W2_S = Workload(
    "W2-S", input_tokens=4096, output_tokens=2000, concurrency=(4,)
)  # NOTE: input_tokens intentionally matches W1-S, not W2's 32768 -- this
# is the LONG-GENERATION DEGENERATION test (per the coordinator's point
# 3), not a long-CONTEXT test; its purpose is isolating how acceptance
# rate drifts with generation DEPTH for a fixed input, which does not
# need W2's expensive 32768-token context to study. A true W2-scale
# (32768in) fixture is not built this round -- see the design doc's
# "not attempted this round" section.

_W1_S_FORMULA = (
    "allowed_tokens[(offset + request_index + arange(input_tokens)) "
    "% len(allowed_tokens)], allowed_tokens = sorted(set(range(vocab_size)) "
    "- set(tokenizer.all_special_ids)), offset = seeded per-request random draw "
    "(matches vllm.benchmarks.datasets.datasets.RandomDataset.generate_token_sequence's "
    "own formula, generated ONCE and frozen -- see generate_synthetic_fixtures.py)"
)

W1_S_FIXTURE = PromptFixture(
    path="w1s_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=16,
    prompt_len=4096,
)

# 2026-07-17 addition: the n=16 fixture above gave only ~1.6 combined
# standard errors on the native-vs-this-runtime gap -- not decisive. This
# larger, SEPARATELY frozen and versioned fixture (same seed, same
# formula -- its first 16 entries are therefore bit-identical to
# W1_S_FIXTURE's, a deliberate cross-check property, not a coincidence)
# extends to num_requests=128 for a properly powered comparison. Kept as
# a genuinely SEPARATE fixture file (not overwriting w1s_prompts.json) so
# the original 16-request round's exact numbers remain independently
# reproducible.
W1_S_FIXTURE_N128 = PromptFixture(
    path="w1s_prompts_n128.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=128,
    prompt_len=4096,
)

# 2026-07-18, Phase D1 (shape-generalization sweep, per the session-review
# doc's own falsifier): the review flagged that context length was only
# ever measured at 4096 (W1-S), and that the true W2-scale 32768-context
# fixture is explicitly "not built" (see W2_S's docstring above). These two
# fixtures are a SAME-FORMULA, SAME-SEED, SAME-num_requests=16 extension of
# W1_S_FIXTURE to prompt_len=16384/32768 -- built ONLY to give Phase D1 a
# comparable frozen input at longer context. They are explicitly NOT the
# official W2/W2-S fixture (no representative -R traffic, no real
# programming-agent replay) -- just this task's own constructed synthetic
# fixture, labeled as such so it is never mistaken for W2_S or a
# `项目实施规划.md`-blessed benchmark.
D1_CTX16K_FIXTURE = PromptFixture(
    path="d1_ctx16k_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=16,
    prompt_len=16384,
)

D1_CTX32K_FIXTURE = PromptFixture(
    path="d1_ctx32k_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=16,
    prompt_len=32768,
)

# 2026-07-18, D1 sweep continuation (native/ours gap trend at 4K/16K/32K:
# 2.08x -> 1.116x, narrowing -- the natural next question is whether this
# runtime overtakes native at 64K). SAME formula/seed/num_requests=16
# extension pattern as the two fixtures above -- NOT the official W2/W2-S
# line, same caveat applies. NOTE (found while wiring this up, see the
# 2026-07-18 §16 safety note): this runtime's own benchmark scripts
# hardcode `blocks_per_slot=640` with `block_size=64` (40960-token/slot
# capacity ceiling) -- a single 65536-token prompt already exceeds that
# ceiling on its own (prefill alone, before any generation), independent
# of concurrency. This fixture is built anyway (cheap, CPU-only, useful
# regardless) but running `mtp_w1s_our_runtime_perf.py --fixture ctx64k`
# against this runtime will raise a capacity RuntimeError until that
# ceiling is deliberately raised -- see the note for the precise scoped fix.
D1_CTX64K_FIXTURE = PromptFixture(
    path="d1_ctx64k_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=16,
    prompt_len=65536,
)

# 2026-07-19, multi-agent-coding-workload re-prioritization: the user's real
# target workload (256K architectural context requirement, concurrency=4 as
# the core scenario at 64K/128K/200K/256K) needs frozen fixtures beyond 64K.
# SAME formula/seed as every fixture above -- deliberately NOT num_requests=16
# like the D1 fixtures: the only concurrency this task's matrix actually needs
# is 4, and a 16-request fixture at 200K/256K would be a ~4x larger frozen
# JSON file (tens of MB) and ~4x more one-time tokenizer-formula compute for
# rows this task never uses -- a deliberate, explicitly-flagged deviation from
# the D1 convention, not an oversight. Still NOT the official W2/W2-S line
# (see W1_S/W2_S's own docstring above).
CTX128K_FIXTURE = PromptFixture(
    path="ctx128k_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=4,
    prompt_len=131072,
)

CTX200K_FIXTURE = PromptFixture(
    path="ctx200k_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=4,
    prompt_len=204800,
)

# ctx256k's prompt_len is deliberately NOT 262144 (the literal "256*1024").
# Confirmed from unsloth/Qwen3.6-27B-NVFP4's real config.json
# (text_config.max_position_embeddings): this model's own native RoPE
# ("default" rope_type, no YaRN/NTK scaling configured) supports AT MOST
# 262144 total positions -- 262144 IS the model's real architectural ceiling,
# not a round-number choice by this project. A prompt_len of exactly 262144
# would leave ZERO room for even one generated token before exceeding that
# ceiling. This fixture's prompt_len=261888 (262144 - 256) is chosen so that
# prompt_len + max_tokens(256, this whole doc's standard) lands EXACTLY at
# the architectural limit -- the most honest way to test "does 256K context
# actually work," at the real edge, not comfortably inside it with slack this
# model does not actually have available. See mtp_w1s_our_runtime_perf.py's
# _MODEL_MAX_POSITION_EMBEDDINGS constant, which caps max_model_len at this
# same 262144 figure for the same reason.
CTX256K_FIXTURE = PromptFixture(
    path="ctx256k_prompts.json",
    tokenizer="unsloth/Qwen3.6-27B-NVFP4",
    generation_formula=_W1_S_FORMULA,
    seed=12345,
    num_requests=4,
    prompt_len=261888,
)


def main() -> None:
    print(json.dumps({name: asdict(workload) for name, workload in WORKLOADS.items()}, indent=2))


if __name__ == "__main__":
    main()
