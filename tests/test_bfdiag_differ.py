"""differ.py: config diff, metric diff, and the comparability verdict.

``test_differ_flags_the_2026_07_27_incident`` is the core acceptance test
called for in the bfdiag task brief: it reconstructs the real incident
where two acceptance-rate measurements (1.000 vs 0.687) were compared as if
they meant something, when in fact they used different prompts. The whole
tool exists to make that mistake loud and immediate.
"""

from __future__ import annotations

from bfdiag.record.differ import (
    DEFAULT_COMPARABLE_FIELDS,
    check_comparability,
    diff_configs,
    diff_metrics,
    diff_records,
    format_text,
    to_jsonable,
)
from bfdiag.record.schema import (
    Fingerprint,
    GitRepoInfo,
    ModelInfo,
    PythonInfo,
    RunRecord,
    WorkloadInfo,
)

_COMMON_GIT = {
    "qwen-sm120-runtime": GitRepoInfo(sha="a" * 40, dirty=False, branch="main"),
    "sparkinfer": GitRepoInfo(sha="b" * 40, dirty=False, branch="main"),
    "vllm": GitRepoInfo(sha="c" * 40, dirty=False, branch="main"),
}


def _record(run_id: str, *, prompt_hash: str, acceptance_rate: float, **workload_overrides):
    workload = dict(
        prompt_hash=prompt_hash,
        prompt_len=65536,
        k=15,
        seed=0,
        greedy=True,
        block_size=64,
        capacity=128,
        max_model_len=67586,
    )
    workload.update(workload_overrides)
    return RunRecord(
        run_id=run_id,
        started_at="2026-07-27T10:00:00+00:00",
        finished_at="2026-07-27T10:05:00+00:00",
        script="benchmarks/laguna_vllm_dflash_baseline.py",
        argv=[],
        status="ok",
        fingerprint=Fingerprint(
            git=dict(_COMMON_GIT),
            env={},
            python=PythonInfo(
                version="3.12.3", torch="2.11.0", vllm="0.26.0", transformers="4.45.0"
            ),
            model=ModelInfo(
                path="poolside/Laguna-S-2.1-NVFP4", revision="0761412", dtype="bfloat16"
            ),
            workload=WorkloadInfo(**workload),
        ),
        metrics={"acceptance_rate": acceptance_rate},
    )


# --- the incident this tool exists to prevent -----------------------------


def test_differ_flags_the_2026_07_27_incident() -> None:
    """Same everything except the prompt -- and wildly different acceptance
    rates as a result. This must be flagged NOT COMPARABLE, and the flagged
    field must be exactly prompt_hash (nothing else differs).
    """
    run_ours = _record(
        "ours0000001", prompt_hash="9c02b1a2c3d4e5f60123456789abcdef", acceptance_rate=0.687
    )
    run_vllm = _record(
        "vllm0000001", prompt_hash="a3f1c2d3e4f5061728394a5b6c7d8e9f", acceptance_rate=1.000
    )

    result = diff_records(run_ours, run_vllm)

    assert result.comparable is False
    assert [b.path for b in result.comparability_breaks] == ["fingerprint.workload.prompt_hash"]

    text = format_text(result)
    first_line = text.splitlines()[0]
    assert first_line.startswith("⚠ NOT COMPARABLE: workload.prompt_hash differs")
    assert "…" in first_line  # the long hashes get truncated, not dumped in full

    payload = to_jsonable(result)
    assert payload["comparable"] is False
    assert payload["comparability_breaks"] == [
        {
            "field": "workload.prompt_hash",
            "a": "9c02b1a2c3d4e5f60123456789abcdef",
            "b": "a3f1c2d3e4f5061728394a5b6c7d8e9f",
        }
    ]

    # The metric diff is still computed and still meaningful to report --
    # the tool doesn't hide the numbers, it labels them untrustworthy.
    metric = {m.name: m for m in result.metric_diffs}["acceptance_rate"]
    assert metric.a == 0.687
    assert metric.b == 1.000


def test_comparable_runs_produce_no_warning() -> None:
    run_a = _record("runA00000001", prompt_hash="deadbeef" * 4, acceptance_rate=0.90)
    run_b = _record("runB00000001", prompt_hash="deadbeef" * 4, acceptance_rate=0.91)

    result = diff_records(run_a, run_b)

    assert result.comparable is True
    assert result.comparability_breaks == []
    text = format_text(result)
    assert "⚠ NOT COMPARABLE" not in text
    assert "OK: all comparability-critical fields match." in text


def test_check_comparability_detects_each_default_field_independently() -> None:
    base = _record("base00000001", prompt_hash="same" * 8, acceptance_rate=1.0)

    # k differs
    other = _record("other0000001", prompt_hash="same" * 8, acceptance_rate=1.0, k=5)
    breaks = check_comparability(base, other)
    assert [b.path for b in breaks] == ["fingerprint.workload.k"]

    # greedy differs
    other = _record("other0000002", prompt_hash="same" * 8, acceptance_rate=1.0, greedy=False)
    breaks = check_comparability(base, other)
    assert [b.path for b in breaks] == ["fingerprint.workload.greedy"]

    # block_size differs
    other = _record("other0000003", prompt_hash="same" * 8, acceptance_rate=1.0, block_size=128)
    breaks = check_comparability(base, other)
    assert [b.path for b in breaks] == ["fingerprint.workload.block_size"]


def test_check_comparability_flags_git_sha_drift() -> None:
    run_a = _record("gitA00000001", prompt_hash="same" * 8, acceptance_rate=1.0)
    run_b = _record("gitB00000001", prompt_hash="same" * 8, acceptance_rate=1.0)
    run_b.fingerprint.git["vllm"] = GitRepoInfo(sha="d" * 40, dirty=False, branch="main")

    breaks = check_comparability(run_a, run_b)
    assert [b.path for b in breaks] == ["fingerprint.git.vllm.sha"]


def test_default_comparable_fields_cover_the_documented_set() -> None:
    expected_leaves = {
        "workload.prompt_hash",
        "model.revision",
        "git.qwen-sm120-runtime.sha",
        "git.sparkinfer.sha",
        "git.vllm.sha",
        "workload.k",
        "workload.greedy",
        "workload.block_size",
        "workload.max_model_len",
        # Added 2026-07-27 after a real incident: a warm daemon defaulting to
        # blocks_per_slot=4096 was compared against a cold-start script
        # deriving 130, acceptance 0.6754 vs 0.452525, and `bf diff` called
        # them comparable because neither field was listed here.
        "workload.blocks_per_slot",
        "workload.capacity",
        # Added by the bfdiag determinism/force-sync work (2026-07-27): see
        # bfdiag/determinism.py + bfdiag/record/fingerprint.py's
        # capture_determinism(). QSR_FORCE_SYNC breaks the async pipeline on
        # purpose, so it must gate comparability like every other field here.
        "extra.determinism.force_sync",
    }
    leaves = {p[len("fingerprint.") :] for p in DEFAULT_COMPARABLE_FIELDS}
    assert leaves == expected_leaves


def test_differ_flags_the_warm_daemon_blocks_per_slot_incident() -> None:
    """The second real incident this tool exists to prevent.

    A warm daemon ran with ``blocks_per_slot=4096`` (its own silent
    default) while the cold-start script derived 130 from the context
    length. Everything else -- prompt, block_size, K, greedy -- matched, so
    the two acceptance rates (0.6754 vs 0.452525) looked directly
    comparable, and the 0.22 gap looked like the block_size=64->128
    regression under investigation. ``blocks_per_slot`` sets sparkinfer's
    decode-workspace ``max_pages``, so it moves float reduction order and
    therefore acceptance; it must gate comparability.
    """
    same_prompt = "7e3a" * 8
    warm = _record(
        "warmdaemon1", prompt_hash=same_prompt, acceptance_rate=0.6754, blocks_per_slot=4096
    )
    cold = _record(
        "coldstart01", prompt_hash=same_prompt, acceptance_rate=0.452525, blocks_per_slot=130
    )

    result = diff_records(warm, cold)

    assert result.comparable is False
    assert [b.path for b in result.comparability_breaks] == ["fingerprint.workload.blocks_per_slot"]
    assert "NOT COMPARABLE" in format_text(result)


# --- config diff -----------------------------------------------------------


def test_diff_configs_reports_only_differing_fields() -> None:
    run_a = _record("cfgA00000001", prompt_hash="same" * 8, acceptance_rate=1.0)
    run_b = _record("cfgB00000001", prompt_hash="same" * 8, acceptance_rate=1.0)
    run_b.fingerprint.python.torch = "2.12.0"

    diffs = {d.path: d for d in diff_configs(run_a, run_b)}
    assert "fingerprint.python.torch" in diffs
    assert diffs["fingerprint.python.torch"].a == "2.11.0"
    assert diffs["fingerprint.python.torch"].b == "2.12.0"
    # unrelated fields (e.g. model.path) must not show up as diffs
    assert "fingerprint.model.path" not in diffs


def test_diff_configs_identical_records_yield_no_diffs() -> None:
    run_a = _record("idA000000001", prompt_hash="same" * 8, acceptance_rate=1.0)
    run_b = _record("idB000000001", prompt_hash="same" * 8, acceptance_rate=1.0)
    assert diff_configs(run_a, run_b) == []


# --- metric diff -------------------------------------------------------


def test_diff_metrics_computes_relative_change_percent() -> None:
    run_a = _record("metA00000001", prompt_hash="same" * 8, acceptance_rate=0.687)
    run_b = _record("metB00000001", prompt_hash="same" * 8, acceptance_rate=1.0)

    diffs = {d.name: d for d in diff_metrics(run_a, run_b)}
    accept = diffs["acceptance_rate"]
    assert accept.a == 0.687
    assert accept.b == 1.0
    expected_pct = (1.0 - 0.687) / 0.687 * 100.0
    assert accept.delta_pct is not None
    assert abs(accept.delta_pct - expected_pct) < 1e-9


def test_diff_metrics_handles_metric_missing_from_one_side() -> None:
    run_a = _record("onlyA0000001", prompt_hash="same" * 8, acceptance_rate=0.9)
    run_b = _record("onlyB0000001", prompt_hash="same" * 8, acceptance_rate=0.9)
    run_b.metrics["tok_per_s"] = 253.0

    diffs = {d.name: d for d in diff_metrics(run_a, run_b)}
    assert diffs["tok_per_s"].a is None
    assert diffs["tok_per_s"].b == 253.0
    assert diffs["tok_per_s"].delta_pct is None  # can't compute a % change from nothing


def test_diff_metrics_zero_baseline_does_not_divide_by_zero() -> None:
    run_a = _record("zeroA0000001", prompt_hash="same" * 8, acceptance_rate=0.0)
    run_b = _record("zeroB0000001", prompt_hash="same" * 8, acceptance_rate=0.5)
    diffs = {d.name: d for d in diff_metrics(run_a, run_b)}
    assert diffs["acceptance_rate"].delta_pct is None
