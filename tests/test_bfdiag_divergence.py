"""Core acceptance tests for bfdiag.divergence.scan.

Builds a synthetic 42-layer activation trace (five submodules per layer:
input_layernorm, self_attn, post_attention_layernorm, mlp, mlp.gate), all on
plain Python floats -- no torch, no GPU, no model. Every layer/submodule
gets tiny, depth-growing "natural drift" noise (candidate != oracle exactly,
but well inside the depth-relaxed thresholds); one test additionally injects
a large, decisive bias at one (layer, submodule) pair and asserts the
scanner finds exactly it. This is the success criterion the whole bfdiag
oracle-divergence task is graded on.
"""

from __future__ import annotations

import math

import pytest

from bfdiag.divergence.capture import FakeCaptureSource
from bfdiag.divergence.cli import _load_prompt_token_ids, _parse_layers, scan_prompt
from bfdiag.divergence.report import format_text_report, to_json_dict
from bfdiag.divergence.scan import ActivationTrace, scan_layers

DIM = 64
NUM_LAYERS = 42
SUBMODULES = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp", "mlp.gate")


def _base_vector(seed: float, dim: int = DIM) -> list[float]:
    """A deterministic, non-degenerate vector with well-separated magnitudes.

    The dominant ``10 * (i + 1)`` ramp keeps every component's rank strictly
    ordered and far apart (gaps of 10.0) so that the tiny natural-drift noise
    below (amplitude <= ~1e-3) can never flip which index is top-1/top-5 --
    only a real injected bug should be able to do that. The ``sin(seed +
    ...)`` term just differentiates vectors across (layer, submodule) seeds.
    """
    return [10.0 * (i + 1) + math.sin(seed + 0.37 * i) for i in range(dim)]


def _natural_drift(vector: list[float], layer_idx: int, amplitude: float = 1e-4) -> list[float]:
    """Tiny additive noise growing with sqrt(layer_idx) -- the same
    random-walk-style model bfdiag.divergence.thresholds relaxes against."""
    eps = amplitude * math.sqrt(layer_idx + 1)
    return [value + eps * math.sin(0.53 * index + layer_idx) for index, value in enumerate(vector)]


def _build_traces(
    *, inject_layer: int | None = None, inject_submodule: str | None = None
) -> tuple[ActivationTrace, ActivationTrace]:
    oracle: dict[int, dict[str, list[float]]] = {}
    candidate: dict[int, dict[str, list[float]]] = {}
    for layer_idx in range(NUM_LAYERS):
        oracle_layer: dict[str, list[float]] = {}
        candidate_layer: dict[str, list[float]] = {}
        for sub_idx, name in enumerate(SUBMODULES):
            base = _base_vector(layer_idx * 3.1 + sub_idx * 7.3)
            oracle_layer[name] = base
            if layer_idx == inject_layer and name == inject_submodule:
                # Decisive divergence: an exact negation drives cosine to
                # -1.0, clearly below any threshold at any depth.
                candidate_layer[name] = [-value for value in base]
            else:
                candidate_layer[name] = _natural_drift(base, layer_idx)
        oracle[layer_idx] = oracle_layer
        candidate[layer_idx] = candidate_layer
    return oracle, candidate


def test_injected_bias_at_layer_17_is_the_first_divergent_layer() -> None:
    oracle, candidate = _build_traces(inject_layer=17, inject_submodule="mlp")
    report = scan_layers(oracle, candidate)

    assert report.has_divergence
    assert report.first_divergent_layer == 17
    assert "mlp" in report.first_divergent_submodules

    for layer in report.layers:
        if layer.layer_idx < 17:
            assert layer.passed, f"layer {layer.layer_idx} unexpectedly failed before the bug"

    layer_17 = next(item for item in report.layers if item.layer_idx == 17)
    mlp_verdict = next(item for item in layer_17.submodules if item.submodule == "mlp")
    assert not mlp_verdict.passed
    assert mlp_verdict.cosine_similarity == pytest.approx(-1.0, abs=1e-6)
    # sibling submodules at the SAME layer must still pass: the drill-down
    # has to name "mlp" specifically, not the whole layer indiscriminately.
    other_verdicts = [item for item in layer_17.submodules if item.submodule != "mlp"]
    assert all(item.passed for item in other_verdicts)


def test_injected_bias_at_a_different_submodule_is_still_pinpointed() -> None:
    # Same construction, different (layer, submodule): proves drill-down
    # isn't hardcoded to "mlp" -- it generalizes to whatever actually broke.
    oracle, candidate = _build_traces(inject_layer=5, inject_submodule="mlp.gate")
    report = scan_layers(oracle, candidate)

    assert report.first_divergent_layer == 5
    assert report.first_divergent_submodules == ("mlp.gate",)


def test_no_bias_at_all_means_no_divergence() -> None:
    oracle, _ = _build_traces()
    identical_candidate = {
        layer_idx: dict(submodules) for layer_idx, submodules in oracle.items()
    }
    report = scan_layers(oracle, identical_candidate)

    assert not report.has_divergence
    assert report.first_divergent_layer is None
    assert all(layer.passed for layer in report.layers)


def test_deep_layer_natural_drift_does_not_false_positive() -> None:
    # No injected bug anywhere -- every layer only has the tiny, depth-
    # growing natural-drift noise. This must not trip the scanner at ANY
    # depth, including the deepest layer where thresholds are loosest.
    oracle, candidate = _build_traces()
    report = scan_layers(oracle, candidate)

    assert not report.has_divergence
    deepest = report.layers[-1]
    assert deepest.layer_idx == NUM_LAYERS - 1
    assert deepest.passed
    # Confirm this is a real (if tiny) drift, not an accidentally-identical
    # vector -- otherwise this test would be indistinguishable from the
    # "no bias at all" case above and would prove nothing extra.
    assert any(item.cosine_similarity < 1.0 for item in deepest.submodules)


def test_missing_layers_on_one_side_are_simply_skipped() -> None:
    oracle, candidate = _build_traces()
    trimmed_candidate = {k: v for k, v in candidate.items() if k != 3}
    report = scan_layers(oracle, trimmed_candidate)
    assert {layer.layer_idx for layer in report.layers} == set(oracle) - {3}


def test_empty_oracle_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        scan_layers({}, {0: {"self_attn": [1.0, 2.0]}})


def test_disjoint_layer_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="no layer indices"):
        scan_layers({0: {"self_attn": [1.0]}}, {1: {"self_attn": [1.0]}})


def test_scan_prompt_wires_two_fake_capture_sources_end_to_end() -> None:
    oracle, candidate = _build_traces(inject_layer=17, inject_submodule="mlp")
    oracle_source = FakeCaptureSource(trace=oracle)
    engine_source = FakeCaptureSource(trace=candidate)

    report = scan_prompt(oracle_source, engine_source, [1, 2, 3, 4])

    assert report.first_divergent_layer == 17
    assert oracle_source.seen_prompts == [(1, 2, 3, 4)]
    assert engine_source.seen_prompts == [(1, 2, 3, 4)]


def test_report_text_marks_the_first_divergent_layer_and_drills_down() -> None:
    oracle, candidate = _build_traces(inject_layer=17, inject_submodule="mlp")
    report = scan_layers(oracle, candidate)

    text = format_text_report(report)

    assert "layer 17" in text
    assert "first divergent layer" in text
    assert "mlp" in text
    assert "第一个发散点: layer 17 的 mlp" in text


def test_report_text_states_no_divergence_when_none_found() -> None:
    oracle, _ = _build_traces()
    identical_candidate = {layer_idx: dict(submods) for layer_idx, submods in oracle.items()}
    report = scan_layers(oracle, identical_candidate)

    text = format_text_report(report)
    assert "未发现发散" in text


def test_report_json_round_trips_the_first_divergence() -> None:
    oracle, candidate = _build_traces(inject_layer=5, inject_submodule="mlp.gate")
    report = scan_layers(oracle, candidate)

    payload = to_json_dict(report)

    assert payload["first_divergent_layer"] == 5
    assert payload["first_divergent_submodules"] == ["mlp.gate"]
    assert payload["has_divergence"] is True
    layer_5 = next(item for item in payload["layers"] if item["layer_idx"] == 5)
    gate_entry = next(item for item in layer_5["submodules"] if item["submodule"] == "mlp.gate")
    assert gate_entry["passed"] is False

    verdict_layer_5 = next(item for item in report.layers if item.layer_idx == 5)
    gate_verdict = next(item for item in verdict_layer_5.submodules if item.submodule == "mlp.gate")
    assert gate_entry["threshold"]["min_cosine"] == pytest.approx(gate_verdict.threshold.min_cosine)


def test_cli_parse_layers_supports_all_ranges_and_lists() -> None:
    assert _parse_layers("all", 5) == (0, 1, 2, 3, 4)
    assert _parse_layers("0-2,4", 10) == (0, 1, 2, 4)
    assert _parse_layers("7", 10) == (7,)


def test_cli_load_prompt_token_ids_reads_a_json_list(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert _load_prompt_token_ids(str(prompt_path)) == [1, 2, 3]


def test_cli_load_prompt_token_ids_reports_a_clear_error_for_missing_file(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(FileNotFoundError, match="token-ids fixture"):
        _load_prompt_token_ids(str(missing))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
