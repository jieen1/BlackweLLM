"""CPU-only contracts for ``bfdiag.divergence.qwen36_capture``.

Mirrors ``tests/test_bfdiag_divergence.py``'s style: plain Python values,
no torch, no GPU, no model -- see that module's docstring and
``bfdiag/divergence/qwen36_capture.py``'s module docstring for why the
capture *functions* themselves (which do call ``torch``) are GPU-only and
NOT exercised here, same as the existing Laguna
``EngineCaptureSource``/``capture_engine_activations`` precedent. What IS
exercised: the pure trace-shape glue (``build_trace_from_captured_values``),
the dataclass wiring (``Qwen36EngineCaptureSource``/
``Qwen36HFOracleCaptureSource`` satisfy ``CaptureSource``), and that the
resulting trace shape actually plugs into ``scan_layers`` without any
special-casing on bfdiag's side.
"""

from __future__ import annotations

from bfdiag.divergence.capture import CaptureSource
from bfdiag.divergence.qwen36_capture import (
    Qwen36EngineCaptureSource,
    Qwen36HFOracleCaptureSource,
    build_trace_from_captured_values,
)
from bfdiag.divergence.scan import scan_layers


class TestBuildTraceFromCapturedValues:
    def test_layer_indices_match_input_order(self) -> None:
        trace = build_trace_from_captured_values(["h0", "h1", "h2"], "logits")
        assert set(trace) == {0, 1, 2, 3}
        assert trace[0] == {"hidden_state": "h0"}
        assert trace[1] == {"hidden_state": "h1"}
        assert trace[2] == {"hidden_state": "h2"}

    def test_logits_land_at_a_sentinel_index_past_the_last_layer(self) -> None:
        trace = build_trace_from_captured_values(["h0", "h1"], "final_logits")
        assert trace[2] == {"logits": "final_logits"}
        # Never collides with a real 0..num_layers-1 hidden_state entry.
        assert "hidden_state" not in trace[2]

    def test_empty_layers_still_places_logits_at_index_zero(self) -> None:
        trace = build_trace_from_captured_values([], "logits")
        assert trace == {0: {"logits": "logits"}}

    def test_num_layers_64_matches_qwen36s_real_depth(self) -> None:
        # Qwen3.6-27B has 64 decoder layers (48 linear_attention + 16
        # full_attention, B0), not Laguna's 48 -- confirm this module
        # doesn't hardcode Laguna's depth anywhere in the indexing math.
        hidden = [f"h{i}" for i in range(64)]
        trace = build_trace_from_captured_values(hidden, "logits")
        assert set(trace) == set(range(65))
        assert trace[63] == {"hidden_state": "h63"}
        assert trace[64] == {"logits": "logits"}


class TestCaptureSourceConformance:
    def test_engine_source_satisfies_the_protocol(self) -> None:
        source = Qwen36EngineCaptureSource(model=object())
        assert isinstance(source, CaptureSource)

    def test_hf_oracle_source_satisfies_the_protocol(self) -> None:
        source = Qwen36HFOracleCaptureSource(hf_model=object())
        assert isinstance(source, CaptureSource)


class TestPluggedIntoScanLayers:
    """The actual point of this module: two traces built the way the real
    capture functions build them must scan cleanly, with zero special
    handling on bfdiag's side for a second architecture."""

    def test_identical_traces_never_diverge(self) -> None:
        hidden = [[float(i), float(i) + 1.0, float(i) + 2.0] for i in range(5)]
        oracle = build_trace_from_captured_values(hidden, [1.0, 2.0, 3.0])
        candidate = build_trace_from_captured_values(hidden, [1.0, 2.0, 3.0])
        report = scan_layers(oracle, candidate)
        assert not report.has_divergence

    def test_a_flipped_layer_is_found_including_the_logits_sentinel(self) -> None:
        hidden = [[10.0 * (i + 1), 10.0 * (i + 2), 10.0 * (i + 3)] for i in range(5)]
        oracle = build_trace_from_captured_values(hidden, [10.0, 20.0, 30.0])
        bad_hidden = list(hidden)
        bad_hidden[2] = [-value for value in hidden[2]]
        candidate = build_trace_from_captured_values(bad_hidden, [10.0, 20.0, 30.0])

        report = scan_layers(oracle, candidate)
        assert report.has_divergence
        assert report.first_divergent_layer == 2
        assert "hidden_state" in report.first_divergent_submodules

    def test_a_flipped_logits_entry_is_found_at_the_sentinel_index(self) -> None:
        hidden = [[10.0 * (i + 1), 10.0 * (i + 2), 10.0 * (i + 3)] for i in range(5)]
        oracle = build_trace_from_captured_values(hidden, [10.0, 20.0, 30.0])
        candidate = build_trace_from_captured_values(hidden, [-10.0, -20.0, -30.0])

        report = scan_layers(oracle, candidate)
        assert report.has_divergence
        assert report.first_divergent_layer == 5
        assert "logits" in report.first_divergent_submodules
