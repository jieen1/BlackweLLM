"""CPU-only tests for the flight recorder's dump/read/panel/CLI layer:
``bfdiag.trace.events``, ``bfdiag.trace.dump``, ``bfdiag.trace.panel``, and
``bfdiag.trace.cli``.
"""

from __future__ import annotations

import argparse
import json

import pytest

from bfdiag.trace import cli as trace_cli
from bfdiag.trace import panel
from bfdiag.trace.dump import read_trace, resolve_run_dir, trace_path_for_run, write_trace
from bfdiag.trace.events import CgMissReason, Path, RoundEvent, path_label, reason_label
from bfdiag.trace.ring import RoundRing


def _make_ring(rows: list[dict]) -> RoundRing:
    ring = RoundRing(max(len(rows), 1), use_cuda=False)
    for row in rows:
        r = ring.begin_round(row.get("slot", 0), row.get("kv_len_before", 0))
        ring.finish_round(
            r,
            3,
            path=row.get("path", Path.CG_REPLAY),
            cg_miss_reason=row.get("cg_miss_reason", CgMissReason.NONE),
            draft_tokens_n=row.get("draft_tokens_n", 15),
            accepted_n=row.get("accepted_n", 15),
            reject_position=row.get("reject_position", -1),
            bonus_token=row.get("bonus_token", 0),
        )
    return ring


class TestEvents:
    def test_path_and_reason_labels_are_stable_strings(self):
        assert path_label(Path.CG_REPLAY) == "cg_replay"
        assert path_label(Path.EAGER) == "eager"
        assert path_label(Path.CG_MISS) == "cg_miss"
        assert reason_label(CgMissReason.NONE) == "none"
        assert reason_label(CgMissReason.BATCH_SIZE_MISMATCH) == "batch_size_mismatch"

    def test_unknown_code_does_not_raise(self):
        assert "unknown" in path_label(99)
        assert "unknown" in reason_label(99)

    def test_round_event_json_roundtrip(self):
        ev = RoundEvent(
            round_idx=1,
            slot=0,
            kv_len_before=10,
            path="cg_replay",
            cg_miss_reason="none",
            draft_tokens_n=15,
            accepted_n=10,
            reject_position=10,
            bonus_token=5,
            mem_allocated=1024,
            t_main_forward=0.0,
            t_draft=1.5,
            t_verify=2.5,
            t_commit=0.3,
            t_round=4.3,
        )
        assert RoundEvent.from_json(ev.to_json()) == ev


class TestDumpReadRoundtrip:
    def test_write_then_read_matches_snapshot(self, tmp_path):
        ring = _make_ring(
            [
                {"kv_len_before": 0, "accepted_n": 15, "reject_position": -1},
                {"kv_len_before": 15, "accepted_n": 3, "reject_position": 3},
            ]
        )
        out = write_trace(ring, tmp_path / "trace.jsonl")
        assert out.exists()
        loaded = read_trace(out)
        assert len(loaded) == 2
        assert [e.kv_len_before for e in loaded] == [0, 15]
        assert loaded[1].reject_position == 3

    def test_write_creates_parent_dirs(self, tmp_path):
        ring = _make_ring([{"kv_len_before": 0}])
        nested = tmp_path / "a" / "b" / "trace.jsonl"
        write_trace(ring, nested)
        assert nested.exists()

    def test_path_helpers(self, tmp_path):
        run_dir = resolve_run_dir(tmp_path, "run-1")
        assert run_dir == tmp_path / "runs" / "run-1"
        assert trace_path_for_run(tmp_path, "run-1") == run_dir / "trace.jsonl"


class TestPanelStats:
    def _rows(self) -> list[RoundEvent]:
        ring = _make_ring(
            [
                {"kv_len_before": i, "accepted_n": 15, "reject_position": -1}
                for i in range(5)
            ]
            + [{"kv_len_before": 5, "accepted_n": 2, "reject_position": 2}]
        )
        return ring.snapshot()

    def test_acceptance_rate(self):
        rows = self._rows()
        stats = panel.compute_stats(rows)
        total_draft = 15 * 6
        total_accepted = 15 * 5 + 2
        assert stats.acceptance_rate == pytest.approx(total_accepted / total_draft)

    def test_reject_position_histogram(self):
        rows = self._rows()
        stats = panel.compute_stats(rows)
        assert stats.reject_position_histogram[-1] == 5
        assert stats.reject_position_histogram[2] == 1

    def test_cg_hit_rate_and_path_counts(self):
        rows = self._rows()
        stats = panel.compute_stats(rows)
        assert stats.cg_hit_rate == 1.0
        assert stats.path_counts == {"cg_replay": 6}

    def test_empty_trace_is_handled(self):
        stats = panel.compute_stats([])
        assert stats.num_rounds == 0
        assert stats.acceptance_rate is None
        assert stats.outliers == []
        assert stats.dropped == 0

    def test_dropped_count_reflects_ring_wraparound(self):
        # A ring of capacity 3 fed 5 rounds has overwritten rounds 0 and 1
        # (their round_idx values) by the time we snapshot -- the earliest
        # surviving row's round_idx (2) IS the drop count, never silent.
        ring = RoundRing(3, use_cuda=False)
        for i in range(5):
            r = ring.begin_round(0, i)
            ring.finish_round(
                r,
                3,
                path=Path.CG_REPLAY,
                cg_miss_reason=CgMissReason.NONE,
                draft_tokens_n=15,
                accepted_n=15,
                reject_position=-1,
                bonus_token=0,
            )
        rows = ring.snapshot()
        stats = panel.compute_stats(rows)
        assert stats.dropped == 2
        assert "dropped" in panel.render_summary(stats)

    def test_no_wraparound_reports_zero_dropped(self):
        rows = self._rows()
        stats = panel.compute_stats(rows)
        assert stats.dropped == 0
        assert "dropped" not in panel.render_summary(stats)

    def test_outlier_detection_flags_the_slow_round(self):
        ring = RoundRing(20, use_cuda=False)
        rows = []
        for i in range(15):
            r = ring.begin_round(0, i)
            ring.finish_round(
                r,
                3,
                path=Path.CG_REPLAY,
                cg_miss_reason=CgMissReason.NONE,
                draft_tokens_n=15,
                accepted_n=15,
                reject_position=-1,
                bonus_token=0,
            )
        rows = ring.snapshot()
        # Manufacture one artificially slow round to stand in for the "270s
        # mystery" -- direct RoundEvent construction is simpler than forcing
        # a real 9-order-of-magnitude timing gap through Timeline marks.
        rows[7].t_round = 9_000.0
        stats = panel.compute_stats(rows)
        assert len(stats.outliers) == 1
        assert stats.outliers[0]["round_idx"] == rows[7].round_idx

    def test_render_summary_and_table_do_not_crash(self):
        rows = self._rows()
        stats = panel.compute_stats(rows)
        table = panel.render_round_table(rows, limit=3)
        assert "showing last 3 of 6" in table
        summary = panel.render_summary(stats)
        assert "acceptance rate" in summary
        rendered_json = panel.render_json(rows, stats)
        parsed = json.loads(rendered_json)
        assert len(parsed["rounds"]) == 6
        assert "summary" in parsed


class TestPanelDiff:
    def test_identical_traces_do_not_diverge(self):
        rows = _make_ring([{"kv_len_before": i} for i in range(3)]).snapshot()
        result = panel.diff_traces(rows, rows)
        assert result.first_divergence_round is None

    def test_first_divergence_is_reported(self):
        a = _make_ring(
            [
                {"kv_len_before": 0, "accepted_n": 15, "reject_position": -1},
                {"kv_len_before": 15, "accepted_n": 15, "reject_position": -1},
                {"kv_len_before": 30, "accepted_n": 15, "reject_position": -1},
            ]
        ).snapshot()
        b = _make_ring(
            [
                {"kv_len_before": 0, "accepted_n": 15, "reject_position": -1},
                {"kv_len_before": 15, "accepted_n": 4, "reject_position": 4},
                {"kv_len_before": 19, "accepted_n": 15, "reject_position": -1},
            ]
        ).snapshot()
        result = panel.diff_traces(a, b)
        assert result.first_divergence_round == 1
        assert "accepted_n" in result.diverging_fields
        assert "reject_position" in result.diverging_fields

    def test_length_mismatch_with_no_field_divergence(self):
        a = _make_ring([{"kv_len_before": 0}]).snapshot()
        b = _make_ring([{"kv_len_before": 0}, {"kv_len_before": 15}]).snapshot()
        result = panel.diff_traces(a, b)
        assert result.first_divergence_round is None
        assert result.len_a == 1
        assert result.len_b == 2
        rendered = panel.render_diff(result)
        assert "lengths differ" in rendered


class TestCli:
    def _run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        trace_cli.register(subparsers)
        args = parser.parse_args(argv)
        return args.func(args)

    def test_show_missing_run_reports_error(self, tmp_path, capsys):
        code = self._run(["trace", "show", "nope", "--bfdiag-dir", str(tmp_path)])
        assert code == 1
        assert "no trace found" in capsys.readouterr().err

    def test_show_renders_table_and_summary(self, tmp_path, capsys):
        ring = _make_ring([{"kv_len_before": i} for i in range(3)])
        write_trace(ring, trace_path_for_run(tmp_path, "run-1"))
        code = self._run(["trace", "show", "run-1", "--bfdiag-dir", str(tmp_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "bfdiag trace summary" in out

    def test_show_json_is_valid_json(self, tmp_path, capsys):
        ring = _make_ring([{"kv_len_before": i} for i in range(2)])
        write_trace(ring, trace_path_for_run(tmp_path, "run-1"))
        code = self._run(["trace", "show", "run-1", "--json", "--bfdiag-dir", str(tmp_path)])
        assert code == 0
        parsed = json.loads(capsys.readouterr().out)
        assert len(parsed["rounds"]) == 2

    def test_diff_between_two_runs(self, tmp_path, capsys):
        write_trace(
            _make_ring([{"kv_len_before": 0}, {"kv_len_before": 15}]),
            trace_path_for_run(tmp_path, "a"),
        )
        write_trace(
            _make_ring(
                [
                    {"kv_len_before": 0},
                    {"kv_len_before": 15, "accepted_n": 1, "reject_position": 1},
                ]
            ),
            trace_path_for_run(tmp_path, "b"),
        )
        code = self._run(["trace", "diff", "a", "b", "--bfdiag-dir", str(tmp_path)])
        assert code == 1
        assert "first divergence at round 1" in capsys.readouterr().out
