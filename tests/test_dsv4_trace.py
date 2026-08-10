"""DSV4-specific CPU-only coverage for the bfdiag flight recorder."""

from __future__ import annotations

import json

from bfdiag.trace import panel
from bfdiag.trace import ring as ring_module
from bfdiag.trace.events import CgMissReason, Path, RoundEvent, reason_label


def test_record_dsv4_prefill_chunk_records_ratio_specific_metadata() -> None:
    ring_module.reset(4, use_cuda=False)

    ring_module.record_dsv4_prefill_chunk(
        0,
        64,
        position=64,
        row_count=32,
        compressor_ratio=4,
        window_entries=96,
        compressed_entries=24,
    )

    rows = ring_module.get_ring().snapshot()
    assert len(rows) == 1
    row = rows[0]
    assert row.event_kind == "prefill_chunk"
    assert row.position == 64
    assert row.row_count == 32
    assert row.compressor_ratio == 4
    assert row.window_entries == 96
    assert row.ratio4_entries == 24
    assert row.ratio128_entries == 0
    assert row.path == "eager"
    assert row.cg_miss_reason == "none"


def test_cg_miss_reason_labels_cover_new_capture_states() -> None:
    assert reason_label(CgMissReason.CAPTURE_FAILED) == "capture_failed"
    assert reason_label(CgMissReason.NOT_CAPTURED) == "not_captured"


def test_record_dsv4_decode_round_keeps_one_row_with_aggregated_counts() -> None:
    ring_module.reset(4, use_cuda=False)

    ring_module.record_dsv4_decode_round(
        1,
        255,
        position=255,
        row_count=1,
        path=Path.CG_MISS,
        cg_miss_reason=CgMissReason.BATCH_SIZE_MISMATCH,
        window_entries=128,
        ratio4_entries=63,
        ratio128_entries=1,
    )

    rows = ring_module.get_ring().snapshot()
    assert len(rows) == 1
    row = rows[0]
    assert row.event_kind == "decode_round"
    assert row.position == 255
    assert row.row_count == 1
    assert row.compressor_ratio == -1
    assert row.window_entries == 128
    assert row.ratio4_entries == 63
    assert row.ratio128_entries == 1
    assert row.path == "cg_miss"
    assert row.cg_miss_reason == "batch_size_mismatch"


def test_dsv4_helper_rows_survive_wraparound_oldest_first() -> None:
    ring_module.reset(2, use_cuda=False)

    ring_module.record_dsv4_prefill_chunk(
        0,
        0,
        position=0,
        row_count=128,
        compressor_ratio=128,
        window_entries=128,
        compressed_entries=1,
    )
    ring_module.record_dsv4_decode_round(
        0,
        128,
        position=128,
        row_count=1,
        path=Path.CG_REPLAY,
        cg_miss_reason=CgMissReason.NONE,
        window_entries=128,
        ratio4_entries=32,
        ratio128_entries=1,
    )
    ring_module.record_dsv4_decode_round(
        0,
        129,
        position=129,
        row_count=1,
        path=Path.EAGER,
        cg_miss_reason=CgMissReason.CG_UNAVAILABLE,
        window_entries=128,
        ratio4_entries=32,
        ratio128_entries=1,
    )

    rows = ring_module.get_ring().snapshot()
    assert [row.round_idx for row in rows] == [1, 2]
    assert [row.position for row in rows] == [128, 129]
    assert [row.event_kind for row in rows] == ["decode_round", "decode_round"]
    assert rows[0].path == "cg_replay"
    assert rows[1].path == "eager"


def test_legacy_json_missing_dsv4_fields_still_parses() -> None:
    legacy = json.dumps(
        {
            "round_idx": 7,
            "slot": 0,
            "kv_len_before": 10,
            "path": "cg_replay",
            "cg_miss_reason": "none",
            "draft_tokens_n": 0,
            "accepted_n": 1,
            "reject_position": -1,
            "bonus_token": -1,
            "mem_allocated": 0,
            "t_main_forward": 0.0,
            "t_draft": 0.0,
            "t_verify": 1.0,
            "t_commit": 0.0,
            "t_round": 1.0,
        }
    )

    row = RoundEvent.from_json(legacy)
    assert row.event_kind == "decode_round"
    assert row.position == -1
    assert row.row_count == 0
    assert row.compressor_ratio == -1
    assert row.window_entries == 0
    assert row.ratio4_entries == 0
    assert row.ratio128_entries == 0


def test_panel_table_and_json_show_dsv4_fields_without_skewing_decode_stats() -> None:
    ring_module.reset(4, use_cuda=False)

    ring_module.record_dsv4_prefill_chunk(
        0,
        0,
        position=0,
        row_count=128,
        compressor_ratio=4,
        window_entries=128,
        compressed_entries=32,
    )
    ring_module.record_dsv4_decode_round(
        0,
        128,
        position=128,
        row_count=1,
        path=Path.CG_REPLAY,
        cg_miss_reason=CgMissReason.NONE,
        window_entries=128,
        ratio4_entries=32,
        ratio128_entries=1,
    )

    rows = ring_module.get_ring().snapshot()
    stats = panel.compute_stats(rows)
    table = panel.render_round_table(rows, limit=0)
    payload = json.loads(panel.render_json(rows, stats))

    assert "kind" in table
    assert "r128" in table
    assert "prefill_chunk" in table
    assert "decode_round" in table
    assert stats.num_rounds == 2
    assert stats.cg_hit_rate == 1.0
    assert stats.path_counts == {"cg_replay": 1}
    assert payload["rounds"][0]["event_kind"] == "prefill_chunk"
    assert payload["rounds"][1]["ratio128_entries"] == 1
