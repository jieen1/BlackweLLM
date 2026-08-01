from __future__ import annotations

import json

import pytest

from bfdiag.cold_capacity import (
    ColdCapacitySpec,
    _driver_memory_from_fields,
    _prefill_slot_limit_from_env,
    _route_tile_trace_from_host,
    _static_headroom_error,
    blocks_for_tokens,
    persist_record,
    run,
    spec_from_env,
)


def test_blocks_for_tokens_rounds_up() -> None:
    assert blocks_for_tokens(1, 64) == 1
    assert blocks_for_tokens(64, 64) == 1
    assert blocks_for_tokens(65, 64) == 2


def test_driver_memory_record_preserves_mib_and_gib() -> None:
    assert _driver_memory_from_fields("97300,97887,587") == {
        "driver_used_mib": 97300,
        "driver_total_mib": 97887,
        "driver_free_mib": 587,
        "driver_used_gib": 95.02,
        "driver_total_gib": 95.593,
        "driver_free_gib": 0.573,
    }


def test_static_headroom_gate_uses_driver_mib() -> None:
    assert _static_headroom_error({"driver_free_mib": 2048}, 2048) is None
    assert _static_headroom_error({"driver_free_mib": None}, 2048) is None
    assert _static_headroom_error({"driver_free_mib": 2047}, 2048) == (
        "StaticHeadroomError: after_load driver_free_mib=2047 is below the required 2048 MiB"
    )
    with pytest.raises(ValueError, match="must not be negative"):
        _static_headroom_error({"driver_free_mib": 0}, -1)


def test_stage_checkpoint_stops_load_before_unsafe_prefill(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import bfdiag.cold_capacity as cold_capacity
    import bfdiag.daemon.provider as provider_module

    class LowHeadroomProvider:
        def __init__(self, **_kwargs) -> None:
            pass

        def load(self, *, on_stage=None) -> None:
            assert on_stage is not None
            on_stage("after_dflash_eager")

        def unload(self) -> None:
            pass

    monkeypatch.setattr(provider_module, "LagunaEngineProvider", LowHeadroomProvider)
    monkeypatch.setattr(
        cold_capacity,
        "_memory_snapshot",
        lambda _torch: {"driver_free_mib": 2047},
    )
    monkeypatch.setattr(cold_capacity, "_git_sha", lambda _path: "test-sha")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    record = run(ColdCapacitySpec("test", 1, 64, 1, 65, 2))

    assert not record["ok"]
    assert record["error"] == (
        "StaticHeadroomError: after_load driver_free_mib=2047 is below the required 2048 MiB"
    )
    assert "after_dflash_eager" in record["memory"]


def test_load_only_returns_after_safe_load(monkeypatch) -> None:
    import os

    torch = pytest.importorskip("torch")

    import bfdiag.cold_capacity as cold_capacity
    import bfdiag.daemon.provider as provider_module

    class LoadedProvider:
        def __init__(self, **_kwargs) -> None:
            pass

        def load(self, *, on_stage=None) -> None:
            assert on_stage is not None
            on_stage("after_dflash_cuda_graphs")

        def unload(self) -> None:
            pass

    monkeypatch.setattr(provider_module, "LagunaEngineProvider", LoadedProvider)
    monkeypatch.setattr(
        cold_capacity,
        "_memory_snapshot",
        lambda _torch: {"driver_free_mib": 2048},
    )
    monkeypatch.setattr(cold_capacity, "_git_sha", lambda _path: "test-sha")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setitem(os.environ, "QSR_COLD_LOAD_ONLY", "1")

    record = run(ColdCapacitySpec("test", 1, 64, 1, 65, 2))

    assert record["ok"]
    assert record["load_only"]
    assert "after_load" in record["memory"]


def test_spec_requires_capacity_for_prompt_and_suffix() -> None:
    with pytest.raises(ValueError, match="required 5 blocks"):
        spec_from_env(
            {
                "QSR_COLD_NUM_SLOTS": "4",
                "QSR_COLD_PROMPT_TOKENS": "257",
                "QSR_COLD_OUTPUT_TOKENS": "1",
                "QSR_COLD_MAX_MODEL_LEN": "258",
                "QSR_COLD_BLOCKS_PER_SLOT": "4",
            }
        )


def test_spec_reads_sku_configuration() -> None:
    spec = spec_from_env(
        {
            "QSR_COLD_SKU": "4x240k",
            "QSR_COLD_NUM_SLOTS": "4",
            "QSR_COLD_PROMPT_TOKENS": "240000",
            "QSR_COLD_OUTPUT_TOKENS": "1000",
            "QSR_COLD_MAX_MODEL_LEN": "241024",
            "QSR_COLD_BLOCKS_PER_SLOT": "3766",
        }
    )
    assert spec.sku == "4x240k"
    assert spec.blocks_per_slot == 3766


def test_prefill_slot_limit_defaults_to_all_slots_and_rejects_invalid_values() -> None:
    assert _prefill_slot_limit_from_env(4, {}) == 4
    assert _prefill_slot_limit_from_env(4, {"QSR_COLD_PREFILL_SLOT_LIMIT": "1"}) == 1
    with pytest.raises(ValueError, match="between 1 and QSR_COLD_NUM_SLOTS"):
        _prefill_slot_limit_from_env(4, {"QSR_COLD_PREFILL_SLOT_LIMIT": "0"})


def test_prefill_slot_limit_checkpoints_each_completed_slot(monkeypatch) -> None:
    import os

    torch = pytest.importorskip("torch")

    import bfdiag.cold_capacity as cold_capacity
    import bfdiag.daemon.provider as provider_module

    class Tokenizer:
        def encode(self, _prompt: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [1]

    class Backend:
        def prefill(self, slot: int, prompt_ids: list[int]) -> int:
            assert slot == 0
            assert prompt_ids == [1] * 64
            return 7

        def decode_batch(self, slot_ids: list[int], tokens: list[int]) -> list[int]:
            assert slot_ids == [0]
            assert tokens == [7]
            return [8]

    class LoadedProvider:
        def __init__(self, **_kwargs) -> None:
            self._tokenizer = Tokenizer()
            self._backend = Backend()

        def load(self, *, on_stage=None) -> None:
            assert on_stage is not None
            on_stage("after_dflash_cuda_graphs")

        def unload(self) -> None:
            pass

    monkeypatch.setattr(provider_module, "LagunaEngineProvider", LoadedProvider)
    monkeypatch.setattr(
        cold_capacity,
        "_memory_snapshot",
        lambda _torch: {"driver_free_mib": 2048},
    )
    monkeypatch.setattr(cold_capacity, "_git_sha", lambda _path: "test-sha")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setitem(os.environ, "QSR_COLD_PREFILL_SLOT_LIMIT", "1")

    checkpoints = []
    record = run(
        ColdCapacitySpec("test", 4, 64, 1, 65, 2),
        on_checkpoint=lambda item: checkpoints.append(json.loads(json.dumps(item))),
    )

    assert record["ok"]
    assert record["prefill_slot_limit"] == 1
    assert record["prefill_seconds_by_slot"]
    assert "after_prefill_slot_1" in record["memory"]
    assert "after_prefill_slot_2" not in record["memory"]
    assert any("after_prefill_slot_1" in item["memory"] for item in checkpoints)


def test_persist_record_writes_a_self_describing_artifact(tmp_path) -> None:
    record = {"ok": False, "spec": {"sku": "4x240k / failure"}}

    path = persist_record(record, tmp_path)

    assert path.parent == tmp_path
    assert path.name.endswith("-4x240k---failure.json")
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert record["artifact_path"] == str(path)


def test_route_tile_trace_preserves_only_valid_physical_rows() -> None:
    pytest.importorskip("sparkinfer")
    trace = _route_tile_trace_from_host(
        token_map=[0, 3, 2, 1, 0, 0],
        expert_row_counts=[2, 2],
        expert_tile_base=[0, 1, 2],
        physical_tiles_capacity=3,
        num_topk=2,
    )

    assert trace == {
        "routed_rows": 4,
        "tile_m": 2,
        "active_tiles": 2,
        "dependency_edges": 2,
        "cyclic_components": 1,
        "largest_cyclic_component_tiles": 2,
        "largest_cyclic_component_route_rows": 4,
    }


def test_failed_load_preserves_spec_and_stage_snapshots(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import bfdiag.cold_capacity as cold_capacity
    import bfdiag.daemon.provider as provider_module

    unloaded = False

    class FailingProvider:
        def __init__(self, **_kwargs) -> None:
            pass

        def load(self, *, on_stage=None) -> None:
            assert on_stage is not None
            on_stage("after_target_backend")
            raise RuntimeError("synthetic target/dflash boundary failure")

        def unload(self) -> None:
            nonlocal unloaded
            unloaded = True

    snapshots = iter(({"stage": "before"}, {"stage": "target"}, {"stage": "failure"}))
    monkeypatch.setattr(provider_module, "LagunaEngineProvider", FailingProvider)
    monkeypatch.setattr(cold_capacity, "_memory_snapshot", lambda _torch: next(snapshots))
    monkeypatch.setattr(cold_capacity, "_git_sha", lambda _path: "test-sha")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    spec = ColdCapacitySpec("test", 1, 64, 1, 65, 2)
    checkpoints = []
    record = run(
        spec,
        on_checkpoint=lambda item: checkpoints.append(json.loads(json.dumps(item))),
    )

    assert not record["ok"]
    assert record["spec"]["sku"] == "test"
    assert record["memory"] == {
        "before_load": {"stage": "before"},
        "after_target_backend": {"stage": "target"},
        "at_failure": {"stage": "failure"},
    }
    assert "synthetic target/dflash boundary failure" in record["error"]
    assert checkpoints[-1]["memory"] == record["memory"]
    assert unloaded
