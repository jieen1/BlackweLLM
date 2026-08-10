from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from bfdiag.workloads import (
    DSV4_WARM_BASELINE_CONTRACT,
    DSV4_WARM_BASELINE_VERSION,
    _percentile,
    _run_dsv4_warm_baseline_sample,
    dsv4_warm_baseline_workloads,
    run_dsv4_warm_baseline,
    run_dsv4_warm_baseline_case,
)


class _FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        seed = (sum(ord(ch) for ch in text) % 997) + 17
        count = max(512, len(text))
        return [seed + (index % 37) for index in range(count)]


class _FakeBackend:
    def __init__(self, *, num_slots: int = 2) -> None:
        self.num_slots = num_slots
        self.max_seq_len = 4096
        self.max_q_rows = 32
        self.bfdiag_model_identity = {
            "path": "/models/dsv4.gguf",
            "revision": "stat:123:456",
        }
        self._kv_len = [0 for _ in range(num_slots)]
        self.prefill_calls: list[tuple[int, int]] = []
        self.decode_calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
        self.reset_calls: list[int] = []

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        self.prefill_calls.append((slot, len(prompt_ids)))
        self._kv_len[slot] = len(prompt_ids)
        return 100 + slot

    def decode_batch_sampled(
        self,
        slot_ids,
        token_ids,
        kv_lengths,
        params_list,
        *,
        return_logprobs=False,
        top_logprobs=0,
    ):
        assert return_logprobs is False
        assert top_logprobs == 0
        self.decode_calls.append((tuple(slot_ids), tuple(token_ids), tuple(kv_lengths)))
        assert len(slot_ids) == len(token_ids) == len(kv_lengths) == len(params_list)
        for slot, kv_len in zip(slot_ids, kv_lengths):
            assert self._kv_len[slot] == kv_len
            self._kv_len[slot] += 1
        return [token + 1 for token in token_ids]

    def slot_state(self, slot: int):
        return SimpleNamespace(kv_len=self._kv_len[slot])

    def reset_slot(self, slot: int) -> None:
        self.reset_calls.append(slot)
        self._kv_len[slot] = 0

    def snapshot(self):
        return SimpleNamespace(dflash_cg_status=(("decode", "captured"),))


class _FakeRunHandle:
    def __init__(self) -> None:
        self.run_id = "run-123"
        self.metrics: dict[str, float] = {}
        self.record = SimpleNamespace(fingerprint=SimpleNamespace(extra={}))

    def metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)


def test_dsv4_warm_baseline_workloads_are_fixed_length_and_distinct():
    workloads = dsv4_warm_baseline_workloads(_FakeTokenizer())

    assert [item["name"] for item in workloads] == [
        "chat-short-96",
        "chat-cross-128",
        "code-instruction-cross-256",
    ]
    assert [item["prompt_len"] for item in workloads] == [96, 129, 257]
    assert len({item["prompt_hash"] for item in workloads}) == len(workloads)


def test_percentile_interpolates_and_validates():
    assert _percentile([1.0, 3.0, 5.0], 0.50) == 3.0
    assert _percentile([10.0, 20.0], 0.95) == pytest.approx(19.5)
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        _percentile([1.0], 1.1)


def test_dsv4_warm_sample_measures_prefill_and_decode_and_resets_slots():
    backend = _FakeBackend(num_slots=2)
    ticks = iter([0.0, 0.3, 1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
    syncs: list[str] = []

    result = _run_dsv4_warm_baseline_sample(
        backend,
        [1, 2, 3, 4],
        batch=2,
        generated_tokens=4,
        synchronize=lambda: syncs.append("sync"),
        reset_peak_memory=lambda: None,
        peak_memory_allocated=lambda: 1234,
        clock=lambda: next(ticks),
    )

    assert result["prefill_tok_s"] == pytest.approx((4 * 2) / 0.3)
    assert result["tok_s"] == pytest.approx(3 / 0.6)
    assert result["aggregate_tok_s"] == pytest.approx((3 * 2) / 0.6)
    assert result["itl_p50_ms"] == pytest.approx(200.0)
    assert result["itl_p95_ms"] == pytest.approx(290.0)
    assert len(result["slot_outputs"]) == 2
    assert result["peak_memory_allocated_bytes"] == 1234
    assert all(len(tokens) == 4 for tokens in result["slot_outputs"])
    assert backend.reset_calls == [0, 1]
    assert len(syncs) == 8


def test_dsv4_warm_baseline_case_records_contract_metrics_and_hashes(monkeypatch):
    backend = _FakeBackend(num_slots=2)
    captured: dict[str, object] = {}
    handle = _FakeRunHandle()

    @contextlib.contextmanager
    def fake_run_record(**kwargs):
        captured.update(kwargs)
        yield handle

    sample_calls = iter(
        [
            {
                "batch": 1,
                "prompt_len": 96,
                "generated_tokens": 4,
                "prefill_s": 0.4,
                "prefill_tok_s": 240.0,
                "decode_s": 0.3,
                "decode_steps": 3,
                "itl_p50_ms": 100.0,
                "itl_p95_ms": 140.0,
                "tok_s": 10.0,
                "aggregate_tok_s": 10.0,
                "output_hash": "warmup-hash",
                "slot_output_hashes": ["slot-a"],
                "slot_outputs": [[1, 2, 3, 4]],
                "decode_itl_ms": [100.0, 120.0, 140.0],
                "peak_memory_allocated_bytes": 2048,
            },
            {
                "batch": 1,
                "prompt_len": 96,
                "generated_tokens": 4,
                "prefill_s": 0.4,
                "prefill_tok_s": 240.0,
                "decode_s": 0.3,
                "decode_steps": 3,
                "itl_p50_ms": 100.0,
                "itl_p95_ms": 140.0,
                "tok_s": 10.0,
                "aggregate_tok_s": 10.0,
                "output_hash": "warmup-hash",
                "slot_output_hashes": ["slot-a"],
                "slot_outputs": [[1, 2, 3, 4]],
                "decode_itl_ms": [100.0, 120.0, 140.0],
                "peak_memory_allocated_bytes": 2048,
            },
        ]
    )

    monkeypatch.setattr("bfdiag.workloads.run_record", fake_run_record)
    monkeypatch.setattr(
        "bfdiag.workloads._run_dsv4_warm_baseline_sample",
        lambda *args, **kwargs: next(sample_calls),
    )

    result = run_dsv4_warm_baseline_case(
        backend,
        _FakeTokenizer(),
        workload_name="chat-short-96",
        batch=1,
        generated_tokens=4,
    )

    assert captured["workload"] == {
        "contract": DSV4_WARM_BASELINE_CONTRACT,
        "contract_version": DSV4_WARM_BASELINE_VERSION,
        "workload_name": "chat-short-96",
        "prompt_hash": result["prompt_hash"],
        "prompt_len": 96,
        "generated_tokens": 4,
        "batch": 1,
        "max_model_len": 4096,
        "max_q_rows": 32,
        "capacity": 2,
        "cuda_graph_status": "captured",
        "warm_only": True,
    }
    assert captured["model"] == {
        "path": "/models/dsv4.gguf",
        "revision": "stat:123:456",
    }
    assert captured["extra"] == {
        "workload_extra": {
            "kind": "chat",
            "graph_capture_status": "captured",
            "warm_only": True,
        }
    }
    assert handle.metrics["prefill_tok_s"] == 240.0
    assert handle.metrics["decode_tok_s"] == 10.0
    assert handle.record.fingerprint.extra["output_hash"] == "warmup-hash"
    assert handle.metrics["peak_memory_allocated_bytes"] == 2048
    assert result["run_id"] == "run-123"


def test_dsv4_warm_baseline_refuses_missing_model_identity():
    backend = _FakeBackend(num_slots=1)
    del backend.bfdiag_model_identity

    with pytest.raises(RuntimeError, match="bfdiag_model_identity"):
        run_dsv4_warm_baseline_case(
            backend,
            _FakeTokenizer(),
            workload_name="chat-short-96",
            generated_tokens=4,
        )


def test_dsv4_warm_baseline_suite_runs_native_batch_buckets(monkeypatch):
    backend = _FakeBackend(num_slots=4)
    calls: list[tuple[str, int, int]] = []

    def fake_case(_backend, _tokenizer, *, workload_name, batch, generated_tokens):
        calls.append((workload_name, batch, generated_tokens))
        return {
            "workload_name": workload_name,
            "batch": batch,
            "generated_tokens": generated_tokens,
        }

    monkeypatch.setattr("bfdiag.workloads.run_dsv4_warm_baseline_case", fake_case)

    result = run_dsv4_warm_baseline(backend, _FakeTokenizer(), generated_tokens=8)

    assert result["contract"] == DSV4_WARM_BASELINE_CONTRACT
    assert calls == [
        ("chat-short-96", 1, 8),
        ("chat-short-96", 2, 8),
        ("chat-short-96", 4, 8),
        ("chat-cross-128", 1, 8),
        ("chat-cross-128", 2, 8),
        ("chat-cross-128", 4, 8),
        ("code-instruction-cross-256", 1, 8),
        ("code-instruction-cross-256", 2, 8),
        ("code-instruction-cross-256", 4, 8),
    ]


def test_dsv4_warm_baseline_case_rejects_unknown_workload():
    with pytest.raises(ValueError, match="unknown workload_name"):
        run_dsv4_warm_baseline_case(
            _FakeBackend(),
            _FakeTokenizer(),
            workload_name="missing",
        )
