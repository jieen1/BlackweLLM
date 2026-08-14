from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from bfdiag.daemon.provider import DeepseekV4EngineProvider, EngineProvider
from runtime.backends.protocol import BackendSnapshot, PrefixSnapshot, SlotSnapshot


class _FakeTokenizerLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def from_pretrained(self, path: str):
        self.calls.append(path)
        return {"tokenizer_path": path}


class _FakeSlotState:
    def __init__(self, kv_len: int) -> None:
        self.kv_len = kv_len


class _FakeBackend:
    def __init__(self) -> None:
        self.reset_calls: list[int] = []
        self.prefill_calls: list[tuple[int, list[int]]] = []
        self.decode_calls: list[tuple[list[int], list[int], list[int]]] = []
        self.capture_calls = 0
        self.prefill_capture_calls = 0
        self.share_calls = 0
        self.release_calls = 0
        self.lifecycle: list[str] = []
        self.kv_len = {0: 0, 1: 0}
        self.committed = {0: [], 1: []}
        self._decode_outputs = [102, 103]
        self.snapshot_payload = BackendSnapshot(
            slots=(
                SlotSnapshot(slot=0, kv_len=0, is_fresh=True),
                SlotSnapshot(slot=1, kv_len=0, is_fresh=True),
            ),
            prefix=(
                PrefixSnapshot(slot=0, cached_kv_len=0, cached_tokens=0, head=()),
                PrefixSnapshot(slot=1, cached_kv_len=0, cached_tokens=0, head=()),
            ),
            dflash_cg_status=(("decode", "captured"),),
            runtime_stats=(("decode_calls", 2),),
            cg_fallback_reasons=(("graph_unavailable", 1),),
        )

    def capture_decode_cuda_graph(self) -> int:
        self.capture_calls += 1
        self.lifecycle.append("capture_decode")
        return 2

    def capture_prefill_cuda_graph(self) -> bool:
        self.prefill_capture_calls += 1
        self.lifecycle.append("capture_prefill")
        return True

    def _share_rope_freqs(self) -> dict[str, int]:
        self.share_calls += 1
        self.lifecycle.append("share_rope")
        return {"kernel_freqs": 1}

    def _free_eager_oracle_caches(self) -> dict[str, int]:
        self.release_calls += 1
        self.lifecycle.append("release_eager")
        return {"eager_oracle_kv": 1}

    def reset_slot(self, slot: int) -> None:
        self.reset_calls.append(slot)
        self.kv_len[slot] = 0
        self.committed[slot] = []

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        self.prefill_calls.append((slot, list(prompt_ids)))
        self.kv_len[slot] = len(prompt_ids)
        self.committed[slot] = list(prompt_ids)
        return 101

    def slot_state(self, slot: int) -> _FakeSlotState:
        return _FakeSlotState(self.kv_len[slot])

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[object],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int]:
        assert return_logprobs is False
        assert top_logprobs == 0
        self.decode_calls.append((list(slot_ids), list(token_ids), list(kv_lengths)))
        slot = slot_ids[0]
        token = token_ids[0]
        self.kv_len[slot] += 1
        self.committed[slot].append(token)
        return [self._decode_outputs.pop(0)]

    def snapshot(self) -> BackendSnapshot:
        return self.snapshot_payload


class _FakeSamplingParams:
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature


def _install_stub_modules(
    monkeypatch,
    backend: _FakeBackend,
    tokenizer_loader: _FakeTokenizerLoader,
):
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = tokenizer_loader
    dsv4_mod = types.ModuleType("runtime.backends.dsv4")
    calls: list[dict[str, object]] = []

    def _load_backend(
        gguf_path: str,
        *,
        num_slots: int,
        max_seq_len: int,
        max_q_rows: int,
        device: str,
    ) -> _FakeBackend:
        calls.append(
            {
                "gguf_path": gguf_path,
                "num_slots": num_slots,
                "max_seq_len": max_seq_len,
                "max_q_rows": max_q_rows,
                "device": device,
            }
        )
        return backend

    dsv4_mod.load_deepseek_v4_backend = _load_backend
    sampling_mod = types.ModuleType("runtime.sampling")
    sampling_mod.SamplingParams = _FakeSamplingParams

    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "runtime.backends.dsv4", dsv4_mod)
    monkeypatch.setitem(sys.modules, "runtime.sampling", sampling_mod)
    return calls


def test_provider_module_imports_without_torch_side_effects() -> None:
    provider_path = Path(__file__).resolve().parents[1] / "bfdiag" / "daemon" / "provider.py"
    targets = ("torch", "transformers", "runtime.backends.dsv4", "runtime.sampling")
    initially_missing = {name for name in targets if name not in sys.modules}
    spec = importlib.util.spec_from_file_location("provider_probe", provider_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in initially_missing:
        assert name not in sys.modules


def test_deepseek_provider_is_a_structural_engine_provider() -> None:
    provider = DeepseekV4EngineProvider()
    assert isinstance(provider, EngineProvider)
    assert provider.describe()["load_config"]["prefill_rows"] == 64


def test_deepseek_provider_load_describe_namespace_and_snapshot_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = _FakeBackend()
    tokenizer_loader = _FakeTokenizerLoader()
    load_calls = _install_stub_modules(monkeypatch, backend, tokenizer_loader)
    stages: list[str] = []
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    provider = DeepseekV4EngineProvider(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_dir),
        num_slots=2,
        max_model_len=4096,
        prefill_rows=17,
        enable_cudagraph=True,
    )

    provider.load(on_stage=stages.append)

    assert load_calls == [
        {
            "gguf_path": str(model_path),
            "num_slots": 2,
            "max_seq_len": 4096,
            "max_q_rows": 17,
            "device": "cuda",
        }
    ]
    assert backend.capture_calls == 1
    assert backend.prefill_capture_calls == 1
    assert backend.share_calls == 1
    assert backend.release_calls == 1
    assert backend.lifecycle == [
        "share_rope",
        "capture_decode",
        "release_eager",
        "capture_prefill",
    ]
    assert tokenizer_loader.calls == [str(tokenizer_dir)]
    assert stages == [
        "after_tokenizer",
        "after_target_backend",
        "after_rope_sharing",
        "after_decode_cuda_graphs",
        "after_eager_oracle_release",
        "after_prefill_cuda_graph",
        "after_reset",
    ]
    assert backend.reset_calls == [0, 1]

    desc = provider.describe()
    assert desc["kind"] == "deepseek_v4"
    assert desc["cg_status"] == {"decode": "captured"}
    assert desc["runtime_stats"] == {"decode_calls": 2}
    assert desc["cg_fallback_reasons"] == {"graph_unavailable": 1}
    assert desc["model_revision"].startswith("stat:4:")
    assert backend.bfdiag_model_identity == {
        "path": str(model_path.resolve()),
        "revision": desc["model_revision"],
    }
    assert desc["load_config"] == {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_dir),
        "num_slots": 2,
        "max_model_len": 4096,
        "prefill_rows": 17,
        "enable_cudagraph": True,
    }
    assert provider.namespace() == {
        "backend": backend,
        "engine": backend,
        "tokenizer": {"tokenizer_path": str(tokenizer_dir)},
        "provider": provider,
    }
    assert provider.is_healthy() is True


def test_deepseek_provider_describe_tolerates_snapshot_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = _FakeBackend()
    backend.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    tokenizer_loader = _FakeTokenizerLoader()
    _install_stub_modules(monkeypatch, backend, tokenizer_loader)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    provider = DeepseekV4EngineProvider(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_dir),
    )
    provider.load()

    desc = provider.describe()
    assert desc["cg_status"] == {}
    assert desc["runtime_stats"] == {}
    assert desc["cg_fallback_reasons"] == {}


def test_deepseek_provider_missing_paths_fail_before_backend_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = _FakeBackend()
    tokenizer_loader = _FakeTokenizerLoader()
    load_calls = _install_stub_modules(monkeypatch, backend, tokenizer_loader)
    provider = DeepseekV4EngineProvider(
        model_path=str(tmp_path / "missing.gguf"),
        tokenizer_path=str(tmp_path / "missing-tokenizer"),
    )

    try:
        provider.load()
    except FileNotFoundError as exc:
        assert "GGUF not found" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected FileNotFoundError")

    assert load_calls == []
    assert tokenizer_loader.calls == []


def test_generate_uses_prefill_anchor_then_decode_inputs_and_resets_all_slots(monkeypatch) -> None:
    backend = _FakeBackend()
    tokenizer_loader = _FakeTokenizerLoader()
    _install_stub_modules(monkeypatch, backend, tokenizer_loader)
    provider = DeepseekV4EngineProvider(num_slots=2)
    provider._backend = backend
    provider._tokenizer = object()

    out = provider.generate([11, 12], 3)

    assert out == [101, 102, 103]
    assert backend.prefill_calls == [(1, [11, 12])]
    assert backend.decode_calls == [
        ([1], [101], [2]),
        ([1], [102], [3]),
    ]
    assert backend.reset_calls == [0, 1, 0, 1]


def test_generate_rejects_sampling_and_zero_tokens(monkeypatch) -> None:
    backend = _FakeBackend()
    tokenizer_loader = _FakeTokenizerLoader()
    _install_stub_modules(monkeypatch, backend, tokenizer_loader)
    provider = DeepseekV4EngineProvider()
    provider._backend = backend
    provider._tokenizer = object()

    assert provider.generate([1], 0) == []
    try:
        provider.generate([1], 1, temperature=0.1)
    except NotImplementedError as exc:
        assert "greedy-only" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected NotImplementedError")


def test_unload_and_memory_snapshot_use_torch_lazily(monkeypatch) -> None:
    backend = _FakeBackend()
    tokenizer_loader = _FakeTokenizerLoader()
    _install_stub_modules(monkeypatch, backend, tokenizer_loader)
    empty_cache_calls: list[str] = []
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        empty_cache=lambda: empty_cache_calls.append("empty_cache"),
        memory_stats=lambda: {
            "allocated_bytes.all.current": 10,
            "reserved_bytes.all.current": 40,
            "num_alloc_retries": 2,
        },
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    provider = DeepseekV4EngineProvider()
    provider._backend = backend
    provider._tokenizer = object()

    assert provider.memory_snapshot() == {
        "kind": "deepseek_v4",
        "allocated_bytes": 10,
        "reserved_bytes": 40,
        "num_alloc_retries": 2,
        "fragmentation_ratio": 0.75,
    }

    provider.unload()

    assert provider._backend is None
    assert provider._tokenizer is None
    assert empty_cache_calls == ["empty_cache"]
