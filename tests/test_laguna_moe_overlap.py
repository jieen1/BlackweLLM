from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from runtime.backends.laguna_moe_overlap import (
    MoESharedOverlapSession,
    active_moe_shared_overlap,
    moe_shared_overlap_session,
)


class _FakeStream:
    def __init__(self, name: str, calls: list[tuple[str, str]]) -> None:
        self.name = name
        self.calls = calls

    def wait_stream(self, other: _FakeStream) -> None:
        self.calls.append((self.name, f"wait:{other.name}"))


class _FakeCuda:
    def __init__(self, current: _FakeStream, calls: list[tuple[str, str]]) -> None:
        self.current = current
        self.calls = calls

    def current_stream(self, *, device: object) -> _FakeStream:
        self.calls.append((self.current.name, f"current:{device}"))
        return self.current

    @contextmanager
    def stream(self, stream: _FakeStream):
        previous = self.current
        self.current = stream
        self.calls.append((stream.name, "enter"))
        try:
            yield
        finally:
            self.calls.append((stream.name, "exit"))
            self.current = previous


def test_session_forks_and_joins_on_its_own_stream():
    calls: list[tuple[str, str]] = []
    main = _FakeStream("main", calls)
    auxiliary = _FakeStream("auxiliary", calls)
    cuda = _FakeCuda(main, calls)
    session = MoESharedOverlapSession("cuda:0", cuda_api=cuda, stream=auxiliary)
    hidden = SimpleNamespace(device="cuda:0")

    output = session.launch(lambda value: ("shared", value), hidden)
    session.join("cuda:0")

    assert output == ("shared", hidden)
    assert calls == [
        ("main", "current:cuda:0"),
        ("auxiliary", "wait:main"),
        ("auxiliary", "enter"),
        ("auxiliary", "exit"),
        ("main", "current:cuda:0"),
        ("main", "wait:auxiliary"),
    ]


def test_session_context_is_nested_and_thread_local_contract_is_restored():
    outer = object()
    inner = object()

    assert active_moe_shared_overlap() is None
    with moe_shared_overlap_session(outer):
        assert active_moe_shared_overlap() is outer
        with moe_shared_overlap_session(inner):
            assert active_moe_shared_overlap() is inner
        assert active_moe_shared_overlap() is outer
    assert active_moe_shared_overlap() is None
