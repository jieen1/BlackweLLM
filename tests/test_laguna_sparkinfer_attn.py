"""CPU-only contract tests for SparkInfer KV descale normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.laguna_cuda_graph import _SparkinferCGExtendImpl  # noqa: E402
from runtime.backends.laguna_sparkinfer_attn import (  # noqa: E402
    SparkinferPrefillWorkspace,
    _paged_descale,
)


def test_paged_descale_normalizes_rank_zero_dflash_default_scale():
    descale = _paged_descale(torch.tensor(0.5), batch_size=1, num_kv_heads=8)

    assert descale.shape == (1,)
    assert torch.equal(descale, torch.tensor([0.5]))


def test_paged_descale_expands_per_head_scale_for_each_request():
    per_head = torch.arange(1, 9, dtype=torch.float32)
    descale = _paged_descale(per_head, batch_size=2, num_kv_heads=8)

    assert descale.shape == (2, 8)
    assert torch.equal(descale[0], per_head)
    assert torch.equal(descale[1], per_head)


def test_paged_descale_rejects_ambiguous_scale_shape():
    with pytest.raises(ValueError, match="KV descale"):
        _paged_descale(torch.ones(3), batch_size=2, num_kv_heads=8)


def test_cuda_graph_attention_rebinds_when_model_buffers_move(monkeypatch):
    # monkeypatch.setattr below resolves its string targets by importing
    # the real sparkinfer submodules -- unlike the plain
    # `from runtime.backends... import ...` above (which only reaches
    # sparkinfer lazily/conditionally), this needs the actual package.
    pytest.importorskip("b12x")
    bindings = []

    def build_binding(**kwargs):
        binding = SimpleNamespace(**kwargs)
        bindings.append(binding)
        return binding

    monkeypatch.setattr(
        "b12x.attention.paged._scratch.build_paged_attention_binding",
        build_binding,
    )
    monkeypatch.setattr(
        "b12x.attention.paged._forward.paged_attention_forward",
        lambda *, binding: None,
    )

    impl = _SparkinferCGExtendImpl(workspace=object(), num_tokens=2)
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    kv_cache = torch.zeros((2, 2, 64, 1, 4), dtype=torch.uint8)
    key = value = torch.empty((2, 1, 4), dtype=torch.bfloat16)
    query = torch.empty((2, 1, 4), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    metadata = SimpleNamespace(num_actual_tokens=2)

    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    assert len(bindings) == 1

    moved_query = torch.empty_like(query)
    moved_output = torch.empty_like(output)
    impl.forward(
        layer,
        moved_query,
        key,
        value,
        kv_cache,
        metadata,
        moved_output,
    )

    assert len(bindings) == 2
    assert bindings[-1].q.data_ptr() == moved_query.data_ptr()
    assert bindings[-1].output.data_ptr() == moved_output.data_ptr()


def test_prefill_workspace_reuses_plan_only_within_one_metadata_object(monkeypatch):
    """Layer-group sharing must not reuse a plan after the next forward."""
    pytest.importorskip("b12x")
    calls: list[str] = []

    class FakeWorkspace:
        @staticmethod
        def eager_extend_work_items_capacity(**kwargs):
            return 1

        @classmethod
        def for_fixed_capacity(cls, **kwargs):
            calls.append("workspace")
            return cls()

        def _ensure_capacity(self, plan):
            calls.append("capacity")

        def _copy_runtime_metadata(self, *args):
            calls.append("runtime")

        def _copy_plan_metadata(self, plan):
            calls.append("plan_metadata")

    monkeypatch.setattr(
        "b12x.attention.paged.workspace.PagedAttentionWorkspace",
        FakeWorkspace,
    )
    monkeypatch.setattr(
        "b12x.attention.paged.planner.create_paged_plan",
        lambda *args, **kwargs: calls.append("plan") or object(),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._scratch.build_paged_attention_binding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._forward.paged_attention_forward",
        lambda *, binding: calls.append("forward"),
    )

    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=64
    )
    q = torch.empty(2, 1, 4)
    cache = torch.empty(3, 2, 1, 4)
    output = torch.empty_like(q)
    metadata = object()

    def run(cache_key):
        workspace.forward(
            q=q,
            k_cache=cache,
            v_cache=cache,
            output=output,
            page_table=torch.empty(1, 2, dtype=torch.int32),
            cache_seqlens=torch.empty(1, dtype=torch.int32),
            cu_seqlens_q=torch.empty(2, dtype=torch.int32),
            plan_cache_key=cache_key,
        )

    run(metadata)
    run(metadata)
    run(object())

    assert calls.count("workspace") == 1
    assert calls.count("plan") == 2
    assert calls.count("runtime") == 2
    assert calls.count("plan_metadata") == 2
    assert calls.count("forward") == 3


def test_prefill_workspace_never_rebuilds_across_varying_real_shapes(monkeypatch):
    """Regression test for the prefill-shape-buckets bug.

    Root cause: sparkinfer's paged_attention_forward JIT-compiles its CuTe
    launch keyed on (among other things) page_table's literal width, which
    is a function of kv_len+qo_len. The old SparkinferPrefillWorkspace
    rebuilt a PagedAttentionWorkspace via for_tensors() -- sized to the
    exact literal q/page_table shape -- every time that shape changed, so
    every previously-unseen (kv_len, qo_len) paid sparkinfer's ~30s CuTe
    recompile. Real multi-turn traffic almost never repeats a shape.

    Before the fix (SparkinferPrefillWorkspace._key() included q.shape and
    the workspace was built via for_tensors()), this test fails: it counts
    a fresh "workspace" build for every distinct (qo_len, page_table_width)
    pair below. After the fix (fixed-capacity workspace, built once per
    (mode, window_left)), it must build exactly once no matter how many
    distinct real shapes are seen.
    """
    pytest.importorskip("b12x")
    calls: list[str] = []

    class FakeWorkspace:
        @staticmethod
        def eager_extend_work_items_capacity(**kwargs):
            return 1

        @classmethod
        def for_fixed_capacity(cls, **kwargs):
            calls.append("workspace")
            return cls()

        def _ensure_capacity(self, plan):
            pass

        def _copy_runtime_metadata(self, *args):
            pass

        def _copy_plan_metadata(self, plan):
            pass

    monkeypatch.setattr(
        "b12x.attention.paged.workspace.PagedAttentionWorkspace",
        FakeWorkspace,
    )
    monkeypatch.setattr(
        "b12x.attention.paged.planner.create_paged_plan",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._scratch.build_paged_attention_binding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._forward.paged_attention_forward",
        lambda *, binding: None,
    )

    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=1088
    )
    cache = torch.empty(3, 64, 8, 128)

    # Distinct real (qo_len, kv_len) pairs -- mirrors a growing multi-turn
    # conversation, where neither the query length nor the accumulated KV
    # length ever repeats.
    for qo_len, kv_len in [(37, 0), (512, 0), (2151, 0), (2839, 4096), (91, 8192)]:
        new_kv_len = kv_len + qo_len
        max_blocks = (new_kv_len + 63) // 64
        q = torch.empty(qo_len, 48, 128)
        output = torch.empty_like(q)
        workspace.forward(
            q=q,
            k_cache=cache,
            v_cache=cache,
            output=output,
            page_table=torch.empty(1, max_blocks, dtype=torch.int32),
            cache_seqlens=torch.empty(1, dtype=torch.int32),
            cu_seqlens_q=torch.empty(2, dtype=torch.int32),
            plan_cache_key=object(),
        )

    assert calls.count("workspace") == 1, (
        "SparkinferPrefillWorkspace rebuilt its PagedAttentionWorkspace for a "
        "new real shape -- this reintroduces the per-shape JIT recompile bug "
        "(see notes/2026-08-01-prefill-shape-buckets-root-cause.md)"
    )


def test_prefill_workspace_verify_mode_without_declared_capacity_raises_loud_error():
    """Regression test for the C-1 capacity bug (notes/2026-08-01-c1-c2-gpu-
    investigation.md): before the fix, mode="verify" silently reused the
    extend-shaped eager_extend_work_items_capacity() estimate, which
    under-provisions verify's real work-item need and only surfaced as
    sparkinfer's own opaque `ValueError: fixed-capacity paged workspace
    exceeded` deep inside _ensure_capacity -- confirmed on real GPU via a
    direct call to DFlashEngine._forward_verify_with_aux (run record
    940b708aa0f8). After the fix, calling mode="verify" before
    declare_verify_capacity() raises a clear, actionable RuntimeError from
    this class itself, without ever reaching sparkinfer.
    """
    pytest.importorskip("b12x")

    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=4096
    )
    q = torch.empty(16, 48, 128)
    cache = torch.empty(4096, 64, 8, 128)
    output = torch.empty_like(q)

    with pytest.raises(RuntimeError, match="declare_verify_capacity"):
        workspace.forward(
            q=q,
            k_cache=cache,
            v_cache=cache,
            output=output,
            page_table=torch.empty(1, 32, dtype=torch.int32),
            cache_seqlens=torch.empty(1, dtype=torch.int32),
            cu_seqlens_q=torch.empty(2, dtype=torch.int32),
            mode="verify",
            window_left=-1,
            plan_cache_key=object(),
        )


def test_declare_verify_capacity_rejects_degenerate_query_len():
    pytest.importorskip("b12x")
    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=4096
    )
    with pytest.raises(ValueError, match="max_query_len"):
        workspace.declare_verify_capacity(1)
    with pytest.raises(ValueError, match="max_query_len"):
        workspace.declare_verify_capacity(0)


def test_declare_verify_capacity_is_monotonic():
    pytest.importorskip("b12x")
    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=4096
    )
    workspace.declare_verify_capacity(16)
    workspace.declare_verify_capacity(8)  # must not shrink an already-declared bound
    assert workspace._max_verify_query_len == 16

    workspace2 = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=4096
    )
    workspace2.declare_verify_capacity(8)
    workspace2.declare_verify_capacity(16)
    assert workspace2._max_verify_query_len == 16


def test_prefill_workspace_verify_mode_sizes_capacity_from_the_real_eager_planner(monkeypatch):
    """Once capacity is declared, mode="verify" must size its fixed capacity
    from sparkinfer's own REAL eager planner (create_paged_plan with
    enable_cuda_graph=False -- the exact function every real verify call
    uses), run once against a synthetic worst-case call -- never the
    extend-shaped eager_extend_work_items_capacity heuristic that
    under-provisioned it, and never the OTHER (CUDA-Graph-mode) capacity
    planner either (planner.plan_verify_graph_capacity) -- that one measured
    wrong on real GPU, see _work_item_capacity's docstring and
    notes/2026-08-01-c1-c2-gpu-investigation.md's follow-up section.
    """
    pytest.importorskip("b12x")
    calls: list[str] = []

    class FakeWorkspace:
        @staticmethod
        def eager_extend_work_items_capacity(**kwargs):
            calls.append("eager_extend_work_items_capacity")
            return 1  # would be silently wrong for verify if ever reached

        @classmethod
        def for_fixed_capacity(cls, **kwargs):
            calls.append("workspace")
            assert kwargs["max_work_items"] == 4321
            assert kwargs["max_partial_rows"] == 99
            return cls()

        def _ensure_capacity(self, plan):
            pass

        def _copy_runtime_metadata(self, *args):
            pass

        def _copy_plan_metadata(self, plan):
            pass

    def fake_create_paged_plan(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, **kwargs
    ):
        calls.append("create_paged_plan")
        assert kwargs["mode"] == "verify"
        assert kwargs["enable_cuda_graph"] is False
        return SimpleNamespace(new_batch_size=4321, total_num_partial_rows=99, split_kv=True)

    def fake_plan_verify_graph_capacity(**kwargs):
        calls.append("plan_verify_graph_capacity")  # must never be called (see docstring)
        raise AssertionError("plan_verify_graph_capacity is the wrong-mode capacity source")

    monkeypatch.setattr(
        "b12x.attention.paged.workspace.PagedAttentionWorkspace",
        FakeWorkspace,
    )
    monkeypatch.setattr(
        "b12x.attention.paged.planner.create_paged_plan",
        fake_create_paged_plan,
    )
    monkeypatch.setattr(
        "b12x.attention.paged.planner.plan_verify_graph_capacity",
        fake_plan_verify_graph_capacity,
    )
    monkeypatch.setattr(
        "b12x.attention.paged._scratch.build_paged_attention_binding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._forward.paged_attention_forward",
        lambda *, binding: None,
    )

    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=4096
    )
    workspace.declare_verify_capacity(16)

    q = torch.empty(16, 48, 128)
    cache = torch.empty(4096, 64, 8, 128)
    output = torch.empty_like(q)
    workspace.forward(
        q=q,
        k_cache=cache,
        v_cache=cache,
        output=output,
        page_table=torch.empty(1, 32, dtype=torch.int32),
        cache_seqlens=torch.empty(1, dtype=torch.int32),
        cu_seqlens_q=torch.empty(2, dtype=torch.int32),
        mode="verify",
        window_left=-1,
        plan_cache_key=object(),
    )

    # Once for the up-front worst-case capacity discovery, once for the real
    # call's own plan.
    assert calls.count("create_paged_plan") == 2
    assert calls.count("eager_extend_work_items_capacity") == 0, (
        "mode='verify' must not size its fixed capacity via the extend-shaped "
        "estimator -- that is exactly the C-1 under-provisioning bug"
    )
    assert calls.count("plan_verify_graph_capacity") == 0, (
        "mode='verify' must not size its fixed capacity via the CUDA-Graph-mode "
        "capacity planner either -- measured wrong on real GPU (different, "
        "smaller chunking policy than the eager path actually uses)"
    )
    assert calls.count("workspace") == 1


def test_prefill_workspace_extend_mode_still_uses_eager_extend_heuristic(monkeypatch):
    """Non-regression check: extend/decode must keep using
    eager_extend_work_items_capacity (unchanged from before the C-1 fix) --
    only verify's capacity source changed.
    """
    pytest.importorskip("b12x")
    calls: list[str] = []

    class FakeWorkspace:
        @staticmethod
        def eager_extend_work_items_capacity(**kwargs):
            calls.append("eager_extend_work_items_capacity")
            return 42

        @classmethod
        def for_fixed_capacity(cls, **kwargs):
            calls.append("workspace")
            assert kwargs["max_work_items"] == 42
            assert kwargs["max_partial_rows"] == 0
            return cls()

        def _ensure_capacity(self, plan):
            pass

        def _copy_runtime_metadata(self, *args):
            pass

        def _copy_plan_metadata(self, plan):
            pass

    monkeypatch.setattr(
        "b12x.attention.paged.workspace.PagedAttentionWorkspace",
        FakeWorkspace,
    )
    monkeypatch.setattr(
        "b12x.attention.paged.planner.create_paged_plan",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._scratch.build_paged_attention_binding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "b12x.attention.paged._forward.paged_attention_forward",
        lambda *, binding: None,
    )

    workspace = SparkinferPrefillWorkspace(
        torch.device("cpu"), max_total_q=8192, max_page_table_width=4096
    )
    q = torch.empty(512, 48, 128)
    cache = torch.empty(4096, 64, 8, 128)
    output = torch.empty_like(q)
    workspace.forward(
        q=q,
        k_cache=cache,
        v_cache=cache,
        output=output,
        page_table=torch.empty(1, 32, dtype=torch.int32),
        cache_seqlens=torch.empty(1, dtype=torch.int32),
        cu_seqlens_q=torch.empty(2, dtype=torch.int32),
        mode="extend",
        window_left=-1,
        plan_cache_key=object(),
    )

    assert calls.count("eager_extend_work_items_capacity") == 1
    assert calls.count("workspace") == 1
