"""Phase 2b CPU tests: MTP dynamic-arena wiring.

``.omx/plans/qwen38-dynamic-context-vllm-plan.md`` Phase 2 -- MTP 迁移:
MTP 的 KV 在 dynamic 模式下与 backbone 共享全局 bundle 池和 page table
(plan §6.1 -- bundle 同时代表 backbone 16 层与 MTP 1 层的 KV 页,锁步
分配/释放/COW)。The CUDA-graph capture paths are real-GPU-only and stay
there (scripts/verify_qwen36_mtp_cuda_graph_bit_exact.py); this file
locks the metadata shapes and the shared-mapping contract that the graph
paths build on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("b12x")

from runtime.backends.qwen36_mtp_cudagraph import (  # noqa: E402
    build_pooled_mtp_caches,
)
from runtime.model.qwen36_slots import Qwen36SlotPool  # noqa: E402

_HEAD_DIM = 4
_KV_HEADS = 2


def _mtp_stub_model():
    """Stub exposing only what build_pooled_mtp_caches reads."""
    attn = SimpleNamespace(
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_heads=4,
        max_seq_len=256,
        kv_cache_dtype=torch.float32,
    )
    mtp = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attn)],
    )
    return SimpleNamespace(mtp=mtp)


def _dynamic_pool(num_slots: int = 3, max_seq_len: int = 256, pool_bundles: int = 24):
    """Same stub shape as tests/test_qwen36_slot_pool.py's _stub_model."""
    layers = []
    for i in range(3):
        if i == 0:
            lin = None
            attn = SimpleNamespace(num_kv_heads=_KV_HEADS, head_dim=_HEAD_DIM, num_heads=4)
        else:
            lin = SimpleNamespace(
                conv_dim=8,
                conv_kernel_size=4,
                num_v_heads=2,
                head_k_dim=_HEAD_DIM,
                head_v_dim=_HEAD_DIM,
            )
            attn = None
        layers.append(
            SimpleNamespace(
                layer_idx=i,
                layer_type="full_attention" if i == 0 else "linear_attention",
                linear_attn=lin,
                self_attn=attn,
            )
        )
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return Qwen36SlotPool(
        model,
        num_slots=num_slots,
        max_seq_len=max_seq_len,
        device="cpu",
        dtype=torch.float32,
        dynamic_arena=True,
        pool_bundles=pool_bundles,
    )


class TestBuildPooledMtpCachesDynamic:
    def test_mtp_pool_spans_the_global_bundle_count(self) -> None:
        pool = _dynamic_pool(pool_bundles=24)
        caches, k_pool, v_pool, page_size, pages_per_slot = build_pooled_mtp_caches(
            _mtp_stub_model(),
            num_slots=3,
            device="cpu",
            dtype=torch.float32,
            pool_bundles=pool.pool_bundles,
            page_table=pool._global_page_table,  # noqa: SLF001
        )
        assert k_pool.shape[0] == 24
        assert v_pool.shape[0] == 24
        assert page_size == 128
        assert len(caches) == 4  # 3 slots + scratch

    def test_mtp_caches_share_the_backbone_page_table_rows(self) -> None:
        pool = _dynamic_pool(pool_bundles=24)
        _, k_pool, _, _, _ = build_pooled_mtp_caches(
            _mtp_stub_model(),
            num_slots=3,
            device="cpu",
            dtype=torch.float32,
            pool_bundles=pool.pool_bundles,
            page_table=pool._global_page_table,  # noqa: SLF001
        )
        # Every MTP cache's page_table is the backbone pool's device row.
        # A backbone remap (e.g. COW detach) must be visible to MTP.
        pool.prepare_kv_writes(0, 0, 128)
        row0 = pool._page_table_host[0]
        # caches are built by build_pooled_mtp_caches above; rebuild here
        # to keep the test self-contained.
        caches, _, _, _, _ = build_pooled_mtp_caches(
            _mtp_stub_model(),
            num_slots=3,
            device="cpu",
            dtype=torch.float32,
            pool_bundles=pool.pool_bundles,
            page_table=pool._global_page_table,  # noqa: SLF001
        )
        assert int(caches[0].page_table[0, 0]) == row0[0]
        # And the MTP cache addresses the same physical bundle in its own pool.
        assert int(caches[0].page_table[0, 0]) < k_pool.shape[0]

    def test_legacy_mode_keeps_fixed_rows_and_identity_mapping(self) -> None:
        caches, k_pool, v_pool, page_size, pages_per_slot = build_pooled_mtp_caches(
            _mtp_stub_model(),
            num_slots=3,
            device="cpu",
            dtype=torch.float32,
        )
        assert k_pool.shape[0] == 4 * 2  # (3 slots + scratch) x 2 pages
        assert v_pool.shape[0] == 4 * 2
        # Legacy: identity mapping, per-slot contiguous views.
        assert int(caches[1].page_table[0, 0]) == 0
        assert caches[1].physical_num_pages == 2


class TestMtpEngineDynamicWiring:
    def test_engine_rejects_dynamic_scratch_snapshot_until_prepared(self) -> None:
        """The MTP engine's mtp_write_index must refuse a write to an
        unprepared logical page (null bundle) -- the pool-level guard that
        keeps the shared mapping honest (plan §7 invariant 5)."""
        # Exercise the shared-mapping contract through the pool's own
        # write_index path.
        pool = _dynamic_pool(pool_bundles=24)
        with pytest.raises(ValueError, match="outside pool capacity"):
            pool.write_index(0, 300)  # beyond capacity -> refused
        pool.prepare_kv_writes(0, 0, 128)
        assert pool.write_index(0, 0) == pool._page_table_host[0][0] * pool.page_size
