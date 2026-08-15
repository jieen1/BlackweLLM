"""CPU tests: shared eager attention workspaces (plan §4.5 P0-M2).

All full-attention layers of this checkpoint share one attention geometry,
so one fixed-capacity workspace per mode must serve the whole group instead
of a per-layer arena (~795 MiB duplicate scratch recovered, step 1); after
every decode batch size replays a captured graph, the eager drivers and the
decode arena are dead residency and are released lazily-rebuildably
(steps 2-3). The workspace constructor is monkeypatched out -- these tests
lock the *sharing and release* contracts, not sparkinfer itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import runtime.model.qwen36_model as qm  # noqa: E402
import runtime.model.qwen36_slots as qs  # noqa: E402
from runtime.model.qwen36_slots import Qwen36SlotPool  # noqa: E402
from tests.test_qwen36_mtp_head import _tiny_config  # noqa: E402
from tests.test_qwen36_slot_pool import _stub_model  # noqa: E402


def _attn_layer(config: dict, layer_idx: int, *, fp8_kv: bool = False) -> qm.Qwen36Attention:
    quantized = {}
    return qm.Qwen36Attention(
        config,
        layer_idx,
        quantized,
        max_seq_len=256,
        enable_fp8_kv=fp8_kv,
    )


def _recording_workspace_factory(created: list[dict]) -> object:
    class FakeWorkspace:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

    return FakeWorkspace


class TestSharedWorkspace:
    def test_same_geometry_layers_share_one_workspace(self, monkeypatch) -> None:
        created: list[dict] = []
        monkeypatch.setattr(qm, "Qwen36AttentionWorkspace", _recording_workspace_factory(created))
        monkeypatch.setattr(qm, "_SHARED_ATTN_WORKSPACES", {})
        config = _tiny_config()
        attn0 = _attn_layer(config, 0)
        attn1 = _attn_layer(config, 1)
        cache0 = attn0.new_cache(device="cpu", dtype=torch.bfloat16)
        cache1 = attn1.new_cache(device="cpu", dtype=torch.bfloat16)

        ws0 = attn0._workspace_for("extend", cache0, torch.bfloat16, torch.device("cpu"))
        ws1 = attn1._workspace_for("extend", cache1, torch.bfloat16, torch.device("cpu"))
        assert ws0 is ws1
        assert len(created) == 1

        # Decode is a different mode -> a second shared workspace, not a third
        # extend arena.
        ws0d = attn0._workspace_for("decode", cache0, torch.bfloat16, torch.device("cpu"))
        ws1d = attn1._workspace_for("decode", cache1, torch.bfloat16, torch.device("cpu"))
        assert ws0d is ws1d
        assert ws0d is not ws0
        assert len(created) == 2
        assert {c["mode"] for c in created} == {"extend", "decode"}

    def test_different_geometry_gets_its_own_workspace(self, monkeypatch) -> None:
        created: list[dict] = []
        monkeypatch.setattr(qm, "Qwen36AttentionWorkspace", _recording_workspace_factory(created))
        monkeypatch.setattr(qm, "_SHARED_ATTN_WORKSPACES", {})
        config = _tiny_config()
        attn0 = _attn_layer(config, 0)
        cache0 = attn0.new_cache(device="cpu", dtype=torch.bfloat16)

        tall = {**config, "head_dim": 16, "num_attention_heads": 2}
        attn1 = _attn_layer(tall, 1)
        cache1 = attn1.new_cache(device="cpu", dtype=torch.bfloat16)

        ws0 = attn0._workspace_for("extend", cache0, torch.bfloat16, torch.device("cpu"))
        ws1 = attn1._workspace_for("extend", cache1, torch.bfloat16, torch.device("cpu"))
        assert ws0 is not ws1
        assert len(created) == 2

    def test_fp8_kv_does_not_share_with_bf16_kv(self, monkeypatch) -> None:
        created: list[dict] = []
        monkeypatch.setattr(qm, "Qwen36AttentionWorkspace", _recording_workspace_factory(created))
        monkeypatch.setattr(qm, "_SHARED_ATTN_WORKSPACES", {})
        config = _tiny_config()
        attn_bf16 = _attn_layer(config, 0)
        attn_fp8 = _attn_layer(config, 1, fp8_kv=True)
        cache_bf16 = attn_bf16.new_cache(device="cpu", dtype=torch.bfloat16)
        cache_fp8 = attn_fp8.new_cache(device="cpu", dtype=torch.bfloat16)

        ws_bf16 = attn_bf16._workspace_for(
            "extend", cache_bf16, torch.bfloat16, torch.device("cpu")
        )
        ws_fp8 = attn_fp8._workspace_for("extend", cache_fp8, torch.bfloat16, torch.device("cpu"))
        assert ws_bf16 is not ws_fp8
        assert created[1]["kv_dtype"] is torch.float8_e4m3fn

    def test_forward_passes_per_layer_descale(self, monkeypatch) -> None:
        """The shared workspace must receive the calling layer's own K/V
        scales at forward time (FP8 KV), not the creating layer's."""
        seen: list[tuple[torch.Tensor, torch.Tensor]] = []

        class FakeWorkspace:
            mode = "decode"

            def __init__(self, **kwargs) -> None:
                pass

            def forward(
                self,
                *,
                q: torch.Tensor,
                k_cache: torch.Tensor,
                v_cache: torch.Tensor,
                output: torch.Tensor,
                page_table: torch.Tensor,
                cache_seqlens: torch.Tensor,
                cu_seqlens_q: torch.Tensor,
                k_descale: torch.Tensor | None = None,
                v_descale: torch.Tensor | None = None,
            ) -> None:
                seen.append((k_descale, v_descale))

        monkeypatch.setattr(qm, "Qwen36AttentionWorkspace", FakeWorkspace)
        monkeypatch.setattr(qm, "_SHARED_ATTN_WORKSPACES", {})
        config = _tiny_config()
        attn0 = _attn_layer(config, 0, fp8_kv=True)
        attn1 = _attn_layer(config, 1, fp8_kv=True)
        cache0 = attn0.new_cache(device="cpu", dtype=torch.bfloat16)
        cache1 = attn1.new_cache(device="cpu", dtype=torch.bfloat16)

        # Materialize the shared workspace through layer 0, then run a decode
        # through layer 1: the descale must be layer 1's own scales.
        attn0._workspace_for("decode", cache0, torch.bfloat16, torch.device("cpu"))
        q = torch.zeros(1, config["num_attention_heads"], config["head_dim"])
        attn1._workspace_for("decode", cache1, torch.bfloat16, torch.device("cpu")).forward(
            q=q,
            k_cache=cache1.k_cache,
            v_cache=cache1.v_cache,
            output=torch.zeros_like(q),
            page_table=cache1.page_table,
            cache_seqlens=torch.tensor([1], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            k_descale=attn1.k_scale,
            v_descale=attn1.v_scale,
        )
        assert len(seen) == 1
        assert seen[0][0] is attn1.k_scale
        assert seen[0][1] is attn1.v_scale


class TestReleaseDecodeResidency:
    """Plan §4.5 P0-M2 steps 2-3: after every batch size replays a captured
    decode graph, the eager batched drivers and the shared decode-mode
    workspace are dead residency -- releasing them must be safe (a degraded
    path rebuilds lazily) and must not touch the extend arena."""

    def test_release_eager_decode_drivers_clears_and_rebuilds(self, monkeypatch) -> None:
        class FakeDriver:
            def __init__(self, **kwargs) -> None:
                pass

        monkeypatch.setattr(qs, "Qwen36BatchedDecodeAttention", FakeDriver)
        pool = Qwen36SlotPool(
            _stub_model(["full_attention"]),
            num_slots=2,
            max_seq_len=256,
            device="cpu",
            dtype=torch.float32,
        )
        pool.attention_driver(1)
        pool.attention_driver(2)
        assert set(pool.decode_attn) == {1, 2}

        assert pool.release_eager_decode_drivers() == 2
        assert pool.decode_attn == {}
        assert pool.release_eager_decode_drivers() == 0  # idempotent

        # A degraded path that needs an eager driver again rebuilds lazily.
        pool.attention_driver(1)
        assert set(pool.decode_attn) == {1}

    def test_release_decode_workspaces_drops_registry_and_layer_caches(self, monkeypatch) -> None:
        created: list[dict] = []
        monkeypatch.setattr(qm, "Qwen36AttentionWorkspace", _recording_workspace_factory(created))
        monkeypatch.setattr(qm, "_SHARED_ATTN_WORKSPACES", {})
        config = _tiny_config()
        attn0 = _attn_layer(config, 0)
        attn1 = _attn_layer(config, 1)
        mtp_attn = _attn_layer(config, 0)
        device = torch.device("cpu")
        cache0 = attn0.new_cache(device="cpu", dtype=torch.bfloat16)
        cache1 = attn1.new_cache(device="cpu", dtype=torch.bfloat16)
        cache_mtp = mtp_attn.new_cache(device="cpu", dtype=torch.bfloat16)

        ws_decode = attn0._workspace_for("decode", cache0, torch.bfloat16, device)
        assert attn1._workspace_for("decode", cache1, torch.bfloat16, device) is ws_decode
        assert mtp_attn._workspace_for("decode", cache_mtp, torch.bfloat16, device) is ws_decode
        ws_extend = attn0._workspace_for("extend", cache0, torch.bfloat16, device)
        assert len(created) == 2

        fake_model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=attn0),
                    SimpleNamespace(self_attn=attn1),
                    SimpleNamespace(self_attn=None),  # a GDN layer
                ]
            ),
            mtp=SimpleNamespace(layers=[SimpleNamespace(self_attn=mtp_attn)]),
        )
        dropped = qm.Qwen36ForCausalLMSelfBuilt.release_decode_workspaces(fake_model)
        assert dropped == 1
        assert attn0._decode_workspace is None
        assert attn1._decode_workspace is None
        assert mtp_attn._decode_workspace is None
        # The extend arena is untouched.
        assert attn0._extend_workspace is ws_extend
        assert qm._SHARED_ATTN_WORKSPACES
        assert all(key[0] != "decode" for key in qm._SHARED_ATTN_WORKSPACES)

        # A later eager decode rebuilds on demand.
        attn0._workspace_for("decode", cache0, torch.bfloat16, device)
        assert len(created) == 3

    def test_release_decode_workspaces_without_mtp(self, monkeypatch) -> None:
        monkeypatch.setattr(qm, "_SHARED_ATTN_WORKSPACES", {})
        fake_model = SimpleNamespace(model=SimpleNamespace(layers=[]), mtp=None)
        assert qm.Qwen36ForCausalLMSelfBuilt.release_decode_workspaces(fake_model) == 0
