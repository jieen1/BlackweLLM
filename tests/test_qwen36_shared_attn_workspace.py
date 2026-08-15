"""CPU tests: shared eager attention workspaces (plan §4.5 P0-M2 step 1).

All full-attention layers of this checkpoint share one attention geometry,
so one fixed-capacity workspace per mode must serve the whole group instead
of a per-layer arena (~795 MiB duplicate scratch recovered). The workspace
constructor is monkeypatched out -- these tests lock the *sharing* contract
(the registry key and per-layer descale passing), not sparkinfer itself.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import runtime.model.qwen36_model as qm  # noqa: E402
from tests.test_qwen36_mtp_head import _tiny_config  # noqa: E402


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
