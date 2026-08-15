"""B3: unit tests for the MTP draft head's construction + weight loading.

CPU-only, tiny synthetic config -- same convention as
``tests/test_qwen36_slot_pool.py`` (real classes, small dimensions,
never a real 27B checkpoint, never a CUDA forward). Construction and
``load_weights`` are plain tensor bookkeeping (``nn.Module.__init__``,
``PlainLinear`` weight allocation, ``default_weight_loader``'s
shape-matched ``.copy_()``) -- none of it calls sparkinfer's paged
attention kernel or FLA's GDN kernels, which only run on a first
``forward()`` call. That real-kernel half is proven correct on GPU
separately (this session's B3 report / GPU probe scripts), matching this
repo's existing convention of keeping kernel-touching proofs out of the
CPU pytest suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
# qwen36_model imports fla/sparkinfer at module level even though
# construction-only tests below never call either -- same guard
# tests/test_qwen36_slot_pool.py and tests/test_qwen36_gdn_spec_rollback.py
# use for the same reason.
pytest.importorskip("fla")
pytest.importorskip("b12x")

import runtime.model.qwen36_model as qwen36_model_module  # noqa: E402
from runtime.model.qwen36_model import (  # noqa: E402
    Qwen36Attention,
    Qwen36ForCausalLMSelfBuilt,
    Qwen36MLP,
    Qwen36MTPHead,
)


def _tiny_config(*, mtp_num_hidden_layers: int = 1) -> dict:
    """A structurally-complete but tiny Qwen3.6-shaped config: two
    linear_attention (GDN) layers and two full_attention layers, so
    Qwen36TextModelSelfBuilt.__init__ exercises both branches, at
    dimensions small enough that constructing the whole model on CPU is
    instant and allocates kilobytes, not gigabytes."""
    return {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "attention_bias": False,
        "attn_output_gate": True,
        "rms_norm_eps": 1e-6,
        "intermediate_size": 64,
        "hidden_act": "silu",
        "layer_types": [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        "num_hidden_layers": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 50,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
        "max_position_embeddings": 64,
        "mtp_num_hidden_layers": mtp_num_hidden_layers,
        "quantization_config": None,
        "tie_word_embeddings": False,
    }


#: The exact 15 mtp.* checkpoint tensor names this session verified
#: against the real nvidia/Qwen3.6-27B-NVFP4 checkpoint's
#: model.safetensors.index.json (see notes / B3 report) -- locked down
#: here so a change to Qwen36MTPHead's module tree that silently drifts
#: from the real checkpoint's naming fails this test, not a live GPU load.
REAL_CHECKPOINT_MTP_NAMES = frozenset(
    {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    }
)


class TestQwen36MTPHeadConstruction:
    def test_disabled_by_default(self) -> None:
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64)
        assert model.mtp is None

    def test_enabled_constructs_the_head(self) -> None:
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        assert isinstance(model.mtp, Qwen36MTPHead)

    def test_parameter_names_match_the_real_checkpoint_exactly(self) -> None:
        """The module tree's own dotted parameter names, restricted to the
        mtp.* prefix, must be BYTE-FOR-BYTE the real checkpoint's 15 tensor
        names -- load_weights (below) relies on this being a literal 1:1
        match, no remapping."""
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        mtp_names = {name for name, _ in model.named_parameters() if name.startswith("mtp.")}
        assert mtp_names == REAL_CHECKPOINT_MTP_NAMES

    def test_rejects_more_than_one_mtp_layer(self) -> None:
        """Only mtp_num_hidden_layers=1 (the real checkpoint's value) is
        verified -- see Qwen36MTPHead's docstring."""
        with pytest.raises(AssertionError):
            Qwen36ForCausalLMSelfBuilt(
                _tiny_config(mtp_num_hidden_layers=2), max_seq_len=64, enable_mtp=True
            )


def _weights_for(model: Qwen36ForCausalLMSelfBuilt) -> list[tuple[str, torch.Tensor]]:
    """Build a synthetic (checkpoint_name, tensor) list covering every
    parameter ``model`` has -- backbone, lm_head, and (if present) mtp --
    with distinguishable random values, keyed by the checkpoint naming
    convention Qwen36ForCausalLMSelfBuilt.load_weights actually expects
    (mirrors the real checkpoint's prefixes, not the module's own
    attribute paths for the backbone -- see that method's docstring)."""
    weights: list[tuple[str, torch.Tensor]] = []
    gen = torch.Generator().manual_seed(0)
    for name, param in model.named_parameters():
        if ".linear_attn.in_proj_qkvz." in name:
            layer = model.get_submodule(name.rsplit(".in_proj_qkvz.", 1)[0])
            assert isinstance(layer, qwen36_model_module.Qwen36GatedDeltaNet)
            suffix = name.rsplit(".in_proj_qkvz.", 1)[1]
            qkv, z = param.split((layer.conv_dim, layer.value_dim), dim=0)
            checkpoint_prefix = (
                "model.language_model." + name[len("model.") :].rsplit(".in_proj_qkvz.", 1)[0]
            )
            weights.extend(
                [
                    (
                        f"{checkpoint_prefix}.in_proj_qkv.{suffix}",
                        torch.randn(qkv.shape, generator=gen),
                    ),
                    (
                        f"{checkpoint_prefix}.in_proj_z.{suffix}",
                        torch.randn(z.shape, generator=gen),
                    ),
                ]
            )
            continue
        if ".linear_attn.in_proj_ba." in name:
            layer = model.get_submodule(name.rsplit(".in_proj_ba.", 1)[0])
            assert isinstance(layer, qwen36_model_module.Qwen36GatedDeltaNet)
            suffix = name.rsplit(".in_proj_ba.", 1)[1]
            b, a = param.split(layer.num_v_heads, dim=0)
            checkpoint_prefix = (
                "model.language_model." + name[len("model.") :].rsplit(".in_proj_ba.", 1)[0]
            )
            weights.extend(
                [
                    (
                        f"{checkpoint_prefix}.in_proj_b.{suffix}",
                        torch.randn(b.shape, generator=gen),
                    ),
                    (
                        f"{checkpoint_prefix}.in_proj_a.{suffix}",
                        torch.randn(a.shape, generator=gen),
                    ),
                ]
            )
            continue
        if name.startswith("model."):
            ckpt_name = "model.language_model." + name[len("model.") :]
        elif name.startswith("lm_head.") or name.startswith("mtp."):
            ckpt_name = name
        else:
            raise AssertionError(f"unexpected top-level parameter name: {name!r}")
        weights.append((ckpt_name, torch.randn(param.shape, generator=gen)))
    return weights


class TestQwen36MTPHeadWeightLoading:
    def test_legacy_gdn_input_tensors_fill_historical_fused_layout(self) -> None:
        """The physical qkvz/ba parameters must preserve checkpoint row order.

        This is deliberately a loader-level test: it proves fusion is a
        direct byte-preserving concatenation of real checkpoint tensor
        families, not a later BF16 conversion or a guessed permutation.
        """
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64)
        weights = _weights_for(donor)
        by_name = dict(weights)
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64)
        model.load_weights(weights)

        gdn = model.model.layers[0].linear_attn
        assert isinstance(gdn, qwen36_model_module.Qwen36GatedDeltaNet)
        checkpoint_prefix = "model.language_model.layers.0.linear_attn"
        assert torch.equal(
            gdn.in_proj_qkvz.weight[: gdn.conv_dim],
            by_name[f"{checkpoint_prefix}.in_proj_qkv.weight"],
        )
        assert torch.equal(
            gdn.in_proj_qkvz.weight[gdn.conv_dim :],
            by_name[f"{checkpoint_prefix}.in_proj_z.weight"],
        )
        assert torch.equal(
            gdn.in_proj_ba.weight[: gdn.num_v_heads],
            by_name[f"{checkpoint_prefix}.in_proj_b.weight"],
        )
        assert torch.equal(
            gdn.in_proj_ba.weight[gdn.num_v_heads :],
            by_name[f"{checkpoint_prefix}.in_proj_a.weight"],
        )

    def test_enable_mtp_true_loads_every_mtp_tensor(self) -> None:
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        weights = _weights_for(donor)

        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        loaded = model.load_weights(weights)

        loaded_mtp = {n for n in loaded if n.startswith("mtp.")}
        assert loaded_mtp == REAL_CHECKPOINT_MTP_NAMES
        assert model.skipped_mtp_count == 0

    def test_enable_mtp_true_actually_copies_the_values_not_just_marks_loaded(self) -> None:
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        weights = _weights_for(donor)
        weights_by_name = dict(weights)

        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        model.load_weights(weights)

        assert torch.equal(model.mtp.fc.weight, weights_by_name["mtp.fc.weight"])
        assert torch.equal(
            model.mtp.layers[0].self_attn.q_proj.weight,
            weights_by_name["mtp.layers.0.self_attn.q_proj.weight"],
        )
        assert torch.equal(
            model.mtp.pre_fc_norm_embedding.weight,
            weights_by_name["mtp.pre_fc_norm_embedding.weight"],
        )

    def test_enable_mtp_false_skips_every_mtp_tensor_like_b1(self) -> None:
        """The exact B1 behavior this class had before B3: mtp.* tensors
        are counted, never loaded, and model.mtp stays None. The count
        (15) is the same real-checkpoint fact REAL_CHECKPOINT_MTP_NAMES
        locks down above."""
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        weights = _weights_for(donor)

        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=False)
        loaded = model.load_weights(weights)

        assert model.mtp is None
        assert not any(n.startswith("mtp.") for n in loaded)
        assert model.skipped_mtp_count == len(REAL_CHECKPOINT_MTP_NAMES)

    def test_backbone_and_lm_head_still_fully_load_with_mtp_enabled(self) -> None:
        """B3 must not regress B1/B2's own loading -- every non-mtp
        parameter still ends up loaded when enable_mtp=True, exactly as
        when it is False."""
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        weights = _weights_for(donor)

        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_mtp=True)
        loaded = model.load_weights(weights)

        all_param_names = {name for name, _ in model.named_parameters()}
        assert loaded == all_param_names


class TestQwen36BatchedVerifyAttention:
    def test_flattens_all_requests_for_rope_kv_write_and_paged_attention(self, monkeypatch) -> None:
        """M-2's B>1 verify must use ``B * qo_len`` token rows throughout.

        The real paged-attention driver is CUDA-only.  This CPU test retains
        its input/output contract with a tiny recording driver, specifically
        catching the old B=1 reshape that discarded the batch dimension after
        Q/K/V had already been flattened.
        """
        attn = Qwen36Attention(_tiny_config(), 1, {}, max_seq_len=64)
        batch, qo_len = 2, 3

        class _VerifyDriver:
            batch = 2
            verify_tokens = 3

            def __init__(self) -> None:
                self.q_shape: tuple[int, ...] | None = None

            def forward(self, *, q, output, **kwargs) -> None:
                del kwargs
                self.q_shape = tuple(q.shape)
                output.copy_(q)

        driver = _VerifyDriver()
        monkeypatch.setattr(
            qwen36_model_module, "apply_rotary_embedding_inplace", lambda *args: None
        )
        hidden = torch.randn(batch, qo_len, 32)
        positions = torch.arange(batch * qo_len)
        k_pool = torch.zeros(2, 8, 2, 8)
        v_pool = torch.zeros_like(k_pool)
        output = torch.empty(batch * qo_len, 4, 8)

        result = attn.verify_batch(
            hidden,
            positions,
            torch.empty(1),
            k_pool=k_pool,
            v_pool=v_pool,
            write_index=torch.arange(batch * qo_len),
            attn=driver,
            output=output,
        )

        assert driver.q_shape == (batch * qo_len, 4, 8)
        assert result.shape == (batch, qo_len, 32)


class TestWeightPrefixOverride:
    """Qwen36Attention/Qwen36MLP's weight_prefix kwarg (added for
    Qwen36MTPHead's reuse of these classes) must default to the exact
    prefix these classes always derived before B3, and must genuinely
    change classification when overridden -- not silently ignored."""

    def test_default_prefix_matches_pre_b3_derivation(self) -> None:
        config = _tiny_config()
        quantized = {"model.language_model.layers.3.self_attn.q_proj": "FP8"}
        attn = Qwen36Attention(config, 3, quantized, max_seq_len=64)
        # FP8 classification only fires if the internally-derived prefix
        # equals "model.language_model.layers.3.self_attn" -- the OLD
        # hardcoded value -- so a PlainLinear-vs-ModelOptFP8Linear check
        # is an indirect but exact proof of what prefix string was used.
        from runtime.model.modelopt_linear import ModelOptFP8Linear

        assert isinstance(attn.q_proj, ModelOptFP8Linear)

    def test_override_prefix_is_actually_used(self) -> None:
        config = _tiny_config()
        # This entry would classify layer 3's q_proj as FP8 under the
        # OLD/default prefix -- but weight_prefix below points elsewhere,
        # so it must NOT match and q_proj must come back unquantized.
        quantized = {"model.language_model.layers.3.self_attn.q_proj": "FP8"}
        attn = Qwen36Attention(
            config, 3, quantized, max_seq_len=64, weight_prefix="mtp.layers.0.self_attn"
        )
        from runtime.model.plain_linear import PlainLinear

        assert isinstance(attn.q_proj, PlainLinear)

    def test_mlp_default_prefix_matches_pre_b3_derivation(self) -> None:
        config = _tiny_config()
        quantized = {"model.language_model.layers.1.mlp.gate_proj": "W4A16_NVFP4"}
        mlp = Qwen36MLP(config, 1, quantized)
        from runtime.model.modelopt_linear import ModelOptNVFP4Linear

        assert isinstance(mlp.gate_proj, ModelOptNVFP4Linear)

    def test_mlp_override_prefix_is_actually_used(self) -> None:
        config = _tiny_config()
        quantized = {"model.language_model.layers.1.mlp.gate_proj": "W4A16_NVFP4"}
        mlp = Qwen36MLP(config, 1, quantized, weight_prefix="mtp.layers.0.mlp")
        from runtime.model.plain_linear import PlainLinear

        assert isinstance(mlp.gate_proj, PlainLinear)
