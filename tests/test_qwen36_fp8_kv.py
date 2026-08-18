"""FP8 KV cache (2026-08-03 follow-up) -- gated by ``enable_fp8_kv``/
``QSR_QWEN36_FP8_KV``, **default ON since 2026-08-03** (`QSR_QWEN36_FP8_KV=0`
opts out) -- measured positive on correctness, memory and speed at once; see
``notes/2026-08-03-fp8-kv-cache.md`` and ``runtime/model_loading.py``.

Same convention as ``tests/test_qwen36_mtp_head.py``/
``tests/test_qwen36_slot_pool.py``: CPU-only, tiny synthetic config or a
minimal stub, never a real 27B checkpoint, never a CUDA forward. The one
thing a CPU suite genuinely cannot exercise -- whether sparkinfer's paged-
attention kernel actually reads an FP8 KV cache back out correctly, and
whether the whole-model output stays inside B1-R's calibrated bars -- is
proven on GPU separately (this session's B1-R full-model-gap run). What
CAN be pinned down here, and would otherwise be genuinely invisible until
a real GPU run produced silently-wrong logits, is:

1. **Scale direction.** ``_kv_to_cache_dtype`` must DIVIDE by scale when
   quantizing (``fp8_stored = real / scale``), matching
   ``runtime/kernels/fused_kv_scatter.py`` and Laguna's own FP8 KV write
   path -- not multiply. Getting this inverted is exactly the class of bug
   this codebase has hit before with a different scale pair
   (``CompressedTensorsNVFP4Linear``'s reciprocal ``weight_global_scale``
   vs. modelopt's ``weight_scale_2``, which produced degenerate
   ``"!!!!!!!!!!!!"`` output on a real GPU run before it was caught) -- a
   model that loads, runs, and emits garbage, with nothing short of
   reading logits to notice. This test reads the resulting FP8 byte
   pattern directly instead, so the inversion is caught on CPU before any
   GPU run is spent on it.

2. **The checkpoint's real 32 k_scale/v_scale tensors actually get
   consumed once the flag is on.** Before this change,
   ``Qwen36Attention`` had no ``k_scale``/``v_scale`` Parameter at all, so
   these checkpoint tensors fell through ``load_weights``'s ``mapped not
   in params_dict: continue`` silently -- the exact failure mode
   ``warn_on_unconsumed_tensor_families`` exists to surface (see
   ``tests/test_unconsumed_checkpoint_tensors.py``'s own docstring for a
   sibling regression of this same shape). A construction-only test
   cannot catch a tensor silently not being loaded; only an actual
   ``load_weights`` round-trip can.

3. **The default-off path is byte-for-byte what main already ships** --
   no new Parameter, no new buffer, BF16 KV cache -- so landing this
   behind a flag genuinely does not change production until the flag is
   flipped.

4. **The memory-saving mechanism itself** (``Qwen36SlotPool``'s k/v pool
   dtype, which is what actually produces the halved resident bytes the
   GPU measurement reports) is plain tensor-shape/dtype bookkeeping, fully
   exercisable against a stub model on CPU exactly like
   ``tests/test_qwen36_slot_pool.py`` already does for the rest of the
   pool's geometry.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# qwen36_model imports fla/sparkinfer at module level even though
# construction-only tests below never call either -- same guard
# tests/test_qwen36_mtp_head.py and tests/test_qwen36_slot_pool.py use for
# the same reason.
pytest.importorskip("fla")
pytest.importorskip("b12x")

from runtime.model.qwen36_model import (  # noqa: E402
    Qwen36Attention,
    Qwen36ForCausalLMSelfBuilt,
    _kv_to_cache_dtype,
    _store_batched_kv_rows,
)
from runtime.model.qwen36_slots import Qwen36SlotPool  # noqa: E402


def _tiny_config(*, mtp_num_hidden_layers: int = 1) -> dict:
    """Same tiny Qwen3.6-shaped config as
    ``tests/test_qwen36_mtp_head.py``'s helper of the same name (kept as
    its own copy here, matching this repo's existing per-file-helper
    convention rather than a shared import)."""
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


def _weights_for(model: Qwen36ForCausalLMSelfBuilt) -> list[tuple[str, torch.Tensor]]:
    """Same helper as ``tests/test_qwen36_mtp_head.py``'s: a synthetic
    ``(checkpoint_name, tensor)`` list covering every parameter ``model``
    has, keyed by the checkpoint naming convention
    ``Qwen36ForCausalLMSelfBuilt.load_weights`` actually expects."""
    weights: list[tuple[str, torch.Tensor]] = []
    gen = torch.Generator().manual_seed(0)
    for name, param in model.named_parameters():
        if name.startswith("model."):
            ckpt_name = "model.language_model." + name[len("model.") :]
        elif name.startswith("lm_head.") or name.startswith("mtp."):
            ckpt_name = name
        else:
            raise AssertionError(f"unexpected top-level parameter name: {name!r}")
        weights.append((ckpt_name, torch.randn(param.shape, generator=gen)))
    return weights


class TestKvToCacheDtypeScaleDirection:
    """``_kv_to_cache_dtype`` is the one function that decides whether the
    checkpoint's k_scale/v_scale gets applied the right way round."""

    def test_fp8_quantizes_by_dividing_by_scale_not_multiplying(self) -> None:
        key = torch.full((2, 3, 4), 1.5, dtype=torch.bfloat16)
        value = torch.full((2, 3, 4), -0.7, dtype=torch.bfloat16)
        # Both exactly representable in float32/bf16 -- no rounding noise
        # in the scale itself to confound the direction check.
        k_scale = torch.tensor([0.03125], dtype=torch.float32)
        v_scale = torch.tensor([0.0625], dtype=torch.float32)

        k_out, v_out = _kv_to_cache_dtype(
            key, value, cache_dtype=torch.float8_e4m3fn, k_scale=k_scale, v_scale=v_scale
        )
        assert k_out.dtype == torch.float8_e4m3fn
        assert v_out.dtype == torch.float8_e4m3fn

        # Computed independently (real/scale), not by re-calling the
        # function under test.
        expected_k = (torch.tensor(1.5, dtype=torch.float32) / 0.03125).to(torch.float8_e4m3fn)
        expected_v = (torch.tensor(-0.7, dtype=torch.float32) / 0.0625).to(torch.float8_e4m3fn)
        assert torch.all(k_out.float() == expected_k.float())
        assert torch.all(v_out.float() == expected_v.float())

        # The inverted (wrong) convention would land on a materially
        # different FP8 grid point -- 1.5*0.03125=0.046875 vs.
        # 1.5/0.03125=48.0 -- so this is a decisive check, not a
        # tautology restating the formula under test.
        inverted_k = (torch.tensor(1.5, dtype=torch.float32) * 0.03125).to(torch.float8_e4m3fn)
        assert k_out[0, 0, 0].float().item() != inverted_k.float().item()

    def test_fp8_round_trip_reconstructs_the_real_value(self) -> None:
        """Dequantizing with a MULTIPLY (what the kernel's k_descale/
        v_descale does -- see Qwen36Attention's docstring) must land close
        to the original real value, using scales measured off the real
        standard checkpoint (layer 3: k_scale=0.0262, v_scale=0.0344)."""
        torch.manual_seed(20260803)
        key = (torch.randn(4, 6, 8) * 2.0).to(torch.bfloat16)
        value = (torch.randn(4, 6, 8) * 2.0).to(torch.bfloat16)
        k_scale = torch.tensor([0.0262451171875], dtype=torch.float32)
        v_scale = torch.tensor([0.034423828125], dtype=torch.float32)

        k_out, v_out = _kv_to_cache_dtype(
            key, value, cache_dtype=torch.float8_e4m3fn, k_scale=k_scale, v_scale=v_scale
        )
        k_reconstructed = k_out.float() * k_scale
        v_reconstructed = v_out.float() * v_scale

        # E4M3 has ~2 decimal digits of precision; a generous absolute
        # tolerance still catches a scale applied in the wrong direction
        # (which would be off by ~1/scale**2, i.e. thousands of times
        # larger than quantization noise, not merely a few percent).
        assert torch.allclose(k_reconstructed.float(), key.float(), atol=0.25)
        assert torch.allclose(v_reconstructed.float(), value.float(), atol=0.25)

    def test_bf16_cache_is_a_plain_cast_ignoring_scale_entirely(self) -> None:
        """The default (flag-off) path: no scale must ever be applied."""
        key = torch.full((2, 2), 5.0, dtype=torch.float32)
        value = torch.full((2, 2), -3.0, dtype=torch.float32)
        # A scale far from 1.0 -- if this were consulted, the BF16 output
        # would visibly differ from a plain cast.
        k_scale = torch.tensor([123.0])
        v_scale = torch.tensor([0.001])

        k_out, v_out = _kv_to_cache_dtype(
            key, value, cache_dtype=torch.bfloat16, k_scale=k_scale, v_scale=v_scale
        )
        assert torch.equal(k_out, key.to(torch.bfloat16))
        assert torch.equal(v_out, value.to(torch.bfloat16))

    def test_bf16_cache_works_with_no_scale_tensors_at_all(self) -> None:
        """The construction-time contract for enable_fp8_kv=False:
        k_scale/v_scale are None, and the BF16 path must not need them."""
        key = torch.ones(2, 2)
        value = torch.zeros(2, 2)
        k_out, v_out = _kv_to_cache_dtype(
            key, value, cache_dtype=torch.bfloat16, k_scale=None, v_scale=None
        )
        assert torch.equal(k_out, key.to(torch.bfloat16))
        assert torch.equal(v_out, value.to(torch.bfloat16))

    def test_fp8_cache_without_scale_tensors_fails_loudly(self) -> None:
        """A FP8 cache_dtype with no real scale is a construction bug --
        must never silently fall back to an implicit 1.0."""
        with pytest.raises(AssertionError):
            _kv_to_cache_dtype(
                torch.ones(2, 2),
                torch.ones(2, 2),
                cache_dtype=torch.float8_e4m3fn,
                k_scale=None,
                v_scale=None,
            )


class TestQwen36AttentionConstructionGating:
    def test_disabled_by_default_no_scale_parameters_at_all(self) -> None:
        attn = Qwen36Attention(_tiny_config(), 3, {}, max_seq_len=64)
        assert attn.kv_cache_dtype == torch.bfloat16
        assert attn.k_scale is None
        assert attn.v_scale is None
        assert "k_scale" not in dict(attn.named_parameters())
        assert "v_scale" not in dict(attn.named_parameters())

    def test_enabled_creates_real_scale_parameters(self) -> None:
        attn = Qwen36Attention(_tiny_config(), 3, {}, max_seq_len=64, enable_fp8_kv=True)
        assert attn.kv_cache_dtype == torch.float8_e4m3fn
        assert isinstance(attn.k_scale, torch.nn.Parameter)
        assert isinstance(attn.v_scale, torch.nn.Parameter)
        assert attn.k_scale.shape == (1,)
        assert attn.v_scale.shape == (1,)
        assert attn.k_scale.dtype == torch.float32
        # Default value before any checkpoint tensor is loaded -- the
        # harmless no-op, same starting point SelfBuiltAttentionPlaceholder
        # (Laguna's own k_scale/v_scale Parameters) uses.
        assert float(attn.k_scale.item()) == 1.0
        assert float(attn.v_scale.item()) == 1.0

    def test_new_cache_uses_this_layers_own_kv_cache_dtype(self) -> None:
        """new_cache's `dtype` argument is accepted for call-site
        compatibility but no longer decides the KV storage dtype -- that
        is `self.kv_cache_dtype` now, so a caller passing the model's
        ordinary BF16 compute dtype still gets an FP8 cache for a layer
        built with enable_fp8_kv=True."""
        attn = Qwen36Attention(_tiny_config(), 3, {}, max_seq_len=64, enable_fp8_kv=True)
        cache = attn.new_cache(device=torch.device("cpu"), dtype=torch.bfloat16)
        assert cache.k_cache.dtype == torch.float8_e4m3fn
        assert cache.v_cache.dtype == torch.float8_e4m3fn

        attn_bf16 = Qwen36Attention(_tiny_config(), 3, {}, max_seq_len=64)
        cache_bf16 = attn_bf16.new_cache(device=torch.device("cpu"), dtype=torch.bfloat16)
        assert cache_bf16.k_cache.dtype == torch.bfloat16
        assert cache_bf16.v_cache.dtype == torch.bfloat16


def test_batched_kv_store_keeps_bf16_fallback_semantics() -> None:
    key = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    value = -key
    k_pool = torch.zeros((2, 4, 2, 3), dtype=torch.bfloat16)
    v_pool = torch.zeros_like(k_pool)
    write_index = torch.tensor([1, 6], dtype=torch.long)

    _store_batched_kv_rows(
        key,
        value,
        k_pool=k_pool,
        v_pool=v_pool,
        write_index=write_index,
        k_scale=None,
        v_scale=None,
    )

    expected_k = key.to(torch.bfloat16)
    expected_v = value.to(torch.bfloat16)
    assert torch.equal(k_pool.view(-1, 2, 3)[write_index], expected_k)
    assert torch.equal(v_pool.view(-1, 2, 3)[write_index], expected_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 fused KV scatter requires CUDA")
def test_batched_kv_store_fp8_matches_scale_cast_on_cuda() -> None:
    key = torch.tensor(
        [[[1.5, -2.0, 3.25], [4.0, -5.5, 6.0]], [[-7.0, 8.0, -9.5], [1.0, 2.0, 3.0]]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = key * -0.5
    k_pool = torch.zeros((2, 4, 2, 3), dtype=torch.float8_e4m3fn, device="cuda")
    v_pool = torch.zeros_like(k_pool)
    write_index = torch.tensor([1, 6], dtype=torch.long, device="cuda")
    k_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    _store_batched_kv_rows(
        key,
        value,
        k_pool=k_pool,
        v_pool=v_pool,
        write_index=write_index,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    expected_k = (key / k_scale).to(torch.float8_e4m3fn)
    expected_v = (value / v_scale).to(torch.float8_e4m3fn)
    assert torch.equal(k_pool.view(-1, 2, 3)[write_index], expected_k)
    assert torch.equal(v_pool.view(-1, 2, 3)[write_index], expected_v)


class TestFullModelDefaultOff:
    def test_default_construction_has_no_kv_scale_parameters_anywhere(self) -> None:
        """The shipped path (no enable_fp8_kv argument at all) must be
        byte-for-byte what main already builds -- no new Parameter."""
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64)
        names = {name for name, _ in model.named_parameters()}
        assert not any(n.endswith(".k_scale") or n.endswith(".v_scale") for n in names)
        for layer in model.model.layers:
            if layer.self_attn is not None:
                assert layer.self_attn.kv_cache_dtype == torch.bfloat16

    def test_explicit_false_matches_the_default(self) -> None:
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_fp8_kv=False)
        for layer in model.model.layers:
            if layer.self_attn is not None:
                assert layer.self_attn.kv_cache_dtype == torch.bfloat16


class TestFullModelFp8KvEnabled:
    def test_every_full_attention_layer_gets_fp8_kv_dtype(self) -> None:
        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_fp8_kv=True)
        full_attn_layers = [layer for layer in model.model.layers if layer.self_attn is not None]
        assert len(full_attn_layers) == 2  # matches _tiny_config's layer_types
        for layer in full_attn_layers:
            assert layer.self_attn.kv_cache_dtype == torch.float8_e4m3fn

    def test_mtp_head_never_gets_fp8_kv_even_when_backbone_does(self) -> None:
        """The MTP checkpoint has no k_scale/v_scale tensor for its own
        self_attn (verified against the real checkpoint's 15 mtp.*
        tensors, none named *_scale) -- enable_fp8_kv must never reach it,
        even implicitly."""
        model = Qwen36ForCausalLMSelfBuilt(
            _tiny_config(), max_seq_len=64, enable_mtp=True, enable_fp8_kv=True
        )
        assert model.mtp is not None
        assert model.mtp.layers[0].self_attn.kv_cache_dtype == torch.bfloat16
        assert model.mtp.layers[0].self_attn.k_scale is None

    def test_the_real_checkpoints_k_scale_v_scale_tensors_get_consumed(self) -> None:
        """The regression this whole feature closes: before enable_fp8_kv
        existed, these 2 tensors per full-attention layer (32 on the real
        checkpoint) fell through load_weights silently. Now they must be
        real, loaded Parameters."""
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_fp8_kv=True)
        weights = _weights_for(donor)
        weights_by_name = dict(weights)

        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_fp8_kv=True)
        loaded = model.load_weights(weights)

        scale_param_names = {
            name for name, _ in model.named_parameters() if name.endswith((".k_scale", ".v_scale"))
        }
        assert len(scale_param_names) == 4  # 2 full-attention layers x (k_scale, v_scale)
        assert scale_param_names <= loaded, (
            "k_scale/v_scale Parameters exist but load_weights did not mark them loaded -- "
            "they would fall through 'mapped not in params_dict: continue' exactly like "
            "before this feature existed"
        )

        all_param_names = {name for name, _ in model.named_parameters()}
        assert loaded == all_param_names, (
            "every Parameter, including k_scale/v_scale, must be reachable from a real "
            "checkpoint tensor name -- assert_all_params_loaded's own contract"
        )

        for layer in model.model.layers:
            if layer.self_attn is None:
                continue
            i = layer.layer_idx
            ckpt_k = weights_by_name[f"model.language_model.layers.{i}.self_attn.k_scale"]
            ckpt_v = weights_by_name[f"model.language_model.layers.{i}.self_attn.v_scale"]
            assert torch.allclose(layer.self_attn.k_scale.data, ckpt_k.to(torch.float32))
            assert torch.allclose(layer.self_attn.v_scale.data, ckpt_v.to(torch.float32))

    def test_checkpoint_without_scale_tensors_fails_loud_not_silent(self) -> None:
        """A checkpoint that never ships k_scale/v_scale (the modelopt
        checkpoint's real situation, per runtime/model/qwen36_model.py's
        module docstring) must not silently run FP8 KV with the
        construction-time default scale of 1.0 -- the caller (load_qwen36_model)
        relies on assert_all_params_loaded to catch exactly this."""
        donor = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_fp8_kv=True)
        weights = _weights_for(donor)
        # Drop every k_scale/v_scale tensor -- reproduces a checkpoint that
        # declares FP8 KV intent but ships no per-layer scale.
        weights = [(n, t) for n, t in weights if not n.endswith((".k_scale", ".v_scale"))]

        model = Qwen36ForCausalLMSelfBuilt(_tiny_config(), max_seq_len=64, enable_fp8_kv=True)
        loaded = model.load_weights(weights)
        all_param_names = {name for name, _ in model.named_parameters()}
        missing = all_param_names - loaded
        assert missing == {
            f"model.layers.{i}.self_attn.{suffix}"
            for i in (1, 3)  # the two full_attention layer indices in _tiny_config
            for suffix in ("k_scale", "v_scale")
        }


class TestSlotPoolFp8KvAllocation:
    """Qwen36SlotPool's k/v pool dtype is what actually produces the
    halved resident bytes the GPU memory measurement reports -- pure
    tensor-shape/dtype bookkeeping, exercisable on CPU exactly like
    tests/test_qwen36_slot_pool.py's existing stub-model tests."""

    _CONV_DIM = 8
    _CONV_K = 4
    _V_HEADS = 2
    _HEAD_DIM = 4
    _KV_HEADS = 2
    _Q_HEADS = 4

    def _stub_model(self, kv_cache_dtypes: dict[int, torch.dtype] | None = None):
        """Same shape as test_qwen36_slot_pool.py's `_stub_model`, plus an
        optional per-layer `kv_cache_dtype` attribute on full-attention
        layers (omitted entirely for layers not in `kv_cache_dtypes`, to
        also cover the getattr(..., dtype) fallback for a stub that
        doesn't know about FP8 KV at all)."""
        kv_cache_dtypes = kv_cache_dtypes or {}
        layer_types = ["full_attention", "linear_attention", "linear_attention"]
        layers = []
        for i, kind in enumerate(layer_types):
            if kind == "linear_attention":
                linear_attn = SimpleNamespace(
                    conv_dim=self._CONV_DIM,
                    conv_kernel_size=self._CONV_K,
                    num_v_heads=self._V_HEADS,
                    head_k_dim=self._HEAD_DIM,
                    head_v_dim=self._HEAD_DIM,
                )
                self_attn = None
            else:
                linear_attn = None
                kwargs = dict(
                    num_kv_heads=self._KV_HEADS, head_dim=self._HEAD_DIM, num_heads=self._Q_HEADS
                )
                if i in kv_cache_dtypes:
                    kwargs["kv_cache_dtype"] = kv_cache_dtypes[i]
                self_attn = SimpleNamespace(**kwargs)
            layers.append(
                SimpleNamespace(
                    layer_idx=i, layer_type=kind, linear_attn=linear_attn, self_attn=self_attn
                )
            )
        return SimpleNamespace(model=SimpleNamespace(layers=layers))

    def _pool(self, model) -> Qwen36SlotPool:
        return Qwen36SlotPool(
            model, num_slots=2, max_seq_len=128, device="cpu", dtype=torch.float32
        )

    def test_fp8_kv_layer_gets_an_fp8_pool(self) -> None:
        pool = self._pool(self._stub_model({0: torch.float8_e4m3fn}))
        assert pool.k_pools[0].dtype == torch.float8_e4m3fn
        assert pool.v_pools[0].dtype == torch.float8_e4m3fn
        assert pool._kv_dtype == torch.float8_e4m3fn

    def test_fp8_kv_pool_uses_exactly_half_the_bytes_of_bf16(self) -> None:
        bf16_pool = self._pool(self._stub_model({0: torch.bfloat16}))
        fp8_pool = self._pool(self._stub_model({0: torch.float8_e4m3fn}))
        assert fp8_pool.geometry.kv_bytes_per_slot * 2 == bf16_pool.geometry.kv_bytes_per_slot

    def test_stub_without_kv_cache_dtype_falls_back_to_pool_dtype(self) -> None:
        """A layer object that doesn't expose kv_cache_dtype at all (the
        exact shape tests/test_qwen36_slot_pool.py's own stub uses) must
        keep working exactly as it did before this feature existed."""
        pool = self._pool(self._stub_model({}))  # no layer declares kv_cache_dtype
        assert pool.k_pools[0].dtype == torch.float32
        assert pool._kv_dtype == torch.float32

    def test_mixed_kv_dtype_across_full_attention_layers_is_refused(self) -> None:
        """One shared decode driver per batch size needs one KV dtype for
        the whole step -- a checkpoint/config that somehow produced two
        different per-layer KV dtypes must fail at construction, not
        silently drive an FP8 pool with a BF16-shaped kernel call (or vice
        versa)."""
        model = self._stub_model({0: torch.float8_e4m3fn})
        # Add a second full-attention layer with a different KV dtype.
        second = SimpleNamespace(
            layer_idx=3,
            layer_type="full_attention",
            linear_attn=None,
            self_attn=SimpleNamespace(
                num_kv_heads=self._KV_HEADS,
                head_dim=self._HEAD_DIM,
                num_heads=self._Q_HEADS,
                kv_cache_dtype=torch.bfloat16,
            ),
        )
        model.model.layers.append(second)
        with pytest.raises(ValueError, match="KV cache dtype"):
            self._pool(model)

    def test_driver_kwargs_reports_the_pools_own_kv_dtype(self) -> None:
        pool = self._pool(self._stub_model({0: torch.float8_e4m3fn}))
        kwargs = pool._driver_kwargs()
        assert kwargs["kv_dtype"] == torch.float8_e4m3fn
        # The compute dtype (query/output) is unaffected -- still the
        # pool's own `dtype`, never the KV dtype.
        assert kwargs["dtype"] == torch.float32
