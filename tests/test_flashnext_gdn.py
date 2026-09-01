"""Flash-Next GDN bring-up: real RadixArk layer-0 weights through the
parameterized qwen36 GDN (sigmoid-gate family), prefill vs decode state
consistency."""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("Flash-Next GDN bring-up requires an SM120 CUDA device", allow_module_level=True)
if not CKPT.is_dir():
    pytest.skip("RadixArk checkpoint not downloaded", allow_module_level=True)

from runtime.model.qwen36_model import GdnLayerState, Qwen36GatedDeltaNet  # noqa: E402


def _config_dict() -> dict:
    with open(CKPT / "config.json") as f:
        tc = json.load(f)["text_config"]
    return {
        "hidden_size": tc["hidden_size"],
        "linear_num_value_heads": tc["linear_num_value_heads"],
        "linear_num_key_heads": tc["linear_num_key_heads"],
        "linear_key_head_dim": tc["linear_key_head_dim"],
        "linear_value_head_dim": tc["linear_value_head_dim"],
        "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "hidden_act": tc["hidden_act"],
    }


def _weight_map() -> dict[str, str]:
    with open(CKPT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _build_layer0() -> Qwen36GatedDeltaNet:
    from safetensors import safe_open

    weight_map = _weight_map()
    prefix = "model.language_model.layers.0.linear_attn"

    def load(name: str) -> torch.Tensor:
        key = f"{prefix}.{name}"
        with safe_open(str(CKPT / weight_map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    module = Qwen36GatedDeltaNet(_config_dict(), layer_idx=0, quantized={})
    with torch.no_grad():
        qkv = load("in_proj_qkv.weight")
        z = load("in_proj_z.weight")
        module.in_proj_qkvz.weight.copy_(torch.cat([qkv, z], dim=0))
        b = load("in_proj_b.weight")
        a = load("in_proj_a.weight")
        module.in_proj_ba.weight.copy_(torch.cat([b, a], dim=0))
        module.conv1d.weight.copy_(load("conv1d.weight"))
        module.dt_bias.copy_(load("dt_bias"))
        module.A_log.copy_(load("A_log"))
        module.norm.weight.copy_(load("norm.weight"))
        module.out_proj.weight.copy_(load("out_proj.weight"))
    return module.to("cuda", torch.bfloat16)


def _new_state(module: Qwen36GatedDeltaNet) -> GdnLayerState:
    return GdnLayerState(
        conv_state=torch.zeros(
            1,
            module.conv_dim,
            module.conv_kernel_size,
            dtype=torch.bfloat16,
            device="cuda",
        ),
        recurrent_state=torch.zeros(
            1,
            module.num_v_heads,
            module.head_k_dim,
            module.head_v_dim,
            dtype=torch.bfloat16,
            device="cuda",
        ),
    )


def test_layer0_prefill_is_finite():
    module = _build_layer0()
    torch.manual_seed(0)
    x = torch.randn(1, 6, module.hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
    state = _new_state(module)
    out = module(x, state)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (1, 6, module.hidden_size)
    assert torch.isfinite(out.float()).all()
    assert state.has_previous_state


def test_layer0_prefill_decode_state_consistency():
    module = _build_layer0()
    torch.manual_seed(1)
    seq = torch.randn(1, 6, module.hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02

    full_state = _new_state(module)
    full_out = module(seq, full_state)

    step_state = _new_state(module)
    outs = []
    for i in range(6):
        outs.append(module(seq[:, i : i + 1], step_state))
    step_out = torch.cat(outs, dim=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(step_out.float(), full_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        step_state.recurrent_state.float(),
        full_state.recurrent_state.float(),
        rtol=2e-2,
        atol=2e-2,
    )
