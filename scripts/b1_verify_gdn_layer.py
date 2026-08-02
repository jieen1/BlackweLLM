"""B1 step 2a: single GDN (linear_attn) layer, real checkpoint weights,
against HF's own ``Qwen3_5GatedDeltaNet`` (same weights, dequantized the
same way). Covers:

  1. Empirically settling the GDN conv_state buffer length question
     (derived by hand in the B1 handoff notes as "the full kernel size (4),
     not kernel_size-1 (3)" from reading transformers/cache_utils.py --
     this prints the real ``cache.layers[0].conv_states.shape`` from a live
     HF ``DynamicCache`` to confirm or refute that derivation directly).
  2. Prefill (multi-token, fresh state) cosine/max_abs_err between
     ``Qwen36GatedDeltaNet`` and ``Qwen3_5GatedDeltaNet``.
  3. Single-token decode continuation from the prefill's state, same
     comparison.

Uses layer 0's REAL checkpoint tensors (not synthetic weights) -- read
directly via safetensors, not through the full model loader (no need to
construct/load all 64 layers for a single-layer check).

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_gdn_layer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/bot/project/qsr-w-b1")
import runtime  # noqa: E402

assert runtime.__file__.startswith("/home/bot/project/qsr-w-b1"), runtime.__file__

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet  # noqa: E402

from runtime.checkpoints import modelopt_checkpoint_path  # noqa: E402
from runtime.loading.modelopt import dequantize_fp8  # noqa: E402
from runtime.model.qwen36_model import Qwen36GatedDeltaNet  # noqa: E402
from runtime.model_loading import _build_qwen36_model_config  # noqa: E402

# Deliberately modelopt (nvidia), not the standard checkpoint: this script
# calls ``runtime.loading.modelopt.dequantize_fp8`` directly and hardcodes
# ``quantized = {...: "FP8"}`` below, which forces the raw tensors it reads
# through ``ModelOptFP8Linear`` -- correct for modelopt's per-*tensor*
# scalar ``weight_scale`` (float32), but silently WRONG for the standard
# checkpoint's per-*channel* ``weight_scale`` ([out, 1], bfloat16) even
# though both checkpoints happen to name the tensors identically (see
# ``runtime/model/compressed_tensors_linear.py``'s module docstring). Do
# not "fix" this to the standard checkpoint without also switching to
# ``CompressedTensorsFP8ChannelLinear``/``dequantize_fp8_channel``.
MODEL_PATH = modelopt_checkpoint_path()
LAYER_IDX = 0  # a linear_attention layer per layer_types
DEVICE = torch.device("cuda")
# No training anywhere in this runtime; matches LagunaBackend.__init__'s
# same call (runtime/backends/laguna.py:266). Not strictly required for
# GDN (FLA's own ops don't dlpack-export), but kept consistent with the
# attention-layer script this one is paired with.
torch.set_grad_enabled(False)


def load_layer_tensors(model_path: str, layer_idx: int) -> dict[str, torch.Tensor]:
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
    names = [n for n in weight_map if n.startswith(prefix)]
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(weight_map[n], []).append(n)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(Path(model_path) / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                out[n[len(prefix) :]] = f.get_tensor(n).to(DEVICE)
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def compare(name: str, mine: torch.Tensor, ref: torch.Tensor) -> None:
    err = (mine.float() - ref.float()).abs().max().item()
    cos = cosine(mine, ref)
    print(f"[{name}] max_abs_err={err:.6g} cosine={cos:.8f}")


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tensors = load_layer_tensors(MODEL_PATH, LAYER_IDX)
    print("loaded layer tensors:", sorted(tensors.keys()))

    model_config = _build_qwen36_model_config(MODEL_PATH)
    quantized = {
        f"model.language_model.layers.{LAYER_IDX}.linear_attn.in_proj_qkv": "FP8",
        f"model.language_model.layers.{LAYER_IDX}.linear_attn.in_proj_z": "FP8",
        f"model.language_model.layers.{LAYER_IDX}.linear_attn.out_proj": "FP8",
    }

    with DEVICE:
        mine = Qwen36GatedDeltaNet(model_config, LAYER_IDX, quantized).to(DEVICE).to(torch.bfloat16)

    # Load real weights into `mine` via each Linear's own weight_loader,
    # exactly like the full loader does (not a shortcut copy).
    params = dict(mine.named_parameters())
    name_map = {
        "in_proj_qkv.weight": "in_proj_qkv.weight",
        "in_proj_qkv.weight_scale": "in_proj_qkv.weight_scale",
        "in_proj_z.weight": "in_proj_z.weight",
        "in_proj_z.weight_scale": "in_proj_z.weight_scale",
        "out_proj.weight": "out_proj.weight",
        "out_proj.weight_scale": "out_proj.weight_scale",
        "in_proj_a.weight": "in_proj_a.weight",
        "in_proj_b.weight": "in_proj_b.weight",
        "dt_bias": "dt_bias",
        "A_log": "A_log",
        "conv1d.weight": "conv1d.weight",
        "norm.weight": "norm.weight",
    }
    missing = set(name_map) - set(tensors)
    assert not missing, f"checkpoint tensors missing: {missing}"
    for ckpt_name, param_name in name_map.items():
        param = params[param_name]
        param.data.copy_(tensors[ckpt_name].to(param.dtype))

    # HF reference: same config, same (dequantized) weights.
    text_config_dict = dict(model_config)
    text_config_dict.pop("quantization_config", None)
    hf_config = Qwen3_5TextConfig(**text_config_dict)
    hf_gdn = Qwen3_5GatedDeltaNet(hf_config, LAYER_IDX).to(DEVICE).to(torch.bfloat16)
    print("\nHF GDN norm class:", type(hf_gdn.norm).__name__)

    in_proj_qkv_bf16 = dequantize_fp8(
        tensors["in_proj_qkv.weight"], tensors["in_proj_qkv.weight_scale"]
    )
    in_proj_z_bf16 = dequantize_fp8(tensors["in_proj_z.weight"], tensors["in_proj_z.weight_scale"])
    out_proj_bf16 = dequantize_fp8(tensors["out_proj.weight"], tensors["out_proj.weight_scale"])
    hf_gdn.in_proj_qkv.weight.data.copy_(in_proj_qkv_bf16)
    hf_gdn.in_proj_z.weight.data.copy_(in_proj_z_bf16)
    hf_gdn.out_proj.weight.data.copy_(out_proj_bf16)
    hf_gdn.in_proj_a.weight.data.copy_(tensors["in_proj_a.weight"].to(torch.bfloat16))
    hf_gdn.in_proj_b.weight.data.copy_(tensors["in_proj_b.weight"].to(torch.bfloat16))
    hf_gdn.dt_bias.data.copy_(tensors["dt_bias"].to(torch.bfloat16))
    hf_gdn.A_log.data.copy_(tensors["A_log"].to(torch.bfloat16))
    hf_gdn.conv1d.weight.data.copy_(tensors["conv1d.weight"].to(torch.bfloat16))
    if hasattr(hf_gdn.norm, "weight"):
        hf_gdn.norm.weight.data.copy_(tensors["norm.weight"].to(hf_gdn.norm.weight.dtype))

    hidden_size = model_config["hidden_size"]
    torch.manual_seed(42)

    # -- Part 1: settle the conv_state buffer length question empirically --
    print("\n=== conv_state buffer length (empirical, live HF DynamicCache) ===")
    hf_cache = DynamicCache(config=hf_config)
    prefill_len = 10
    x_prefill = torch.randn(1, prefill_len, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    hf_gdn(x_prefill, cache_params=hf_cache, attention_mask=None)
    conv_shape = tuple(hf_cache.layers[LAYER_IDX].conv_states.shape)
    print(f"cache.layers[{LAYER_IDX}].conv_states.shape = {conv_shape}")
    kernel_size = model_config["linear_conv_kernel_dim"]
    match_desc = "MATCHES" if conv_shape[-1] == kernel_size else "does NOT match"
    print(f"conv_kernel_size (config) = {kernel_size} -- buffer last-dim {match_desc} full kernel")

    # -- Part 2: prefill comparison (fresh state, same input) --
    print(f"\n=== prefill (fresh state, T={prefill_len}) ===")
    my_state = mine.new_state(batch=1, device=DEVICE, dtype=torch.bfloat16)
    my_out_prefill = mine(x_prefill, my_state)

    hf_cache2 = DynamicCache(config=hf_config)
    hf_out_prefill, *_ = (hf_gdn(x_prefill, cache_params=hf_cache2, attention_mask=None),)
    compare("prefill", my_out_prefill, hf_out_prefill)

    # -- Part 3: single-token decode continuation --
    print("\n=== decode step (continuation from prefill state) ===")
    x_decode = torch.randn(1, 1, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    my_out_decode = mine(x_decode, my_state)
    hf_out_decode = hf_gdn(x_decode, cache_params=hf_cache2, attention_mask=None)
    compare("decode", my_out_decode, hf_out_decode)

    # -- Part 4: a second decode step, to check state doesn't diverge --
    print("\n=== decode step 2 ===")
    x_decode2 = torch.randn(1, 1, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    my_out_decode2 = mine(x_decode2, my_state)
    hf_out_decode2 = hf_gdn(x_decode2, cache_params=hf_cache2, attention_mask=None)
    compare("decode2", my_out_decode2, hf_out_decode2)

    print(
        "\nrecurrent_state dtype: mine=",
        my_state.recurrent_state.dtype,
        "hf=",
        hf_cache2.layers[LAYER_IDX].recurrent_states.dtype,
    )
    print("conv_state shape: mine=", my_state.conv_state.shape)


if __name__ == "__main__":
    main()
