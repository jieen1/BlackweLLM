"""B1 step 2b: single full-attention layer, real checkpoint weights,
against HF's own ``Qwen3_5Attention`` (same weights, dequantized the same
way). Exercises the sparkinfer paged-attention path
(``Qwen36Attention``/``Qwen36PagedAttentionCache``) for both prefill
("extend") and single-token decode.

Uses layer 3's REAL checkpoint tensors (the first full_attention layer per
``layer_types``), read directly via safetensors -- no need to construct
all 64 layers for a single-layer check.

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_attention_layer.py
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
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention  # noqa: E402

from runtime.checkpoints import modelopt_checkpoint_path  # noqa: E402
from runtime.kernels.rope import compute_cos_sin_cache_default  # noqa: E402
from runtime.loading.modelopt import dequantize_fp8  # noqa: E402
from runtime.model.qwen36_model import Qwen36Attention  # noqa: E402
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
LAYER_IDX = 3  # first full_attention layer per layer_types
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 64
# Required: sparkinfer's paged-attention kernel exports its inputs via
# __dlpack__, which torch refuses for any tensor requiring grad. No
# training anywhere in this runtime -- matches LagunaBackend.__init__
# (runtime/backends/laguna.py:266).
torch.set_grad_enabled(False)


def load_layer_tensors(model_path: str, layer_idx: int) -> dict[str, torch.Tensor]:
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.self_attn."
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


def make_causal_mask(seq_q: int, total_kv: int, past_len: int, dtype, device) -> torch.Tensor:
    q_pos = torch.arange(past_len, past_len + seq_q, device=device).unsqueeze(-1)
    kv_pos = torch.arange(total_kv, device=device).unsqueeze(0)
    allowed = kv_pos <= q_pos
    mask = torch.zeros(seq_q, total_kv, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_q, total_kv)


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tensors = load_layer_tensors(MODEL_PATH, LAYER_IDX)
    print("loaded layer tensors:", sorted(tensors.keys()))

    model_config = _build_qwen36_model_config(MODEL_PATH)
    quantized = {
        f"model.language_model.layers.{LAYER_IDX}.self_attn.q_proj": "FP8",
        f"model.language_model.layers.{LAYER_IDX}.self_attn.k_proj": "FP8",
        f"model.language_model.layers.{LAYER_IDX}.self_attn.v_proj": "FP8",
        f"model.language_model.layers.{LAYER_IDX}.self_attn.o_proj": "FP8",
    }

    with DEVICE:
        mine = Qwen36Attention(model_config, LAYER_IDX, quantized, max_seq_len=MAX_SEQ_LEN)
        mine = mine.to(DEVICE).to(torch.bfloat16)

    params = dict(mine.named_parameters())
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        params[f"{proj}.weight"].data.copy_(tensors[f"{proj}.weight"])
        params[f"{proj}.weight_scale"].data.copy_(tensors[f"{proj}.weight_scale"])
    params["q_norm.weight"].data.copy_(tensors["q_norm.weight"].to(torch.bfloat16))
    params["k_norm.weight"].data.copy_(tensors["k_norm.weight"].to(torch.bfloat16))

    text_config_dict = dict(model_config)
    text_config_dict.pop("quantization_config", None)
    hf_config = Qwen3_5TextConfig(**text_config_dict)
    hf_config._attn_implementation = "eager"
    hf_attn = Qwen3_5Attention(hf_config, LAYER_IDX).to(DEVICE).to(torch.bfloat16)

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w_bf16 = dequantize_fp8(tensors[f"{proj}.weight"], tensors[f"{proj}.weight_scale"])
        getattr(hf_attn, proj).weight.data.copy_(w_bf16)
    hf_attn.q_norm.weight.data.copy_(tensors["q_norm.weight"].to(torch.bfloat16))
    hf_attn.k_norm.weight.data.copy_(tensors["k_norm.weight"].to(torch.bfloat16))

    hidden_size = model_config["hidden_size"]
    rope_params = model_config["rope_parameters"]
    rotary_dim = int(model_config["head_dim"] * rope_params["partial_rotary_factor"])
    cos_sin_cache = compute_cos_sin_cache_default(
        rotary_dim, model_config["max_position_embeddings"], float(rope_params["rope_theta"]),
        torch.bfloat16, device=DEVICE,
    )

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    from runtime.model.qwen36_model import Qwen36PagedAttentionCache

    hf_rope = Qwen3_5TextRotaryEmbedding(hf_config).to(DEVICE)

    torch.manual_seed(7)
    prefill_len = 10
    x_prefill = torch.randn(1, prefill_len, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1

    print(f"\n=== prefill (extend mode, T={prefill_len}) ===")
    my_cache = Qwen36PagedAttentionCache(
        num_kv_heads=model_config["num_key_value_heads"], head_dim=model_config["head_dim"],
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16, device=DEVICE,
    )
    my_positions = torch.arange(0, prefill_len, device=DEVICE, dtype=torch.long)
    my_out_prefill = mine(x_prefill, my_positions, cos_sin_cache, my_cache)

    hf_cache = DynamicCache(config=hf_config)
    pos_ids = my_positions.unsqueeze(0)
    cos, sin = hf_rope(x_prefill, pos_ids)
    attn_mask = make_causal_mask(prefill_len, prefill_len, 0, torch.bfloat16, DEVICE)
    hf_out_prefill, _ = hf_attn(
        x_prefill, position_embeddings=(cos, sin),
        attention_mask=attn_mask, past_key_values=hf_cache,
    )
    compare("prefill", my_out_prefill, hf_out_prefill)

    print("\n=== decode step (continuation) ===")
    x_decode = torch.randn(1, 1, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    my_positions_d = torch.arange(prefill_len, prefill_len + 1, device=DEVICE, dtype=torch.long)
    my_out_decode = mine(x_decode, my_positions_d, cos_sin_cache, my_cache)

    pos_ids_d = my_positions_d.unsqueeze(0)
    cos_d, sin_d = hf_rope(x_decode, pos_ids_d)
    attn_mask_d = make_causal_mask(1, prefill_len + 1, prefill_len, torch.bfloat16, DEVICE)
    hf_out_decode, _ = hf_attn(
        x_decode, position_embeddings=(cos_d, sin_d),
        attention_mask=attn_mask_d, past_key_values=hf_cache,
    )
    compare("decode", my_out_decode, hf_out_decode)

    print("\n=== decode step 2 ===")
    x_decode2 = torch.randn(1, 1, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    my_positions_d2 = torch.arange(
        prefill_len + 1, prefill_len + 2, device=DEVICE, dtype=torch.long
    )
    my_out_decode2 = mine(x_decode2, my_positions_d2, cos_sin_cache, my_cache)

    pos_ids_d2 = my_positions_d2.unsqueeze(0)
    cos_d2, sin_d2 = hf_rope(x_decode2, pos_ids_d2)
    attn_mask_d2 = make_causal_mask(1, prefill_len + 2, prefill_len + 1, torch.bfloat16, DEVICE)
    hf_out_decode2, _ = hf_attn(
        x_decode2, position_embeddings=(cos_d2, sin_d2), attention_mask=attn_mask_d2,
        past_key_values=hf_cache,
    )
    compare("decode2", my_out_decode2, hf_out_decode2)


if __name__ == "__main__":
    main()
