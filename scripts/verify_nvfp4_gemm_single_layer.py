"""Single-layer validation: ModelOptNVFP4Linear's new block-scaled NVFP4 GEMM
forward path (sparkinfer.gemm.blockscaled.mm) vs the legacy BF16-dequant
forward path, on one *real* checkpoint layer (nvidia/Qwen3.6-27B-NVFP4,
layer 5 down_proj/gate_proj -- real quantized weight + block scale + global
scale, not synthetic).

Not a pytest test (needs the GPU lock + a real checkpoint on disk) -- run
manually, one shot, under /tmp/gpu_lock.sh. Reports cosine similarity and
max abs error between the two forward paths across several M (decode-like
M=1 up to prefill-like M=512), matching the task's "single-layer: NVFP4-GEMM
output vs existing BF16 output, report cosine and max_abs_err" requirement.
"""

from __future__ import annotations

import json
import pathlib
import time

import torch
from safetensors import safe_open

from runtime.model.modelopt_linear import ModelOptNVFP4Linear

CKPT = pathlib.Path(
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots"
)
LAYER = 5
DEVICE = "cuda"


def _find_ckpt() -> pathlib.Path:
    snaps = sorted(CKPT.iterdir())
    assert snaps, f"no snapshot under {CKPT}"
    return snaps[0]


def load_proj(ckpt: pathlib.Path, layer: int, proj: str) -> dict[str, torch.Tensor]:
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.{proj}"
    needed = {
        f"{prefix}.weight": None,
        f"{prefix}.weight_scale": None,
        f"{prefix}.weight_scale_2": None,
    }
    shards = {weight_map[k] for k in needed}
    out: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed:
                    out[k.rsplit(".", 1)[-1]] = f.get_tensor(k)
    assert set(out) == {"weight", "weight_scale", "weight_scale_2"}, out.keys()
    return out


def build_layer(ckpt: pathlib.Path, layer: int, proj: str) -> ModelOptNVFP4Linear:
    raw = load_proj(ckpt, layer, proj)
    weight = raw["weight"]
    out_features, packed_in = weight.shape
    in_features = packed_in * 2
    mod = ModelOptNVFP4Linear(in_features, out_features, bias=False)
    mod.weight.data.copy_(weight.to(DEVICE))
    mod.weight_scale.data.copy_(raw["weight_scale"].to(DEVICE))
    mod.weight_scale_2.data.copy_(raw["weight_scale_2"].to(DEVICE))
    mod = mod.to(DEVICE)
    return mod


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    return (a64 @ b64 / (a64.norm() * b64.norm() + 1e-30)).item()


def run_case(mod: ModelOptNVFP4Linear, m: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = (torch.randn(m, mod.input_size, device=DEVICE, dtype=torch.bfloat16) * 0.02).contiguous()

    # legacy BF16-dequant reference (still intact, opt-in only)
    mod._ensure_ready()
    ref = torch.nn.functional.linear(x, mod._weight_bf16, mod.bias)
    mod._weight_bf16 = None  # don't let the reference path leave a resident cache either

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_iters = 20 if m <= 8 else 5
    for _ in range(n_iters):
        out = mod(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters

    max_abs_err = (out.double() - ref.double()).abs().max().item()
    cos = cosine(out, ref)
    ref_norm = ref.double().norm().item()
    rel_err = max_abs_err / (ref.double().abs().max().item() + 1e-30)
    print(
        f"  M={m:4d}  cosine={cos:.6f}  max_abs_err={max_abs_err:.6f}  "
        f"rel_to_max={rel_err:.6f}  ref_norm={ref_norm:.2f}  "
        f"latency={dt * 1e6:.1f}us/call"
    )


def main() -> None:
    ckpt = _find_ckpt()
    print(f"checkpoint: {ckpt}")
    for proj in ("down_proj", "gate_proj", "up_proj"):
        print(f"\n=== layer {LAYER}.{proj} ===")
        mod = build_layer(ckpt, LAYER, proj)
        print(f"  in={mod.input_size} out={mod.output_size}")
        for m in (1, 2, 8, 32, 128, 512):
            run_case(mod, m, seed=1234 + m)
        del mod
        torch.cuda.empty_cache()
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\npeak allocated during this script: {mem:.3f} GiB")


if __name__ == "__main__":
    main()
