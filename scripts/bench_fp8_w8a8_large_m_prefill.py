"""Large-M FP8 W8A8 prefill-shape microbench: self-owned .so vs torch.

Attributes the prefill FP8-dense GEMM cost of one 4096-token prefill across
the checkpoint's real (N, K) shapes, and measures the self-owned CUTLASS
port against torch._scaled_mm at each shape.  Pure tensor bench -- loads no
model weights beyond the enumerated shape metadata, and runs under the
single-job GPU lock.
"""

from __future__ import annotations

import ctypes
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch

from runtime.checkpoints import standard_checkpoint_path
from runtime.fp8_w8a8 import ABI_VERSION, artifact_paths


def enumerate_fp8_shapes() -> Counter:
    index = json.loads(
        (Path(standard_checkpoint_path()) / "model.safetensors.index.json").read_text()
    )["weight_map"]
    bases = {
        re.sub(r"\.(weight_scale|weight_packed|weight_global_scale|weight)$", "", n)
        for n in index
    }
    # FP8-channel = has weight_scale but no packed NVFP4 storage.
    fp8_bases = {
        b
        for b in bases
        if f"{b}.weight_scale" in index and f"{b}.weight_packed" not in index
    }
    # (N, K) per kind, from hidden=5120 / head_dim=256 / 24 q-heads /
    # 4 kv-heads / intermediate=17408; linear_attn qkv packs q+k+v+gates.
    shapes: dict[str, tuple[int, int]] = {
        "model.language_model.layers.#.linear_attn.in_proj_qkv": (12288, 5120),
        "model.language_model.layers.#.linear_attn.in_proj_z": (5120, 5120),
        "model.language_model.layers.#.linear_attn.out_proj": (5120, 5120),
        "model.language_model.layers.#.mlp.gate_proj": (17408, 5120),
        "model.language_model.layers.#.mlp.up_proj": (17408, 5120),
        "model.language_model.layers.#.mlp.down_proj": (5120, 17408),
        "model.language_model.layers.#.self_attn.q_proj": (6144, 5120),
        "model.language_model.layers.#.self_attn.k_proj": (1024, 5120),
        "model.language_model.layers.#.self_attn.v_proj": (1024, 5120),
        "model.language_model.layers.#.self_attn.o_proj": (5120, 6144),
        "lm_head": (248320, 5120),
    }
    result = Counter()
    for base in fp8_bases:
        kind = re.sub(r"\d+", "#", base)
        if kind not in shapes:
            raise SystemExit(f"unknown FP8 linear kind: {kind}")
        result[shapes[kind]] += 1
    return result


def load_native() -> ctypes.CDLL:
    library_path, _manifest = artifact_paths()
    lib = ctypes.CDLL(str(library_path))
    abi = lib.qsr_fp8_w8a8_abi_version
    abi.argtypes, abi.restype = [], ctypes.c_int
    assert abi() == ABI_VERSION
    for name, argtypes in (
        ("qsr_fp8_w8a8_workspace_size_sm120",
         [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int, ctypes.c_int,
          ctypes.c_int, ctypes.c_int]),
        ("qsr_fp8_w8a8_scaled_mm_sm120",
         [ctypes.c_void_p] * 6 + [ctypes.c_size_t] + [ctypes.c_int] * 4 +
         [ctypes.c_void_p]),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, ctypes.c_int
    return lib


def bench_native(lib, x_fp8, w_t, a_scale, b_scale, m, n, k, iters=50) -> float:
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    ws_size = ctypes.c_size_t()
    status = lib.qsr_fp8_w8a8_workspace_size_sm120(
        ctypes.byref(ws_size), m, n, k, 0)
    assert status == 0, status
    workspace = torch.empty(ws_size.value, dtype=torch.uint8, device="cuda")
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def run() -> None:
        status = lib.qsr_fp8_w8a8_scaled_mm_sm120(
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(x_fp8.data_ptr()),
            ctypes.c_void_p(w_t.data_ptr()), ctypes.c_void_p(a_scale.data_ptr()),
            ctypes.c_void_p(b_scale.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()) if workspace.numel() else None,
            workspace.numel(), m, n, k, 0, stream)
        assert status == 0, status

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_torch(x_fp8, w_t, a_scale, b_scale, iters=50) -> float:
    def run():
        return torch._scaled_mm(
            x_fp8, w_t, scale_a=a_scale, scale_b=b_scale,
            out_dtype=torch.bfloat16)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    torch.cuda.init()
    shapes = enumerate_fp8_shapes()
    lib = load_native()
    # Production prefill projects only the per-request final rows through the
    # 248k-wide lm_head (B<=4 at c=4), not all 4096 rows.
    row_override = {(248320, 5120): 4}
    total_native = total_torch = 0.0
    print(f"{'shape':>16} {'M':>5} {'cnt':>4} {'native ms':>10} {'torch ms':>10} "
          f"{'ratio':>6} {'native TF/s':>12} {'prefill ms':>10}")
    for (n, k), count in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
        m = row_override.get((n, k), 4096)
        x = torch.randn(m, k, device="cuda").clamp(-4, 4).to(torch.bfloat16)
        amax = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-8)
        a_scale = (amax / 448.0).clamp_min(1.0 / (448.0 * 512.0))
        x_fp8 = (x.float() / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        a_scale = a_scale.reshape(m, 1)
        w = torch.randn(n, k, device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
        w_t = w.t()
        b_scale_row = torch.rand(1, n, device="cuda").add(0.5)
        b_scale_col = b_scale_row.reshape(n, 1).contiguous()
        ms_native = bench_native(lib, x_fp8, w_t, a_scale.reshape(m, 1),
                                 b_scale_col, m, n, k)
        ms_torch = bench_torch(x_fp8, w_t, a_scale, b_scale_row)
        tflops = 2.0 * m * n * k / (ms_native * 1e-3) / 1e12
        prefill = ms_native * count
        total_native += prefill
        total_torch += ms_torch * count
        print(f"({n:>6},{k:>5}) {m:>5} {count:>4} {ms_native:>10.3f} {ms_torch:>10.3f} "
              f"{ms_torch / ms_native:>6.2f} {tflops:>12.1f} {prefill:>10.1f}")
    print(f"\none 4096-token prefill FP8-dense GEMM: native {total_native:.1f} ms "
          f"vs torch {total_torch:.1f} ms (x{total_torch / total_native:.2f})")


if __name__ == "__main__":
    main()
