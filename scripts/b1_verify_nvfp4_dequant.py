"""B1 step-0 check: does ``runtime.loading.modelopt``'s hand-rolled NVFP4
(E2M1) dequantization -- LUT + nibble packing order -- agree with torch's
own, independently-implemented ``float4_e2m1fn_x2`` view+cast?

Why this exists (see runtime/loading/modelopt.py's module docstring): this
environment has no ``modelopt``-aware HF quantizer and no ``nvidia-modelopt``
package, so there is no second, independently-implemented NVFP4 dequantizer
on this machine to diff against -- except torch's own native dtype cast,
which IS independent (it is PyTorch/CUDA's own OCP E2M1 decode, not derived
from this module). This script is that cross-check. It does NOT validate
that modelopt's *export* convention matches torch's *native* convention --
only that the two are self-consistent, which is the strongest check
available without pulling in an external oracle.

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_nvfp4_dequant.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/bot/project/qsr-w-b1")
import runtime  # noqa: E402

assert runtime.__file__.startswith("/home/bot/project/qsr-w-b1"), runtime.__file__

import torch  # noqa: E402

from runtime.loading.modelopt import unpack_nvfp4_to_fp32  # noqa: E402


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Every possible byte value (0..255), reshaped into a [16, 16] matrix so
    # unpack_nvfp4_to_fp32 gets a proper [out, in//2] shape.
    all_bytes = torch.arange(256, dtype=torch.uint8, device=device).reshape(16, 16)

    ours = unpack_nvfp4_to_fp32(all_bytes)  # [16, 32] float32

    native = all_bytes.view(torch.float4_e2m1fn_x2).to(torch.float32)
    print("native view+cast shape:", native.shape, "dtype path: float4_e2m1fn_x2 -> float32")

    if native.shape != ours.shape:
        print(f"RESULT: SHAPE MISMATCH ours={tuple(ours.shape)} native={tuple(native.shape)}")
        return

    max_abs_err = (ours - native).abs().max().item()
    n_mismatch = (ours != native).sum().item()
    print(f"RESULT: max_abs_err={max_abs_err} mismatched_elements={n_mismatch}/{ours.numel()}")
    if n_mismatch:
        mism = (ours != native).nonzero()[:10]
        for idx in mism:
            r, c = idx.tolist()
            byte_val = all_bytes[r, c].item()
            ours_pair = (ours[r, 2 * c].item(), ours[r, 2 * c + 1].item())
            native_pair = (native[r, 2 * c].item(), native[r, 2 * c + 1].item())
            print(
                f"  byte={byte_val:3d} (0x{byte_val:02x}) low_nibble={byte_val & 0xF} "
                f"high_nibble={(byte_val >> 4) & 0xF} ours={ours_pair} native={native_pair}"
            )
    else:
        print("RESULT: EXACT MATCH -- LUT values and nibble packing order both confirmed.")

    # Sanity: is our LUT's positive half exactly the textbook E2M1 table?
    print("\nLUT sanity (byte 0x01, 0x23, 0x67, 0xFE):")
    for b in (0x01, 0x23, 0x67, 0xFE):
        t = torch.tensor([[b]], dtype=torch.uint8, device=device)
        print(f"  0x{b:02x}: ours={unpack_nvfp4_to_fp32(t).tolist()}")


if __name__ == "__main__":
    main()
