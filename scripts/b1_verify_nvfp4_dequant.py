"""B1 step-0 check: is there a live, independent oracle on this machine to
cross-validate ``runtime.loading.modelopt``'s hand-rolled NVFP4 (E2M1)
dequantization (LUT + nibble packing order) against?

**Result of running this (2026-08-02, this exact torch build): no.**
``torch.float4_e2m1fn_x2`` exists as a dtype but its elementwise cast is
not functional in either direction on ``torch==2.13.0a0+gitcf30153`` --
both failure modes are captured below. This was the planned cross-check;
see ``runtime/loading/modelopt.py``'s module docstring for what the
dequant math's correctness rests on instead (the E2M1 value table is
derived from the format spec, not guessed; the packing order is the one
genuinely unverified assumption, flagged there as the first thing to
flip if the full-model smoke test ever looks wrong).

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_nvfp4_dequant.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"imported runtime from {runtime.__file__}, expected under {_ROOT}"
)

import torch  # noqa: E402

from runtime.loading.modelopt import unpack_nvfp4_to_fp32  # noqa: E402

# Each direction is run in its own subprocess: a device-side assert (as
# seen below) leaves the CUDA context unusable for the rest of the
# process, so probing both directions in-process would make the second
# probe's failure spurious (context-corruption, not the thing being
# tested).
_DEQUANT_PROBE = (
    "import torch; b = torch.tensor([0x21,0x67,0x00,0xFF], dtype=torch.uint8, device='cuda'); "
    "v = b.view(torch.float4_e2m1fn_x2); f = v.to(torch.float32); "
    "torch.cuda.synchronize(); print('OK', f)"
)
_QUANT_PROBE = (
    "import torch; x = torch.tensor([0.5,1.0,1.5,-6.0], dtype=torch.float32, device='cuda'); "
    "q = x.to(torch.float4_e2m1fn_x2); torch.cuda.synchronize(); print('OK', q.dtype, q.shape)"
)


def _run_probe(label: str, code: str) -> None:
    print(f"=== {label} ===")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        print(f"RESULT: cast succeeded -> {result.stdout.strip()}")
    else:
        last_line = next(
            (line for line in reversed(result.stderr.splitlines()) if line.strip()), ""
        )
        print(f"RESULT: FAILED (returncode={result.returncode}) -- {last_line}")


def print_lut_and_math_derivation() -> None:
    print("\n=== E2M1 LUT sanity: our decode of every nibble 0x0..0xF ===")
    all_nibbles = torch.arange(16, dtype=torch.uint8, device="cuda").reshape(1, 16)
    decoded = unpack_nvfp4_to_fp32(all_nibbles)
    # unpack_nvfp4_to_fp32 expects packed bytes; feeding raw nibble values
    # 0..15 as whole bytes means high_nibble=0 always, so read the LOW
    # (even-index) output column, which is this byte's actual value.
    low_vals = decoded[0, 0::2].tolist()
    print("nibble -> value:", list(enumerate(low_vals)))
    print(
        "Expected from the E2M1 format definition (1 sign + 2 exp[bias=1] + 1 "
        "mantissa): 0,0.5,1,1.5,2,3,4,6 then the same magnitudes negated."
    )


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    _run_probe("torch.float4_e2m1fn_x2 -> float32 (the planned cross-check)", _DEQUANT_PROBE)
    _run_probe("float32 -> torch.float4_e2m1fn_x2 (the other direction)", _QUANT_PROBE)
    print_lut_and_math_derivation()
    print(
        "\nCONCLUSION: no live independent oracle available on this machine for "
        "the NVFP4 dequant. See runtime/loading/modelopt.py's module docstring "
        "for what this module's correctness rests on instead, and what remains "
        "genuinely unverified (packing order)."
    )


if __name__ == "__main__":
    main()
