"""Memory + throughput measurement for the direct-NVFP4-GEMM fused MLP path
(``work/nvfp4-gemm-20260802``, extended to the compressed-tensors checkpoint
format by ``work/std-model-fuse-20260803``): does removing the BF16 dequant
cache actually shrink resident memory, and what does eager (no CUDA graph)
greedy decode throughput look like?

External ``nvidia-smi`` polling (not ``torch.cuda.*`` counters), matching
``notes/2026-08-02-gpu-memory-audit.md``'s methodology -- so the number is
directly comparable to that note's 27.3 -> 76.1 GiB warmup jump.

``--model-path`` (2026-08-03 follow-up): originally hardcoded to nvidia's
modelopt checkpoint only. Parameterized, default UNCHANGED (still nvidia's
checkpoint), so this stays the exact same measurement code either checkpoint
runs through -- required for the unsloth (compressed-tensors) checkpoint's
throughput number to be comparable to nvidia's 6.547 tok/s at all (a
different script/methodology would not be).

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"-- rerun with PYTHONPATH={_ROOT}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.checkpoints import modelopt_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

# Default deliberately stays nvidia's modelopt checkpoint -- NOT because
# this measurement code is format-specific (it isn't; ``--model-path``
# below already runs the standard checkpoint through the identical code
# path), but because the default exists specifically to reproduce nvidia's
# own historical 6.547 tok/s figure on demand (see the module docstring
# above). Changing the default would silently break that reproducibility
# for anyone who omits ``--model-path``. Pass ``--model-path`` explicitly
# to grade the standard checkpoint instead.
DEFAULT_MODEL_PATH = modelopt_checkpoint_path()
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)


def nvidia_smi_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()
    model_path = args.model_path

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    print("model_path:", model_path)
    before_load = nvidia_smi_used_mib()
    print(f"nvidia-smi before load: {before_load} MiB")

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    t0 = time.perf_counter()
    model = load_qwen36_model(model_path, device=DEVICE, max_seq_len=512, enable_mtp=False)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    after_load = nvidia_smi_used_mib()
    print(f"nvidia-smi after load: {after_load} MiB (+{after_load - before_load} MiB)")

    prompt = "Once upon a time, in a small village near the mountains,"
    prompt_ids = tok(prompt, return_tensors=None)["input_ids"]

    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    hidden = model(ids, state)
    logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
    token = int(logits.argmax().item())

    after_prefill = nvidia_smi_used_mib()
    print(f"nvidia-smi after prefill (1 fwd, {len(prompt_ids)} tok): {after_prefill} MiB")

    # Warm every decode-shape kernel once (cutlass DSL JIT-compiles per
    # unique shape on first call -- exclude that from the timed loop).
    step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
    hidden = model(step, state)
    logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
    token = int(logits.argmax().item())
    torch.cuda.synchronize()
    after_warm_decode = nvidia_smi_used_mib()
    print(f"nvidia-smi after 1 warm decode step: {after_warm_decode} MiB")

    n_decode = 30
    t0 = time.perf_counter()
    for _ in range(n_decode):
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        hidden = model(step, state)
        logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
        token = int(logits.argmax().item())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    after_decode = nvidia_smi_used_mib()

    print(f"\n{n_decode} eager decode steps in {dt:.2f}s -> {n_decode / dt:.3f} tok/s")
    print(
        f"nvidia-smi after decode loop: {after_decode} MiB "
        "(peak resident, no growth beyond warmup expected)"
    )
    print(f"\nsummary: before_load={before_load} after_load={after_load} "
          f"after_prefill={after_prefill} after_warm_decode={after_warm_decode} "
          f"after_decode_loop={after_decode} (MiB)")
    print(f"total resident (GiB): {after_decode / 1024:.2f}")


if __name__ == "__main__":
    main()
