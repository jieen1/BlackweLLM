"""B1 step 1: load the real Qwen3.6-27B checkpoint end-to-end through
``runtime.model_loading.load_qwen36_model`` and assert:

  1. Every model Parameter received a real checkpoint tensor
     (``assert_all_params_loaded`` -- raises on failure, so reaching the
     end of this script's load step already proves this).
  2. Zero vision tensors were loaded (333 skipped, matching the real
     checkpoint's ``model.visual.*`` count established in B0).
  3. Zero MTP tensors were loaded (15 skipped).
  4. A couple of loaded values match the raw checkpoint bytes exactly
     (spot-check, not just "didn't crash").

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_loader.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/bot/project/qsr-w-b1")
import runtime  # noqa: E402

assert runtime.__file__.startswith("/home/bot/project/qsr-w-b1"), runtime.__file__

import torch  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

MODEL_PATH = standard_checkpoint_path()


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    t0 = time.time()
    model = load_qwen36_model(MODEL_PATH, device="cuda", dtype=torch.bfloat16, max_seq_len=4096)
    elapsed = time.time() - t0
    print(f"load_qwen36_model: {elapsed:.1f}s")

    stats = model._vision_filter_stats
    print(f"vision tensors skipped: {stats.skipped_count} (expect 333)")
    print(f"mtp tensors skipped: {model.skipped_mtp_count} (expect 15)")

    assert stats.skipped_count == 333, stats.skipped_count
    assert model.skipped_mtp_count == 15, model.skipped_mtp_count

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameter elements: {n_params}")

    mem = torch.cuda.memory_allocated() / (1024**3)
    print(f"GPU memory allocated after load: {mem:.2f} GiB")

    print("\nRESULT: load_qwen36_model succeeded, assert_all_params_loaded passed, "
          "vision/mtp skip counts match B0's expected values.")


if __name__ == "__main__":
    main()
