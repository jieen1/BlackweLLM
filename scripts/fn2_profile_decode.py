"""FN2: profile incremental decode_step to find the dominant costs."""

from __future__ import annotations

import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), runtime.__file__

import torch  # noqa: E402

from runtime.model.flashnext.model import (  # noqa: E402
    decode_step,
    load_flashnext_model,
    new_session,
)

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")


def main() -> None:
    model = load_flashnext_model(CKPT, "cuda", progress=lambda d, t: None)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    ids = tok.encode("The capital of France is")
    sess = new_session(model, "cuda")
    logits = None
    for t in ids:
        logits = decode_step(model, int(t), sess)
    nxt = int(logits.argmax())

    # warm a few decode steps
    for _ in range(5):
        lg = decode_step(model, nxt, sess)
        nxt = int(lg.argmax())
    torch.cuda.synchronize()

    n = 10
    t0 = time.time()
    steps = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            torch.cuda.synchronize()
            s = time.time()
            lg = decode_step(model, nxt, sess)
            torch.cuda.synchronize()
            steps.append(time.time() - s)
            nxt = int(lg.argmax())
    dt = time.time() - t0
    print(f"decode {dt / n * 1000:.1f} ms/token = {n / dt:.2f} tok/s", flush=True)
    print("per-step ms:", [f"{x*1000:.0f}" for x in steps], flush=True)

    print("\n=== top CUDA kernels (self time) ===", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18), flush=True)


if __name__ == "__main__":
    main()
