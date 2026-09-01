"""FN3: profile graph-mode decode: replay cost vs PLE prelude vs gaps."""

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
    FlashNextGraphEngine,
    load_flashnext_model,
    new_session,
    ple_prelude,
    prepare_graph_buffers,
)

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")


def main() -> None:
    model = load_flashnext_model(CKPT, "cuda", progress=lambda d, t: None,
                                   ple_cache_rows=8_000_000)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    ids = tok.encode("The capital of France is")
    sess = new_session(model, "cuda")
    prepare_graph_buffers(model, sess, "cuda", max_seq=4096)
    engine = FlashNextGraphEngine(model, sess, "cuda")
    engine.capture()
    logits = None
    for t in ids:
        logits = engine.step(int(t))
    nxt = int(logits.argmax())
    for _ in range(5):
        lg = engine.step(nxt)
        nxt = int(lg.argmax())
    torch.cuda.synchronize()

    # phase split: prelude vs replay
    n = 20
    t_pre = 0.0
    t_rep = 0.0
    for _ in range(n):
        torch.cuda.synchronize()
        a = time.time()
        ple_prelude(model, sess, nxt)
        torch.cuda.synchronize()
        b = time.time()
        sess.token_buf.fill_(nxt)
        sess.pos_buf.fill_(sess.pos)
        engine.graph.replay()
        torch.cuda.synchronize()
        c = time.time()
        sess.pos += 1
        nxt = int(engine._logits.argmax())
        t_pre += b - a
        t_rep += c - b
    print(f"over {n} steps: prelude {t_pre / n * 1000:.2f} ms | "
          f"replay {t_rep / n * 1000:.2f} ms | "
          f"total ~{(t_pre + t_rep) / n * 1000:.2f} ms/token "
          f"= {n / (t_pre + t_rep):.2f} tok/s", flush=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            engine.graph.replay()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12), flush=True)


if __name__ == "__main__":
    main()
