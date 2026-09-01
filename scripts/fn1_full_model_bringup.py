"""FN1: Flash-Next full-model bring-up on the self-built runtime.

Loads all 48 layers with real RadixArk weights (NVFP4 experts via b12x,
GDN/QSA/hyper-connection/PLE modules from runtime/model/flashnext), runs a
prefill forward, and greedy-extends a few tokens by stateless re-forward.
Milestone gate: finite logits, in-vocab greedy tokens, coherent text.
"""

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
    prepare_graph_buffers,
)

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")


def main() -> None:
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        print(f"[{time.time() - t0:7.1f}s] layers loaded {done}/{total}", flush=True)

    model = load_flashnext_model(CKPT, "cuda", progress=progress)
    print(f"[{time.time() - t0:.1f}s] model loaded; "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    prompt = "The capital of France is"
    ids = tok.encode(prompt)
    print(f"prompt tokens: {ids}", flush=True)

    sess = new_session(model, "cuda")
    prepare_graph_buffers(model, sess, "cuda", max_seq=4096)
    engine = FlashNextGraphEngine(model, sess, "cuda")
    engine.capture()
    print(f"[{time.time() - t0:.1f}s] CUDA graph captured", flush=True)

    logits = None
    for t in ids:
        logits = engine.step(int(t))
    torch.cuda.synchronize()
    print(f"[{time.time() - t0:.1f}s] graph prefill OK; "
          f"logits finite={torch.isfinite(logits).all().item()}", flush=True)

    generated = list(ids)
    nxt = int(logits.argmax())
    generated.append(nxt)
    print(f"step 0: token={nxt} piece={tok.decode([nxt])!r}", flush=True)

    n_gen = 24
    t1 = time.time()
    for step in range(1, n_gen):
        lg = engine.step(nxt)
        torch.cuda.synchronize()
        nxt = int(lg.argmax())
        generated.append(nxt)
        if step < 8:
            print(f"step {step}: token={nxt} piece={tok.decode([nxt])!r}", flush=True)
    dt = time.time() - t1
    tps = (n_gen - 1) / dt if dt > 0 else 0.0
    print(f"FINAL TEXT: {tok.decode(generated)!r}", flush=True)
    print(f"DECODE SPEED: {tps:.2f} tok/s over {n_gen - 1} tokens "
          f"(CUDA graph, no MTP)", flush=True)
    for layer in model.layers:
        if layer.ple is not None:
            tbl = layer.ple.table
            tot = tbl.cache_hits + tbl.cache_misses
            hr = tbl.cache_hits / tot if tot else 0.0
            print(f"PLE cache: {tbl.cache_hits} hits / {tbl.cache_misses} misses "
                  f"= {hr*100:.1f}% hit rate", flush=True)


if __name__ == "__main__":
    main()
