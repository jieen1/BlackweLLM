"""B3 CUDA-Graph follow-up: GPU bit-exactness proof for
``runtime.backends.qwen36_mtp_cudagraph``'s anchor, draft, and verify graphs.

**What this proves, and why it needs a real GPU + real checkpoint**: these
graphs replace EXISTING, already-verified eager code paths
(``Qwen36MTPEngine.round``'s anchor-advance forward, chained ``mtp_step``
calls, and K-token target verify) with a
DIFFERENT kernel code path (``decode_batch()`` against a pooled KV tensor,
vs ``forward()``/``mtp_step()`` against a standalone
``Qwen36PagedAttentionCache``). Same weights, same math on paper -- but a
different kernel schedule, same class of claim
``scripts/b2_verify_serving.py`` already established for the MAIN decode
path ("batched decode is bit-exact against B1's eager path") and which
that script's own docstring says explicitly is "a claim about a real
checkpoint on a real GPU", not something a CPU/stub test can make. This
script makes the SAME kind of claim for all MTP graphs specifically
-- nothing before this landing exercised ``Qwen36Attention.decode_batch``
against MTP's own head/cache at all.

**Method**: run ``NUM_ROUNDS`` real MTP rounds (prefill bootstrap +
``round()`` calls) on TWO DIFFERENT slots of the SAME loaded model/backend
-- one with the captured graphs live (the default once ``enable_mtp``
succeeds on a real GPU), one with them forced off
(``engine._anchor_cg = engine._draft_cg = engine._verify_cg = None`` for the duration, which
routes every call through the pre-existing eager path unchanged). Greedy
decoding is deterministic given fixed weights and fixed inputs, so if both
graphs replay bit-exactly, the two slots' full per-round traces
(committed tokens, num_accepted, next_anchor, next_draft_tokens) must be
IDENTICAL, not merely "close" -- any float noise from a genuinely different
kernel schedule would first show up as an argmax flip in ``committed``,
which this comparison catches structurally (a list-equality check), not
just numerically.

Two different slots (not two separate model loads) so this only pays the
checkpoint's cold-load cost once; the two runs are still fully independent
(``backend.reset_slot`` clears one slot's own KV/GDN/MTP-cache state, never
touching the other's).

Not a pytest (needs the GPU lock + a real checkpoint on disk), matching
this repo's convention (``verify_w4a16_cuda_graph_scratch_rootcause.py``):

    ~/.venvs/vllm/bin/python scripts/verify_qwen36_mtp_cuda_graph_bit_exact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"-- rerun with PYTHONPATH including {_ROOT}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

DEVICE = "cuda"
MODEL_PATH = standard_checkpoint_path()
MAX_SEQ_LEN = 512
NUM_SLOTS = 2
K = 4
NUM_ROUNDS = 8
PROMPT = "Once upon a time, in a small village near the mountains,"


def _run(
    backend: Qwen36Backend, slot: int, *, use_graphs: bool, prompt_ids: list[int]
) -> list[dict]:
    engine = backend._mtp
    backend.reset_slot(slot)
    saved = (engine._anchor_cg, engine._draft_cg, engine._verify_cg)
    if not use_graphs:
        engine._anchor_cg = None
        engine._draft_cg = None
        engine._verify_cg = None
    try:
        state = backend.prefill_chunked_begin([slot], [prompt_ids], params_per_slot={})
        anchor = state.result[slot]["anchor"]
        drafts = state.result[slot]["draft_tokens"]
        trace: list[dict] = []
        for _ in range(NUM_ROUNDS):
            result = engine.round(slot, anchor, list(drafts))
            trace.append(
                {
                    "committed": list(result["committed"]),
                    "num_accepted": result["num_accepted"],
                    "next_anchor": result["next_anchor"],
                    "next_draft_tokens": list(result["next_draft_tokens"]),
                }
            )
            anchor = result["next_anchor"]
            drafts = result["next_draft_tokens"]
        return trace
    finally:
        engine._anchor_cg, engine._draft_cg, engine._verify_cg = saved


def main() -> None:
    print(f"checkpoint: {MODEL_PATH}")
    model = load_qwen36_model(
        MODEL_PATH,
        device=DEVICE,
        dtype=torch.bfloat16,
        max_seq_len=MAX_SEQ_LEN,
        enable_mtp=True,
    )
    backend = Qwen36Backend(
        model,
        num_slots=NUM_SLOTS,
        max_seq_len=MAX_SEQ_LEN,
        block_size=64,
        device=DEVICE,
        dtype=torch.bfloat16,
        enable_prefix_cache=False,
    )
    graph_batch = backend.capture_decode_cuda_graph()
    print(f"backbone decode CUDA Graph captured up to batch={graph_batch}")
    backend.enable_mtp(num_speculative_tokens=K, enable_resync=False)
    engine = backend._mtp
    print("MTP cg_status:", engine.cg_status)
    assert engine.cg_status.get("anchor") == "captured", (
        f"anchor CUDA Graph did not capture: {engine.cg_status}"
    )
    assert engine.cg_status.get("draft") == "captured", (
        f"draft CUDA Graph did not capture: {engine.cg_status}"
    )
    assert engine.cg_status.get("verify") == "captured", (
        f"verify CUDA Graph did not capture: {engine.cg_status}"
    )
    assert engine._anchor_cg is not None
    assert engine._draft_cg is not None
    assert engine._verify_cg is not None

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_ids = tok(PROMPT, return_tensors=None)["input_ids"]
    print(f"prompt_ids: {len(prompt_ids)} tokens")

    trace_graph = _run(backend, 0, use_graphs=True, prompt_ids=prompt_ids)
    trace_eager = _run(backend, 1, use_graphs=False, prompt_ids=prompt_ids)

    match = trace_graph == trace_eager
    print("MATCH" if match else "MISMATCH")
    if not match:
        for i, (g, e) in enumerate(zip(trace_graph, trace_eager)):
            if g != e:
                print(f"  round {i}: graph={g}")
                print(f"  round {i}: eager={e}")
    for i, t in enumerate(trace_graph):
        print(f"  round {i}: accepted={t['num_accepted']}/{K} committed={t['committed']}")

    assert match, (
        "Qwen36MTP anchor/draft/verify CUDA Graph replay diverged from "
        "eager Qwen36MTPEngine.round for identical inputs -- NOT bit-exact"
    )
    print(f"PASS: MTP CUDA-Graph replay is bit-exact vs eager for {NUM_ROUNDS} rounds, K={K}")


if __name__ == "__main__":
    main()
