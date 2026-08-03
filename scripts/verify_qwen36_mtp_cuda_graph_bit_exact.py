"""⚠️ 本脚本的判据是错的，不要拿它当放行条件。

**"图重放对 eager 逐位相同"这件事，本仓库早已证明做不到**，与 MTP 无关：
``notes/2026-08-02-eager-verify-cg-verify-divergence.md`` 的结论是
**CG 冻结在 1 个 KV 分块、eager 用 4~16 个**，而"块数不一致本身不是错误，
两种块数在注意力算子这一层都算对了"（cos ≥ 0.999997，kv_len=64/400/500 全成立）。
全模型 logits 上看到的 argmax 翻转是**近似平局位置的翻转**，不是某条路径算错了。

2026-08-03 的实测复现了同一机制：跑到第 4 轮（首次全接受）结束时，
``accepts``/``anchor``/``live_col``/committed token **两条路径完全一致**，
而 48 个 GDN 层里 45 个的状态差在 **conv 3.1e-02 / recurrent 2e-03**——
**bf16 精度尺度，不是写错行的量级**（写错行会给出完全不同的值）。
到第 5 轮累积漂移把一个近似平局翻了过去，于是本脚本判失败。

**这个失败三次误导了修复方向**：先后被归因为"地址被烤死"和"anchor 写进候选行"，
两个假设都按历史代码认真查证过、也都改过，而签名分毫未变——**因为没有东西可修，是尺子错了。**

**正确的判据是两条**，都已在本仓库建立：

1. **B1-R 的 gap-error**（``docs/b1-correctness-criterion.md``，对照已校准的 bar）
   —— W4A4 / FP8 W8A8 / FP8 KV 都是这么判的。这个判据存在的理由，
   恰恰就是 bit-exactness 被证明达不到。
2. **投机 vs 非投机在同一条路径内逐 token 一致**
   （``scripts/b3_mtp_e2e_acceptance_throughput.py``）——这才是投机解码的正确性定义。

本脚本保留有一个用处、且只有这一个：**确认 ``cg_status`` 三项都是 ``captured``**，
即捕获没有静默退化成 eager。它确实抓到过两次真问题（``commit_verify`` 分支导致
一开图就崩、multistep 的连续性要求导致 verify 捕获失败）。**捕获与否是二值的、可判的；
逐位相同不是。**
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
