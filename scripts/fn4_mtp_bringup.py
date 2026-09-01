"""FN4: MTP draft module bring-up -- load weights, run one draft, check logits.

Validates the FlashNextMTP module (fusion + 1 QSA layer + BF16 MoE) before
wiring the full speculative loop.
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
    FlashNextTextConfig,
    load_flashnext_model,
    new_session,
    prepare_graph_buffers,
)
from runtime.model.flashnext.mtp import load_flashnext_mtp  # noqa: E402
from runtime.model.flashnext.qsa import QsaDecodeAttention  # noqa: E402

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")
MAX_SEQ = 4096


def main() -> None:
    cfg = FlashNextTextConfig.from_checkpoint(CKPT)
    t0 = time.time()
    model = load_flashnext_model(CKPT, "cuda", progress=lambda d, t: None)
    print(f"[{time.time() - t0:.1f}s] main model loaded", flush=True)
    mtp = load_flashnext_mtp(CKPT, cfg, model, "cuda")
    print(f"[{time.time() - t0:.1f}s] mtp loaded", flush=True)

    # MTP QSA pools + decode attention
    attn = mtp.attn
    mtp.qsa_pad = mtp.indexer.block_topk * mtp.indexer.compress_ratio
    mtp.decode_attn = QsaDecodeAttention(attn, mtp.qsa_pad)
    dev = "cuda"
    mtp_k_pool = torch.zeros(MAX_SEQ, attn.num_kv_heads, attn.head_dim,
                             dtype=torch.bfloat16, device=dev)
    mtp_v_pool = torch.zeros_like(mtp_k_pool)
    mtp_idx_k_pool = torch.zeros(MAX_SEQ, mtp.indexer.head_dim,
                                 dtype=torch.bfloat16, device=dev)

    class _S:
        pass

    sess = _S()
    sess.mtp_k_pool = mtp_k_pool
    sess.mtp_v_pool = mtp_v_pool
    sess.mtp_idx_k_pool = mtp_idx_k_pool

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT))

    # get main-model hc_hidden for a prompt via the graph engine
    mks = new_session(model, "cuda")
    prepare_graph_buffers(model, mks, "cuda", max_seq=MAX_SEQ)
    mks.want_hc_hidden = True
    eng = FlashNextGraphEngine(model, mks, "cuda")
    eng.capture()
    ids = tok.encode("The capital of France is")
    logits = None
    for t in ids:
        logits = eng.step(int(t))
    torch.cuda.synchronize()
    main_next = int(logits.argmax())
    hc_hidden = mks.hc_hidden_buf.clone()
    print(f"main model next token: {main_next} "
          f"{tok.decode([main_next])!r}", flush=True)
    print(f"hc_hidden shape {tuple(hc_hidden.shape)} "
          f"finite={torch.isfinite(hc_hidden.float()).all().item()}", flush=True)

    # MTP draft: predict the token AFTER main_next
    pos_mtp = 0
    embeds = model.embed_tokens(
        torch.tensor([main_next], dtype=torch.long, device=dev)
    )
    positions = torch.tensor([pos_mtp], dtype=torch.long, device=dev)
    draft = mtp.forward(embeds, hc_hidden.unsqueeze(0), positions, sess)
    draft_logits = model.lm_head(draft.squeeze(0).float()).float()
    torch.cuda.synchronize()
    mtp_next = int(draft_logits.argmax())
    print(f"mtp logits shape {tuple(draft_logits.shape)} "
          f"finite={torch.isfinite(draft_logits).all().item()}", flush=True)
    print(f"mtp drafted next token: {mtp_next} {tok.decode([mtp_next])!r}", flush=True)
    print("MTP BRING-UP OK", flush=True)


if __name__ == "__main__":
    main()
