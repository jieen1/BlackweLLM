"""FN5: speculative decode with the Flash-Next MTP draft (K=1 greedy).

Cycle: main-graph step produces one real token + hc_hidden; MTP drafts the
token after it; if the main graph's next real token equals the draft, two
tokens are banked for one replay. No rollback needed (greedy bullet
verification; the main model only ever advances).
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
    mtp = load_flashnext_mtp(CKPT, cfg, model, "cuda")
    attn = mtp.attn
    mtp.qsa_pad = mtp.indexer.block_topk * mtp.indexer.compress_ratio
    mtp.decode_attn = QsaDecodeAttention(attn, mtp.qsa_pad)

    class _S:
        pass

    msess = _S()
    msess.mtp_k_pool = torch.zeros(MAX_SEQ, attn.num_kv_heads, attn.head_dim,
                                   dtype=torch.bfloat16, device="cuda")
    msess.mtp_v_pool = torch.zeros_like(msess.mtp_k_pool)
    msess.mtp_idx_k_pool = torch.zeros(MAX_SEQ, mtp.indexer.head_dim,
                                       dtype=torch.bfloat16, device="cuda")
    print(f"[{time.time() - t0:.1f}s] models loaded", flush=True)

    sess = new_session(model, "cuda")
    prepare_graph_buffers(model, sess, "cuda", max_seq=MAX_SEQ)
    sess.want_hc_hidden = True
    eng = FlashNextGraphEngine(model, sess, "cuda")
    eng.capture()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    ids = tok.encode("The capital of France is")
    logits = None
    for t in ids:
        logits = eng.step(int(t))  # advances GDN/QSA/PLE states
    torch.cuda.synchronize()

    generated = list(ids) + [int(logits.argmax())]

    def mtp_sync(token: int) -> torch.Tensor | None:
        """Teacher-force one real (token, hc_hidden) pair through the MTP
        layer (the engine's _sync_real_suffix equivalent) and return its
        logits -- MTP's prediction for the token after ``token``."""
        embeds = model.embed_tokens(
            torch.tensor([token], dtype=torch.long, device="cuda")
        )
        pos_mtp = torch.tensor([msess.mtp_pos], dtype=torch.long, device="cuda")
        draft = mtp.forward(embeds, sess.hc_hidden_buf.unsqueeze(0), pos_mtp, msess)
        msess.mtp_pos += 1
        return model.lm_head(draft.squeeze(0)).float()


    msess.mtp_pos = 0
    # warmup: advance states a few steps
    nxt = generated[-1]
    for _ in range(3):
        lg = eng.step(nxt)
        nxt = int(lg.argmax())
        mtp_sync(nxt)
    torch.cuda.synchronize()

    # Measure MTP-1-step acceptance over N steps: after the main graph
    # produces hc_hidden at position p and samples token a (= position p+1),
    # MTP drafts the token it expects at p+2; acceptance = match with what
    # the main model actually produces there. This is the upper bound of
    # K>=2 speculative gain.
    n_steps = 40
    hits = 0
    tried = 0
    examples = []
    cur = nxt

    for _ in range(n_steps):
        lg = eng.step(cur)
        real = int(lg.argmax())
        # teacher-force cur (with the just-produced hc_hidden) through MTP;
        # MTP's own logits at this position predict `real` -- same-position
        # acceptance, comparable with sglang's accept-rate semantics.
        mlp_out = mtp_sync(cur)
        if mlp_out is not None:
            tried += 1
            mtop1 = int(mlp_out.argmax())
            if mtop1 == real:
                hits += 1
            elif len(examples) < 6:
                mtop = [int(i) for i in torch.topk(mlp_out.float(), 3).indices.tolist()]
                examples.append((mtop, real))
        cur = real
    torch.cuda.synchronize()
    print(f"MTP 1-step acceptance: {hits}/{tried} = {hits / max(tried, 1):.2f}",
          flush=True)



if __name__ == "__main__":
    main()
