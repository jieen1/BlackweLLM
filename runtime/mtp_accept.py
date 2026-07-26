"""B5 模块化：MTP accept/reject 域。

从 direct_model_runner.py 提取的 determine_accept_reject* 纯函数。
纯移动不改逻辑（B5 parity 门禁）。
"""

from __future__ import annotations

import torch


def determine_accept_reject_from_predictions(
    draft_tokens: list[int],
    predicted_tokens: list[int],
) -> dict:
    """Greedy accept/reject from already-sampled target token ids.

    ``draft_tokens`` contains the pending anchor followed by K draft tokens.
    ``predicted_tokens`` contains K verifier predictions plus the final target
    bonus prediction.  ``num_accepted`` counts only matching draft tokens;
    the recovery/bonus token is committed output but is never counted as an
    accepted draft.
    """
    k = len(draft_tokens) - 1
    if k < 0:
        raise ValueError("draft_tokens must contain an anchor token")
    if len(predicted_tokens) < k + 1:
        raise ValueError(
            "predicted_tokens must contain K verifier predictions plus one bonus "
            f"(need {k + 1}, got {len(predicted_tokens)})"
        )

    committed: list[int] = []
    for p in range(k):
        predicted = predicted_tokens[p]
        if predicted == draft_tokens[p + 1]:
            committed.append(draft_tokens[p + 1])
        else:
            committed.append(predicted)
            return {"num_accepted": p, "committed": committed, "rejected_at": p}
    committed.append(predicted_tokens[k])
    return {"num_accepted": k, "committed": committed, "rejected_at": None}


def determine_accept_reject(draft_tokens: list[int], verify_logits) -> dict:
    """Greedy MTP accept/reject (2026-07-17, moved here from
    ``benchmarks/mtp_accept_reject_check.py`` so the real
    ``mtp_verify_and_commit`` coordinator and that benchmark's regression
    test share ONE implementation, not two copies). ``draft_tokens`` has
    K+1 entries (anchor + K drafts); ``verify_logits`` is shaped
    ``[K+1, vocab]`` for ONE request. Returns ``num_accepted`` (0..K), the
    committed real token ids (accepted drafts, if any, plus exactly one
    recovery/bonus token), and the rejection position (``None`` if all K
    were accepted)."""
    k = len(draft_tokens) - 1
    predicted_tokens = verify_logits[: k + 1].argmax(dim=-1).tolist()
    return determine_accept_reject_from_predictions(draft_tokens, predicted_tokens)


def determine_accept_reject_batch(
    slots: list[int], drafts: dict[int, list[int]], verify_logits: torch.Tensor, k: int
) -> dict[int, dict]:
    """Batched analogue of ``determine_accept_reject`` -- computes the SAME
    greedy accept/reject decision for every slot in ONE vectorized GPU op
    plus exactly ONE host round-trip, instead of a Python loop calling
    ``determine_accept_reject`` once per slot (each of which does up to
    ``k+1`` sequential ``.item()`` calls -- 2026-07-17, Phase 3 of
    ``notes/2026-07-17-post-ragged-round-next-steps.md``, directly
    targeting that doc's section 7.4 finding that the compute-phase
    no-kernel gap is dominated by per-launch host dispatch, not GPU work).

    ``verify_logits`` is shaped ``[len(slots)*(k+1), vocab]`` in
    request-then-position order (``verify_batch``'s / the verify graph's
    own output convention). Returns a dict keyed by slot id, each value
    byte-for-byte the same shape as ``determine_accept_reject``'s own
    return dict (``num_accepted``/``committed``/``rejected_at``) -- this is
    a strict re-derivation of the same greedy rule, not a different one:
    for slot ``s`` with drafts ``d = drafts[s]`` (``k+1`` entries, anchor +
    k draft continuations) and per-position argmax predictions ``pred``,
    ``committed = [d[p+1] for p in range(num_accepted)] + [pred[num_accepted]]``
    is exactly what the original sequential version produces in EITHER
    branch (a genuine reject at position ``num_accepted < k``, where
    ``pred[num_accepted]`` is the recovery token; or a full accept where
    ``num_accepted == k`` and ``pred[k]`` is the bonus token) -- verified by
    direct comparison against ``determine_accept_reject`` in
    ``benchmarks/mtp_verify_cudagraph_check.py``.

    Vectorization: ``verify_logits.argmax(dim=-1)`` computes every
    position's greedy prediction in ONE kernel launch (instead of
    ``len(slots)*(k+1)`` separate ``.argmax().item()`` calls); comparing
    against each slot's own draft-continuation tokens and taking a
    cumulative-AND ("still matching every earlier position") over the
    position axis is a second vectorized op that yields ``num_accepted``
    for every slot at once. Only the FINAL small result tensor (shape
    ``[len(slots), k+2]``) is pulled to host via a single ``.tolist()`` --
    everything upstream of that stays on-GPU.
    """
    num_reqs = len(slots)
    predicted = verify_logits.argmax(dim=-1).view(num_reqs, k + 1)  # [num_reqs, k+1], int64
    draft_next = torch.tensor(
        [drafts[s][1:] for s in slots], dtype=predicted.dtype, device=predicted.device
    )  # [num_reqs, k] -- each slot's k candidate continuation tokens (drafts[s][1:])
    matches = predicted[:, :k] == draft_next  # [num_reqs, k] bool
    # True at position p iff every position <= p matched (the greedy
    # "still on the accepted prefix" condition) -- a cumulative product
    # over bools is exactly a running AND.
    still_matching = (
        matches.cumprod(dim=1).bool()
        if k > 0
        else matches.new_zeros((num_reqs, 0), dtype=torch.bool)
    )
    num_accepted = still_matching.sum(dim=1)  # [num_reqs], int64, values 0..k

    # ONE combined host round-trip for the whole batch: num_accepted plus
    # every position's raw prediction (needed to build "committed" below).
    combined = torch.cat([num_accepted.unsqueeze(1), predicted], dim=1)  # [num_reqs, 1 + (k+1)]
    combined_list = combined.tolist()

    decisions: dict[int, dict] = {}
    for i, s in enumerate(slots):
        row = combined_list[i]
        na = row[0]
        pred_row = row[1:]
        committed = [drafts[s][p + 1] for p in range(na)] + [pred_row[na]]
        decisions[s] = {
            "num_accepted": na,
            "committed": committed,
            "rejected_at": na if na < k else None,
        }
    return decisions
