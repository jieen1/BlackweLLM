"""B5 模块化：MTP accept/reject 域。

从 direct_model_runner.py 提取的 determine_accept_reject* 纯函数。
纯移动不改逻辑（B5 parity 门禁）。

E2-a (docs/e2e-and-quality-plan.md §2.2) adds the non-greedy sibling of the
functions above: rejection-sampling accept/reject for ``temperature>0``
speculative decoding (``sample_accept_reject`` and its two building blocks,
``acceptance_probability`` / ``residual_distribution``, near the bottom of
this file). The greedy functions never sample (``argmax`` is deterministic
and needs no distribution-preservation argument); the non-greedy ones draw
random numbers, so "produces *a* token" is not enough -- it must produce a
token whose marginal distribution equals the target model's own sampling
distribution, otherwise speculative decoding silently changes what the
model outputs. See ``sample_accept_reject``'s docstring for the algorithm
and ``tests/test_mtp_accept_sampling.py`` for the exact-rational proof that
this file's formulas satisfy that property (not just "looks right on a
synthetic example").
"""

from __future__ import annotations

import torch

from runtime.round_profile import round_profile
from runtime.sampling import sample_from_distribution, validate_sampling_distribution


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
    slots: list[int],
    drafts: dict[int, list[int]] | torch.Tensor,
    verify_logits: torch.Tensor,
    k: int,
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
    everything upstream of that stays on-GPU.  ``drafts`` may alternatively
    be a device tensor ``[len(slots), k+1]`` (anchor + K drafts, the
    graph-path fast form): candidates are then compared entirely on device
    and ``committed`` is built from the verifier's own predictions, which
    equal the accepted draft values by the match condition -- no draft host
    round-trip is needed.
    """
    num_reqs = len(slots)
    predicted = verify_logits.argmax(dim=-1).view(num_reqs, k + 1)  # [num_reqs, k+1], int64
    if isinstance(drafts, torch.Tensor):
        if tuple(drafts.shape) != (num_reqs, k + 1):
            raise ValueError(
                "device drafts must have shape [num_reqs, k+1] (anchor + K candidates)"
            )
        draft_next = drafts[:, 1 : k + 1]  # [num_reqs, k] -- device slice, no H2D
    else:
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
    # QSR_PROFILE_ROUNDS sub-phase: everything before this line is the GPU
    # verify+lmhead queue drain; everything after is pure host decision work.
    round_profile.phase("accept_gpu_wait")

    decisions: dict[int, dict] = {}
    for i, s in enumerate(slots):
        row = combined_list[i]
        na = row[0]
        pred_row = row[1:]
        if isinstance(drafts, torch.Tensor):
            # Every position p < na matched (predicted[p] == draft[p+1]), so
            # the verifier's own predictions are exactly the committed draft
            # values -- read them from the already-synchronised combined row.
            committed = [pred_row[p] for p in range(na)] + [pred_row[na]]
        else:
            committed = [drafts[s][p + 1] for p in range(na)] + [pred_row[na]]
        decisions[s] = {
            "num_accepted": na,
            "committed": committed,
            "rejected_at": na if na < k else None,
        }
    return decisions


# --------------------------------------------------------------------------
# E2-a: non-greedy (rejection-sampling) accept/reject.
# --------------------------------------------------------------------------
#
# Reference: Leviathan, Kalman & Matias, "Fast Inference from Transformers
# via Speculative Decoding" (https://arxiv.org/abs/2211.17192) and Chen et
# al., "Accelerating Large Language Model Decoding with Speculative
# Sampling" (https://arxiv.org/abs/2302.01318) -- the same algorithm vLLM
# ships in ``vllm/v1/sample/rejection_sampler.py`` (read locally at
# ``/home/bot/vllm/vllm/v1/sample/rejection_sampler.py`` while implementing
# this) and SGLang implements for its EAGLE path. This module is a plain
# eager/CPU-tensor re-derivation of that same algorithm -- not a novel one --
# scoped for E2-a (CPU-only correctness) and reusable by E2-b's GPU
# integration. Flash-Next's sampled MTP path now supplies the draft
# distributions from its graph-backed proposer; other backends may continue
# to use this helper independently.
#
# ``determine_accept_reject*`` above only ever needs to prove "picks the
# same token the greedy target model would have picked" -- an equality
# check. The functions below instead must prove a *distributional*
# property: draw K draft tokens ``x_0..x_{K-1}`` from a draft distribution
# ``q_0..q_{K-1}`` (independent per position, one row per verify position);
# for each position in turn, accept ``x_p`` with probability
# ``min(1, p_p(x_p) / q_p(x_p))``; on the first rejection at position ``r``,
# resample from the *residual* distribution
# ``norm(max(0, p_r - q_r))`` and stop (later positions are discarded, same
# as the greedy path -- their target distributions were computed under a
# now-invalidated hypothesized context and are never real). If every
# position is accepted, sample one "bonus" token directly from ``p_K``
# (position K's target distribution needed no acceptance test -- it was
# never proposed by the draft model, so there is nothing to correct).
#
# The correctness claim -- unconditional on ``q`` -- is that the output
# token's marginal distribution at any position that is *reached* equals
# ``p`` at that position, exactly:
#
#   P(output = x) = q(x) * min(1, p(x)/q(x))                    [accepted]
#                  + (1 - sum_y q(y)*min(1, p(y)/q(y))) * r(x)   [rejected]
#                  = min(p(x), q(x)) + max(0, p(x) - q(x))       [algebra]
#                  = p(x)                                        for every x
#
# (``min(a,b) + max(0,a-b) == a`` for any reals ``a,b`` -- case split on
# ``a>=b`` vs ``a<b`` -- so this holds token-by-token with no assumption
# about ``q`` at all, in particular without requiring ``q`` and ``p`` to
# share support). ``tests/test_mtp_accept_sampling.py`` checks this
# algebraic identity with ``fractions.Fraction`` (exact rational arithmetic,
# not float comparison) across hand-picked edge cases (identical
# distributions, disjoint support, one-hot/deterministic distributions) and
# randomized rational distributions, then separately checks that *this
# file's* ``acceptance_probability``/``residual_distribution`` functions
# reproduce those same exact values, then separately runs a statistical
# (chi-square) goodness-of-fit test on actual samples drawn through
# ``sample_accept_reject`` to confirm the RNG-driven code path (not just the
# closed-form formula) reproduces the target distribution.


def acceptance_probability(
    target_row: torch.Tensor, draft_row: torch.Tensor, token: int
) -> torch.Tensor:
    """``min(1, p(token)/q(token))`` -- the probability of accepting a draft
    token proposed at this position, given the target distribution
    ``target_row`` (``p``) and the draft distribution ``draft_row`` (``q``)
    at the SAME position. Returns a 0-d tensor so callers can compare it
    against a uniform draw without a host round trip.

    ``q(token) <= 0`` returns 0 (always reject): a token actually sampled
    from ``q`` has ``q(token) > 0`` almost surely, so this branch is a
    defensive floor against float underflow, not a case the algorithm's
    derivation depends on (mirrors vLLM's
    ``accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob``).
    """
    q_x = draft_row[token]
    if q_x <= 0:
        return torch.zeros((), dtype=target_row.dtype, device=target_row.device)
    p_x = target_row[token]
    return torch.clamp(p_x / q_x, max=1.0)


def residual_distribution(target_row: torch.Tensor, draft_row: torch.Tensor) -> torch.Tensor:
    """``norm(max(0, p - q))`` -- the distribution a rejection resamples
    from. Deterministic (no RNG): the random part of "resample from the
    residual" is entirely in the caller's ``torch.multinomial`` draw, so
    this function's output can be checked exactly against a hand-derived
    expectation without needing to reason about any sampler.

    ``total <= 0`` (only possible when ``target_row == draft_row``
    everywhere, i.e. ``p == q``) falls back to ``p`` itself. This is an
    unreachable branch in the real accept/reject loop -- a rejection at
    token ``x`` requires ``p(x) < q(x)``, which by conservation of
    probability mass (``sum p == sum q == 1``) forces
    ``sum_{y != x} (p(y) - q(y)) == q(x) - p(x) > 0``, hence residual mass
    strictly greater than 0 elsewhere; see
    ``tests/test_mtp_accept_sampling.py`` for the exact-rational proof of
    that conservation argument. Kept anyway as a defensive floor against
    float cancellation (``target_row - draft_row`` summing to a tiny
    negative or zero value instead of the true positive one).
    """
    residual = (target_row - draft_row).clamp_min(0.0)
    total = residual.sum()
    if total <= 0:
        residual = target_row.clamp_min(0.0)
        total = residual.sum()
    return residual / total


def sample_accept_reject(
    draft_tokens: list[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> dict:
    """Non-greedy (rejection-sampling) MTP accept/reject -- the
    ``temperature>0`` sibling of ``determine_accept_reject_from_predictions``.

    Args:
        draft_tokens: K draft continuation tokens (NOT including the
            anchor -- unlike the greedy functions above, there is no
            "anchor" concept here: position ``p``'s acceptance test only
            ever needs ``draft_tokens[p]``, ``draft_probs[p]`` and
            ``target_probs[p]``).
        draft_probs: ``[K, vocab]``. Row ``p`` is the draft model's own
            sampling distribution (post temperature/top-k/top-p, i.e. the
            same distribution ``runtime.sampling.sample_from_logits`` would
            have normalized to) at the position where it produced
            ``draft_tokens[p]``. MUST be the distribution actually sampled
            from -- passing ``argmax``-derived probabilities here would
            silently change the acceptance math.  The proposer must retain
            the non-degenerate ``q`` row it actually sampled from; an
            argmax-derived one-hot row does not implement sampled MTP.
        target_probs: ``[K+1, vocab]``. Rows ``0..K-1`` are the target
            (main) model's distribution at each of the K verify positions,
            used for that position's acceptance test. Row ``K`` is the
            target distribution one position past the last draft token,
            used ONLY for the bonus token when every draft is accepted.
        generator: optional seeded ``torch.Generator`` (CPU or CUDA,
            matching the tensors' device) for reproducible sampling --
            same role as ``runtime.sampling.sample_from_logits``'s
            ``generator`` argument. Every random draw in this function
            (acceptance tests, residual resampling, bonus sampling) reads
            from this ONE generator in sequence, so a
            ``runtime.sampling.PersistentSeed``-backed generator advances
            consistently across an entire request's decode, same as the
            existing sampled decode path.

    Returns:
        Same contract as ``determine_accept_reject_from_predictions``:
        ``{"num_accepted": int, "committed": list[int], "rejected_at":
        int | None}``. ``committed`` is the accepted draft prefix plus
        EXACTLY one trailing recovery-or-bonus token.  Flash-Next uses this
        return shape directly in its MTP verify/commit loop; DFlash callers
        can use the same helper when their proposer exposes matching ``q``
        rows.
    """
    k = len(draft_tokens)
    if draft_probs.shape[0] != k:
        raise ValueError(
            f"draft_probs must have one row per draft token (need {k}, got {draft_probs.shape[0]})"
        )
    if target_probs.shape[0] < k + 1:
        raise ValueError(
            "target_probs must contain K verifier distributions plus one bonus "
            f"distribution (need {k + 1}, got {target_probs.shape[0]})"
        )

    # Validate both model distributions before the first random draw.  If a
    # graph produced NaN/Inf, calling CUDA multinomial first would poison the
    # context with a device-side assert and make slot recovery impossible.
    validate_sampling_distribution(draft_probs, context="MTP draft")
    validate_sampling_distribution(target_probs, context="MTP target")

    device = draft_probs.device
    committed: list[int] = []
    for p in range(k):
        token = draft_tokens[p]
        accept_prob = acceptance_probability(target_probs[p], draft_probs[p], token)
        u = torch.rand((), generator=generator, dtype=torch.float64, device=device)
        if bool(u < accept_prob):
            committed.append(token)
            continue
        residual = residual_distribution(target_probs[p], draft_probs[p])
        recovered = int(
            sample_from_distribution(residual, generator=generator).item()
        )
        committed.append(recovered)
        return {"num_accepted": p, "committed": committed, "rejected_at": p}

    bonus = int(sample_from_distribution(target_probs[k], generator=generator).item())
    committed.append(bonus)
    return {"num_accepted": k, "committed": committed, "rejected_at": None}


def sample_accept_reject_sparse(
    draft_tokens: list[int],
    draft_indices: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> dict:
    """Rejection sampling when the draft distribution is top-k sparse.

    DFlash2's selector samples from a distribution over its ``top_k`` target
    candidates rather than materializing a ``[K, vocab]`` row.  The standard
    acceptance proof is unchanged: ``q(token)`` is the sparse probability for
    the proposed token, and the rejection residual subtracts each candidate's
    mass from the target row with ``scatter_add_``.  This keeps the exact
    sampled distribution while avoiding a second full-vocabulary draft tensor.
    """

    k = len(draft_tokens)
    if draft_indices.ndim != 2 or draft_indices.shape[0] != k:
        raise ValueError(
            "draft_indices must have shape [K, top_k], "
            f"got {tuple(draft_indices.shape)} for K={k}"
        )
    if draft_probs.shape != draft_indices.shape:
        raise ValueError(
            "draft_probs must have the same shape as draft_indices, "
            f"got {tuple(draft_probs.shape)} vs {tuple(draft_indices.shape)}"
        )
    if target_probs.ndim != 2 or target_probs.shape[0] < k + 1:
        raise ValueError(
            "target_probs must contain K verifier rows plus one bonus row; "
            f"got {tuple(target_probs.shape)} for K={k}"
        )
    if draft_indices.device != draft_probs.device or draft_indices.device != target_probs.device:
        raise ValueError(
            "sparse draft indices, probabilities, and target probabilities "
            "must share a device"
        )

    validate_sampling_distribution(draft_probs, context="sparse MTP draft")
    validate_sampling_distribution(target_probs, context="sparse MTP target")

    committed: list[int] = []
    for position, token in enumerate(draft_tokens):
        indices = draft_indices[position].long()
        sparse_row = draft_probs[position]
        match = indices == int(token)
        q_token = sparse_row.masked_select(match).sum()
        p_token = target_probs[position, int(token)]
        if q_token <= 0:
            accept_prob = torch.zeros((), dtype=target_probs.dtype, device=target_probs.device)
        else:
            accept_prob = torch.clamp(p_token / q_token, max=1.0)
        uniform = torch.rand(
            (), generator=generator, dtype=torch.float64, device=target_probs.device
        )
        if bool(uniform < accept_prob):
            committed.append(int(token))
            continue

        residual = target_probs[position].clone()
        residual.scatter_add_(0, indices, -sparse_row)
        residual.clamp_min_(0.0)
        total = residual.sum()
        if total > 0:
            residual = residual / total
        else:
            residual = target_probs[position]
        recovered = int(
            sample_from_distribution(residual, generator=generator).item()
        )
        committed.append(recovered)
        return {
            "num_accepted": position,
            "committed": committed,
            "rejected_at": position,
        }

    bonus = int(sample_from_distribution(target_probs[k], generator=generator).item())
    committed.append(bonus)
    return {"num_accepted": k, "committed": committed, "rejected_at": None}
