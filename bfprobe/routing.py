"""P2b MoE routing probe: extract router_logits / topk_ids / topk_weights.

Integration point: ``runtime/backends/laguna.py``'s sparkinfer-patched MoE
forward (``_patched_forward``, built by ``_make_patched_forward`` inside
``LagunaBackend._patch_moe_sparkinfer``), immediately after the call to
vLLM's own ``fused_topk_bias`` router function
(``runtime/backends/laguna.py:537-545``). This is the exact tensor pair the
head-of-line investigation (``notes/2026-07-27-acceptance-rate-gap-vllm-vs-
ours-same-prompt.md``) needs: if the vLLM oracle side (see
``notes/2026-07-27-bfprobe-moe-routing-and-vllm-tap.md`` for the oracle-side
tap point) routes to different expert ids on the same prompt, that is a
routing bug, not a numerics bug -- one bit of information settles it.

Site ids (this module's assigned band is 300-399; see the probe site table
in the design doc for the global allocation):

    SITE_ROUTER_LOGITS = 300   T2  (M, num_experts) float32, post-softcap
                                    gate output (P-ROUTER-LOGITS)
    SITE_TOPK_IDS      = 301   T2  (M, top_k) int32, pre-EPLB logical expert
                                    ids, in the router kernel's native order
                                    (P-TOPK -- highest-value site in the repo)
    SITE_TOPK_WEIGHTS  = 302   T2  (M, top_k) float32, renormalized routing
                                    weights (routed_scaling_factor is *not*
                                    baked in here -- see the semantic-parity
                                    checklist in the design note)

Cost: 47 MoE layers x 16 verify tokens x 10 experts x (4+4) bytes ~= 60 KB
per DFlash round, against a 44.16 ms/round budget -- 0.0001%, per
``notes/2026-07-27-probe-system-design-and-plan.md`` section 4.

The probe deliberately captures only rows up to DFlash's M=16 verify
shape. Capturing a full long-context prefill would retain 47 copies of
``(65536, 256)`` router logits in the local bus and can OOM the diagnostic
process; prefill routing is not a target of this probe.

Ordering convention (no explicit layer index is emitted, to keep the hot
path to one function call): ``capture_routing`` is called once per MoE
layer per forward pass, and MoE layers are patched once, at model-load
time, in ascending checkpoint layer order (1..47) by
``_patch_moe_sparkinfer``'s ``model.named_modules()`` walk; every forward
pass re-invokes those same closures in that same fixed order. Offline
consumers reconstruct the (round, layer, token) grid from this implicit
call order plus the known layer count, exactly as the T1 per-layer
signature ring already does (design doc section 4's "48x4 signatures" row
has the same property).
"""

from __future__ import annotations

from typing import Any

try:
    from bfprobe.bus import PROBE_ENABLED, emit_tensor
except ImportError:
    from bfprobe._bus_stub import PROBE_ENABLED, emit_tensor

SITE_ROUTER_LOGITS = 300
SITE_TOPK_IDS = 301
SITE_TOPK_WEIGHTS = 302
MAX_CAPTURE_ROWS = 16


def capture_routing(router_logits: Any, topk_ids: Any, topk_weights: Any) -> None:
    """Emit one MoE layer's routing decision to the probe bus (T2).

    No-op (a single module-level boolean check) when the probe bus is
    disabled, so the production hot path pays nothing when
    ``PROBE_ENABLED`` is ``False``. Call once per MoE layer, per forward
    pass, right after ``fused_topk_bias`` returns -- see the integration in
    ``runtime/backends/laguna.py``.

    Args:
        router_logits: Gate output for this layer, post-softcap, shape
            ``(M, num_experts)``.
        topk_ids: Selected expert ids, shape ``(M, top_k)``.
        topk_weights: Renormalized routing weights, shape ``(M, top_k)``.
    """
    if not PROBE_ENABLED:
        return
    if topk_ids.shape[0] > MAX_CAPTURE_ROWS:
        return
    emit_tensor(SITE_ROUTER_LOGITS, router_logits)
    emit_tensor(SITE_TOPK_IDS, topk_ids)
    emit_tensor(SITE_TOPK_WEIGHTS, topk_weights)
