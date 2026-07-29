"""Pre-compute GemmaRMSNorm weight+1.0 to eliminate per-forward copy kernels.

GemmaRMSNorm.forward_native() does ``self.weight.float() + 1.0`` on EVERY
forward call, generating an elementwise copy kernel each time.  With 161
instances × ~2 calls/step (verify + draft), this produces ~300+ unnecessary
kernel launches per decode step (profiling: 1.06ms/step in copy kernels).

Fix: cache the float32 weight+1.0 tensor on first access.  The weight is a
frozen parameter (inference-only), so this is safe.

Usage::

    from runtime.gemma_norm_patch import patch_gemma_rms_norm
    patch_gemma_rms_norm()
"""

from __future__ import annotations

from torch import Tensor

_patched = False


def patch_gemma_rms_norm() -> None:
    """Monkey-patch GemmaRMSNorm to cache weight.float()+1.0.  Idempotent."""
    global _patched
    if _patched:
        return
    _patched = True

    from runtime.legacy_qwen36_vllm import get_vllm_ir

    ir = get_vllm_ir()
    from runtime.legacy_qwen36_vllm import get_gemma_rms_norm

    GemmaRMSNorm = get_gemma_rms_norm()

    def _forward_native(
        self,
        x: Tensor,
        residual: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Lazily compute and cache weight+1.0 on first call
        cached = getattr(self, "_cached_weight_plus_one", None)
        if cached is None:
            cached = self.weight.float() + 1.0
            self._cached_weight_plus_one = cached
        if residual is None:
            return ir.ops.rms_norm(x, cached, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(x, residual, cached, self.variance_epsilon)

    GemmaRMSNorm.forward_native = _forward_native
    GemmaRMSNorm.forward_cuda = _forward_native
