"""Self-built plain (unquantized) Linear layer -- Phase 2 of the vLLM removal plan.

Replaces vLLM's ``ColumnParallelLinear``/``QKVParallelLinear``/
``RowParallelLinear``/``ReplicatedLinear`` for the TP=1, no-quantization
case. Verified against the real checkpoint's quantization_config
(models--poolside--Laguna-S-2.1-NVFP4, config.json) before writing this:
the ``ignore`` list explicitly excludes every attention projection
(q/k/v/o/g_proj, all layers), the dense MLP (layer 0, the only
``mlp_only_layers`` entry), the MoE router gate, and every MoE
shared_expert projection from NVFP4 quantization -- all of these are
plain BF16 on disk. Only ``experts.N.{gate,up,down}_proj`` (inside
``FusedMoE``, still vLLM-owned, see laguna_decoder.py) are NVFP4.
This directly contradicts the original Phase 2 plan's assumption that
these four module classes needed ``NvFp4Linear`` -- see nvfp4_linear.py's
docstring for the (still valid, just not applicable *here*) NVFP4 port;
that class currently has no live call site in the Laguna production path.

Same weight_loader-closure design as ``NvFp4Linear``: each Parameter gets
a bound closure matching vLLM's ``weight_loader(param, loaded_weight,
shard_id=None)`` convention, so the existing generic dispatch in
``LagunaModelSelfBuilt.load_weights`` (stacked_params_mapping for QKV,
default_weight_loader fallback otherwise) needs zero changes to route
weights into this class -- verified by reading that function, not assumed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_SHARD_ID_TO_IDX = {"q": 0, "k": 1, "v": 2}


class PlainLinear(nn.Module):
    """Unquantized Linear, TP=1, with optional output-dim shard fusion.

    ``shard_sizes=None`` (default): one logical weight matrix (o_proj,
    down_proj, gate_proj, up_proj, g_proj, MoE gate/shared_expert.*).

    ``shard_sizes=[q_size, k_size, v_size]``: fused QKV-style layer,
    mirroring vLLM's ``QKVParallelLinear`` output layout exactly (shards
    concatenated along dim 0 in q/k/v order).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        shard_sizes: list[int] | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.shard_sizes = list(shard_sizes) if shard_sizes else [output_size]
        assert sum(self.shard_sizes) == output_size, (
            f"shard_sizes {self.shard_sizes} must sum to output_size {output_size}"
        )
        offsets = []
        running = 0
        for s in self.shard_sizes:
            offsets.append(running)
            running += s
        self.shard_offsets = offsets

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

        self.weight.weight_loader = self._make_weight_loader("weight")
        if self.bias is not None:
            self.bias.weight_loader = self._make_weight_loader("bias")

    def _make_weight_loader(self, param_name: str):
        def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id=None):
            if (
                shard_id is None
                and len(self.shard_sizes) > 1
                and loaded_weight.shape[0] == param.data.shape[0]
            ):
                # A multi-shard Linear (e.g. qkv_proj, constructed to accept
                # 3 separately-stacked q/k/v checkpoint tensors, matching
                # the main model's checkpoint layout) can ALSO be fed a
                # single already-fused checkpoint tensor covering the full
                # output dim at once (verified against the real DFlash
                # draft checkpoint: its qkv_proj.weight is one [11264,3072]
                # tensor, not separate q/k/v -- a different on-disk layout
                # for the SAME LagunaAttentionSelfBuilt class, reused
                # unmodified between main and draft). Unambiguous: this
                # only fires when the caller has no shard_id at all AND
                # the incoming tensor's size matches the FULL param, which
                # could not have succeeded as a real single-shard load
                # anyway (shape mismatch), so this only adds a previously-
                # impossible-to-succeed case rather than changing any
                # currently-working one.
                param.data.copy_(loaded_weight)
                return
            if shard_id is None:
                shard_idx = 0
            elif isinstance(shard_id, str):
                shard_idx = _SHARD_ID_TO_IDX[shard_id]
            else:
                shard_idx = shard_id
            offset = self.shard_offsets[shard_idx]
            size = self.shard_sizes[shard_idx]
            dst = param.data.narrow(0, offset, size)
            assert dst.shape == loaded_weight.shape, (
                f"PlainLinear {param_name} shard {shard_idx}: dst "
                f"{tuple(dst.shape)} vs loaded {tuple(loaded_weight.shape)}"
            )
            dst.copy_(loaded_weight)

        return weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
