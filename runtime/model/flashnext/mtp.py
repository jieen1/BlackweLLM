"""Flash-Next MTP (multi-token prediction) draft model -- bring-up.

One extra decoder layer (QSA + BF16 MoE) that, given the main model's
hc-hidden state and the embedding of the just-sampled token, predicts the
logits of the token AFTER that. Used for NEXTN-style speculative decoding.

Structure pinned from the checkpoint + sglang ``qwen4_exp_mtp.py``:
input fusion (``_fuse_residual_linear_shared``), one full-attention layer
with MoE, a final hyper-connection mixer, then the shared lm_head.
"""

from __future__ import annotations

import json
import os
import pathlib

import torch
from torch import nn
from torch.nn import functional as F

from runtime.model.flashnext.hyper_connection import GatedResidual
from runtime.model.flashnext.mtp_kernels import (
    mtp_expert_matvec,
    mtp_weighted_route_reduce,
)
from runtime.model.flashnext.qsa import (
    FlashNextQSAAttention,
    QsaDecodeAttention,
    QSAIndexer,
    _qsa_cache_is_quantized,
    qsa_cache_index_copy_,
    quantize_qsa_kv,
)


def mtp_expert_weight_dtype() -> torch.dtype:
    """Return the resident dtype for MTP routed-expert weights.

    The checkpoint stores the MTP experts in BF16.  Keeping that as the
    explicit ``bf16`` compatibility path remains available, but production
    defaults to a per-output-row FP8 E4M3 representation.  It cuts the
    resident expert footprint roughly in half while retaining a small FP32
    scale plane.  The setting is resolved at model-load time so CUDA Graphs
    never change dtype after capture.  On an older torch build without FP8,
    an omitted or blank environment variable falls back to BF16; an explicit
    FP8 request fails closed with a useful error.
    """
    requested = os.environ.get("QSR_FLASHNEXT_MTP_EXPERT_DTYPE")
    value = requested.strip().lower() if requested is not None else "fp8_e4m3fn"
    if not value:
        value = "fp8_e4m3fn"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp8", "fp8_e4m3", "fp8_e4m3fn", "e4m3", "float8_e4m3fn"}:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            if requested is None:
                return torch.bfloat16
            raise RuntimeError("MTP FP8 experts requested but torch.float8_e4m3fn is unavailable")
        return dtype
    raise ValueError(
        "QSR_FLASHNEXT_MTP_EXPERT_DTYPE must be bf16 or fp8_e4m3fn, "
        f"got {value!r}"
    )


def _mtp_expert_is_fp8(dtype: torch.dtype) -> bool:
    return dtype == getattr(torch, "float8_e4m3fn", None)


def quantize_mtp_expert_weight(
    values: torch.Tensor,
    *,
    dtype: torch.dtype,
    chunk_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an expert matrix per output row for resident FP8 storage.

    ``values`` is ``[experts, output, input]``.  A scale per expert/output
    row avoids sharing a single scale across unrelated routed experts and is
    cheap (under 8 MiB for the full MTP layer).  The returned scale is FP32
    because it is multiplied inside the FP32 accumulation path.
    """
    if not _mtp_expert_is_fp8(dtype):
        raise ValueError(f"per-row MTP quantization requires FP8 E4M3, got {dtype}")
    if values.ndim != 3:
        raise ValueError(f"MTP expert weights must be rank-3, got {tuple(values.shape)}")
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    quant_max = float(torch.finfo(dtype).max)
    # Do not materialize ``values.float()`` for the complete 3.3 GiB FC1
    # tensor on this machine (the host has only 23 GiB and the service may
    # already occupy most of the card).  Row chunks keep the transient below
    # one GiB while the output remains one contiguous FP8 allocation.
    quantized = torch.empty(values.shape, dtype=dtype, device=values.device)
    scales = torch.empty(
        values.shape[:-1], dtype=torch.float32, device=values.device
    )
    for start in range(0, values.shape[0], chunk_rows):
        end = min(start + chunk_rows, values.shape[0])
        chunk = values[start:end].float()
        chunk_scales = chunk.abs().amax(dim=-1).div(quant_max).clamp_min(1e-8)
        quantized[start:end].copy_(
            (chunk / chunk_scales.unsqueeze(-1)).clamp(-quant_max, quant_max).to(dtype)
        )
        scales[start:end].copy_(chunk_scales)
    return quantized, scales


def _shared_sparse_captured_len(sess) -> int:
    """Read graph-owned sparse length when eager falls back after replay."""
    sparse = getattr(sess, "sparse_graph_buffers", None)
    captured_len = getattr(sparse, "shared_captured_len", None)
    if torch.is_tensor(captured_len):
        # This helper is used only by eager reuse (the graph path passes the
        # device tensor through to the fixed ABI), so the scalar read is safe.
        return int(captured_len.item())
    return int(getattr(sess, "shared_sparse_captured_len", 0))


def _shared_sparse_reuse(sess) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the latest graph/eager sparse row and validity mask."""
    sparse = getattr(sess, "sparse_graph_buffers", None)
    indices = getattr(sparse, "shared_indices", None)
    valid = getattr(sparse, "shared_valid", None)
    if torch.is_tensor(indices) and torch.is_tensor(valid):
        # The fixed graph buffers are updated by replay.  They are therefore
        # authoritative when eager execution resumes after a graph path.
        return indices, valid
    return (
        getattr(sess, "shared_sparse_indices", None),
        getattr(sess, "shared_sparse_valid", None),
    )


class _PlainGemmaRMSNorm(nn.Module):
    """Whole-tensor Gemma-style RMSNorm: no branch grouping."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.eps) * (1.0 + self.weight.float())).to(dtype)


class BF16MoE(nn.Module):
    """Decode MoE over resident BF16 or optional FP8 expert weights.

    The class name is retained for API compatibility with the original MTP
    bring-up.  FP8 is storage-only: routing, activations, and all non-expert
    projections remain BF16, while the custom matvec kernel dequantizes each
    weight row into its FP32 accumulator.
    """

    def __init__(
        self,
        hidden: int,
        num_experts: int,
        inter: int,
        top_k: int,
        *,
        expert_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.inter = inter
        self.expert_dtype = expert_dtype
        if not (
            expert_dtype == torch.bfloat16
            or _mtp_expert_is_fp8(expert_dtype)
        ):
            raise ValueError(
                "MTP expert_dtype must be torch.bfloat16 or torch.float8_e4m3fn, "
                f"got {expert_dtype}"
            )
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        if _mtp_expert_is_fp8(expert_dtype):
            # Expert payloads are inference-only storage.  Registering them as
            # non-persistent buffers also keeps ``state_dict`` compatible with
            # the checkpoint's BF16 source tensors.
            self.register_buffer(
                "gate_up_proj",
                torch.empty(num_experts, 2 * inter, hidden, dtype=expert_dtype),
                persistent=False,
            )
            self.register_buffer(
                "down_proj",
                torch.empty(num_experts, hidden, inter, dtype=expert_dtype),
                persistent=False,
            )
            self.register_buffer(
                "gate_up_proj_scale",
                torch.empty(num_experts, 2 * inter, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "down_proj_scale",
                torch.empty(num_experts, hidden, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.gate_up_proj = nn.Parameter(
                torch.empty(num_experts, 2 * inter, hidden)
            )
            self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter))
        self.shared_gate = nn.Linear(hidden, inter, bias=False)
        self.shared_up = nn.Linear(hidden, inter, bias=False)
        self.shared_down = nn.Linear(inter, hidden, bias=False)
        self.shared_gate_lin = nn.Linear(hidden, 1, bias=False)
        # ``torch.bincount(minlength=...)`` stages its minlength metadata from
        # host memory and is therefore illegal during CUDA Graph capture.
        # Keep the tiny expert histogram resident on device and rebuild it
        # with scatter_add_ for every grouped-GEMM dispatch.
        self.register_buffer(
            "_expert_counts",
            torch.zeros(num_experts, dtype=torch.int32),
            persistent=False,
        )

    @property
    def expert_is_fp8(self) -> bool:
        return _mtp_expert_is_fp8(self.expert_dtype)

    def _save_to_state_dict(
        self,
        destination: dict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        """Include inference-only FP8 payloads in generic Torch checkpoints.

        The resident expert planes are non-persistent buffers deliberately so
        loading the BF16 safetensors checkpoint does not collide with their
        source names.  Once the module has been quantized, however, omitting
        those planes would make ``state_dict`` round-trips silently restore an
        uninitialised expert cache.  Serialize them under their normal module
        names and consume them in ``_load_from_state_dict`` below.
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if not self.expert_is_fp8:
            return
        for name in (
            "gate_up_proj",
            "gate_up_proj_scale",
            "down_proj",
            "down_proj_scale",
        ):
            value = getattr(self, name)
            destination[prefix + name] = value if keep_vars else value.detach()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Restore the explicit FP8 payload emitted by ``state_dict``."""
        # Pull the custom payload out before delegating.  ``nn.Module`` treats
        # non-persistent buffers as unknown keys while running its strict-key
        # check; leaving them in place would report every valid FP8 plane as
        # ``unexpected_keys``.
        payload: dict[str, torch.Tensor | None] = {}
        if self.expert_is_fp8:
            for name in (
                "gate_up_proj",
                "gate_up_proj_scale",
                "down_proj",
                "down_proj_scale",
            ):
                payload[name] = state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not self.expert_is_fp8:
            return
        with torch.no_grad():
            for name in (
                "gate_up_proj",
                "gate_up_proj_scale",
                "down_proj",
                "down_proj_scale",
            ):
                key = prefix + name
                value = payload[name]
                target = getattr(self, name)
                if value is None:
                    missing_keys.append(key)
                    continue
                if tuple(value.shape) != tuple(target.shape):
                    error_msgs.append(
                        f"size mismatch for {key}: copying a param with shape "
                        f"{tuple(value.shape)} from checkpoint, the shape in "
                        f"current model is {tuple(target.shape)}"
                    )
                    continue
                try:
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
                except (RuntimeError, TypeError) as exc:
                    error_msgs.append(f"while loading {key}: {exc}")

    def set_fp8_expert_weights(
        self,
        gate_up_proj: torch.Tensor,
        gate_up_proj_scale: torch.Tensor,
        down_proj: torch.Tensor,
        down_proj_scale: torch.Tensor,
    ) -> None:
        """Install already-quantized FP8 expert payloads in-place."""
        if not self.expert_is_fp8:
            raise RuntimeError("set_fp8_expert_weights requires an FP8 MTP MoE")
        for name, value, expected_shape in (
            (
                "gate_up_proj",
                gate_up_proj,
                (self.num_experts, 2 * self.inter, self.gate.in_features),
            ),
            (
                "gate_up_proj_scale",
                gate_up_proj_scale,
                (self.num_experts, 2 * self.inter),
            ),
            (
                "down_proj",
                down_proj,
                (self.num_experts, self.gate.in_features, self.inter),
            ),
            (
                "down_proj_scale",
                down_proj_scale,
                (self.num_experts, self.gate.in_features),
            ),
        ):
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} != expected {expected_shape}"
                )
            if value.dtype != getattr(self, name).dtype:
                raise TypeError(
                    f"{name} dtype {value.dtype} != expected {getattr(self, name).dtype}"
                )
            getattr(self, name).copy_(value)

    def _expert_rows(
        self,
        values: torch.Tensor,
        ids: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> torch.Tensor:
        selected = values[ids]
        if scales is None:
            return selected
        return (selected.float() * scales[ids].unsqueeze(-1)).to(torch.bfloat16)

    def _indexed_experts(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reference expert path used off CUDA and by correctness tests."""
        gu = self._expert_rows(
            self.gate_up_proj,
            ids,
            self.gate_up_proj_scale if self.expert_is_fp8 else None,
        )  # [T, K, 2I, H]
        h = torch.einsum("td,tkid->tki", x, gu)
        g, u = h.split(self.inter, dim=-1)
        act = F.silu(g) * u
        dn = self._expert_rows(
            self.down_proj,
            ids,
            self.down_proj_scale if self.expert_is_fp8 else None,
        )  # [T, K, H, I]
        return torch.einsum("tki,tkdi->tkd", act, dn)

    def _grouped_experts(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        *,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run routed experts without materialising selected weight copies.

        The old ``weight[ids]`` implementation expands both expert matrices
        to ``[T, top_k, ...]``.  Flash-Next prompt sync has roughly one
        hundred rows and ten experts per row, so that transient is multiple
        GiB and dominates end-to-end latency.  Grouped GEMM instead sorts the
        token/expert assignments, consumes each resident expert matrix in
        place, and restores the original token-major routing order afterward.
        """
        if self.expert_is_fp8:
            if routing_weights is None:
                return self._indexed_experts(x, ids)
            return mtp_expert_matvec(
                x,
                ids,
                routing_weights,
                self.gate_up_proj,
                self.down_proj,
                gate_up_scales=self.gate_up_proj_scale,
                down_scales=self.down_proj_scale,
            )

        flat_ids = ids.flatten()
        order = torch.argsort(flat_ids, stable=True)
        sorted_ids = flat_ids.index_select(0, order)
        token_rows = torch.div(order, self.top_k, rounding_mode="floor")
        sorted_x = x.index_select(0, token_rows)
        counts = self._expert_counts
        counts.zero_()
        counts.scatter_add_(
            0,
            sorted_ids,
            torch.ones_like(sorted_ids, dtype=torch.int32),
        )
        offsets = counts.cumsum(0, dtype=torch.int32)

        gate_up = F.grouped_mm(
            sorted_x,
            self.gate_up_proj.transpose(-2, -1),
            offs=offsets,
        )
        gate, up = gate_up.split(self.inter, dim=-1)
        activated = F.silu(gate) * up
        sorted_out = F.grouped_mm(
            activated,
            self.down_proj.transpose(-2, -1),
            offs=offsets,
        )

        if routing_weights is not None:
            return mtp_weighted_route_reduce(sorted_out, order, routing_weights)

        out = torch.empty_like(sorted_out)
        out.index_copy_(0, order, sorted_out)
        return out.view(x.shape[0], self.top_k, x.shape[-1])

    def forward(self, x: torch.Tensor, *, graph_direct: bool = False) -> torch.Tensor:
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        if graph_direct or (self.expert_is_fp8 and x.is_cuda):
            if self.expert_is_fp8:
                routed = mtp_expert_matvec(
                    x,
                    ids,
                    weights,
                    self.gate_up_proj,
                    self.down_proj,
                    gate_up_scales=self.gate_up_proj_scale,
                    down_scales=self.down_proj_scale,
                )
            else:
                routed = mtp_expert_matvec(
                    x,
                    ids,
                    weights,
                    self.gate_up_proj,
                    self.down_proj,
                )
        elif x.is_cuda and x.dtype == torch.bfloat16:
            if os.getenv("QSR_FLASHNEXT_MTP_LEGACY_ROUTE_REDUCE", "0") == "1":
                # A/B seam for the pre-optimization restore + FP32 reduction.
                # This is intentionally opt-in and exists for same-process
                # measurements against the fused sorted-route reducer.
                out = self._grouped_experts(x, ids)
                routed = (out.float() * weights.unsqueeze(-1)).sum(dim=1).to(x.dtype)
            else:
                routed = self._grouped_experts(x, ids, routing_weights=weights)
        else:
            out = self._indexed_experts(x, ids)
            routed = (out.float() * weights.unsqueeze(-1)).sum(dim=1).to(x.dtype)

        sg = torch.sigmoid(self.shared_gate_lin(x))
        shared = self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))
        return routed + sg * shared


class FlashNextMTP(nn.Module):
    """MTP draft model (1 QSA+MoE layer) over the hc-hidden stream."""

    def __init__(self, cfg, *, expert_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        hs = cfg.hidden_size
        hc = cfg.hc_count
        self.hc_count = hc
        self.hidden_size = hs
        # NOTE: pre-fusion norms are PLAIN (ungrouped) Gemma RMSNorm over the
        # full width -- the reference uses GemmaRMSNorm here, unlike the
        # per-branch hc norms inside decoder layers.
        self.pre_fc_norm_embedding = _PlainGemmaRMSNorm(hs, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = _PlainGemmaRMSNorm(hc * hs, eps=cfg.rms_norm_eps)
        self.fc_embedding = nn.Linear(hs, hs, bias=False)
        self.fc_hidden = nn.Linear(hs, hs, bias=False)
        self.hyper_connection_mixer = GatedResidual(
            hc_count=hc,
            hidden_size=hs,
            lowrank=cfg.hc_lowrank,
            eps=cfg.rms_norm_eps,
            use_mix=True,
            use_combine=False,
        )
        self.attn_hc = GatedResidual(
            hc_count=hc,
            hidden_size=hs,
            lowrank=cfg.hc_lowrank,
            eps=cfg.rms_norm_eps,
        )
        self.mlp_hc = GatedResidual(
            hc_count=hc,
            hidden_size=hs,
            lowrank=cfg.hc_lowrank,
            eps=cfg.rms_norm_eps,
        )
        self.indexer = QSAIndexer(hidden_size=hs, rope_theta=cfg.rope_theta, eps=cfg.rms_norm_eps)
        self.attn = FlashNextQSAAttention(
            hidden_size=hs,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rotary_dim=int(cfg.head_dim * 0.25),
            rope_theta=cfg.rope_theta,
        )
        self.mlp = BF16MoE(
            hidden=hs,
            num_experts=cfg.num_experts,
            inter=cfg.moe_intermediate_size,
            top_k=cfg.num_experts_per_tok,
            expert_dtype=expert_dtype,
        )
        self.qsa_pad = 0
        self.decode_attn: QsaDecodeAttention | None = None

    def fuse(self, embeds: torch.Tensor, hc_hidden: torch.Tensor) -> torch.Tensor:
        """``embeds [T, hs]`` + ``hc_hidden [T, hc*hs]`` -> fused [T, hc*hs]."""
        e = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        h = self.pre_fc_norm_hidden(hc_hidden)
        hv = h.view(*h.shape[:-1], self.hc_count, self.hidden_size)
        enc = self.fc_hidden(hv)
        return (e.unsqueeze(-2) + enc).flatten(-2)

    def _store_sparse_reuse(
        self,
        sess,
        idx: torch.Tensor,
        valid: torch.Tensor,
        captured_len: int | torch.Tensor,
    ) -> None:
        graph_capture = (
            torch.is_tensor(captured_len)
            and captured_len.is_cuda
            and torch.cuda.is_current_stream_capturing()
        )
        # During graph capture these outputs are caller-owned static buffers;
        # retaining views keeps graph replay and eager fallback coherent and
        # avoids allocating/cloning a new graph-private tensor each capture.
        if graph_capture:
            sess.shared_sparse_indices = idx[-1:]
            sess.shared_sparse_valid = valid[-1:]
        else:
            sess.shared_sparse_indices = idx[-1:].clone()
            sess.shared_sparse_valid = valid[-1:].clone()
        if torch.is_tensor(captured_len):
            if captured_len.numel() != 1:
                raise ValueError(
                    f"captured_len tensor must be scalar-like, got {tuple(captured_len.shape)}"
                )
            sparse = sess.sparse_graph_buffers
            if sparse is not None:
                sparse.shared_indices.copy_(idx[-1:])
                sparse.shared_valid.copy_(valid[-1:])
                sparse.shared_captured_len.copy_(
                    captured_len.to(
                        device=sparse.shared_captured_len.device,
                        dtype=sparse.shared_captured_len.dtype,
                    ).reshape_as(sparse.shared_captured_len)
                )
            # A graph capture cannot perform a host scalar read.  Keep the
            # Python fallback mirror current whenever this is ordinary eager
            # execution; post-replay eager fallback reads the static buffer
            # when capture left the mirror unchanged.
            if not captured_len.is_cuda or not torch.cuda.is_current_stream_capturing():
                sess.shared_sparse_captured_len = int(captured_len.item())
            return
        sess.shared_sparse_captured_len = int(captured_len)
        sparse = sess.sparse_graph_buffers
        if sparse is not None:
            sparse.shared_indices.copy_(idx[-1:])
            sparse.shared_valid.copy_(valid[-1:])
            sparse.shared_captured_len.fill_(int(captured_len))

    def forward(
        self,
        embeds: torch.Tensor,
        hc_hidden: torch.Tensor,
        positions: torch.Tensor,
        sess,
        *,
        capture_sparse_indices: bool = False,
        reuse_sparse_indices: bool = False,
        graph_sparse_capacity: int | None = None,
        graph_dense_capacity: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(mixed [T, hs], own hc stream [T, hc*hs])``.

        The hc stream is the MTP layer's own residual stream -- feed it back
        as ``hc_hidden`` to chain drafts (K>=2)."""
        if embeds.dtype != self.fc_embedding.weight.dtype:
            embeds = embeds.to(self.fc_embedding.weight.dtype)
        if hc_hidden.dtype != self.fc_hidden.weight.dtype:
            hc_hidden = hc_hidden.to(self.fc_hidden.weight.dtype)
        x = self.fuse(embeds, hc_hidden)
        if not getattr(self, "_dtype_hooks", False):
            for name, mod in self.named_modules():
                if isinstance(mod, nn.Linear):
                    def hook(mod, inp, out, _name=name):
                        if inp[0].dtype != mod.weight.dtype:
                            raise RuntimeError(
                                f"dtype mismatch at {_name}: input "
                                f"{inp[0].dtype} vs weight {mod.weight.dtype}"
                            )
                    mod.register_forward_hook(hook)
            self._dtype_hooks = True
        mixed, residuals = self.attn_hc.mix(x)
        pos = positions
        q, k, v, gate = self.attn.project(mixed, positions)
        k_scales = getattr(sess, "mtp_k_scale_pool", None)
        v_scales = getattr(sess, "mtp_v_scale_pool", None)
        rows = int(pos.shape[0])
        if rows <= 0:
            raise ValueError("MTP QSA update requires at least one position")
        if _qsa_cache_is_quantized(sess.mtp_k_pool.dtype):
            k_rows = getattr(sess, "mtp_k_rows", None)
            v_rows = getattr(sess, "mtp_v_rows", None)
            k_scale_rows = getattr(sess, "mtp_k_scale_rows", None)
            v_scale_rows = getattr(sess, "mtp_v_scale_rows", None)
            if any(value is None for value in (k_rows, v_rows, k_scale_rows, v_scale_rows)):
                raise RuntimeError(
                    "quantized MTP QSA updates require graph-owned row scratch"
                )
            if rows > k_rows.shape[0]:
                # The proposal graph only needs K+1 rows, but teacher-forced
                # suffix sync can contain the entire prompt.  Grow an eager
                # temporary for that path; replacing the graph-owned scratch
                # during capture would invalidate captured addresses.
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError(
                        "MTP QSA row scratch is too small during graph capture: "
                        f"rows={rows}, capacity={k_rows.shape[0]}"
                    )
                scratch_shape = (rows, *k.shape[1:])
                k_rows = torch.empty(scratch_shape, dtype=k_rows.dtype, device=k.device)
                v_rows = torch.empty(scratch_shape, dtype=v_rows.dtype, device=v.device)
                scale_shape = (rows, *k_scale_rows.shape[1:])
                k_scale_rows = torch.empty(
                    scale_shape, dtype=k_scale_rows.dtype, device=k.device
                )
                v_scale_rows = torch.empty(
                    scale_shape, dtype=v_scale_rows.dtype, device=v.device
                )
            k_rows = k_rows[:rows]
            v_rows = v_rows[:rows]
            k_scale_rows = k_scale_rows[:rows]
            v_scale_rows = v_scale_rows[:rows]
            quantize_qsa_kv(k, k_rows, k_scale_rows)
            quantize_qsa_kv(v, v_rows, v_scale_rows)
            qsa_cache_index_copy_(sess.mtp_k_pool, pos, k_rows)
            qsa_cache_index_copy_(sess.mtp_v_pool, pos, v_rows)
            if k_scales is not None:
                k_scales.index_copy_(0, pos, k_scale_rows)
            if v_scales is not None:
                v_scales.index_copy_(0, pos, v_scale_rows)
        else:
            # ``pool[pos]`` is a copy for tensor ``pos``; index_copy_ both
            # preserves graph-stable addresses and commits the BF16 rows.
            sess.mtp_k_pool.index_copy_(0, pos, k)
            sess.mtp_v_pool.index_copy_(0, pos, v)

        def decode_sparse(idx, valid):
            # Decode gatherers pack valid lanes (including the partial tail)
            # at the front of the fixed row.  The valid-prefix count is the
            # only safe loop bound for sparse top-k rows; absolute positions
            # include blocks that were not selected.
            selected_counts = valid.to(torch.int32).sum(dim=-1)
            if k_scales is None and v_scales is None:
                return self.decode_attn(
                    q,
                    gate,
                    sess.mtp_k_pool,
                    sess.mtp_v_pool,
                    idx,
                    valid,
                    selected_counts=selected_counts,
                )
            return self.decode_attn(
                q,
                gate,
                sess.mtp_k_pool,
                sess.mtp_v_pool,
                idx,
                valid,
                k_scales,
                v_scales,
                selected_counts,
            )
        if graph_sparse_capacity is not None:
            sparse = sess.sparse_graph_buffers
            if sparse is None:
                raise RuntimeError("MTP sparse graph replay requires prepared sparse buffers")
            if positions.shape[0] > sparse.row_block_ends.shape[0]:
                raise ValueError(
                    "MTP sparse graph buffer rows are too small: "
                    f"positions={positions.shape[0]}, buffers={sparse.row_block_ends.shape[0]}"
                )
            if graph_sparse_capacity > sess.mtp_k_pool.shape[0]:
                raise ValueError(
                    "MTP sparse graph capacity exceeds cache capacity: "
                    f"{graph_sparse_capacity} > {sess.mtp_k_pool.shape[0]}"
                )
            if reuse_sparse_indices:
                idx, valid = self.indexer.batch_decode_reuse_indices_fixed(
                    sparse.shared_indices,
                    sparse.shared_valid,
                    positions,
                    sparse.shared_captured_len,
                    out_tokens=sparse.reuse_indices[: positions.shape[0]],
                    out_valid=sparse.reuse_valid[: positions.shape[0]],
                    tail_tokens=sparse.reuse_tail_indices[: positions.shape[0]],
                )
            else:
                qi, index_keys = self.indexer.project_qk(mixed, positions)
                self.indexer.update_index_cache_fixed(
                    sess.mtp_idx_k_pool,
                    sess.mtp_pooled_k_pool,
                    index_keys,
                    positions,
                )
                row_block_ends = sparse.row_block_ends[: positions.shape[0]]
                row_block_ends.copy_(
                    torch.div(
                        positions + 1,
                        self.indexer.compress_ratio,
                        rounding_mode="floor",
                    )
                )
                block_logits = self.indexer.score_blocks_fixed(
                    qi,
                    sess.mtp_pooled_k_pool,
                    row_block_ends,
                    out=sparse.block_logits[: positions.shape[0]],
                    column_ids=sparse.pooled_columns,
                )
                block_indices = self.indexer.select_blocks_fixed(
                    block_logits,
                    row_block_ends,
                    out=sparse.block_indices[: positions.shape[0]],
                )
                idx, valid = self.indexer.batch_decode_gather_indices_fixed(
                    block_indices,
                    positions,
                    self.qsa_pad,
                    out_tokens=sparse.gather_indices[: positions.shape[0]],
                    out_valid=sparse.gather_valid[: positions.shape[0]],
                    tail_tokens=sparse.tail_indices[: positions.shape[0]],
                )
                if capture_sparse_indices:
                    self._store_sparse_reuse(sess, idx, valid, positions[-1:] + 1)
            attn_out = decode_sparse(idx, valid)
            x = self.attn_hc.combine(attn_out, residuals)
            mixed2, res2 = self.mlp_hc.mix(x)
            x = self.mlp_hc.combine(self.mlp(mixed2, graph_direct=True), res2)
            mixed, _ = self.hyper_connection_mixer.mix(x)
            return mixed, x
        if graph_dense_capacity is not None:
            # Teacher-sync graphs must populate the indexer cache even though
            # short-prefix attention itself is dense.  Those real keys become
            # the compressed history when the session later crosses into the
            # sparse-QSA regime.
            _, index_keys = self.indexer.project_qk(mixed, positions)
            self.indexer.update_index_cache_fixed(
                sess.mtp_idx_k_pool,
                sess.mtp_pooled_k_pool,
                index_keys,
                positions,
            )
            attn_out = self.decode_attn.causal_prefix_fixed(
                q,
                gate,
                sess.mtp_k_pool,
                sess.mtp_v_pool,
                pos,
                graph_dense_capacity,
                k_scales,
                v_scales,
            )
            x = self.attn_hc.combine(attn_out, residuals)
            mixed2, res2 = self.mlp_hc.mix(x)
            x = self.mlp_hc.combine(self.mlp(mixed2, graph_direct=True), res2)
            mixed, _ = self.hyper_connection_mixer.mix(x)
            return mixed, x

        visible_end = int(pos[-1].item()) + 1
        dense_prefix = visible_end <= self.indexer.block_topk * self.indexer.compress_ratio
        if reuse_sparse_indices:
            shared_indices, shared_valid = _shared_sparse_reuse(sess)
            if not dense_prefix and (shared_indices is None or shared_valid is None):
                raise RuntimeError("MTP sparse-index reuse requires a captured sync row")
            if not dense_prefix and pos.numel() != 1:
                raise ValueError("MTP sparse-index reuse currently requires one decode row")
            if not dense_prefix:
                current_pos = int(pos.item())
                captured_len = _shared_sparse_captured_len(sess)
                if current_pos < captured_len:
                    raise RuntimeError(
                        "MTP sparse-index position precedes captured prefix: "
                        f"position={current_pos}, captured_len={captured_len}"
                    )
                # Captured rows keep a fixed storage width for CUDA Graphs;
                # eager continuation can compact the valid prefix before
                # appending the newly visible dense tail.  Leaving the
                # padding gap in the concatenated row would make the
                # selected-count loop bound skip that tail.
                shared_indices = shared_indices.masked_select(shared_valid).reshape(1, -1)
                shared_valid = torch.ones_like(shared_indices, dtype=torch.bool)
                tail = torch.arange(
                    captured_len,
                    current_pos + 1,
                    dtype=torch.long,
                    device=pos.device,
                ).unsqueeze(0)
                idx = torch.cat([shared_indices, tail], dim=1)
                valid = torch.cat(
                    [
                        shared_valid,
                        torch.ones_like(tail, dtype=torch.bool),
                    ],
                    dim=1,
                )
        else:
            qi, ki = self.indexer.project_qk(mixed, positions)
            idx_pool = sess.mtp_idx_k_pool
            self.indexer.update_index_cache_eager(
                idx_pool,
                sess.mtp_pooled_k_pool,
                ki,
                start=int(pos[0].item()),
            )
            pooled = sess.mtp_pooled_k_pool[: visible_end // self.indexer.compress_ratio]
            ends = torch.clamp((pos - 3) // self.indexer.compress_ratio + 1, min=0)
            # A long teacher-sync suffix can otherwise allocate the complete
            # [rows, pooled_keys] FP32 score matrix at once.  Keep its peak
            # bounded to the same 128 MiB row-tiled budget as target QSA
            # prefill; the per-row top-k result is exactly unchanged.
            workspace_mb = int(
                os.environ.get("QSR_FLASHNEXT_MTP_SCORE_WORKSPACE_MB", "128")
            )
            if workspace_mb <= 0:
                raise ValueError(
                    "QSR_FLASHNEXT_MTP_SCORE_WORKSPACE_MB must be positive, "
                    f"got {workspace_mb}"
                )
            bounded_selector = getattr(self.indexer, "select_blocks_bounded", None)
            if bounded_selector is None:
                # Keep small protocol/test doubles source-compatible.  The
                # production QSAIndexer always takes the bounded path.
                scores = self.indexer.score_blocks(qi, pooled, ends)
                blocks = self.indexer.select_blocks(scores, ends)
            else:
                blocks = bounded_selector(
                    qi,
                    pooled,
                    ends,
                    logits_workspace_bytes=workspace_mb * 1024 * 1024,
                )
            idx, valid = self.indexer.batch_decode_gather_indices(
                blocks, pos, self.qsa_pad
            )
            if capture_sparse_indices:
                self._store_sparse_reuse(sess, idx, valid, int(pos[-1].item()) + 1)
        if dense_prefix:
            attn_out = self.decode_attn.causal_prefix(
                q,
                gate,
                sess.mtp_k_pool,
                sess.mtp_v_pool,
                pos,
                k_scales,
                v_scales,
            )
        else:
            attn_out = decode_sparse(idx, valid)
        x = self.attn_hc.combine(attn_out, residuals)
        mixed2, res2 = self.mlp_hc.mix(x)
        x = self.mlp_hc.combine(self.mlp(mixed2), res2)
        mixed, _ = self.hyper_connection_mixer.mix(x)
        return mixed, x


def load_flashnext_mtp(
    ckpt: pathlib.Path | str, cfg, main_model, device: str = "cuda"
) -> FlashNextMTP:
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt)
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    def load(name: str) -> torch.Tensor:
        with safe_open(str(ckpt / weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    expert_dtype = mtp_expert_weight_dtype()
    m = FlashNextMTP(cfg, expert_dtype=expert_dtype)
    with torch.no_grad():
        m.pre_fc_norm_embedding.weight.copy_(load("mtp.pre_fc_norm_embedding.weight"))
        m.pre_fc_norm_hidden.weight.copy_(load("mtp.pre_fc_norm_hidden.weight"))
        m.fc_embedding.weight.copy_(load("mtp.fc_embedding.weight"))
        m.fc_hidden.weight.copy_(load("mtp.fc_hidden.weight"))
        p = "mtp.hyper_connection_mixer"
        m.hyper_connection_mixer.hc_norm.weight.copy_(load(f"{p}.hc_norm.weight"))
        m.hyper_connection_mixer.input_mix_weight_down.weight.copy_(
            load(f"{p}.input_mix_weight_down.weight")
        )
        m.hyper_connection_mixer.input_mix_weight_up.weight.copy_(
            load(f"{p}.input_mix_weight_up.weight")
        )
        p = "mtp.layers.0.attn_hyper_connection"
        m.attn_hc.hc_norm.weight.copy_(load(f"{p}.hc_norm.weight"))
        m.attn_hc.input_mix_weight_down.weight.copy_(load(f"{p}.input_mix_weight_down.weight"))
        m.attn_hc.input_mix_weight_up.weight.copy_(load(f"{p}.input_mix_weight_up.weight"))
        m.attn_hc.block_inject_weight.weight.copy_(load(f"{p}.block_inject_weight.weight"))
        p = "mtp.layers.0.mlp_hyper_connection"
        m.mlp_hc.hc_norm.weight.copy_(load(f"{p}.hc_norm.weight"))
        m.mlp_hc.input_mix_weight_down.weight.copy_(load(f"{p}.input_mix_weight_down.weight"))
        m.mlp_hc.input_mix_weight_up.weight.copy_(load(f"{p}.input_mix_weight_up.weight"))
        m.mlp_hc.block_inject_weight.weight.copy_(load(f"{p}.block_inject_weight.weight"))
        p = "mtp.layers.0.self_attn.indexer"
        m.indexer.index_qk_proj.weight.copy_(load(f"{p}.index_qk_proj.weight"))
        m.indexer.q_layernorm.copy_(load(f"{p}.q_layernorm.weight"))
        m.indexer.k_layernorm.copy_(load(f"{p}.k_layernorm.weight"))
        p = "mtp.layers.0.self_attn"
        m.attn.q_proj.weight.copy_(load(f"{p}.q_proj.weight"))
        m.attn.k_proj.weight.copy_(load(f"{p}.k_proj.weight"))
        m.attn.v_proj.weight.copy_(load(f"{p}.v_proj.weight"))
        m.attn.o_proj.weight.copy_(load(f"{p}.o_proj.weight"))
        m.attn.q_norm.copy_(load(f"{p}.q_norm.weight"))
        m.attn.k_norm.copy_(load(f"{p}.k_norm.weight"))
        p = "mtp.layers.0.mlp"
        m.mlp.gate.weight.copy_(load(f"{p}.gate.weight"))
        gate_up = load(f"{p}.experts.gate_up_proj")
        down = load(f"{p}.experts.down_proj")
        if m.mlp.expert_is_fp8:
            gate_up_q, gate_up_scale = quantize_mtp_expert_weight(
                gate_up,
                dtype=expert_dtype,
            )
            down_q, down_scale = quantize_mtp_expert_weight(
                down,
                dtype=expert_dtype,
            )
            m.mlp.set_fp8_expert_weights(
                gate_up_q,
                gate_up_scale,
                down_q,
                down_scale,
            )
            del gate_up_q, gate_up_scale, down_q, down_scale
        else:
            m.mlp.gate_up_proj.copy_(gate_up)
            m.mlp.down_proj.copy_(down)
        del gate_up, down
        m.mlp.shared_gate.weight.copy_(load(f"{p}.shared_expert.gate_proj.weight"))
        m.mlp.shared_up.weight.copy_(load(f"{p}.shared_expert.up_proj.weight"))
        m.mlp.shared_down.weight.copy_(load(f"{p}.shared_expert.down_proj.weight"))
        m.mlp.shared_gate_lin.weight.copy_(load(f"{p}.shared_expert_gate.weight"))
    if m.mlp.expert_is_fp8:
        # ``Module.to(dtype=...)`` would convert the FP8 expert buffers back
        # to BF16.  Move first, then cast only registered Parameters; the
        # inference-only expert buffers and their FP32 scales stay compact.
        m.to(device)
        for parameter in m.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    else:
        m.to(device, torch.bfloat16)
    return m
