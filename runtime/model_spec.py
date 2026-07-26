"""E1 Phase 1: ModelSpec — explicit model architecture configuration.

Extracts the implicit Qwen3.6-27B assumptions from DirectModelRunner into
a frozen dataclass. The runner accesses architecture parameters via
``self.spec`` instead of scattered config lookups.

This is the first step toward multi-model support (E1 Phase 3) and enables
B5 MTP methods extraction (Phase 2) by making the model boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Frozen model architecture specification.

    Created once at runner initialization from the vLLM config and
    static_forward_context discovery. All fields are read-only after
    construction.
    """

    # Identity
    model_id: str
    architecture: str

    # Layer structure (discovered from static_forward_context)
    attn_layer_names: tuple[str, ...]
    gdn_layer_names: tuple[str, ...]

    # MTP draft model (None if speculative decoding not configured)
    mtp_model_id: str | None = None
    mtp_attn_layer_names: tuple[str, ...] = ()
    num_speculative_tokens: int = 0

    # KV cache configuration
    kv_dtype: str = "fp8_e4m3"
    block_size: int = 64

    # Derived properties
    @property
    def num_attn_layers(self) -> int:
        return len(self.attn_layer_names)

    @property
    def num_gdn_layers(self) -> int:
        return len(self.gdn_layer_names)

    @property
    def num_layers(self) -> int:
        return self.num_attn_layers + self.num_gdn_layers

    @property
    def has_mtp(self) -> bool:
        return self.mtp_model_id is not None and self.num_speculative_tokens > 0

    @property
    def verify_qo_len(self) -> int:
        """qo_len for MTP verify forward (K+1 positions)."""
        return self.num_speculative_tokens + 1

    @classmethod
    def from_runner_init(
        cls,
        model_id: str,
        architecture: str,
        attn_layer_names: list[str],
        gdn_layer_names: list[str],
        mtp_model_id: str | None = None,
        mtp_attn_layer_names: list[str] | None = None,
        num_speculative_tokens: int = 0,
        kv_dtype: str = "fp8_e4m3",
        block_size: int = 64,
    ) -> ModelSpec:
        """Construct from runner initialization parameters."""
        return cls(
            model_id=model_id,
            architecture=architecture,
            attn_layer_names=tuple(attn_layer_names),
            gdn_layer_names=tuple(gdn_layer_names),
            mtp_model_id=mtp_model_id,
            mtp_attn_layer_names=tuple(mtp_attn_layer_names or []),
            num_speculative_tokens=num_speculative_tokens,
            kv_dtype=kv_dtype,
            block_size=block_size,
        )
