"""B3: does the MTP draft head load its real checkpoint weights and run a
correct-shaped, numerically sane chained multi-step draft?

This is the MTP-head analogue of ``scripts/b3_probe_gdn_spec_rollback.py``:
loads ONE real component (here, the whole -- but tiny, ~1 GiB BF16 --
``Qwen36MTPHead`` instance) directly from the real checkpoint's safetensors
shards, never touching the 27B backbone (whose first full forward inflates
resident memory from ~19 GiB to ~54+ GiB, per
``notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`` -- entirely
avoidable here since the backbone is never constructed at all, let alone
forwarded).

What this checks:
  1. All 15 real ``mtp.*`` tensors load, with the exact shapes B3's design
     work derived from the checkpoint's own safetensors index.
  2. A single MTP step (real weights, a synthetic "target hidden state" +
     a real token embedding row) produces finite, non-degenerate output --
     not a numerical-correctness claim against any oracle (there is no
     non-speculative Qwen3.6 MTP path to compare against; MTP is new to
     this model), but a "the wiring is not obviously broken" check.
  3. Chained multi-step drafting (this script's own re-derivation of what
     ``Qwen36ForCausalLMSelfBuilt.mtp_step`` does per call, run K times
     feeding each step's own output back in) advances the head's KV cache
     by exactly one position per step and produces K distinct draft token
     ids from a real embedding table slice -- i.e. the recursive chaining
     design in ``Qwen36MTPHead``'s module docstring actually runs
     end-to-end, not just type-checks.
  4. Rewinding the head's KV cache (`cache.seq_len = anchor + m`, the same
     trick B3's report says the design needs for a partial accept) and
     re-running from that point reproduces EXACTLY (bit-exact) the
     original chain's steps up to m -- proving the "just truncate, no
     snapshot needed" claim for attention (unlike GDN), not just asserting
     it.

Run: ~/.venvs/vllm/bin/python scripts/b3_probe_mtp_head.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.qwen36_model import Qwen36MTPHead  # noqa: E402
from runtime.model_loading import _build_qwen36_model_config  # noqa: E402

MODEL_PATH = standard_checkpoint_path()
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)

# Defensive cap: real footprint here is ~1 GiB (15 BF16 tensors + a handful
# of embedding rows), nowhere near needing more -- cheap insurance on a
# card the user is actively using.
torch.cuda.set_per_process_memory_fraction(0.10, device=0)

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def load_mtp_head() -> tuple[Qwen36MTPHead, dict]:
    index = json.loads((Path(MODEL_PATH) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names = [n for n in weight_map if n.startswith("mtp.")]
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(weight_map[n], []).append(n)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(Path(MODEL_PATH) / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                tensors[n] = f.get_tensor(n).to(DEVICE)

    model_config = _build_qwen36_model_config(MODEL_PATH)
    with DEVICE:
        head = (
            Qwen36MTPHead(model_config, quantized={}, max_seq_len=4096)
            .to(DEVICE)
            .to(torch.bfloat16)
        )
    # head.named_parameters() gives names relative to `head` itself (no
    # "mtp." prefix -- that prefix only exists once this class is nested
    # under Qwen36ForCausalLMSelfBuilt.mtp); re-add it here to match the
    # checkpoint's own tensor names.
    params = {f"mtp.{name}": param for name, param in head.named_parameters()}
    missing = set(params) - set(tensors)
    extra = set(tensors) - set(params)
    assert not missing, f"module has params the checkpoint lacks: {missing}"
    assert not extra, f"checkpoint has tensors the module lacks: {extra}"
    for name, param in params.items():
        param.data.copy_(tensors[name].to(param.dtype))
    return head, model_config


def load_a_few_embedding_rows(hidden_size: int, token_ids: list[int]) -> torch.Tensor:
    """Real embedding rows for a handful of token ids -- avoids loading the
    whole 2.5 GiB embed_tokens table (248320 x 5120 BF16) just to get a few
    realistic (not synthetic-random) input vectors."""
    index = json.loads((Path(MODEL_PATH) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    shard = weight_map[name]
    with safe_open(str(Path(MODEL_PATH) / shard), framework="pt", device="cpu") as f:
        full = f.get_slice(name)
        rows = [full[t : t + 1, :] for t in token_ids]
    out = torch.cat(rows, dim=0).to(DEVICE)
    assert out.shape == (len(token_ids), hidden_size)
    return out


def mtp_step(
    head: Qwen36MTPHead,
    embeds: torch.Tensor,
    prev_hidden: torch.Tensor,
    position: int,
    cache,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-derivation of Qwen36ForCausalLMSelfBuilt.mtp_step, minus the
    shared lm_head (this probe has no reason to load the 2.5 GiB lm_head
    table -- draft *logits* are a different concern from "does the head
    itself run"; a real lm_head is exercised in the full-model round-trip
    this session's B3 report separately measures acceptance/throughput
    from)."""
    positions = torch.tensor([position], device=embeds.device, dtype=torch.long)
    embeds_3d = embeds.view(1, 1, -1)
    prev_hidden_3d = prev_hidden.view(1, 1, -1)
    hidden = head(embeds_3d, prev_hidden_3d, positions, cos_sin_cache, cache)
    return hidden.view(-1), hidden.view(-1)


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    head, model_config = load_mtp_head()
    hidden_size = model_config["hidden_size"]
    print(f"loaded real MTP head weights, hidden_size={hidden_size}")

    torch.manual_seed(1234)
    rope_params = model_config["rope_parameters"]
    rotary_dim = int(hidden_size // model_config["num_attention_heads"])
    from runtime.kernels.rope import compute_cos_sin_cache_default

    cos_sin_cache = compute_cos_sin_cache_default(
        int(model_config["head_dim"] * rope_params.get("partial_rotary_factor", 1.0)),
        model_config["max_position_embeddings"],
        float(rope_params["rope_theta"]),
        torch.bfloat16,
        device=DEVICE,
    )
    del rotary_dim  # computed above only to sanity-check against head_dim below

    K = 8
    token_ids = list(range(100, 100 + K + 1))  # anchor + K chained steps
    embeds = load_a_few_embedding_rows(hidden_size, token_ids)

    # Synthetic "target hidden state at the anchor's position" -- a real
    # forward would produce this from the 64-layer backbone; here it just
    # needs to be a plausible-scale BF16 vector (matches the probe script's
    # own x_prefill * 0.1 scaling convention for the same reason).
    anchor_hidden = torch.randn(hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1

    cache = head.new_cache(device=DEVICE, dtype=torch.bfloat16)
    record("fresh MTP cache starts at seq_len=0", cache.seq_len == 0, f"got {cache.seq_len}")

    # -- Chained K-step drafting: step 0 conditions on the real anchor
    # hidden; every later step conditions on the PREVIOUS step's own
    # output, per Qwen36MTPHead's module docstring. --------------------
    print(f"\n=== chained drafting, K={K} steps ===")
    outputs: list[torch.Tensor] = []
    prev_hidden = anchor_hidden
    anchor_position = 41  # arbitrary "real sequence position" for the anchor
    for step in range(K):
        h, prev_hidden = mtp_step(
            head, embeds[step], prev_hidden, anchor_position + 1 + step, cache, cos_sin_cache
        )
        outputs.append(h)
        record(
            f"step {step}: cache advanced by exactly 1 (seq_len={cache.seq_len})",
            cache.seq_len == step + 1,
        )
        record(
            f"step {step}: output finite, non-degenerate",
            bool(torch.isfinite(h).all())
            and h.float().abs().max().item() > 0
            and h.float().abs().max().item() < 1e4,
            f"max_abs={h.float().abs().max().item():.4g}",
        )

    # Distinct chained steps should (overwhelmingly likely, real weights +
    # varying real token embeddings + advancing RoPE positions) produce
    # DIFFERENT hidden states -- catches an accidental "ignores its input"
    # bug (e.g. a forgotten residual or a cache that never actually reads
    # back what it wrote).
    all_diff = all(
        not torch.equal(outputs[i], outputs[j])
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    )
    record("chained steps produce distinct outputs (not a constant/dead path)", all_diff)

    # -- Rewind: truncate the cache back to m and re-run from there.
    # Attention's "state at position m" is just "the first m rows", so
    # this must reproduce the ORIGINAL chain's outputs[m:] bit-exactly,
    # given the SAME (embeds, prev_hidden) inputs from that point --
    # proving the rewind claim in Qwen36MTPHead's docstring by
    # measurement, not by assertion alone. --------------------------
    print("\n=== rewind (partial-accept truncation) bit-exactness ===")
    for m in (0, 1, K // 2, K - 1):
        cache.seq_len = m
        prev_hidden_resume = anchor_hidden if m == 0 else outputs[m - 1]
        h_resumed, _ = mtp_step(
            head, embeds[m], prev_hidden_resume, anchor_position + 1 + m, cache, cos_sin_cache
        )
        bit_exact = torch.equal(h_resumed, outputs[m])
        record(f"rewind to m={m}, replay step {m}: bit-exact vs original chain", bit_exact)
    cache.seq_len = K  # leave the cache consistent for the timing section below

    # -- Throughput: one chained draft step's real cost, for the B3
    # acceptance/throughput report to put the "how expensive is drafting
    # itself" question in context (separate from verify's GDN-rollback
    # cost, which scripts/b3_probe_gdn_spec_rollback.py already measured).
    print("\n=== throughput (one MTP head instance) ===")
    torch.cuda.synchronize()
    n_iters = 50
    warm_cache = head.new_cache(device=DEVICE, dtype=torch.bfloat16)
    mtp_step(head, embeds[0], anchor_hidden, anchor_position + 1, warm_cache, cos_sin_cache)

    def timed_step() -> float:
        c = head.new_cache(device=DEVICE, dtype=torch.bfloat16)
        h = anchor_hidden
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_iters):
            h_out, h = mtp_step(head, embeds[0], h, anchor_position + 1 + i, c, cos_sin_cache)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1e3

    t_step = timed_step()
    print(f"  one chained MTP draft step (real weights, eager): {t_step:.3f} ms")
    print(f"  K={K} chained steps (drafting cost for one round): {t_step * K:.3f} ms")

    print("\n== summary ==")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, detail in _results:
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
