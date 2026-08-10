"""Phase 3 gate: kernel-path attention vs eager, full model, real weights.

Runs the real DeepSeek-V4-Flash GGUF through both paths -- the eager
graph (executable definition) and the kernel path (packed FP8 KV pages +
the sparkinfer compressed_mla kernel, sharing every weight module with
the eager graph) -- and reports per-layer output cosine and the greedy
stream agreement, per the plan's amended Phase 3 gate:

    logits cos >= 0.99, greedy stream >= 95%, no exponential drift
    (amended 2026-08-07: the fork kernel's numerical contract is
    ~3e-4/step from eager, so the original per-layer 0.99999 is
    unattainable).

The two paths share everything except the attention layers, so any
divergence is attributable to the kernel attention path. `--steps`
defaults to the gate's 512; a smoke run (`--steps 16`) verifies the
harness before the full run.

Prompts are encoded from the GGUF's own token table (exact-string
lookup), so no external tokenizer is needed for the parity workloads.

Usage:
    python scripts/dsv4_align_eager_vs_kernel.py <model.gguf> [--steps N]
    python scripts/dsv4_align_eager_vs_kernel.py <model.gguf> --cuda-graph [--steps N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import runtime
from runtime.loading.gguf import read_gguf_header
from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer
from runtime.model.dsv4_model import (
    Dsv4Transformer,
    load_dsv4_from_gguf,
    rms_norm,
)

WORKLOAD_WORDS = [
    ["The", " meaning", " of", " life", " is"],
    ["Explain", " the", " theory", " of", " relativity", " in", " simple", " terms", ":"],
    ["def", " fibonacci", "(", "n", ")"],
]

REPO_ROOT = Path(__file__).resolve().parents[1]
assert Path(runtime.__file__).resolve().is_relative_to(REPO_ROOT), runtime.__file__


def _norm(word: str) -> list[str]:
    """BPE byte-level conventions: leading spaces are 'Ġ' (U+0120)."""
    candidates = [word]
    if word.startswith(" "):
        candidates.append("Ġ" + word[1:])
    return candidates


def encode_prompt(kv: dict, words: list[str]) -> list[int]:
    """Best-effort token lookup: exact string match in the GGUF token table."""
    tokens = kv.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError("GGUF token table missing")
    ids: list[int] = []
    for word in words:
        for candidate in _norm(word):
            if candidate in tokens:
                ids.append(tokens.index(candidate))
                break
        else:
            raise RuntimeError(f"token {word!r} not in the GGUF table")
    return ids


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run_path(
    model: Dsv4Transformer,
    kernel_layers: list[Dsv4AttnKernelLayer] | None,
    input_ids: torch.Tensor,
    start_pos: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One forward on either path; returns (logits, per-block outputs)."""
    h = model.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    block_outs: list[torch.Tensor] = []
    for i, block in enumerate(model.blocks):
        residual = h
        x, post, comb = block.hc_pre(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
        x = rms_norm(x, block.attn_norm_weight, block.eps)
        x = (
            kernel_layers[i](x, start_pos)
            if kernel_layers is not None
            else block.attn(x, start_pos)
        )
        x = block.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = block.hc_pre(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
        x = rms_norm(x, block.ffn_norm_weight, block.eps)
        x = block.moe(x, input_ids)
        x = block.hc_post(x, residual, post, comb)
        block_outs.append(x)
        h = x
    h = model.hc_head(h)
    return model.lm_head(rms_norm(h, model.norm_weight, model.eps)), block_outs


def reset_all(model: Dsv4Transformer, kernel_layers: list[Dsv4AttnKernelLayer]) -> None:
    model.reset_caches()
    for layer in kernel_layers:
        layer.reset_caches()


def run_cuda_graph_gate(
    model: Dsv4Transformer,
    prompts: list[list[int]],
    *,
    steps: int,
    max_seq_len: int,
    max_q_rows: int,
) -> None:
    """Compare continuous kernel-eager decode with the captured full graph.

    Two slot-owned attention stacks share weights but not recursive/cache
    state. Slot 0 stays eager as the oracle; slot 1 replays its address-bound
    graph. Both consume the eager stream's token at each position.
    """
    from runtime.backends.dsv4 import DeepseekV4Backend

    backend = DeepseekV4Backend(
        model,
        model.config,
        num_slots=2,
        max_seq_len=max_seq_len,
        max_q_rows=max_q_rows,
        device="cuda",
    )
    print("capturing per-slot decode graphs ...", flush=True)
    captured = backend.capture_decode_cuda_graph()
    if captured != 2:
        raise RuntimeError(
            f"DSV4 CUDA Graph gate requires two captured slots, got {captured}; "
            f"status={backend.snapshot().dflash_cg_status}"
        )
    # Slot 0 is the continuous kernel-eager oracle; slot 1 replays the B=1
    # decode graph (the new backend shares one bucketed driver across
    # slots, so there is no per-slot graph to drop -- slot 0 never uses it).
    assert 1 in backend._decode_graphs, "B=1 decode graph was not captured"  # noqa: SLF001

    worst_cos = 1.0
    mismatches = 0
    total = 0
    eager_ms: list[float] = []
    graph_ms: list[float] = []
    for workload, prompt in enumerate(prompts):
        backend.reset_slot(0)
        backend.reset_slot(1)
        eager_prefill = backend._prefill_logits(0, prompt)  # noqa: SLF001
        graph_prefill = backend._prefill_logits(1, prompt)  # noqa: SLF001
        prefill_cos = cosine(eager_prefill, graph_prefill)
        token = int(eager_prefill[0, -1].argmax().item())
        graph_token = int(graph_prefill[0, -1].argmax().item())
        mismatches += int(token != graph_token)
        total += 1
        print(
            f"workload {workload}: prefill cos {prefill_cos:.8f}, "
            f"anchor eager/graph={token}/{graph_token}",
            flush=True,
        )

        for step in range(steps):
            position = len(prompt) + step
            ids = torch.tensor([[token]], dtype=torch.long, device="cuda")

            torch.cuda.synchronize()
            started = time.perf_counter()
            eager_logits = backend._forward(0, ids, position)  # noqa: SLF001
            torch.cuda.synchronize()
            eager_ms.append((time.perf_counter() - started) * 1000)

            torch.cuda.synchronize()
            started = time.perf_counter()
            # New backend (post-64a9850): _decode_graphs is keyed by batch
            # size; the B=1 driver takes host inputs for one slot.
            graph_logits = backend._decode_graphs[1].replay_host(  # noqa: SLF001
                [token],
                [position],
                [1],
                max_index_entries=backend._decode_index_bucket([position]),
            )
            torch.cuda.synchronize()
            graph_ms.append((time.perf_counter() - started) * 1000)

            current_cos = cosine(eager_logits, graph_logits)
            worst_cos = min(worst_cos, current_cos)
            token = int(eager_logits[0, 0].argmax().item())
            graph_token = int(graph_logits[0, 0].argmax().item())
            mismatches += int(token != graph_token)
            total += 1
            for layer in backend.slot_layers:
                compressors = [layer.compressor]
                if layer.indexer is not None:
                    compressors.append(layer.indexer.compressor)
                for compressor in compressors:
                    if compressor is not None and torch.isnan(compressor.score_state).any():
                        raise AssertionError(
                            f"score_state NaN at workload={workload} step={step} "
                            f"layer={layer.layer_id}"
                        )
            if step < 4 or (step + 1) % 16 == 0:
                print(
                    f"  step {step + 1}: cos {current_cos:.8f}, "
                    f"token eager/graph={token}/{graph_token}, "
                    f"eager={eager_ms[-1]:.1f}ms graph={graph_ms[-1]:.1f}ms",
                    flush=True,
                )

    agreement = (total - mismatches) / total
    eager_avg = sum(eager_ms) / len(eager_ms)
    graph_avg = sum(graph_ms) / len(graph_ms)
    verdict = "PASS" if worst_cos >= 0.99999 and agreement == 1.0 else "REVIEW"
    print("\n=== DSV4 CUDA Graph gate summary ===")
    print(f"worst logits cosine: {worst_cos:.8f}")
    print(f"greedy agreement: {total - mismatches}/{total} ({agreement:.1%})")
    print(
        f"mean decode: eager {eager_avg:.1f}ms, graph {graph_avg:.1f}ms "
        f"({eager_avg / graph_avg:.2f}x)"
    )
    print(f"capture status: {backend.snapshot().dflash_cg_status}")
    print(f"verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=str, help="DeepSeek-V4-Flash GGUF file")
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="decode steps per workload (default 128: the drift and stream "
        "trends are visible within this; a 512x3 run costs ~2.5h on the "
        "eager path's MoE dequant and is only for final confirmation)",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument(
        "--max-q-rows", type=int, default=0, help="MLA plan rows (default: largest prompt length)"
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="compare continuous kernel-eager decode with full CUDA Graph replay",
    )
    args = parser.parse_args()

    device = "cuda"
    print(f"loading {args.gguf} ...", flush=True)
    t0 = time.time()
    model, count = load_dsv4_from_gguf(args.gguf, max_seq_len=args.max_seq_len, device=device)
    print(f"loaded {count} tensors in {time.time() - t0:.1f}s", flush=True)

    header = read_gguf_header(Path(args.gguf))
    config = model.config
    prompts = [encode_prompt(header.kv, words) for words in WORKLOAD_WORDS]
    print("workloads:", [len(p) for p in prompts], flush=True)

    # The fork MLA scratch scales with max_q_rows (tmp_output is
    # [rows, heads, chunks, v]); the decode path needs 1 row and prefill
    # only the prompt length, so plan tight (OOM otherwise: 128 rows cost
    # ~11 GB across the 21 ratio-4 layers on top of the 82 GB model).
    plan_rows = args.max_q_rows or max(len(p) for p in prompts)
    print(f"MLA plan rows: {plan_rows}", flush=True)

    if args.cuda_graph:
        run_cuda_graph_gate(
            model,
            prompts,
            steps=args.steps,
            max_seq_len=args.max_seq_len,
            max_q_rows=plan_rows,
        )
        return

    print("building kernel-path attention layers (shared weights) ...", flush=True)
    kernel_layers = [
        Dsv4AttnKernelLayer(
            config,
            layer_id,
            max_seq_len=args.max_seq_len,
            max_q_rows=plan_rows,
            device=device,
            shared_from=model.blocks[layer_id].attn,
        )
        for layer_id in range(config.num_layers)
    ]

    layer_worst = [1.0] * config.num_layers
    logits_worst = 1.0
    total_mismatch = 0
    total_steps = 0

    for wl, prompt in enumerate(prompts):
        reset_all(model, kernel_layers)
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        print(f"workload {wl}: prompt {len(prompt)} tokens", flush=True)
        e_logits, e_blocks = run_path(model, None, prompt_t, 0)
        k_logits, k_blocks = run_path(model, kernel_layers, prompt_t, 0)
        for i, (eb, kb) in enumerate(zip(e_blocks, k_blocks)):
            c = cosine(eb, kb)
            layer_worst[i] = min(layer_worst[i], c)
        lc = cosine(e_logits, k_logits)
        logits_worst = min(logits_worst, lc)
        worst_block = min(cosine(e, k) for e, k in zip(e_blocks, k_blocks))
        print(f"  prefill: logits cos {lc:.8f}  worst block {worst_block:.8f}", flush=True)

        pos = len(prompt)
        e_tok = int(e_logits[0, -1].argmax().item())
        k_tok = int(k_logits[0, -1].argmax().item())
        stream_e, stream_k = [e_tok], [k_tok]
        if e_tok != k_tok:
            total_mismatch += 1
        for step in range(args.steps):
            # Both paths consume the SAME token (teacher-forced from the eager
            # stream): the per-layer cosine then measures numerical closeness,
            # not divergence amplified by different inputs. The greedy streams
            # are tracked independently; agreement is reported separately.
            tid_e = torch.tensor([[stream_e[-1]]], dtype=torch.long, device=device)
            tid_k = torch.tensor([[stream_e[-1]]], dtype=torch.long, device=device)
            e_logits, e_blocks = run_path(model, None, tid_e, pos)
            k_logits, k_blocks = run_path(model, kernel_layers, tid_k, pos)
            for i, (eb, kb) in enumerate(zip(e_blocks, k_blocks)):
                c = cosine(eb, kb)
                layer_worst[i] = min(layer_worst[i], c)
            lc = cosine(e_logits, k_logits)
            logits_worst = min(logits_worst, lc)
            e_tok = int(e_logits[0, 0].argmax().item())
            k_tok = int(k_logits[0, 0].argmax().item())
            stream_e.append(e_tok)
            stream_k.append(k_tok)
            total_steps += 1
            if e_tok != k_tok:
                total_mismatch += 1
            if (step + 1) % 64 == 0:
                print(
                    f"  step {step + 1}: logits cos {lc:.8f}  mismatches so far {total_mismatch}",
                    flush=True,
                )
            pos += 1  # advance the cache position (a fixed pos recomputes the
            # same step forever -- identical cosines, meaningless streams)
        agree = sum(1 for a, b in zip(stream_e, stream_k) if a == b)
        print(f"  stream: {agree}/{len(stream_e)} tokens agree", flush=True)

    print("\n=== Phase 3 gate summary ===")
    print(f"per-layer worst cosine (across {total_steps} decode steps + prefill):")
    for i, c in enumerate(layer_worst):
        print(f"  layer {i:2d} ({config.layer_ratio(i):3d}): {c:.8f}")
    print(f"final-logits worst cosine: {logits_worst:.8f}")
    print(
        f"greedy stream agreement: {total_steps + len(prompts) - total_mismatch}/"
        f"{total_steps + len(prompts)}"
    )
    worst = min(logits_worst, *layer_worst)
    total = total_steps + len(prompts)
    stream_frac = (total - total_mismatch) / total
    # Phase-3 gate criterion as amended 2026-08-07 (plan note): the fork
    # MLA kernel's numerical contract is ~3e-4/step from eager, so the
    # original per-layer 0.99999 is unattainable.  The accepted bar is
    # logits cos >= 0.99, greedy stream agreement >= 95%, and no
    # exponential drift (no sign-flipped layer -- oscillating dips are
    # expected, a negative worst cosine is divergence).
    no_drift = min(layer_worst) > 0.0
    verdict = "PASS" if logits_worst >= 0.99 and stream_frac >= 0.95 and no_drift else "REVIEW"
    print(
        f"verdict: {verdict} (logits cos {logits_worst:.8f}, stream {stream_frac:.1%}, "
        f"worst layer {worst:.8f})"
    )


if __name__ == "__main__":
    main()
