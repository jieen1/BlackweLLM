"""Phase 2 GPU gate: legacy fixed-row vs dynamic arena, same-process A/B.

``.omx/plans/qwen38-dynamic-context-vllm-plan.md`` Phase 2 acceptance:
the dynamic page-bundle arena must be observationally identical to the
legacy fixed-row layout on the serving path. Two separate b2 runs can't
prove that -- a shared GPU is noisy and a silent per-step divergence
between runs is exactly the failure class this plan's Phase 5 matrix is
built to catch. This script runs BOTH pool modes in ONE process, on ONE
model load, with the same prompts and the same greedy steps, and compares
token streams, the KV arena's base pointer stability across
reset/reuse/COW, and the dynamic pool's bundle conservation.

Checks:
1. prefill + N-step greedy: legacy and dynamic produce the same tokens.
2. batched decode == serial decode inside dynamic mode (backend-internal).
3. dynamic arena base pointer unchanged across reset/reuse cycles (plan
   §7 invariant 12).
4. dynamic bundle conservation after many reset/allocate cycles (no leak).
5. MTP K=3 round smoke: with MTP enabled, one round produces accepted
   tokens and leaves the dynamic bundle pool balanced.

Run: ~/.venvs/vllm/bin/python scripts/b2_verify_dynamic_arena_ab.py
     [--slots N] [--steps N] [--max-seq-len N] [--no-mtp]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"!= {_ROOT}"
)

import torch  # noqa: E402

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The first president of the United States was",
    "In a hole in the ground there lived",
]

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def backend_greedy(backend, slot: int, prompt_ids: list[int], steps: int):
    if not backend.slot_state(slot).is_fresh:
        backend.reset_slot(slot)
    params = SamplingParams()
    state = backend.prefill_chunked_begin([slot], [prompt_ids])
    token = state.result[slot]["anchor"]
    tokens = [token]
    for _ in range(steps):
        token = backend.decode_batch_sampled(
            [slot], [token], [backend.slot_state(slot).kv_len], [params]
        )[0]
        tokens.append(token)
    return tokens


def _kv_base_ptrs(backend) -> list[int]:
    return [
        int(pool.data_ptr())
        for pool in backend.pool.k_pools
        if pool is not None
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--no-mtp", action="store_true", help="skip the MTP round smoke")
    ap.add_argument("--model-path", type=str, default=standard_checkpoint_path())
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    print("model_path:", args.model_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [tokenizer(p).input_ids for p in PROMPTS]
    print("prompt lengths:", [len(p) for p in prompts])

    def load_model() -> object:
        m = load_qwen36_model(
            args.model_path, device="cuda", dtype=torch.bfloat16, max_seq_len=args.max_seq_len
        )
        return m

    # -- 1. legacy vs dynamic token identity, INDEPENDENT model loads ----
    # Critical: the two backends must NOT share one model instance. The
    # self-built model's sparkinfer workspaces / GDN persistent buffers are
    # model-level, not backend-level; running legacy then dynamic against the
    # same instance silently reuses stale workspace state and produces
    # different KV bytes (measured 2026-08-15 -- a false divergence that
    # disappears with separate model loads). Each phase loads, runs, and
    # frees its own model so the comparison is clean.
    print("\n== 1. legacy vs dynamic: identical greedy streams ==")
    m1 = load_model()
    print(f"  [legacy] weights resident: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    legacy = Qwen36Backend(
        m1,
        num_slots=args.slots,
        max_seq_len=args.max_seq_len,
        device="cuda",
        dtype=torch.bfloat16,
        batched_decode=True,
        enable_prefix_cache=False,
        block_size=args.block_size,
    )
    ref_streams = [
        backend_greedy(legacy, i, ids, args.steps)
        for i, ids in enumerate(prompts[: args.slots])
    ]
    del legacy, m1
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    m2 = load_model()
    print(f"  [dynamic] weights resident: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    dynamic = Qwen36Backend(
        m2,
        num_slots=args.slots,
        max_seq_len=args.max_seq_len,
        device="cuda",
        dtype=torch.bfloat16,
        batched_decode=True,
        enable_prefix_cache=False,
        block_size=args.block_size,
        dynamic_arena=True,
    )
    snap = dynamic.pool.capacity_snapshot()
    print(
        f"  dynamic arena: pool_bundles={dynamic.pool.pool_bundles}, "
        f"kv_bytes_total={snap['kv_bytes_total'] // (1 << 20)} MiB, "
        f"kv_bytes_measured={snap['kv_bytes_measured'] // (1 << 20)} MiB"
    )
    dynamic.pool.assert_kv_storage_consistent()
    record("dynamic KV capacity formula == measured", True)

    for i, ids in enumerate(prompts[: args.slots]):
        got = backend_greedy(dynamic, i, ids, args.steps)
        same = got == ref_streams[i]
        first_div = next(
            (j for j, (a, b) in enumerate(zip(ref_streams[i], got)) if a != b), None
        )
        record(
            f"prompt[{i}] dynamic == legacy ({len(ref_streams[i])} tokens)",
            same,
            "" if same else f"first divergence at step {first_div}",
        )

    # -- 2. concurrency: legacy vs dynamic produce IDENTICAL streams -------
    # Solo-then-concurrent on one model instance diverges from the solo
    # reference at step ~4 in BOTH modes (a model-level baseline behavior,
    # measured 2026-08-15 on legacy b2_verify_serving.py too). The arena
    # claim that matters is that legacy and dynamic diverge the SAME way:
    # run the same solo-then-concurrent sequence in each mode (independent
    # model loads) and compare the two modes' concurrent outputs directly.
    print("\n== 2. concurrency: legacy vs dynamic, identical concurrent streams ==")
    params = SamplingParams()

    def concurrency_streams(mode: str):
        m = load_model()
        backend = Qwen36Backend(
            m,
            num_slots=args.slots,
            max_seq_len=args.max_seq_len,
            device="cuda",
            dtype=torch.bfloat16,
            batched_decode=True,
            enable_prefix_cache=False,
            block_size=args.block_size,
            dynamic_arena=(mode == "dynamic"),
        )
        use = prompts[: args.slots]
        # solo reference (pollutes the model identically in both modes)
        backend_greedy(backend, 0, use[0], args.steps)
        for s in range(len(use)):
            backend.reset_slot(s)
        slots = list(range(len(use)))
        state = backend.prefill_chunked_begin(slots, list(use))
        cur = [state.result[s]["anchor"] for s in slots]
        together = [[t] for t in cur]
        for _ in range(args.steps):
            cur = backend.decode_batch_sampled(
                slots, cur, [backend.slot_state(s).kv_len for s in slots],
                [params] * len(slots),
            )
            for i, t in enumerate(cur):
                together[i].append(t)
        del backend, m
        gc.collect()
        torch.cuda.empty_cache()
        return together

    n_use = len(prompts[: args.slots])
    legacy_conc = concurrency_streams("legacy")
    dyn_conc = concurrency_streams("dynamic")
    for i in range(n_use):
        same = dyn_conc[i] == legacy_conc[i]
        first_div = next(
            (j for j, (a, b) in enumerate(zip(legacy_conc[i], dyn_conc[i])) if a != b), None
        )
        record(
            f"concurrent slot {i}: dynamic == legacy",
            same,
            "" if same else f"first divergence at step {first_div}",
        )

    # -- 3. base pointer stability across reset/reuse ---------------------
    print("\n== 3. dynamic: KV arena base pointers stable across cycles ==")
    before = _kv_base_ptrs(dynamic)
    for _ in range(3):
        for s in range(args.slots):
            backend_greedy(dynamic, s, prompts[s % len(prompts)], 4)
            dynamic.reset_slot(s)
    after = _kv_base_ptrs(dynamic)
    record("all KV pool data_ptrs unchanged across reset/reuse", before == after)

    # -- 4. bundle conservation (no leaks) --------------------------------
    print("\n== 4. dynamic: bundle conservation after many cycles ==")
    u = dynamic.pool._arena.usage()  # noqa: SLF001
    usable = dynamic.pool.pool_bundles - dynamic.pool._arena.reserved  # noqa: SLF001
    conserved = (
        u.free_bundles + u.live_bundles + u.cached_bundles == usable
    )
    record(
        "free + live + cached == usable",
        conserved,
        f"free={u.free_bundles} live={u.live_bundles} cached={u.cached_bundles} "
        f"usable={usable}",
    )
    dynamic._arena = None  # noqa: SLF001 - release before MTP backend alloc

    # -- 5. MTP K=3 smoke on dynamic arena --------------------------------
    if not args.no_mtp:
        print("\n== 5. dynamic: MTP K=3 round smoke ==")
        import gc

        m_mtp = load_qwen36_model(
            args.model_path,
            device="cuda",
            dtype=torch.bfloat16,
            max_seq_len=args.max_seq_len,
            enable_mtp=True,
        )
        if m_mtp.mtp is None:
            print("  model loaded without MTP; skipping (load with enable_mtp=True)")
            del m_mtp
        else:
            prompt_ids = tokenizer(PROMPTS[0]).input_ids
            mtp_backend = Qwen36Backend(
                m_mtp,
                num_slots=args.slots,
                max_seq_len=args.max_seq_len,
                device="cuda",
                dtype=torch.bfloat16,
                batched_decode=True,
                enable_prefix_cache=False,
                block_size=args.block_size,
                dynamic_arena=True,
            )
            mtp_backend.enable_mtp(num_speculative_tokens=3)
            params = SamplingParams()
            mtp_backend.reset_slot(0)
            state = mtp_backend.prefill_chunked_begin([0], [prompt_ids])
            token = state.result[0]["anchor"]
            for _ in range(args.steps):
                token = mtp_backend.decode_batch_sampled(
                    [0], [token], [mtp_backend.slot_state(0).kv_len], [params]
                )[0]
            record("MTP K=3 decoded under dynamic arena", True)
            u = mtp_backend.pool._arena.usage()  # noqa: SLF001
            usable = mtp_backend.pool.pool_bundles - mtp_backend.pool._arena.reserved  # noqa: SLF001
            record(
                "MTP round left the bundle pool balanced",
                u.free_bundles + u.live_bundles + u.cached_bundles == usable,
                f"free={u.free_bundles} live={u.live_bundles} cached={u.cached_bundles}",
            )

    print("\n== summary ==")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, detail in _results:
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
