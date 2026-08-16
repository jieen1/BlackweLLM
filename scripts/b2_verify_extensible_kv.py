"""Phase 5.5 GPU gate: extensible (VMM) physical KV vs dynamic arena A/B.

``notes/2026-08-16-vllm-extensible-kv-cache.md`` Phase B acceptance: the
extensible pool must be observationally identical to the dynamic arena's
full-commit pool on the serving path, while committing physical memory
lazily. Same-process, independent-model-loads A/B (same discipline as
``b2_verify_dynamic_arena_ab.py`` -- one shared model instance silently
reuses stale workspace state and produces false divergences).

Checks:
1. Extensible pool construction commits ~0 physical KV memory (vs the
   dynamic pool's full commit), while reserving the full VA capacity.
2. prefill + greedy: extensible == dynamic token streams, with the pool
   still mostly uncommitted (only the touched prefix is backed).
3. Base pointers stable across ensure -> greedy -> commit (graphs bound
   to the pre-commit addresses stay valid by construction).
4. After commit_kv_blocks(capacity): physical bytes == full pool, and
   greedy streams still == dynamic (post-commit correctness).
5. MTP K=3 round smoke under the extensible pool (pooled MTP caches
   joined the lockstep VMM pool).

Run: ~/.venvs/vllm/bin/python scripts/b2_verify_extensible_kv.py
     [--slots N] [--steps N] [--max-seq-len N] [--no-mtp]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} != {_ROOT}"
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


def backend_greedy(backend, slot: int, prompt_ids: list[int], steps: int) -> list[int]:
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
        for pool in list(backend.pool.k_pools) + list(backend.pool.v_pools)
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

    def load_model(**kw):
        return load_qwen36_model(
            args.model_path,
            device="cuda",
            dtype=torch.bfloat16,
            max_seq_len=args.max_seq_len,
            **kw,
        )

    # -- 1. dynamic reference streams (independent model load) -------------
    print("\n== 1. dynamic arena reference greedy streams ==")
    m1 = load_model()
    dynamic = Qwen36Backend(
        m1,
        num_slots=args.slots,
        max_seq_len=args.max_seq_len,
        device="cuda",
        dtype=torch.bfloat16,
        batched_decode=True,
        enable_prefix_cache=False,
        block_size=args.block_size,
        dynamic_arena=True,
    )
    dynamic.pool.assert_kv_storage_consistent()
    ref_streams = [
        backend_greedy(dynamic, i, ids, args.steps) for i, ids in enumerate(prompts[: args.slots])
    ]
    del dynamic, m1
    gc.collect()
    torch.cuda.empty_cache()

    # -- 2. extensible pool: construction commits ~0 physical KV ----------
    print("\n== 2. extensible pool: lazy physical commit ==")
    m2 = load_model()
    ext = Qwen36Backend(
        m2,
        num_slots=args.slots,
        max_seq_len=args.max_seq_len,
        device="cuda",
        dtype=torch.bfloat16,
        batched_decode=True,
        enable_prefix_cache=False,
        block_size=args.block_size,
        dynamic_arena=True,
        extensible_kv=True,
    )
    cap = ext.pool.pool_bundles
    ext.pool.assert_kv_storage_consistent()  # formula == view size (VA capacity)
    phys0 = ext.pool.physical_kv_bytes
    view_bytes = ext.pool.kv_storage_bytes()
    print(
        f"  extensible pool: capacity={cap} bundles, VA view={view_bytes // (1 << 20)} MiB, "
        f"physical committed={phys0 // (1 << 20)} MiB"
    )
    record(
        "extensible construction commits ~0 physical KV",
        phys0 <= 4 << 20,
        f"physical={phys0 // (1 << 20)} MiB of {view_bytes // (1 << 20)} MiB VA",
    )

    # -- 3. pre-commit greedy == dynamic, with pool mostly uncommitted -----
    print("\n== 3. extensible (pre-commit) == dynamic token streams ==")
    ext.ensure_kv_blocks(1 + args.slots * 4)  # warmup+capture-scale prefix
    streams = [
        backend_greedy(ext, i, ids, args.steps) for i, ids in enumerate(prompts[: args.slots])
    ]
    for i in range(args.slots):
        same = streams[i] == ref_streams[i]
        first_div = next(
            (j for j, (a, b) in enumerate(zip(ref_streams[i], streams[i])) if a != b), None
        )
        record(
            f"prompt[{i}] extensible == dynamic",
            same,
            "" if same else f"first divergence at step {first_div}",
        )
    phys_after_greedy = ext.pool.physical_kv_bytes
    committed_after = ext.pool.extensible_buffers.num_blocks_committed
    print(
        f"  after greedy: physical={phys_after_greedy // (1 << 20)} MiB "
        f"({committed_after} bundles committed of {cap})"
    )
    record(
        "greedy ran against a partially committed pool",
        committed_after < cap,
        f"{committed_after}/{cap} bundles committed",
    )

    # -- 4. base pointers stable; post-commit still == dynamic -------------
    print("\n== 4. base pointers + post-commit identity ==")
    before_ptrs = _kv_base_ptrs(ext)
    committed = ext.commit_kv_cache(cap)
    after_ptrs = _kv_base_ptrs(ext)
    record(
        "KV base pointers unchanged across commit",
        before_ptrs == after_ptrs,
        f"{len(before_ptrs)} pools",
    )
    record(
        "commit_kv_cache committed the full capacity",
        committed == cap,
        f"{committed}/{cap}",
    )
    phys_final = ext.pool.physical_kv_bytes
    print(
        f"  after commit: physical={phys_final // (1 << 20)} MiB "
        f"(view={view_bytes // (1 << 20)} MiB)"
    )
    record(
        "post-commit physical == full VA capacity",
        phys_final >= view_bytes - (2 << 20),
        f"physical={phys_final // (1 << 20)} MiB view={view_bytes // (1 << 20)} MiB",
    )
    for i in range(args.slots):
        ext.reset_slot(i)
        got = backend_greedy(ext, i, prompts[i], args.steps)
        record(
            f"prompt[{i}] post-commit extensible == dynamic",
            got == ref_streams[i],
            "" if got == ref_streams[i] else "divergence after commit",
        )

    # -- 5. MTP K=3 smoke under the extensible pool ------------------------
    if not args.no_mtp:
        print("\n== 5. extensible: MTP K=3 round smoke ==")
        m3 = load_model(enable_mtp=True)
        if m3.mtp is None:
            print("  model loaded without MTP; skipping")
            del m3
        else:
            prompt_ids = tokenizer(PROMPTS[0]).input_ids
            mtp = Qwen36Backend(
                m3,
                num_slots=args.slots,
                max_seq_len=args.max_seq_len,
                device="cuda",
                dtype=torch.bfloat16,
                batched_decode=True,
                enable_prefix_cache=False,
                block_size=args.block_size,
                dynamic_arena=True,
                extensible_kv=True,
            )
            mtp.ensure_kv_blocks(1 + args.slots * 5)  # K=3: verify K+1 pages/slot
            mtp.enable_mtp(num_speculative_tokens=3)
            mtp_backend_ok = True
            params = SamplingParams()
            mtp.reset_slot(0)
            state = mtp.prefill_chunked_begin([0], [prompt_ids])
            token = state.result[0]["anchor"]
            for _ in range(args.steps):
                token = mtp.decode_batch_sampled(
                    [0], [token], [mtp.slot_state(0).kv_len], [params]
                )[0]
            record("MTP K=3 decoded under extensible pool", mtp_backend_ok)
            mtp.commit_kv_cache(mtp.pool.pool_bundles)
            u = mtp.pool._arena.usage()  # noqa: SLF001
            usable = mtp.pool.pool_bundles - mtp.pool._arena.reserved  # noqa: SLF001
            record(
                "MTP rounds left the extensible pool balanced",
                u.free_bundles + u.live_bundles + u.cached_bundles == usable,
                f"free={u.free_bundles} live={u.live_bundles} cached={u.cached_bundles}",
            )
            del mtp, m3

    print("\n== summary ==")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, detail in _results:
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
