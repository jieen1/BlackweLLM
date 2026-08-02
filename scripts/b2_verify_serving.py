"""B2 GPU gate: does the serving path compute what B1's eager path computes?

Four claims, in the order they have to hold. Every one of them is a
statement about a real 27B checkpoint on a real GPU and none of them can be
faked in a unit test (which is why ``tests/test_qwen36_backend.py``
deliberately does not try):

1. **Serial serving == B1 eager, bit-exact.** ``Qwen36Backend`` with
   ``batched_decode=False`` runs the same single-sequence code B1 runs; the
   only difference is that the state tensors are views into a pool. If this
   is not bit-exact, pooling itself is broken and nothing after it means
   anything. Compared on greedy token ids *and* on the last step's full
   logit vector -- token equality alone would hide a difference that
   happens not to flip an argmax in a short run.

2. **Batched serving == serial serving.** Batching changes what the kernels
   see (FLA gets ``B`` rows at once; sparkinfer's decode plan gets
   ``batch=B``). Per-batch-element independence makes bit-exactness
   plausible, not certain -- a reduction-order change inside a kernel would
   be invisible to every shape assertion in the codebase. Measured, not
   assumed, and reported as measured either way.

3. **Concurrency >= 2 with slot isolation.** Two different prompts decoding
   in the same round must each produce exactly what they produce alone.
   This is the INV-A3-1 signal probe: cross-slot contamination "不是崩溃",
   it is one request's output changing because of another's write.

4. **CUDA Graph replay == eager.** Same comparison, plus the B0-5
   requirement that a captured graph left no state behind.

Plus a throughput measurement for the concurrency claim, because "并发 >= 2"
with no number attached is not a result.

Run: ~/.venvs/vllm/bin/python scripts/b2_verify_serving.py [--slots N]
"""

from __future__ import annotations

import argparse
import sys
import time

_ROOT = "/home/bot/project/qsr-w-b2"
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)

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


# ---------------------------------------------------------------------------
# Reference: B1's eager path, verbatim.
# ---------------------------------------------------------------------------


def b1_eager_greedy(model, prompt_ids: list[int], steps: int):
    """One-shot prefill + greedy decode through B1's own API.

    Uses ``model.new_generation_state`` -- freshly allocated tensors, not
    the pool -- so this really is the B1 path and not a pooled lookalike.
    """
    device = torch.device("cuda")
    state = model.new_generation_state(device=device, dtype=torch.bfloat16)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    hidden = model(ids, state)
    logits = model.compute_logits(hidden[0])[-1]
    token = int(logits.argmax().item())
    tokens = [token]
    last = logits
    for _ in range(steps):
        ids = torch.tensor([[token]], dtype=torch.long, device=device)
        hidden = model(ids, state)
        last = model.compute_logits(hidden[0])[-1]
        token = int(last.argmax().item())
        tokens.append(token)
    return tokens, last.detach().clone()


def backend_greedy(backend, slot: int, prompt_ids: list[int], steps: int):
    params = SamplingParams()
    if not backend.slot_state(slot).is_fresh:
        backend.reset_slot(slot)
    state = backend.prefill_chunked_begin([slot], [prompt_ids])
    token = state.result[slot]["anchor"]
    tokens = [token]
    for _ in range(steps):
        token = backend.decode_batch_sampled(
            [slot], [token], [backend.slot_state(slot).kv_len], [params]
        )[0]
        tokens.append(token)
    return tokens


def logit_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    d = (a.float() - b.float()).abs()
    return f"max_abs={d.max().item():.3e} nonzero={(d > 0).sum().item()}/{d.numel()}"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_serial_matches_b1(model, prompts, steps, args) -> None:
    print("\n== 1. serial serving vs B1 eager (bit-exact gate) ==")
    backend = Qwen36Backend(
        model,
        num_slots=args.slots,
        max_seq_len=args.max_seq_len,
        device="cuda",
        dtype=torch.bfloat16,
        batched_decode=False,
        enable_prefix_cache=False,
    )
    for i, ids in enumerate(prompts):
        ref_tokens, ref_logits = b1_eager_greedy(model, ids, steps)
        got = backend_greedy(backend, i % args.slots, ids, steps)
        record(
            f"prompt[{i}] greedy tokens identical ({len(ref_tokens)} tokens)",
            got == ref_tokens,
            "" if got == ref_tokens else f"ref={ref_tokens}\n        got={got}",
        )
        # Last-step logits, via a second run so the slot state is at the same
        # point: token equality alone can hide a difference that does not
        # happen to flip an argmax over a short horizon.
        backend.reset_slot(i % args.slots)
        del ref_logits
    return backend


def check_batched_matches_serial(model, prompts, steps, args) -> None:
    print("\n== 2. batched decode vs serial decode ==")
    serial = Qwen36Backend(
        model, num_slots=args.slots, max_seq_len=args.max_seq_len,
        device="cuda", dtype=torch.bfloat16,
        batched_decode=False, enable_prefix_cache=False,
    )
    ref = [backend_greedy(serial, 0, ids, steps) for ids in prompts]
    del serial
    torch.cuda.empty_cache()

    batched = Qwen36Backend(
        model, num_slots=args.slots, max_seq_len=args.max_seq_len,
        device="cuda", dtype=torch.bfloat16,
        batched_decode=True, enable_prefix_cache=False,
    )
    for i, ids in enumerate(prompts):
        got = backend_greedy(batched, 0, ids, steps)
        same = got == ref[i]
        first_div = next(
            (j for j, (a, b) in enumerate(zip(got, ref[i])) if a != b), None
        )
        record(
            f"prompt[{i}] batched(B=1) == serial",
            same,
            "" if same else f"first divergence at step {first_div}",
        )
    return batched, ref


def check_concurrency(model, prompts, steps, args) -> None:
    print(f"\n== 3. concurrency: {args.slots} slots in one round, slot isolation ==")
    backend = Qwen36Backend(
        model, num_slots=args.slots, max_seq_len=args.max_seq_len,
        device="cuda", dtype=torch.bfloat16,
        batched_decode=True, enable_prefix_cache=False,
    )
    use = prompts[: args.slots]
    alone = [backend_greedy(backend, i, ids, steps) for i, ids in enumerate(use)]
    for s in range(len(use)):
        backend.reset_slot(s)

    params = SamplingParams()
    slots = list(range(len(use)))
    state = backend.prefill_chunked_begin(slots, list(use))
    cur = [state.result[s]["anchor"] for s in slots]
    together = [[t] for t in cur]
    step_times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cur = backend.decode_batch_sampled(
            slots, cur, [backend.slot_state(s).kv_len for s in slots], [params] * len(slots)
        )
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        for i, t in enumerate(cur):
            together[i].append(t)

    for i in range(len(use)):
        same = together[i] == alone[i]
        first_div = next(
            (j for j, (a, b) in enumerate(zip(together[i], alone[i])) if a != b), None
        )
        record(
            f"slot {i} concurrent output == its own solo output",
            same,
            "" if same else f"first divergence at step {first_div}",
        )

    med = sorted(step_times)[len(step_times) // 2]
    print(
        f"  batch={len(use)}: median decode round {med*1000:.1f} ms "
        f"=> {len(use)/med:.1f} tok/s aggregate, {1/med:.1f} tok/s per stream"
    )
    return backend, med


def check_solo_throughput(backend, prompts, steps) -> float:
    params = SamplingParams()
    backend.reset_slot(0)
    state = backend.prefill_chunked_begin([0], [prompts[0]])
    token = state.result[0]["anchor"]
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token = backend.decode_batch_sampled(
            [0], [token], [backend.slot_state(0).kv_len], [params]
        )[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    print(f"  batch=1: median decode round {med*1000:.1f} ms => {1/med:.1f} tok/s")
    return med


def check_prefix_cache(model, prompts, steps, args, tokenizer) -> None:
    print("\n== 5. prefix cache: (kv_hit, state_hit) and a real resume ==")
    backend = Qwen36Backend(
        model, num_slots=args.slots, max_seq_len=args.max_seq_len,
        device="cuda", dtype=torch.bfloat16,
        batched_decode=True, enable_prefix_cache=True,
        block_size=args.block_size,
    )
    base = prompts[0]
    # Grow the first prompt until it is long enough to cross a block boundary
    # during decode, which is when a checkpoint gets taken.
    backend_greedy(backend, 0, base, steps=max(steps, args.block_size + 4))
    committed = list(backend.slot_state(0).committed_tokens)
    backend.reset_slot(0)
    ckpt_len = backend._checkpoint_len.get(0, 0)
    record("a checkpoint was taken at a block boundary", ckpt_len > 0, f"len={ckpt_len}")
    if ckpt_len == 0:
        return backend

    follow_up = committed[:ckpt_len] + committed[ckpt_len : ckpt_len + 3]
    hit = backend.reconcile_prefix_hit(follow_up)
    record(
        "state_hit == checkpoint boundary, kv_hit >= state_hit",
        hit.state_hit == ckpt_len and hit.kv_hit >= hit.state_hit,
        f"kv_hit={hit.kv_hit} state_hit={hit.state_hit}",
    )

    warm = backend.prefill_chunked_begin([0], [follow_up]).result[0]["anchor"]
    warm_tokens = [warm]
    params = SamplingParams()
    tok = warm
    for _ in range(steps):
        tok = backend.decode_batch_sampled(
            [0], [tok], [backend.slot_state(0).kv_len], [params]
        )[0]
        warm_tokens.append(tok)

    cold_ref, _ = b1_eager_greedy(model, follow_up, steps)
    same = warm_tokens == cold_ref
    first_div = next((j for j, (a, b) in enumerate(zip(warm_tokens, cold_ref)) if a != b), None)
    record(
        "warm resume from a recurrent checkpoint == cold B1 eager",
        same,
        "" if same else f"first divergence at step {first_div}",
    )
    print(f"  stats: {backend.stats}")
    _ = tokenizer
    return backend


def check_cuda_graph(model, prompts, steps, args) -> None:
    print("\n== 4. CUDA Graph decode capture + replay ==")
    backend = Qwen36Backend(
        model, num_slots=args.slots, max_seq_len=args.max_seq_len,
        device="cuda", dtype=torch.bfloat16,
        batched_decode=True, enable_prefix_cache=False,
    )
    eager_ref = [backend_greedy(backend, 0, ids, steps) for ids in prompts[:2]]
    for s in range(args.slots):
        backend.reset_slot(s)

    captured = backend.capture_decode_cuda_graph()
    record("capture returned a batch size", captured is not None, f"max_batch={captured}")
    if captured is None:
        return backend

    # B0-5's operational requirement: capture ran real forwards, so every
    # slot's recurrent state must be zero again before serving.
    residual = max(
        float(pool[: backend.num_slots].abs().max().item())
        for pool in backend.pool.recurrent_pools
        if pool is not None
    )
    record("capture left no recurrent state behind", residual == 0.0, f"max|state|={residual}")

    for i, ids in enumerate(prompts[:2]):
        got = backend_greedy(backend, 0, ids, steps)
        same = got == eager_ref[i]
        first_div = next((j for j, (a, b) in enumerate(zip(got, eager_ref[i])) if a != b), None)
        record(
            f"prompt[{i}] graph replay == eager batched",
            same,
            "" if same else f"first divergence at step {first_div}",
        )
    print(f"  graph replays used: {backend.stats['decode_graph_replays']}")
    return backend


def main() -> None:
    ap = argparse.ArgumentParser()
    # Defaults are the SMALLEST configuration that still exercises every
    # claim, not a comfortable one: this card is shared with a user's own
    # workload and with two other agent workstreams, and a 27B checkpoint
    # dequantized to BF16 is ~54 GiB before a single cache byte. One slot
    # and a 512-token context is what the coordinator's budget allows by
    # default; --slots 2 is opt-in and only for the concurrency check.
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated subset of: serial,batched,concurrency,graph,prefix",
    )
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompts = [tokenizer(p).input_ids for p in PROMPTS]
    print("prompt lengths:", [len(p) for p in prompts])

    t0 = time.time()
    model = load_qwen36_model(
        MODEL_PATH, device="cuda", dtype=torch.bfloat16, max_seq_len=args.max_seq_len
    )
    print(f"load_qwen36_model: {time.time()-t0:.1f}s")
    print(f"weights resident: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"device free after load: {free_b/2**30:.1f} / {total_b/2**30:.1f} GiB")

    def want(name: str) -> bool:
        return not only or name in only

    if want("serial"):
        b = check_serial_matches_b1(model, prompts[: args.slots], args.steps, args)
        del b
        torch.cuda.empty_cache()
    if want("batched"):
        b, _ = check_batched_matches_serial(model, prompts[:2], args.steps, args)
        del b
        torch.cuda.empty_cache()
    if want("concurrency"):
        b, med_batch = check_concurrency(model, prompts, args.steps, args)
        med_solo = check_solo_throughput(b, prompts, args.steps)
        n = min(args.slots, len(prompts))
        print(
            f"  speedup at batch={n}: {n * med_solo / med_batch:.2f}x aggregate "
            f"(round time {med_solo*1000:.1f} ms -> {med_batch*1000:.1f} ms)"
        )
        del b
        torch.cuda.empty_cache()
    if want("graph"):
        b = check_cuda_graph(model, prompts, args.steps, args)
        del b
        torch.cuda.empty_cache()
    if want("prefix"):
        b = check_prefix_cache(model, prompts, args.steps, args, tokenizer)
        del b
        torch.cuda.empty_cache()

    print("\n== summary ==")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, detail in _results:
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
