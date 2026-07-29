"""Cold GPU smoke for the Laguna runtime with vLLM unavailable.

Run in a production environment where vLLM is not installed, or pass
``--block-vllm`` while developing in an oracle environment.  This validates
the real serving path: self-built model load, eager prefill/decode, decode
CUDA Graph capture, and DFlash verification.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import os
import sys


class _BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise ModuleNotFoundError(f"blocked production dependency: {fullname}")
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Laguna checkpoint directory")
    parser.add_argument(
        "--block-vllm",
        action="store_true",
        help="Block vLLM imports in this process",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--blocks-per-slot", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.block_vllm:
        sys.meta_path.insert(0, _BlockVllm())
    elif importlib.util.find_spec("vllm") is not None:
        raise RuntimeError(
            "vLLM is installed; rerun in a clean production environment or pass --block-vllm"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    from transformers import AutoTokenizer

    from runtime.backends.laguna import LagunaBackend
    from runtime.backends.laguna_dflash import DFlashEngine
    from runtime.laguna_config import build_laguna_config

    config = build_laguna_config(
        args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.80,
    )
    backend = LagunaBackend(
        config,
        num_slots=1,
        block_size=64,
        blocks_per_slot=args.blocks_per_slot,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 32,
        add_special_tokens=False,
    )[:128]

    try:
        first_token = backend.prefill(0, prompt_ids)
        next_token = backend.decode(0, first_token)
        captured_batch_size = backend.capture_decode_cuda_graph()
        backend.reset_slot(0)
        dflash = DFlashEngine(backend)
        _, dflash_stats = dflash.generate_verify_only(
            prompt_ids=prompt_ids,
            max_tokens=32,
            enable_prefix_cache=False,
            slot=0,
        )
        loaded_vllm = sorted(
            name for name in sys.modules if name == "vllm" or name.startswith("vllm.")
        )
        if loaded_vllm:
            raise RuntimeError(f"Laguna loaded vLLM modules: {loaded_vllm}")
        print(
            {
                "eager_tokens": [first_token, next_token],
                "captured_batch_size": captured_batch_size,
                "dflash_acceptance": dflash_stats["acceptance_rate"],
                "dflash_tok_per_s": dflash_stats["tok_per_s"],
                "vllm_modules": loaded_vllm,
            }
        )
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
