"""Temporary diagnostic: 2-slot simultaneous cold-full vs warm-hit comparison
(mirrors the proven P3.4 hook structure, avoiding single-slot sequential
state concerns), across several suffix lengths, to pinpoint whether the 10K
warm-suffix decode degeneration is a harness artifact or a real runtime
characteristic of the unchunked hit continue-prefill."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

def main():
    import torch
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa
    from benchmarks.workloads import D1_CTX64K_FIXTURE, load_prompt_token_ids
    from oracle.qwen36_vllm.direct_model_runner import (DirectModelRunner, build_vllm_config,
                                             _physical_slot)
    P = D1_CTX64K_FIXTURE.prompt_len
    prefix = load_prompt_token_ids(D1_CTX64K_FIXTURE)[0]
    max_tokens = 256
    suffix_lens = [64, 1024, 4096, 10240]
    max_suffix = max(suffix_lens)
    blocks_per_slot = -(-(P + max_suffix + max_tokens + 8) // 16) + 16
    cfg = build_vllm_config(model=MODEL, kv_cache_dtype="fp8_e4m3",
                            max_model_len=min(P + max_suffix + max_tokens + 2048, 262144),
                            gpu_memory_utilization=0.85, speculative_config=SPECULATIVE_CONFIG)
    runner = DirectModelRunner(cfg, num_slots=2, block_size=64,
                               blocks_per_slot=blocks_per_slot, enable_block_table=True,
                               enable_prefix_cache=True, enable_persistent_prefix_cache=True,
                               enable_cudagraph=False)
    block_size=64
    G = ((P - 1) // block_size) * block_size
    gdn0 = runner.gdn_layer_names[0]

    def clear():
        for s in range(runner.num_slots):
            if runner.slot_kv_len[s] != 0 or runner.block_table[s]:
                runner.reset_slot(s)
        runner.block_pool.hash_to_block.clear()
        for b in runner.block_pool.blocks:
            b.block_hash = None
        runner.gdn_ckpt_meta.clear()
        runner._gdn_ckpt_by_hash.clear()
        runner._gdn_ckpt_free = list(range(runner.gdn_ckpt_max_checkpoints))
        runner._gdn_ckpt_lru.clear()
        for s in range(runner.num_slots):
            runner.slot_block_hashes[s] = []
            runner.slot_published_blocks[s] = 0
            runner.slot_committed_tokens[s] = []

    def decode(slot, anchor, drafts, n=12):
        toks = []
        a, d = anchor, drafts
        for _ in range(n):
            dec = runner.mtp_verify_and_commit_batch([slot], {slot: a}, {slot: d})[slot]
            toks.extend(dec["committed"])
            a, d = dec["next_anchor"], dec["next_draft_tokens"]
        return toks

    print(f"P={P} G={G} blocks_per_slot={blocks_per_slot}")
    for slen in suffix_lens:
        suffix = [(prefix[-1] + 1 + i) % 151936 for i in range(slen)]
        full = prefix + suffix
        # slot 1 = warm hit; slot 0 = cold-full reference (simultaneous).
        clear()
        # cold populate P on slot 0 -> cache has P at G
        runner.mtp_prefill_with_cache([0], [prefix], 8192)
        # warm hit P+suffix on slot 1
        warm_pr = runner.mtp_prefill_with_cache([1], [full], 8192)
        warm_L = runner.reconcile_prefix_hit(full)
        warm_anchor = warm_pr[1]["anchor"]
        # cold-full reference on slot 0: reset slot 0, clear hash index only
        runner.reset_slot(0)
        runner.block_pool.hash_to_block.clear()
        for b in runner.block_pool.blocks:
            b.block_hash = None
        runner.gdn_ckpt_meta.clear()
        runner._gdn_ckpt_by_hash.clear()
        runner._gdn_ckpt_free = list(range(runner.gdn_ckpt_max_checkpoints))
        runner._gdn_ckpt_lru.clear()
        ref_pr = runner.mtp_prefill_with_cache([0], [full], 8192)
        ref_anchor = ref_pr[0]["anchor"]
        # GDN layer-0 compare (slot 0 ref vs slot 1 warm)
        conv_state, ssm_state = runner.kv_caches[gdn0]
        cp, wp = _physical_slot(0), _physical_slot(1)
        cr = conv_state[cp].shape[0] - K
        conv_exact = bool(torch.equal(conv_state[cp][:cr], conv_state[wp][:cr]))
        ssm_diff = (ssm_state[cp].float() - ssm_state[wp].float()).abs().max().item()
        # decode both
        ref_toks = decode(0, ref_anchor, ref_pr[0]["draft_tokens"])
        warm_toks = decode(1, warm_anchor, warm_pr[1]["draft_tokens"])
        exact = ref_toks == warm_toks
        first_mm = next((i for i,(a,b) in enumerate(zip(ref_toks,warm_toks)) if a!=b), None)
        print(f"\nsuffix={slen}: warm_L={warm_L} (G={G}) ref_anchor={ref_anchor} "
              f"warm_anchor={warm_anchor} anchor_match={ref_anchor==warm_anchor}")
        print(f"  conv_exact={conv_exact} ssm_max_diff={ssm_diff:.5f}")
        print(f"  decode exact={exact} first_mismatch_idx={first_mm}")
        print(f"  ref_toks[:12]={ref_toks[:12]}")
        print(f"  warm_toks[:12]={warm_toks[:12]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
