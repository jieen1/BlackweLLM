"""Profile one DFlash round: where does time go?"""
import torch, time

# Warm up with a short prompt
ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 2000, add_special_tokens=False)[:4096]
backend.reset_slot(0)
tokens, stats = engine.generate_verify_only(prompt_ids=ids, max_tokens=128, enable_prefix_cache=False, slot=0)
print(f"Warmup: accept={stats['acceptance_rate']*100:.1f}% tok/s={stats['tok_per_s']:.1f}")

# Now profile a 64K generation with detailed timing
ids64k = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 50000, add_special_tokens=False)[:65536]
backend.reset_slot(0)
torch.cuda.synchronize()

# Time prefill separately
t0 = time.perf_counter()
first_token, aux = backend.prefill_with_aux(0, ids64k)
torch.cuda.synchronize()
t_prefill = time.perf_counter() - t0

# Time bulk precompute
t1 = time.perf_counter()
if aux is not None:
    aux_len = aux[0].shape[0]
    aux_offset = len(ids64k) - aux_len
    engine._bulk_precompute_context_kv(0, aux, aux_len, aux_offset)
torch.cuda.synchronize()
t_precompute = time.perf_counter() - t1
del aux

# Time initial draft
kv_len = backend.slot_kv_len[0]
t2 = time.perf_counter()
if engine._draft_cg is not None:
    draft_tokens = engine._draft_cg.replay(0, first_token, kv_len)
else:
    draft_tokens = engine._draft_forward(0, first_token, kv_len)
torch.cuda.synchronize()
t_draft = time.perf_counter() - t2

# Time one verify round
verify_tokens = [first_token] + draft_tokens
t3 = time.perf_counter()
if engine._verify_cg is not None:
    verify_logits, verify_aux = engine._verify_cg.replay_with_aux(0, verify_tokens, kv_len)
else:
    verify_logits, verify_aux = engine._forward_verify_with_aux(0, verify_tokens, kv_len, len(verify_tokens))
torch.cuda.synchronize()
t_verify = time.perf_counter() - t3

# Time accept/reject
t4 = time.perf_counter()
all_argmax = verify_logits[:16].argmax(dim=-1).tolist()
from runtime.backends.laguna_dflash import _verify_only_accept_reject
decision = _verify_only_accept_reject(all_argmax, draft_tokens, first_token)
torch.cuda.synchronize()
t_accept = time.perf_counter() - t4

# Time context KV update (post-accept)
context_count = decision["context_count"]
t5 = time.perf_counter()
if verify_aux is not None:
    aux_slice = [a[:context_count] for a in verify_aux]
    combined_input = torch.cat(aux_slice, dim=-1)
    combined = engine.draft_model.combine_hidden_states(combined_input)
    bs = engine.block_size
    from runtime.backends.laguna import _physical_slot
    phys = _physical_slot(0)
    draft_base = phys * engine._draft_blocks_per_slot
    ring_slots = engine._draft_blocks_per_slot * bs
    context_positions = torch.arange(kv_len, kv_len + context_count, dtype=torch.long, device=engine.device)
    ring_blocks = (context_positions % ring_slots) // bs
    ring_offs = context_positions % bs
    slot_mappings = (draft_base + ring_blocks) * bs + ring_offs
    engine.draft_model.precompute_and_store_context_kv(combined, context_positions, slot_mappings)
torch.cuda.synchronize()
t_ctx_kv = time.perf_counter() - t5

# Time next draft
new_kv_len = kv_len + context_count
new_bonus = decision["next_anchor"]
t6 = time.perf_counter()
if engine._draft_cg is not None:
    next_draft = engine._draft_cg.replay(0, new_bonus, new_kv_len)
else:
    next_draft = engine._draft_forward(0, new_bonus, new_kv_len)
torch.cuda.synchronize()
t_next_draft = time.perf_counter() - t6

total_round = t_verify + t_accept + t_ctx_kv + t_next_draft
print(f"\n{'='*60}")
print(f"DFlash Round Profile (64K context)")
print(f"{'='*60}")
print(f"  Prefill (64K):       {t_prefill*1000:8.1f} ms")
print(f"  Draft KV precompute: {t_precompute*1000:8.1f} ms")
print(f"  Initial draft:       {t_draft*1000:8.1f} ms")
print(f"  ---")
print(f"  Verify (M=16):       {t_verify*1000:8.1f} ms  ({t_verify/total_round*100:.0f}%)")
print(f"  Accept/reject:       {t_accept*1000:8.1f} ms  ({t_accept/total_round*100:.0f}%)")
print(f"  Context KV update:   {t_ctx_kv*1000:8.1f} ms  ({t_ctx_kv/total_round*100:.0f}%)")
print(f"  Next draft:          {t_next_draft*1000:8.1f} ms  ({t_next_draft/total_round*100:.0f}%)")
print(f"  ---")
print(f"  ROUND TOTAL:         {total_round*1000:8.1f} ms")
print(f"  Accepted this round: {decision['num_accepted']}/15")
print(f"  Effective tok/s:     {(decision['num_accepted']+1)/total_round:.1f}")
print(f"  CG: draft={engine._draft_cg is not None} verify={engine._verify_cg is not None}")
