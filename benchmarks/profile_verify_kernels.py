"""Kernel-level profile of verify forward."""
import torch, time

# Setup
ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 50000, add_special_tokens=False)[:65536]
backend.reset_slot(0)
tokens, stats = engine.generate_verify_only(prompt_ids=ids, max_tokens=128, enable_prefix_cache=False, slot=0)

backend.reset_slot(0)
first_token, aux = backend.prefill_with_aux(0, ids)
if aux is not None:
    engine._bulk_precompute_context_kv(0, aux, aux[0].shape[0], len(ids) - aux[0].shape[0])
del aux
kv_len = backend.slot_kv_len[0]
draft_tokens = engine._draft_cg.replay(0, first_token, kv_len)
verify_tokens = [first_token] + draft_tokens

# Warm up
for _ in range(3):
    engine._verify_cg.replay_with_aux(0, verify_tokens, kv_len)
torch.cuda.synchronize()

# Profile with torch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(5):
        engine._verify_cg.replay_with_aux(0, verify_tokens, kv_len)
    torch.cuda.synchronize()

# Print top kernels
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
