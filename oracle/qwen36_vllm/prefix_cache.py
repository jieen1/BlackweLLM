"""B5: prefix cache operations extracted from DirectModelRunner.

Content-addressed prefix cache (P0-P3 three-tier) methods: block hash
computation, hit reconciliation, cold/warm prefill paths, and committed
block publication.

``self._r`` is the owning runner instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner

from runtime.block_pool import (
    BlockHash,
    hash_block_tokens,
)


class PrefixCacheOps:
    """Prefix cache operations.

    Mechanically extracted from DirectModelRunner.
    ``self._r`` is the owning runner instance.
    """

    def __init__(self, runner: DirectModelRunner) -> None:
        self._r = runner

    def _publish_committed_blocks(self, slot: int, token_ids: list[int], committed_len: int) -> int:
        # Populate-on-completion (attention half, P3.1 step 5/6): publish the
        # full committed blocks [slot_published_blocks[slot], committed_len //
        # block_size) to the content index, growing this slot's chained hash.
        # ONLY committed tokens are hashed/published -- the partial tail and any
        # draft/verify tokens beyond commit are never touched (INV4; mirrors
        # vLLM kv_cache_manager.py:456-465). Write-time dedup (step 6, sec 3.8):
        # if get_cached_block(h_i) hits an existing B', paranoid-verify
        # num_tokens (R7), then swap block_table[slot][i] -> B', touch([B']),
        # free([fresh]) (the recomputed duplicate's memory is reclaimed -- the
        # A>0,G=0 compute-miss reclamation). Else publish fresh. Returns the
        # deepest published boundary in tokens. The draft layer needs no
        # separate publish: it is the 17th attention-group member, so the same
        # block_table[slot] blocks hold its KV (sec 3.1).
        if not self._r.enable_persistent_prefix_cache:
            return self._r.slot_published_blocks[slot] * self._r.block_size
        # P3.2: keep the slot's full committed-token sequence available for
        # hashing decode-produced blocks (which may straddle the prompt tail +
        # decode head). At prefill this seeds it from the prompt
        # (token_ids[:committed_len]); during decode populate the caller has
        # already extended it to slot_kv_len, so this is a no-op there.
        if len(self._r.slot_committed_tokens[slot]) < committed_len:
            self._r.slot_committed_tokens[slot] = list(token_ids[:committed_len])
        block_size = self._r.block_size
        extra_keys = (self._r.kv_cache_dtype,)
        full_blocks = committed_len // block_size
        cursor = self._r.slot_published_blocks[slot]
        parent_hash = self._r.slot_block_hashes[slot][cursor - 1].value if cursor > 0 else None
        for i in range(cursor, full_blocks):
            block_tokens = token_ids[i * block_size : (i + 1) * block_size]
            h_i = hash_block_tokens(parent_hash, block_tokens, extra_keys)
            block_hash = BlockHash(h_i, (i + 1) * block_size)
            self._r.slot_block_hashes[slot].append(block_hash)
            fresh_block_id = self._r.block_table[slot][i]
            existing = self._r.block_pool.get_cached_block(h_i)
            if existing is not None and existing.block_id != fresh_block_id:
                if (
                    existing.block_hash is None
                    or existing.block_hash.num_tokens != (i + 1) * block_size
                ):
                    raise RuntimeError(
                        f"prefix-cache dedup collision: block {existing.block_id} "
                        f"num_tokens={getattr(existing.block_hash, 'num_tokens', None)} "
                        f"!= {(i + 1) * block_size} for hash {h_i} (R7)"
                    )
                self._r.block_table[slot][i] = existing.block_id
                self._r.block_pool.touch([existing.block_id])
                self._r.block_pool.free([fresh_block_id])
            else:
                self._r.block_pool.cache_block(fresh_block_id, block_hash)
            parent_hash = h_i
        self._r.slot_published_blocks[slot] = full_blocks
        return full_blocks * block_size

    def publish_committed_decode_blocks(self, slot: int, committed_token_ids: list[int]) -> None:
        """Decode-position populate (attention half, P3.2 step 4). Called by
        both verify-commit funnels AFTER ``slot_kv_len`` advances by the REAL
        committed length: append the newly-committed tokens to the slot's
        committed sequence, then publish any newly-FULL committed blocks
        ``[slot_published_blocks[slot], slot_kv_len[slot] // block_size)``,
        chaining each hash from the last published block (via the incremental
        ``_publish_committed_blocks``).

        ``committed_token_ids`` are the tokens newly written into KV this round
        (``[anchor] + committed[:-1]`` -- the recovery/bonus token is NOT yet
        written, so it is excluded). ONLY committed tokens ever reach here
        (INV4): rejected drafts never advance ``slot_kv_len``, so they are never
        hashed or published (mirrors vLLM ``kv_cache_manager.py:456-465``).
        No-op when the flag is off, and a no-op publish when no NEW full block
        exists yet (the cursor simply does not advance)."""
        if not self._r.enable_persistent_prefix_cache:
            return
        self._r.slot_committed_tokens[slot].extend(committed_token_ids)
        self._publish_committed_blocks(
            slot, self._r.slot_committed_tokens[slot], self._r.slot_kv_len[slot]
        )

    def _compute_prompt_block_hashes(
        self, token_ids: list[int], max_tokens: int
    ) -> list[BlockHash]:
        # Chained hashes of full blocks, capped at max_tokens (= len(T) - 1 on
        # lookup so the last token is always recomputed for logits; vLLM
        # kv_cache_manager.py:225-231). Pure CPU, O(blocks). Block i's hash
        # depends on all tokens 0..(i+1)*block_size via the chain.
        block_size = self._r.block_size
        extra_keys = (self._r.kv_cache_dtype,)
        num_blocks = max_tokens // block_size if max_tokens > 0 else 0
        hashes: list[BlockHash] = []
        parent_hash = None
        for i in range(num_blocks):
            block_tokens = token_ids[i * block_size : (i + 1) * block_size]
            h_i = hash_block_tokens(parent_hash, block_tokens, extra_keys)
            hashes.append(BlockHash(h_i, (i + 1) * block_size))
            parent_hash = h_i
        return hashes

    def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
        # Reconciliation (sec 3.4), specialized to two cache groups (no
        # iterative solver): L = G <= A.
        #   A = attention match -- walk hashes left-to-right, stop at first miss
        #       (the attention group is downward-closed: any prefix of a hit is
        #       a hit). A = matched_blocks * block_size.
        #   G = GDN boundary -- the largest checkpoint boundary Lc <= A with a
        #       GDN checkpoint under the SAME chained hash at Lc. In P3.1
        #       checkpoints exist only at completion boundaries, so G is that
        #       boundary or 0.
        #   L = G (always <= A, always block-aligned). A>0,G=0 => compute miss
        #       (L=0, prefill fresh -- vLLM v1's rule); write-time dedup still
        #       reclaims the recomputed attention blocks.
        if not self._r.enable_persistent_prefix_cache:
            return 0
        block_size = self._r.block_size
        hashes = self._compute_prompt_block_hashes(token_ids, len(token_ids) - 1)
        matched_blocks = 0
        for bh in hashes:
            if self._r.block_pool.get_cached_block(bh.value) is None:
                break
            matched_blocks += 1
        a = matched_blocks * block_size
        if a == 0:
            return 0
        g = 0
        for boundary_blocks in range(matched_blocks, 0, -1):
            hash_value = hashes[boundary_blocks - 1].value
            ckpt_key = self._r._gdn_ckpt_by_hash.get(hash_value)
            if ckpt_key is None:
                continue
            meta = self._r.gdn_ckpt_meta.get(ckpt_key)
            if meta is not None and meta["num_tokens"] == boundary_blocks * block_size:
                g = boundary_blocks * block_size
                break
        return g

    def restore_cached_prefix(self, slot: int, token_ids: list[int], L: int) -> None:
        # The sec 3.5 reuse steps 1-4 for a FRESH slot: reserve-and-touch the
        # [0, L) attention blocks BEFORE any forward (R4/INV9), restore the GDN
        # checkpoint at L (reusing the existing cross-slot restore -- P3 writes
        # no second restore), and set the bookkeeping to exactly what computing
        # [0, L) fresh would have produced. R1 addressing proof hook: the
        # checkpoint at L must be tagged with the SAME chained hash as this
        # prompt's boundary block at L -- a wrong-prefix checkpoint is REJECTED,
        # not used.
        block_size = self._r.block_size
        num_blocks = L // block_size
        if num_blocks <= 0:
            raise RuntimeError(f"restore_cached_prefix requires L >= block_size, got L={L}")
        if self._r.block_table[slot]:
            raise RuntimeError(f"restore_cached_prefix: slot {slot} is not fresh")
        hashes = self._compute_prompt_block_hashes(token_ids, len(token_ids) - 1)
        if len(hashes) < num_blocks:
            raise RuntimeError(
                f"restore_cached_prefix: prompt yields {len(hashes)} blocks < {num_blocks}"
            )
        matched_ids: list[int] = []
        for i in range(num_blocks):
            block = self._r.block_pool.get_cached_block(hashes[i].value)
            if block is None:
                raise RuntimeError(
                    f"prefix-cache hit lost block {i} (hash {hashes[i].value}) mid-restore"
                )
            matched_ids.append(block.block_id)
        boundary_hash = hashes[num_blocks - 1].value
        ckpt_key = self._r._gdn_ckpt_by_hash.get(boundary_hash)
        if ckpt_key is None:
            raise RuntimeError(
                f"prefix-cache hit at L={L} has no GDN checkpoint (hash {boundary_hash})"
            )
        meta = self._r.gdn_ckpt_meta[ckpt_key]
        if meta["hash_value"] != boundary_hash:
            raise RuntimeError(
                f"R1 reject: GDN checkpoint hash {meta['hash_value']} != prompt boundary "
                f"hash {boundary_hash} -- a wrong-prefix checkpoint is rejected, not used"
            )
        # Step 1: reference the [0, L) attention blocks (all 17 attention layers
        # share the one block-id namespace, sec 3.1). touch revives any block
        # parked at ref_cnt == 0 in the free queue (a freed-but-published block).
        self._r.block_table[slot] = list(matched_ids)
        self._r.block_pool.touch(matched_ids)
        # Step 2: restore the GDN checkpoint at L.
        view = self._r.checkpoint_view(ckpt_key)
        if view is None:
            raise RuntimeError(f"prefix-cache hit at L={L}: checkpoint view is None")
        self._r.restore_gdn_state(slot, view, allow_cross_slot=True)
        self._r.slot_gdn_initialized[slot] = True
        # Steps 3-4: bookkeeping reproduces computing [0, L) fresh.
        self._r.slot_draft_sync_len[slot] = L
        self._r.slot_kv_len[slot] = L
        self._r.slot_num_accepted_tokens[slot] = 1
        self._r.slot_block_hashes[slot] = list(hashes[:num_blocks])
        self._r.slot_published_blocks[slot] = num_blocks

    def _prefill_cold_with_populate(self, slot: int, prompt: list[int]) -> dict:
        # Two-phase cold prefill that materializes a CORRECT GDN completion
        # checkpoint at G = block_align_down(prompt_len - 1). A single-shot
        # prefill's live GDN state is at prompt_len, NOT at G -- so to capture
        # the state AT G, phase 1 prefills [0, G) (its GDN forward ENDS at G),
        # publishes [0, G//16) + materializes the checkpoint, then phase 2
        # continue-prefills [G, prompt_len). Token-identical to a single-shot
        # cold prefill (it IS chunked prefill with one boundary at G); mirrors
        # the proven P2 fan-out leader two-phase pattern. This is the dedicated
        # test's producing path (the only path that creates a correct completion
        # checkpoint in P3.1).
        if self._r.slot_kv_len[slot] != 0 or self._r.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        self._r.slot_num_accepted_tokens[slot] = 1
        prompt_len = len(prompt)
        k = self._r.num_speculative_tokens
        g = ((prompt_len - 1) // self._r.block_size) * self._r.block_size
        if g >= self._r.block_size:
            phase1_logits, phase1_hidden = self._r._forward_batch(
                [slot],
                [prompt[:g]],
                [0],
                qo_len=g,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )
            self._publish_committed_blocks(slot, prompt, g)
            num_g_blocks = g // self._r.block_size
            self._r.materialize_gdn_checkpoint(
                slot,
                key=self._r.block_table[slot][num_g_blocks - 1],
                hash_value=self._r.slot_block_hashes[slot][num_g_blocks - 1].value,
                num_tokens=g,
            )
            suffix_len = prompt_len - g
            suffix_tokens = prompt[g:]
            suffix_logits, suffix_hidden = self._r._forward_batch(
                [slot],
                [suffix_tokens] if suffix_len > 1 else [suffix_tokens[0]],
                [g],
                qo_len=suffix_len,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )
            anchor = int(suffix_logits[0].argmax(dim=-1).item())
            hidden = torch.cat([phase1_hidden, suffix_hidden], dim=0)
            draft_tokens_by_slot = self._r._mtp_sync_and_propose_batch(
                [slot],
                [prompt[1:] + [anchor]],
                hidden,
                [0],
                num_new_tokens=prompt_len,
                k=k,
                step0_logits_last_position_only=True,
            )
            self._publish_committed_blocks(slot, prompt, prompt_len)
            self._r.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
            return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}
        # Prompt too short for a full-block boundary < prompt_len
        # (prompt_len <= block_size): plain single-shot cold prefill; publish
        # whatever full blocks exist (the completion checkpoint needs a forward
        # ending at G >= block_size, impossible here).
        target_logits, target_hidden = self._r._forward_batch(
            [slot],
            [prompt] if prompt_len > 1 else [prompt[0]],
            [0],
            qo_len=prompt_len,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        anchor = int(target_logits[0].argmax(dim=-1).item())
        draft_tokens_by_slot = self._r._mtp_sync_and_propose_batch(
            [slot],
            [prompt[1:] + [anchor]],
            target_hidden,
            [0],
            num_new_tokens=prompt_len,
            k=k,
            step0_logits_last_position_only=True,
        )
        self._publish_committed_blocks(slot, prompt, prompt_len)
        self._r.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
        return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}

    def _prefill_hit_with_cache(self, slot: int, prompt: list[int], L: int) -> dict:
        # Restore-and-continue hit (sec 3.5): restore the [0, L) attention
        # blocks + GDN checkpoint at L (restore_cached_prefix), then continue-
        # prefill the suffix [L, prompt_len) via the EXACT validated continuation
        # the P2 fan-out sibling path uses (_forward_batch([s],[suffix],[L],
        # qo_len=suffix_len, commit, is_decode=False) + _mtp_sync_and_propose_
        # batch([s],[prompt[L+1:]+[anchor]], hidden,[L], num_new_tokens=
        # suffix_len, k=K)). L=0 never reaches here.
        if self._r.slot_kv_len[slot] != 0 or self._r.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        self.restore_cached_prefix(slot, prompt, L)
        prompt_len = len(prompt)
        suffix_len = prompt_len - L
        k = self._r.num_speculative_tokens
        suffix_tokens = prompt[L:]
        suffix_logits, suffix_hidden = self._r._forward_batch(
            [slot],
            [suffix_tokens] if suffix_len > 1 else [suffix_tokens[0]],
            [L],
            qo_len=suffix_len,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        anchor = int(suffix_logits[0].argmax(dim=-1).item())
        draft_tokens_by_slot = self._r._mtp_sync_and_propose_batch(
            [slot],
            [prompt[L + 1 :] + [anchor]],
            suffix_hidden,
            [L],
            num_new_tokens=suffix_len,
            k=k,
            step0_logits_last_position_only=True,
        )
        # Publish the suffix's full committed blocks (attention) so future
        # longer requests can hit deeper. The GDN checkpoint at the new
        # completion boundary is deferred (live GDN state is at prompt_len, not
        # a block boundary -- a correct one needs a forward ending there).
        self._publish_committed_blocks(slot, prompt, prompt_len)
        self._r.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
        return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}
