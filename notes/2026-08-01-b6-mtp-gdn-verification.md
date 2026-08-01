# B-6: Does Qwen3.6's in-checkpoint MTP layer carry a GDN layer?

> Investigation-queue reference: `docs/investigation-queue.md` B-6.
> Related prior art: `notes/2026-08-01-hybrid-cache-prior-art.md` §3, which raised
> this exact question and is the source of the vLLM quote being checked here.

## Verdict

**No.** All six locally cached Qwen3.6-27B checkpoint variants carry a single MTP
layer that is architecturally a plain full-attention transformer block — the same
tensor schema as the main model's `full_attention` layers. Zero GDN/linear-attention
tensors exist anywhere under the `mtp.*` namespace, in any variant, quantization
format, or provenance (official NVIDIA/Qwen FP8, official quantized AWQ-INT4, or
either community NVFP4 fork).

**But the premise this question was framed under is wrong**, and that matters more
than the yes/no: confirming "MTP has no GDN" does **not** delete Track B3's GDN
recursive-state rollback problem. That problem lives in the *main* model's 48 GDN
layers during the *verify* forward pass, not in the MTP head. See §3.

## 1. Evidence: `config.json`

`layer_types` (nested under `text_config`, since all six checkpoints are the
multimodal wrapper's config even for the "text-only" ones) is a flat 64-element list
covering the **main model's** layers only:

```
text_config.layer_types (len=64) unique={'full_attention', 'linear_attention'}
text_config.mtp_num_hidden_layers = 1
text_config.mtp_use_dedicated_embeddings = False
```

There is no separate `mtp_layer_types` field — the MTP layer's internal architecture
is not declared in `config.json` at all. `config.json` alone cannot answer this
question; it only tells you *how many* MTP layers exist (1), not what's inside one.
This is why the tensor-name evidence in §2 is the one that actually settles it.

Checked: `nvidia/Qwen3.6-27B-NVFP4`, `unsloth/Qwen3.6-27B-NVFP4`,
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, `morosystems/ThinkingCap-Qwen3.6-27B-NVFP4`,
`Qwen/Qwen3.6-27B-FP8`, `cyankiwi/Qwen3.6-27B-AWQ-INT4` — all six local HF cache
entries under `~/.cache/huggingface/hub/`. All six report `mtp_num_hidden_layers = 1`
and the same 64-element `layer_types` split (48 `linear_attention` + 16
`full_attention`).

## 2. Evidence: tensor names (the decisive check)

Comparing the main model's `full_attention` layer (index 3), `linear_attention` layer
(index 0), and the MTP layer, all from `nvidia/Qwen3.6-27B-NVFP4`'s
`model.safetensors.index.json`:

```
--- model.language_model.layers.3 (full_attention) ---
  input_layernorm.weight
  mlp.{down,gate,up}_proj.{weight,weight_scale,weight_scale_2,input_scale}
  post_attention_layernorm.weight
  self_attn.{q,k,v,o}_proj.{weight,weight_scale,input_scale (no o/k scale_2)}
  self_attn.{q,k}_norm.weight

--- model.language_model.layers.0 (linear_attention / GDN) ---
  input_layernorm.weight
  linear_attn.A_log
  linear_attn.conv1d.weight
  linear_attn.dt_bias
  linear_attn.in_proj_{a,b,qkv,z}.{weight[,weight_scale,input_scale]}
  linear_attn.norm.weight
  linear_attn.out_proj.{weight,weight_scale,input_scale}
  mlp.{down,gate,up}_proj...
  post_attention_layernorm.weight

--- mtp.layers.0 ---
  input_layernorm.weight
  mlp.{down,gate,up}_proj.weight
  post_attention_layernorm.weight
  self_attn.{q,k,v,o}_proj.weight
  self_attn.{q,k}_norm.weight
```

`mtp.layers.0.*` is exactly the `self_attn.*` + `mlp.*` schema of the main model's
`full_attention` layers. There is no `linear_attn.*` prefix, no `A_log`, no
`conv1d`, no `dt_bias`, no `in_proj_{a,b,qkv,z}` — the tensor signature that marks
every one of the main model's 48 GDN layers is absent from the MTP layer in every
single checkpoint checked.

Full `mtp.*` tensor inventory, by checkpoint (suffix pattern, layer index
normalized to `N`):

| Checkpoint | mtp.* tensor count | Pattern |
|---|---|---|
| `Qwen/Qwen3.6-27B-FP8` | 22 | `self_attn.*` (+ `weight_scale_inv` for FP8) + `mlp.*` + norms — **no `linear_attn.*`** |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | 36 | same, + AWQ `weight_packed/weight_zero_point/weight_shape` — **no `linear_attn.*`** |
| `morosystems/ThinkingCap-Qwen3.6-27B-NVFP4` | 15 | `self_attn.*` + `mlp.*` + norms — **no `linear_attn.*`** |
| `nvidia/Qwen3.6-27B-NVFP4` | 15 | identical to above |
| `unsloth/Qwen3.6-27B-NVFP4` | 15 | identical to above |
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | 15 | identical to above (read via safetensors header directly — single-file checkpoint, no index.json) |

The exact 15-tensor pattern (`mtp.fc.weight`, `mtp.norm.weight`,
`mtp.pre_fc_norm_{embedding,hidden}.weight`, and per-layer
`input_layernorm` / `post_attention_layernorm` / `mlp.{gate,up,down}_proj.weight` /
`self_attn.{q,k,v,o}_proj.weight` / `self_attn.{q,k}_norm.weight`) is unanimous
across all four unquantized-in-the-mtp-block checkpoints; the FP8 and AWQ variants
carry the same structure with extra per-tensor quantization scale/zero-point
tensors, never additional GDN-shaped ones.

## 3. Why this does *not* delete Track B3's rollback item — correcting the premise

The question was framed as: *"vLLM says draft models have no mamba layers, so no
eagle shift — if our MTP has no GDN, the GDN recursive-state rollback problem
doesn't exist."* The first half is now confirmed. The inference in the second half
does not follow, and independent evidence contradicts it:

- The vLLM comment being quoted (`reachable_block_mask`, see
  `notes/2026-08-01-hybrid-cache-prior-art.md` §3) is about the **draft model's own**
  recurrent state during the drafting step — this is the piece MTP having no GDN
  genuinely eliminates. There is no "eagle shift" to do because the MTP head has no
  recurrent state of its own to shift.
- But **verification** of the MTP's proposed tokens still runs those tokens through
  the **main model** — all 64 layers, including the 48 GDN layers — in one forward
  pass, exactly as it does for any other decode step. That pass updates the main
  model's GDN recurrent state as if every proposed token were real. If the verify
  step then rejects some suffix of them, the GDN state has already absorbed
  tokens that turned out not to happen, and — unlike a KV cache, which can just have
  the rejected blocks dropped — a gated-delta-rule recurrent state cannot be
  subtracted back to an earlier point; the update is not invertible.
- This is confirmed directly by vLLM's own upstream tracking, found via web search
  while grounding this conclusion: **vLLM issue #47572** ("[RFC]: ReplaySSM: cache
  SSM inputs instead of state for faster standard and speculative decode (Mamba2 +
  GDN)") states explicitly: *"Speculative decoding must roll back rejected draft
  tokens, but the SSM state update is irreversible... to support rollback, the
  current implementation keeps a separate recurrent state per draft token."* That
  RFC is about the **base/main model's** SSM layers under speculative decoding in
  general — it says nothing about the draft/MTP head's own architecture, because it
  doesn't need to: the cost it's fixing is incurred by the main model regardless of
  what the draft head looks like.
- This project's own investigation queue already independently flagged this exact
  RFC as **D-3** ("ReplaySSM Ring Spec-Verify: 投机 scratch 显存降 6.4×,
  11.5 GB → 1.8 GB") — that memory cost *is* the "keep N+1 copies of recurrent
  state so you can roll back to any of them" workaround this section is describing.
  D-3 and B3's GDN-rollback item are the same problem, seen from memory-footprint
  and correctness-mechanism angles respectively; they should be tracked together,
  not as one being deleted by the other.

**Net effect on Track B3 ("MTP draft / verify... 含GDN递归状态的推测回滚")**:

- Confirmed simplification: the MTP/draft side of speculative decoding needs no
  recurrent-state bookkeeping of its own (no separate conv/ssm state to allocate,
  checkpoint, or shift for the draft head) — this is a genuine, if narrower,
  win than the roadmap currently credits explicitly.
- Not eliminated: the main model's GDN state still needs a rollback story for
  rejected verify tokens. The item should stay in B3, reworded to make clear the
  rollback target is the main model's 48 GDN layers (not the MTP head), and
  explicitly cross-referenced to D-3 (ReplaySSM) as the concrete candidate
  mechanism — ReplaySSM turns the "irreversible update" problem into an O(1)
  ring-buffer pointer move by caching recent *inputs* rather than *state*, which
  is exactly the fix this rollback item needs regardless of whether it ships as a
  from-scratch implementation or a vendored port.

## 4. Roadmap impact

- `docs/roadmap.md` §2.3's differentiation matrix row ("投机解码 | ... | 不同机制，
  且 GDN 状态回滚是难点") is still accurate and should **not** be softened based on
  this finding — the GDN rollback difficulty is real, just not because of what the
  matrix row's adjacent MTP-mechanism description might suggest.
- Track B3's bullet ("MTP draft / verify（Qwen3.6 自带 1 层 MTP），含**GDN 递归状态的
  推测回滚**") should be kept, not deleted, but merged with D-3 in scheduling: they
  are the same engineering problem. (I did not edit `roadmap.md` or
  `implementation-plan.md` directly — out of scope for this pass — but whoever picks
  up B3/D-3 scheduling should read this note first.)
- One genuine, if smaller, win: B1/B2's scope is a little lighter than the
  differentiation matrix implies, since "MTP draft" work involves zero recurrent
  state design for the draft head itself — all of the GDN complexity Track B
  has to solve is main-model-side and already covered by A3's hybrid-cache work
  (D-1) plus this rollback item.
