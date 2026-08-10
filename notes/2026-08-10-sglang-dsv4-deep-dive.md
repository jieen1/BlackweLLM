# SGLang DeepSeek V4 完整调研

> 来源：/home/bot/project/sglang（HEAD b296e1a503）深度调研。
> SGLang 无"魔法 M=1 kernel"；破墙靠 DSpark/MTP verify 批量（M=γ+1）+ FP8/NVFP4 减字节。

日期：2026-08-10

## 1. 模型结构（deepseek_v4.py，3121 行）

三层：`DeepseekV4ForCausalLM`(L2373) → `DeepseekV4Model`(L2036) → `DeepseekV4DecoderLayer`(L1288)。

**MLA**（MqaAttentionBase L373 / MQALayer L546）：
- `wqkv_a` 融合投影（L456-463，`SGLANG_OPT_FUSE_WQA_WKV` 默认开）。
- `fused_q_norm_rope`（L653-665）：norm+rope 单 kernel。
- KV：`set_swa_key_buffer_radix_fused_norm_rope`（L667-695）= norm+rope+FP8量化+写 cache 单 kernel，
  **无 bf16 KV 中间量**。
- 输出：wo_a（L490-499）+ wo_b（L509-518），wo_a 有 FP8 路径。

**MoE**：复用 DeepseekV2MoE（deepseek_v2.py L537），DSV4 特化：
- hash+topk 混合（前 num_hash_layers 用 HashTopK，其余 sqrtsoftplus topk）。
- shared expert 融进 MoE kernel（第 256 个 expert，L2814-2821）。

**HC**：hc_pre（L1380-1499）+ hc_post（L1501-1540）。TileLang `mhc_pre`（L1424-1444）
把 rmsnorm+hc mix 融合。跨层 MHC 融合 `mhc_fused_post_pre`（L1560-1580）把上层 post 与下层 pre 融合。

## 2. SM120 注意力 kernel 选择（deepseek_v4_backend.py L1582-1732）

- **默认 FlashInfer `sparse_mla_sm120_decode_dsv4`**（flash_mla_sm120.py L205-268）。
  页 256 → Triton 拆 64 子页（L330-388）。
- **Triton 备选**（flash_mla_sm120_triton.py L40-206）：tiled V2，grid=(B,H)，
  BLOCK_T∈{16,32} autotune，base-2 exp2 在线 softmax。**可抄模板**。
- KV 页：每 token 584B = 448 FP8 nope + 128 bf16 rope + 8 scale。SWA 128。

## 3. decode forward 关键优化

- `SGLANG_PREP_IN_CUDA_GRAPH=1`（默认开）：attn 元数据在 graph 内重建（backend L1081-1143）。
- **multi-stream 重叠**（`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` 默认开）：KV 投影/压缩器/indexer
  放 alt stream 与 Q 投影并行（L718-781）。M=1 也生效（_multi_stream_bs_limit=128）。
- **wo_a FP8**（L1232-1261）：o[T,G,D] 量化成 FP8 + group-major scale，
  `deep_gemm.fp8_einsum`，权重带宽减半。

## 4. MTP（deepseek_v4_nextn.py）
- e_proj(当前 embed) + h_proj(**上一时刻主模型 hidden**) → 单层 DecoderLayer → hc_head。
- 主模型 hidden 经 `forward_batch.spec_info.hidden_states` 切 [T*hc_mult,d] 进 draft（L148-152）。
- draft 的 γ token 批量穿主模型 1 层，verify 的 γ+1 token 穿全部层。

## 5. DSpark（deepseek_v4_dspark.py 892 行 + dspark_components/）
- draft：γ 个 DSparkV4Stage（γ=7），`project_target_hidden` 把主模型 hidden 投影进 draft KV。
- **verify 批量核心**（dspark_worker_v2.py L477-674）：propose → schedule_layout →
  verify 前向（bs×verify_num_draft_tokens 一次穿主模型全部层）→ accept_and_finalize →
  commit_hidden（把目标 hidden 写回 draft KV，自举闭环）。
- **折叠优化**：greedy 时 accept/finalize/KV-commit 全折进 verify CUDA graph（DsparkVerifyEpilogue）。

## 6. decode 调度（decode_cuda_graph_runner.py）
- 按 bs 分桶 capture_bs（L273-275），replay 时 raw_bs pad 到桶（L1106-1123）。
- DSpark verify 用 token-keyed ragged graph。

## 7. 对我们 runtime 的建议
1. **DSpark/MTP verify 批量**（最高优先）：M=1→M=γ+1，q8_0 19.4ms→~2.5ms(M=8)。
   verify 用 [bs,γ+1] 2D 窗口 + per-position causal。commit_hidden 是自举闭环关键。
2. **weight 降字节**：wo_a/wo_b 换 FP8（DSV4 checkpoint 自带）→ 带宽减半。
   q8_0 170 GB/s 若真（先查 kernel 是否 streaming 满带宽，对照 flash_mla_sm120_triton.py）。
3. **融合 kernel 减中间流量**：silu+mul+quant 单 kernel；norm+rope+quant+store 单 kernel。
4. **multi-stream 重叠**：隐藏 M=1 串行小 kernel 的 latency。
5. **SGLANG_PREP_IN_CUDA_GRAPH**：decode 元数据全进 graph。

## 8. 差异对照
| 维度 | SGLang | 我们 |
|---|---|---|
| M=1 | pad 进 bs 桶 + multi-stream；真正破墙靠 DSpark | 纯串行 M=1，无摊销 |
| 权重格式 | attention FP8、expert NVFP4、wo_a FP8 | q8_0(~8.25bit)、iq2xs(~3bit) |
| 注意力 | FlashInfer sparse_mla_sm120 或 Triton tiled | 自有 MLA（非瓶颈） |
| 融合度 | Q/KV/MoE 全融合 kernel | 需自查中间张量落全局次数 |
