# 2026-08-05：persistent prefix 完整命中路径修复 + Codex CLI 接入

Status: **fixed and verified**（单元/集成测试全绿；服务端连续同 prompt 命中
复测通过；Codex CLI 端到端任务已完成）。

## 1. 背景

质量重跑（见 `notes/2026-08-05-qwen36-quality-rerun.md`）无回退，随后按
用户要求把服务切到“最佳配置”：MTP K=3、prefix cache、decode CUDA Graph +
MTP 自己的 anchor/draft/sync/verify CUDA Graph、3 × 256K 上下文
（`scripts/run_qwen36_quality.sh server start best`），并用 Codex CLI
（`model_providers.blackwellm`，Responses 协议）做端到端验证。

端到端验证暴露出两个 **persistent prefix（scratch 池）完整命中路径**的
runtime bug：第二次相同请求（usage.cached_tokens=16）输出变成
`'' + 10 个乱 token`，第三次请求又正常（“隔次必错”）。

## 2. 主 bug：完整命中把 live GDN 状态清零

现象（调试日志）：

```
run 1（冷）:  text='responses-ok'，155 tokens，第一个 verify 预测 579
run 2（完整命中）: text=''，10 个乱 token，第一个 verify 预测 846
run 3（又 miss）: 恢复正确
```

同一输入 `[8160,579,264,7047]`、相同 `past_len=16`，冷路径 row0 topk 是
`579`，命中路径是 `846` —— 说明命中路径的 target GDN 递推状态是空的。

根因：`runtime/backends/qwen36_mtp.py::Qwen36MTPEngine.restore_prefix_from_scratch`
在恢复 MTP KV 后调用 `self._spec_rows.reset_slot(target_slot)`。
`Qwen36MTPGDNRows.reset_slot` 会把 **包括 column 0（即 target 的 live GDN
状态）在内的所有 K+1 个 spec 行清零**。而调用顺序是：

```
pool.restore_recurrent_state(slot, entry.checkpoint)   # 先恢复 live GDN
pool.rewind_slot(...)
mtp.restore_prefix_from_scratch(...)                  # 再把 col 0 清零 ✗
```

随后第一个 verify 就从“零 GDN 状态”开始，logits 全错；之后每轮都像从
系统模板重新起步，最终第 4 轮提交 EOS → 10 个 token 后结束。

修复：scratch restore **只把 source 列指针钉回 column 0**
（`_spec_rows.activate(target_slot, 0)`），不再 `reset_slot`。候选列
（destination rows 0..K）每次 verify 都会被覆盖，陈旧字节无害。
同 slot 路径 `restore_prefix` 本来就不碰 spec 行，无需改动。

## 3. 次 bug：slot-local checkpoint 覆盖 persistent hash 索引

完整命中后 `_commit_prefill` 会再次触发 `_maybe_checkpoint(slot)`，在
同一个 block 边界（16）用**相同的 hash** 注册 `(slot, 16)`。
`RecurrentStatePool._by_hash` 是 one-to-one dict，新注册把
`("persistent", hash)` 覆盖成 `(slot, 16)`；于是
`_persistent_prefix_entry` 的
`get_by_hash(hash) != entry.checkpoint_key` 恒失败，persistent 条目
“永久 miss”，表现为第三次请求又回到冷计算。

修复：`_maybe_checkpoint` 注册前先查 `get_by_hash`；若同 hash 已被
persistent 条目占用，直接跳过重复注册（persistent 条目严格更完整：还带
MTP scratch 快照和 anchor hidden）。

## 4. 回归测试

- `tests/test_qwen36_backend.py::test_repeated_full_prompt_hits_stay_persistent_across_generations`
  —— 同一 prompt 连续两次完整命中，第三次仍命中（修复前第三次必然 miss）。
- `tests/test_qwen36_mtp_engine.py::test_restore_from_scratch_never_clears_the_live_gdn_column`
  —— scratch restore 不得调用 `reset_slot`，只能固定 source 列。

验证：`/home/bot/.venvs/vllm/bin/python -m pytest -q` → 1871 passed,
3 skipped；`/tmp/ci-sim/bin/python -m pytest -q` → 1150 passed,
192 skipped；`ruff check` 通过。

## 5. Codex CLI 接入

Codex CLI 0.146+ 已移除自定义 provider 的 `wire_api = "chat"`，只走
Responses 协议。因此新增：

- `server/formats/responses.py` + `server/app.py` 的 `/v1/responses`
  （含 SSE 流式）适配层，翻译成现有 chat 管线。
- `.codex/blackwellm.config.toml`：独立 profile，
  `model_provider = "blackwellm"`，`base_url = http://127.0.0.1:8300/v1`，
  `wire_api = "responses"`，256K 上下文窗口。

运行方式：

```bash
CODEX_HOME="$PWD/.codex" codex exec -p blackwellm \
  -C /home/bot/project/qwen-sm120-runtime "<任务>"
```

## 6. 复现/验证命令

```bash
bash scripts/run_qwen36_quality.sh server start best
# 同一 prompt 连续 curl 3 次，每次都应是 responses-ok
curl -sS --noproxy '*' http://127.0.0.1:8300/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","input":"Say exactly: responses-ok","max_output_tokens":32}'
```

服务端统计里 `prefix_persistent_restores` 应随每次完整命中递增，
`mtp_verify_graph_replays` 非零（证明 MTP CUDA Graph 在真实服务路径回放）。

## 7. 流式 SSE 生命周期事件缺 `type`（Codex CLI 反复重连）

首轮 Codex CLI 端到端虽然通过黑盒服务完成了两个文件，但 CLI 输出多次
`ERROR: stream disconnected before completion: stream closed before
response.completed`，最终非 0 退出。实测长流（19.7s / 172 事件）在
`response.completed` / `response.done` 之后断流：解析器每个事件的顶层
`type` 都是 None，客户端不认为已完成。

修复（`server/app.py`）：

- 四个生命周期事件（`response.created` / `in_progress` / `completed` /
  `done`）的 data 改为
  `{"type": ..., "response": snapshot}`（规范要求顶层 `type`），
  其余事件（`output_item.added`、`content_part.added`、`output_text.delta`
  等）本来就带 `type`，未动。
- 流内空闲 ≥15s 时发 SSE 注释 `: keepalive\n\n`，避免长思考空档被客户端
  空闲读超时掐断。

验证（2026-08-05 17:10+，服务以 best profile 重启后）：

```bash
curl -sS --noproxy '*' -N --max-time 240 http://127.0.0.1:8300/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","input":"请详细分析这个仓库的架构设计取舍，写出至少 300 字的中文分析。","stream":true,"max_output_tokens":1200}'
```

统计：29 个事件全部带顶层 `type`，事件序列完整（created → in_progress →
delta ×20 → done ×3 → completed → done），`max_gap=6.8s`，无缺 type、无断流。

Codex CLI 端到端重跑（同 §5 命令）：

- 退出码 **0**，无 `Reconnecting` / `stream disconnected` 报错；
- `/tmp/codex-e2e-hello.txt` = `runtime-e2e-ok`（14 字节，精确匹配）；
- `/tmp/codex-e2e-summary.md` = 28 行中文总结（任务要求 ≤30 行）；
- Codex 会话实际读完 README.md、docs/architecture.md、qwen36.py 入口与
  状态机后完成文件任务；会话用量 228,992 tokens，落在 256K 上下文内。

服务端日志全部 `POST /v1/responses 200 OK`。
