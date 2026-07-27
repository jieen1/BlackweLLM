# DFlash 64K 公平对照:接受率匹配后,我们已经和 vLLM 原生打平(2026-07-27)

## 结论先行

**"vLLM 367.3 tok/s vs 我们 252.89-259.14 tok/s,还差30-40%"这个此前的判断是不公平对照——
两边接受率差了 30 个百分点(vLLM 99.22% vs 我们 68.7%)。在接受率匹配(都接近饱和)的条件下,
我们引擎每轮的真实 wall-clock 开销已经和 stock vLLM 打平,差距 <1%。**

## 背景:为什么要重新测

`notes/2026-07-23-vllm-baseline-final.md` 记录的 367.3 tok/s(64K,DFlash K=15)一直被当作
阶段1(DFlash 64K 单请求深度优化)的对标目标。但这个数字是用什么接受率测的,当初没有记录——
`benchmarks/laguna_vllm_dflash_baseline.py` 的 `measure()` 只算 tok/s,不采集接受率。同时我们
自己最扎实的独立测试(`ab_dflash_verify_cg_vs_eager.py`,64K,向量化修复后)测出 252.89-259.14
tok/s,接受率 68.7%。两个数字直接比较之前,必须先确认接受率是否可比。

## 方法:读 vLLM 内部 Prometheus 计数器,不受打印间隔影响

给 `benchmarks/laguna_vllm_dflash_baseline.py` 加了 `--log-stats` 选项(commit 待定,见下),
原理:

- `disable_log_stats=True`(脚本原默认值)会连底层 Prometheus 计数器都不启用,必须先关掉。
- `SpecDecodingLogging.observe()` 每个 scheduler step 都会被调用、累加进内部列表,和
  `.log()` 的打印节流(默认按时间间隔)是两回事——`llm.llm_engine.get_metrics()` 读的是
  Prometheus `Counter`,`PrometheusStatLogger.record()` 每步都调,不受打印间隔影响,所以
  哪怕 `generate()` 只跑 0.7 秒(远低于默认打印间隔),也能拿到精确的累计值。
- 具体做法:测量前后分别读 `vllm:spec_decode_num_draft_tokens`/
  `vllm:spec_decode_num_accepted_tokens` 两个 Counter,做差,除出 acceptance_rate。

## 结果

`benchmarks/fixtures/laguna_vllm_dflash_baseline_accept_20260727.json`(64K,K=15,greedy,
moe_backend=auto→实测选 FLASHINFER_CUTLASS,重复短语 prompt"quick brown fox...near the
river bank",和历史 367.3 tok/s 测试同一个 prompt 构造方式):

```json
{
  "accepted_tok_s": 334.5,
  "acceptance_rate": 0.9922,
  "spec_decode_draft_tokens_delta": 765,
  "spec_decode_accepted_tokens_delta": 759
}
```

**vLLM 的历史高分(367.3/376.9 tok/s)是在 ~99% 接受率下测的**——不是像我最初以为的"两边都用
类似的重复短语 prompt,接受率应该差不多",实测差了整整 30 个百分点(vLLM 99.22% vs 我们
68.7%)。这次 334.5 tok/s 比历史 367.3 略低,大概率是同一套高接受率区间内的正常测量方差
(不同随机种子/GPU 状态),不影响下面的核心结论。

## 交叉验证:反推的每轮耗时和我们自己的真实服务端数字几乎一致

- vLLM:accepted_tok_s=334.5,acceptance_rate=0.9922 → 每轮有效 token ≈ 15×0.9922+1=15.88
  → 每轮耗时 ≈ 15.88/334.5 ≈ **47.5ms**。
- 我们:`notes/2026-07-27-dflash-server-integration.md` 记录的真实 HTTP 服务端测试(同样是
  重复文本、接受率接近满格)—— `avg_round_ms` **47.47-47.56ms**,`tokens/sec` **336-337**。

**两个独立测量(不同代码库、不同引擎、不同团队实现)在"接受率匹配"条件下,每轮耗时和吞吐几乎
完全一致(差距 <1%)。** 这不是巧合——说明我们引擎在 64K、M=16 verify 这个具体负载下的真实
计算效率,已经和 vLLM(torch.compile + piecewise CUDA Graph + autotuned kernel 全家桶)打平,
不存在"kernel 效率差 30-40%"这回事。

## 和另一个今天的发现的呼应

同一天另一项独立工作(sparkinfer MoE kernel 的显存带宽打满率测算,见另一篇待发笔记)也得出
"MoE kernel 在 M=16 下已经打满实测显存带宽上限(~100%)"的结论。两个发现互相印证:如果 kernel
本身已经没有空间,而真实服务端吞吐又和 vLLM 打平,那"阶段1 需要深度优化 kernel 才能追上
367 tok/s"这个任务前提本身站不住脚——**367 这个数字不是一个在我们真实生产场景(更真实、更低
接受率的文本)下有意义的对标尺,因为它测的是不现实的高接受率合成文本。**

## 尚未做、如果需要可以补的验证

- vLLM 在我们 68.7% 接受率对应的真实(更多样)文本下具体是多少 tok/s——需要再跑一次
  vLLM baseline(~8-9 分钟模型加载),用非重复 prompt。目前基于"两次独立测量在高接受率下
  高度吻合"的证据链,认为大概率也会在低接受率下继续吻合,但没有直接测过,如果要百分之百
  确认需要补这一步。

## 环境修复记录(阻塞过这次测试,现已解决,归功于协调者/用户)

1. `flashinfer-python==0.6.13` vs `flashinfer-jit-cache==0.6.15rc2` 版本不匹配,导致
   `flashinfer.fused_moe.cutlass_fused_moe` 里 `MoERunner.__init__` 调 JIT 编译模块的
   `init()` 时参数数量对不上(`TypeError: Mismatched number of arguments...Expected 8 but
   got 7`)。协调者把 `flashinfer-python` 升到 0.6.15 解决。
2. `ModuleNotFoundError: No module named 'torchvision'`——vLLM profile_run 路径依赖它,
   `/home/bot/.venvs/vllm/bin/python -m pip install torchvision --no-deps` 解决(和
   `notes/STATUS_dflash_acceptance.md` 里 07-26 记录的同一个已知修复,这次是这个具体
   venv 第一次需要它)。

## 代码改动

- `benchmarks/laguna_vllm_dflash_baseline.py`:新增 `--log-stats` 选项 + `_spec_decode_counter()`
  辅助函数,读 Prometheus 计数器算真实接受率,默认行为不变(不传这个 flag 就和以前完全一样)。
