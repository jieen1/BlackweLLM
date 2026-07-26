# 历史 80.4 tok/s 复现验证（2026-07-27 00:12）

## 结论

**复现成功。** 独立 venv + 历史环境组合跑出 round1 **80.6 tok/s / 12.41ms/step**，
三轮均值 76.8 tok/s / 13.05ms/step，round2/3 的波动（72.6/77.1 tok/s）与此前
`FULL_STATUS_20260726.md` 记录的 GPU 时钟/功耗波动现象一致，不是新问题。
最优单轮数据已匹配/略超 2026-07-22 记录的 80.4 tok/s / 12.45ms/step。

## 精确环境（复现时刻的真实状态，非事后回忆）

| 组件 | 版本/commit | 备注 |
|------|-------------|------|
| 本项目代码 | commit `66d5913`（worktree `/tmp/qsr-66d`） | "Full benchmark: 64K/128K/200K all pass — 80.4/73.0/68.3 tok/s" |
| vLLM | `e12b91b032`（detached HEAD，`/home/bot/vllm`） | + 本地补丁（见下）+ 该 commit 上已重新编译好的 C++ 扩展（.so 时间戳 07-26 23:47-23:48） |
| vLLM 本地补丁 | `git stash@{0}` "local-patches-v0.25"，已 apply 到工作区 | 5 文件 +135/−52 行，diff 已存档于 `notes/2026-07-22-vllm-fork-diff.patch`（本次核对：内容逐字节一致，仅 git blob 哈希缩写长度不同） |
| vLLM 未跟踪自研文件 | `vllm/v1/attention/backends/sm120_gqa.py`（1115 行） | **本次新备份**：`notes/2026-07-22-vllm-sm120_gqa-backup.py`（md5 校验与源文件一致；`.bak` 后缀命中 `.gitignore`，改名为 `-backup.py` 才能入库）。此前只有文字描述（`2026-07-22-vllm-fork-archive.md`），没有完整内容备份——这是本次补上的关键缺口 |
| vLLM 未跟踪生成产物 | `csrc/moe/marlin_moe_wna16/`、`csrc/quantization/marlin/` | 确认是 `CMakeLists.txt` 里 `generate_kernels.py` 的标准构建期产物，不是手改补丁，**不需要备份**，重新 build 会自动生成 |
| vLLM 未跟踪第三方代码 | `vllm/third_party/tml_fa4/`（FlashAttention4 CuTeDSL，1.4MB） | 未在 `.gitignore`/`CMakeLists.txt` 显式引用，推测是 build 期下载的 vendor 代码而非我们手写，**未备份**——如果之后发现真的复现不了，这是第一个要复查的点 |
| sparkinfer | `0a7b143`（detached HEAD，`/home/bot/project/sparkinfer`） | "Fix decode graph capacity underestimation for windowed attention"；**不是**当前 `blackforge-main`（HEAD `3fa9b54`，多了 2 个正确性修复：`d2d8cb9` K/V 竞态、`989723d` MoE 确定性、`3fa9b54` bincount→scatter_add） |
| PyTorch | `2.13.0a0+gitcf30153`，editable `/home/bot/pytorch-build` @ `cf30153` | 未改动 |
| 模型 | `poolside/Laguna-S-2.1-NVFP4` snapshot `07614121b31898586430f189d27a25a0be310843` | |
| **Python venv** | **`/home/bot/.venvs/vllm-repro80`**（本次新建） | `cp -a /home/bot/.venvs/vllm /home/bot/.venvs/vllm-repro80`，**不改动主 venv `/home/bot/.venvs/vllm`**。editable install 仍指向上面几个共享源码目录，所以「独立」只隔离了 venv 本身的 site-packages/可执行文件，不隔离共享源码树的 git 状态 |

## 复现命令（可直接照抄）

```bash
# 1. 确认源码树处于上表状态（本次复现时已经是，如果被其他工作改动过，需要先切回）
cd /home/bot/vllm && git log -1 --oneline   # 应为 e12b91b032
cd /home/bot/project/sparkinfer && git log -1 --oneline   # 应为 0a7b143

# 2. 建独立 venv（如果 vllm-repro80 已存在就跳过）
cp -a /home/bot/.venvs/vllm /home/bot/.venvs/vllm-repro80

# 3. 跑 benchmark（脚本已存档：benchmarks/repro_80tok_m1_decode_cg.py，
#    等价于 /tmp/bench_m1_66d.py，指向 /tmp/qsr-66d 这个 66d5913 worktree）
cd /tmp/qsr-66d && \
  CUDA_VISIBLE_DEVICES=0 USE_LIBUV=0 HF_HUB_OFFLINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  /home/bot/.venvs/vllm-repro80/bin/python /path/to/benchmarks/repro_80tok_m1_decode_cg.py
```

若 `/tmp/qsr-66d` 不存在，用
`git worktree add /tmp/qsr-66d 66d5913`（在本仓库任意 checkout 下执行）重建。

## 结果

原始 JSON：`benchmarks/fixtures/repro_80tok_20260727_0009.json`

| Round | step_ms | tok/s | prefill_s | mem_GiB |
|-------|---------|-------|-----------|---------|
| 1 | 12.41 | **80.6** | 11.94 | 78.6 |
| 2 | 13.78 | 72.6 | 12.28 | 78.7 |
| 3 | 12.97 | 77.1 | 12.38 | 78.4 |
| **均值** | 13.05 | 76.8 | — | — |

对比历史记录（2026-07-22，同一套环境）：80.4 tok/s / 12.45ms/step。

## 与 FULL_STATUS_20260726.md 差距分解的关系

前一 session 已经用同一路数据拆出三个差距来源：

1. C++ 扩展版本差异（贡献 ~6 tok/s）
2. vLLM Python 代码差异 v0.25.0+patch vs v0.26.0（贡献 ~4 tok/s）
3. SM80 WMMA kernel 被 cuBLAS 选中在 SM120 上跑（`notes/STATUS_speed_optimization_0726.md`，潜在 ~2ms/step）

本次复现证实了差距来源 1+2 的基线端确实能打满 80.4——即历史数字真实、可信、可重复，不是测量误差或环境记错。接下来（任务 #5）要做的是反过来验证：**main 分支现在的代码**（`8e04775`，含 vLLM 0.26.0 适配）跑在**这套历史环境**（vLLM 0.25.0+patch）下会是多少——这能把"运行时代码演进"和"vLLM 版本"两个变量解耦，是目前唯一还没测过的组合。

## 遗留风险 / 待办

- 环境目前仍停留在历史态（vLLM `e12b91b032`+patch、sparkinfer `0a7b143`），**不是**主分支平时依赖的状态（vLLM 0.26.0、sparkinfer `blackforge-main`）。在切回之前，main 分支代码不应该被拿来跑真实 serving。
- `vllm/third_party/tml_fa4/` 的来源没有 100% 确认，如果以后复现失败，先查这个目录是不是被覆盖/丢失。
- 独立 venv 只隔离了 venv 本身，vLLM/sparkinfer/PyTorch 的 editable 源码树仍是共享的——如果要同时保留"历史可复现环境"和"随时可跑最新 main"两个互不干扰的完整环境，需要连源码树一起复制（工作量大，本次未做，先用"记录精确 commit + 按需切换"的方式代替）。
