# bfdiag 方案 1:常驻热引擎 daemon

负责范围:`bfdiag/daemon/` 全部文件(`protocol.py`/`provider.py`/`session.py`/
`canary.py`/`server.py`/`client.py`/`queue.py`/`cli.py`)、`bfdiag/__init__.py`、
三个测试文件、本笔记。全程在无 GPU 的 worktree 里开发,`LagunaEngineProvider`
的真实路径**只写代码,一次都没跑过**——这是本笔记最后一节要重点交代的风险。

## 1. 做了什么(文件清单)

```
bfdiag/__init__.py                       # 与其它 3 个 agent 逐字节一致的包 docstring
bfdiag/daemon/__init__.py                # 子包说明
bfdiag/daemon/protocol.py                # newline-delimited JSON 协议,纯函数可测
bfdiag/daemon/provider.py                # EngineProvider 协议 + FakeEngineProvider + LagunaEngineProvider
bfdiag/daemon/session.py                 # reset 清单(RESET_CHECKLIST)+ reset_laguna_engine()
bfdiag/daemon/canary.py                  # 金丝雀自检 + 基线持久化
bfdiag/daemon/server.py                  # Daemon:socket server、flock、FIFO worker、超时/TAINTED/重启
bfdiag/daemon/client.py                  # Client:Unix socket 客户端
bfdiag/daemon/queue.py                   # bf submit:FIFO + 笛卡尔积 env 扫描
bfdiag/daemon/cli.py                     # register(subparsers) 挂载 daemon/exec/repl/submit
tests/test_bfdiag_protocol.py            # 21 个用例,纯编解码
tests/test_bfdiag_daemon.py              # 17 个用例,daemon 全生命周期(FakeEngineProvider)
tests/test_bfdiag_canary.py              # 11 个用例,金丝雀 + 失败重启集成测试
notes/2026-07-27-bfdiag-warm-daemon.md   # 本文件
```

验证方式(均在本 worktree 根目录执行):

```bash
# lint(系统 python 没装 ruff,用仓库自带 venv 里的 ruff,纯静态分析、不执行代码)
/home/bot/project/qwen-sm120-runtime/.venv/bin/python -m ruff check .

# 单测(用系统 python——它连 torch 都没装,物理上不可能碰 GPU,是最安全的执行环境)
python -m pytest -q tests/test_bfdiag_protocol.py tests/test_bfdiag_daemon.py tests/test_bfdiag_canary.py
python -m pytest -q   # 全仓库回归,确认没有引入新的失败

# 端到端手动 smoke test(全部走 --provider fake,详见 §6)
QSR_BFDIAG_DIR=/tmp/x python -m bfdiag.daemon.cli daemon start --provider fake --wait-s 5
QSR_BFDIAG_DIR=/tmp/x python -m bfdiag.daemon.cli exec -c "result = 1 + 1"
QSR_BFDIAG_DIR=/tmp/x python -m bfdiag.daemon.cli daemon stop
```

结果:`ruff check .` 全绿;`pytest -q`(全仓库)275 passed, 49 skipped, 2 failed——
这 2 个失败是**改动前就存在**的基线失败(`test_regression_unit.py` 的一个用例、
`test_vllm_dependency_boundary.py` 的迁移台账断言),都在 `runtime/` 里、都不是我
改的文件,任务约束也明确不让碰 `runtime/`,所以原样记录、不处理。改动前的基线是
`226 passed, 49 skipped, 2 failed`(同样 2 个),`275-226=49` 正好等于我新增的用
例数,说明没有引入新的回归。

## 2. 协议设计

`protocol.py`:每行一个 UTF-8 JSON 对象,双向都是,不做长度前缀、不传二进制——
方便用 `socat`/`nc -U` 之类工具手工戳 socket 调试这套诊断平台本身。

请求 `Request`:`op`(`exec`/`ping`/`reset`/`status`/`shutdown`)+ `code` +
`args` + `timeout_s` + `run_id`。`op` 校验、`exec` 必须带非空 `code` 都在
`__post_init__` 里做,构造即校验,不合法的请求在**客户端**就直接抛
`ProtocolError`,不会发到 socket 上。

响应 `Response`:`ok` + `result` + `stdout`/`stderr` + `traceback` + `error`
+ `elapsed_s` + `state`(daemon 处理完这条请求后的状态)+ `run_id`。

框架层 `read_line`/`write_line` 只依赖 `.readline()`/`.write()`,不关心底层是
`socket.makefile()` 还是 `io.BytesIO`,所以协议本身可以完全脱离 socket 单测
(`tests/test_bfdiag_protocol.py`,21 个用例,全部纯函数)。

**踩过的一个坑**:`write_line` 一开始被我直接传了裸 `socket.socket` 对象(它
只有 `.send()`/`.sendall()`,没有 `.write()`),写完测试一跑就是
`AttributeError: 'socket' object has no attribute 'write'`——两处调用点
(`client.py::_request`、`server.py::_safe_write`)都需要先 `sock.makefile("wb")`
包一层再传给 `write_line`。这是测试真正跑起来才抓到的 bug,再次印证了"写完代码
不代表对",也是为什么 §7 的 GPU 待办清单里,每一条都要求"先在 FakeEngineProvider
上跑通同类路径,再上 GPU"。

## 3. daemon 的并发/超时/污染处理模型

`server.py::Daemon` 只用**一个 FIFO worker 线程**碰 engine:`ping`/`status`/
`shutdown` 由接收连接的线程直接应答(不用等排在前面的长任务跑完,`bf daemon
status` 永远秒回);`reset`/`exec` 一律扔进 `queue.Queue`,由唯一的 worker 线程
按提交顺序串行处理——这就是"同一时刻只有一个实验碰 GPU"这条硬约束在代码层面
的体现,不需要额外加锁,`queue.Queue` 本身的 FIFO 语义就够了。

超时处理:CPython 里**没有安全的办法强杀一个正在跑的线程**,尤其是卡在 C 扩展
里的调用——一个真正卡住的 CUDA kernel 更是只能杀掉整个 OS 进程才能收场,单进程
内任何"强制中断"手段都做不到。所以 `_run_with_timeout` 的做法是:worker 线程
把 `exec` 丢给一个短生命周期的子线程,`join(timeout_s)`;超时了就**放弃**这个
子线程(它可能还在后台跑,`daemon=True` 不会阻塞进程退出),立刻给客户端返回
超时错误,并把 daemon 标记 `TAINTED`。如果配置了 `restart_on_taint=True`(默认
开),就用 `provider_factory()` 现造一个全新 provider 实例并 `load()`,替换掉
`self._provider`——被放弃的那个子线程手里还攥着**旧的** provider 引用,永远碰
不到新实例,不会有"僵尸线程污染下一个实验"的问题。

但这只是**进程内的 Python 对象级"重启"**,不是真正杀掉 OS 进程:如果卡住的是
一次真实的 CUDA 调用,旧线程仍然会占着 GPU 显存/算力,新 provider 重新
`load()` 大概率会因为显存不够而失败,或者两边在 GPU 上打架。这条在 §7 里单独
列了待验证项。

普通异常(`exec` 的代码自己 `raise`)**不会**让 daemon 染上 `TAINTED`——只有
「金丝雀不匹配」和「超时放弃」这两种"我们已经不知道 engine 内部状态还对不对"
的情况才会 TAINTED。异常本身只是原样把 traceback 传回去,daemon 继续正常服务
下一个请求;如果这次异常真的弄脏了状态,靠的是**下一次 `exec` 之前的金丝雀检查**
来兜底。这也是为什么金丝雀是核心安全机制而不是可选项——普通的 try/except 完全
覆盖不了"代码没报错,但悄悄改坏了共享状态"这种最危险的情况。

单实例锁:`socket_path` 旁边有个 `<socket>.lock` 文件,`start()` 时
`fcntl.flock(fd, LOCK_EX | LOCK_NB)`,拿不到直接抛 `AlreadyRunningError`——
这台机器只有一块 GPU,`bf daemon start` 绝不能让两个 daemon 同时抢。持有锁之后
如果发现旧的 socket 文件还在,可以放心删掉重建(能拿到锁就说明自己是唯一实例,
残留 socket 文件一定是上次没能优雅退出留下的死文件)。

## 4. 金丝雀机制(canary.py)

固定 8 个 token 的假 prompt(`DEFAULT_CANARY_PROMPT_IDS`)、greedy、固定 8 步
(`DEFAULT_CANARY_STEPS`),每次 `exec` **之前**都跑一遍(`QSR_BFD_CANARY=0`
可关)。第一次跑记录基线到
`${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/canary_baseline.json`,基线带指纹
`f"{model_revision}:{git_sha}"`——指纹变了(比如真的换了模型或者改了代码)就
认为是预期之内的变化,直接重新记录基线,不算污染;指纹没变但 token 序列对不上,
就是真正的状态污染,拒绝执行这次 `exec`、标记 `TAINTED`、按配置重启。

`git_sha` 默认走 `git rev-parse --short HEAD`(2 秒超时,失败就退化成
`"unknown"`),可以用 `QSR_BFD_GIT_SHA` 环境变量或者构造函数参数覆盖,方便测试
里精确控制指纹、不依赖真实 git 状态。

`tests/test_bfdiag_canary.py::TestCanaryFailureRestartIntegration` 把整条链路
用 `FakeEngineProvider` 完整跑通了:第一次 `exec` 记基线 → `exec_code
("provider.pollute()")` 模拟一次"实验做完没清理干净" → 第二次 `exec` 的金丝雀
预检测到 token 不匹配 → 拒绝执行、状态变 `TAINTED`、自动重启(新实例
`dirty=0`)→ 第三次 `exec` 恢复正常。另有 `test_crash_on_reset_taints_and_
restarts` 覆盖"`reset()` 本身抛异常"这条崩溃恢复路径。

## 5. reset 必须清空的完整状态清单

这是本任务里我认为最容易"漏一个就是错误结论"的部分,所以专门去读了
`runtime/backends/laguna.py`、`laguna_dflash.py`、`laguna_cuda_graph.py`、
`laguna_dflash_cudagraph.py` 的源码(用 `codegraph_explore` 定位 + 关键行直接
`Read` 核对行号,没有大面积 grep),完整清单和代码位置写在
`bfdiag/daemon/session.py::RESET_CHECKLIST` 里(机器可读,和这里的表格逐条对
应,以后两边任一方改了都要同步)。

| 状态 | 适用 Laguna? | 代码位置 | 备注 |
|---|---|---|---|
| `slot_kv_len`/`slot_committed_tokens`/full+SWA KV cache 物理块 | 是 | `runtime/backends/laguna.py:1504 reset_slot` | 必须对 `range(backend.num_slots)` **每个** slot 都调,不能只调实验用过的 slot——原因见下一行 |
| CUDA Graph 捕获时的 warmup 残留 | 是 | `laguna_cuda_graph.py:294`(decode CG 用尾部 slot warmup)、`:702`(verify CG 用 **slot 0**,`warmup_kv=64`);`laguna_dflash_cudagraph.py:301`/`:544`(DFlash verify/draft CG 都用 **slot 0**) | **关键发现**:Laguna 的 `RESERVED_PHYSICAL_SLOTS = 0`(`laguna.py:40`),和有独立预留物理槽位的 DirectModelRunner/BlockPool 路径不一样——四个 CUDA Graph 捕获类的 warmup 都是直接把 dummy token(id=1 或 MASK_TOKEN_ID)写进**真实的逻辑 slot 0**(以及 M=1 decode CG 用到的尾部 slot)。也就是说 `DFlashEngine.__init__()`/`LagunaEngineProvider.load()` **天生不是 pristine 的**——`load()` 的最后一步必须无条件调用一次 `reset()`,否则冷启动后的第一个金丝雀/实验就会悄悄跑在一个还留着满长度 dummy KV 的 slot 0 上 |
| DFlash draft KV cache(6 层 SWA ring buffer) | 是 | `laguna_dflash.py:297 _alloc_draft_kv_cache`(`self._draft_kv_caches`) | 不属于 `LagunaBackend.reset_slot`——它在 `DFlashEngine` 这一层,现有诊断脚本(`benchmarks/diag_acceptance_v2.py:87-88`)已经手工 `kv_tensor.zero_()` 过,这正是本任务想消灭的样板代码。ring buffer 寻址理论上不会读到没写过的位置,所以省略这一步"目前没被证明会改变结果"——但这是未经测试的假设,正是金丝雀存在的意义:假设一旦错了,金丝雀能抓到 |
| Laguna 自己的轻量前缀复用(`find_prefix_match`) | 是 | `laguna_dflash.py:1043 generate_verify_only` | **本任务里最重要的一条发现**:`generate_verify_only` 默认 `enable_prefix_cache=True`,同一个 slot 连续调用两次 **不会**自动从头开始——它会用 `backend.find_prefix_match` 复用上一次留下的 KV/ring 状态。金丝雀(以及任何想"固定 prompt 纯函数式生成"的调用)必须显式传 `enable_prefix_cache=False`,并且前后都 `backend.reset_slot()`,否则金丝雀紧跟在一个共享前缀的实验后面跑,会误"命中"残留状态而不是真正冷跑模型,金丝雀就形同虚设 |
| GDN(Gated DeltaNet)递归状态 | **否** | `runtime/gdn_state.py`、`direct_model_runner.py:1593` | 规格里点名要看这个文件,但读代码发现 `LagunaBackend` 构造时传 `gdn_layer_names=[]`(`laguna.py:383`,注释写着"Laguna has no GDN/SSM recursive state")——这是 Qwen3.6/`DirectModelRunner` 那条路径的状态,`LagunaEngineProvider` 根本不加载它。规格与代码现实的分歧,按约定记录后继续 |
| 内容寻址持久前缀缓存(BlockPool 哈希索引) | **否** | `runtime/block_pool.py`、`runtime/prefix_cache.py` | 同上不适用:`LagunaBackend.reconcile_prefix_hit` 是显式 stub("E1: Laguna has no persistent content-addressed prefix cache yet ... every admission is a cold miss",`laguna.py:1520-1523`)。同样属于 DirectModelRunner 路径 |
| `LagunaBackend.generate()` 自己的 CUDA-Graph 贪心解码步数计数器 | **否(且是个 bug)** | `laguna.py:1643 self._decode_cg.reset()` | **顺手发现的运行时 bug**(只读代码没有修,`runtime/` 不在本任务范围内,记录给 `runtime/backends/laguna.py` 的 owner):`LagunaCudaGraphDecode`(`laguna_cuda_graph.py`)整个文件里 `grep -n reset` 零匹配,根本没有 `reset()` 方法,而 `LagunaBackend.generate()` 在温度=0 且走 CUDA Graph 分支时(`QSR_DECODE_CUDA_GRAPH` 默认就是开的,`laguna.py:342`)会调用这个不存在的方法,必然 `AttributeError`。因此 `LagunaEngineProvider` 完全不走这条 `backend.generate()` 路径,而是直接调 `DFlashEngine.generate_verify_only()`(生产用的真实入口,CUDA Graph 对象是它自己独立管理的一套,不会碰到这个坏方法) |

`bfdiag/daemon/session.py::reset_laguna_engine(engine)` 就是按前 3 条"适用"
清单写的真实重置逻辑:对每个 slot 调 `backend.reset_slot(slot)`,再把
`engine._draft_kv_caches` 全部 `zero_()`。`LagunaEngineProvider.load()` 末尾
无条件调一次,`generate()`(金丝雀用的那个)前后各调一次。**这段代码从未执行
过**,见 §7。

## 6. 端到端验证情况(全部 FakeEngineProvider / --provider fake)

- 单元测试:协议编解码(21)、daemon 生命周期/并发/超时/单实例锁/协议健壮性
  (17)、金丝雀 + 失败重启(11),共 49 个,全绿。
- 手动 smoke test(§1 的命令):`bf daemon start --provider fake` → `status`
  → `exec -c` → `exec <file>` → `submit --sweep` (2 变体,`.bfdiag/runs/
  <run_id>/queue_{request,response}.json` 落盘正确)→ 再次 `daemon start`
  确认复用而非报错(pid 不变)→ `daemon stop` → `status` 报"not running"。
  另外单独验证了 `bf repl`(管道喂 `result = 3 + 4` 一行,输出 `7`)。
- 一个测试(`TestCliSubprocessLifecycle`)是真·spawn 了一个
  `python -m bfdiag.daemon.server --provider fake` 子进程,走完整的
  start→ping→status→exec→shutdown,验证的是 `bf daemon start` 实际驱动的那条
  子进程生命周期路径——但全程 `--provider fake`,不会 import torch。

## 7. 需要 GPU 才能验证的待办清单(下一步串行安排的唯一依据)

`LagunaEngineProvider`、`session.reset_laguna_engine`、以及金丝雀在真实
Laguna+DFlash 引擎上的行为,**一次都没有运行过**——都是照着当前 `runtime/`
源码(直接 `Read`/`codegraph_explore` 核对过,不是拍脑袋写的)写出来的。上 GPU
之后必须按下面顺序逐条过一遍,漏一项就可能漏一个真实 bug:

1. **`LagunaEngineProvider.load()` 能不能跑通**:构造 `EngineArgs` → 加载
   `LagunaBackend` → 构造 `DFlashEngine`(权重加载 + draft 模型 + CUDA Graph
   捕获)→ `AutoTokenizer.from_pretrained` → `self.reset()`。这一串照抄自
   `benchmarks/diag_acceptance_v2.py::build_vllm_config`,但从没跑过,模型
   路径、`dtype`、`gpu_memory_utilization` 等默认值都需要在真机上确认仍然有效
   (`benchmarks/` 下的脚本这几周改动很快,详见 git log)。
2. **`load()` 末尾的 `self.reset()` 是否真的把 slot 0/尾部 slot 清干净了**:
   验证方法——`load()` 跑完之后,不做任何 reset,直接 dump 一次
   `backend.kv_caches[name][phys_slot_range]` 的统计量(比如非零元素数量),
   确认握手前确实有残留;再验证 `reset()` 之后变干净。这是 §5 表格第 2 行的
   直接验证。
3. **金丝雀在真实引擎上首次记录的基线是否稳定**:连续跑 3~5 次
   `LagunaEngineProvider.generate(DEFAULT_CANARY_PROMPT_IDS, DEFAULT_CANARY_
   STEPS)`(中间穿插 `reset()`),确认 token 序列逐位相同——如果模型本身在
   `enforce_eager=True`/CUDA Graph 混合模式下有任何非确定性(比如某个 kernel
   没有完全关掉 TF32/某个 reduce 顺序不固定),金丝雀本身就会误报,需要先确认
   "干净状态下重复调用是确定的"这个前提成立。
4. **`enable_prefix_cache=False` 是否真的完全绕开了 `find_prefix_match`**:
   验证同一个 slot 连续跑两个不同的 fixed prompt,确认第二次是完整重新
   prefill(而不是复用),可以加日志观察 `backend.find_prefix_match` 是否被
   跳过、或者用不同长度的 prompt 观察 prefill 耗时是否符合"从 0 开始"的预期。
5. **`generate_verify_only` 返回的 `tokens` 是否只包含新生成的部分**(不含
   prompt):`LagunaEngineProvider.generate()`/canary 都假设返回值就是"生成的
   延续部分",这个假设需要用一次已知输出的短 prompt 验证。
6. **draft KV cache 不清零是否真的不影响结果**——§5 表格第 3 行提到的假设。
   建议:先用现有 `benchmarks/diag_acceptance_v2.py` 的方法论(不同长度 prompt
   轮跑),对比"清零 draft KV"和"不清零"两种情况下 acceptance rate 是否一致;
   如果一致,`reset_laguna_engine` 里保留 zero_() 只是防御性开销;如果不一致,
   说明这一步是必要的、而且现有其它诊断脚本里可能还有没加这行的,需要回头排查。
7. **超时放弃 + 进程内重启在真实 CUDA 场景下的实际后果**(§3 提到的告警):
   人为制造一次"卡住"(比如给一个诊断脚本传一个会死循环/挂起的 kernel 调用),
   验证——(a) 客户端确实在 `timeout_s` 附近拿到超时响应而不是等到进程 hang;
   (b) `_maybe_restart` 现造的新 `LagunaEngineProvider` 在旧线程仍占着显存时
   `load()` 是报 OOM 还是别的错误;(c) 确认这种情况下**必须**升级为杀掉整个
   daemon 进程(而不是信任进程内重启),需要的话在 `server.py` 加一个"重启也
   失败就 `os._exit()`,让外层脚本/`systemd`/`bf daemon start` 的下一次调用
   重新拉起整个进程"的兜底路径——当前实现里没有这一层,是已知缺口。
8. **单实例 flock 在多 agent 排队用同一块 GPU 时的实际表现**:目前只在单进程
   内用 `AlreadyRunningError` 验证过语义,没有在"两个真实终端同时跑
   `bf daemon start --provider laguna`"的场景下人工验证过锁文件路径、权限、
   NFS/网络文件系统(如果 `.bfdiag/` 真的建在网络盘上,`flock` 的语义可能不
   可靠)等边界情况。
9. **`bf daemon start --provider laguna` 的 `--wait-s` 默认值(10s)对真实
   冷启动(权重加载 26.76s + draft 模型 3.97s + CUDA Graph 捕获,参考
   `our_server.log:39,49`)显然不够**——当前行为是超时后打印"still starting"
   并返回 0,不是 bug,但需要在真实环境里确认这个"打印提示、不阻塞"的行为
   符合预期,而不是应该把默认 `--wait-s` 调大或做成异步轮询。
10. **`describe()` 里的 `model_revision`(`_extract_revision`)对真实 HF
    缓存路径的解析是否正确**:只在字符串上测试过
    `.../snapshots/<hash>/` 模式,没有对着真实 `~/.cache/huggingface/hub/...`
    路径跑过 `os.path.expanduser` 之后的最终形态。

## 8. 已知限制 / 遗留问题

- **`bfdiag/cli.py` 的 dispatcher 约定是假设的**:另一个 agent 拥有
  `bfdiag/cli.py`,规格只给了 `register(subparsers) -> None` 这一个签名,没有
  说清楚解析完 `args` 之后怎么分发。我按最常见的 argparse 惯例实现——每个子
  parser `set_defaults(func=callable)`,`callable(args) -> int`。如果对方
  `cli.py` 用别的分发约定(比如按 `args.command` 手工 if/elif),需要合并时对
  一下接口,我自己的 `if __name__ == "__main__":` 里就是照这个假设写的,可以
  直接跑通作为参考实现。
- **超时放弃后的"进程内重启"不是真正的进程级重启**,§3/§7 第 7 条已经说得
  很清楚:真实 CUDA 卡死场景下这条恢复路径大概率不够,需要 GPU 验证后决定要
  不要加一层"重启也失败就 `os._exit()`"的兜底。
- **金丝雀目前只覆盖 DFlash 的 eager/verify-only 路径**,不覆盖
  `LagunaCudaGraphVerify`/`DFlashDraftCudaGraph` 的 **CUDA Graph replay**
  子路径本身的状态(比如 `_fill_buffers` 的实现细节,git log 里提到过的
  "Vectorize LagunaCudaGraphVerify._fill_buffers" 那次改动)——如果污染恰好
  发生在 replay 特有的 buffer 里而不是 KV cache/slot 计数器上,今天的金丝雀
  不一定能测出来。这是一个已知的覆盖面缺口,不是遗漏,只是本任务优先级
  (a)(b)(c) 排在前面,没有时间做到 (c) 之外的加固。
- **`.bfdiag/runs/<run_id>/` 下的文件命名**:`queue.py` 只写
  `queue_request.json`/`queue_response.json`,刻意加前缀避免和其它 3 个
  agent 的 run-record 模块打架;如果最终合并后发现命名冲突或者希望统一 schema,
  需要跨 agent 对一下,我这边没有权限创建共享的 `bfdiag/paths.py` 之类模块
  (不在文件清单里),所以 `bfdiag_dir()`/`default_socket_path()` 是在
  `server.py` 里独立实现的,其它 agent 大概率也会各自实现一份同名逻辑,这是
  4-agent 并行拆分下预期之内的重复,不是 bug。
- **`bf repl` 是最简实现**:空行提交整段缓冲代码,没有语法级"这一行需要续行"
  的智能判断,复杂的多行控制流(`if`/`for` 嵌套)需要用户自己拼好整段代码再
  空行提交,不如真正的 Python REPL 好用,但满足"交互式对已加载引擎发命令"的
  基本需求,优先级 (d) 里排在最后,没有进一步打磨。
