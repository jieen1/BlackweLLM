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
tests/test_bfdiag_daemon.py              # 41 个用例,daemon 全生命周期 + idle-TTL + 热/冷边界(FakeEngineProvider)
tests/test_bfdiag_canary.py              # 11 个用例,金丝雀 + 失败重启集成测试
notes/2026-07-27-bfdiag-warm-daemon.md   # 本文件
```

**2026-07-27 补充**(协调者反馈,同一天):加了 idle-TTL 自动释放
(§7)、热/冷边界的产品化(§8,`requires_cold_restart` + `bf submit` 扫描
守卫 + `bf run --cold`)、显存快照可见性(§9),并把"热态 vs 冷启动零假设"
提到 GPU 待办清单(§10)第一位。三件事共享同一个动机:**这台机器只有一块
GPU,用户本人也要用它**——daemon 不能无限期占着卡,不能让扫描静默产生假
数据,不能让"热引擎跑久了显存布局变了"这件事对使用者不可见。

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

结果:`ruff check .` 全绿;`pytest -q`(全仓库)299 passed, 49 skipped, 2
failed——这 2 个失败是**改动前就存在**的基线失败(`test_regression_unit.py`
的一个用例、`test_vllm_dependency_boundary.py` 的迁移台账断言),都在
`runtime/` 里、都不是我改的文件,任务约束也明确不让碰 `runtime/`,所以原样
记录、不处理。改动前的基线是 `226 passed, 49 skipped, 2 failed`(同样 2
个),`299-226=73` 正好等于我新增的用例总数(首版 49 个 + 补充需求新增 24
个),说明没有引入新的回归。

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

- 单元测试:协议编解码(21)、daemon 生命周期/并发/超时/单实例锁/协议健壮性/
  idle-TTL/显存快照/热冷边界(41)、金丝雀 + 失败重启(11),共 73 个,全绿。
- 手动 smoke test(§1 的命令):`bf daemon start --provider fake` → `status`
  → `exec -c` → `exec <file>` → `submit --sweep` (2 变体,`.bfdiag/runs/
  <run_id>/queue_{request,response}.json` 落盘正确)→ 再次 `daemon start`
  确认复用而非报错(pid 不变)→ `daemon stop` → `status` 报"not running"。
  另外单独验证了 `bf repl`(管道喂 `result = 3 + 4` 一行,输出 `7`)。
- 一个测试(`TestCliSubprocessLifecycle`)是真·spawn 了一个
  `python -m bfdiag.daemon.server --provider fake` 子进程,走完整的
  start→ping→status→exec→shutdown,验证的是 `bf daemon start` 实际驱动的那条
  子进程生命周期路径——但全程 `--provider fake`,不会 import torch。
- 补充需求(idle-TTL/热冷边界/显存快照)额外手动验证过一遍真实子进程场景:
  `--idle-ttl-s 3` 的 daemon 在两条命令之间的间隙里**真的自己退出了**(日志
  `idle for 3.8s >= --idle-ttl-s=3.0s, auto-shutting down to release the
  GPU`),`bf submit --sweep 'QSR_DFLASH_CUDA_GRAPH=0,1'` 真的报错退出
  (exit=2)而不是静默跑,`bf run --cold --sweep 'MYVAR=x,y'` 真的起了两个
  独立进程、各自读到自己的环境变量值。

## 7. Idle-TTL 自动释放

`server.py::Daemon` 新增 `--idle-ttl-s`(默认 **900 秒**,`QSR_BFD_IDLE_TTL_S`
环境变量可覆盖,`0` 表示禁用)。语义:

- 计时基准是**最近一次跑完的 `exec`/`reset`**——不是"收到请求"的那一刻,是
  "处理完"的那一刻(`_worker_loop` 在 `job.done.set()` 之前调
  `_bump_activity()`)。这样一个跑得比 TTL 还久的实验,不会在它自己跑完的瞬间
  就被判定为"已经空闲了 TTL 那么久"。
- **`ping`/`status` 绝不重置计时器**——它们走 `_handle_immediate`,根本不经过
  `_submit_and_wait`/`_execute_job`,所以一个轮询脚本天然不可能让 daemon 永远
  活着。
- 空闲检测**在 `state == "BUSY"` 时永远跳过**(`_idle_watchdog_loop` 的第一个
  判断),这是防止"一个正常跑着的长实验被空闲看门狗当场拔线"的硬保证——不依赖
  时钟推算,而是直接看当前状态。
- 触发后:`provider.unload()`(新加进 `EngineProvider` 协议的方法,释放
  显存)→ 设置 `_shutdown_event` → 复用已有的 `shutdown` 清理路径(socket
  unlink、`flock` 释放、worker 线程退出)。`python -m bfdiag.daemon.server`
  跑在真实子进程里时,`serve_forever()` 返回后 `main()` 也跟着返回,进程自然
  退出——不需要额外的 `os._exit()`。

**可注入时钟**(`server.py::ManualClock`):`Daemon(..., clock=...)` 默认
`time.monotonic`,测试传一个 `ManualClock()` 进去,`clock.advance(seconds)`
瞬间"快进",不需要真的 `sleep` 900 秒。`tests/test_bfdiag_daemon.py::
TestIdleTTL` 七个用例覆盖:超时触发关闭、活动重置计时器、ping/status 不重置、
`--idle-ttl-s 0` 禁用、忙碌期间永不触发 + 完成后计时器归零、`status` 暴露
`idle_s`/`idle_ttl_s`、`exec_count` 只在真正执行时才加一。

## 8. 热/冷边界的产品化

用户问"性能测试能不能直接在热引擎里跑",答案分两类,这一版把边界做进了代码
而不是只写在文档里:

- **热引擎可靠**:稳态 decode 性能、接受率、精度实验、换 prompt/换 K/换
  运行时开关。
- **热引擎不可靠、必须冷启动**:冷启动 prefill 性能(见
  `notes/2026-07-20-cold-prefill-allocation-sensitivity-investigation.md`)、
  load-time 配置(`block_size`/容量/`gpu_memory_utilization`/
  `max_model_len`/量化后端)、显存压力/OOM 边界、首次 CUDA Graph 捕获耗时。

**代码里的边界**(`provider.py`):

- `LOAD_TIME_CONFIG_KEYS`:`LagunaEngineProvider` 构造函数参数里,构造完就
  锁死、只能靠重启才能改的那一组(`model_path`/`num_slots`/`blocks_per_slot`/
  `dtype`/`max_model_len`/`gpu_memory_utilization`/`dflash_model_path`)。
  `describe()["load_config"]` 把这组值暴露出来,`bf daemon status` 能直接
  看到当前 daemon 是用什么配置起的。
- `LOAD_TIME_ENV_VARS`:读代码(不是猜)确认的、`LagunaBackend`/`DFlashEngine`
  构造时**只读一次**的环境变量——`QSR_PREFILL_CHUNK`(`laguna.py:305`)、
  `QSR_DECODE_CUDA_GRAPH`(`laguna.py:342`)、`QSR_DFLASH_CUDA_GRAPH`
  (`laguna_dflash.py:168`)、`QSR_VERIFY_CUDA_GRAPH`(`laguna_dflash.py:384`)。
  这几个之所以重要,是因为原始的 `bf submit --sweep` 例子里就有一个
  (`QSR_DFLASH_CUDA_GRAPH=0,1`)——在热 daemon 里扫这个变量,构造时已经读过
  一次的布尔值根本不会因为之后改了 env var 而变化,扫出来的"4 个变体"里至少
  一半是同一个引擎配置跑出来的两份完全相同的假数据,而且从表面上完全看不
  出来。
- `requires_cold_restart(current_cfg, requested_cfg, locked_keys=None) ->
  list[str]`:纯函数,比较两个 config 字典在锁定 key 上的差异,返回不一致的
  key 列表(空列表 = 可以安全热复用)。两处用到:
  1. `bf daemon start` 检测到已有实例时,不再只看 `ping().ok`,还会拿当前
     CLI 请求的配置和正在跑的 daemon 的 `load_config` 比一遍——配置对不上就
     **拒绝复用**(打印哪些 key 不一致,提示先 `bf daemon stop`),而不是像
     以前那样默默复用一个配置早就不一样的旧 daemon。
  2. `queue.check_sweep_is_hot_safe(specs)`:`bf submit --sweep` 一旦扫到
     `LOAD_TIME_ENV_VARS` 里的变量名,`submit()` **在联系 daemon 之前**就
     直接 `raise ValueError`,清楚说明原因并指向 `bf run --cold`。这是硬性
     要求的落地:宁可报错,不能静默产生一份"看起来是 4 组不同配置、实际上
     只有 2 组真数据"的结果。
- `bf run --cold <script> [--sweep ...]`(`cli.py::_cmd_run`):不经过
  daemon,`--cold` 是**必须显式传的** argparse 必填 flag(测试专门验证过缺省
  会被 argparse 拒绝),每个 sweep 变体起一个独立的 `subprocess.run([sys.
  executable, script], env=...)` 真实进程,跑完退出——这才是 load-time 配置
  扫描的正路,替代了"热 daemon 里硬扫、数字看着像真的但其实没变"的陷阱。

## 9. 显存可见性

热引擎跑得越久,PyTorch caching allocator 越可能碎片化,第 50 个实验的显存
布局不等于刚启动那一刻——孤立的一个 tok/s 数字在热引擎里因此是不可比的,必须
连同当时的显存状态一起看。落地方式:

- `EngineProvider` 协议新增 `memory_snapshot() -> dict`。`server.py::_do_exec`
  在每次 `exec` **前后各调一次**(`_safe_memory_snapshot`,失败也不会弄坏
  exec 本身),写进 `Response.memory_before`/`memory_after`(`protocol.py`
  新增的两个字段)。
- `FakeEngineProvider.memory_snapshot()` 返回一个占位字典(`allocated_bytes`
  等全部是 `None`,只是为了让协议/daemon 路径端到端可测)。
  `LagunaEngineProvider.memory_snapshot()`——**代码照写,一次没跑过**——读
  `torch.cuda.memory_stats()`,取 `allocated_bytes.all.current`/
  `reserved_bytes.all.current`/`num_alloc_retries`,算一个简单的碎片率
  `(reserved - allocated) / reserved`。
- `status()` 新增 `exec_count`(真正执行过的实验数,`ping`/`status`/`reset`
  都不算)和 `since_cold_start_s`(距**进程**上次真正冷启动过去多久,不随
  进程内的 TAINTED 重启而归零——那是 provider 对象级别的重建,不是这里想追踪
  的"进程/CUDA context 活了多久")。
- **给使用指南的直接结论**:任何从这个 daemon 里拿到的 tok/s、ITL 数字,报告
  的时候都应该带上当时的 `memory_before`/`memory_after`(或者至少
  `exec_count`/`since_cold_start_s`)——两次测量如果 `fragmentation_ratio`
  差异很大,那两次数字本身就不该被当成同一个基准去比较,这不是噪声,是
  热引擎的真实副作用。

## 10. 需要 GPU 才能验证的待办清单(下一步串行安排的唯一依据)

`LagunaEngineProvider`、`session.reset_laguna_engine`、以及金丝雀在真实
Laguna+DFlash 引擎上的行为,**一次都没有运行过**——都是照着当前 `runtime/`
源码(直接 `Read`/`codegraph_explore` 核对过,不是拍脑袋写的)写出来的。上 GPU
之后必须按下面顺序逐条过一遍,漏一项就可能漏一个真实 bug。**第 1 条是这份
清单里最重要的一条,必须排最前面**:

1. **零假设:同一个 benchmark,daemon 热态跑 vs 冷启动跑,差异必须在噪声内
   (建议 p50 相对偏差 < 1%)**。具体做法:选一个现有的、有基线的 benchmark
   (比如 `benchmarks/` 下测 tok/s 或 ITL 的某个脚本),(a) 冷启动跑一次拿到
   基线,(b) 在热 daemon 里、已经跑过若干个"无关"实验之后(模拟真实使用场景,
   不是刚 `load()` 完就测),`bf exec` 同一段测量代码,(c) 对比两次的 p50(以
   及 §9 提到的显存快照)。**如果差异显著,那就是"热引擎不适合测这类性能"的
   实证,不是要去调参数消除的噪声**——必须把这个结论写进使用指南(哪些指标
   只能在热引擎测、哪些必须冷启动测,§8 目前只是"设计上认为"的分类,这一条
   验证的是"设计上认为"是否等于"实测如此")。不能假设它没问题就跳过这一条。
2. **`LagunaEngineProvider.load()` 能不能跑通**:构造 `EngineArgs` → 加载
   `LagunaBackend` → 构造 `DFlashEngine`(权重加载 + draft 模型 + CUDA Graph
   捕获)→ `AutoTokenizer.from_pretrained` → `self.reset()`。这一串照抄自
   `benchmarks/diag_acceptance_v2.py::build_vllm_config`,但从没跑过,模型
   路径、`dtype`、`gpu_memory_utilization` 等默认值都需要在真机上确认仍然有效
   (`benchmarks/` 下的脚本这几周改动很快,详见 git log)。
3. **`load()` 末尾的 `self.reset()` 是否真的把 slot 0/尾部 slot 清干净了**:
   验证方法——`load()` 跑完之后,不做任何 reset,直接 dump 一次
   `backend.kv_caches[name][phys_slot_range]` 的统计量(比如非零元素数量),
   确认握手前确实有残留;再验证 `reset()` 之后变干净。这是 §5 表格第 2 行的
   直接验证。
4. **金丝雀在真实引擎上首次记录的基线是否稳定**:连续跑 3~5 次
   `LagunaEngineProvider.generate(DEFAULT_CANARY_PROMPT_IDS, DEFAULT_CANARY_
   STEPS)`(中间穿插 `reset()`),确认 token 序列逐位相同——如果模型本身在
   `enforce_eager=True`/CUDA Graph 混合模式下有任何非确定性(比如某个 kernel
   没有完全关掉 TF32/某个 reduce 顺序不固定),金丝雀本身就会误报,需要先确认
   "干净状态下重复调用是确定的"这个前提成立。
5. **`enable_prefix_cache=False` 是否真的完全绕开了 `find_prefix_match`**:
   验证同一个 slot 连续跑两个不同的 fixed prompt,确认第二次是完整重新
   prefill(而不是复用),可以加日志观察 `backend.find_prefix_match` 是否被
   跳过、或者用不同长度的 prompt 观察 prefill 耗时是否符合"从 0 开始"的预期。
6. **`generate_verify_only` 返回的 `tokens` 是否只包含新生成的部分**(不含
   prompt):`LagunaEngineProvider.generate()`/canary 都假设返回值就是"生成的
   延续部分",这个假设需要用一次已知输出的短 prompt 验证。
7. **draft KV cache 不清零是否真的不影响结果**——§5 表格第 3 行提到的假设。
   建议:先用现有 `benchmarks/diag_acceptance_v2.py` 的方法论(不同长度 prompt
   轮跑),对比"清零 draft KV"和"不清零"两种情况下 acceptance rate 是否一致;
   如果一致,`reset_laguna_engine` 里保留 zero_() 只是防御性开销;如果不一致,
   说明这一步是必要的、而且现有其它诊断脚本里可能还有没加这行的,需要回头排查。
8. **超时放弃 + 进程内重启在真实 CUDA 场景下的实际后果**(§3 提到的告警):
   人为制造一次"卡住"(比如给一个诊断脚本传一个会死循环/挂起的 kernel 调用),
   验证——(a) 客户端确实在 `timeout_s` 附近拿到超时响应而不是等到进程 hang;
   (b) `_maybe_restart` 现造的新 `LagunaEngineProvider` 在旧线程仍占着显存时
   `load()` 是报 OOM 还是别的错误;(c) 确认这种情况下**必须**升级为杀掉整个
   daemon 进程(而不是信任进程内重启),需要的话在 `server.py` 加一个"重启也
   失败就 `os._exit()`,让外层脚本/`systemd`/`bf daemon start` 的下一次调用
   重新拉起整个进程"的兜底路径——当前实现里没有这一层,是已知缺口。
9. **单实例 flock 在多 agent 排队用同一块 GPU 时的实际表现**:目前只在单进程
   内用 `AlreadyRunningError` 验证过语义,没有在"两个真实终端同时跑
   `bf daemon start --provider laguna`"的场景下人工验证过锁文件路径、权限、
   NFS/网络文件系统(如果 `.bfdiag/` 真的建在网络盘上,`flock` 的语义可能不
   可靠)等边界情况。
10. **`bf daemon start --provider laguna` 的 `--wait-s` 默认值(10s)对真实
   冷启动(权重加载 26.76s + draft 模型 3.97s + CUDA Graph 捕获,参考
   `our_server.log:39,49`)显然不够**——当前行为是超时后打印"still starting"
   并返回 0,不是 bug,但需要在真实环境里确认这个"打印提示、不阻塞"的行为
   符合预期,而不是应该把默认 `--wait-s` 调大或做成异步轮询。
11. **`describe()` 里的 `model_revision`(`_extract_revision`)对真实 HF
    缓存路径的解析是否正确**:只在字符串上测试过
    `.../snapshots/<hash>/` 模式,没有对着真实 `~/.cache/huggingface/hub/...`
    路径跑过 `os.path.expanduser` 之后的最终形态。
12. **`LagunaEngineProvider.unload()`/`memory_snapshot()` 从未跑过**:
    `unload()` 里 `del` 引用 + `gc.collect()` + `torch.cuda.empty_cache()`
    这套组合拳是否真的把显存还给了系统(而不是被某个我没想到的地方持有的
    引用挡住),以及 idle-TTL 触发 `unload()` 之后进程退出前这段时间窗口
    (`_stop()` 还要做 socket/`flock` 清理)会不会有其它代码路径意外碰
    already-unloaded 的 `self._engine`/`self._backend`(理论上不会,因为
    `unload()` 只在关闭序列的最后调用,但从未实测)。`memory_snapshot()`
    里 `torch.cuda.memory_stats()` 的具体 key 名(`allocated_bytes.all.
    current` 等)需要对着实际安装的 torch 版本核实一遍,不同版本这些 key
    名曾经变过。

## 11. 已知限制 / 遗留问题

- **`bfdiag/cli.py` 的 dispatcher 约定是假设的**:另一个 agent 拥有
  `bfdiag/cli.py`,规格只给了 `register(subparsers) -> None` 这一个签名,没有
  说清楚解析完 `args` 之后怎么分发。我按最常见的 argparse 惯例实现——每个子
  parser `set_defaults(func=callable)`,`callable(args) -> int`。如果对方
  `cli.py` 用别的分发约定(比如按 `args.command` 手工 if/elif),需要合并时对
  一下接口,我自己的 `if __name__ == "__main__":` 里就是照这个假设写的,可以
  直接跑通作为参考实现。
- **超时放弃后的"进程内重启"不是真正的进程级重启**,§3/§10 第 8 条已经说得
  很清楚:真实 CUDA 卡死场景下这条恢复路径大概率不够,需要 GPU 验证后决定要
  不要加一层"重启也失败就 `os._exit()`"的兜底。
- **金丝雀目前只覆盖 DFlash 的 eager/verify-only 路径**,不覆盖
  `LagunaCudaGraphVerify`/`DFlashDraftCudaGraph` 的 **CUDA Graph replay**
  子路径本身的状态(比如 `_fill_buffers` 的实现细节,git log 里提到过的
  "Vectorize LagunaCudaGraphVerify._fill_buffers" 那次改动)——如果污染恰好
  发生在 replay 特有的 buffer 里而不是 KV cache/slot 计数器上,今天的金丝雀
  不一定能测出来。这是一个已知的覆盖面缺口,不是遗漏,只是本任务优先级
  (a)(b)(c) 排在前面,没有时间做到 (c) 之外的加固。
- **§8 的"热引擎可靠 vs 必须冷启动"分类目前是设计判断,不是实测结论**——
  §10 第 1 条(零假设验证)就是专门用来证伪或证实这个分类的,在那条跑完之前,
  不应该把 §8 的分类当成"已验证"来对外宣传。
- **idle-TTL 触发的进程退出依赖 `main()` 正常返回**:真实子进程
  (`python -m bfdiag.daemon.server`)下这条路径已经手动验证过(见 §6 最后一
  条),但如果未来有人把 `Daemon` 嵌到一个不通过 `main()`/`serve_forever()`
  的宿主进程里(比如某种进程内嵌入式用法),需要自己确保 `_auto_shutdown_
  for_idle` 触发后宿主也会跟着退出——`Daemon` 本身只负责释放 GPU 和停止
  accept 循环,不会主动 `sys.exit()` 整个宿主进程。
- **`requires_cold_restart`/`LOAD_TIME_CONFIG_KEYS`/`LOAD_TIME_ENV_VARS` 是
  两套独立的 key 空间**,分别对应"构造函数参数"和"读一次的环境变量",刻意
  没有合并成一套,因为 `bf daemon start` 的复用检测(比较 CLI 参数)和
  `bf submit --sweep` 的守卫(比较环境变量名)本来就是不同粒度的问题——如果
  以后 Laguna 侧新增了"读环境变量决定的 load-time 参数"(而不是构造函数
  参数),需要同时更新两套集合,不会自动同步。
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
