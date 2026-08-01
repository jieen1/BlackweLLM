# Prefill/extend 每轮 30+ 秒卡顿：根因是 page_table 宽度，不是 q.shape[0]

## 症状

用户 agent 每轮对话等 30–110 秒。小请求（几十 token）0.9 秒，长 prompt
30–110 秒，且**同一个 prompt 第二次就快**。插桩（`bf`/临时 TIMING 日志）显示
时间 100% 花在 `sparkinfer.attention.paged._forward.paged_attention_forward`
的 JIT 编译上：

```
TIMING bind=0.000s  forward=34.502s  mode=extend  q=(2151, 72, 128)
TIMING bind=0.000s  forward=33.921s  mode=extend  q=(2839, 72, 128)
```

`create_paged_plan` / `build_paged_attention_binding` 均 0.000s（纯 host
端簿记，不是问题）。编译结果落盘到 `~/.cache/sparkinfer`，同一形状第二次
免费——但真实 agent 流量里 prompt 长度、累计上下文长度每轮都变，于是每轮
都撞“新形状”。

## 最初假设（错的）与纠正

最初假设：sparkinfer 按 `q.shape[0]`（本次送入 attention 的 token 数）特化
编译。**这个假设被源码和本机的实测编译缓存直接推翻**：

- `sparkinfer/attention/paged/_forward.py` 的 `forward_cache_key` /
  `_tensor_meta_key`：非 CUDA-graph-decode 路径下
  `dynamic_first_dim = (0,)`，即 **q 的 token 维（dim 0）被显式标记为
  "dynamic"，从编译缓存键里排除**。`page_table` 同样只有 dim 0（num_reqs）
  被标记为 dynamic；**dim 1（block-table 宽度，即
  `ceil((kv_len+qo_len)/block_size)`）没有被排除，是编译键里的精确整数**。
- 本机 `~/.cache/sparkinfer/compile/*/*.json` 的实测统计（2026-08-01，
  1609 个缓存条目）：`mode="extend"` 的 879 条缓存里，`page_table` 的
  第二维（宽度）取值分布在 1 到 1152+ 之间、**超过 100 个不同值**；而 `q`
  的静态维度（`(num_heads, head_dim)`，dim 0 排除在外）只有
  `(48,128)`（full attention）和 `(72,128)`（SWA）**两个值**——和层数/架构
  一一对应，和 token 数完全无关。

**结论**：真正无界增长、逐次触发重新编译的轴是 **block-table 宽度**（一个
`kv_len + qo_len` 的函数），不是 q 的 token 数。这也统一解释了两类看似不同
的症状——prompt 长度不同（`kv_len=0` 时宽度直接由 `qo_len` 决定）和
多轮对话变长（`kv_len` 每轮增长，宽度也跟着变）——本质是同一个根因。

## 修复：SparkinferPrefillWorkspace 固定容量，不是按形状重建

`runtime/backends/laguna_sparkinfer_attn.py` 的 `SparkinferPrefillWorkspace`
过去每次遇到新的 `(q.shape, page_table.shape, ...)` 组合都通过
`PagedAttentionWorkspace.for_tensors()` 重建一个新 workspace——这正是重新
编译的触发点。修复后，每个 layer group（按 `window_left` 区分：full
attention 是 `-1`，每个 SWA window 是 `window-1`）在首次使用时通过
`PagedAttentionWorkspace.for_fixed_capacity()` 建一个**固定容量**的
workspace（容量由 `LagunaBackend.__init__` 算出的
`max_total_q` / `max_page_table_width` 决定，见
`self._prefill_capacity_by_window_left`），之后每次调用只是把当次真实的
`page_table` / `cache_seqlens` / `cu_seqlens_q` 拷进这个固定大小的缓冲区
（`_ensure_capacity` + `_copy_runtime_metadata`），**workspace 本身、连带
sparkinfer 的编译产物都不再重建**。`SparkinferPrefillWorkspace._key()`
现在显式排除 `q.shape[0]` 和 `k_cache.shape[0]`/`v_cache.shape[0]`
（page/block 总数）——这些本来就是这个类要吸收掉的“每次调用都变”的维度。

验证：`tests/test_laguna_sparkinfer_attn.py::
test_prefill_workspace_never_rebuilds_across_varying_real_shapes` 喂入 5 组
互不相同的 `(qo_len, kv_len)`（覆盖 `page_table` 宽度 1 到 130+），断言
`for_fixed_capacity` 只被调用一次；改回旧的 `for_tensors()`/`_key()`（含
`q.shape`）此测试会失败（每个不同形状都新建一次 workspace）。

## 曾经考虑过、后来放弃的方案：block-table 宽度的桶阶梯

在发现 sparkinfer 自带的 `for_fixed_capacity`/`eager_extend_work_items_capacity`
之前，曾经在 `_build_common_attn_metadata` / `_build_swa_attn_metadata` 里把
`block_table` 的宽度向上取整到一个有界的 2 的幂阶梯（8, 16, 32, ...,
`blocks_per_slot`），padding 列填充为该请求自己的首块（不会被读到，
`cache_seqlens` 已经界定了真实的 KV 范围）。这个方案本身是安全的（不写
KV、不影响 causal mask、不越界到其它 slot），但**在 SparkinferPrefillWorkspace
改成固定容量之后就是纯粹多余的复杂度**——真正的编译边界现在完全由
`SparkinferPrefillWorkspace` 的 `(mode, window_left)` 粒度决定，跟调用者
传入的 `page_table` 实际形状无关（只要不超过声明的 `max_page_table_width`）。
已从 `laguna.py` 里移除，仅在此记录以免日后重新发明。

## 启动期预热

`LagunaBackend.warmup_paged_attention_shapes()`：一次 `_forward` 调用即可
同时触发全部 layer group（full attention + 每个 SWA window）各自的一次性
CuTe 编译（因为它们都在同一次 `self.model.forward()` 里跑到）。由
`server/engine.py` 的 `ServerEngine._load_laguna_model` 在模型加载完、任何
slot 还未被真实请求占用之前调用（`QSR_SERVER_WARMUP_PAGED_ATTENTION=0`
可关闭）。只有 `~/.cache/sparkinfer` 是"冷"的那一次（换机器、换 sparkinfer
版本、或清空过缓存目录）才会真的付编译的钱；之后每次重启都是磁盘缓存命中。

## 诸如 DFlash 草稿模型的类似路径

`runtime/backends/laguna_dflash.py` 里草稿模型的注意力也走同一个
`SparkinferPrefillWorkspace`，但它的 `qo_len` 本来就固定
（`NUM_QUERY_PER_REQ`，见 `_draft_forward`），从未真正触发过这个 bug；
仍然改成 `for_fixed_capacity`（`prefill_capacity_by_window_left`）只是为了
接口一致，不是修复一个已观察到的症状。

## 已知缺口（本次未修，留给后续）：DFlash eager verify 路径未被预热覆盖

`LagunaBackend.warmup_paged_attention_shapes()` 只预热主模型自己的两个
layer group（`window_left` 分别是 `-1` 和 SWA window-1，各 extend/decode
两种 mode）。DFlash 的 `dflash_round()`（`laguna_dflash.py:1521` 附近）在
`self._verify_cg is None` 时会退回 `_forward_verify_with_aux`，这条路径
显式传 `mode="verify"`（`laguna_dflash.py:1692,1698`）——对
`SparkinferPrefillWorkspace._key()` 来说这是**跟 extend/decode 完全独立
的第三个 contract**，我的预热函数不会碰它。

正常生产配置下 `DFlashEngine._init_cuda_graph()` 在 `__init__` 里就同步
捕获 verify/draft CG（不是懒捕获），所以 `_forward_verify_with_aux`
理论上不该在真实流量里被打到——但没有直接证据证明这条路径在本机
100% 命中 CG（`_capture_verify_cg()`/`_capture_draft_cg()` 的成功日志是
`logger.info`，在这个仓库当前的日志配置下默认被吞掉，看不到；只有失败
才会打 `logger.warning` 冒出来，本次验证过程中没看到这条失败日志，但这
只是"没证据说它失败"，不等于"证明它成功走了 CG"）。

真实验证中出现过一次跟这个假设吻合的现象（一次热身后的全新长度扫描里，
第一个请求和跨 8192 chunk 边界的长请求异常慢，随后同一进程内重跑就再也
没出现过）——但同一时间窗口里另有一个探针脚本刚触发过 CUDA illegal
memory access 崩溃，也不能排除是 GPU/driver 恢复期的假象。两个假设都没
彻底坐实，所以：

- 不在这个 PR 里动 DFlash 的任何代码。
- 后续如果要坐实，思路是：给 `_forward_verify_with_aux`/`_draft_forward`
  也接上跟 `SparkinferPrefillWorkspace.forward()` 同款的阈值诊断日志（本
  PR 已经加在 extend/decode 路径上），然后专门在 DFlash 开启、且故意让
  verify CG 捕获失败（或干脆强制走 eager 分支）的配置下复现一次，看
  `mode=verify` 是否真的补一次编译。
- 如果坐实，修法应该跟本 PR 完全同构：给 DFlash 的 `enable_dflash()` 也
  声明一份 `prefill_capacity_by_window_left`（`mode="verify"` 也需要走
  `for_fixed_capacity`，如果它现在还是老的按形状重建），并在
  `warmup_paged_attention_shapes()`（或专门给 DFlash 加一个姊妹函数）里
  补一次 mode="verify" 的 dummy 调用。
