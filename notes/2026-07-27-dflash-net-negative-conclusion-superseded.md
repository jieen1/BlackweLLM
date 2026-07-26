# 旧结论作废:DFlash "净负收益" 判断已被推翻(2026-07-27)

`notes/ANALYSIS_speed_regression.md`(07-26)和本文档系列前几篇今天的笔记里反复出现的
"DFlash 整体比不投机解码慢""M=16 verify 天然比 M=1 贵 16 倍"这两个结论**已被真实
kernel 级 profiling 推翻**,详见 `notes/2026-07-27-dflash-profiling-and-optimization.md`。

- 真实瓶颈是 `LagunaCudaGraphVerify._fill_buffers` 里一个未向量化的 Python 循环
  (64K 下 1024 次零碎 page table 写入,182.5ms 纯 CPU 调度开销),和 M=16 的计算量
  本身无关。
- 向量化修复后(commit `1443143`):DFlash 64K 端到端 tok/s 从 45-46 提升到
  **252-259**,是不开 DFlash 的纯 decode(~80 tok/s)的 **~3.2 倍**。
- 之前基于"DFlash 净负收益"做出的判断(任务 #12 暂缓接入服务主循环)**已经反转**——
  现在这是明确的高优先级项。

后续引用 DFlash 性能相关结论时,以这次 profiling 的真实数据为准,不要再引用旧文档
里"净负收益"的说法。
