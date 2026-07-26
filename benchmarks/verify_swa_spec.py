"""加载模型，验证 static_forward_context 各层 get_kv_cache_spec 是否区分 SWA/full。"""
import os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QSR_A2_CUSTOM_GEMM", "0")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
from runtime.compat_vllm import EngineArgs
from runtime.backends.laguna import LagunaBackend
MODEL = "poolside/Laguna-S-2.1-NVFP4"
config = EngineArgs(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.85,
                    enforce_eager=True, dtype="bfloat16", disable_log_stats=True).create_engine_config()
backend = LagunaBackend(config, num_slots=1, block_size=64, blocks_per_slot=64)
sfc = config.compilation_config.static_forward_context
from collections import Counter
specs = {}
for name in backend.attn_layer_names:
    layer = sfc[name]
    if hasattr(layer, "get_kv_cache_spec"):
        spec = layer.get_kv_cache_spec(config)
        cls = type(spec).__name__
        sw = getattr(spec, "sliding_window", None)
        specs[name] = (cls, sw)
# 统计
cnt = Counter((cls, sw) for cls, sw in specs.values())
print("=== per-layer KV cache spec 统计 ===")
for (cls, sw), n in cnt.items():
    print(f"  {cls} sliding_window={sw}: {n} 层")
# 抽样打印前 6 层
print("=== 前 6 层明细 ===")
for name in backend.attn_layer_names[:6]:
    print(f"  {name}: {specs[name]}")
print(f"总 attention 层: {len(backend.attn_layer_names)}")
